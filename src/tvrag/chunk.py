"""Split documents into heading-level chunks and assign event time (date heading > frontmatter date > created).

Usage: python -m tvrag.chunk --config configs/local.yaml [--out PATH]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from tvrag.ingest import Document, extract_links

# Length limits in bge-m3 tokens (XLM-R sentencepiece, no special tokens).
# MAX_TOKENS: well under the model's 8192 window but small enough that one chunk is
# about one topic and the top-5 context stays short. MIN_TOKENS: below this a section is
# usually just a heading or a one-line pointer with too little text to embed on its own.
MAX_TOKENS = 512
MIN_TOKENS = 64

HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
# Only full dates at the start of a heading count. The snapshot survey found
# yyyy-mm-dd only; dotted/slashed full dates are accepted too because they are
# unambiguous. Month-only ("yyyy년 m월") or year-less ("m/d") headings are NOT
# turned into dates: that would require guessing a day or a year.
DATE_HEADING_RE = re.compile(r"^(\d{4})[-./](\d{1,2})[-./](\d{1,2})(?!\d)")

TokenCounter = Callable[[str], int]


@dataclass
class Section:
    level: int  # 0 = text before the first heading
    title: str
    heading_path: list[str]
    lines: list[str] = field(default_factory=list)
    event_date: str | None = None
    date_source: str = "none"

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


@dataclass
class Chunk:
    chunk_id: str
    doc_path: str
    heading_path: list[str]
    text: str
    event_date: str | None
    date_source: str  # heading | frontmatter_date | created | none
    doc_date: str | None
    doc_type: str
    tags: list[str]
    aliases: list[str]
    links: list[str]
    headings: list[str]  # titles of all sections merged into this chunk
    n_tokens: int


def heading_date(title: str) -> str | None:
    m = DATE_HEADING_RE.match(title)
    if not m:
        return None
    try:
        return dt.date(*map(int, m.groups())).isoformat()
    except ValueError:
        return None


def doc_fallback_date(doc: Document) -> tuple[str | None, str]:
    if doc.date:
        return doc.date, "frontmatter_date"
    if doc.created:
        return doc.created, "created"
    return None, "none"


def split_sections(doc: Document) -> list[Section]:
    """One Section per heading. Date headings (level >= 2) date their own subtree.

    Level-1 headings are document titles; their date (if any) is the document date,
    which frontmatter already carries, so they do not count as date headings.
    """
    fb_date, fb_source = doc_fallback_date(doc)
    sections = [Section(0, "", [], event_date=fb_date, date_source=fb_source)]
    stack: list[tuple[int, str, str | None]] = []  # (level, title, date set at this level)
    in_fence = False
    for line in doc.body.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
        m = None if in_fence else HEADING_RE.match(line)
        if not m:
            sections[-1].lines.append(line)
            continue
        level, title = len(m.group(1)), m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        own = heading_date(title) if level >= 2 else None
        stack.append((level, title, own))
        inherited = next((d for _, _, d in reversed(stack) if d), None)
        sec = Section(level, title, [t for _, t, _ in stack], lines=[line])
        sec.event_date, sec.date_source = (inherited, "heading") if inherited else (fb_date, fb_source)
        sections.append(sec)
    return [s for s in sections if s.text]


def _split_long(text: str, count: TokenCounter, max_tokens: int) -> list[str]:
    """Pack paragraphs (blank-line separated) up to max_tokens; fall back to lines."""
    def units_of(t: str) -> list[str]:
        paras = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
        out = []
        for p in paras:
            if count(p) <= max_tokens:
                out.append(p)
            else:  # one huge paragraph or a list without blank lines: split on lines
                out.extend(ln for ln in p.split("\n") if ln.strip())
        return out

    def heading_only(parts: list[str]) -> bool:
        return all(HEADING_RE.match(ln) for p in parts for ln in p.split("\n") if ln.strip())

    pieces, cur = [], []
    queue = units_of(text)
    while queue:
        u = queue.pop(0)
        cand = "\n\n".join(cur + [u])
        if cur and count(cand) > max_tokens and heading_only(cur) and "\n" in u.strip():
            # never leave a heading alone: break the next paragraph into lines instead
            queue[:0] = [ln for ln in u.split("\n") if ln.strip()]
            continue
        if cur and count(cand) > max_tokens and not heading_only(cur):
            pieces.append("\n\n".join(cur))
            cur = [u]
        else:
            cur.append(u)
    if cur:
        pieces.append("\n\n".join(cur))
    return pieces  # a single line longer than max_tokens stays whole


def chunk_document(
    doc: Document,
    count: TokenCounter,
    max_tokens: int = MAX_TOKENS,
    min_tokens: int = MIN_TOKENS,
) -> list[Chunk]:
    sections = split_sections(doc)

    # 1) merge short sections forward into the next one with the same event date,
    #    so a merge never mixes two dated sections.
    groups: list[list[Section]] = []
    pending: list[Section] = []
    for sec in sections:
        if pending and pending[0].event_date != sec.event_date:
            groups.append(pending)
            pending = []
        pending.append(sec)
        if count("\n\n".join(s.text for s in pending)) >= min_tokens:
            groups.append(pending)
            pending = []
    if pending:
        # trailing short remainder: attach to the previous group if same date, else keep
        if groups and groups[-1][0].event_date == pending[0].event_date and count(
            "\n\n".join(s.text for s in groups[-1] + pending)
        ) <= max_tokens:
            groups[-1].extend(pending)
        else:
            groups.append(pending)

    # 2) split long groups at paragraph boundaries
    fb_date, _ = doc_fallback_date(doc)
    chunks: list[Chunk] = []
    for grp in groups:
        head = grp[0]
        text = "\n\n".join(s.text for s in grp)
        parts = [text] if count(text) <= max_tokens else _split_long(text, count, max_tokens)
        for part in parts:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.path}::{len(chunks):03d}",
                    doc_path=doc.path,
                    heading_path=head.heading_path,
                    text=part,
                    event_date=head.event_date,
                    date_source=head.date_source,
                    doc_date=fb_date,
                    doc_type=doc.doc_type,
                    tags=doc.tags,
                    aliases=doc.aliases,
                    links=extract_links(part),
                    headings=[s.title for s in grp if s.title],
                    n_tokens=count(part),
                )
            )
    return chunks


def chunk_documents(docs: Iterable[Document], count: TokenCounter, **kw) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d, count, **kw)]


def bge_m3_counter(model_name: str = "BAAI/bge-m3") -> TokenCounter:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    return lambda t: len(tok(t, add_special_tokens=False)["input_ids"])


def write_jsonl(chunks: Iterable[Chunk], out: str | Path) -> int:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> None:
    from tvrag.config import load_config
    from tvrag.ingest import load_snapshot

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="default: <private_root>/chunks/<snapshot_id>.jsonl")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    root = Path(cfg["paths"]["private_root"])
    snap = cfg["snapshot_id"]
    ccfg = cfg.get("chunk") or {}
    docs = load_snapshot(root / "snapshots" / snap, (cfg.get("ingest") or {}).get("doc_type_dirs"))
    count = bge_m3_counter(cfg["embedding"]["model"])
    chunks = chunk_documents(
        docs,
        count,
        max_tokens=ccfg.get("max_tokens", MAX_TOKENS),
        min_tokens=ccfg.get("min_tokens", MIN_TOKENS),
    )
    out = Path(args.out) if args.out else root / "chunks" / f"{snap}.jsonl"
    n = write_jsonl(chunks, out)
    print(f"docs={len(docs)} chunks={n} -> {out}")


if __name__ == "__main__":
    main()
