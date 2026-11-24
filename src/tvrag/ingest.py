"""Read a vault snapshot: Markdown files -> documents with frontmatter and body."""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
# [[target]], [[target|label]], [[target#heading|label]], ![[embed]]; inside Markdown
# tables the pipe is escaped as "\|", so a trailing backslash on the target is dropped.
WIKILINK_RE = re.compile(r"!?\[\[([^\]\|#]*?)\\?(?:#[^\]\|]*?)?\\?(?:\|[^\]]*)?\]\]")


@dataclass
class Document:
    path: str  # relative to the snapshot root, forward slashes
    doc_type: str  # "topic" | "session" | "other"
    body: str
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    source: str | None = None
    date: str | None = None  # frontmatter `date`, ISO yyyy-mm-dd
    created: str | None = None  # frontmatter `created`, ISO yyyy-mm-dd
    links: list[str] = field(default_factory=list)


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    return [s.strip() for s in str(value).split(",") if s.strip()]


_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")


def _as_date(value) -> str | None:
    """Normalise a frontmatter date to yyyy-mm-dd; anything unparseable -> None."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    m = _ISO_DATE_RE.match(str(value).strip())
    if not m:
        return None
    try:
        return dt.date(*map(int, m.groups())).isoformat()
    except ValueError:
        return None


def split_frontmatter(text: str) -> tuple[dict, str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta = yaml.safe_load(m.group(1)) or {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, text[m.end():]


def extract_links(text: str) -> list[str]:
    """Wiki-link targets in order of first appearance (heading/label parts dropped)."""
    seen: dict[str, None] = {}
    for m in WIKILINK_RE.finditer(text):
        target = m.group(1).strip()
        if target:
            seen.setdefault(target, None)
    return list(seen)


def classify(rel_path: str, doc_type_dirs: dict[str, str] | None) -> str:
    """Document type from folder: any path component equal to a configured folder name."""
    parts = Path(rel_path).parts[:-1]
    for doc_type, folder in (doc_type_dirs or {}).items():
        if folder in parts:
            return doc_type
    return "other"


def parse_document(text: str, rel_path: str, doc_type_dirs: dict[str, str] | None = None) -> Document:
    text = text.lstrip("﻿")
    meta, body = split_frontmatter(text)
    return Document(
        path=rel_path,
        doc_type=classify(rel_path, doc_type_dirs),
        body=body,
        tags=_as_list(meta.get("tags")),
        aliases=_as_list(meta.get("aliases")),
        source=None if meta.get("source") is None else str(meta.get("source")),
        date=_as_date(meta.get("date")),
        created=_as_date(meta.get("created")),
        links=extract_links(body),
    )


def load_snapshot(root: str | Path, doc_type_dirs: dict[str, str] | None = None) -> list[Document]:
    """All Markdown files under `root`, sorted by relative path."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"snapshot folder not found: {root}")
    docs = []
    for p in sorted(root.rglob("*.md")):
        rel = p.relative_to(root).as_posix()
        docs.append(parse_document(p.read_text(encoding="utf-8"), rel, doc_type_dirs))
    return docs
