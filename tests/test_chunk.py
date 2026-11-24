"""ingest + chunk on synthetic Markdown only (no real vault content)."""
import json

from tvrag.chunk import (
    chunk_document,
    heading_date,
    main,
    split_sections,
    write_jsonl,
)
from tvrag.ingest import extract_links, load_snapshot, parse_document

DIRS = {"topic": "notes", "session": "logs"}


def words(text: str) -> int:
    """Test token counter: whitespace words."""
    return len(text.split())


TOPIC = """---
tags: [fitness, sleep]
aliases: [rest notes]
source: imaginary diary
created: 2025-01-05
---

# Rest and recovery

Intro line about a made-up person named Robin.

## 2025-02-10 Nap experiment

Robin tried a 20 minute nap. See [[logs/2025-02-10 nap|nap log]].

### Follow-up

Felt fine afterwards; linked [[Recovery hub#Naps|hub]].

## 2025.03.01 Dotted date heading

Dotted heading body.

## 2025년 3월 plan

Month-only heading must not become a date.

## 3/4 short form

Year-less heading must not become a date.

## Undated section

| a | b |
|---|---|
| [[Table target\\|shown]] | x |
"""

SESSION = """---
tags: [log]
date: 2025-02-10
created: 2025-04-01
---

# 2025-02-10 nap

## What happened

Short.
"""


def test_frontmatter_and_links():
    doc = parse_document(TOPIC, "vault/notes/rest.md", DIRS)
    assert doc.doc_type == "topic"
    assert doc.tags == ["fitness", "sleep"] and doc.aliases == ["rest notes"]
    assert doc.source == "imaginary diary"
    assert doc.date is None and doc.created == "2025-01-05"
    assert doc.links == ["logs/2025-02-10 nap", "Recovery hub", "Table target"]


def test_extract_links_variants():
    text = "[[a]] [[b|B]] [[c#h|C]] [[#local]] ![[d.png]] [[a]]"
    assert extract_links(text) == ["a", "b", "c", "d.png"]


def test_doc_type_by_folder():
    assert parse_document(SESSION, "x/logs/s.md", DIRS).doc_type == "session"
    assert parse_document(SESSION, "x/other/s.md", DIRS).doc_type == "other"


def test_heading_date_forms():
    assert heading_date("2025-02-10 x") == "2025-02-10"
    assert heading_date("2025.3.1 x") == "2025-03-01"
    assert heading_date("2025/03/01") == "2025-03-01"
    assert heading_date("2025년 3월 plan") is None
    assert heading_date("3/4 short") is None
    assert heading_date("2025-02-30 bad") is None
    assert heading_date("20250210") is None


def test_date_heading_covers_subsections_and_falls_back():
    doc = parse_document(TOPIC, "vault/notes/rest.md", DIRS)
    by_title = {s.title: s for s in split_sections(doc)}
    assert by_title["2025-02-10 Nap experiment"].date_source == "heading"
    fu = by_title["Follow-up"]
    assert (fu.event_date, fu.date_source) == ("2025-02-10", "heading")
    assert fu.heading_path == ["Rest and recovery", "2025-02-10 Nap experiment", "Follow-up"]
    assert by_title["2025.03.01 Dotted date heading"].event_date == "2025-03-01"
    for t in ("2025년 3월 plan", "3/4 short form", "Undated section", "Rest and recovery"):
        assert (by_title[t].event_date, by_title[t].date_source) == ("2025-01-05", "created")


def test_frontmatter_date_beats_created_and_h1_date_is_not_heading():
    doc = parse_document(SESSION, "v/logs/s.md", DIRS)
    for s in split_sections(doc):
        assert (s.event_date, s.date_source) == ("2025-02-10", "frontmatter_date")


def test_no_dates_at_all():
    doc = parse_document("# T\n\n## A\n\ntext here\n", "v/notes/n.md", DIRS)
    chunks = chunk_document(doc, words, max_tokens=50, min_tokens=1)
    assert chunks
    assert all(c.event_date is None and c.date_source == "none" and c.doc_date is None for c in chunks)


def test_short_sections_merge_only_with_same_date():
    doc = parse_document(TOPIC, "vault/notes/rest.md", DIRS)
    chunks = chunk_document(doc, words, max_tokens=200, min_tokens=20)
    for c in chunks:
        assert c.text  # nothing empty
    nap = next(c for c in chunks if "Nap experiment" in c.headings[0])
    assert nap.headings == ["2025-02-10 Nap experiment", "Follow-up"]  # same date -> merged
    assert nap.links == ["logs/2025-02-10 nap", "Recovery hub"]
    # dated and undated neighbours never share a chunk
    for c in chunks:
        assert len({heading_date(h) or "-" for h in c.headings if heading_date(h)}) <= 1
    assert sum("Dotted heading body" in c.text for c in chunks) == 1
    # every chunk keeps document metadata
    assert all(c.doc_type == "topic" and c.tags == ["fitness", "sleep"] for c in chunks)
    assert all(c.doc_date == "2025-01-05" for c in chunks)
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)


def test_long_section_splits_on_paragraphs():
    paras = "\n\n".join(f"para {i} " + "word " * 30 for i in range(6))
    doc = parse_document(f"# T\n\n## 2025-05-05 Long\n\n{paras}\n", "v/notes/l.md", DIRS)
    chunks = chunk_document(doc, words, max_tokens=80, min_tokens=5)
    chunks = [c for c in chunks if "Long" in c.heading_path[-1]]
    assert len(chunks) >= 3
    assert all(c.n_tokens <= 80 for c in chunks)
    assert all(c.event_date == "2025-05-05" and c.date_source == "heading" for c in chunks)
    joined = " ".join(c.text for c in chunks)
    assert all(f"para {i} " in joined for i in range(6))


def test_heading_never_left_alone_when_splitting():
    lines = "\n".join(f"- item {i} " + "word " * 10 for i in range(5))
    doc = parse_document(f"## 2025-05-05 Head\n\n{lines}\n", "v/notes/h.md", DIRS)
    chunks = chunk_document(doc, words, max_tokens=60, min_tokens=5)
    assert len(chunks) >= 2
    assert chunks[0].text.startswith("## 2025-05-05 Head\n") and "item 0" in chunks[0].text
    assert all(c.n_tokens <= 60 for c in chunks)


def test_code_fence_heading_ignored():
    body = "# T\n\n## 2025-01-01 Real\n\n```\n## 2030-01-01 not a heading\n```\n"
    doc = parse_document(body, "v/notes/f.md", DIRS)
    assert [s.title for s in split_sections(doc)] == ["T", "2025-01-01 Real"]


def test_load_snapshot_and_cli(tmp_path, monkeypatch):
    snap = tmp_path / "priv" / "snapshots" / "abc1234"
    (snap / "vault" / "notes").mkdir(parents=True)
    (snap / "vault" / "logs").mkdir(parents=True)
    (snap / "vault" / "notes" / "rest.md").write_text(TOPIC, encoding="utf-8")
    (snap / "vault" / "logs" / "nap.md").write_text(SESSION, encoding="utf-8")
    docs = load_snapshot(snap, DIRS)
    assert [d.path for d in docs] == ["vault/logs/nap.md", "vault/notes/rest.md"]

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"snapshot_id: abc1234\npaths:\n  private_root: {(tmp_path / 'priv').as_posix()}\n"
        "ingest:\n  doc_type_dirs: {topic: notes, session: logs}\n"
        "chunk: {max_tokens: 100, min_tokens: 3}\nembedding: {model: fake}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("tvrag.chunk.bge_m3_counter", lambda name: words)
    main(["--config", str(cfg)])
    rows = [json.loads(l) for l in (tmp_path / "priv/chunks/abc1234.jsonl").read_text("utf-8").splitlines()]
    assert {r["doc_path"] for r in rows} == {"vault/logs/nap.md", "vault/notes/rest.md"}
    keys = {"chunk_id", "doc_path", "heading_path", "text", "event_date", "date_source",
            "doc_date", "doc_type", "tags", "aliases", "links"}
    assert all(keys <= r.keys() for r in rows)


def test_write_jsonl_roundtrip(tmp_path):
    doc = parse_document(SESSION, "v/logs/s.md", DIRS)
    out = tmp_path / "o.jsonl"
    assert write_jsonl(chunk_document(doc, words), out) == 1
    assert json.loads(out.read_text("utf-8"))["event_date"] == "2025-02-10"
