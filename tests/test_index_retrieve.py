"""Index build/load and modes A/B with a fake embedder (no model download)."""
import hashlib
import json

import numpy as np
import pytest

from tvrag.index import Tokenizer, build_index, is_current, load_index, passage_text
from tvrag.retrieve import Retriever, recency_weight, rrf, select_distinct_dates


class FakeEmbedder:
    """Deterministic bag-of-characters vectors: similar strings -> similar vectors."""

    dim = 64

    def encode(self, texts, show_progress=False):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for ch in t:
                out[i, int(hashlib.md5(ch.encode()).hexdigest(), 16) % self.dim] += 1
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return out


def chunk(cid, doc, head, text, date, source="frontmatter_date", doc_type="session"):
    return {"chunk_id": cid, "doc_path": doc, "heading_path": [head], "headings": [head], "text": text,
            "event_date": date, "date_source": source, "doc_date": date, "doc_type": doc_type, "n_tokens": 10}


CHUNKS = [
    chunk("logs/2025-01-01 a.md::000", "logs/2025-01-01 a.md", "측정", "아침 공복 체중은 70.5kg이었다.", "2025-01-01"),
    chunk("logs/2025-02-01 b.md::000", "logs/2025-02-01 b.md", "식단", "하루 칼로리 한도를 1,800kcal로 낮췄다.", "2025-02-01"),
    chunk("notes/sleep.md::000", "notes/sleep.md", "수면", "자정 전에 잠들고 커피는 오후에 마시지 않는다.",
          "2025-01-05", "created", "topic"),
]


def write_chunks(path):
    path.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in CHUNKS), encoding="utf-8")
    return path


@pytest.fixture
def index_dir(tmp_path):
    cp = write_chunks(tmp_path / "chunks.jsonl")
    out = tmp_path / "idx"
    build_index(cp, out, {"model": "fake"}, embedder=FakeEmbedder(), show_progress=False)
    return cp, out


def test_rrf_hand_computed():
    fused = rrf([["a", "b", "c"], ["c", "a"]], k=60)
    assert [c for c, _ in fused] == ["a", "c", "b"]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1][1] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[2][1] == pytest.approx(1 / 62)


def test_tokenizer_keeps_content_morphemes():
    toks = Tokenizer()("공복 체중을 70.5kg으로 쟀다")
    assert {"공복", "체중", "70.5", "kg"} <= set(toks)
    assert "을" not in toks and "다" not in toks


def test_passage_text_has_title_and_headings():
    assert passage_text(CHUNKS[0]).startswith("2025-01-01 a > 측정\n\n")


def test_build_is_reproducible_and_current(index_dir, tmp_path):
    cp, out = index_dir
    m1 = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    m2 = build_index(cp, tmp_path / "idx2", {"model": "fake"}, embedder=FakeEmbedder(), show_progress=False)
    assert m1 == m2
    assert is_current(out, cp, "fake") and not is_current(out, cp, "other-model")
    idx = load_index(out)
    assert [r["chunk_id"] for r in idx.rows] == sorted(c["chunk_id"] for c in CHUNKS)


def test_modes(index_dir):
    _, out = index_dir
    r = Retriever(load_index(out), FakeEmbedder(), cfg={"rrf_k": 60, "candidates": 3})
    a = r.retrieve("칼로리 한도", "A", k=2)
    assert len(a) == 2 and a[0].mode == "A" and a[0].rank == 1
    b = r.retrieve("칼로리 한도", "B", k=3, as_of="2025-01-15")
    assert b[0].chunk_id == "logs/2025-02-01 b.md::000"  # the only BM25 match
    assert b[0].components["bm25_rank"] == 1 and b[0].components["as_of"] == "2025-01-15"
    assert b[0].event_date == "2025-02-01" and b[0].text
    # no fact table: D falls back to C
    d = r.retrieve("칼로리 한도", "D", k=2, reference_date="2025-03-01")
    c = r.retrieve("칼로리 한도", "C", k=2, reference_date="2025-03-01")
    assert [h.chunk_id for h in d] == [h.chunk_id for h in c] and d[0].mode == "D"
    with pytest.raises(ValueError):
        r.retrieve("x", "Z")


@pytest.mark.slow
def test_bge_m3_index_is_reproducible(tmp_path):
    """Two builds with the real model give byte-identical vectors."""
    from tvrag.config import load_config

    cp = write_chunks(tmp_path / "chunks.jsonl")
    emb = load_config("configs/demo.yaml")["embedding"]
    m1 = build_index(cp, tmp_path / "i1", emb, show_progress=False)
    m2 = build_index(cp, tmp_path / "i2", emb, show_progress=False)
    assert m1["vectors_sha256"] == m2["vectors_sha256"] and m1["embedding_dim"] == 1024


def test_recency_weight():
    assert recency_weight(0, 60) == 1.0
    assert recency_weight(60, 60) == pytest.approx(0.5)
    assert recency_weight(120, 60) == pytest.approx(0.25)
    assert recency_weight(-5, 60) == 1.0          # future content counts as "now"
    assert recency_weight(500, 0) == 1.0          # 0 = no decay
    assert recency_weight(None, 60) == pytest.approx(0.5 ** 4)


def test_select_distinct_dates():
    pool = [("a", 0.9, {}), ("b", 0.8, {}), ("c", 0.7, {}), ("d", 0.6, {})]
    date = {"a": "2025-01-01", "b": "2025-01-01", "c": "2025-02-01", "d": "2025-03-01"}.get
    assert [h[0] for h in select_distinct_dates(pool, 3, date)] == ["a", "c", "d"]
    assert [h[0] for h in select_distinct_dates(pool, 4, date)] == ["a", "c", "d", "b"]  # filled


def test_mode_c_time_logic(index_dir):
    _, out = index_dir
    r = Retriever(load_index(out), FakeEmbedder(), cfg={"candidates": 3, "c": {"half_life_days": 30}})
    # as-of before the February chunk: it is filtered out even though it matches best
    hits = r.retrieve("칼로리 한도", "C", k=3, as_of="2025-01-15")
    assert "logs/2025-02-01 b.md::000" not in [h.chunk_id for h in hits]
    assert all(h.date_min <= "2025-01-15" for h in hits)
    assert hits[0].components["qtype"] == "시점 고정" and hits[0].components["cue"] == "argument"
    # the same as-of read from the question text
    hits = r.retrieve("1월 15일 기준 칼로리 한도", "C", k=3, reference_date="2025-03-01")
    assert hits[0].components["as_of"] == "2025-01-15"
    assert "logs/2025-02-01 b.md::000" not in [h.chunk_id for h in hits]
    # post-filter gives the same set here
    r.as_of_filter = "post"
    assert {h.chunk_id for h in r.retrieve("칼로리 한도", "C", 3, as_of="2025-01-15")} == {h.chunk_id for h in hits}
    # current: newest content first when relevance is close
    cur = r.retrieve("지금 기록", "C", k=3, reference_date="2025-03-01")
    assert cur[0].components["qtype"] == "현재 값"
    assert cur[0].date_max == max(h.date_max for h in cur)
    # change: one chunk per date, in time order
    ch = r.retrieve("어떻게 바뀌었어", "C", k=3, reference_date="2025-03-01")
    assert [h.date_min for h in ch] == sorted(h.date_min for h in ch)
    # labels override reading
    lab = r.retrieve("아무 질문", "C", k=2, qtype="현재 값", reference_date="2025-03-01")
    assert lab[0].components["qtype"] == "현재 값" and lab[0].components["cue"] == "label"
