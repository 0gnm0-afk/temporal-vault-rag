"""Metric checks against hand-computed examples (synthetic paths only)."""
import json

import pytest

from tvrag.eval import (
    bootstrap_ci,
    check_gold,
    dates_in_text,
    load_eval_set,
    normalize_path,
    run_eval,
    score_question,
)


def row(cid, doc, date=None, headings=(), text=""):
    return {"chunk_id": cid, "doc_path": doc, "event_date": date, "headings": list(headings),
            "heading_path": list(headings[:1]), "text": text}


def test_normalize_path_cuts_prefix_and_extension():
    assert normalize_path("root/vault/area/notes/x.md", "vault/area") == "notes/x"
    assert normalize_path("vault\\area\\notes\\x.md", "vault/area/") == "notes/x"
    assert normalize_path("notes/x.md", "vault/area") == "notes/x"  # prefix absent: unchanged
    assert normalize_path("notes/x", None) == "notes/x"


def test_as_of_question_hand_computed():
    # gold: docs a, b; sections a#X, b#Y; as_of 2025-02-15
    q = {"id": "q", "type": "시점 고정", "as_of": "2025-02-15", "gold_docs": ["a.md", "b.md"],
         "gold_sections": [{"doc": "a.md", "heading": "## X"}, {"doc": "b.md", "heading": "Y"}]}
    ranked = [
        row("c1", "c.md", "2025-03-01", ["Z"]),   # not gold, after as_of
        row("c2", "a.md", "2025-02-01", ["X"]),   # gold doc + gold section
        row("c3", "a.md", "2025-02-20", ["W"]),   # gold doc, after as_of
        row("c4", "d.md", "2025-01-01"),
        row("c5", "d.md", None),                  # undated: not counted as contamination
        row("c6", "b.md", "2025-01-10", ["Y"]),   # rank 6: outside k=5
    ]
    r = score_question(q, ranked, ranked, None, k=5, history_k=10)
    assert r["recall_doc"] == 0.5          # a found, b only at rank 6
    assert r["mrr_doc"] == 0.5             # first gold doc at rank 2
    assert r["recall_section"] == 0.5      # a#X found, b#Y not in top 5
    assert r["mrr_section"] == 0.5
    assert r["contamination"] == pytest.approx(2 / 5)  # c1, c3
    assert r["contaminated"] == ["c1", "c3"]
    assert r["missed_docs"] == ["b"]


def test_no_hit_gives_zero_mrr():
    q = {"id": "q", "type": "일반", "gold_docs": ["a.md"]}
    r = score_question(q, [row("c1", "b.md")], [], None, 5, 10)
    assert r["mrr_doc"] == 0.0 and r["recall_doc"] == 0.0 and r["first_doc_rank"] is None


def test_recency_hand_computed():
    q = {"id": "q", "type": "현재 값", "gold_docs": ["a.md"]}
    all_rows = [row("a1", "a.md", "2025-01-01"), row("a2", "a.md", "2025-03-01"), row("o1", "o.md", "2025-03-01")]
    # top-1 is another doc with the same date as the latest gold chunk
    r = score_question(q, [all_rows[2], all_rows[1]], all_rows, None, 5, 10)
    assert r["latest_gold_date"] == "2025-03-01"
    assert r["recency"] == 1.0 and r["recency_strict"] == 0.0
    r = score_question(q, [all_rows[0], all_rows[1]], all_rows, None, 5, 10)
    assert r["recency"] == 0.0


def test_history_hand_computed():
    q = {"id": "q", "type": "변화 추적", "gold_docs": ["a.md"],
         "change_points": [{"date": "2025-01-01"}, {"date": "2025-02-01"}, {"date": "2025-03-01"}]}
    ranked = [row("c1", "a.md", "2025-01-01"), row("c2", "s.md", "2025-05-01", text="2025년 2월 1일에 바꿨다")]
    r = score_question(q, ranked, ranked, None, 5, 10)
    assert r["history"] == pytest.approx(2 / 3)         # event_date 01-01 + text mention 02-01
    assert r["history_strict"] == pytest.approx(1 / 3)  # event_date only
    assert r["history_missing"] == ["2025-03-01"]
    r = score_question(q, ranked, ranked, None, 5, history_k=1)  # window limits coverage
    assert r["history"] == pytest.approx(1 / 3)


def test_dates_in_text_formats():
    assert dates_in_text("2025-03-17, 2025.4.2, 2025/05/09, 2025년 6월 1일, 2025-13-01") == {
        "2025-03-17", "2025-04-02", "2025-05-09", "2025-06-01"}


def test_bootstrap_ci():
    assert bootstrap_ci([1.0, 1.0, 1.0]) == (1.0, 1.0, 1.0)
    m, lo, hi = bootstrap_ci([0.0, 1.0] * 10, n_boot=500, seed=1)
    assert m == 0.5 and 0.0 <= lo < 0.5 < hi <= 1.0
    assert bootstrap_ci([0.0, 1.0] * 10, n_boot=500, seed=1) == (m, lo, hi)  # seeded


def test_load_eval_set_validates(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text(json.dumps({"id": "1", "type": "시점 고정", "question": "?", "gold_docs": []}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="as_of"):
        load_eval_set(p)
    p.write_text(json.dumps({"id": "1", "type": "기타", "question": "?", "gold_docs": []}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown type"):
        load_eval_set(p)


def test_check_gold_and_run_eval():
    rows = {r["chunk_id"]: r for r in [row("c1", "x/a.md", "2025-01-01", ["H"]), row("c2", "x/b.md", "2025-02-01")]}
    qs = [{"id": "1", "type": "일반", "question": "?", "gold_docs": ["root/x/a.md"],
           "gold_sections": [{"doc": "root/x/a.md", "heading": "H"}, {"doc": "root/x/a.md", "heading": "nope"}]}]
    assert check_gold(qs, list(rows.values()), "x") == ["1: gold section not in index: a # nope"]
    out = run_eval(qs, lambda q, m, n: ["c2", "c1"], rows, ["A"], prefix="x", k=5, n_boot=100)
    s = out["A"]["summary"]["전체"]
    assert s["mrr_doc"]["mean"] == 0.5 and s["recall_doc"]["mean"] == 1.0
