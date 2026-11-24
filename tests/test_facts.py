"""Fact table (mode D): build, answer selection, evidence ranking, value scoring, rule candidates.
All strings are invented (a fictional person's log)."""
import json

import pytest

from tests.test_index_retrieve import FakeEmbedder, chunk
from tvrag.facts import (FactStore, build_table, coverage, detect_attrs, extract_candidates, load_observations,
                         numbers_in, numbers_match, table_markdown, value_scores)
from tvrag.index import build_index, load_index
from tvrag.retrieve import Retriever

DATES = {"s1": "2025-01-05", "s2": "2025-02-01", "s3": "2025-03-01", "s4": "2025-03-10", "t1": "2025-01-01"}
TYPES = {"t1": "topic"}


def obs(i, attr, value, date, ev, series="", nature="측정", unit="kg", **kw):
    return {"id": i, "attr": attr, "value": value, "unit": unit, "date": date, "series": series, "nature": nature,
            "evidence": ev, **kw}


OBS = [
    obs("w1", "체중", 80.0, "2025-01-05", ["s1", "t1"], "아침 공복"),
    obs("w2", "체중", 79.0, "2025-02-01", ["s2", "t1"], "아침 공복"),
    obs("w3", "체중", 81.5, "2025-02-10", ["s3"], "식후"),          # another series: must not close w2
    obs("w4", "체중", 78.2, "2025-03-10", ["s4"], "아침 공복"),
    obs("g1", "체중", 70.0, "2025-03-01", ["s3"], "아침 공복", nature="제안"),  # proposal: own chain
    obs("k1", "식단 한도", 2000, "2025-01-05", ["s1"], unit="kcal", nature="사용자 결정"),
    obs("k2", "식단 한도", "1,800-1,900", "2025-03-01", ["s3"], unit="kcal", nature="사용자 결정"),
    obs("r1", "운동 루틴", "주 3회 전신", "2025-01-05", ["s1"], unit=None, nature="사용자 결정"),
]


@pytest.fixture
def facts():
    return build_table(OBS, DATES)


def by_id(facts):
    return {f.id: f for f in facts}


def test_chain_fills_only_valid_to(facts):
    f = by_id(facts)
    assert (f["w1"].valid_to, f["w2"].valid_to, f["w4"].valid_to) == ("2025-02-01", "2025-03-10", None)
    assert f["w3"].valid_to is None and f["g1"].valid_to is None        # separate chains
    assert f["w1"].superseded_by == "w2" and f["w1"].closed_at == "2025-02-01"
    # a topic chunk dated 2025-01-01 cannot have recorded a 2025-02-01 value before that day
    assert f["w2"].recorded_at == "2025-02-01" and f["w1"].recorded_at == "2025-01-05"
    assert len(facts) == len(OBS)
    assert f["k2"].numeric and not f["r1"].numeric


def test_back_dated_row_is_inserted():
    late = obs("w2b", "체중", 79.5, "2025-01-20", ["s4"], "아침 공복")   # recorded 03-10, valid from 01-20
    f = by_id(build_table(OBS + [late], DATES))
    assert f["w1"].valid_to == "2025-01-20" and f["w2b"].valid_to == "2025-02-01"
    assert f["w2b"].recorded_at == "2025-03-10" and f["w1"].closed_at == "2025-03-10"


def test_build_rejects_bad_input():
    with pytest.raises(ValueError):
        build_table([obs("x", "체중", 1.0, "2025-01-05", ["nope"])], DATES)
    with pytest.raises(ValueError):
        build_table([obs("a", "체중", 1.0, "2025-01-05", ["s1"]), obs("b", "체중", 2.0, "2025-01-05", ["s2"])], DATES)


def test_load_observations(tmp_path):
    p = tmp_path / "o.jsonl"
    p.write_text(json.dumps(OBS[0], ensure_ascii=False) + "\n// comment\n", encoding="utf-8")
    assert load_observations(p)[0]["id"] == "w1"
    p.write_text(json.dumps({**OBS[0], "nature": "모름"}, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_observations(p)


def test_detect_attrs():
    known = {"체중", "골격근량", "체지방량", "체지방률", "식단 한도", "운동 루틴", "스쿼트 작업 중량"}
    assert detect_attrs("지금 몸무게 얼마야", known) == ["체중"]
    assert detect_attrs("인바디 결과", known) == ["체중", "골격근량", "체지방량", "체지방률"]
    assert detect_attrs("체지방은 얼마", known) == ["체지방량", "체지방률"]
    assert detect_attrs("체지방률만", known) == ["체지방률"]
    assert detect_attrs("인바디 체지방률은?", known) == ["체지방률"]          # group word not expanded
    assert detect_attrs("인바디 체중과 골격근", known) == ["체중", "골격근량"]
    assert detect_attrs("인바디 체지방은?", known) == ["체지방량", "체지방률"]
    assert detect_attrs("하루 칼로리 한도랑 스쿼트 무게", known) == ["식단 한도", "스쿼트 작업 중량"]
    assert detect_attrs("단백질 목표", known) == []            # not in the table
    assert detect_attrs("수면 습관", known) == []


def test_numbers():
    assert numbers_in("1,850kcal, 체중 72.1kg") == [(1850.0, "kcal"), (72.1, "kg")]
    assert numbers_in("2025-03-01 07:10에 3/4 먹고 70.2") == [(70.2, None)]   # dates, times skipped
    assert numbers_match([(72.1, "kg")], [(72.1, None)]) and not numbers_match([(72.1, "kg")], [(72.1, "%")])
    assert numbers_match([(1800.0, None)], [(1800.0, "kcal")], "kcal")


def store(facts):
    return FactStore(facts, {"default_series": {"체중": "아침 공복"}, "change_series": {}}, dict(DATES), dict(TYPES))


def test_answer_as_of_current_change(facts):
    s = store(facts)
    a = s.answer("2월 15일 체중은?", "시점 고정", "2025-02-15")
    assert [f.id for f in a.rows] == ["w2"] and a.previous["체중"].id == "w1"
    assert [f.id for f in a.reference] == ["w3"]                    # other series shown as 참고 only
    cur = s.answer("지금 체중", "현재 값", "2025-03-20")
    assert [f.id for f in cur.rows] == ["w4"]
    ch = s.answer("체중 어떻게 바뀌었어", "변화 추적", None)
    assert [f.id for f in ch.rows] == ["w1", "w2", "w4"]             # default series only, no proposal
    assert "g1" in [f.id for f in ch.reference]
    assert s.answer("수면은?", "현재 값", "2025-03-20") is None
    assert s.answer("체중", "일반", None) is None
    assert s.answer("1월 1일 체중", "시점 고정", "2024-12-31") is None    # before any row
    text = s.format(a)
    assert "79.0kg" in text and "참고(식후)" in text


def test_evidence_order_and_cover(facts):
    s = store(facts)
    a = s.answer("2월 15일 체중은?", "시점 고정", "2025-02-15")
    assert s.evidence_ids(a, 5) == ["s2", "t1"]          # record-day session first, then the topic note
    early = s.answer("2월 15일 체중은?", "시점 고정", "2025-02-15")
    early.when = "2025-01-31"                               # evidence written after as_of is dropped
    assert "s2" not in s.evidence_ids(early, 5)
    cur = s.answer("지금 식단 한도", "현재 값", "2025-02-20")
    cur.rows[0].evidence = ["s1", "s3"]                    # latest statement first for 현재 값
    assert s.evidence_ids(cur, 5) == ["s3", "s1"]
    ch = s.answer("체중 변화", "변화 추적", None)
    assert s.evidence_ids(ch, 2) == ["t1", "s4"]          # t1 covers w1+w2, then s4; time order
    assert s.evidence_ids(ch, 1) == ["t1"]


def test_value_scores():
    rows = [{"value": 79.0, "unit": "kg", "numeric": True, "valid_from": "2025-02-01"}]
    assert value_scores({"type": "시점 고정", "answer": "79.0kg였다"}, rows) == {"value_answered": 1.0, "value_acc": 1.0}
    assert value_scores({"type": "현재 값", "answer": "78.2kg"}, rows)["value_acc"] == 0.0
    assert value_scores({"type": "현재 값", "answer": "78.2kg"}, []) == {"value_answered": 0.0, "value_acc": 0.0}
    assert value_scores({"type": "현재 값", "answer": "아침 공복으로 잰다"}, rows) == {}   # no numeric gold
    assert value_scores({"type": "일반", "answer": "79.0kg"}, rows) == {}
    routine = [{"value": "주 3회 전신", "unit": None, "numeric": False, "valid_from": "2025-01-05"}]
    assert value_scores({"type": "현재 값", "answer": "주 3회"}, routine) == {"value_nonnumeric": 1.0}
    cps = {"type": "변화 추적", "change_points": [{"date": "2025-01-04", "value": "80.0kg"},
                                               {"date": "2025-02-01", "value": "79.0kg"},
                                               {"date": "2025-03-10", "value": "분할로 변경"}]}
    hist = [{"value": 80.0, "unit": "kg", "valid_from": "2025-01-05"},
            {"value": 79.5, "unit": "kg", "valid_from": "2025-02-01"}]
    assert value_scores(cps, hist) == {"value_answered": 1.0, "value_acc": 0.5}


def test_candidates_and_coverage(facts):
    rows = [{"chunk_id": "c1", "event_date": "2025-02-01", "text": "아침 공복 체중 79.0kg\n하루 한도 2,000kcal 유지"},
            {"chunk_id": "c2", "event_date": "2025-03-12", "text": "3/10 체중 78.2kg, 스쿼트 60kg 5세트"}]
    c = extract_candidates(rows)
    got = {(x["attr"], x["date"], x["value"]) for x in c}
    assert ("체중", "2025-02-01", 79.0) in got and ("체중", "2025-03-10", 78.2) in got
    assert ("식단 한도", "2025-02-01", 2000.0) in got and ("스쿼트 작업 중량", "2025-03-10", 60.0) in got
    cov = coverage(facts, c)
    assert {"w2", "w4"} <= set(cov["found"]) and "r1" in cov["non_numeric"] and "w1" in cov["missed"]


def test_table_markdown(facts):
    md = table_markdown(facts, "t")
    assert "## 체중 (5행)" in md and "| w2 | 79.0kg | 아침 공복 |" in md


CHUNKS = [
    chunk("logs/2025-01-05 a.md::000", "logs/2025-01-05 a.md", "측정", "아침 공복 체중 80.0kg. 하루 한도 2,000kcal.", "2025-01-05"),
    chunk("logs/2025-02-01 b.md::000", "logs/2025-02-01 b.md", "측정", "아침 공복 체중 79.0kg.", "2025-02-01"),
    chunk("logs/2025-03-10 c.md::000", "logs/2025-03-10 c.md", "측정", "아침 공복 체중 78.2kg.", "2025-03-10"),
    chunk("notes/sleep.md::000", "notes/sleep.md", "수면", "자정 전에 잠든다. 커피는 오후에 마시지 않는다.", "2025-01-01",
          "created", "topic"),
]


def test_mode_d_on_retriever(tmp_path):
    cp = tmp_path / "chunks.jsonl"
    cp.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in CHUNKS), encoding="utf-8")
    build_index(cp, tmp_path / "idx", {"model": "fake"}, embedder=FakeEmbedder(), show_progress=False)
    r = Retriever(load_index(tmp_path / "idx"), FakeEmbedder(), cfg={"candidates": 4})
    ids = [c["chunk_id"] for c in CHUNKS]
    o = [obs("w1", "체중", 80.0, "2025-01-05", [ids[0]], "아침 공복"),
         obs("w2", "체중", 79.0, "2025-02-01", [ids[1]], "아침 공복"),
         obs("w3", "체중", 78.2, "2025-03-10", [ids[2]], "아침 공복")]
    dates = {c["chunk_id"]: c["event_date"] for c in CHUNKS}
    r.facts = FactStore(build_table(o, dates), {"default_series": {"체중": "아침 공복"}}, dates,
                        {c["chunk_id"]: c["doc_type"] for c in CHUNKS})
    d = r.retrieve("2월 15일 체중은?", "D", k=3, reference_date="2025-03-31")
    assert d[0].chunk_id == ids[1] and d[0].components["fact_rows"] == ["w2"]
    assert d[0].components["facts"][0]["value"] == 79.0 and d[0].components["qtype"] == "시점 고정"
    assert len({h.chunk_id for h in d}) == len(d) == 3
    cur = r.retrieve("지금 체중", "D", k=2, reference_date="2025-03-31")
    assert cur[0].chunk_id == ids[2]
    lab = r.retrieve("체중 알려줘", "D", k=2, qtype="현재 값", reference_date="2025-03-31")
    assert lab[0].chunk_id == ids[2] and lab[0].components["cue"] == "label"
    q = "커피는 언제 마시나"
    gen = r.retrieve(q, "D", k=2, reference_date="2025-03-31")
    assert gen[0].components["facts"] == []
    assert [h.chunk_id for h in gen] == [h.chunk_id for h in r.retrieve(q, "C", k=2, reference_date="2025-03-31")]
