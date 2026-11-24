"""Date mentions -> range, and the question-reading rules for mode C.

The question strings below are the development set for tvrag.query: written for these
rules, not taken from any evaluation set.
"""
import datetime as dt

import pytest

from tvrag.dates import date_range, infer_year, mentioned_dates
from tvrag.query import AS_OF, CHANGE, CURRENT, GENERAL, parse_question

D = dt.date.fromisoformat


# ---------------------------------------------------------------- dates
def test_formats_and_exclusions():
    text = (
        "2025-03-02 측정. 3/9에는 쉬었다. 4월 1일 다시 시작.\n"
        "링크 [[2025-01-05 다른 기록|그날]] 은 세지 않는다.\n"
        "체성분 71.2/30.1/22.4, 세트 10/12회, 밥 1/2 공기, 소수 3.5/7.25\n"
        "8-12회, 6월 말, 25.08.10"
    )
    got = mentioned_dates(text, D("2025-03-05"))
    assert got == [D("2025-03-02"), D("2025-03-09"), D("2025-04-01")]


def test_slash_date_after_sentence_period_counts():
    assert mentioned_dates("기록을 남겼다.3/9 오후에 측정", D("2025-03-10")) == [D("2025-03-09")]


def test_year_inference_closest_to_reference():
    assert infer_year(12, 20, D("2026-01-10")) == D("2025-12-20")   # just before New Year
    assert infer_year(4, 30, D("2026-11-11")) == D("2027-04-30")    # goal date ahead
    assert infer_year(10, 6, D("2026-09-02")) == D("2026-10-06")
    assert infer_year(2, 29, D("2025-03-01")) == D("2024-02-29")    # only valid year


def test_reference_uses_latest_full_date_in_text():
    # event date is an old `created`; the text shows it was edited in October
    assert mentioned_dates("2026-10-21 갱신. 10/6 대비 -0.7kg", D("2026-01-15")) == [D("2026-10-21"), D("2026-10-06")]


def test_range_rules():
    # record date (frontmatter) joins the range
    assert date_range("3/1에 잰 값", "2025-03-10", "frontmatter_date") == ("2025-03-01", "2025-03-10", 1)
    # a created date does not
    assert date_range("2025-02-01 기록, 2025-02-20 기록", "2025-09-02", "created") == ("2025-02-01", "2025-02-20", 2)
    # no mention -> event_date
    assert date_range("날짜 없음", "2025-09-02", "created") == ("2025-09-02", "2025-09-02", 0)
    # mentions after the ceiling are goals, not events
    assert date_range("목표일 2026-04-30, 오늘 2025-11-20", "2025-11-20", "frontmatter_date",
                      ceiling="2025-11-23") == ("2025-11-20", "2025-11-20", 1)


# ---------------------------------------------------------------- question reading
REF = "2025-11-23"


@pytest.mark.parametrize("q, qtype, as_of", [
    # as-of: explicit and relative dates; periods read as their last day
    ("11/12 당시 체중은?", AS_OF, "2025-11-12"),
    ("10월 말 기준으로 한도가 얼마였어", AS_OF, "2025-10-31"),
    ("9월 초에는 어떤 루틴이었지", AS_OF, "2025-09-10"),
    ("8월 중순 단백질 목표", AS_OF, "2025-08-20"),
    ("7월에 먹던 보충제는", AS_OF, "2025-07-31"),
    ("2024년 3월 5일 측정값", AS_OF, "2024-03-05"),
    ("2025-06-01 무렵 식단", AS_OF, "2025-06-01"),
    ("12월 20일 기록 기준으로", AS_OF, "2024-12-20"),        # after the reference -> last year
    ("2주 전 체중은?", AS_OF, "2025-11-09"),
    ("지난달 한도", AS_OF, "2025-10-31"),
    ("어제 잰 수치", AS_OF, "2025-11-22"),
    # change wins over dates and "now"
    ("3월부터 지금까지 체중이 어떻게 바뀌었어?", CHANGE, None),
    ("칼로리 한도 변화 과정을 알려줘", CHANGE, None),
    ("체지방률 추이", CHANGE, None),
    ("루틴이 어떻게 달라졌지", CHANGE, None),
    # current
    ("지금 하루 한도는?", CURRENT, None),
    ("요즘 운동 루틴", CURRENT, None),
    ("가장 최근 인바디 결과", CURRENT, None),
    ("현재 단백질 목표가 뭐야", CURRENT, None),
    # general
    ("무릎이 아플 때 어떻게 했어?", GENERAL, None),
    ("외식할 때 원칙은?", GENERAL, None),
    ("주 4회 분할은 어떻게 나눠?", GENERAL, None),
    ("12개월 동안 할 일", GENERAL, None),                       # "12개월" is not a month
])
def test_parse_question(q, qtype, as_of):
    it = parse_question(q, REF)
    assert (it.qtype, it.as_of) == (qtype, as_of)
