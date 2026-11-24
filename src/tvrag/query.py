"""Rule-based question reading for mode C: question type and as-of date.

Types: 일반 | 시점 고정 | 현재 값 | 변화 추적. Precedence when cues collide:
  1. change cue ("어떻게 바뀌었", "변화", "추이", "흐름", ...)      -> 변화 추적
  2. an explicit or relative date ("11/12 당시", "10월 말", "2주 전") -> 시점 고정
  3. a "now" cue ("지금", "현재", "요즘", "최근", ...)              -> 현재 값
  4. otherwise                                                       -> 일반
A change question may name dates ("3월부터 지금까지"); the change cue wins because the
answer is a sequence, not one value. A dated question with "지금" ("10월 말과 지금")
is rare; the date wins and the question is read as as-of.

Date -> one as-of day. A period is read as its LAST day: an as-of question asks what
was true at that time, and records written during the period belong to it, so the
filter keeps them. 초 -> day 10, 중순 -> day 20, 말 / month only -> last day,
"첫 주" -> day 7. Year: written year, else the reference date's year; a date that
would fall after the reference date is moved one year back. Relative expressions
("N일/주/달 전", "지난주", "지난달", "어제") count back from the reference date.
"""
from __future__ import annotations

import calendar
import datetime as dt
import re
from dataclasses import dataclass

GENERAL, AS_OF, CURRENT, CHANGE = "일반", "시점 고정", "현재 값", "변화 추적"

CHANGE_RE = re.compile(
    r"바뀌|바꿔\s*왔|변화|변했|변해\s*왔|변동|변경\s*(?:이력|과정|내역)|추이|흐름|경과|달라졌|달라진|달라져"
    r"|이력|과정을|어떻게\s*(?:늘|줄|올라|내려|움직)|순서대로|차례로|시간순"
)
CURRENT_RE = re.compile(r"지금|현재|요즘|요새|최근|최신|이번\s*주|오늘|아직|여전히|현행|마지막으로|가장\s*늦")

_NUM_KO = {"한": 1, "하루": 1, "두": 2, "이틀": 2, "세": 3, "사흘": 3, "네": 4, "나흘": 4, "다섯": 5, "여섯": 6}
ISO_Q = re.compile(r"(?<!\d)(\d{4})[-./](\d{1,2})[-./](\d{1,2})(?!\d)")
YMD_KO = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월(?:\s*(\d{1,2})\s*일|\s*(초|중순|말|첫\s*주|마지막\s*주))?")
MD_KO = re.compile(r"(?<!\d)(\d{1,2})\s*월\s*(\d{1,2})\s*일")
MONTH_PART = re.compile(r"(?<![\d년])(?<!년\s)(\d{1,2})\s*월\s*(초|중순|말|첫\s*주|마지막\s*주)?(?!\s*\d)")
SLASH_Q = re.compile(r"(?<![\d/.])(\d{1,2})/(\d{1,2})(?![\d/])")
REL_Q = re.compile(r"(\d+|한|두|세|네|다섯|여섯)\s*(일|주|주일|달|개월)\s*전|(하루|이틀|사흘|나흘)\s*전")
LAST_Q = re.compile(r"지난\s*(주|달|번)|어제|그저께|엊그제")


@dataclass
class Intent:
    qtype: str
    as_of: str | None = None
    cue: str | None = None  # the text that decided the type (for reports)


def _part_day(year: int, month: int, part: str | None) -> int:
    last = calendar.monthrange(year, month)[1]
    part = (part or "").replace(" ", "")
    return {"초": 10, "중순": 20, "첫주": 7}.get(part, last)


def _fit_year(year: int | None, month: int, day_fn, ref: dt.date) -> dt.date | None:
    """Build the date; without a written year use ref's year, one year back if in the future."""
    try:
        if year:
            return dt.date(year, month, day_fn(year))
        d = dt.date(ref.year, month, day_fn(ref.year))
        if d > ref:
            d = dt.date(ref.year - 1, month, day_fn(ref.year - 1))
        return d
    except ValueError:
        return None


def find_as_of(question: str, ref: dt.date) -> tuple[str | None, str | None]:
    """(as_of ISO date, matched text) for the first date expression, else (None, None)."""
    q = question
    m = ISO_Q.search(q)
    if m:
        try:
            return dt.date(*map(int, m.groups())).isoformat(), m.group(0)
        except ValueError:
            pass
    m = YMD_KO.search(q)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            day = int(m.group(3)) if m.group(3) else None
            d = _fit_year(y, mo, (lambda _y: day) if day else (lambda _y: _part_day(_y, mo, m.group(4))), ref)
            if d:
                return d.isoformat(), m.group(0)
    for pat in (MD_KO, SLASH_Q):
        m = pat.search(q)
        if m:
            mo, day = int(m.group(1)), int(m.group(2))
            if 1 <= mo <= 12 and 1 <= day <= 31:
                d = _fit_year(None, mo, lambda _y: day, ref)
                if d:
                    return d.isoformat(), m.group(0)
    m = MONTH_PART.search(q)
    if m:
        mo = int(m.group(1))
        if 1 <= mo <= 12:
            d = _fit_year(None, mo, lambda _y: _part_day(_y, mo, m.group(2)), ref)
            if d:
                return d.isoformat(), m.group(0)
    m = REL_Q.search(q)
    if m:
        if m.group(3):
            n, unit = _NUM_KO[m.group(3)], "일"
        else:
            raw, unit = m.group(1), m.group(2)
            n = int(raw) if raw.isdigit() else _NUM_KO.get(raw, 1)
        days = {"일": 1, "주": 7, "주일": 7, "달": 30, "개월": 30}[unit] * n
        return (ref - dt.timedelta(days=days)).isoformat(), m.group(0)
    m = LAST_Q.search(q)
    if m:
        w = m.group(0).replace(" ", "")
        if w == "어제":
            d = ref - dt.timedelta(days=1)
        elif w in ("그저께", "엊그제"):
            d = ref - dt.timedelta(days=2)
        elif w == "지난주":
            d = ref - dt.timedelta(days=7)
        elif w == "지난달":
            first = ref.replace(day=1)
            d = first - dt.timedelta(days=1)
        else:  # 지난번: no date
            return None, None
        return d.isoformat(), m.group(0)
    return None, None


def parse_question(question: str, reference_date: str | dt.date) -> Intent:
    ref = reference_date if isinstance(reference_date, dt.date) else dt.date.fromisoformat(str(reference_date)[:10])
    m = CHANGE_RE.search(question)
    if m:
        return Intent(CHANGE, None, m.group(0))
    as_of, cue = find_as_of(question, ref)
    if as_of:
        return Intent(AS_OF, as_of, cue)
    m = CURRENT_RE.search(question)
    if m:
        return Intent(CURRENT, None, m.group(0))
    return Intent(GENERAL, None, None)
