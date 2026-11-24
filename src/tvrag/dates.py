"""Date mentions in chunk text -> content date range (date_min, date_max).

Formats are the ones a survey of the vault actually found (counts outside links):
  yyyy-mm-dd    full ISO date (most common)
  m/d           year-less slash date (second most common)
  m월 d일       year-less Korean date (rare)
Not recognised on purpose: "m월 말/초/중순" and month-only mentions (a range, not a
day: left out of the first pass), "m-d" (in the vault these are rep or value ranges
such as "8-12"), "m.d" (decimals), "yy.mm.dd" (a single occurrence).

Exclusions:
  - dates inside wiki links [[...]]: a link names another note (often a dated session),
    so its date is that note's date, not an event described here
  - m/d that is part of a number run or a count: preceded by a digit, "/" or a
    decimal ("76.4/31.8/26.8"), followed by "/" or a decimal ("8/26.8"), or by a
    count unit ("12/15회", "1/2 공기")
Frontmatter and file names are not part of chunk text, so they never count; the
chunker already turns date headings and frontmatter into `event_date`.

Year for year-less dates: reference = the later of the chunk's event_date and the
latest full ISO date in its text (topic notes carry a migration `created` date but are
edited later, and their ISO dates show when). Among years ref-1, ref, ref+1 take the
date closest to the reference. Why not "prefer the past": the vault refers forward to
goal dates months ahead ("from 11/23 to 4/30 is 158 days"), which a past-leaning rule
puts in the wrong year. Known limit: a year-less date about half a year away from the
reference is ambiguous either way.

Range: all recognised mentions plus the chunk's event_date when that date is a real
record date (date heading or frontmatter `date`); a `created` date is not (it is a
file-creation or migration date). Mentions after `ceiling` (the snapshot's latest
record date) are goals or plans, not events, and stay out of the range. No mention
and no real date -> event_date alone.
"""
from __future__ import annotations

import datetime as dt
import re

RECORD_DATE_SOURCES = ("heading", "frontmatter_date")

LINK_RE = re.compile(r"!?\[\[[^\]]*\]\]")
ISO_RE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
_UNITS = r"(?:회|개|세트|set|rep|공기|인분|컵|번|kg|g(?![a-z])|%|명|잔|알|장|줄|분|초|cm|ml|L)"
SLASH_RE = re.compile(r"(?<![\d/])(?<!\d\.)(\d{1,2})/(\d{1,2})(?![\d/])(?!\.\d)(?!\s*" + _UNITS + r")")
KOREAN_MD_RE = re.compile(r"(?<!\d)(\d{1,2})\s*월\s*(\d{1,2})\s*일")


def _date(y: int, m: int, d: int) -> dt.date | None:
    try:
        return dt.date(y, m, d)
    except ValueError:
        return None


def infer_year(month: int, day: int, ref: dt.date) -> dt.date | None:
    """The date in ref.year-1..ref.year+1 closest to ref (earlier one on a tie)."""
    cands = [c for c in (_date(y, month, day) for y in (ref.year - 1, ref.year, ref.year + 1)) if c]
    return min(cands, key=lambda c: (abs((c - ref).days), c)) if cands else None


def strip_links(text: str) -> str:
    return LINK_RE.sub(" ", text)


def full_dates(text: str) -> list[dt.date]:
    return [d for d in (_date(*map(int, m.groups())) for m in ISO_RE.finditer(text)) if d]


def mentioned_dates(text: str, ref: dt.date | None) -> list[dt.date]:
    """All recognised date mentions outside links, in text order (year-less ones need ref)."""
    body = strip_links(text)
    out: list[tuple[int, dt.date]] = []
    for m in ISO_RE.finditer(body):
        d = _date(*map(int, m.groups()))
        if d:
            out.append((m.start(), d))
    isos = [d for _, d in out]
    anchor = max([ref, *isos]) if ref and isos else (ref or (max(isos) if isos else None))
    if anchor:
        for pat in (SLASH_RE, KOREAN_MD_RE):
            for m in pat.finditer(body):
                mo, da = int(m.group(1)), int(m.group(2))
                if 1 <= mo <= 12 and 1 <= da <= 31:
                    d = infer_year(mo, da, anchor)
                    if d:
                        out.append((m.start(), d))
    return [d for _, d in sorted(out, key=lambda x: x[0])]


def date_range(text: str, event_date: str | None, date_source: str | None,
               ceiling: str | None = None) -> tuple[str | None, str | None, int]:
    """(date_min, date_max, n_mentions) as ISO strings; n_mentions counts in-range mentions."""
    ref = dt.date.fromisoformat(event_date) if event_date else None
    found = mentioned_dates(text, ref)
    if ceiling:
        top = dt.date.fromisoformat(ceiling)
        found = [d for d in found if d <= top]
    pool = list(found)
    if ref and date_source in RECORD_DATE_SOURCES:
        pool.append(ref)
    if not pool:
        return event_date, event_date, 0
    return min(pool).isoformat(), max(pool).isoformat(), len(found)
