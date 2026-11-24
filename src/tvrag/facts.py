"""Bitemporal fact table for mode D: value-level facts with validity intervals.

Row = (subject, attribute, value, unit, condition, nature, valid_from, valid_to,
       evidence chunks, recorded_at), plus review status and notes.

Source and build
  The table is built from a curated observation file (JSONL, one observation per line;
  the vault's own data, so it lives next to the vault data, not in code). Observations
  are written by a person with help from `extract_candidates` (regex rules over chunks,
  no LLM): the rules propose (attribute, date, value) candidates and `coverage` reports
  which curated rows they found, so the manual share is measurable.

  Observation fields:
    id            stable id
    subject       whose value (default "사용자")
    attr          attribute name (see ATTR_KEYWORDS for the names the question reader knows)
    value, unit   a number, or text for non-numeric values (routines) and ranges ("1,700-1,900")
    detail        optional extra text (reps x sets, weekly total ...); its numbers count as
                  the row's numbers for value scoring
    date          valid_from: when the value became true (measurement or decision day)
    time          optional clock time of a measurement
    series        condition series (e.g. a measurement-time category). Validity is chained
                  per (subject, attr, series): a value in one series never closes another
                  series' interval. "" = one chain for the attribute.
    device, condition, cond_source
                  measuring device, free-text condition, where the condition came from
    nature        측정 | 사용자 결정 | 사용자 보고 | 제안 | 추정
    evidence      chunk ids that state the value
    review        자동 (session checked) | 확인 (user confirmed) | 수정 (user corrected)
    note          free text

  build_table replays observations in recording order (recorded_at = earliest event_date
  of the evidence chunks). A new observation is appended to its chain; the row that was
  open before it gets its valid_to filled (and closed_at = when that became known). No
  row is ever deleted or overwritten; a back-dated observation is inserted between two
  rows and takes over the remainder of the earlier row's interval.

Answering (mode D, see tvrag.retrieve)
  detect_attrs maps question words to attributes (dictionary below, built from the
  schema and the vault vocabulary, not from eval questions). Rows are then chosen by
  question type (types and as-of come from tvrag.query, the same reading as mode C):
    시점 고정  per attribute, the row valid at as_of (valid_from <= as_of < valid_to) with
               the latest valid_from among the attribute's default series and its
               `change_series`; condition-exception series are shown as 참고 only
    현재 값    the row valid at the reference date in the default series
    변화 추적  all rows of the default series + change_series, in time order
    일반       no fact answer (D = C)
  Only natures in ANSWER_NATURES answer; 제안·추정 rows are listed as 참고(채택 미확인).
  Ranked list (mode D): the answer rows' evidence chunks, then mode C's list.
    시점 고정  per row the record-day session chunk first; chunks written after as_of dropped
    현재 값    per row the latest chunk stating the value first (it shows the value still held;
               chosen on the demo set, where the record-day chunk lost to later summaries)
    변화 추적  greedy cover: fewest chunks that evidence the most history rows, time order
  `max_evidence_per_row` chunks per row (시점 고정, 현재 값).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ANSWER_NATURES = ("측정", "사용자 결정", "사용자 보고")
REFERENCE_NATURES = ("제안", "추정")
NATURES = ANSWER_NATURES + REFERENCE_NATURES
REVIEW = ("자동", "확인", "수정")

# Question word -> attributes. Order matters only for display. A word may name several
# attributes ("인바디" = the whole body-composition sheet, bare "체지방" = mass and rate).
ATTR_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"인바디|체성분", ("체중", "골격근량", "체지방량", "체지방률")),
    (r"체중|몸무게", ("체중",)),
    (r"골격근|근육량", ("골격근량",)),
    (r"체지방량|지방량", ("체지방량",)),
    (r"체지방률|체지방\s*%|체지방\s*퍼센트|체지방\s*비율", ("체지방률",)),
    (r"체지방(?!량|률|\s*%|\s*퍼센트|\s*비율)", ("체지방량", "체지방률")),
    (r"한도|칼로리\s*목표|목표\s*칼로리|열량\s*목표|목표\s*열량|섭취\s*목표|칼로리\s*예산|열량\s*예산|하루\s*칼로리|주간\s*칼로리|kcal\s*목표",
     ("식단 한도",)),
    (r"스쿼트", ("스쿼트 작업 중량",)),
    (r"루틴|분할|A일|B일|운동\s*구성|종목\s*구성|어떤\s*운동|무슨\s*운동", ("운동 루틴",)),
    (r"단백질", ("단백질 목표",)),
)
_ATTR_RES = tuple((re.compile(p, re.IGNORECASE), attrs) for p, attrs in ATTR_KEYWORDS)


def detect_attrs(question: str, known: set[str] | None = None) -> list[str]:
    """Attributes the question names, in order of first mention; only `known` ones.
    A group word ("인바디", bare "체지방") expands to its attributes only when the question
    names none of them specifically ("인바디 체지방률" -> 체지방률 only)."""
    found: list[tuple[int, int, str]] = []
    specific: set[str] = set()
    groups = []
    for order, (rx, attrs) in enumerate(_ATTR_RES):
        m = rx.search(question)
        if not m:
            continue
        if len(attrs) > 1:
            groups.append((m.start(), order, attrs))
        else:
            specific.add(attrs[0])
            found.append((m.start(), order * 10, attrs[0]))
    for start, order, attrs in sorted(groups, key=lambda g: len(g[2])):  # narrow groups first
        if not specific & set(attrs):
            found += [(start, order * 10 + j, a) for j, a in enumerate(attrs)]
            specific |= set(attrs)
    out: list[str] = []
    for _, _, a in sorted(found):
        if a not in out and (known is None or a in known):
            out.append(a)
    return out


# ---------------------------------------------------------------- numbers
_DATE_LIKE = re.compile(r"\d{4}[-./]\d{1,2}[-./]\d{1,2}|(?<![\d.])\d{1,2}/\d{1,2}(?![\d/])|\d{1,2}\s*월\s*\d{1,2}\s*일|\d{1,2}:\d{2}")
_NUM = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(kg|㎏|%|kcal|g(?![a-z])|L|회|세트)?", re.IGNORECASE)
_UNIT_NORM = {"㎏": "kg", "kg": "kg", "%": "%", "kcal": "kcal", "g": "g", "l": "L", "회": "회", "세트": "세트"}


def numbers_in(text) -> list[tuple[float, str | None]]:
    """(number, unit) pairs in text; dates and clock times are skipped."""
    if text is None:
        return []
    if isinstance(text, (int, float)):
        return [(float(text), None)]
    s = _DATE_LIKE.sub(" ", str(text))
    out = []
    for m in _NUM.finditer(s):
        unit = m.group(2)
        out.append((float(m.group(1).replace(",", "")), _UNIT_NORM.get(unit.lower(), unit) if unit else None))
    return out


def numbers_match(a: list[tuple[float, str | None]], b: list[tuple[float, str | None]], unit_hint: str | None = None) -> bool:
    """True if some number appears in both lists (to 0.05), with compatible units.
    A side without a unit takes `unit_hint` (the fact row's unit)."""
    for x, ux in a:
        for y, uy in b:
            ux2, uy2 = ux or unit_hint, uy or unit_hint
            if abs(x - y) < 0.05 and (ux2 is None or uy2 is None or ux2 == uy2):
                return True
    return False


# ---------------------------------------------------------------- table
@dataclass
class Fact:
    id: str
    attr: str
    value: object
    valid_from: str
    subject: str = "사용자"
    unit: str | None = None
    detail: str | None = None
    time: str | None = None
    series: str = ""
    device: str | None = None
    condition: str | None = None
    cond_source: str | None = None
    nature: str = "측정"
    evidence: list[str] = field(default_factory=list)
    review: str = "자동"
    note: str | None = None
    valid_to: str | None = None
    recorded_at: str | None = None
    closed_at: str | None = None    # recorded_at of the observation that set valid_to
    superseded_by: str | None = None

    @property
    def chain(self) -> tuple[str, str, str, bool]:
        """Validity is chained per (subject, attr, series) and separately for answer rows
        and 제안·추정 rows: a proposal never closes a decided or measured value."""
        return (self.subject, self.attr, self.series, self.nature in ANSWER_NATURES)

    @property
    def numeric(self) -> bool:
        """Numbers, and text values with a unit ("1,700-1,900" kcal); text without a unit
        (a routine) is not numeric even if it mentions counts."""
        return isinstance(self.value, (int, float)) or bool(self.unit and numbers_in(self.value))

    def valid_at(self, day: str) -> bool:
        return self.valid_from <= day and (self.valid_to is None or day < self.valid_to)

    def value_text(self) -> str:
        v = self.value
        if isinstance(v, float):
            v = f"{v:g}" if v != int(v) else f"{v:.1f}"
        s = f"{v}{self.unit or ''}" if isinstance(self.value, (int, float)) else f"{v}"
        if isinstance(self.value, str) and self.unit and self.unit not in s:
            s += self.unit
        return s + (f" ({self.detail})" if self.detail else "")

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        return d


FIELDS = tuple(Fact.__dataclass_fields__)


def load_observations(path: str | Path) -> list[dict]:
    obs = []
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip() or line.lstrip().startswith("//"):
                continue
            o = json.loads(line)
            for key in ("id", "attr", "value", "date", "evidence"):
                if key not in o:
                    raise ValueError(f"{path}:{n}: missing {key}")
            if o.get("nature", "측정") not in NATURES:
                raise ValueError(f"{path}:{n}: unknown nature {o.get('nature')!r}")
            if o.get("review", "자동") not in REVIEW:
                raise ValueError(f"{path}:{n}: unknown review {o.get('review')!r}")
            dt.date.fromisoformat(o["date"])
            obs.append(o)
    ids = [o["id"] for o in obs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate observation ids")
    return obs


def build_table(observations: list[dict], chunk_dates: dict[str, str]) -> list[Fact]:
    """Replay observations in recording order and chain validity per (subject, attr, series).
    `chunk_dates`: chunk_id -> event_date (evidence must exist)."""
    facts = []
    for o in observations:
        missing = [c for c in o["evidence"] if c not in chunk_dates]
        if missing:
            raise ValueError(f"{o['id']}: evidence chunk not found: {missing[0]}")
        if not o["evidence"]:
            raise ValueError(f"{o['id']}: no evidence")
        kw = {k: v for k, v in o.items() if k in FIELDS and k not in ("valid_to", "recorded_at", "closed_at", "superseded_by")}
        kw["valid_from"] = o["date"]
        kw.setdefault("subject", "사용자")
        f = Fact(**kw)
        # a chunk cannot have recorded a value before it existed: topic notes carry an early
        # `created` date but list later values, so each chunk counts from max(its date, valid_from)
        f.recorded_at = min(max(chunk_dates[c] or f.valid_from, f.valid_from) for c in f.evidence)
        facts.append(f)

    chains: dict[tuple, list[Fact]] = {}
    for f in sorted(facts, key=lambda f: (f.recorded_at, f.valid_from, f.id)):
        rows = chains.setdefault(f.chain, [])
        if any(r.valid_from == f.valid_from for r in rows):
            raise ValueError(f"{f.id}: two rows for {f.chain} on {f.valid_from} (merge them into one observation)")
        rows.append(f)
        rows.sort(key=lambda r: r.valid_from)
        i = rows.index(f)
        if i > 0:  # the row before it ends where this one starts (only valid_to is filled)
            prev = rows[i - 1]
            prev.valid_to, prev.closed_at, prev.superseded_by = f.valid_from, f.recorded_at, f.id
        if i + 1 < len(rows):  # back-dated: runs until the next known row
            nxt = rows[i + 1]
            f.valid_to, f.closed_at, f.superseded_by = nxt.valid_from, f.recorded_at, nxt.id
    return sorted(facts, key=lambda f: (f.attr, f.series, f.valid_from, f.id))


def write_table(facts: list[Fact], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for fact in facts:
            f.write(json.dumps(fact.to_dict(), ensure_ascii=False) + "\n")


def load_table(path: str | Path) -> list[Fact]:
    with Path(path).open(encoding="utf-8") as f:
        return [Fact(**json.loads(line)) for line in f if line.strip()]


def table_markdown(facts: list[Fact], title: str, short=lambda cid: cid) -> str:
    lines = [f"# {title}", "", f"{len(facts)}행. 유효 구간은 [유효시작, 유효끝), 빈 유효끝 = 열림. "
             "기록 = 근거 청크 중 가장 이른 기록일, 닫힘 = 유효끝이 알려진 날.", ""]
    for attr in sorted({f.attr for f in facts}):
        rows = [f for f in facts if f.attr == attr]
        lines += [f"## {attr} ({len(rows)}행)", "",
                  "| id | 값 | 계열 | 성격 | 유효시작 | 유효끝 | 기록 | 닫힘 | 측정·조건 | 조건 근거 | 검토 | 근거 청크 | 메모 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for f in sorted(rows, key=lambda r: (r.series, r.valid_from)):
            cond = " ".join(x for x in (f.time, f.device, f.condition) if x)
            lines.append("| " + " | ".join(str(x if x is not None else "").replace("|", "/") for x in (
                f.id, f.value_text(), f.series, f.nature, f.valid_from, f.valid_to or "", f.recorded_at,
                f.closed_at or "", cond, f.cond_source or "", f.review, "<br>".join(short(c) for c in f.evidence),
                f.note or "")) + " |")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- answering
@dataclass
class FactAnswer:
    qtype: str
    attrs: list[str]
    when: str | None                      # as_of or reference date
    rows: list[Fact]                      # the answer (one per attr, or the history)
    previous: dict[str, Fact] = field(default_factory=dict)   # attr -> row before the answer
    reference: list[Fact] = field(default_factory=list)       # 참고: other series / 제안·추정

    def info(self) -> list[dict]:
        return [{"id": f.id, "attr": f.attr, "value": f.value, "unit": f.unit, "detail": f.detail,
                 "numeric": f.numeric, "series": f.series, "nature": f.nature, "valid_from": f.valid_from,
                 "valid_to": f.valid_to, "evidence": f.evidence} for f in self.rows]


class FactStore:
    def __init__(self, facts: list[Fact], cfg: dict | None = None, chunk_dates: dict[str, str] | None = None,
                 chunk_types: dict[str, str] | None = None):
        cfg = cfg or {}
        self.facts = facts
        self.attrs = {f.attr for f in facts}
        self.default_series: dict[str, str] = dict(cfg.get("default_series") or {})
        self.change_series: dict[str, list[str]] = {k: list(v) for k, v in (cfg.get("change_series") or {}).items()}
        self.max_evidence = int(cfg.get("max_evidence_per_row", 2))
        self.chunk_dates = chunk_dates or {}
        self.chunk_types = chunk_types or {}

    @classmethod
    def from_config(cls, cfg: dict, rows: list[dict] | None = None) -> "FactStore | None":
        fcfg = cfg.get("facts") or {}
        from tvrag.config import data_paths

        path = data_paths(cfg).get("facts")
        if not path or not Path(path).exists():
            return None
        dates = {r["chunk_id"]: r.get("event_date") for r in rows or []}
        types = {r["chunk_id"]: r.get("doc_type") for r in rows or []}
        return cls(load_table(path), fcfg, dates, types)

    # series used to answer --------------------------------------------------
    def _answer_rows(self, attr: str, series: list[str] | None) -> list[Fact]:
        return [f for f in self.facts if f.attr == attr and f.nature in ANSWER_NATURES
                and (series is None or f.series in series)]

    def _series(self, attr: str, history: bool) -> list[str] | None:
        if attr not in self.default_series:
            return None  # all series
        s = [self.default_series[attr]]
        return s + self.change_series.get(attr, []) if history else s

    def answer(self, question: str, qtype: str, when: str | None) -> FactAnswer | None:
        from tvrag.query import AS_OF, CHANGE, CURRENT

        attrs = detect_attrs(question, self.attrs)
        if not attrs or qtype not in (AS_OF, CURRENT, CHANGE) or (qtype != CHANGE and not when):
            return None
        ans = FactAnswer(qtype, attrs, when, [])
        for attr in attrs:
            if qtype == CHANGE:
                rows = sorted(self._answer_rows(attr, self._series(attr, True)), key=lambda f: (f.valid_from, f.series))
                ans.rows += rows
                ans.reference += [f for f in self.facts if f.attr == attr and f not in rows]
                continue
            series = self._series(attr, qtype == AS_OF)
            valid = [f for f in self._answer_rows(attr, series) if f.valid_at(when)]
            if not valid:
                continue
            best = max(valid, key=lambda f: (f.valid_from, f.series == self.default_series.get(attr)))
            ans.rows.append(best)
            prev = [f for f in self.facts if f.chain == best.chain and f.valid_to == best.valid_from]
            if prev:
                ans.previous[attr] = prev[0]
            others = [f for f in self.facts if f.attr == attr and f is not best and f.valid_from <= when
                      and f.valid_from >= best.valid_from and f.chain != best.chain]
            ans.reference += sorted(others, key=lambda f: f.valid_from)[-3:]
        return ans if ans.rows or ans.reference else None

    # evidence -> ranked chunk ids -------------------------------------------
    def _ordered_evidence(self, f: Fact, before: str | None = None, latest_first: bool = False) -> list[str]:
        """Default (시점 고정): the record-day session chunk first, then the rest by date, and
        chunks written after `before` are dropped. latest_first (현재 값): the most recent chunk
        that states the value first, since it shows the value still held then."""
        ev = [c for c in f.evidence if before is None or (self.chunk_dates.get(c) or "") <= before]

        def key(c):
            d = self.chunk_dates.get(c) or ""
            return (d != f.valid_from, self.chunk_types.get(c) != "session", d, c)

        if latest_first:
            return sorted(ev, key=lambda c: (self.chunk_dates.get(c) or "", self.chunk_types.get(c) == "session", c),
                          reverse=True)
        return sorted(ev, key=key)

    def evidence_ids(self, ans: FactAnswer, n: int) -> list[str]:
        from tvrag.query import AS_OF, CHANGE

        if ans.qtype == CHANGE:
            return self._cover(ans.rows, n)
        before = ans.when if ans.qtype == AS_OF else None
        out: list[str] = []
        for f in ans.rows:
            for c in self._ordered_evidence(f, before, latest_first=ans.qtype != AS_OF)[: self.max_evidence]:
                if c not in out:
                    out.append(c)
        return out[:n]

    def _cover(self, rows: list[Fact], n: int) -> list[str]:
        """Greedy cover: repeatedly take the chunk that evidences the most rows not yet
        covered (ties: a chunk written on the row's own day, then session, then earlier),
        until every row is covered or n chunks are taken; return them in time order."""
        covers: dict[str, set[int]] = {}
        own_day: dict[str, bool] = {}
        for i, f in enumerate(rows):
            for c in f.evidence:
                covers.setdefault(c, set()).add(i)
                own_day[c] = own_day.get(c, False) or self.chunk_dates.get(c) == f.valid_from
        left, picked = set(range(len(rows))), []
        while left and len(picked) < n:
            best = min((c for c in covers if covers[c] & left),
                       key=lambda c: (-len(covers[c] & left), not own_day[c], self.chunk_types.get(c) != "session",
                                      self.chunk_dates.get(c) or "", c), default=None)
            if best is None:
                break
            picked.append(best)
            left -= covers[best]
        return sorted(picked, key=lambda c: (self.chunk_dates.get(c) or "", c))

    def format(self, ans: FactAnswer, short=lambda cid: cid) -> str:
        from tvrag.query import CHANGE

        out = []
        if ans.qtype == CHANGE:
            for attr in ans.attrs:
                rows = [f for f in ans.rows if f.attr == attr]
                if not rows:
                    continue
                out.append(f"[D] {attr} 변화 ({len(rows)}행, 계열: {', '.join(sorted({f.series or '-' for f in rows}))})")
                for f in rows:
                    out.append(f"    {f.valid_from}  {f.value_text()}  [{f.series or '-'}·{f.nature}]  근거: {short(f.evidence[0])}")
        else:
            label = "시점" if ans.qtype != "현재 값" else "기준일"
            for f in ans.rows:
                cond = " · ".join(x for x in (f.device, f.series) if x)
                out.append(f"[D] {f.attr}{' · ' + cond if cond else ''}  ({label} {ans.when})")
                out.append(f"    값 {f.value_text()}  유효 {f.valid_from} ~ {f.valid_to or '(열림)'}  성격: {f.nature}  "
                           f"기록 {f.recorded_at}  근거: {', '.join(short(c) for c in f.evidence[:3])}")
                if f.cond_source:
                    out.append(f"    조건 근거: {f.cond_source}")
                p = ans.previous.get(f.attr)
                if p:
                    out.append(f"    직전 {p.value_text()}  {p.valid_from} ~ {p.valid_to}")
        for f in ans.reference:
            tag = "참고(채택 미확인)" if f.nature in REFERENCE_NATURES else f"참고({f.series or '다른 계열'})"
            out.append(f"    {tag}: {f.attr} {f.value_text()} ({f.valid_from}, {f.nature})")
        return "\n".join(out)


# ---------------------------------------------------------------- value scoring (D only)
def value_scores(q: dict, facts: list[dict] | None, window_days: int = 3) -> dict:
    """Automatic number match (user decision 5-가). Returns {} when the question has no
    numeric gold. 시점 고정 / 현재 값: share of D's answer rows whose number appears in
    `answer`. 변화 추적: share of numeric change_points matched by a D row within
    ±window_days with the same number. No D answer -> 0. Non-numeric rows (routines) are
    not scored; a question D answered only with such rows is left out (value_nonnumeric)."""
    t = q.get("type")
    res: dict = {}
    all_facts = facts or []
    facts = [f for f in all_facts if f.get("numeric", True)]
    if all_facts and not facts and t in ("시점 고정", "현재 값", "변화 추적"):
        return {"value_nonnumeric": 1.0}
    if t in ("시점 고정", "현재 값"):
        gold = numbers_in(q.get("answer"))
        if not gold:
            return {}
        res["value_answered"] = float(bool(facts))
        if not facts:
            res["value_acc"] = 0.0
            return res
        hits = [numbers_match(_row_numbers(f), gold, f.get("unit")) for f in facts]
        res["value_acc"] = sum(hits) / len(hits)
    elif t == "변화 추적":
        cps = [(str(c.get("date"))[:10], numbers_in(c.get("value"))) for c in q.get("change_points") or []]
        cps = [(d, n) for d, n in cps if n]
        if not cps:
            return {}
        res["value_answered"] = float(bool(facts))
        ok = 0
        for d, nums in cps:
            day = dt.date.fromisoformat(d)
            for f in facts:
                if abs((dt.date.fromisoformat(f["valid_from"]) - day).days) <= window_days and \
                        numbers_match(_row_numbers(f), nums, f.get("unit")):
                    ok += 1
                    break
        res["value_acc"] = ok / len(cps)
    return res


def _row_numbers(f: dict) -> list[tuple[float, str | None]]:
    if not f.get("numeric", True):
        return []
    nums = numbers_in(f.get("value"))
    if isinstance(f.get("value"), (int, float)) and f.get("unit"):
        nums = [(v, f["unit"]) for v, _ in nums]
    return nums + numbers_in(f.get("detail"))


# ---------------------------------------------------------------- rule candidates
_CAND_RULES: tuple[tuple[str, re.Pattern, re.Pattern], ...] = (
    ("체중", re.compile(r"체중|몸무게|weight", re.I), re.compile(r"(?<![\d.])([4-9]\d\.\d|1[0-4]\d\.\d)\s*(?:kg|㎏)?", re.I)),
    ("골격근량", re.compile(r"골격근|skeletal muscle", re.I), re.compile(r"(?<![\d.])([2-4]\d\.\d)\s*(?:kg|㎏)?", re.I)),
    ("체지방량", re.compile(r"체지방량|지방량|체지방\s*\d|fat mass|body fat(?! percentage)", re.I),
     re.compile(r"(?<![\d.])(\d{1,2}\.\d)\s*(?:kg|㎏)(?!/m)", re.I)),
    ("체지방률", re.compile(r"체지방률|체지방|body fat( percentage)?", re.I), re.compile(r"(?<![\d.])(\d{1,2}\.\d)\s*%", re.I)),
    ("식단 한도", re.compile(r"한도|목표|예산|target|budget|limit", re.I),
     re.compile(r"(?<![\d.,])(1,?[0-9]{3}|[2-3],?[0-9]{3})\s*kcal", re.I)),
    ("스쿼트 작업 중량", re.compile(r"스쿼트|squat", re.I), re.compile(r"(?<![\d.])(\d{2,3})\s*kg", re.I)),
    ("단백질 목표", re.compile(r"단백질.{0,10}(목표|하루)", re.I), re.compile(r"(?<![\d.])(\d{2,3})\s*g(?![a-z])", re.I)),
)
_LINE_DATE = (re.compile(r"(?<!\d)(\d{4})[-.](\d{1,2})[-.](\d{1,2})(?!\d)"),
              re.compile(r"(?<!\d)(\d{2})\.\s*(\d{1,2})\.\s*(\d{1,2})\.?(?!\d)"),
              re.compile(r"(?<![\d/.])(\d{1,2})/(\d{1,2})(?![\d/])"))


def _line_date(line: str, ref: dt.date) -> str | None:
    from tvrag.dates import infer_year

    m = _LINE_DATE[0].search(line)
    if m:
        try:
            return dt.date(*map(int, m.groups())).isoformat()
        except ValueError:
            pass
    m = _LINE_DATE[1].search(line)
    if m:
        try:
            return dt.date(2000 + int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
        except ValueError:
            pass
    m = _LINE_DATE[2].search(line)
    if m and 1 <= int(m.group(1)) <= 12 and 1 <= int(m.group(2)) <= 31:
        d = infer_year(int(m.group(1)), int(m.group(2)), ref)
        return d.isoformat() if d else None
    return None


def extract_candidates(rows: list[dict], attrs: set[str] | None = None) -> list[dict]:
    """Regex candidates (attr, date, value, chunk) per line. Date: a date on the line
    (yyyy-mm-dd, yy. mm. dd., m/d), else the chunk's event_date. High recall, low
    precision by design: a person keeps or drops them."""
    out = []
    for r in rows:
        ref = dt.date.fromisoformat(r["event_date"]) if r.get("event_date") else dt.date.today()
        for line in r["text"].split("\n"):
            for attr, cue, val in _CAND_RULES:
                if attrs is not None and attr not in attrs:
                    continue
                if not cue.search(line):
                    continue
                for m in val.finditer(line):
                    out.append({"attr": attr, "value": float(m.group(1).replace(",", "")),
                                "date": _line_date(line, ref) or r.get("event_date"), "chunk_id": r["chunk_id"]})
    seen, uniq = set(), []
    for c in out:
        key = (c["attr"], c["date"], c["value"], c["chunk_id"])
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def coverage(facts: list[Fact], candidates: list[dict], window_days: int = 0) -> dict:
    """Which numeric fact rows the rules found (same attr, number, date within window)."""
    by_attr: dict[str, list[dict]] = {}
    for c in candidates:
        by_attr.setdefault(c["attr"], []).append(c)
    found, missed, nonnum = [], [], []
    for f in facts:
        nums = [v for v, _ in numbers_in(f.value)] if f.numeric else []
        if not nums:
            nonnum.append(f.id)
            continue
        d0 = dt.date.fromisoformat(f.valid_from)
        hit = any(c["value"] in nums and c["date"] and abs((dt.date.fromisoformat(c["date"]) - d0).days) <= window_days
                  for c in by_attr.get(f.attr, []))
        (found if hit else missed).append(f.id)
    return {"numeric": len(found) + len(missed), "found": found, "missed": missed, "non_numeric": nonnum,
            "n_candidates": len(candidates)}


# ---------------------------------------------------------------- CLI
def build_from_config(cfg: dict, verbose: bool = True) -> list[Fact] | None:
    from tvrag.config import data_paths
    from tvrag.index import load_chunks

    paths = data_paths(cfg)
    src = paths.get("facts_source")
    if not src or not Path(src).exists():
        if verbose:
            print(f"[facts] no observation file ({src}); mode D will equal C")
        return None
    rows = load_chunks(paths["chunks"])
    facts = build_table(load_observations(src), {r["chunk_id"]: r["event_date"] for r in rows})
    write_table(facts, paths["facts"])
    md = Path(paths["facts"]).with_suffix(".md")
    short = _shortener(rows)
    md.write_text(table_markdown(facts, f"사실표 ({cfg.get('snapshot_id')})", short) + "\n", encoding="utf-8", newline="\n")
    if verbose:
        print(f"[facts] {len(facts)} rows -> {paths['facts']}")
    return facts


def _shortener(rows: list[dict]):
    def short(cid: str) -> str:
        stem, _, n = cid.rpartition("::")
        return f"{Path(stem).stem[:24]}::{n}"
    return short


def main(argv: list[str] | None = None) -> None:
    from tvrag.config import data_paths, load_config
    from tvrag.index import load_chunks

    ap = argparse.ArgumentParser(description="Build the fact table from the observation file.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--candidates", default=None, help="also write rule candidates + coverage to this JSON file")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    facts = build_from_config(cfg)
    if facts is None or not args.candidates:
        return
    rows = load_chunks(data_paths(cfg)["chunks"])
    cands = extract_candidates(rows, {f.attr for f in facts})
    cov = coverage(facts, cands)
    cov3 = coverage(facts, cands, window_days=3)
    Path(args.candidates).write_text(json.dumps({"coverage": cov, "coverage_3d": cov3, "candidates": cands},
                                                ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"[facts] rule candidates {len(cands)}; numeric rows found by rules {len(cov['found'])}/{cov['numeric']} "
          f"(±3d {len(cov3['found'])}), non-numeric rows {len(cov['non_numeric'])} -> {args.candidates}")


if __name__ == "__main__":
    main()
