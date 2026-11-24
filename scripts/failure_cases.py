"""List mode C failures by type with an automatic first-pass cause.

Reads a results JSON written by tvrag.eval (mode C or CL), the eval set and the chunk
file, and writes a Markdown report. The report quotes questions and chunk text, so
write it to a private location; the cause labels alone are safe to publish.

Causes (first matching rule):
  시점 고정
    as_of_convention   the question's period was read as a later day than the label
                       ("4월 중순" -> 20th vs 13th) and a contaminated chunk falls in between
    mixed_time_chunk   a top-k chunk was recorded after as_of but mentions dates up to as_of
                       (date_min <= as_of < event_date): passes the date_min filter
    gold_filtered      a gold chunk has date_min > as_of, so the filter removed it
    relevance          gold not in top-k for other reasons
  현재 값
    stale_top1_date    top-1 is a gold chunk whose event_date is not the latest gold date
                       (e.g. a "current" section dated by `created`)
    newer_non_gold     top-1 is not gold but has newer content than the latest gold chunk
    older_non_gold     top-1 is not gold and older
    gold_not_in_pool   no gold chunk in the top-k at all
  변화 추적 (per missing change date, by the official `history` metric)
    metric_format        a top-history_k chunk does state the date, but only as "m/d" or
                         "m월 d일", which `history` does not read (history_anyfmt does)
    point_not_in_pool    some chunk states the date (any format) or carries it as event_date,
                         but none reached the top-history_k
    point_not_in_corpus  no chunk has the date in any form
  일반
    hub_chunks           a gold doc/section is missing and a non-gold list chunk (3+ wiki
                         links, e.g. a topic note's session-summary list) holds a top-k slot
    section_miss / doc_miss  otherwise
A chunk whose own text mentions dates on both sides of as_of / of a change point is
counted separately as mixed-time evidence for the fact table (S6).

Usage: python scripts/failure_cases.py <results_C.json> <eval.jsonl> <chunks.jsonl> <out.md> [--doc-prefix P]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from tvrag.dates import mentioned_dates
from tvrag.eval import dates_in_text, gold_of, chunk_doc, section_matches

import datetime as dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("eval", type=Path)
    ap.add_argument("chunks", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--doc-prefix", default=None)
    ap.add_argument("--history-k", type=int, default=10)
    args = ap.parse_args()

    res = json.loads(args.results.read_text(encoding="utf-8"))
    qs = {q["id"]: q for q in map(json.loads, args.eval.read_text(encoding="utf-8").splitlines()) if q}
    rows = {r["chunk_id"]: r for r in map(json.loads, args.chunks.read_text(encoding="utf-8").splitlines()) if r}
    pre = args.doc_prefix
    mode = res["meta"]["mode"]

    def gold_rows(q):
        g = gold_of(q, pre)
        if g.sections:
            return [r for r in rows.values() if section_matches(r, pre, g)]
        return [r for r in rows.values() if chunk_doc(r, pre) in g.docs]

    def is_gold(r, q):
        g = gold_of(q, pre)
        return bool(section_matches(r, pre, g)) if g.sections else chunk_doc(r, pre) in g.docs

    def anyfmt(r):
        ref = dt.date.fromisoformat(r["event_date"]) if r.get("event_date") else None
        return {x.isoformat() for x in mentioned_dates(r["text"], ref)}

    def spans(r, d):
        ref = dt.date.fromisoformat(r["event_date"]) if r.get("event_date") else None
        ds = [x.isoformat() for x in mentioned_dates(r["text"], ref)]
        return bool(ds) and min(ds) <= d < max(ds)

    cases = []
    for pq in res["questions"]:
        q = qs[pq["id"]]
        top = [rows[c] for c in pq.get("top_k", [])]
        causes, mixed = [], []
        if q["type"] == "시점 고정":
            a = q["as_of"]
            bad = [rows[c] for c in pq.get("contaminated", [])]
            parsed = (pq.get("intent") or {}).get("as_of")
            if bad and parsed and parsed > a and any(a < (r.get("event_date") or "") <= parsed for r in bad):
                causes.append("as_of_convention")
            if any((r.get("date_min") or "") <= a for r in bad):
                causes.append("mixed_time_chunk")
            if pq.get("recall_doc", 1) < 1 or pq.get("recall_section", 1) < 1:
                gr = gold_rows(q)
                if gr and all((r.get("date_min") or "") > a for r in gr):
                    causes.append("gold_filtered")
                else:
                    causes.append("relevance")
            mixed = [r["chunk_id"] for r in top if spans(r, a)]
        elif q["type"] == "현재 값":
            if pq.get("recency", 1) < 1 and top:
                t1, latest = top[0], pq.get("latest_gold_date")
                if is_gold(t1, q):
                    causes.append("stale_top1_date")
                elif (t1.get("date_max") or "") >= (latest or ""):
                    causes.append("newer_non_gold")
                else:
                    causes.append("older_non_gold")
            if not any(is_gold(r, q) for r in top):
                causes.append("gold_not_in_pool")
            latest = pq.get("latest_gold_date")
            if latest:
                mixed = [r["chunk_id"] for r in top if spans(r, latest)]
        elif q["type"] == "변화 추적":
            hist = [rows[c] for c in pq.get("hist_ids", [])]
            hist_any = set().union(*(anyfmt(r) for r in hist)) if hist else set()
            for d in pq.get("history_missing", []):
                if d in hist_any:
                    causes.append("metric_format")
                elif any(r.get("event_date") == d or d in anyfmt(r) for r in rows.values()):
                    causes.append("point_not_in_pool")
                else:
                    causes.append("point_not_in_corpus")
            pts = sorted({str(c["date"])[:10] for c in q.get("change_points", [])})
            mixed = [r["chunk_id"] for r in hist if sum(d in anyfmt(r) for d in pts) >= 2]
        else:
            hub = any(len(r.get("links") or []) >= 3 and not is_gold(r, q) for r in top)
            if pq.get("recall_doc", 1) < 1 or pq.get("recall_section", 1) < 1:
                causes.append("hub_chunks" if hub else ("doc_miss" if pq.get("recall_doc", 1) < 1 else "section_miss"))
        if causes:
            cases.append((q, pq, sorted(set(causes)), mixed, top))

    by_type = collections.defaultdict(list)
    for c in cases:
        by_type[c[0]["type"]].append(c)
    lines = [f"# 모드 {mode} 실패 사례 ({args.results.name})", "",
             "원인은 규칙으로 1차 분류한 것이다(스크립트 docstring 참고). 질문 원문·청크 본문 포함: 비공개.", "",
             "## 요약", "", "| 유형 | 실패 질문 | 원인(질문 수) | 시점 섞인 청크가 상위에 있는 질문 |", "|---|---|---|---|"]
    for t in ("일반", "시점 고정", "현재 값", "변화 추적"):
        cs = by_type.get(t, [])
        cnt = collections.Counter(x for c in cs for x in c[2])
        lines.append(f"| {t} | {len(cs)} | {', '.join(f'{k} {v}' for k, v in cnt.most_common())} | {sum(bool(c[3]) for c in cs)} |")
    for t in ("일반", "시점 고정", "현재 값", "변화 추적"):
        lines += ["", f"## {t}", ""]
        for q, pq, causes, mixed, top in by_type.get(t, []):
            lines += [f"### {q['id']}: {', '.join(causes)}",
                      f"- 질문: {q['question']}",
                      f"- 정답: {q.get('answer', '')}",
                      f"- 라벨 as_of {q.get('as_of')}, 해석 {pq.get('intent')}",
                      f"- 지표: recall_doc {pq.get('recall_doc')}, recall_section {pq.get('recall_section')}, "
                      f"오염 {pq.get('contamination')}, 최신성 {pq.get('recency')}, 이력 {pq.get('history')}, 누락 {pq.get('history_missing')}",
                      f"- 시점 섞인 상위 청크: {len(mixed)}"]
            for r in top:
                mark = "G" if is_gold(r, q) else " "
                txt = " ".join(r["text"].split())[:140]
                lines.append(f"  - [{mark}] {r['chunk_id'].split('/')[-1]} ev {r.get('event_date')} [{r.get('date_min')}..{r.get('date_max')}] {txt}")
            lines.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(cases)} failing questions -> {args.out}")


if __name__ == "__main__":
    main()
