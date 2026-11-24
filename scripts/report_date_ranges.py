"""Report on content date ranges (date_min/date_max) for chunks whose event date came
from `created`, i.e. the file date rather than a record date.

Usage: python scripts/report_date_ranges.py <chunks.jsonl> <out.md> [--doc-type topic] [--samples 10]
The report quotes chunk text, so write it to a private location.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--doc-type", default="topic")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.chunks.open(encoding="utf-8") if line.strip()]
    sel = [r for r in rows if r["date_source"] == "created" and r["doc_type"] == args.doc_type]
    has = [r for r in sel if r.get("n_date_mentions", 0) > 0]
    span = lambda r: (dt.date.fromisoformat(r["date_max"]) - dt.date.fromisoformat(r["date_min"])).days

    lines = [
        f"# 날짜 구간 점검: date_source=created, doc_type={args.doc_type}",
        "",
        f"- 청크 파일: `{args.chunks.name}`, 전체 {len(rows)}청크 중 대상 {len(sel)}청크",
        f"- 본문 날짜 언급이 있어 구간을 받은 청크: {len(has)}, 언급이 없어 event_date(created)를 그대로 쓴 청크: {len(sel) - len(has)}",
        f"- 대상 청크의 event_date 값: {dict(collections.Counter(r['event_date'] for r in sel))}",
        "",
        "## 구간 길이(일) 분포 (언급 있는 청크)",
        "",
        "| 구간 길이 | 청크 수 |",
        "|---|---|",
    ]
    bins = [(0, 0, "0 (하루)"), (1, 7, "1-7"), (8, 31, "8-31"), (32, 92, "32-92"), (93, 10**6, "93 이상")]
    for lo, hi, label in bins:
        lines.append(f"| {label} | {sum(lo <= span(r) <= hi for r in has)} |")
    lines += ["", "## date_max 월별 분포 (언급 있는 청크)", "", "| 월 | 청크 수 |", "|---|---|"]
    for m, n in sorted(collections.Counter(r["date_max"][:7] for r in has).items()):
        lines.append(f"| {m} | {n} |")
    lines += ["", "## date_min 월별 분포 (언급 있는 청크)", "", "| 월 | 청크 수 |", "|---|---|"]
    for m, n in sorted(collections.Counter(r["date_min"][:7] for r in has).items()):
        lines.append(f"| {m} | {n} |")
    lines += ["", "## 언급 개수 분포", "", "| 언급 수 | 청크 수 |", "|---|---|"]
    for lo, hi, label in [(0, 0, "0"), (1, 1, "1"), (2, 5, "2-5"), (6, 20, "6-20"), (21, 10**6, "21 이상")]:
        lines.append(f"| {label} | {sum(lo <= r.get('n_date_mentions', 0) <= hi for r in sel)} |")

    # Dates written right before a wiki link to a note named with the same date, e.g.
    # "- 2026-01-02 [[2026-01-02 some session|...]] - summary". Link text itself never
    # counts (tvrag.dates strips links); this counts the plain-text date beside it.
    import re

    adj = re.compile(r"(\d{4}-\d{2}-\d{2})\s*(?:\S{0,3}\s*)?\[\[[^\]]*?\1")
    n_adj = sum(len(adj.findall(r["text"])) for r in rows)
    n_adj_sel = sum(len(adj.findall(r["text"])) for r in sel)
    lines += [
        "",
        "## 링크 옆 날짜",
        "",
        f"- 같은 날짜로 시작하는 노트 링크 바로 앞에 쓴 날짜: 전체 {n_adj}건, 대상 청크 {n_adj_sel}건.",
        "- 링크 안 날짜는 세지 않는다(다른 노트의 이름). 링크 옆 날짜는 세는데, 이런 줄은 그날 세션 내용을 요약한 것이라",
        "  그 날짜가 요약된 사건의 날짜이기도 하다. 다만 주제노트의 \"근거 세션 목록\" 청크는 이 때문에 구간이 목록 전체로 넓어진다.",
    ]

    rng = random.Random(args.seed)
    picks = rng.sample(sel, min(args.samples, len(sel)))
    lines += ["", f"## 표본 {len(picks)}개 (seed {args.seed})", ""]
    for r in sorted(picks, key=lambda r: r["chunk_id"]):
        text = " ".join(r["text"].split())
        lines += [
            f"### {r['chunk_id']}",
            f"- 구간 {r['date_min']} ~ {r['date_max']} ({span(r)}일), 언급 {r.get('n_date_mentions', 0)}개, event_date {r['event_date']}",
            f"- 헤딩: {' > '.join(r['heading_path'])}",
            f"- 본문 앞부분: {text[:300]}",
            "",
        ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(sel)} chunks ({len(has)} with mentions) -> {args.out}")


if __name__ == "__main__":
    main()
