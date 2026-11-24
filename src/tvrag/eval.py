"""Evaluation runner: Recall@5, MRR, as-of contamination, recency accuracy, history completeness, with CIs.

Eval set: JSONL, one question per line with fields
  id, type (일반 | 시점 고정 | 현재 값 | 변화 추적), question, answer,
  gold_docs [path], gold_sections [{doc, heading}], as_of (시점 고정),
  change_points [{date, value}] (변화 추적), confidence, confidence_reason,
  reference_date, snapshot

Metric definitions (all per question, then averaged; CIs by bootstrap over questions):
  recall_doc@k      share of gold_docs that own at least one of the top-k chunks
  recall_section@k  share of gold_sections matched by at least one top-k chunk
  mrr_doc / mrr_section
                    1 / rank of the first chunk (in the retrieved list of `depth`
                    chunks) that belongs to a gold doc / matches a gold section; 0 if none
  contamination@k   (시점 고정) share of top-k chunks with event_date > as_of
  recency           (현재 값) top-1 chunk's event_date equals the latest event_date among
                    gold chunks (chunks matching gold_sections, else chunks of gold_docs)
  recency_strict    same, and the top-1 chunk is itself a gold chunk
  history           (변화 추적) share of change_points dates covered by the top-`history_k`
                    chunks. A chunk covers a date if its event_date equals it OR its text
                    states that full date (yyyy-mm-dd, yyyy.mm.dd, yyyy/mm/dd, yyyy년 m월 d일).
                    Why both: a dated log entry is evidence by its event_date; a summary or
                    topic note that writes "on 2025-03-17 the limit went to X" is evidence
                    too, but its own event_date is the note's date, so event_date alone
                    would undercount it. history_strict (event_date only) is reported too.
  history_anyfmt    (added after the first private run in S5, marked 사후 in tables) as
                    history, but text dates are read with tvrag.dates, so year-less "m/d"
                    and "m월 d일" mentions count too. The vault writes many change dates
                    only as "m/d", which `history` cannot credit.
  value_acc         (mode D only, S6) automatic number match of D's fact answer against
                    the eval set: 시점 고정 / 현재 값 = share of D's answer rows whose
                    number appears in `answer`; 변화 추적 = share of numeric change_points
                    matched by a D row within ±3 days with the same number. Questions
                    whose gold has no number are left out; no D answer scores 0
                    (tvrag.facts.value_scores).
  value_answered    (mode D) share of those numeric questions D answered from the table

Path matching: gold_docs may be written relative to another root than chunk doc_path.
Both sides are cut after `eval.doc_prefix` (from the config, so no vault name is hard-
coded) when the prefix occurs, slashes are normalised and a trailing ".md" is dropped.

Usage: python -m tvrag.eval --config configs/demo.yaml --modes A B
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from tvrag.dates import mentioned_dates

TYPES = ("일반", "시점 고정", "현재 값", "변화 추적")
REQUIRED = ("id", "type", "question", "gold_docs")

METRICS = (
    ("recall_doc", "Recall@k 문서"),
    ("recall_section", "Recall@k 섹션"),
    ("mrr_doc", "MRR 문서"),
    ("mrr_section", "MRR 섹션"),
    ("contamination", "시점 오염률"),
    ("recency", "최신성 정확도"),
    ("recency_strict", "최신성(정답 청크)"),
    ("history", "이력 완결도"),
    ("history_strict", "이력(event_date만)"),
    ("history_anyfmt", "이력(m/d 포함, 사후)"),
    ("value_acc", "값 정확도(D)"),
    ("value_answered", "D 응답률"),
)


# ---------------------------------------------------------------- loading
def load_eval_set(path: str | Path) -> list[dict]:
    qs = []
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            q = json.loads(line)
            missing = [k for k in REQUIRED if k not in q]
            if missing:
                raise ValueError(f"line {n}: missing {missing}")
            if q["type"] not in TYPES:
                raise ValueError(f"line {n}: unknown type {q['type']!r}")
            if q["type"] == "시점 고정" and not q.get("as_of"):
                raise ValueError(f"line {n}: 시점 고정 needs as_of")
            if q["type"] == "변화 추적" and not q.get("change_points"):
                raise ValueError(f"line {n}: 변화 추적 needs change_points")
            qs.append(q)
    ids = [q["id"] for q in qs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate question ids")
    return qs


# ---------------------------------------------------------------- matching
def normalize_path(path: str, prefix: str | None = None) -> str:
    p = path.replace("\\", "/").strip().lstrip("./")
    if prefix:
        pre = prefix.replace("\\", "/").strip("/") + "/"
        i = p.find(pre)
        if i >= 0:
            p = p[i + len(pre):]
    return p[:-3] if p.lower().endswith(".md") else p


def normalize_heading(h: str) -> str:
    return " ".join(h.lstrip("#").split()).casefold()


@dataclass
class Gold:
    docs: set[str]
    sections: list[tuple[str, str]]  # (normalized doc, normalized heading)


def gold_of(q: dict, prefix: str | None) -> Gold:
    return Gold(
        {normalize_path(d, prefix) for d in q.get("gold_docs") or []},
        [(normalize_path(s["doc"], prefix), normalize_heading(s["heading"])) for s in q.get("gold_sections") or []],
    )


def chunk_doc(row: dict, prefix: str | None) -> str:
    return normalize_path(row["doc_path"], prefix)


def chunk_sections(row: dict) -> set[str]:
    return {normalize_heading(h) for h in [*(row.get("headings") or []), *(row.get("heading_path") or [])] if h}


def section_matches(row: dict, prefix: str | None, gold: Gold) -> list[int]:
    """Indices of gold sections this chunk matches."""
    doc, heads = chunk_doc(row, prefix), chunk_sections(row)
    return [i for i, (d, h) in enumerate(gold.sections) if d == doc and h in heads]


DATE_PATTERNS = (
    re.compile(r"(?<!\d)(\d{4})[-./](\d{1,2})[-./](\d{1,2})(?!\d)"),
    re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일"),
)


def dates_in_text(text: str) -> set[str]:
    out = set()
    for pat in DATE_PATTERNS:
        for m in pat.finditer(text):
            try:
                out.add(dt.date(*map(int, m.groups())).isoformat())
            except ValueError:
                pass
    return out


def check_gold(questions: list[dict], rows: list[dict], prefix: str | None) -> list[str]:
    """Gold docs / sections that match no chunk (usually a path-prefix or heading typo)."""
    docs = {chunk_doc(r, prefix) for r in rows}
    problems = []
    for q in questions:
        g = gold_of(q, prefix)
        for d in sorted(g.docs - docs):
            problems.append(f"{q['id']}: gold doc not in index: {d}")
        hit = {i for r in rows for i in section_matches(r, prefix, g)}
        for i, (d, h) in enumerate(g.sections):
            if i not in hit:
                problems.append(f"{q['id']}: gold section not in index: {d} # {h}")
    return problems


# ---------------------------------------------------------------- per question
def score_question(q: dict, ranked: list[dict], all_rows: list[dict], prefix: str | None, k: int, history_k: int,
                   top: list[dict] | None = None, hist: list[dict] | None = None) -> dict:
    """`ranked`: retrieved chunk rows in rank order (depth) for MRR. `top` / `hist`: the
    lists retrieved with k / history_k (default: prefixes of `ranked`; a mode whose
    output for a smaller k is not a prefix, such as time-sorted mode C, passes them)."""
    gold = gold_of(q, prefix)
    top = ranked[:k] if top is None else top[:k]
    hist = ranked[:history_k] if hist is None else hist[:history_k]
    res: dict = {"id": q["id"], "type": q["type"]}

    doc_hits = [chunk_doc(r, prefix) in gold.docs for r in ranked]
    sec_hits = [section_matches(r, prefix, gold) for r in ranked]
    res["ranked"] = [
        {
            "rank": i + 1,
            "chunk_id": r["chunk_id"],
            "event_date": r.get("event_date"),
            "gold_doc": doc_hits[i],
            "gold_section": bool(sec_hits[i]),
        }
        for i, r in enumerate(ranked)
    ]
    first_doc = next((i + 1 for i, h in enumerate(doc_hits) if h), None)
    first_sec = next((i + 1 for i, h in enumerate(sec_hits) if h), None)
    res["first_doc_rank"], res["first_section_rank"] = first_doc, first_sec
    res["top_k"] = [r["chunk_id"] for r in top]

    if gold.docs:
        found = {chunk_doc(r, prefix) for r in top} & gold.docs
        res["recall_doc"] = len(found) / len(gold.docs)
        res["mrr_doc"] = 1 / first_doc if first_doc else 0.0
        res["missed_docs"] = sorted(gold.docs - found)
    if gold.sections:
        covered = {i for r in top for i in section_matches(r, prefix, gold)}
        res["recall_section"] = len(covered) / len(gold.sections)
        res["mrr_section"] = 1 / first_sec if first_sec else 0.0

    if q["type"] == "시점 고정" and q.get("as_of"):
        bad = [r["chunk_id"] for r in top if r.get("event_date") and r["event_date"] > q["as_of"]]
        res["contamination"] = len(bad) / len(top) if top else 0.0
        res["contaminated"] = bad

    if q["type"] == "현재 값":
        if gold.sections:
            gold_rows = [r for r in all_rows if section_matches(r, prefix, gold)]
        else:
            gold_rows = [r for r in all_rows if chunk_doc(r, prefix) in gold.docs]
        dates = [r["event_date"] for r in gold_rows if r.get("event_date")]
        if dates and top:
            latest = max(dates)
            res["latest_gold_date"] = latest
            res["top1_date"] = top[0].get("event_date")
            res["recency"] = float(top[0].get("event_date") == latest)
            top1_gold = (bool(section_matches(top[0], prefix, gold)) if gold.sections
                         else chunk_doc(top[0], prefix) in gold.docs)
            res["recency_strict"] = float(res["recency"] == 1.0 and top1_gold)

    if q["type"] == "변화 추적" and q.get("change_points"):
        want = sorted({str(c["date"])[:10] for c in q["change_points"]})
        hk = hist
        ev = {r.get("event_date") for r in hk}
        mention = set().union(*(dates_in_text(r.get("text", "")) for r in hk)) if hk else set()
        cov = [d for d in want if d in ev or d in mention]
        res["history"] = len(cov) / len(want)
        res["history_strict"] = len([d for d in want if d in ev]) / len(want)
        res["history_missing"] = [d for d in want if d not in cov]
        anyfmt = set()
        for r in hk:
            ref = dt.date.fromisoformat(r["event_date"]) if r.get("event_date") else None
            anyfmt |= {d.isoformat() for d in mentioned_dates(r.get("text", ""), ref)}
        res["history_anyfmt"] = len([d for d in want if d in ev or d in anyfmt]) / len(want)
        res["hist_ids"] = [r["chunk_id"] for r in hk]
    return res


# ---------------------------------------------------------------- aggregation
def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean and percentile CI, resampling questions with replacement."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def summarize(per_q: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    groups = {"전체": per_q, **{t: [r for r in per_q if r["type"] == t] for t in TYPES}}
    out = {}
    for g, rows in groups.items():
        if not rows:
            continue
        out[g] = {"n": len(rows)}
        for key, _ in METRICS:
            vals = [r[key] for r in rows if key in r]
            if vals:
                m, lo, hi = bootstrap_ci(vals, n_boot, seed)
                out[g][key] = {"mean": m, "lo": lo, "hi": hi, "n": len(vals)}
    return out


def _fmt(cell: dict | None) -> str:
    if not cell:
        return "-"
    return f"{cell['mean']:.2f} [{cell['lo']:.2f}, {cell['hi']:.2f}]"


def markdown_table(summaries: dict[str, dict], meta: dict) -> str:
    """One table per mode, rows = question groups, columns = metrics present."""
    k = meta["k"]
    lines = [
        f"# 평가 결과 ({meta['eval_set']}, 스냅샷 {meta.get('snapshot')}, {meta['date']})",
        "",
        f"질문 {meta['n_questions']}개, k={k}, MRR 깊이 {meta['depth']}, 이력 k={meta['history_k']}, "
        f"부트스트랩 {meta['n_boot']}회 95% 구간(질문 단위 재표집). 칸: 평균 [하한, 상한].",
        "",
    ]
    for mode, summ in summaries.items():
        present = [(key, label.replace("@k", f"@{k}")) for key, label in METRICS if any(key in summ[g] for g in summ)]
        lines += [f"## 모드 {mode}", "", "| 그룹 | n | " + " | ".join(l for _, l in present) + " |",
                  "|---|---|" + "---|" * len(present)]
        for g, row in summ.items():
            lines.append(f"| {g} | {row['n']} | " + " | ".join(_fmt(row.get(key)) for key, _ in present) + " |")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- runner
def run_eval(
    questions: list[dict],
    search: Callable[[dict, str, int], list[str]],
    rows_by_id: dict[str, dict],
    modes: list[str],
    prefix: str | None = None,
    k: int = 5,
    depth: int = 20,
    history_k: int = 10,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, dict]:
    """`search(question_dict, mode, n) -> [chunk_id, ...]` or `([chunk_id, ...], info)`.
    Each cutoff (depth for MRR, k, history_k) is retrieved separately, so modes whose
    top-k is not a prefix of their top-depth are scored on what they return for k.
    Returns {mode: {summary, questions}}."""
    all_rows = list(rows_by_id.values())
    out = {}
    for mode in modes:
        per_q = []
        for q in questions:
            lists, info = {}, None
            for n in sorted({depth, k, history_k}, reverse=True):
                got = search(q, mode, n)
                ids, inf = got if isinstance(got, tuple) else (got, None)
                lists[n] = [rows_by_id[c] for c in ids]
                info = info or inf
            res = score_question(q, lists[depth], all_rows, prefix, k, history_k, lists[k], lists[history_k])
            if info and "facts" in info:
                from tvrag.facts import value_scores

                res.update(value_scores(q, info["facts"]))
            if info:
                res["intent"] = info
            per_q.append(res)
        out[mode] = {"summary": summarize(per_q, n_boot, seed), "questions": per_q}
    return out


MODE_LABELS = {"CL": "C(정답 라벨)", "DL": "D(정답 라벨)"}
LABEL_MODES = {"CL": "C", "DL": "D"}  # same mode, eval labels (type, as_of) instead of question reading

COMPARE_ROWS = (
    ("recall_doc", None), ("recall_section", None), ("mrr_doc", None),
    ("contamination", "시점 고정"), ("recency", "현재 값"), ("recency_strict", "현재 값"),
    ("history", "변화 추적"), ("history_strict", "변화 추적"), ("history_anyfmt", "변화 추적"),
    ("value_acc", None), ("value_answered", None),
)


def comparison_table(summaries: dict[str, dict], k: int) -> str:
    """Modes side by side: rows = (group, metric), cells = mean [CI]."""
    labels = dict(METRICS)
    modes = list(summaries)
    head = "| 그룹 | 지표 | " + " | ".join(MODE_LABELS.get(m, m) for m in modes) + " |"
    lines = [head, "|---|---|" + "---|" * len(modes)]
    groups = ["전체", *TYPES]
    for g in groups:
        for key, only in COMPARE_ROWS:
            if only and g not in (only, "전체"):
                continue
            cells = [summaries[m].get(g, {}).get(key) for m in modes]
            if not any(cells):
                continue
            label = labels[key].replace("@k", f"@{k}")
            lines.append(f"| {g} | {label} | " + " | ".join(_fmt(c) for c in cells) + " |")
    return "\n".join(lines)


def question_reading(questions: list[dict], default_ref: str | None = None, tol_days: int = 7) -> dict:
    """Compare tvrag.query's reading with the eval labels (labels are used for scoring only).
    Returns accuracy numbers and per-id flags (no question text)."""
    from tvrag.query import parse_question

    per, conf = [], {}
    for q in questions:
        ref = q.get("reference_date") or default_ref or dt.date.today().isoformat()
        it = parse_question(q["question"], ref)
        row = {"id": q["id"], "label": q["type"], "pred": it.qtype, "type_ok": it.qtype == q["type"]}
        if q["type"] == "시점 고정":
            row["as_of_label"], row["as_of_pred"] = q.get("as_of"), it.as_of
            if it.as_of and q.get("as_of"):
                diff = abs((dt.date.fromisoformat(it.as_of) - dt.date.fromisoformat(q["as_of"][:10])).days)
                row["as_of_diff_days"] = diff
            row["as_of_exact"] = it.as_of == (q.get("as_of") or "")[:10]
            row["as_of_close"] = row.get("as_of_diff_days") is not None and row["as_of_diff_days"] <= tol_days
        per.append(row)
        conf.setdefault(q["type"], {}).setdefault(it.qtype, 0)
        conf[q["type"]][it.qtype] += 1
    asof = [r for r in per if r["label"] == "시점 고정"]
    return {
        "type_accuracy": sum(r["type_ok"] for r in per) / len(per) if per else None,
        "as_of_exact": sum(r["as_of_exact"] for r in asof) / len(asof) if asof else None,
        "as_of_within_days": tol_days,
        "as_of_close": sum(r["as_of_close"] for r in asof) / len(asof) if asof else None,
        "confusion": conf,
        "questions": per,
    }


def reading_table(rd: dict) -> str:
    lines = [f"질문 유형 정확도 {rd['type_accuracy']:.2f}, 시점 고정의 as_of 일치 {rd['as_of_exact']:.2f} "
             f"(±{rd['as_of_within_days']}일 이내 {rd['as_of_close']:.2f}).", "",
             "| 라벨 \\ 해석 | " + " | ".join(TYPES) + " |", "|---|" + "---|" * len(TYPES)]
    for t in TYPES:
        row = rd["confusion"].get(t, {})
        lines.append(f"| {t} | " + " | ".join(str(row.get(p, 0)) for p in TYPES) + " |")
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(cfg: dict, modes: list[str], eval_set: str | Path | None = None, out_dir: str | Path | None = None,
             date: str | None = None) -> dict:
    from tvrag.config import data_paths
    from tvrag.retrieve import Retriever

    paths = data_paths(cfg)
    ecfg = cfg.get("eval") or {}
    eval_path = Path(eval_set) if eval_set else paths["eval_set"]
    if not eval_path or not eval_path.exists():
        raise FileNotFoundError(f"eval set not found: {eval_path} (set paths.eval_set or pass --eval)")
    questions = load_eval_set(eval_path)
    retriever = Retriever.from_config(cfg)
    rows_by_id = {r["chunk_id"]: r for r in retriever.index.rows}
    prefix = ecfg.get("doc_prefix") or None
    k, depth, history_k = int(ecfg.get("k", 5)), int(ecfg.get("depth", 20)), int(ecfg.get("history_k", 10))
    n_boot, seed = int(ecfg.get("bootstrap", 2000)), int(ecfg.get("seed", 0))

    gold_problems = check_gold(questions, retriever.index.rows, prefix)
    for msg in gold_problems:
        print(f"[eval] warning: {msg}")

    def search(q: dict, mode: str, n: int):
        ref = q.get("reference_date")
        if mode in ("C", "D"):  # reads type and as-of from the question text itself
            hits = retriever.retrieve(q["question"], mode, n, reference_date=ref)
        elif mode in LABEL_MODES:  # same mode, labels instead of question reading
            hits = retriever.retrieve(q["question"], LABEL_MODES[mode], n, q.get("as_of"), qtype=q["type"],
                                      reference_date=ref)
        else:
            hits = retriever.retrieve(q["question"], mode, n, q.get("as_of"))
        info = None
        if mode in ("C", "CL", "D", "DL") and hits:
            c = hits[0].components
            info = {"qtype": c.get("qtype"), "as_of": c.get("as_of"), "cue": c.get("cue")}
            if mode in ("D", "DL"):
                info["facts"] = c.get("facts") or []
                info["fact_attrs"] = c.get("fact_attrs") or []
        return [h.chunk_id for h in hits], info

    results = run_eval(questions, search, rows_by_id, modes, prefix, k, depth, history_k, n_boot, seed)

    date = date or dt.date.today().isoformat()
    meta = {
        "date": date,
        "eval_set": eval_path.name,
        "eval_sha256": _sha256(eval_path),
        "snapshot": cfg.get("snapshot_id"),
        "index_manifest": retriever.index.manifest,
        "n_questions": len(questions),
        "k": k,
        "depth": depth,
        "history_k": history_k,
        "n_boot": n_boot,
        "seed": seed,
        "retrieve": cfg.get("retrieve") or {},
        "gold_problems": gold_problems,
    }
    if any(m in ("D", "DL") for m in modes):
        meta["facts"] = {"table": str(paths.get("facts")), "rows": len(retriever.facts.facts) if retriever.facts else 0,
                         "sha256": _sha256(Path(paths["facts"])) if retriever.facts else None,
                         "config": cfg.get("facts") or {}}
    if any(m in ("C", "CL", "D", "DL") for m in modes):
        meta["question_reading"] = question_reading(questions)
    out_dir = Path(out_dir) if out_dir else paths["results"]
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for mode, res in results.items():
        p = out_dir / f"{date}_{mode}.json"
        p.write_text(json.dumps({"meta": {**meta, "mode": mode}, **res}, ensure_ascii=False, indent=1) + "\n",
                     encoding="utf-8", newline="\n")
        (out_dir / f"{date}_{mode}.md").write_text(
            markdown_table({mode: res["summary"]}, meta) + "\n", encoding="utf-8", newline="\n")
        written.append(p)
    tag = "".join(modes)
    table = markdown_table({MODE_LABELS.get(m, m): r["summary"] for m, r in results.items()}, meta)
    if len(results) > 1:
        table += "\n## 모드 비교\n\n" + comparison_table({m: r["summary"] for m, r in results.items()}, k) + "\n"
    if "question_reading" in meta:
        table += "\n## 질문 해석 (모드 C)\n\n" + reading_table(meta["question_reading"]) + "\n"
    (out_dir / f"{date}_{tag}.md").write_text(table + "\n", encoding="utf-8", newline="\n")
    return {"meta": meta, "results": results, "table": table, "out_dir": str(out_dir)}


def main(argv: list[str] | None = None) -> None:
    from tvrag.config import load_config

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--modes", nargs="+", default=["A", "B"])
    ap.add_argument("--eval", default=None, help="eval set JSONL (default: paths.eval_set)")
    ap.add_argument("--out", default=None, help="results folder (default: paths.results)")
    ap.add_argument("--date", default=None, help="date tag for result files (default: today)")
    args = ap.parse_args(argv)
    r = evaluate(load_config(args.config), [m.upper() for m in args.modes], args.eval, args.out, args.date)
    print(r["table"])
    print(f"-> {r['out_dir']}")


if __name__ == "__main__":
    main()
