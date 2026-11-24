"""Compare mode C settings on a development eval set (the demo set; never the private one).

Runs mode C with the gold question type and as-of ("CL") so that only the time logic
varies, one setting at a time around the configured defaults.

Usage: python scripts/tune_c.py --config configs/demo.yaml
"""
from __future__ import annotations

import argparse

from tvrag.config import data_paths, load_config
from tvrag.eval import load_eval_set, run_eval
from tvrag.retrieve import Retriever


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ecfg = cfg.get("eval") or {}
    qs = load_eval_set(data_paths(cfg)["eval_set"])
    r = Retriever.from_config(cfg)
    rows = {x["chunk_id"]: x for x in r.index.rows}
    k, depth, hk = ecfg.get("k", 5), ecfg.get("depth", 20), ecfg.get("history_k", 10)

    def search(q, mode, n):
        return [h.chunk_id for h in r.retrieve(q["question"], "C", n, q.get("as_of"), qtype=q["type"],
                                               reference_date=q.get("reference_date"))]

    def score(qtype, keys):
        sub = [q for q in qs if q["type"] == qtype]
        s = run_eval(sub, search, rows, ["CL"], ecfg.get("doc_prefix") or None, k, depth, hk, n_boot=10)["CL"]["summary"][qtype]
        return "  ".join(f"{key}={s[key]['mean']:.2f}" for key in keys if key in s)

    base = (r.as_of_filter, r.half_life_days, r.change_pool)
    print("시점 고정: as_of_filter")
    for v in ("pre", "post"):
        r.as_of_filter = v
        print(f"  {v:5s}", score("시점 고정", ["recall_doc", "recall_section", "mrr_doc", "contamination"]))
    r.as_of_filter = base[0]
    print("현재 값: half_life_days (0 = no decay)")
    for v in (0, 7, 14, 30, 60, 90, 180, 365):
        r.half_life_days = v
        print(f"  {v:5d}", score("현재 값", ["recall_doc", "recall_section", "mrr_doc", "recency", "recency_strict"]))
    r.half_life_days = base[1]
    print("변화 추적: change_pool")
    for v in (10, 20, 30, 50):
        r.change_pool = v
        print(f"  {v:5d}", score("변화 추적", ["recall_doc", "recall_section", "history", "history_strict"]))
    r.change_pool = base[2]


if __name__ == "__main__":
    main()
