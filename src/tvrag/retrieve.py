"""Search modes A-D: A vector, B +BM25 (RRF), C +time (as-of filter, recency, multi-point history), D +fact table.

Common interface: retrieve(question, mode, k, as_of=None) -> list[Hit]
Every hit carries the mode that produced it and its evidence chunk (id, doc, headings,
event date, content date range, text). `as_of` is recorded but not used by A and B.
Mode C = B plus time, by question type (read by tvrag.query):
  일반       same as B
  시점 고정  drop chunks with date_min > as_of (pre-filter in both retrievers by default)
  현재 값    B candidates re-ranked by score * 0.5 ** (age / half_life), age = now - date_max
  변화 추적  wider B pool, one chunk per distinct date_max in relevance order, sorted by time
Mode D = C plus the fact table (tvrag.facts): the same question reading picks fact rows
(value valid at as_of / now, or the history); their evidence chunks go first in the
ranked list, the mode C list fills the rest. The fact answer itself (value, validity,
condition, nature, evidence) rides on the first hit as components["facts"]. Without a
fact table, or when no attribute is named, or for 일반 questions, D = C.

Usage: python -m tvrag.retrieve --config configs/local.yaml -m B -k 5 "question"
"""
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from tvrag.index import Embedder, Index, Tokenizer, load_index

MODES = ("A", "B", "C", "D")


@dataclass
class Hit:
    chunk_id: str
    score: float
    rank: int
    mode: str
    doc_path: str
    heading_path: list[str]
    headings: list[str]
    event_date: str | None
    text: str
    components: dict = field(default_factory=dict)  # per-retriever ranks/scores, mode C intent
    date_min: str | None = None
    date_max: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank), rank from 1.

    Ties are broken by best single rank, then by id, so the output is deterministic.
    """
    score: dict[str, float] = {}
    best: dict[str, int] = {}
    for ranking in rankings:
        for r, cid in enumerate(ranking, start=1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + r)
            best[cid] = min(best.get(cid, r), r)
    return sorted(score.items(), key=lambda kv: (-kv[1], best[kv[0]], kv[0]))


class Retriever:
    def __init__(self, index: Index, embedder: Embedder, tokenizer: Tokenizer | None = None, cfg: dict | None = None):
        cfg = cfg or {}
        self.index = index
        self.embedder = embedder
        self.tokenizer = tokenizer or Tokenizer()
        self.rrf_k = int(cfg.get("rrf_k", 60))
        self.candidates = int(cfg.get("candidates", 50))
        c = cfg.get("c") or {}
        self.as_of_filter = c.get("as_of_filter", "pre")  # pre | post
        self.half_life_days = float(c.get("half_life_days", 60))
        self.change_pool = int(c.get("change_pool", 30))
        self.reference_date = c.get("reference_date")  # default "now" for mode C
        self._qvec: dict[str, np.ndarray] = {}
        self.facts = None  # tvrag.facts.FactStore for mode D

    @classmethod
    def from_config(cls, cfg: dict, index_dir: str | Path | None = None) -> "Retriever":
        from tvrag.config import data_paths

        e = cfg["embedding"]
        emb = Embedder(e["model"], e.get("device", "cpu"), e.get("batch_size", 8), e.get("normalize", True))
        r = cls(load_index(index_dir or data_paths(cfg)["index"]), emb, cfg=cfg.get("retrieve") or {})
        from tvrag.facts import FactStore

        r.facts = FactStore.from_config(cfg, r.index.rows)
        return r

    # --- single retrievers -------------------------------------------------
    def _encode(self, question: str) -> np.ndarray:
        if question not in self._qvec:
            self._qvec[question] = self.embedder.encode([question])[0]
        return self._qvec[question]

    def vector_search(self, question: str, n: int, max_date_min: str | None = None) -> list[tuple[str, float]]:
        """Exact cosine search in LanceDB (no ANN index: brute force over all rows).
        `max_date_min`: pre-filter to chunks whose content starts on or before that day."""
        q = self._encode(question)
        search = self.index.table.search(q.tolist()).distance_type("cosine")
        if max_date_min:
            search = search.where(f"date_min IS NULL OR date_min <= '{max_date_min}'", prefilter=True)
        res = search.select(["chunk_id", "_distance"]).limit(n).to_list()
        hits = [(r["chunk_id"], 1.0 - float(r["_distance"])) for r in res]
        # stable order for equal similarity
        return sorted(hits, key=lambda h: (-round(h[1], 6), h[0]))

    def bm25_search(self, question: str, n: int, max_date_min: str | None = None) -> list[tuple[str, float]]:
        toks = self.tokenizer(question)
        if not toks:
            return []
        scores = np.asarray(self.index.bm25.get_scores(toks))
        rows = self.index.rows
        idx = range(len(scores))
        if max_date_min:
            idx = [i for i in idx if not rows[i].get("date_min") or rows[i]["date_min"] <= max_date_min]
        order = sorted(idx, key=lambda i: (-scores[i], i))
        return [(rows[i]["chunk_id"], float(scores[i])) for i in order[:n] if scores[i] > 0]

    def fused(self, question: str, n: int, max_date_min: str | None = None) -> list[tuple[str, float, dict]]:
        """Mode B core: top-n of the vector and BM25 lists fused by RRF."""
        vec = self.vector_search(question, n, max_date_min)
        bm = self.bm25_search(question, n, max_date_min)
        vr = {cid: r for r, (cid, _) in enumerate(vec, 1)}
        br = {cid: r for r, (cid, _) in enumerate(bm, 1)}
        return [(cid, s, {"vec_rank": vr.get(cid), "bm25_rank": br.get(cid)})
                for cid, s in rrf([[c for c, _ in vec], [c for c, _ in bm]], self.rrf_k)]

    def _row(self, cid: str) -> dict:
        return self.index.rows[self.index.pos[cid]]

    # --- mode C pieces -------------------------------------------------------
    def c_as_of(self, question: str, k: int, as_of: str) -> list[tuple[str, float, dict]]:
        """Drop chunks whose content starts after as_of (date_min > as_of)."""
        n = max(self.candidates, k)
        if self.as_of_filter == "pre":
            return self.fused(question, n, max_date_min=as_of)[:k]
        kept = [h for h in self.fused(question, n) if (self._row(h[0]).get("date_min") or "") <= as_of]
        return kept[:k]

    def c_current(self, question: str, k: int, reference_date: str) -> list[tuple[str, float, dict]]:
        """Re-rank B candidates by score * 0.5 ** (age / half_life), age = ref - date_max."""
        ref = dt.date.fromisoformat(reference_date)
        out = []
        for cid, s, comp in self.fused(question, max(self.candidates, k)):
            dmax = self._row(cid).get("date_max")
            age = max(0, (ref - dt.date.fromisoformat(dmax)).days) if dmax else None
            w = recency_weight(age, self.half_life_days)
            out.append((cid, s * w, {**comp, "rrf": s, "age_days": age, "recency_w": round(w, 4)}))
        return sorted(out, key=lambda h: (-h[1], h[0]))[:k]

    def c_change(self, question: str, k: int) -> list[tuple[str, float, dict]]:
        """Widen the pool, keep the most relevant chunk per distinct date, sort by time."""
        n = max(self.change_pool, k)
        pool = self.fused(question, max(self.candidates, n))[:n]
        picked = select_distinct_dates(pool, k, lambda cid: self._row(cid).get("date_max"))

        def when(h):
            r = self._row(h[0])
            return (r.get("date_min") or "", r.get("date_max") or "", h[0])

        return sorted(picked, key=when)

    def read_intent(self, question: str, as_of: str | None = None, qtype: str | None = None,
                    reference_date: str | None = None):
        """(Intent, reference date). `qtype` (a label) replaces the reading; an explicit
        `as_of` overrides the parsed one."""
        from tvrag.query import AS_OF, Intent, parse_question

        ref = str(reference_date or self.reference_date or dt.date.today().isoformat())[:10]
        intent = parse_question(question, ref)
        if qtype:
            intent = Intent(qtype, as_of if qtype == AS_OF else None, "label")
        elif as_of:
            intent = Intent(AS_OF, as_of, "argument")
        return intent, ref

    def mode_c(self, question: str, k: int, intent, ref: str) -> list[tuple[str, float, dict]]:
        from tvrag.query import AS_OF, CHANGE, CURRENT

        if intent.qtype == AS_OF and intent.as_of:
            return self.c_as_of(question, k, intent.as_of)
        if intent.qtype == CURRENT:
            return self.c_current(question, k, ref)
        if intent.qtype == CHANGE:
            return self.c_change(question, k)
        return self.fused(question, max(self.candidates, k))[:k]

    def fact_answer(self, question: str, intent, ref: str):
        from tvrag.query import AS_OF

        if self.facts is None:
            return None
        return self.facts.answer(question, intent.qtype, intent.as_of if intent.qtype == AS_OF else ref)

    def mode_d(self, question: str, k: int, intent, ref: str, c_scored: list[tuple[str, float, dict]]):
        """Evidence chunks of the chosen fact rows first, then mode C's list."""
        ans = self.fact_answer(question, intent, ref)
        if ans is None:
            return c_scored, {"facts": [], "fact_attrs": []}
        ev = [c for c in self.facts.evidence_ids(ans, k) if c in self.index.pos]
        rows_of = {c: [f.id for f in ans.rows if c in f.evidence] for c in ev}
        out = [(c, 1.0 + (len(ev) - i), {"fact_rows": rows_of[c]}) for i, c in enumerate(ev)]
        out += [h for h in c_scored if h[0] not in rows_of]
        return out[:k], {"facts": ans.info(), "fact_attrs": ans.attrs}

    # --- modes ---------------------------------------------------------------
    def retrieve(self, question: str, mode: str = "B", k: int = 5, as_of: str | None = None, *,
                 qtype: str | None = None, reference_date: str | None = None) -> list[Hit]:
        """Mode C reads the question (type, as-of) with tvrag.query unless `qtype` is
        given; an explicit `as_of` overrides the parsed one. `reference_date` is "now"
        (default: retrieve.c.reference_date, else today)."""
        mode = mode.upper()
        intent = None
        extra: dict = {}
        if mode == "A":
            scored = [(cid, s, {"vec_rank": r}) for r, (cid, s) in enumerate(self.vector_search(question, k), 1)]
        elif mode == "B":
            scored = self.fused(question, max(self.candidates, k))[:k]
        elif mode in ("C", "D"):
            intent, ref = self.read_intent(question, as_of, qtype, reference_date)
            scored = self.mode_c(question, k, intent, ref)
            if mode == "D":
                scored, extra = self.mode_d(question, k, intent, ref, scored)
        else:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        hits = []
        for rank, (cid, score, comp) in enumerate(scored, 1):
            row = self._row(cid)
            if intent is not None:
                comp = {**comp, "qtype": intent.qtype, "as_of": intent.as_of, "cue": intent.cue}
            elif as_of:
                comp = {**comp, "as_of": as_of}
            if rank == 1 and extra:
                comp = {**comp, **extra}
            hits.append(
                Hit(
                    chunk_id=cid,
                    score=float(score),
                    rank=rank,
                    mode=mode,
                    doc_path=row["doc_path"],
                    heading_path=list(row["heading_path"] or []),
                    headings=list(row["headings"] or []),
                    event_date=row["event_date"],
                    text=row["text"],
                    components=comp,
                    date_min=row.get("date_min"),
                    date_max=row.get("date_max"),
                )
            )
        return hits


def recency_weight(age_days: int | None, half_life_days: float) -> float:
    """0.5 ** (age / half_life); no half-life (<= 0 or inf) -> 1; unknown age -> 0.5 ** 4."""
    if half_life_days <= 0 or half_life_days == float("inf"):
        return 1.0
    if age_days is None:
        return 0.5 ** 4
    return 0.5 ** (max(0, age_days) / half_life_days)


def select_distinct_dates(pool: list[tuple[str, float, dict]], k: int, date_of) -> list[tuple[str, float, dict]]:
    """Walk `pool` in relevance order taking one chunk per distinct date; if fewer than k
    dates exist, fill with the next most relevant chunks."""
    seen, picked, rest = set(), [], []
    for h in pool:
        d = date_of(h[0])
        if d not in seen:
            seen.add(d)
            picked.append(h)
        else:
            rest.append(h)
        if len(picked) == k:
            break
    for h in rest:
        if len(picked) >= k:
            break
        picked.append(h)
    return picked[:k]


_DEFAULT: Retriever | None = None


def retrieve(question: str, mode: str = "B", k: int = 5, as_of: str | None = None, retriever: Retriever | None = None,
             **kw) -> list[Hit]:
    """Module-level entry point; loads the configured index once on first use."""
    global _DEFAULT
    if retriever is None:
        if _DEFAULT is None:
            from tvrag.config import load_config

            _DEFAULT = Retriever.from_config(load_config())
        retriever = _DEFAULT
    return retriever.retrieve(question, mode, k, as_of, **kw)


def main(argv: list[str] | None = None) -> None:
    from tvrag.config import load_config

    ap = argparse.ArgumentParser(description="Search the index with one mode.")
    ap.add_argument("question", nargs="+")
    ap.add_argument("--config", default=None)
    ap.add_argument("-m", "--mode", default="B")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--ref", default=None, help="reference date ('now') for mode C")
    ap.add_argument("--chars", type=int, default=120, help="text preview length")
    args = ap.parse_args(argv)
    r = Retriever.from_config(load_config(args.config))
    question = " ".join(args.question)
    if args.mode.upper() == "D" and r.facts is not None:
        intent, ref = r.read_intent(question, args.as_of, None, args.ref)
        ans = r.fact_answer(question, intent, ref)
        print(r.facts.format(ans) if ans else f"[D] 사실표 답 없음 ({intent.qtype}) -> C 결과만")
        print("[C 근거 청크]")
    for h in r.retrieve(question, args.mode, args.k, args.as_of, reference_date=args.ref):
        preview = " ".join(h.text.split())[: args.chars]
        print(f"[{h.mode}] #{h.rank} {h.score:.4f} {h.event_date} [{h.date_min}..{h.date_max}] {h.doc_path} :: {' > '.join(h.heading_path)} {h.components}")
        print(f"    {preview}")


if __name__ == "__main__":
    main()
