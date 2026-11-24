"""Search modes A-D: A vector, B +BM25 (RRF), C +time (as-of filter, recency, multi-point history), D +fact table.

Common interface: retrieve(question, mode, k, as_of=None) -> list[Hit]
Every hit carries the mode that produced it and its evidence chunk (id, doc, headings,
event date, text). Modes C and D are placeholders until S5/S6; `as_of` is accepted and
recorded but not used by A and B.

Usage: python -m tvrag.retrieve --config configs/local.yaml -m B -k 5 "question"
"""
from __future__ import annotations

import argparse
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
    components: dict = field(default_factory=dict)  # per-retriever ranks/scores (mode B)

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

    @classmethod
    def from_config(cls, cfg: dict, index_dir: str | Path | None = None) -> "Retriever":
        from tvrag.config import data_paths

        e = cfg["embedding"]
        emb = Embedder(e["model"], e.get("device", "cpu"), e.get("batch_size", 8), e.get("normalize", True))
        return cls(load_index(index_dir or data_paths(cfg)["index"]), emb, cfg=cfg.get("retrieve") or {})

    # --- single retrievers -------------------------------------------------
    def vector_search(self, question: str, n: int) -> list[tuple[str, float]]:
        """Exact cosine search in LanceDB (no ANN index: brute force over all rows)."""
        q = self.embedder.encode([question])[0]
        res = (
            self.index.table.search(q.tolist())
            .distance_type("cosine")
            .select(["chunk_id", "_distance"])
            .limit(n)
            .to_list()
        )
        hits = [(r["chunk_id"], 1.0 - float(r["_distance"])) for r in res]
        # stable order for equal similarity
        return sorted(hits, key=lambda h: (-round(h[1], 6), h[0]))

    def bm25_search(self, question: str, n: int) -> list[tuple[str, float]]:
        toks = self.tokenizer(question)
        if not toks:
            return []
        scores = np.asarray(self.index.bm25.get_scores(toks))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        rows = self.index.rows
        return [(rows[i]["chunk_id"], float(scores[i])) for i in order[:n] if scores[i] > 0]

    # --- modes ---------------------------------------------------------------
    def retrieve(self, question: str, mode: str = "B", k: int = 5, as_of: str | None = None) -> list[Hit]:
        mode = mode.upper()
        if mode == "A":
            scored = [(cid, s, {"vec_rank": r}) for r, (cid, s) in enumerate(self.vector_search(question, k), 1)]
        elif mode == "B":
            n = max(self.candidates, k)
            vec = self.vector_search(question, n)
            bm = self.bm25_search(question, n)
            vr = {cid: r for r, (cid, _) in enumerate(vec, 1)}
            br = {cid: r for r, (cid, _) in enumerate(bm, 1)}
            fused = rrf([[c for c, _ in vec], [c for c, _ in bm]], self.rrf_k)
            scored = [(cid, s, {"vec_rank": vr.get(cid), "bm25_rank": br.get(cid)}) for cid, s in fused[:k]]
        elif mode in ("C", "D"):
            raise NotImplementedError(f"mode {mode} is implemented in a later stage (C: S5, D: S6)")
        else:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        hits = []
        for rank, (cid, score, comp) in enumerate(scored, 1):
            row = self.index.rows[self.index.pos[cid]]
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
                    components={**comp, "as_of": as_of} if as_of else comp,
                )
            )
        return hits


_DEFAULT: Retriever | None = None


def retrieve(question: str, mode: str = "B", k: int = 5, as_of: str | None = None, retriever: Retriever | None = None) -> list[Hit]:
    """Module-level entry point; loads the configured index once on first use."""
    global _DEFAULT
    if retriever is None:
        if _DEFAULT is None:
            from tvrag.config import load_config

            _DEFAULT = Retriever.from_config(load_config())
        retriever = _DEFAULT
    return retriever.retrieve(question, mode, k, as_of)


def main(argv: list[str] | None = None) -> None:
    from tvrag.config import load_config

    ap = argparse.ArgumentParser(description="Search the index with one mode.")
    ap.add_argument("question", nargs="+")
    ap.add_argument("--config", default=None)
    ap.add_argument("-m", "--mode", default="B")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--chars", type=int, default=120, help="text preview length")
    args = ap.parse_args(argv)
    r = Retriever.from_config(load_config(args.config))
    for h in r.retrieve(" ".join(args.question), args.mode, args.k, args.as_of):
        preview = " ".join(h.text.split())[: args.chars]
        print(f"[{h.mode}] #{h.rank} {h.score:.4f} {h.event_date} {h.doc_path} :: {' > '.join(h.heading_path)} {h.components}")
        print(f"    {preview}")


if __name__ == "__main__":
    main()
