"""Build and load the vector store (bge-m3 embeddings in LanceDB) and the BM25 index (kiwipiepy morphemes).

Usage: python -m tvrag.index --config configs/local.yaml [--chunks PATH] [--out DIR]

Index folder layout (default: <private_root>/index/<snapshot_id>/):
  lancedb/            LanceDB database, table "chunks" (metadata + vector)
  bm25_tokens.jsonl   one {"chunk_id", "tokens"} per chunk, BM25 is rebuilt from it on load
  manifest.json       inputs and checksums; no timestamps, so a rebuild from the same
                      inputs yields the same manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TABLE = "chunks"
EMBED_TEMPLATE_VERSION = 1
METADATA_VERSION = 2  # 2: date_min/date_max columns

# BM25 keeps content morphemes only: nouns (NNG general, NNP proper, NR numerals),
# verb/adjective stems (VV, VA), roots (XR), and foreign/hanja/number tokens
# (SL, SH, SN) because units and values ("kg", "kcal", "75.8") carry the facts the
# questions ask about. Particles (J*), endings (E*), suffixes (XS*), bound nouns
# (NNB: 것, 수, 회 ...) and punctuation are dropped: they occur everywhere and only
# add noise to term overlap.
BM25_TAGS = frozenset({"NNG", "NNP", "NR", "VV", "VA", "XR", "SL", "SH", "SN"})


def load_chunks(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return sorted(rows, key=lambda r: r["chunk_id"])


def doc_title(doc_path: str) -> str:
    return Path(doc_path).stem


def passage_text(chunk: dict) -> str:
    """Text that is embedded and BM25-tokenized: document title and heading path
    give a split-off piece the context its first line may lack."""
    head = " > ".join([doc_title(chunk["doc_path"]), *chunk.get("heading_path", [])])
    return f"{head}\n\n{chunk['text']}"


class Tokenizer:
    def __init__(self) -> None:
        from kiwipiepy import Kiwi

        self.kiwi = Kiwi()

    def __call__(self, text: str) -> list[str]:
        out = []
        for t in self.kiwi.tokenize(text):
            if t.tag in BM25_TAGS:
                out.append(t.form.lower() if t.tag == "SL" else t.form)
        return out


class Embedder:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 8, normalize: bool = True):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.manual_seed(0)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.normalize = normalize

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(emb, dtype=np.float32)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _model_revision(model_name: str) -> str | None:
    """Commit hash the HF cache resolves `main` to (what from_pretrained loads)."""
    try:
        from huggingface_hub import constants

        ref = Path(constants.HF_HUB_CACHE) / f"models--{model_name.replace('/', '--')}" / "refs" / "main"
        return ref.read_text(encoding="utf-8").strip() if ref.exists() else None
    except Exception:
        return None


def build_index(
    chunks_path: str | Path,
    out_dir: str | Path,
    emb_cfg: dict,
    embedder: Embedder | None = None,
    show_progress: bool = True,
) -> dict:
    """Embed + tokenize all chunks and write the index folder. Returns the manifest."""
    import kiwipiepy
    import lancedb
    import pyarrow as pa

    chunks_path, out_dir = Path(chunks_path), Path(out_dir)
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"no chunks in {chunks_path}")
    texts = [passage_text(c) for c in chunks]

    embedder = embedder or Embedder(
        emb_cfg["model"], emb_cfg.get("device", "cpu"), emb_cfg.get("batch_size", 8), emb_cfg.get("normalize", True)
    )
    vectors = embedder.encode(texts, show_progress=show_progress)
    tok = Tokenizer()
    tokens = [tok(t) for t in texts]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    dim = vectors.shape[1]
    schema = pa.schema(
        [
            ("chunk_id", pa.string()),
            ("doc_path", pa.string()),
            ("heading_path", pa.list_(pa.string())),
            ("headings", pa.list_(pa.string())),
            ("text", pa.string()),
            ("event_date", pa.string()),
            ("date_source", pa.string()),
            ("doc_date", pa.string()),
            ("date_min", pa.string()),
            ("date_max", pa.string()),
            ("doc_type", pa.string()),
            ("n_tokens", pa.int32()),
            ("vector", pa.list_(pa.float32(), dim)),
        ]
    )
    rows = [
        {
            "chunk_id": c["chunk_id"],
            "doc_path": c["doc_path"],
            "heading_path": c.get("heading_path", []),
            "headings": c.get("headings", []),
            "text": c["text"],
            "event_date": c.get("event_date"),
            "date_source": c.get("date_source"),
            "doc_date": c.get("doc_date"),
            # chunks written before S5 have no range: fall back to the event date
            "date_min": c.get("date_min") or c.get("event_date"),
            "date_max": c.get("date_max") or c.get("event_date"),
            "doc_type": c.get("doc_type"),
            "n_tokens": c.get("n_tokens", 0),
            "vector": v.tolist(),
        }
        for c, v in zip(chunks, vectors)
    ]
    db = lancedb.connect(str(out_dir / "lancedb"))
    db.create_table(TABLE, data=pa.Table.from_pylist(rows, schema=schema), mode="overwrite")

    with (out_dir / "bm25_tokens.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for c, t in zip(chunks, tokens):
            f.write(json.dumps({"chunk_id": c["chunk_id"], "tokens": t}, ensure_ascii=False) + "\n")

    manifest = {
        "chunks_file": chunks_path.name,
        "chunks_sha256": _sha256_file(chunks_path),
        "n_chunks": len(chunks),
        "embedding_model": emb_cfg["model"],
        "embedding_revision": _model_revision(emb_cfg["model"]),
        "embedding_dim": int(dim),
        "normalize": bool(emb_cfg.get("normalize", True)),
        "embed_template_version": EMBED_TEMPLATE_VERSION,
        "metadata_version": METADATA_VERSION,
        "vectors_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
        "bm25_tags": sorted(BM25_TAGS),
        "kiwipiepy_version": kiwipiepy.__version__,
        "bm25_tokens_sha256": _sha256_file(out_dir / "bm25_tokens.jsonl"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def read_manifest(index_dir: str | Path) -> dict | None:
    p = Path(index_dir) / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def is_current(index_dir: str | Path, chunks_path: str | Path, model_name: str) -> bool:
    m = read_manifest(index_dir)
    return bool(
        m
        and Path(chunks_path).exists()
        and m.get("chunks_sha256") == _sha256_file(Path(chunks_path))
        and m.get("embedding_model") == model_name
        and m.get("embed_template_version") == EMBED_TEMPLATE_VERSION
        and m.get("metadata_version") == METADATA_VERSION
        and m.get("bm25_tags") == sorted(BM25_TAGS)
    )


@dataclass
class Index:
    """In-memory view of an index folder."""

    dir: Path
    table: object  # lancedb table
    rows: list[dict]  # metadata in chunk_id order (no vectors)
    pos: dict[str, int]  # chunk_id -> row position
    bm25: object
    manifest: dict


def load_index(index_dir: str | Path) -> Index:
    import lancedb
    from rank_bm25 import BM25Okapi

    index_dir = Path(index_dir)
    manifest = read_manifest(index_dir)
    if manifest is None:
        raise FileNotFoundError(f"no index at {index_dir} (build it with: python -m tvrag.index)")
    table = lancedb.connect(str(index_dir / "lancedb")).open_table(TABLE)
    data = table.to_arrow().drop_columns(["vector"]).to_pylist()
    rows = sorted(data, key=lambda r: r["chunk_id"])
    pos = {r["chunk_id"]: i for i, r in enumerate(rows)}
    toks: list[list[str] | None] = [None] * len(rows)
    with (index_dir / "bm25_tokens.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            toks[pos[r["chunk_id"]]] = r["tokens"]
    if any(t is None for t in toks):
        raise ValueError("bm25_tokens.jsonl does not cover every chunk")
    # rank_bm25 divides by the average length; guard against an all-empty corpus
    bm25 = BM25Okapi([t or ["\u0000"] for t in toks])
    return Index(index_dir, table, rows, pos, bm25, manifest)


def main(argv: list[str] | None = None) -> None:
    from tvrag.config import data_paths, load_config

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--chunks", default=None, help="default: paths.chunks")
    ap.add_argument("--out", default=None, help="default: paths.index")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    paths = data_paths(cfg)
    chunks = Path(args.chunks) if args.chunks else paths["chunks"]
    out = Path(args.out) if args.out else paths["index"]
    m = build_index(chunks, out, cfg["embedding"])
    print(f"chunks={m['n_chunks']} dim={m['embedding_dim']} vectors_sha256={m['vectors_sha256'][:12]} -> {out}")


if __name__ == "__main__":
    main()
