"""One command: chunk -> index -> fact table -> evaluate.

  python -m tvrag.run --config configs/demo.yaml            # demo vault, modes A B
  python -m tvrag.run --config configs/local.yaml --eval <eval.jsonl>

Chunks are rebuilt when the chunk file is missing or --rechunk is given (the demo config
sets run.rechunk so edits to the demo vault are always picked up). The index is rebuilt
when its manifest does not match the chunk file, model, or tokenizer settings, or with
--reindex. The fact table (mode D) is rebuilt from its observation file on every run
(cheap); without an observation file mode D equals C.
"""
from __future__ import annotations

import argparse

from tvrag.config import data_paths, load_config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--modes", nargs="+", default=["A", "B"])
    ap.add_argument("--eval", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--date", default=None)
    ap.add_argument("--rechunk", action="store_true")
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument("--no-eval", action="store_true", help="stop after building the index")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    paths = data_paths(cfg)
    rcfg = cfg.get("run") or {}

    if args.rechunk or rcfg.get("rechunk") or not paths["chunks"].exists():
        from tvrag.chunk import build_chunks

        n_docs, n_chunks = build_chunks(cfg, paths["chunks"])
        print(f"[chunk] docs={n_docs} chunks={n_chunks} -> {paths['chunks']}")
    else:
        print(f"[chunk] reuse {paths['chunks']}")

    from tvrag.index import build_index, is_current

    if args.reindex or not is_current(paths["index"], paths["chunks"], cfg["embedding"]["model"]):
        m = build_index(paths["chunks"], paths["index"], cfg["embedding"], show_progress=False)
        print(f"[index] built chunks={m['n_chunks']} vectors_sha256={m['vectors_sha256'][:12]} -> {paths['index']}")
    else:
        print(f"[index] reuse {paths['index']}")

    from tvrag.facts import build_from_config

    build_from_config(cfg)

    if args.no_eval:
        return
    from tvrag.eval import evaluate

    r = evaluate(cfg, [m.upper() for m in args.modes], args.eval, args.out, args.date)
    print(r["table"])
    print(f"[eval] -> {r['out_dir']}")


if __name__ == "__main__":
    main()
