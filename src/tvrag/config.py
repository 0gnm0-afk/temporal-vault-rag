"""Environment and config loading.

Must be imported before anything that touches Hugging Face, so that model
caches are redirected to the directory named in .env (outside the repo).
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parents[2]
_PATH_KEYS = ("HF_HOME", "TVRAG_CONFIG")


def _resolve(value: str) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else (REPO_ROOT / p).resolve())


def load_env() -> None:
    """Load .env (or .env.example as fallback) without overriding real env vars."""
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        env_file = REPO_ROOT / ".env.example"
    for key, value in dotenv_values(env_file).items():
        if value is None or key in os.environ:
            continue
        os.environ[key] = _resolve(value) if key in _PATH_KEYS else value
    for key in ("HF_HOME",):
        if key in os.environ:
            Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Read the YAML config; paths under `paths:` are resolved against the repo root."""
    load_env()
    cfg_path = Path(path or os.environ.get("TVRAG_CONFIG", REPO_ROOT / "configs/example.yaml"))
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    if not cfg_path.exists():
        cfg_path = REPO_ROOT / "configs/example.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cfg["paths"] = {k: _resolve(v) for k, v in (cfg.get("paths") or {}).items()}
    cfg["_config_file"] = str(cfg_path)
    return cfg


load_env()


def data_paths(cfg: dict) -> dict[str, Path | None]:
    """Data locations. Explicit `paths.<name>` wins; otherwise derived from
    `paths.private_root` and `snapshot_id` (snapshots/<id>, chunks/<id>.jsonl,
    index/<id>, results, facts/<id>.obs.jsonl, facts/<id>.jsonl)."""
    p = cfg.get("paths") or {}
    root = Path(p.get("private_root") or REPO_ROOT)
    snap = str(cfg.get("snapshot_id") or "snapshot")

    def pick(key: str, default: Path | None) -> Path | None:
        return Path(p[key]) if p.get(key) else default

    return {
        "snapshot": pick("snapshot", root / "snapshots" / snap),
        "chunks": pick("chunks", root / "chunks" / f"{snap}.jsonl"),
        "index": pick("index", root / "index" / snap),
        "eval_set": pick("eval_set", None),
        "results": pick("results", root / "results"),
        # mode D: curated observations (input) and the built fact table (output, + .md)
        "facts_source": pick("facts_source", root / "facts" / f"{snap}.obs.jsonl"),
        "facts": pick("facts", root / "facts" / f"{snap}.jsonl"),
    }
