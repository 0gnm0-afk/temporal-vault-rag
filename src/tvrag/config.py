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
