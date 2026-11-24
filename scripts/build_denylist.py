"""Build the leak denylist used by .githooks/pre-commit.

The denylist lives outside the repo (default: leak-denylist.txt in paths.private_root)
so that private strings never appear in tracked files. Entries:
  - generic private-location markers
  - every directory name inside the snapshot (vault folder names)
  - every note's file stem and first H1 title (titles shorter than
    MIN_LEN are skipped to avoid false positives)
  - extra literals passed with --extra

Usage: python scripts/build_denylist.py <snapshot_dir> [--extra STR ...]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from tvrag.config import load_config

MIN_LEN = 6
# Built by concatenation so this file does not match its own patterns.
GENERIC = ["private" + "/", "private" + "\\", "D" + ":/", "D" + ":\\", "d" + ":/"]
H1 = re.compile(r"^#\s+(.+?)\s*$", re.M)


def collect(snapshot: Path) -> set[str]:
    out: set[str] = set()
    for p in snapshot.rglob("*"):
        if p.is_dir():
            out.add(p.name)
        elif p.suffix == ".md":
            out.add(p.stem)
            m = H1.search(p.read_text(encoding="utf-8", errors="ignore"))
            if m:
                out.add(m.group(1))
    return {s for s in out if len(s) >= MIN_LEN or s[:1].isdigit()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", type=Path)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = args.out or Path(load_config()["paths"]["private_root"]) / "leak-denylist.txt"
    entries = sorted(set(GENERIC) | collect(args.snapshot) | set(args.extra))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(entries) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(entries)} entries -> {out}")


if __name__ == "__main__":
    main()
