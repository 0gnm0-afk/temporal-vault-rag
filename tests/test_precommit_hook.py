"""The leak guard blocks denylisted strings and paths, and fails closed."""
import shutil
import subprocess
from pathlib import Path

import pytest

from tvrag.config import REPO_ROOT

HOOK = REPO_ROOT / ".githooks" / "pre-commit"
DUMMY = "ZZ_DENY_한글_TOKEN"  # stand-in for a private string

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not found")


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8"
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "r"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "user.email", "t@example.invalid")
    git(r, "config", "user.name", "t")
    hooks = r / ".githooks"
    hooks.mkdir()
    shutil.copy(HOOK, hooks / "pre-commit")
    git(r, "config", "core.hooksPath", ".githooks")
    deny = tmp_path / "deny.txt"
    deny.write_text(f"{DUMMY}\nsecret-dir\n", encoding="utf-8")
    git(r, "config", "tvrag.denylist", str(deny))
    return r


def commit(repo: Path, name: str, text: str) -> subprocess.CompletedProcess:
    p = repo / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    git(repo, "add", name)
    return git(repo, "commit", "-q", "-m", "x")


def test_clean_commit_passes(repo):
    assert commit(repo, "ok.txt", "nothing private here\n").returncode == 0


def test_content_blocked(repo):
    r = commit(repo, "a.txt", f"see {DUMMY} note\n")
    assert r.returncode != 0 and "blocked" in r.stderr


def test_path_blocked(repo):
    r = commit(repo, "secret-dir/a.txt", "x\n")
    assert r.returncode != 0 and "path" in r.stderr


def test_missing_denylist_fails_closed(repo, tmp_path):
    git(repo, "config", "tvrag.denylist", str(tmp_path / "missing.txt"))
    assert commit(repo, "ok.txt", "x\n").returncode != 0
