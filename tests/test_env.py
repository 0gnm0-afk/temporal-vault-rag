"""Environment checks: everything must work under a non-ASCII (Korean) path."""
import os
from pathlib import Path

import numpy as np
import pytest

from tvrag.config import REPO_ROOT, load_config


def test_repo_path_reported():
    # Not an assertion on the path itself; documents where the checks ran.
    print("repo root:", REPO_ROOT, "ascii:", str(REPO_ROOT).isascii())


def test_config_loads_and_redirects_cache():
    cfg = load_config()
    assert cfg["embedding"]["model"] == "BAAI/bge-m3"
    assert "HF_HOME" in os.environ
    assert not Path(os.environ["HF_HOME"]).is_relative_to(Path.home() / ".cache")


def test_kiwi_tokenizes_korean():
    from kiwipiepy import Kiwi

    kiwi = Kiwi()
    forms = [t.form for t in kiwi.tokenize("오늘 체중을 재고 단백질을 먹었다.")]
    assert "체중" in forms and "단백질" in forms


def test_lancedb_write_read_under_repo(tmp_path_factory):
    import lancedb

    # Write inside the repo tree (non-ASCII path), not only in the temp dir.
    target = REPO_ROOT / ".pytest_cache" / "한글_lancedb"
    db = lancedb.connect(str(target))
    tbl = db.create_table(
        "t", data=[{"id": "a", "text": "한글", "vector": [0.1, 0.2, 0.3]}], mode="overwrite"
    )
    hit = tbl.search([0.1, 0.2, 0.3]).limit(1).to_list()[0]
    assert hit["id"] == "a" and hit["text"] == "한글"


@pytest.mark.slow
def test_bge_m3_embeds_korean():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(load_config()["embedding"]["model"], device="cpu")
    emb = model.encode(
        ["아침 공복 체중은 72kg이었다.", "오늘 아침 몸무게를 쟀다."], normalize_embeddings=True
    )
    assert emb.shape == (2, 1024)
    assert np.isfinite(emb).all()
    assert float(emb[0] @ emb[1]) > 0.5
