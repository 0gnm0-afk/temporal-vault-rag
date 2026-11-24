"""Download the embedding model into the cache dir set in .env (HF_HOME)."""
import os

from tvrag import config  # noqa: F401  (sets HF_HOME before HF imports)
from huggingface_hub import snapshot_download

cfg = config.load_config()
model = cfg["embedding"]["model"]
# Dense-only use via sentence-transformers: skip ONNX export, images and the
# colbert/sparse heads.
path = snapshot_download(model, ignore_patterns=["onnx/*", "imgs/*", "*.jpg", "*.pt", "*.md", ".gitattributes"])
print("HF_HOME", os.environ["HF_HOME"])
print("downloaded", model, "->", path)
