"""Pre-trained GloVe word embeddings (glove-wiki-gigaword-100, 400k uncased words)."""
from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from . import EMB_DIR

GLOVE_NAME = "glove-wiki-gigaword-100"
GLOVE_URL = f"https://github.com/RaRe-Technologies/gensim-data/releases/download/{GLOVE_NAME}/{GLOVE_NAME}.gz"
GLOVE_DIM = 100


def download_glove(emb_dir: Path = EMB_DIR) -> Path:
    emb_dir.mkdir(parents=True, exist_ok=True)
    path = emb_dir / f"{GLOVE_NAME}.gz"
    if not path.exists():
        print(f"Downloading GloVe ({GLOVE_URL}) ...")
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(GLOVE_URL) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(path)
    return path


def load_glove(wanted: set[str] | None = None, top_n: int = 0, emb_dir: Path = EMB_DIR) -> dict[str, np.ndarray]:
    """Return {word: vector}; restricted to `wanted` words plus the `top_n` most frequent
    GloVe words when `wanted` is given (much less memory than all 400k)."""
    path = download_glove(emb_dir)
    vectors = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        first = f.readline().split()
        if len(first) != 2:  # no word2vec header -> first line is a vector
            f.seek(0)
        for rank, line in enumerate(f):
            word, rest = line.split(" ", 1)
            if wanted is None or rank < top_n or word in wanted:
                vectors[word] = np.asarray(rest.split(), dtype=np.float32)
    return vectors


def build_matrix(words: list[str], glove: dict[str, np.ndarray], dim: int = GLOVE_DIM, seed: int = 0):
    """Embedding matrix aligned with `words`; words missing from GloVe get small random vectors."""
    rng = np.random.default_rng(seed)
    scale = np.sqrt(3.0 / dim)
    mat = rng.uniform(-scale, scale, size=(len(words), dim)).astype(np.float32)
    mat[0] = 0.0  # <pad>
    hits = 0
    for i, w in enumerate(words):
        v = glove.get(w)
        if v is not None:
            mat[i] = v
            hits += 1
    return mat, hits
