"""Local sentence embeddings. Catalog text never leaves the machine."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Protocol

import numpy as np

from .text import normalize, tokenize


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode([normalize(t) for t in texts], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


class HashingEmbedder:
    """Deterministic bag-of-tokens embedder for tests. Not semantic; never used in evals."""

    dim = 256

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


@lru_cache(maxsize=2)
def get_embedder(model_name: str) -> Embedder:
    if model_name == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(model_name)
