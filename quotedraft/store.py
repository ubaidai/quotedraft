"""Catalog search: vector, keyword and hybrid (reciprocal rank fusion).

Two backends with the same interface:
  * PgStore    - PostgreSQL + pgvector (production; docker compose)
  * LocalStore - in-process numpy + BM25 (dev machines without Postgres, and tests)

Both normalise text the same way and fuse with the same RRF, so hybrid behaviour matches.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from .embed import Embedder
from .models import Candidate, Product
from .text import normalize, tokenize

Mode = Literal["vector", "keyword", "hybrid"]
RRF_K = 60
# Keyword rank counts 5x the vector rank in fusion. Chosen on RFQs 01-12 only and
# reported on the held-out RFQs 13-25 (eval/run.py --retrieval-only).
KEYWORD_WEIGHT = float(__import__("os").getenv("QD_KEYWORD_WEIGHT", "5"))


def load_catalog(path: Path) -> list[Product]:
    with open(path, encoding="utf-8") as fh:
        return [Product(sku=r["sku"], name=r["name"], description=r["description"], family=r["family"],
                        unit=r["unit"], price=float(r["price"]), attributes=json.loads(r["attributes"]))
                for r in csv.DictReader(fh)]


def rrf(rankings: list[list[str]], weights: list[float] | None = None, k: int = RRF_K) -> list[tuple[str, float]]:
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, sku in enumerate(ranking):
            scores[sku] = scores.get(sku, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def hybrid(vector: list[str], keyword: list[str]) -> list[tuple[str, float]]:
    return rrf([vector, keyword], [1.0, KEYWORD_WEIGHT])


class CatalogStore(Protocol):
    def search(self, query: str, k: int = 5, mode: Mode = "hybrid") -> list[Candidate]: ...
    def get(self, sku: str) -> Product | None: ...


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avgdl = sum(map(len, docs)) / len(docs)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, q: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out[i] = s
        return out


class LocalStore:
    def __init__(self, products: list[Product], embedder: Embedder):
        self.products = products
        self.by_sku = {p.sku: p for p in products}
        self.embedder = embedder
        texts = [p.search_text() for p in products]
        self.matrix = embedder.encode(texts)
        self.bm25 = BM25([tokenize(t) for t in texts])

    def _vector_rank(self, query: str, n: int) -> list[tuple[str, float]]:
        q = self.embedder.encode([query])[0]
        sims = self.matrix @ q
        idx = np.argsort(-sims)[:n]
        return [(self.products[i].sku, float(sims[i])) for i in idx]

    def _keyword_rank(self, query: str, n: int) -> list[tuple[str, float]]:
        sc = self.bm25.scores(tokenize(query))
        idx = [i for i in np.argsort(-sc)[:n] if sc[i] > 0]
        return [(self.products[i].sku, float(sc[i])) for i in idx]

    def search(self, query: str, k: int = 5, mode: Mode = "hybrid") -> list[Candidate]:
        pool = max(k, 30)
        if mode == "vector":
            ranked = self._vector_rank(query, k)
        elif mode == "keyword":
            ranked = self._keyword_rank(query, k)
        else:
            ranked = hybrid([s for s, _ in self._vector_rank(query, pool)], [s for s, _ in self._keyword_rank(query, pool)])[:k]
        return [self._cand(sku, score) for sku, score in ranked]

    def _cand(self, sku: str, score: float) -> Candidate:
        p = self.by_sku[sku]
        return Candidate(sku=p.sku, name=p.name, unit=p.unit, price=p.price, score=round(score, 4))

    def get(self, sku: str) -> Product | None:
        return self.by_sku.get(sku)


class PgStore:
    """PostgreSQL + pgvector. Keyword side uses Postgres full-text search over normalised text."""

    def __init__(self, dsn: str, embedder: Embedder):
        import psycopg
        from pgvector.psycopg import register_vector

        self.embedder = embedder
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)

    def init_schema(self):
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS products (
                sku text PRIMARY KEY,
                name text NOT NULL,
                description text NOT NULL,
                family text NOT NULL,
                unit text NOT NULL,
                price numeric(12,2) NOT NULL,
                attributes jsonb NOT NULL DEFAULT '{{}}',
                search_text text NOT NULL,
                tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', search_text)) STORED,
                embedding vector({self.embedder.dim}) NOT NULL
            )""")
        self.conn.execute("CREATE INDEX IF NOT EXISTS products_tsv_idx ON products USING gin (tsv)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS products_emb_idx ON products USING hnsw (embedding vector_cosine_ops)")

    def load(self, products: list[Product]):
        self.init_schema()
        vecs = self.embedder.encode([p.search_text() for p in products])
        with self.conn.cursor() as cur:
            for p, v in zip(products, vecs):
                cur.execute(
                    """INSERT INTO products (sku, name, description, family, unit, price, attributes, search_text, embedding)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (sku) DO UPDATE SET name=EXCLUDED.name, description=EXCLUDED.description,
                         price=EXCLUDED.price, attributes=EXCLUDED.attributes, search_text=EXCLUDED.search_text,
                         embedding=EXCLUDED.embedding""",
                    (p.sku, p.name, p.description, p.family, p.unit, p.price, json.dumps(p.attributes),
                     " ".join(tokenize(p.search_text())), v))

    def _vector_rank(self, query: str, n: int) -> list[tuple[str, float]]:
        q = self.embedder.encode([query])[0]
        rows = self.conn.execute(
            "SELECT sku, 1 - (embedding <=> %s) FROM products ORDER BY embedding <=> %s LIMIT %s", (q, q, n)).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def _keyword_rank(self, query: str, n: int) -> list[tuple[str, float]]:
        toks = tokenize(query)
        if not toks:
            return []
        tsq = " | ".join(t.replace("'", "") for t in toks)
        rows = self.conn.execute(
            """SELECT sku, ts_rank_cd(tsv, q) AS r FROM products, to_tsquery('simple', %s) q
               WHERE tsv @@ q ORDER BY r DESC LIMIT %s""", (tsq, n)).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def search(self, query: str, k: int = 5, mode: Mode = "hybrid") -> list[Candidate]:
        pool = max(k, 30)
        if mode == "vector":
            ranked = self._vector_rank(query, k)
        elif mode == "keyword":
            ranked = self._keyword_rank(query, k)
        else:
            ranked = hybrid([s for s, _ in self._vector_rank(query, pool)], [s for s, _ in self._keyword_rank(query, pool)])[:k]
        out = []
        for sku, score in ranked:
            p = self.get(sku)
            out.append(Candidate(sku=p.sku, name=p.name, unit=p.unit, price=p.price, score=round(score, 4)))
        return out

    def get(self, sku: str) -> Product | None:
        r = self.conn.execute(
            "SELECT sku, name, description, family, unit, price, attributes FROM products WHERE sku=%s", (sku,)).fetchone()
        if not r:
            return None
        return Product(sku=r[0], name=r[1], description=r[2], family=r[3], unit=r[4], price=float(r[5]), attributes=r[6])


def build_store(settings, products: list[Product] | None = None) -> CatalogStore:
    from .config import DATA_DIR
    from .embed import get_embedder

    embedder = get_embedder(settings.embed_model)
    if settings.store == "postgres":
        return PgStore(settings.database_url, embedder)
    return LocalStore(products or load_catalog(DATA_DIR / "catalog.csv"), embedder)
