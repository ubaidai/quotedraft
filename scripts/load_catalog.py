"""Load data/catalog.csv into Postgres + pgvector (QD_STORE=postgres, DATABASE_URL set)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quotedraft.config import DATA_DIR, get_settings  # noqa: E402
from quotedraft.embed import get_embedder  # noqa: E402
from quotedraft.store import PgStore, load_catalog  # noqa: E402

if __name__ == "__main__":
    s = get_settings()
    if not s.database_url:
        sys.exit("Set DATABASE_URL, e.g. postgresql://quotedraft:quotedraft@localhost:5433/quotedraft")
    products = load_catalog(DATA_DIR / "catalog.csv")
    store = PgStore(s.database_url, get_embedder(s.embed_model))
    store.load(products)
    n = store.conn.execute("SELECT count(*) FROM products").fetchone()[0]
    print(f"products table now has {n} rows (embedding dim {store.embedder.dim})")
