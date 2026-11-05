"""Create the catalog products in a running Odoo 17 (ODOO_BACKEND=xmlrpc settings).

First time only: open http://localhost:8069, create a database named as ODOO_DB with
the admin password you put in ODOO_PASSWORD, and install the Sales app.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quotedraft.config import DATA_DIR, get_settings  # noqa: E402
from quotedraft.odoo import XmlRpcOdoo  # noqa: E402
from quotedraft.store import load_catalog  # noqa: E402

if __name__ == "__main__":
    s = get_settings()
    odoo = XmlRpcOdoo(s.odoo_url, s.odoo_db, s.odoo_user, s.odoo_password)
    created = odoo.upsert_products(load_catalog(DATA_DIR / "catalog.csv"))
    print(f"Odoo: {created} products created, the rest updated")
