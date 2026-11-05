"""Step 5: write the draft quotation to Odoo. Drafts only: nothing is confirmed or emailed.

`XmlRpcOdoo` talks to a real Odoo 17 over its external XML-RPC API.
`MockOdoo` keeps the same interface in a JSON file, for machines without Odoo running.
"""
from __future__ import annotations

import json
import threading
import xmlrpc.client
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class QuoteLine:
    sku: str
    name: str
    quantity: float
    unit_price: float


class OdooClient(Protocol):
    backend: str

    def find_or_create_partner(self, name: str | None, email: str | None) -> int: ...
    def create_draft_quote(self, partner_id: int, lines: list[QuoteLine], note: str, client_ref: str) -> int: ...
    def get_quote(self, quote_id: int) -> dict | None: ...


def order_line_commands(product_ids: dict[str, int], lines: list[QuoteLine]) -> list:
    """Odoo one2many 'create' commands: (0, 0, values)."""
    return [(0, 0, {"product_id": product_ids[ln.sku], "product_uom_qty": ln.quantity, "price_unit": ln.unit_price})
            for ln in lines]


class XmlRpcOdoo:
    backend = "odoo-xmlrpc"

    def __init__(self, url: str, db: str, user: str, password: str, transport=None):
        self.url, self.db, self.password = url.rstrip("/"), db, password
        common = xmlrpc.client.ServerProxy(f"{self.url}/xmlrpc/2/common", transport=transport, allow_none=True)
        self.uid = common.authenticate(db, user, password, {})
        if not self.uid:
            raise RuntimeError("Odoo authentication failed: check ODOO_DB, ODOO_USER and ODOO_PASSWORD")
        self.models = xmlrpc.client.ServerProxy(f"{self.url}/xmlrpc/2/object", transport=transport, allow_none=True)

    def _x(self, model: str, method: str, *args, **kw):
        return self.models.execute_kw(self.db, self.uid, self.password, model, method, list(args), kw)

    def find_or_create_partner(self, name, email):
        if email:
            ids = self._x("res.partner", "search", [["email", "=ilike", email]], limit=1)
            if ids:
                return ids[0]
        return self._x("res.partner", "create", {"name": name or email or "Unknown RFQ sender", "email": email or False})

    def product_ids(self, skus: list[str]) -> dict[str, int]:
        rows = self._x("product.product", "search_read", [["default_code", "in", skus]], fields=["id", "default_code"])
        found = {r["default_code"]: r["id"] for r in rows}
        missing = set(skus) - found.keys()
        if missing:
            raise RuntimeError(f"SKUs not found in Odoo (run scripts/setup_odoo.py): {sorted(missing)}")
        return found

    def create_draft_quote(self, partner_id, lines, note, client_ref):
        pids = self.product_ids([ln.sku for ln in lines])
        vals = {"partner_id": partner_id, "client_order_ref": client_ref, "note": note,
                "order_line": order_line_commands(pids, lines)}
        return self._x("sale.order", "create", vals)  # new sale.order records start in state 'draft'

    def get_quote(self, quote_id):
        rows = self._x("sale.order", "read", [quote_id], fields=["name", "state", "partner_id", "amount_untaxed", "client_order_ref"])
        return rows[0] if rows else None

    def upsert_products(self, products) -> int:
        n = 0
        for p in products:
            existing = self._x("product.product", "search", [["default_code", "=", p.sku]], limit=1)
            vals = {"name": p.name, "default_code": p.sku, "list_price": p.price, "sale_ok": True,
                    "description_sale": p.description}
            if existing:
                tmpl = self._x("product.product", "read", existing, fields=["product_tmpl_id"])[0]["product_tmpl_id"][0]
                self._x("product.template", "write", [tmpl], vals)
            else:
                self._x("product.template", "create", vals)
                n += 1
        return n


class MockOdoo:
    """Same interface as XmlRpcOdoo, stored in a JSON file. States are always 'draft'."""

    backend = "mock"

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"partners": [], "orders": []}), encoding="utf-8")

    def _load(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data):
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def find_or_create_partner(self, name, email):
        with self._lock:
            data = self._load()
            for p in data["partners"]:
                if email and p["email"] and p["email"].lower() == email.lower():
                    return p["id"]
            pid = len(data["partners"]) + 1
            data["partners"].append({"id": pid, "name": name or email or "Unknown RFQ sender", "email": email})
            self._save(data)
            return pid

    def create_draft_quote(self, partner_id, lines, note, client_ref):
        with self._lock:
            data = self._load()
            oid = len(data["orders"]) + 1
            data["orders"].append({
                "id": oid, "name": f"S{oid:05d}", "state": "draft", "partner_id": partner_id, "client_order_ref": client_ref,
                "note": note, "amount_untaxed": round(sum(ln.quantity * ln.unit_price for ln in lines), 2),
                "order_line": [ln.__dict__ for ln in lines]})
            self._save(data)
            return oid

    def get_quote(self, quote_id):
        for o in self._load()["orders"]:
            if o["id"] == quote_id:
                return o
        return None


def build_odoo(settings) -> OdooClient:
    from .config import DATA_DIR

    if settings.odoo_backend == "xmlrpc":
        return XmlRpcOdoo(settings.odoo_url, settings.odoo_db, settings.odoo_user, settings.odoo_password)
    return MockOdoo(DATA_DIR / "mock_odoo.json")
