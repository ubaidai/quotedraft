"""Deterministic unit check between what the customer asked for and how the product is sold.

Added after the first full eval: OCR read "10x6m" on a scan, the model quoted 60 lengths of
steel angle instead of 10, and reported 0.93 confidence. Model confidence does not catch unit
errors, so code does.
"""
from __future__ import annotations

import math
import re

FAMILIES = {
    "pcs": {"pcs", "pc", "piece", "pieces", "off", "no", "nos", "units", "unit", "each", "ea", ""},
    "length": {"length", "lengths", "bar", "bars", "lengths (6m)", "x 6m", "6m lengths"},
    "pack": {"pack", "packs", "pkt", "pkts", "bag", "bags"},
    "metre": {"m", "metre", "metres", "meter", "meters", "mtr", "mtrs", "lm"},
}
LENGTH_M = 6.0  # every length-sold product in this catalog is a 6 m length


def family(unit: str) -> str | None:
    u = re.sub(r"\s+", " ", unit.strip().lower())
    for fam, words in FAMILIES.items():
        if u in words:
            return fam
    if "length" in u:
        return "length"
    if u.endswith("m") and u[:-1].replace(".", "").isdigit():
        return "metre"
    return None


def reconcile(quantity: float, customer_unit: str, product_unit: str) -> tuple[float, str | None]:
    """Returns (quantity in the product's unit, warning or None). A warning always flags the line."""
    cf, pf = family(customer_unit), family(product_unit)
    if cf == pf:
        return quantity, None
    if pf == "length" and cf == "metre":
        n = math.ceil(quantity / LENGTH_M - 1e-9)
        return float(n), f"customer asked for {quantity:g} m; sold in {LENGTH_M:g} m lengths, so quoted {n} lengths. Check."
    if cf is None:
        return quantity, f"unit '{customer_unit}' not recognised; quoted {quantity:g} {product_unit}. Check."
    return quantity, f"customer unit '{customer_unit}' differs from product unit '{product_unit}'. Check the quantity."
