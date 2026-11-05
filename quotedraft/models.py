"""Pydantic schemas shared across the pipeline."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Product(BaseModel):
    sku: str
    name: str
    description: str
    family: str
    unit: str
    price: float
    attributes: dict[str, str] = Field(default_factory=dict)

    def search_text(self) -> str:
        attrs = " ".join(f"{k} {v}" for k, v in self.attributes.items())
        return f"{self.name}. {self.description}. {attrs}"


# ---- LLM extraction output -------------------------------------------------

class ExtractedLine(BaseModel):
    description: str = Field(description="The item as the customer wrote it, copied closely, with all sizes and specs.")
    quantity: float = Field(description="Total quantity requested, in the unit given. Multiply packs out, e.g. '2 boxes of 100' -> 200.")
    unit: str = Field(description="Unit as written: pcs, m, lengths, boxes, etc. Use 'pcs' if none is given.")


class ExtractedRFQ(BaseModel):
    line_items: list[ExtractedLine] = Field(description="Every product line the customer asks to be quoted, in order.")
    notes: str = Field(default="", description="Delivery dates or other requirements that are not product lines. Empty if none.")


# ---- matching ----------------------------------------------------------------

class Candidate(BaseModel):
    sku: str
    name: str
    unit: str
    price: float
    score: float


class LineMatch(BaseModel):
    line_index: int
    sku: Optional[str]  # None = NO_MATCH
    confidence: float
    reason: str
    model: str = ""


class ReviewLine(BaseModel):
    index: int
    description: str
    quantity: float
    unit: str
    sku: Optional[str]
    product_name: Optional[str] = None
    unit_price: Optional[float] = None
    line_total: Optional[float] = None
    confidence: float
    reason: str
    model: str
    flagged: bool


class ReviewSheet(BaseModel):
    run_id: str
    customer_email: Optional[str]
    customer_name: Optional[str]
    lines: list[ReviewLine]
    total: float
    notes: str = ""
    draft_id: Optional[int] = None
    odoo_backend: str
    cost_usd: float
    latency_ms: int
    escalations: int
    warnings: list[str] = Field(default_factory=list)
