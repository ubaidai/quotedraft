"""The whole flow: RFQ email (+ attachment) -> review sheet + Odoo draft quotation."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .calllog import CallLog
from .config import Settings
from .extract import extract
from .ingest import ingest
from .llm import LLM
from .models import ReviewLine, ReviewSheet
from .odoo import OdooClient, QuoteLine
from .redact import leaks, redact
from .store import CatalogStore
from .units import reconcile


@dataclass
class Pipeline:
    settings: Settings
    llm: LLM
    store: CatalogStore
    odoo: OdooClient
    calllog: CallLog
    create_drafts: bool = True

    def run(self, raw_email: str, attachment: Path | None = None, run_id: str | None = None) -> ReviewSheet:
        from .match import match_lines

        run_id = run_id or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        warnings: list[str] = []

        doc = ingest(raw_email, attachment)
        if "pdf-ocr" in doc.sources:
            warnings.append("Attachment had no text layer and was read with OCR. Check quantities against the original.")
        names = [doc.sender_name] if doc.sender_name else []
        red = redact(doc.text, known_names=names)
        leaked = leaks(red.text, [doc.sender_email or "", doc.sender_name or ""])
        if leaked:  # hard stop: never send identified personal data to the model
            raise RuntimeError(f"redaction left personal data in the prompt: {leaked}")

        rfq, _model, esc_extract = extract(self.llm, red.text, run_id=run_id, routing=self.settings.routing)
        matches, esc_match = match_lines(self.llm, self.store, rfq.line_items, run_id=run_id,
                                         routing=self.settings.routing, threshold=self.settings.confidence_threshold)

        review: list[ReviewLine] = []
        quote_lines: list[QuoteLine] = []
        for i, li in enumerate(rfq.line_items):
            m = matches[i]
            prod = self.store.get(m.sku) if m.sku else None
            qty, unit, reason = li.quantity, li.unit, m.reason
            unit_warning = None
            if prod is not None:
                qty, unit_warning = reconcile(li.quantity, li.unit, prod.unit)
                if unit_warning:
                    unit, reason = prod.unit, f"{unit_warning} {m.reason}"
            flagged = prod is None or m.confidence < self.settings.confidence_threshold or unit_warning is not None
            row = ReviewLine(index=i, description=li.description, quantity=qty, unit=unit, sku=m.sku,
                             product_name=prod.name if prod else None, unit_price=prod.price if prod else None,
                             line_total=round(prod.price * qty, 2) if prod else None,
                             confidence=round(m.confidence, 2), reason=reason, model=m.model, flagged=flagged)
            review.append(row)
            if prod is not None:
                quote_lines.append(QuoteLine(sku=prod.sku, name=prod.name, quantity=qty, unit_price=prod.price))

        unmatched = [r for r in review if r.sku is None]
        note_parts = ["Drafted by QuoteDraft. Review every line before confirming."]
        if unmatched:
            note_parts.append("Not quoted (no catalog match): " + "; ".join(f'"{r.description}" x{r.quantity:g}' for r in unmatched))
        if rfq.notes:
            note_parts.append(f"Customer notes: {rfq.notes}")
        if any(r.flagged and r.sku for r in review):
            warnings.append("Some lines are low-confidence matches and are flagged for review.")

        draft_id = None
        if self.create_drafts and quote_lines:
            partner = self.odoo.find_or_create_partner(doc.sender_name, doc.sender_email)
            draft_id = self.odoo.create_draft_quote(partner, quote_lines, "\n".join(note_parts), client_ref=doc.subject[:120])

        totals = self.calllog.run_totals(run_id)
        return ReviewSheet(
            run_id=run_id, customer_email=doc.sender_email, customer_name=doc.sender_name, lines=review,
            total=round(sum(r.line_total or 0 for r in review), 2), notes=rfq.notes, draft_id=draft_id,
            odoo_backend=self.odoo.backend, cost_usd=round(totals["cost_usd"], 6),
            latency_ms=int((time.perf_counter() - t0) * 1000), escalations=esc_extract + esc_match, warnings=warnings)


def build_pipeline(settings: Settings | None = None, create_drafts: bool = True) -> Pipeline:
    from .calllog import CallLog
    from .config import get_settings
    from .llm import AnthropicLLM
    from .odoo import build_odoo
    from .store import build_store

    settings = settings or get_settings()
    calllog = CallLog(settings.calllog_path)
    return Pipeline(settings=settings, llm=AnthropicLLM(calllog, settings.daily_budget_usd), store=build_store(settings),
                    odoo=build_odoo(settings), calllog=calllog, create_drafts=create_drafts)
