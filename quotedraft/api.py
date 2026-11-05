"""HTTP API and a plain upload page.

  uvicorn quotedraft.api:app --port 8000
"""
from __future__ import annotations

import html
import json
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from .calllog import BudgetExceeded, CallLog
from .config import DATA_DIR, get_settings
from .models import ReviewSheet

app = FastAPI(title="QuoteDraft", description="RFQ email + spec -> draft Odoo quotation for human approval")
RUNS = DATA_DIR / "runs"


@lru_cache(maxsize=1)
def pipeline():
    from .pipeline import build_pipeline

    return build_pipeline()


def _save(sheet: ReviewSheet):
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{sheet.run_id}.json").write_text(sheet.model_dump_json(indent=2), encoding="utf-8")


async def _run(email_text: str, file: UploadFile | None) -> ReviewSheet:
    att = None
    if file is not None and file.filename:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "attachment must be a PDF")
        data = await file.read()
        if len(data) > 15 * 1024 * 1024:
            raise HTTPException(413, "attachment larger than 15 MB")
        tmp = Path(tempfile.mkdtemp()) / "attachment.pdf"
        tmp.write_bytes(data)
        att = tmp
    try:
        sheet = pipeline().run(email_text, att)
    except BudgetExceeded as exc:
        raise HTTPException(429, str(exc)) from exc
    _save(sheet)
    return sheet


@app.post("/rfq", response_model=ReviewSheet)
async def post_rfq(email_text: str = Form(...), file: UploadFile | None = File(None)):
    return await _run(email_text, file)


@app.get("/rfq/{run_id}", response_model=ReviewSheet)
def get_rfq(run_id: str):
    path = RUNS / f"{run_id}.json"
    if not run_id.isalnum() or not path.exists():
        raise HTTPException(404, "unknown run id")
    return ReviewSheet.model_validate_json(path.read_text(encoding="utf-8"))


@app.get("/stats")
def stats():
    s = get_settings()
    log = CallLog(s.calllog_path)
    return {"spent_today_usd": round(log.spent_today(), 6), "daily_budget_usd": s.daily_budget_usd, **log.stats()}


@app.get("/health")
def health():
    s = get_settings()
    return {"ok": True, "store": s.store, "odoo_backend": s.odoo_backend, "routing": s.routing}


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuoteDraft</title><style>
body{font:15px/1.5 system-ui,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#1b1f24;background:#fafafa}
textarea{width:100%;height:220px;font:13px ui-monospace,monospace}table{border-collapse:collapse;width:100%;margin-top:12px}
td,th{border:1px solid #ddd;padding:6px 8px;text-align:left;font-size:14px;vertical-align:top}th{background:#f0f0f0}
.flag{background:#fff4d6}.muted{color:#666;font-size:13px}button{padding:8px 16px;font-size:15px}
</style></head><body><h1>QuoteDraft</h1>
<p class="muted">Paste an RFQ email (with its From: line) and attach the PDF spec if there is one. QuoteDraft creates a <b>draft</b> quotation for a person to review. Nothing is sent to the customer.</p>
<form method="post" action="/ui" enctype="multipart/form-data">
<textarea name="email_text" required placeholder="From: Name &lt;name@company.example&gt;&#10;Subject: RFQ&#10;&#10;Please quote...">{email}</textarea>
<p><input type="file" name="file" accept="application/pdf"> <button type="submit">Draft quote</button></p></form>{result}</body></html>"""


def _render(sheet: ReviewSheet) -> str:
    e = html.escape
    rows = "".join(
        f"<tr class='{'flag' if ln.flagged else ''}'><td>{ln.index + 1}</td><td>{e(ln.description)}</td><td>{ln.quantity:g} {e(ln.unit)}</td>"
        f"<td>{e(ln.sku or 'NO MATCH')}<br><span class='muted'>{e(ln.product_name or '')}</span></td>"
        f"<td>{'' if ln.unit_price is None else f'{ln.unit_price:.2f}'}</td><td>{'' if ln.line_total is None else f'{ln.line_total:.2f}'}</td>"
        f"<td>{ln.confidence:.2f}<br><span class='muted'>{e(ln.model.replace('claude-', ''))}</span></td><td class='muted'>{e(ln.reason)}</td></tr>"
        for ln in sheet.lines)
    warn = "".join(f"<p class='flag'>{e(w)}</p>" for w in sheet.warnings)
    return (f"<h2>Review sheet</h2>{warn}<p>Draft #{sheet.draft_id} in {e(sheet.odoo_backend)} &middot; total <b>{sheet.total:.2f}</b> &middot; "
            f"cost ${sheet.cost_usd:.4f} &middot; {sheet.latency_ms} ms &middot; {sheet.escalations} escalation(s)</p>"
            "<table><tr><th>#</th><th>Customer wrote</th><th>Qty</th><th>Matched SKU</th><th>Unit</th><th>Total</th><th>Conf.</th><th>Why</th></tr>"
            f"{rows}</table><p class='muted'>Highlighted rows need a human decision.</p>")


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE.replace("{email}", "").replace("{result}", "")


@app.post("/ui", response_class=HTMLResponse)
async def ui(email_text: str = Form(...), file: UploadFile | None = File(None)):
    try:
        sheet = await _run(email_text, file)
        result = _render(sheet)
    except HTTPException as exc:
        result = f"<p class='flag'>{html.escape(str(exc.detail))}</p>"
    except Exception as exc:  # noqa: BLE001
        result = f"<p class='flag'>Failed: {html.escape(type(exc).__name__)}: {html.escape(str(exc))}</p>"
    return PAGE.replace("{email}", html.escape(email_text)).replace("{result}", result)
