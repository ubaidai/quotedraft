"""Run one RFQ end to end and print the review sheet. Costs well under one cent on Haiku.

  python scripts/smoke.py rfq_02
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quotedraft.config import DATA_DIR, provider  # noqa: E402
from quotedraft.pipeline import build_pipeline  # noqa: E402

if __name__ == "__main__":
    import json

    rid = sys.argv[1] if len(sys.argv) > 1 else "rfq_02"
    labels = json.loads((DATA_DIR / "labels.json").read_text(encoding="utf-8"))
    att = DATA_DIR / "rfqs" / labels[rid]["attachment"] if labels[rid].get("attachment") else None
    pipe = build_pipeline(create_drafts=True)
    print(f"provider={provider()}  structured-output mode will be reported below")
    sheet = pipe.run((DATA_DIR / "rfqs" / f"{rid}.txt").read_text(encoding="utf-8"), att)
    print(f"parse mode: {'tool fallback' if pipe.llm.tool_parse else 'messages.parse (structured outputs)'}")
    for ln in sheet.lines:
        print(f"  {ln.description!r:55} qty {ln.quantity:g} -> {ln.sku} ({ln.confidence:.2f}, {ln.model}){'  FLAG' if ln.flagged else ''}")
    print(f"gold: {[ (l['text'], l['quantity'], l['sku']) for l in labels[rid]['lines']]}")
    print(f"draft #{sheet.draft_id} ({sheet.odoo_backend})  total {sheet.total}  cost ${sheet.cost_usd:.5f}  {sheet.latency_ms} ms  escalations {sheet.escalations}")
