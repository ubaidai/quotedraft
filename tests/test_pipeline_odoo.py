import xmlrpc.client

import pytest

from quotedraft.calllog import BudgetExceeded, CallLog, cost_usd
from quotedraft.config import CHEAP_MODEL, STRONG_MODEL, Settings
from quotedraft.match import _validate
from quotedraft.models import ExtractedLine, ExtractedRFQ
from quotedraft.odoo import MockOdoo, QuoteLine, XmlRpcOdoo, order_line_commands
from quotedraft.pipeline import Pipeline

from .conftest import FakeLLM

RAW = """From: Megan Doyle <mdoyle@caldervale.example>
Subject: RFQ - Caldervale - 99

Hello,

1. 6205-2RS - 100 pcs
2. hydraulic cylinder 50mm bore - 2 pcs

Regards,
Megan Doyle
Caldervale Facilities
Tel: +44 7700 912345
"""


def make_pipeline(tmp_path, store, matches, routing="haiku_first"):
    extracted = ExtractedRFQ(line_items=[ExtractedLine(description="6205-2RS", quantity=100, unit="pcs"),
                                         ExtractedLine(description="hydraulic cylinder 50mm bore", quantity=2, unit="pcs")])
    llm = FakeLLM(extracted, matches)
    s = Settings()
    s.routing = routing
    calllog = CallLog(":memory:")
    return Pipeline(settings=s, llm=llm, store=store, odoo=MockOdoo(tmp_path / "odoo.json"), calllog=calllog), llm


def test_pipeline_drafts_matched_lines_and_notes_unmatched(tmp_path, store):
    pipe, llm = make_pipeline(tmp_path, store, [
        {"line_index": 0, "sku": "BRG-6205-2RS", "confidence": 0.95, "reason": "exact code"},
        {"line_index": 1, "sku": None, "confidence": 0.9, "reason": "not stocked"}])
    sheet = pipe.run(RAW)
    assert sheet.draft_id == 1
    q = pipe.odoo.get_quote(1)
    assert q["state"] == "draft"
    assert [ln["sku"] for ln in q["order_line"]] == ["BRG-6205-2RS"]
    assert "hydraulic cylinder" in q["note"]
    assert sheet.lines[1].flagged and sheet.lines[1].sku is None
    assert sheet.total == pytest.approx(100 * store.get("BRG-6205-2RS").price)


def test_no_personal_data_reaches_the_model(tmp_path, store):
    pipe, llm = make_pipeline(tmp_path, store, [
        {"line_index": 0, "sku": "BRG-6205-2RS", "confidence": 0.95, "reason": ""},
        {"line_index": 1, "sku": None, "confidence": 0.9, "reason": ""}])
    pipe.run(RAW)
    everything = "\n".join(llm.prompts).lower()
    for secret in ["mdoyle@caldervale.example", "megan", "doyle", "7700 912345", "caldervale facilities"]:
        assert secret not in everything


def test_low_confidence_lines_escalate_to_stronger_model(tmp_path, store):
    pipe, llm = make_pipeline(tmp_path, store, [
        {"line_index": 0, "sku": "BRG-6205-2RS", "confidence": 0.3, "reason": "unsure"},
        {"line_index": 1, "sku": None, "confidence": 0.9, "reason": ""}])
    sheet = pipe.run(RAW)
    assert ("match", CHEAP_MODEL) in llm.calls and ("match", STRONG_MODEL) in llm.calls
    assert sheet.escalations == 1


def test_sonnet_only_never_calls_haiku(tmp_path, store):
    pipe, llm = make_pipeline(tmp_path, store, [
        {"line_index": 0, "sku": "BRG-6205-2RS", "confidence": 0.95, "reason": ""},
        {"line_index": 1, "sku": None, "confidence": 0.9, "reason": ""}], routing="sonnet_only")
    pipe.run(RAW)
    assert {m for _, m in llm.calls} == {STRONG_MODEL}


def test_matcher_cannot_invent_a_sku(store):
    out = _validate([{"line_index": 0, "sku": "BRG-9999-XX", "confidence": 0.99, "reason": "made up"},
                     {"line_index": 1, "sku": "HB-M12-050-ZP", "confidence": 0.9, "reason": "real but never shown"}],
                    wanted={0, 1, 2}, shown={"BRG-6205-2RS"}, store=store, model="m")
    assert out[0].sku is None and out[0].confidence == 0
    assert out[1].sku is None
    assert out[2].sku is None and "no match" in out[2].reason


def test_budget_guard_blocks_calls():
    log = CallLog(":memory:")
    log.record(run_id="r", step="extract", model=STRONG_MODEL, input_tokens=1_000_000, output_tokens=0, latency_ms=1, success=True)
    assert log.spent_today() == pytest.approx(2.0)
    with pytest.raises(BudgetExceeded):
        log.check_budget(1.0)
    assert cost_usd(CHEAP_MODEL, 1_000_000, 1_000_000) == pytest.approx(0.60)


class FakeOdooServer:
    """Stands in for xmlrpc.client.ServerProxy and records execute_kw calls."""

    calls: list = []

    def __init__(self, url, **kw):
        self.url = url

    def authenticate(self, db, user, pw, ctx):
        return 7

    def execute_kw(self, db, uid, pw, model, method, args, kw):
        FakeOdooServer.calls.append((model, method, args, kw))
        if (model, method) == ("product.product", "search_read"):
            return [{"id": 100 + i, "default_code": sku} for i, sku in enumerate(args[0][0][2])]
        if (model, method) == ("sale.order", "create"):
            return 55
        return []


def test_xmlrpc_creates_draft_sale_order_only(monkeypatch):
    monkeypatch.setattr(xmlrpc.client, "ServerProxy", FakeOdooServer)
    FakeOdooServer.calls = []
    odoo = XmlRpcOdoo("http://odoo", "db", "admin", "pw")
    qid = odoo.create_draft_quote(3, [QuoteLine("BRG-6205-2RS", "Bearing", 100, 5.3)], "note", "REF")
    assert qid == 55
    create = [c for c in FakeOdooServer.calls if c[:2] == ("sale.order", "create")][0]
    vals = create[2][0]
    assert vals["partner_id"] == 3
    assert vals["order_line"] == [(0, 0, {"product_id": 100, "product_uom_qty": 100, "price_unit": 5.3})]
    methods = {c[1] for c in FakeOdooServer.calls}
    assert not methods & {"action_confirm", "action_quotation_send", "message_post"}


def test_unit_check_converts_metres_to_lengths_and_flags():
    from quotedraft.units import reconcile

    assert reconcile(60, "m", "length") == (10.0, reconcile(60, "m", "length")[1])
    assert "10 lengths" in reconcile(60, "m", "length")[1]
    assert reconcile(61, "metres", "length")[0] == 11.0
    assert reconcile(10, "lengths", "length") == (10, None)
    assert reconcile(200, "pcs", "pcs") == (200, None)
    assert reconcile(5, "packs", "pcs")[1] is not None


def test_unit_mismatch_flags_line_even_at_high_confidence(tmp_path, store):
    from quotedraft.models import ExtractedLine, ExtractedRFQ

    llm = FakeLLM(ExtractedRFQ(line_items=[ExtractedLine(description="equal angle 75x75x8 S275", quantity=60, unit="m")]),
                  [{"line_index": 0, "sku": "EA-75x75x8-S275", "confidence": 0.93, "reason": "exact section"}])
    pipe = Pipeline(settings=Settings(), llm=llm, store=store, odoo=MockOdoo(tmp_path / "o.json"), calllog=CallLog(":memory:"))
    line = pipe.run("From: A B <a@b.example>\nSubject: x\n\nangle").lines[0]
    assert line.quantity == 10 and line.unit == "length" and line.flagged


def test_order_line_commands_shape():
    cmds = order_line_commands({"A": 1}, [QuoteLine("A", "a", 2, 3.5)])
    assert cmds == [(0, 0, {"product_id": 1, "product_uom_qty": 2, "price_unit": 3.5})]
