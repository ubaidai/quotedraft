from fastapi.testclient import TestClient

from quotedraft import api
from quotedraft.calllog import CallLog
from quotedraft.config import Settings
from quotedraft.models import ExtractedLine, ExtractedRFQ
from quotedraft.odoo import MockOdoo
from quotedraft.pipeline import Pipeline

from .conftest import FakeLLM


def test_ui_and_api_roundtrip(tmp_path, store, monkeypatch):
    llm = FakeLLM(ExtractedRFQ(line_items=[ExtractedLine(description="6205-2RS", quantity=10, unit="pcs")]),
                  [{"line_index": 0, "sku": "BRG-6205-2RS", "confidence": 0.9, "reason": "code"}])
    pipe = Pipeline(settings=Settings(), llm=llm, store=store, odoo=MockOdoo(tmp_path / "o.json"), calllog=CallLog(":memory:"))
    monkeypatch.setattr(api, "pipeline", lambda: pipe)
    monkeypatch.setattr(api, "RUNS", tmp_path / "runs")
    c = TestClient(api.app)
    assert "QuoteDraft" in c.get("/").text
    r = c.post("/rfq", data={"email_text": "From: A B <a@b.example>\nSubject: x\n\n6205-2RS x10"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["lines"][0]["sku"] == "BRG-6205-2RS" and body["draft_id"] == 1
    assert c.get(f"/rfq/{body['run_id']}").json()["run_id"] == body["run_id"]
    assert c.get("/rfq/../../etc").status_code == 404
    page = c.post("/ui", data={"email_text": "From: A B <a@b.example>\nSubject: x\n\n6205-2RS x10"}).text
    assert "BRG-6205-2RS" in page and "Review sheet" in page
    assert c.post("/rfq", data={"email_text": "x"}, files={"file": ("a.txt", b"hi")}).status_code == 400


def test_mcp_server_exposes_three_tools():
    import asyncio

    from quotedraft.mcp_server import mcp

    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert names == {"search_catalog", "draft_quote_from_text", "get_quote_status"}
