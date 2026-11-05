"""MCP server so Claude Desktop / Claude Code can use QuoteDraft as tools.

  python -m quotedraft.mcp_server          (stdio transport)
"""
from __future__ import annotations

from functools import lru_cache

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("quotedraft")


@lru_cache(maxsize=1)
def _pipeline():
    from .pipeline import build_pipeline

    return build_pipeline()


@mcp.tool()
def search_catalog(query: str, k: int = 8) -> list[dict]:
    """Search the product catalog (hybrid keyword + semantic). Returns sku, name, unit and price."""
    return [c.model_dump() for c in _pipeline().store.search(query, k=max(1, min(k, 20)))]


@mcp.tool()
def draft_quote_from_text(rfq_text: str) -> dict:
    """Turn an RFQ email (include its 'From:' line) into a DRAFT quotation for human review.
    Returns the review sheet: matched SKUs, prices, confidence, flagged lines and the draft id.
    Never confirms or sends anything."""
    return _pipeline().run(rfq_text).model_dump()


@mcp.tool()
def get_quote_status(quote_id: int) -> dict:
    """Read a draft quotation back from Odoo (or the mock store): name, state, partner and amount."""
    q = _pipeline().odoo.get_quote(quote_id)
    return q or {"error": f"no quote {quote_id}"}


if __name__ == "__main__":
    mcp.run()
