"""Starts the MCP server as a subprocess and talks to it over stdio, as Claude Desktop does.
Only calls search_catalog (local, no LLM cost)."""
import asyncio
import json
import sys

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .conftest import ROOT


async def _roundtrip():
    params = StdioServerParameters(command=sys.executable, args=["-m", "quotedraft.mcp_server"], cwd=str(ROOT),
                                   env={"QD_STORE": "local"})
    async with stdio_client(params) as streams:
        read, write = streams[0], streams[1]
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name for t in (await session.list_tools()).tools}
            res = await session.call_tool("search_catalog", {"query": "6205 2RS bearing", "k": 3})
            return tools, res


def test_mcp_server_over_stdio():
    tools, res = asyncio.run(asyncio.wait_for(_roundtrip(), timeout=180))
    assert tools == {"search_catalog", "draft_quote_from_text", "get_quote_status"}
    text = "".join(getattr(c, "text", "") for c in res.content)
    payload = res.structured_content if getattr(res, "structured_content", None) else json.loads(text) if text.startswith("[") else text
    assert "BRG-6205-2RS" in json.dumps(payload, default=str)
