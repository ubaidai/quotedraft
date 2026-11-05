# QuoteDraft: build brief

Paste this file into Claude Code in an empty folder and say: "Build this, phase by phase. Stop after each phase so I can run it."

## What it is

An AI agent that reads an incoming request-for-quote (an email plus a PDF or scanned spec), works out which catalog products the customer wants, and creates a **draft** quotation in Odoo for a person to approve. It never confirms or sends anything on its own.

## Why this project

It mirrors the AI Engineer job description almost line by line:

| Job description says | QuoteDraft has |
|---|---|
| Agents that read emails, specs and documents and prepare quotes and purchase orders for approval | The whole pipeline, ending in an Odoo draft |
| RAG with embeddings and a vector database such as pgvector | Catalog search in Postgres + pgvector, hybrid with full-text |
| LLM APIs, structured outputs, tool calling | Pydantic-validated extraction, tool calls to search and Odoo |
| Connect LLMs to Odoo through tool calling | Odoo XML-RPC tools: find customer, find product, create draft `sale.order` |
| Testing, logging and cost tracking for every AI feature | Labelled eval set, per-call token, cost and latency log |
| No company or customer data sent without approval | PII redaction before any LLM call |
| Nice to have: MCP, OCR, n8n | MCP server, Tesseract OCR fallback, optional n8n email trigger |

It also covers both reels: evals, LLMOps (cost and latency), auto-routing between models, MCP, and shipping a working product.

## Stack

- Python 3.12, FastAPI, Pydantic v2
- PostgreSQL 16 + pgvector (Docker)
- Odoo 17 Community (Docker, official image) with the Sales module installed
- Anthropic API: `claude-haiku-5-5` first, escalating to `claude-sonnet-5-5` (see routing)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, run locally (no catalog data leaves the machine)
- PDF text: `pdfplumber`; scanned pages: `pytesseract`
- MCP: official `mcp` Python SDK (FastMCP)
- `docker compose up` starts everything

## Pipeline

```
RFQ email + attachment
  -> 1. extract text (pdfplumber, Tesseract fallback for scans)
  -> 2. redact PII (emails, phones, names -> placeholders, kept in a local map)
  -> 3. LLM structured extraction -> RFQ { customer_ref, line_items[{description, qty, unit, spec}] }
         validated by Pydantic; on failure: retry once with the validation error, then escalate model
  -> 4. per line item: hybrid search over catalog (pgvector + Postgres full-text, merged with RRF)
         -> top 5 candidates -> LLM picks a SKU or returns NO_MATCH with a reason
  -> 5. tool calls to Odoo: find/create customer, create sale.order in DRAFT state with lines
  -> 6. return the draft plus a review sheet: each line, the matched SKU, confidence, and why
         low-confidence or NO_MATCH lines are flagged for the human
```

## Model routing (auto-routing)

- Every step runs on Haiku first.
- Escalate to Sonnet only when Pydantic validation fails twice, the match confidence is under 0.6, or the item is flagged ambiguous.
- Log which model handled each step. The eval report shows the cost of routing versus "always Sonnet". That comparison is your interview story.

## Data (all synthetic, mark it as fake)

- `data/catalog.csv`: 200 products for a made-up industrial supplier (fasteners, valves, steel sections, pumps). Columns: sku, name, description, unit, price, attributes.
- `data/rfqs/`: 25 synthetic RFQ emails with PDF attachments. Make 5 of them scanned images and 5 messy (typos, mixed units, "same as last time", one item that isn't in the catalog).
- `data/labels.json`: the correct line items and SKUs for every RFQ. Write the labels by hand. Don't let the model label its own test.

## Eval harness (`python -m eval.run`)

Reports per run, saved to `eval/results/<timestamp>.json` and printed as a table:

- Line-item extraction precision and recall
- SKU match accuracy@1 and recall@5 for retrieval alone
- Retrieval comparison: vector-only vs full-text-only vs hybrid
- NO_MATCH correctness (did it refuse the item that isn't in the catalog?)
- Quote total exact-match rate
- Cost per RFQ, p50 and p95 latency, % of steps escalated to Sonnet

Write down one real failure the eval found and the change that fixed it, in the README. That beats any claim.

## Logging and cost

- `llm_calls` table: timestamp, step, model, input and output tokens, cost in USD, latency in ms, success.
- `GET /stats` returns totals per day and per step.
- A hard budget guard: refuse new runs once daily spend passes a limit set in `.env`.

## MCP server

Expose these tools so Claude Desktop or Claude Code can use QuoteDraft:
`search_catalog(query)`, `draft_quote_from_text(rfq_text)`, `get_quote_status(quote_id)`.
Include the `claude_desktop_config.json` snippet in the README.

## API

- `POST /rfq` (email text plus file upload) returns the review sheet and the Odoo draft id
- `GET /rfq/{id}`
- `GET /stats`
- A small HTML page with an upload form and the review sheet. Keep it plain.

## Phases (about 10 to 12 hours)

1. **(1.5h) Setup:** docker compose with Postgres+pgvector and Odoo 17. Create the Odoo database and install Sales. Generate the synthetic catalog and load it into both Postgres and Odoo.
2. **(1.5h) Retrieval:** embed the catalog, write hybrid search with RRF, and a retrieval-only eval (recall@5) that compares the three strategies.
3. **(2h) Extraction:** PDF and OCR text, PII redaction, Pydantic schema, retry and escalation.
4. **(2h) Matching + Odoo tools:** SKU picking, then XML-RPC tools that create the draft `sale.order`. Verify the draft appears in the Odoo UI.
5. **(1.5h) Eval harness + cost logging:** full eval, the `llm_calls` table, budget guard, routing comparison.
6. **(1h) MCP server + FastAPI + upload page.**
7. **(1h) README:** architecture diagram, eval table with real numbers, one failure and its fix, a 60-second Loom demo, how to run.

**If Odoo eats more than 2 hours:** put the Odoo calls behind an interface, add a `MockOdoo` that writes drafts to Postgres, finish everything else, and come back to real Odoo last. Never let one integration kill the day.

**Stretch, only if done early:** an n8n workflow that watches a Gmail label and posts new RFQs to `POST /rfq`.

## Rules for the build

- Tests for the parser, redaction, hybrid search and Odoo tool payloads (pytest).
- No secrets in the repo: `.env.example` only.
- Nothing is ever confirmed or emailed. Drafts only.
- Every number in the README comes from an actual eval run.

## The resume bullet it earns (fill in real numbers after the eval)

> **QuoteDraft: RFQ-to-quotation agent** | Python, FastAPI, PostgreSQL, pgvector, Odoo, Anthropic API, MCP, Docker
> Agent that reads RFQ emails and scanned specs and creates draft Odoo quotations for human approval. Hybrid pgvector + full-text retrieval raised SKU recall@5 from X% to Y%. Haiku-first routing cut cost per RFQ by Z% versus Sonnet-only at equal accuracy on a hand-labelled 25-RFQ eval set. PII is redacted before every LLM call; every call is logged with tokens, cost and latency.
