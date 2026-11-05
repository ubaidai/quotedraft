"""Runtime configuration, read from environment variables (and a local .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # dotenv is optional
    pass

DATA_DIR = ROOT / "data"

# USD per 1M tokens (input, output). Source: Anthropic pricing, checked 2026-11-05.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}

CHEAP_MODEL = "claude-haiku-5-5"
STRONG_MODEL = "claude-sonnet-5-5"

# Provider: "anthropic" (ANTHROPIC_API_KEY) or "openrouter" (OPENROUTER_API_KEY, Anthropic-compatible endpoint).
# Internal model names stay the same; only the id sent over the wire changes.
OPENROUTER_BASE_URL = "https://openrouter.ai/api"
OPENROUTER_IDS = {"claude-haiku-5-5": "anthropic/claude-haiku-5.5", "claude-sonnet-5-5": "anthropic/claude-sonnet-5.5"}


def provider() -> str:
    explicit = os.getenv("QD_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    return "openrouter" if os.getenv("OPENROUTER_API_KEY") and not os.getenv("ANTHROPIC_API_KEY") else "anthropic"


@dataclass
class Settings:
    # storage: "local" (in-process, no server needed) or "postgres" (pgvector)
    store: str = field(default_factory=lambda: os.getenv("QD_STORE", "local"))
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    # odoo: "mock" (JSON file) or "xmlrpc" (real Odoo)
    odoo_backend: str = field(default_factory=lambda: os.getenv("ODOO_BACKEND", "mock"))
    odoo_url: str = field(default_factory=lambda: os.getenv("ODOO_URL", "http://localhost:8069"))
    odoo_db: str = field(default_factory=lambda: os.getenv("ODOO_DB", "quotedraft"))
    odoo_user: str = field(default_factory=lambda: os.getenv("ODOO_USER", "admin"))
    odoo_password: str = field(default_factory=lambda: os.getenv("ODOO_PASSWORD", ""))
    # routing: "haiku_first" (default), "haiku_only", "sonnet_only"
    routing: str = field(default_factory=lambda: os.getenv("QD_ROUTING", "haiku_first"))
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("QD_CONFIDENCE_THRESHOLD", "0.6"))
    )
    daily_budget_usd: float = field(default_factory=lambda: float(os.getenv("QD_DAILY_BUDGET_USD", "2.0")))
    embed_model: str = field(
        default_factory=lambda: os.getenv("QD_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    calllog_path: Path = field(default_factory=lambda: Path(os.getenv("QD_CALLLOG", str(DATA_DIR / "quotedraft.db"))))


def get_settings() -> Settings:
    return Settings()
