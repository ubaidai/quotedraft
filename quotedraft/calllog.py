"""Every LLM call is logged with tokens, cost and latency. Also enforces the daily budget."""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .config import PRICES

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    run_id TEXT,
    step TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    latency_ms INTEGER NOT NULL,
    success INTEGER NOT NULL,
    error TEXT
)"""


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICES[model]
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


class BudgetExceeded(RuntimeError):
    pass


class CallLog:
    """SQLite by default. Point QD_CALLLOG at a path; in Postgres deployments the same table
    lives next to the catalog (see docker/init.sql)."""

    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(SCHEMA)
        self._conn.commit()

    def record(self, *, run_id: str | None, step: str, model: str, input_tokens: int, output_tokens: int,
               latency_ms: int, success: bool, error: str | None = None) -> float:
        c = cost_usd(model, input_tokens, output_tokens)
        with self._lock:
            self._conn.execute(
                "INSERT INTO llm_calls (ts, run_id, step, model, input_tokens, output_tokens, cost_usd, latency_ms, success, error)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), run_id, step, model, input_tokens, output_tokens, c,
                 latency_ms, int(success), error))
            self._conn.commit()
        return c

    def spent_today(self) -> float:
        today = datetime.now(timezone.utc).date().isoformat()
        with self._lock:
            row = self._conn.execute("SELECT COALESCE(SUM(cost_usd),0) FROM llm_calls WHERE substr(ts,1,10)=?", (today,)).fetchone()
        return float(row[0])

    def check_budget(self, limit_usd: float):
        spent = self.spent_today()
        if spent >= limit_usd:
            raise BudgetExceeded(f"Daily LLM budget reached: ${spent:.4f} of ${limit_usd:.2f}. Raise QD_DAILY_BUDGET_USD to continue.")

    def run_totals(self, run_id: str) -> dict:
        with self._lock:
            r = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(cost_usd),0), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0)"
                " FROM llm_calls WHERE run_id=?", (run_id,)).fetchone()
        return {"calls": r[0], "cost_usd": r[1], "input_tokens": r[2], "output_tokens": r[3]}

    def stats(self) -> dict:
        with self._lock:
            by_day = self._conn.execute(
                "SELECT substr(ts,1,10) d, COUNT(*), SUM(cost_usd), SUM(input_tokens), SUM(output_tokens)"
                " FROM llm_calls GROUP BY d ORDER BY d DESC LIMIT 30").fetchall()
            by_step = self._conn.execute(
                "SELECT step, model, COUNT(*), SUM(cost_usd), AVG(latency_ms), SUM(1-success)"
                " FROM llm_calls GROUP BY step, model ORDER BY step").fetchall()
        return {
            "by_day": [{"date": d, "calls": n, "cost_usd": round(c, 6), "input_tokens": i, "output_tokens": o} for d, n, c, i, o in by_day],
            "by_step": [{"step": s, "model": m, "calls": n, "cost_usd": round(c, 6), "avg_latency_ms": round(lat), "failures": f}
                        for s, m, n, c, lat, f in by_step],
        }
