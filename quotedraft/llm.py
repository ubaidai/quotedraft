"""Thin wrapper over the Anthropic SDK that logs every call and enforces the budget.

The pipeline talks to the `LLM` protocol, so tests swap in a fake without touching the network.
"""
from __future__ import annotations

import time
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from .calllog import CallLog

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    def parse(self, *, step: str, model: str, system: str, user: str, output_format: type[T],
              run_id: str | None, max_tokens: int = 4000) -> T: ...

    def create(self, *, step: str, model: str, system: str, messages: list[dict], tools: list[dict],
               run_id: str | None, max_tokens: int = 4000) -> Any: ...


class AnthropicLLM:
    def __init__(self, calllog: CallLog, budget_usd: float):
        import anthropic

        import os

        from .config import OPENROUTER_BASE_URL, OPENROUTER_IDS, provider

        self._anthropic = anthropic
        self.provider = provider()
        if self.provider == "openrouter":
            key = os.getenv("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("QD_PROVIDER=openrouter but OPENROUTER_API_KEY is not set in .env")
            # OpenRouter's Anthropic-compatible endpoint takes the key as a bearer token
            self.client = anthropic.Anthropic(api_key=None, auth_token=key, base_url=OPENROUTER_BASE_URL)
            self._ids = OPENROUTER_IDS
        else:
            self.client = anthropic.Anthropic()
            self._ids = {}
        # Structured outputs (messages.parse) first; if the endpoint rejects them, fall back to
        # a single tool whose input schema is the Pydantic model, validated the same way.
        self.tool_parse = False
        self.calllog = calllog
        self.budget_usd = budget_usd

    def _id(self, model: str) -> str:
        return self._ids.get(model, model)

    def _parse_via_tool(self, model, system, user, output_format, max_tokens):
        tool = {"name": "record_result", "description": "Record the extracted result.",
                "input_schema": output_format.model_json_schema()}
        resp = self.client.messages.create(
            model=self._id(model), max_tokens=max_tokens, tools=[tool],
            system=system + "\n\nReturn your answer only by calling the record_result tool.",
            messages=[{"role": "user", "content": user}])
        tu = next((b for b in resp.content if getattr(b, "type", None) == "tool_use"), None)
        parsed = output_format.model_validate(tu.input) if tu is not None else None
        return resp, parsed

    def _log(self, step, model, run_id, usage, t0, ok, err=None):
        self.calllog.record(run_id=run_id, step=step, model=model,
                            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
                            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
                            latency_ms=int((time.perf_counter() - t0) * 1000), success=ok, error=err)

    def parse(self, *, step, model, system, user, output_format, run_id, max_tokens=4000):
        self.calllog.check_budget(self.budget_usd)
        t0 = time.perf_counter()
        try:
            if self.tool_parse:
                resp, parsed = self._parse_via_tool(model, system, user, output_format, max_tokens)
            else:
                try:
                    resp = self.client.messages.parse(
                        model=self._id(model), max_tokens=max_tokens, system=system,
                        messages=[{"role": "user", "content": user}], output_format=output_format)
                    parsed = resp.parsed_output
                except self._anthropic.BadRequestError as exc:
                    if "output" not in str(exc).lower() and "format" not in str(exc).lower():
                        raise
                    self.tool_parse = True  # endpoint does not support output_config.format
                    resp, parsed = self._parse_via_tool(model, system, user, output_format, max_tokens)
        except self._anthropic.APIStatusError as exc:
            self._log(step, model, run_id, None, t0, False, f"{exc.status_code}: {exc.message}"[:500])
            raise LLMError(str(exc)) from exc
        except self._anthropic.APIConnectionError as exc:
            self._log(step, model, run_id, None, t0, False, "connection error")
            raise LLMError(str(exc)) from exc
        except Exception as exc:  # validation failure of the parsed output
            self._log(step, model, run_id, None, t0, False, f"{type(exc).__name__}: {exc}"[:500])
            raise LLMError(str(exc)) from exc
        if resp.stop_reason in ("refusal", "max_tokens") or parsed is None:
            self._log(step, model, run_id, resp.usage, t0, False, f"stop_reason={resp.stop_reason}")
            raise LLMError(f"no parsed output (stop_reason={resp.stop_reason})")
        self._log(step, model, run_id, resp.usage, t0, True)
        return parsed

    def create(self, *, step, model, system, messages, tools, run_id, max_tokens=4000):
        self.calllog.check_budget(self.budget_usd)
        t0 = time.perf_counter()
        try:
            resp = self.client.messages.create(model=self._id(model), max_tokens=max_tokens, system=system,
                                               messages=messages, tools=tools)
        except (self._anthropic.APIStatusError, self._anthropic.APIConnectionError) as exc:
            self._log(step, model, run_id, None, t0, False, str(exc)[:500])
            raise LLMError(str(exc)) from exc
        self._log(step, model, run_id, resp.usage, t0, resp.stop_reason != "refusal")
        return resp
