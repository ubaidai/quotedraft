import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quotedraft.embed import HashingEmbedder  # noqa: E402
from quotedraft.store import LocalStore, load_catalog  # noqa: E402

DATA = ROOT / "data"


@pytest.fixture(scope="session")
def catalog():
    return load_catalog(DATA / "catalog.csv")


@pytest.fixture(scope="session")
def labels():
    return json.loads((DATA / "labels.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def store(catalog):
    return LocalStore(catalog, HashingEmbedder())


class FakeLLM:
    """Scripted stand-in for AnthropicLLM. Records every prompt it receives."""

    def __init__(self, extracted, matches):
        self.extracted = extracted  # ExtractedRFQ to return
        self.matches = matches      # list of dicts for submit_matches
        self.prompts: list[str] = []
        self.calls: list[tuple[str, str]] = []

    def parse(self, *, step, model, system, user, output_format, run_id, max_tokens=4000):
        self.prompts.append(system + "\n" + user)
        self.calls.append((step, model))
        return self.extracted

    def create(self, *, step, model, system, messages, tools, run_id, max_tokens=4000):
        self.prompts.append(system + "\n" + json.dumps(messages, default=str))
        self.calls.append((step, model))
        block = SimpleNamespace(type="tool_use", id="tu_1", name="submit_matches", input={"matches": self.matches})
        return SimpleNamespace(content=[block], stop_reason="tool_use")
