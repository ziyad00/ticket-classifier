from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from app.classifier import ModelPermanentError, ModelTransientError
from app.config import Settings
from app.main import create_app
from app.prompt import Prompt

SAMPLES = json.loads((Path(__file__).parent.parent / "data" / "sample_tickets.json").read_text())


class ScriptedClassifier:
    """Returns (or raises) each scripted item in order; the last one repeats forever."""

    name = "scripted"

    def __init__(self, *script: str | Exception, latency: float = 0.0) -> None:
        self.script = list(script)
        self.latency = latency
        self.calls = 0
        self.prompts: list[Prompt] = []
        self.active = 0
        self.max_active = 0

    async def complete(self, prompt: Prompt) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.latency:
                await asyncio.sleep(self.latency)
            item = self.script[min(self.calls, len(self.script)) - 1]
            if isinstance(item, Exception):
                raise item
            return item
        finally:
            self.active -= 1


GOOD = '{"category": "billing", "priority": "high", "summary": "Customer was charged twice and wants a refund."}'
TRANSIENT = ModelTransientError("529 overloaded")
PERMANENT = ModelPermanentError("401 invalid api key")


def fast_settings(**overrides) -> Settings:
    base = dict(
        db_path=":memory:",
        worker_concurrency=2,
        max_attempts=3,
        retry_base_delay=0.01,
        llm_timeout=1.0,
        lease_seconds=60,
        poll_interval=0.02,
        shutdown_grace=1.0,
        fake_latency=0.0,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
async def make_client():
    """Yields a factory: `client, app = await make_client(classifier, **settings)`. Runs lifespan."""
    stack: list = []

    async def _make(classifier=None, **overrides):
        app = create_app(fast_settings(**overrides), classifier)
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        stack.append((client, lifespan))
        return client, app

    yield _make
    for client, lifespan in reversed(stack):
        await client.aclose()
        await lifespan.__aexit__(None, None, None)


async def wait_for_status(client: httpx.AsyncClient, ticket_id: str, *statuses: str, timeout: float = 3.0) -> dict:
    """Poll until the ticket reaches one of `statuses`. Fails loudly instead of hanging."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        r = await client.get(f"/tickets/{ticket_id}")
        data = r.json()
        if data.get("status") in statuses:
            return data
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError(f"{ticket_id} stuck in {data.get('status')!r}: {data}")
        await asyncio.sleep(0.02)


async def post_ticket(client: httpx.AsyncClient, id: str = "t-1", subject: str = "Subject", body: str = "Body text") -> httpx.Response:
    return await client.post("/tickets", json={"id": id, "subject": subject, "body": body})
