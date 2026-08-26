"""A fake model: plausible answers, deterministic per seed, occasionally broken.

Breakage is on purpose. The parser and the retry policy are only worth anything
if something upstream misbehaves, so this fake produces the same kinds of
garbage a real model does: prose instead of JSON, invented enum values,
truncated output, and transient errors.
"""

from __future__ import annotations

import asyncio
import json
import random
import re

from app.prompt import Prompt

from .base import ModelTransientError

_KEYWORDS: dict[str, list[str]] = {
    "billing": ["charge", "refund", "invoice", "subscription", "payment", "card", "bill"],
    "technical": ["500", "error", "api", "upload", "timeout", "export", "broken", "bug", "crash", "integration"],
    "account": ["password", "log in", "login", "email address", "account", "profile", "credentials"],
}
_HIGH = re.compile(r"urgent|blocking|production|outage|cannot|can't|locked out|not fixed|still broken|twice", re.I)
_LOW = re.compile(r"not urgent|nice to have|feature request|would love|suggestion", re.I)


def _heuristic(subject: str, body: str) -> dict:
    text = f"{subject}\n{body}".lower()
    scores = {cat: sum(text.count(k) for k in kws) for cat, kws in _KEYWORDS.items()}
    category = max(scores, key=lambda c: scores[c]) if any(scores.values()) else "other"
    if _LOW.search(text):
        priority = "low"
    elif _HIGH.search(text):
        priority = "high"
    else:
        priority = "medium"
    topic = (subject.strip() or body.strip()[:80]).rstrip(".")
    return {"category": category, "priority": priority, "summary": f"Customer needs help with: {topic}."}


def _break(good: dict, rng: random.Random) -> str:
    """Return one of several realistic malformed responses."""
    kind = rng.choice(["prose", "bad_category", "bad_priority", "truncated", "missing_field", "long_summary", "wrong_type"])
    if kind == "prose":
        return f"Sure! This looks like a {good['category']} issue with {good['priority']} priority."
    if kind == "bad_category":
        return json.dumps({**good, "category": rng.choice(["refund", "urgent", "bug", "Billing/Technical"])})
    if kind == "bad_priority":
        return json.dumps({**good, "priority": rng.choice(["critical", "P1", "normal", 3])})
    if kind == "truncated":
        s = json.dumps(good)
        return s[: len(s) // 2]
    if kind == "missing_field":
        return json.dumps({k: v for k, v in good.items() if k != "summary"})
    if kind == "long_summary":
        return json.dumps({**good, "summary": good["summary"] + " Additionally, " * 40})
    return json.dumps({"category": [good["category"]], "priority": good["priority"], "summary": ""})


def _decorate(good: dict, rng: random.Random) -> str:
    """Valid content, sometimes wrapped in things a parser must tolerate."""
    s = json.dumps(good)
    style = rng.choice(["plain", "plain", "fenced", "preamble", "shouty"])
    if style == "fenced":
        return f"```json\n{s}\n```"
    if style == "preamble":
        return f"Here is the classification:\n\n{s}"
    if style == "shouty":
        return json.dumps({**good, "category": good["category"].upper(), "priority": good["priority"].title()})
    return s


class FakeClassifier:
    name = "fake"
    billable = False

    def __init__(self, failure_rate: float = 0.15, latency: float = 0.05, seed: int = 42) -> None:
        self.failure_rate = failure_rate
        self.latency = latency
        self.seed = seed
        self.calls = 0

    async def complete(self, prompt: Prompt) -> str:
        self.calls += 1
        rng = random.Random(f"{self.seed}:{self.calls}:{prompt.user}")
        if self.latency:
            await asyncio.sleep(self.latency)

        ticket = json.loads(prompt.user.split("\n", 1)[1])
        good = _heuristic(ticket["subject"], ticket["body"])

        if rng.random() < self.failure_rate:
            if rng.random() < 0.3:
                raise ModelTransientError("simulated 529 overloaded_error")
            return _break(good, rng)
        return _decorate(good, rng)
