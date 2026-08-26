"""The model boundary.

A Classifier turns a prompt into *text*. That is all we assume about it. Turning
that text into something trustworthy happens in `parse.py`, never here.
"""

from __future__ import annotations

from typing import Protocol

from app.prompt import Prompt


class ModelError(Exception):
    """Base for anything that goes wrong talking to the model."""


class ModelTransientError(ModelError):
    """Worth retrying: rate limit, overload, network, timeout."""


class ModelPermanentError(ModelError):
    """Not worth retrying: bad credentials, bad request, refusal, unknown model."""


class InvalidModelOutput(ModelError):
    """The model answered, but not with something we can store."""


class Classifier(Protocol):
    name: str
    billable: bool  # False for fakes; drives the re-classification cost estimate

    async def complete(self, prompt: Prompt) -> str:
        """Return the raw model text. May raise ModelError."""
        ...
