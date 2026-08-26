"""API schemas and the validated shape of a classification.

`Classification` is the single gate between model output and the data store:
nothing reaches the database without passing through it.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Category(StrEnum):
    billing = "billing"
    technical = "technical"
    account = "account"
    other = "other"


class Priority(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


class Status(StrEnum):
    pending = "pending"
    classified = "classified"
    failed = "failed"


MAX_SUMMARY_CHARS = 300
_WS = re.compile(r"\s+")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_DROP_CATEGORIES = {"Cc", "Cf", "Cs", "Co", "Cn"}  # control, format (RTL override, zero-width), surrogates, private, unassigned


def plain_text(v: str) -> str:
    """Reduce model output to something safe to show in someone else's UI."""
    v = unicodedata.normalize("NFKC", v)  # fullwidth '＜' becomes '<' before we strip it
    v = "".join(ch for ch in v if unicodedata.category(ch) not in _DROP_CATEGORIES or ch in "\t\n\r")
    v = v.replace("<", "").replace(">", "")
    v = _URL.sub("[link]", v)
    return _WS.sub(" ", v).strip()


class Classification(BaseModel):
    """What the model must produce. Strict on values, lenient on unknown extra keys."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    category: Category
    priority: Priority
    summary: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARS)

    @field_validator("category", "priority", mode="before")
    @classmethod
    def _normalise_enum(cls, v: object) -> object:
        # "Billing" and " HIGH " are accepted; "refund" or "critical" are not.
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("summary", mode="before")
    @classmethod
    def _plain_text(cls, v: object) -> object:
        # The summary is model output and ends up in other systems' UIs: keep it plain text.
        return plain_text(v) if isinstance(v, str) else v


# ---- API ------------------------------------------------------------------

class TicketIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    subject: str = Field(default="", max_length=500)
    body: str = Field(min_length=1, max_length=20_000)


class TicketOut(BaseModel):
    id: str
    subject: str
    body: str
    status: Status
    classification: Classification | None
    attempts: int
    last_error: str | None
    injection_suspected: bool
    prompt_version: str | None
    created_at: datetime
    updated_at: datetime


class TicketList(BaseModel):
    items: list[TicketOut]
    total: int
    limit: int
    offset: int


class ReclassifyStaleRequest(BaseModel):
    confirm: bool = False
    max_usd: float | None = Field(default=None, ge=0, description="Refuse if the estimate exceeds this")


class ReclassifyPreview(BaseModel):
    dry_run: bool
    affected: dict
    estimate: dict
    current_prompt_version: str


class RequeueResult(BaseModel):
    requeued: int
    estimate: dict


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


def to_datetime(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)
