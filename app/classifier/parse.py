"""Turn raw model text into a validated Classification, or refuse."""

from __future__ import annotations

import json

from pydantic import ValidationError

from app.models import Classification
from app.safety import looks_like_injection

from .base import InvalidModelOutput

MAX_RAW_CHARS = 10_000


def parse_classification(raw: str) -> Classification:
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidModelOutput("empty response")
    if len(raw) > MAX_RAW_CHARS:
        raise InvalidModelOutput(f"response too long ({len(raw)} chars)")

    # Tolerate a preamble or code fences; the first '{' to the last '}' must be the object.
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise InvalidModelOutput("no JSON object in response")
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as e:
        raise InvalidModelOutput(f"malformed JSON: {e.msg}") from e
    except (ValueError, RecursionError) as e:  # absurd numbers, pathological nesting
        raise InvalidModelOutput(f"malformed JSON: {type(e).__name__}") from e
    if not isinstance(data, dict):
        raise InvalidModelOutput("JSON is not an object")

    try:
        result = Classification.model_validate(data)
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc'])) or '?'}: {err['msg']}" for err in e.errors())
        raise InvalidModelOutput(f"schema violation: {problems}") from e

    # A summary that reads like an instruction means the model echoed injected text.
    if looks_like_injection("", result.summary):
        raise InvalidModelOutput("summary echoes instruction-like text")
    return result
