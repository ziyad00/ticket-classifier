"""Turn raw model text into a validated Classification, or refuse."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from app.models import Classification
from app.safety import looks_like_injection

from .base import InvalidModelOutput

MAX_RAW_CHARS = 10_000
MAX_BRACKETS = 20  # a three-field object needs 1; anything deeply nested is not an answer
MAX_SUMMARY_WORDS = 40
_LETTERS = re.compile(r"[^\W\d_]")  # any letter in any script
_SELF_REFERENCE = re.compile(r"\b(prompt|instructions?|assistant|ai|llm|language model|classifier|system message)\b", re.I)


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    d: dict = {}
    for k, v in pairs:
        if k in d:
            raise InvalidModelOutput(f"ambiguous: key {k!r} appears more than once")
        d[k] = v
    return d


def _exactly_one_object(text: str) -> None:
    """Reject 'here are two possible answers' style responses."""
    decoder = json.JSONDecoder()
    count, i = 0, 0
    while (i := text.find("{", i)) != -1:
        try:
            _, end = decoder.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        count += 1
        i = end
    if count > 1:
        raise InvalidModelOutput(f"ambiguous: {count} JSON objects in response")


def parse_classification(raw: str) -> Classification:
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidModelOutput("empty response")
    if len(raw) > MAX_RAW_CHARS:
        raise InvalidModelOutput(f"response too long ({len(raw)} chars)")

    if raw.count("{") + raw.count("[") > MAX_BRACKETS:
        raise InvalidModelOutput("response is not a flat object (too many brackets)")

    # Tolerate a preamble or code fences; the first '{' to the last '}' must be the object.
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise InvalidModelOutput("no JSON object in response")
    _exactly_one_object(raw)
    try:
        data = json.loads(raw[start : end + 1], object_pairs_hook=_no_duplicate_keys)
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

    # The summary must be a short sentence about the customer, not about the classifier.
    if looks_like_injection("", result.summary):
        raise InvalidModelOutput("summary echoes instruction-like text")
    if not _LETTERS.search(result.summary):
        raise InvalidModelOutput("summary contains no words")
    if len(result.summary.split()) > MAX_SUMMARY_WORDS:
        raise InvalidModelOutput(f"summary longer than {MAX_SUMMARY_WORDS} words")
    if m := _SELF_REFERENCE.search(result.summary):
        raise InvalidModelOutput(f"summary refers to the classifier ({m.group(0)!r})")
    return result
