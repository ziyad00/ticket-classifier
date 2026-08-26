"""The parser is the only door into the data store. These are the locks."""

import pytest

from app.classifier import InvalidModelOutput, parse_classification


def test_plain_json():
    c = parse_classification('{"category": "billing", "priority": "high", "summary": "Refund wanted."}')
    assert (c.category, c.priority, c.summary) == ("billing", "high", "Refund wanted.")


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"category": "technical", "priority": "low", "summary": "x"}\n```',
        'Here is the classification:\n{"category": "technical", "priority": "low", "summary": "x"}',
        '{"category": "TECHNICAL", "priority": " Low ", "summary": "x"}',
        '{"category": "technical", "priority": "low", "summary": "x", "confidence": 0.9}',
        '{"category": "technical", "priority": "low", "summary": "  lots   of\\n whitespace  "}',
    ],
)
def test_tolerated_quirks(raw):
    c = parse_classification(raw)
    assert c.category == "technical" and c.priority == "low"
    assert "\n" not in c.summary and "  " not in c.summary


@pytest.mark.parametrize(
    "raw, fragment",
    [
        ("", "empty"),
        ("Sure! This is a billing issue.", "no JSON"),
        ('{"category": "billing", "priority": "hi}', "malformed"),
        ('["billing", "high", "x"]', "no JSON object"),
        ('{"a": 1} {"b": }', "malformed"),
        ('{"category": "refund", "priority": "high", "summary": "x"}', "category"),
        ('{"category": "billing", "priority": "critical", "summary": "x"}', "priority"),
        ('{"category": "billing", "priority": 3, "summary": "x"}', "priority"),
        ('{"category": ["billing"], "priority": "high", "summary": "x"}', "category"),
        ('{"category": "billing", "priority": "high"}', "summary"),
        ('{"category": "billing", "priority": "high", "summary": ""}', "summary"),
        ('{"category": "billing", "priority": "high", "summary": "   "}', "summary"),
        ('{"category": "billing", "priority": "high", "summary": "' + "x" * 301 + '"}', "summary"),
        ('{"category": null, "priority": "high", "summary": "x"}', "category"),
    ],
)
def test_rejected(raw, fragment):
    with pytest.raises(InvalidModelOutput) as exc:
        parse_classification(raw)
    assert fragment in str(exc.value)


def test_rejects_absurdly_long_response():
    with pytest.raises(InvalidModelOutput, match="too long"):
        parse_classification("{" + " " * 20_000 + "}")
