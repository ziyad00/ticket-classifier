"""Heuristics for untrusted ticket content.

This does not block anything. It marks tickets that look like they are trying to
steer the classifier so a human can eyeball them. The real defences are in
`prompt.py` (content is passed as data) and `models.py` (output is validated).
"""

from __future__ import annotations

import re

_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) instructions",
    r"disregard (all |any )?(previous|prior|above) instructions",
    r"you are now\b",
    r"\bsystem prompt\b",
    r"\bclassify (it|this|the ticket) as\b",
    r"\bsummari[sz]e (it|this) as\b",
    r"\bset (the )?priority to\b",
    r"\bthis (ticket|message) is from the (ceo|cto|admin|owner|founder)\b",
]
_INJECTION = re.compile("|".join(f"(?:{p})" for p in _PATTERNS), re.IGNORECASE)


def looks_like_injection(subject: str, body: str) -> bool:
    return bool(_INJECTION.search(f"{subject}\n{body}"))
