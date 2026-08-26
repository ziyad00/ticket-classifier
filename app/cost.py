"""Rough cost estimate for re-running the classifier over N tickets.

Deliberately simple: ~4 characters per token, a fixed output allowance per
ticket, list prices from settings. It exists so that "requeue everything" is a
decision made with a number in front of you, not a guess.
"""

from __future__ import annotations

from app.config import Settings
from app.prompt import SYSTEM

CHARS_PER_TOKEN = 4
PROMPT_OVERHEAD_CHARS = 60  # the "Ticket to classify:" wrapper + JSON punctuation
OUTPUT_TOKENS_PER_TICKET = 60  # a JSON object with a one-sentence summary


def estimate(n_tickets: int, ticket_chars: int, settings: Settings, classifier_name: str, billable: bool) -> dict:
    system_tokens = len(SYSTEM) // CHARS_PER_TOKEN
    input_tokens = n_tickets * (system_tokens + PROMPT_OVERHEAD_CHARS // CHARS_PER_TOKEN) + ticket_chars // CHARS_PER_TOKEN
    output_tokens = n_tickets * OUTPUT_TOKENS_PER_TICKET
    usd = (input_tokens * settings.price_input_per_mtok + output_tokens * settings.price_output_per_mtok) / 1_000_000
    return {
        "classifier": classifier_name,
        "model": settings.llm_model if billable else None,
        "model_calls": n_tickets,
        "model_calls_worst_case": n_tickets * settings.max_attempts,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "usd": round(usd, 4) if billable else 0.0,
        "usd_worst_case": round(usd * settings.max_attempts, 4) if billable else 0.0,
        "basis": (
            f"~{CHARS_PER_TOKEN} chars/token, {OUTPUT_TOKENS_PER_TICKET} output tokens/ticket, "
            f"${settings.price_input_per_mtok}/M in, ${settings.price_output_per_mtok}/M out; "
            f"worst case assumes every ticket needs {settings.max_attempts} attempts"
        ),
    }
