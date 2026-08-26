"""Real provider. Optional: `pip install -e ".[anthropic]"` and set CLASSIFIER=anthropic.

The request asks the API to constrain decoding to CLASSIFICATION_SCHEMA, so the
model cannot emit an out-of-set category or priority. The adapter still returns
plain text and the shared parser still runs on it: the schema is enforced by a
remote service we do not control, and the parser also does what a schema cannot
(plain-text reduction, rejecting echoed instructions).
"""

from __future__ import annotations

from app.models import CLASSIFICATION_SCHEMA
from app.prompt import Prompt

from .base import ModelPermanentError, ModelTransientError


class AnthropicClassifier:
    name = "anthropic"
    billable = True

    def __init__(self, model: str) -> None:
        import anthropic  # imported lazily so the fake path has no dependency on it

        self._anthropic = anthropic
        self._client = anthropic.AsyncAnthropic(max_retries=0)  # the worker owns retry policy
        self.model = model

    async def complete(self, prompt: Prompt) -> str:
        a = self._anthropic
        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=300,
                system=prompt.system,
                messages=[{"role": "user", "content": prompt.user}],
                output_config={"format": {"type": "json_schema", "schema": CLASSIFICATION_SCHEMA}},
            )
        except (a.RateLimitError, a.APIConnectionError) as e:
            raise ModelTransientError(str(e)) from e
        except a.APIStatusError as e:
            if e.status_code >= 500:
                raise ModelTransientError(f"{e.status_code}: {e.message}") from e
            raise ModelPermanentError(f"{e.status_code}: {e.message}") from e

        if response.stop_reason == "refusal":
            raise ModelPermanentError("model refused the request")
        # stop_reason == "max_tokens" is not an error here: the parser will reject truncated JSON.
        return "".join(block.text for block in response.content if block.type == "text")
