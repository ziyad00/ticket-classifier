"""Real provider. Optional: `pip install -e ".[anthropic]"` and set CLASSIFIER=anthropic.

Deliberately does not use structured outputs / tool schemas: the exercise is about
treating the model as a source of text, so this adapter returns text and the
shared parser decides whether it is acceptable. In production I would turn on
`output_config.format` as an extra layer and keep the parser as the last line.
"""

from __future__ import annotations

from app.prompt import Prompt

from .base import ModelPermanentError, ModelTransientError


class AnthropicClassifier:
    name = "anthropic"

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
