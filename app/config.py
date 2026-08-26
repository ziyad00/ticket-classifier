"""Runtime configuration. Every value comes from the environment and has a default."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    db_path: str = "tickets.db"
    classifier: str = "fake"  # "fake" | "anthropic"
    llm_model: str = "claude-opus-5"
    worker_concurrency: int = 3
    max_attempts: int = 3
    retry_base_delay: float = 1.0  # seconds; doubles per attempt
    llm_timeout: float = 30.0  # seconds per model call
    lease_seconds: float = 60.0  # a claim older than this is considered abandoned
    poll_interval: float = 1.0  # how often idle workers re-check the queue
    shutdown_grace: float = 10.0  # seconds to let in-flight work finish on stop
    fake_failure_rate: float = 0.15
    fake_latency: float = 0.05
    fake_seed: int = 42

    def __post_init__(self) -> None:
        if self.lease_seconds <= self.llm_timeout:
            raise ValueError(
                f"LEASE_SECONDS ({self.lease_seconds}) must be greater than LLM_TIMEOUT ({self.llm_timeout}); "
                "otherwise a slow but live worker could lose its ticket mid-call"
            )
        if self.worker_concurrency < 1 or self.max_attempts < 1:
            raise ValueError("WORKER_CONCURRENCY and MAX_ATTEMPTS must be >= 1")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            db_path=_env("DB_PATH", cls.db_path),
            classifier=_env("CLASSIFIER", cls.classifier),
            llm_model=_env("LLM_MODEL", cls.llm_model),
            worker_concurrency=int(_env("WORKER_CONCURRENCY", str(cls.worker_concurrency))),
            max_attempts=int(_env("MAX_ATTEMPTS", str(cls.max_attempts))),
            retry_base_delay=float(_env("RETRY_BASE_DELAY", str(cls.retry_base_delay))),
            llm_timeout=float(_env("LLM_TIMEOUT", str(cls.llm_timeout))),
            lease_seconds=float(_env("LEASE_SECONDS", str(cls.lease_seconds))),
            fake_failure_rate=float(_env("FAKE_FAILURE_RATE", str(cls.fake_failure_rate))),
            fake_latency=float(_env("FAKE_LATENCY", str(cls.fake_latency))),
            fake_seed=int(_env("FAKE_SEED", str(cls.fake_seed))),
        )
