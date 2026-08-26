"""Asynchronous classification workers.

N asyncio tasks share one event loop. Each loops: claim a ticket → call the
model → validate → store. Concurrency is bounded by N; there is no separate
semaphore because a worker only ever has one ticket in flight.
"""

from __future__ import annotations

import asyncio
import logging

from app.classifier import Classifier, InvalidModelOutput, ModelPermanentError, parse_classification
from app.config import Settings
from app.prompt import PROMPT_VERSION, build_prompt
from app.store import TicketRow, TicketStore

log = logging.getLogger("worker")


class WorkerPool:
    def __init__(self, store: TicketStore, classifier: Classifier, settings: Settings) -> None:
        self.store = store
        self.classifier = classifier
        self.settings = settings
        self._tasks: list[asyncio.Task] = []
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()
        self.in_flight = 0
        self._budget_warned = False

    # ---- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        released = self.store.release_all_leases()
        if released:
            log.info("released %d lease(s) left over from a previous run", released)
        self._stop.clear()
        self._tasks = [asyncio.create_task(self._run(i), name=f"worker-{i}") for i in range(self.settings.worker_concurrency)]

    async def stop(self) -> None:
        """Let in-flight tickets finish (bounded), then cancel whatever is left."""
        self._stop.set()
        self._wake.set()
        done, pending = await asyncio.wait(self._tasks, timeout=self.settings.shutdown_grace)
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        # Anything cancelled mid-flight still holds a lease; drop it so a restart picks it up immediately.
        self.store.release_all_leases()

    def notify(self) -> None:
        """Called after ingest so idle workers do not wait for the next poll tick."""
        self._wake.set()

    # ---- loop ------------------------------------------------------------

    def budget_exhausted(self) -> bool:
        limit = self.settings.daily_model_call_limit
        return limit > 0 and self.store.model_calls_today() >= limit

    async def _run(self, index: int) -> None:
        while not self._stop.is_set():
            if self.budget_exhausted():
                # Tickets stay pending; nothing is lost, nothing is paid for. Resumes at UTC midnight or a higher limit.
                if not self._budget_warned:
                    log.warning("daily model call limit (%d) reached; pausing classification", self.settings.daily_model_call_limit)
                    self._budget_warned = True
                await asyncio.sleep(self.settings.poll_interval)
                continue
            self._budget_warned = False
            ticket = self.store.claim_next()
            if ticket is None:
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=self.settings.poll_interval)
                except asyncio.TimeoutError:
                    pass
                self._wake.clear()
                continue
            self.in_flight += 1
            try:
                await self._process(ticket)
            except Exception:  # never let one bad ticket kill a worker
                log.exception("worker %d: unexpected error on %s", index, ticket.id)
                self._retry_or_fail(ticket, "internal error")
            finally:
                self.in_flight -= 1

    async def _process(self, ticket: TicketRow) -> None:
        assert ticket.locked_at is not None
        prompt = build_prompt(ticket.subject, ticket.body)
        self.store.record_model_call()
        try:
            raw = await asyncio.wait_for(self.classifier.complete(prompt), timeout=self.settings.llm_timeout)
            result = parse_classification(raw)
        except ModelPermanentError as e:
            log.warning("%s: permanent model error, failing: %s", ticket.id, e)
            self.store.mark_failed(ticket.id, ticket.locked_at, f"permanent: {e}")
        except asyncio.TimeoutError:
            self._retry_or_fail(ticket, f"timeout after {self.settings.llm_timeout}s")
        except InvalidModelOutput as e:
            log.warning("%s: rejected model output: %s", ticket.id, e)
            self._retry_or_fail(ticket, f"invalid output: {e}")
        except Exception as e:  # ModelTransientError and anything else unexpected
            self._retry_or_fail(ticket, f"{type(e).__name__}: {e}")
        else:
            ok = self.store.mark_classified(ticket.id, ticket.locked_at, result, PROMPT_VERSION)
            if not ok:
                log.warning("%s: lease lost before result could be stored; discarding", ticket.id)

    def _retry_or_fail(self, ticket: TicketRow, error: str) -> None:
        assert ticket.locked_at is not None
        if ticket.attempts >= self.settings.max_attempts:
            log.warning("%s: giving up after %d attempt(s): %s", ticket.id, ticket.attempts, error)
            self.store.mark_failed(ticket.id, ticket.locked_at, error)
        else:
            delay = self.settings.retry_base_delay * (2 ** (ticket.attempts - 1))
            log.info("%s: attempt %d failed (%s); retrying in %.1fs", ticket.id, ticket.attempts, error, delay)
            self.store.mark_retry(ticket.id, ticket.locked_at, error, delay)
