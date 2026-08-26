"""HTTP API. `create_app()` wires everything; tests call it with their own settings/classifier."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.classifier import Classifier
from app.classifier.fake import FakeClassifier
from app.config import Settings
from app.cost import estimate as cost_estimate
from app.guard import Guard, ModelGuard, RegexGuard
from app.models import (
    Category,
    Classification,
    ErrorResponse,
    Priority,
    ReclassifyPreview,
    ReclassifyStaleRequest,
    RequeueResult,
    Status,
    TicketIn,
    TicketList,
    TicketOut,
    to_datetime,
)
from app.prompt import PROMPT_VERSION
from app.ratelimit import RateLimiter
from app.store import TicketRow, TicketStore
from app.worker import WorkerPool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@dataclass
class AppState:
    settings: Settings
    store: TicketStore
    classifier: Classifier
    guard: Guard
    pool: WorkerPool
    limiter: RateLimiter


def make_classifier(settings: Settings) -> Classifier:
    if settings.classifier == "anthropic":
        from app.classifier.anthropic_client import AnthropicClassifier

        return AnthropicClassifier(model=settings.llm_model)
    if settings.classifier == "fake":
        return FakeClassifier(settings.fake_failure_rate, settings.fake_latency, settings.fake_seed)
    raise ValueError(f"unknown CLASSIFIER={settings.classifier!r}")


def make_guard(settings: Settings) -> Guard:
    if settings.guard == "model":
        return ModelGuard(model=settings.guard_model)
    if settings.guard == "regex":
        return RegexGuard()
    raise ValueError(f"unknown GUARD={settings.guard!r}")


def to_out(row: TicketRow) -> TicketOut:
    classification = None
    if row.status == Status.classified:
        classification = Classification(category=row.category, priority=row.priority, summary=row.summary)
    return TicketOut(
        id=row.id,
        subject=row.subject,
        body=row.body,
        status=Status(row.status),
        classification=classification,
        attempts=row.attempts,
        last_error=row.last_error,
        injection_suspected=row.injection_suspected,
        prompt_version=row.prompt_version,
        created_at=to_datetime(row.created_at),
        updated_at=to_datetime(row.updated_at),
    )


def _error(status: int, code: str, message: str, details: list | None = None) -> JSONResponse:
    body = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body)


def create_app(
    settings: Settings | None = None, classifier: Classifier | None = None, guard: Guard | None = None
) -> FastAPI:
    settings = settings or Settings.from_env()
    classifier = classifier or make_classifier(settings)
    guard = guard or make_guard(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = TicketStore(settings.db_path, lease_seconds=settings.lease_seconds)
        pool = WorkerPool(store, classifier, settings, guard)
        app.state.ctx = AppState(settings, store, classifier, guard, pool, RateLimiter(settings.ingest_rate_per_minute))
        await pool.start()
        try:
            yield
        finally:
            await pool.stop()
            store.close()

    app = FastAPI(title="Ticket classifier", lifespan=lifespan, responses={422: {"model": ErrorResponse}})

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        return _error(422, "validation_error", "request failed validation", exc.errors())

    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException):
        code = {401: "unauthorized", 404: "not_found", 409: "conflict", 429: "rate_limited"}.get(exc.status_code, "error")
        return _error(exc.status_code, code, str(exc.detail))

    def ctx() -> AppState:
        return app.state.ctx

    def rate_limited(request: Request) -> None:
        """Per-client-IP limit on endpoints that can cause model calls."""
        key = request.client.host if request.client else "unknown"
        if not ctx().limiter.allow(key):
            raise HTTPException(429, f"rate limit: {ctx().settings.ingest_rate_per_minute} requests per minute")

    def admin_only(request: Request) -> None:
        """If ADMIN_TOKEN is set, require it as a bearer token. Open otherwise (local dev)."""
        token = ctx().settings.admin_token
        if token and request.headers.get("authorization") != f"Bearer {token}":
            raise HTTPException(401, "admin token required")

    # ---- routes ----------------------------------------------------------

    @app.post(
        "/tickets",
        response_model=TicketOut,
        status_code=201,
        responses={200: {"model": TicketOut}, 429: {"model": ErrorResponse}},
        dependencies=[Depends(rate_limited)],
    )
    def create_ticket(ticket: TicketIn):
        """201 when created, 200 with the existing record when the id was seen before."""
        c = ctx()
        created = c.store.insert_if_absent(ticket.id, ticket.subject, ticket.body)
        row = c.store.get(ticket.id)
        assert row is not None
        if created:
            c.pool.notify()
            return to_out(row)
        return JSONResponse(status_code=200, content=to_out(row).model_dump(mode="json"))

    # Literal paths must be registered before /tickets/{ticket_id} or they match as an id.
    def _stale_preview(limit: int | None = None) -> ReclassifyPreview:
        """Preview of a re-run. With `limit`, the estimate covers only the next batch."""
        c = ctx()
        affected = c.store.stale_summary(PROMPT_VERSION)
        chars = affected.pop("ticket_chars")
        n = affected["total"] if limit is None else min(limit, affected["total"])
        if limit is not None and affected["total"]:
            chars = chars * n // affected["total"]  # proportional share for the batch
            affected["this_call"] = n
        est = cost_estimate(n, chars, c.settings, c.classifier.name, c.classifier.billable)
        return ReclassifyPreview(dry_run=True, affected=affected, estimate=est, current_prompt_version=PROMPT_VERSION)

    @app.get("/tickets/reclassify-stale", response_model=ReclassifyPreview, dependencies=[Depends(admin_only)])
    def preview_reclassify_stale():
        """What a re-run would touch and roughly what it would cost. Changes nothing."""
        return _stale_preview()

    @app.post(
        "/tickets/reclassify-stale",
        response_model=RequeueResult,
        status_code=202,
        responses={200: {"model": ReclassifyPreview}, 409: {"model": ErrorResponse}},
        dependencies=[Depends(admin_only), Depends(rate_limited)],
    )
    def reclassify_stale(body: ReclassifyStaleRequest | None = None):
        """Requeue failed tickets and tickets classified under an older prompt, `limit` at a time.

        Without `{"confirm": true}` this is a dry run (200 + preview). With `max_usd`, the
        call is refused (409) if the estimate for this batch is above it. The response
        reports `remaining`; call again to continue — already-requeued tickets are skipped.
        """
        body = body or ReclassifyStaleRequest()
        preview = _stale_preview(limit=body.limit)
        if not body.confirm:
            return JSONResponse(status_code=200, content=preview.model_dump(mode="json"))
        if body.max_usd is not None and preview.estimate["usd"] > body.max_usd:
            raise HTTPException(
                409, f"estimated cost ${preview.estimate['usd']} exceeds max_usd ${body.max_usd}; nothing requeued"
            )
        c = ctx()
        n = c.store.requeue_stale(PROMPT_VERSION, limit=body.limit)
        if n:
            c.pool.notify()
        remaining = c.store.stale_summary(PROMPT_VERSION)["total"]
        return RequeueResult(requeued=n, remaining=remaining, estimate=preview.estimate)

    @app.get("/tickets/{ticket_id}", response_model=TicketOut, responses={404: {"model": ErrorResponse}})
    def get_ticket(ticket_id: str):
        row = ctx().store.get(ticket_id)
        if row is None:
            raise HTTPException(404, f"ticket {ticket_id!r} not found")
        return to_out(row)

    @app.get("/tickets", response_model=TicketList)
    def list_tickets(
        category: Category | None = None,
        priority: Priority | None = None,
        status: Status | None = None,
        limit: int = Query(20, ge=1, le=100),
        offset: int = Query(0, ge=0),
    ):
        rows, total = ctx().store.list(
            category=category, priority=priority, status=status, limit=limit, offset=offset
        )
        return TicketList(items=[to_out(r) for r in rows], total=total, limit=limit, offset=offset)

    @app.post(
        "/tickets/{ticket_id}/reclassify",
        response_model=TicketOut,
        status_code=202,
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
        dependencies=[Depends(admin_only), Depends(rate_limited)],
    )
    def reclassify_ticket(ticket_id: str):
        """Send a classified or failed ticket through the classifier again."""
        c = ctx()
        row = c.store.get(ticket_id)
        if row is None:
            raise HTTPException(404, f"ticket {ticket_id!r} not found")
        if not c.store.requeue(ticket_id):
            raise HTTPException(409, f"ticket {ticket_id!r} is already pending")
        c.pool.notify()
        return to_out(c.store.get(ticket_id))

    @app.get("/health")
    def health():
        """Liveness only. Operational detail lives in /stats."""
        return {"status": "ok"}

    @app.get("/stats", dependencies=[Depends(admin_only)])
    def stats():
        c = ctx()
        return {
            "classifier": c.classifier.name,
            "guard": c.guard.name,
            "prompt_version": PROMPT_VERSION,
            "workers": c.settings.worker_concurrency,
            "in_flight": c.pool.in_flight,
            "pending": c.store.count("pending"),
            "classified": c.store.count("classified"),
            "failed": c.store.count("failed"),
            "model_calls_today": c.store.model_calls_today(),
            "daily_model_call_limit": c.settings.daily_model_call_limit or None,
            "paused_for_budget": c.pool.budget_exhausted(),
        }

    return app


app = create_app()
