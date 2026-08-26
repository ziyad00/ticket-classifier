"""HTTP API. `create_app()` wires everything; tests call it with their own settings/classifier."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.classifier import Classifier
from app.classifier.fake import FakeClassifier
from app.config import Settings
from app.models import (
    Category,
    Classification,
    ErrorResponse,
    Priority,
    RequeueResult,
    Status,
    TicketIn,
    TicketList,
    TicketOut,
    to_datetime,
)
from app.prompt import PROMPT_VERSION
from app.safety import looks_like_injection
from app.store import TicketRow, TicketStore
from app.worker import WorkerPool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@dataclass
class AppState:
    settings: Settings
    store: TicketStore
    classifier: Classifier
    pool: WorkerPool


def make_classifier(settings: Settings) -> Classifier:
    if settings.classifier == "anthropic":
        from app.classifier.anthropic_client import AnthropicClassifier

        return AnthropicClassifier(model=settings.llm_model)
    if settings.classifier == "fake":
        return FakeClassifier(settings.fake_failure_rate, settings.fake_latency, settings.fake_seed)
    raise ValueError(f"unknown CLASSIFIER={settings.classifier!r}")


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


def create_app(settings: Settings | None = None, classifier: Classifier | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    classifier = classifier or make_classifier(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = TicketStore(settings.db_path, lease_seconds=settings.lease_seconds)
        pool = WorkerPool(store, classifier, settings)
        app.state.ctx = AppState(settings, store, classifier, pool)
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
        code = {404: "not_found", 409: "conflict"}.get(exc.status_code, "error")
        return _error(exc.status_code, code, str(exc.detail))

    def ctx() -> AppState:
        return app.state.ctx

    # ---- routes ----------------------------------------------------------

    @app.post("/tickets", response_model=TicketOut, status_code=201, responses={200: {"model": TicketOut}})
    def create_ticket(ticket: TicketIn, request: Request):
        """201 when created, 200 with the existing record when the id was seen before."""
        c = ctx()
        created = c.store.insert_if_absent(
            ticket.id, ticket.subject, ticket.body, looks_like_injection(ticket.subject, ticket.body)
        )
        row = c.store.get(ticket.id)
        assert row is not None
        if created:
            c.pool.notify()
            return to_out(row)
        return JSONResponse(status_code=200, content=to_out(row).model_dump(mode="json"))

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

    @app.post("/tickets/reclassify-stale", response_model=RequeueResult, status_code=202)
    def reclassify_stale():
        """Requeue every failed ticket and every ticket classified under an older prompt version."""
        c = ctx()
        n = c.store.requeue_stale(PROMPT_VERSION)
        if n:
            c.pool.notify()
        return RequeueResult(requeued=n)

    @app.get("/health")
    def health():
        c = ctx()
        return {
            "status": "ok",
            "classifier": c.classifier.name,
            "prompt_version": PROMPT_VERSION,
            "workers": c.settings.worker_concurrency,
            "in_flight": c.pool.in_flight,
            "pending": c.store.count("pending"),
            "failed": c.store.count("failed"),
        }

    return app


app = create_app()
