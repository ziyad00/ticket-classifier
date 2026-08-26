"""SQLite-backed ticket store. Also the job queue.

One table holds both the ticket and its position in the classification
lifecycle. A worker "claims" a pending ticket by writing a lease (`locked_at`);
a lease older than `lease_seconds` is treated as abandoned and may be claimed
again. Status never leaves {pending, classified, failed}.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass

from app.models import Classification

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id                  TEXT PRIMARY KEY,
    subject             TEXT NOT NULL,
    body                TEXT NOT NULL,
    status              TEXT NOT NULL CHECK (status IN ('pending', 'classified', 'failed')),
    category            TEXT CHECK (category IN ('billing', 'technical', 'account', 'other')),
    priority            TEXT CHECK (priority IN ('low', 'medium', 'high')),
    summary             TEXT,
    prompt_version      TEXT,
    attempts            INTEGER NOT NULL DEFAULT 0,
    last_error          TEXT,
    injection_suspected INTEGER NOT NULL DEFAULT 0,
    locked_at           REAL,
    next_attempt_at     REAL NOT NULL DEFAULT 0,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tickets_queue ON tickets (status, next_attempt_at, created_at);
CREATE INDEX IF NOT EXISTS idx_tickets_filter ON tickets (category, priority, created_at);
CREATE TABLE IF NOT EXISTS model_calls (
    day   TEXT PRIMARY KEY,   -- UTC date, YYYY-MM-DD
    calls INTEGER NOT NULL
);
"""


@dataclass(frozen=True)
class TicketRow:
    id: str
    subject: str
    body: str
    status: str
    category: str | None
    priority: str | None
    summary: str | None
    prompt_version: str | None
    attempts: int
    last_error: str | None
    injection_suspected: bool
    locked_at: float | None
    next_attempt_at: float
    created_at: float
    updated_at: float

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> "TicketRow":
        d = dict(r)
        d["injection_suspected"] = bool(d["injection_suspected"])
        return cls(**d)


class TicketStore:
    def __init__(self, path: str = ":memory:", lease_seconds: float = 60.0) -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()
        self.lease_seconds = lease_seconds

    def close(self) -> None:
        self._conn.close()

    # ---- ingest / read ---------------------------------------------------

    def insert_if_absent(self, id: str, subject: str, body: str) -> bool:
        """Returns True if the ticket was created, False if the id already existed."""
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO tickets (id, subject, body, status, created_at, updated_at)"
                " VALUES (?, ?, ?, 'pending', ?, ?)",
                (id, subject, body, now, now),
            )
            return cur.rowcount == 1

    def set_injection_suspected(self, id: str, flag: bool) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE tickets SET injection_suspected = ?, updated_at = ? WHERE id = ?", (int(flag), time.time(), id)
            )

    def get(self, id: str) -> TicketRow | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM tickets WHERE id = ?", (id,)).fetchone()
        return TicketRow.from_row(r) if r else None

    def list(
        self,
        *,
        category: str | None = None,
        priority: str | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[TicketRow], int]:
        clauses, params = [], []
        for col, val in (("category", category), ("priority", priority), ("status", status)):
            if val is not None:
                clauses.append(f"{col} = ?")
                params.append(val)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            total = self._conn.execute(f"SELECT COUNT(*) FROM tickets {where}", params).fetchone()[0]
            rows = self._conn.execute(
                f"SELECT * FROM tickets {where} ORDER BY created_at, id LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        return [TicketRow.from_row(r) for r in rows], total

    def count(self, status: str) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM tickets WHERE status = ?", (status,)).fetchone()[0]

    # ---- queue -----------------------------------------------------------

    def claim_next(self) -> TicketRow | None:
        """Lease the oldest runnable pending ticket. Increments `attempts`."""
        now = time.time()
        stale_before = now - self.lease_seconds
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                r = self._conn.execute(
                    "SELECT id FROM tickets WHERE status = 'pending' AND next_attempt_at <= ?"
                    " AND (locked_at IS NULL OR locked_at < ?) ORDER BY created_at, id LIMIT 1",
                    (now, stale_before),
                ).fetchone()
                if r is None:
                    return None
                self._conn.execute(
                    "UPDATE tickets SET locked_at = ?, attempts = attempts + 1, updated_at = ? WHERE id = ?",
                    (now, now, r["id"]),
                )
                row = self._conn.execute("SELECT * FROM tickets WHERE id = ?", (r["id"],)).fetchone()
            finally:
                self._conn.execute("COMMIT")
        return TicketRow.from_row(row)

    def _finish(self, sql: str, params: tuple, id: str, lease: float) -> bool:
        """Apply a terminal/retry update only if we still hold the lease."""
        with self._lock:
            cur = self._conn.execute(sql + " WHERE id = ? AND locked_at = ?", (*params, id, lease))
            return cur.rowcount == 1

    def mark_classified(self, id: str, lease: float, result: Classification, prompt_version: str) -> bool:
        return self._finish(
            "UPDATE tickets SET status = 'classified', category = ?, priority = ?, summary = ?,"
            " prompt_version = ?, last_error = NULL, locked_at = NULL, updated_at = ?",
            (result.category.value, result.priority.value, result.summary, prompt_version, time.time()),
            id,
            lease,
        )

    def mark_retry(self, id: str, lease: float, error: str, delay: float) -> bool:
        now = time.time()
        return self._finish(
            "UPDATE tickets SET status = 'pending', last_error = ?, locked_at = NULL, next_attempt_at = ?, updated_at = ?",
            (error, now + delay, now),
            id,
            lease,
        )

    def mark_failed(self, id: str, lease: float, error: str) -> bool:
        return self._finish(
            "UPDATE tickets SET status = 'failed', last_error = ?, locked_at = NULL, updated_at = ?",
            (error, time.time()),
            id,
            lease,
        )

    def release_all_leases(self) -> int:
        """On startup/shutdown of a single-process deployment, no lease can be live."""
        with self._lock:
            return self._conn.execute("UPDATE tickets SET locked_at = NULL WHERE locked_at IS NOT NULL").rowcount

    # ---- spend accounting ------------------------------------------------

    @staticmethod
    def _today() -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def record_model_call(self) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO model_calls (day, calls) VALUES (?, 1) ON CONFLICT(day) DO UPDATE SET calls = calls + 1",
                (self._today(),),
            )

    def model_calls_today(self) -> int:
        with self._lock:
            r = self._conn.execute("SELECT calls FROM model_calls WHERE day = ?", (self._today(),)).fetchone()
        return r[0] if r else 0

    # ---- re-classification ----------------------------------------------

    def requeue(self, id: str) -> bool:
        """Put a classified/failed ticket back to pending. Leaves the old result in place until overwritten."""
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE tickets SET status = 'pending', attempts = 0, next_attempt_at = 0, locked_at = NULL, updated_at = ?"
                " WHERE id = ? AND status IN ('classified', 'failed')",
                (now, id),
            )
            return cur.rowcount == 1

    def stale_summary(self, current_prompt_version: str) -> dict:
        """What requeue_stale would touch, without touching it."""
        with self._lock:
            total, chars = self._conn.execute(
                f"SELECT COUNT(*), COALESCE(SUM(LENGTH(subject) + LENGTH(body)), 0) FROM tickets WHERE {self._STALE}",
                (current_prompt_version,),
            ).fetchone()
            failed = self._conn.execute("SELECT COUNT(*) FROM tickets WHERE status = 'failed'").fetchone()[0]
            by_version = self._conn.execute(
                "SELECT prompt_version, COUNT(*) FROM tickets WHERE status = 'classified' AND prompt_version IS NOT ?"
                " GROUP BY prompt_version",
                (current_prompt_version,),
            ).fetchall()
        return {
            "total": total,
            "failed": failed,
            "stale": total - failed,
            "by_prompt_version": {(r[0] or "none"): r[1] for r in by_version},
            "ticket_chars": chars,
        }

    _STALE = "status = 'failed' OR (status = 'classified' AND prompt_version IS NOT ?)"

    def requeue_stale(self, current_prompt_version: str, limit: int, batch_size: int = 500) -> int:
        """Requeue up to `limit` stale tickets, oldest first, in short transactions of `batch_size`.

        Resumable by construction: a requeued ticket is `pending` and no longer matches the
        stale predicate, so calling again continues where the last call stopped.
        """
        requeued = 0
        while requeued < limit:
            n = min(batch_size, limit - requeued)
            now = time.time()
            with self._lock:
                cur = self._conn.execute(
                    "UPDATE tickets SET status = 'pending', attempts = 0, next_attempt_at = 0, locked_at = NULL, updated_at = ?"
                    f" WHERE id IN (SELECT id FROM tickets WHERE {self._STALE} ORDER BY created_at, id LIMIT ?)",
                    (now, current_prompt_version, n),
                )
            if cur.rowcount == 0:
                break
            requeued += cur.rowcount
        return requeued
