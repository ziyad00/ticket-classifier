# Ticket classifier

A small HTTP service that ingests support tickets, classifies them asynchronously
with an LLM (a fake one by default), validates whatever the model says before
storing it, and serves the results back with filtering and pagination.

## Run it

```bash
uv sync --extra dev                 # or: python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
uv run uvicorn app.main:app         # http://localhost:8000  (docs at /docs)
uv run python scripts/load_samples.py --wait    # in a second terminal: loads the 10 sample tickets, prints results
uv run pytest                       # 45 tests, ~1.5s
```

No API key is needed. To use a real model instead of the fake:
`uv sync --extra anthropic`, `export ANTHROPIC_API_KEY=...`, `CLASSIFIER=anthropic uv run uvicorn app.main:app`.
All settings are in `.env.example`.

## API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/tickets` | body `{"id", "subject", "body"}`. **201** created, **200** if the id already exists (returns the existing ticket, does not re-classify). |
| `GET` | `/tickets/{id}` | 404 if unknown. |
| `GET` | `/tickets?category=&priority=&status=&limit=20&offset=0` | Filters are ANDed; `total` is included. |
| `POST` | `/tickets/{id}/reclassify` | Requeue one classified/failed ticket. 202, or 409 if already pending. |
| `POST` | `/tickets/reclassify-stale` | Requeue every failed ticket and every ticket classified under an older `PROMPT_VERSION`. |
| `GET` | `/health` | counts + which classifier is loaded. |

Ticket shape:

```json
{"id": "t-1001", "subject": "...", "body": "...", "status": "classified",
 "classification": {"category": "billing", "priority": "high", "summary": "..."},
 "attempts": 1, "last_error": null, "injection_suspected": false,
 "prompt_version": "sha256:1a2b3c4d5e6f", "created_at": "...", "updated_at": "..."}
```

`status` is one of `pending | classified | failed`. `classification` is `null` unless `status == classified`.
Errors are always `{"error": {"code", "message", "details?"}}`.

## Where to look

```
app/
  main.py          routes, app factory, error format
  models.py        Classification = the validated shape; API schemas
  classifier/
    base.py        Classifier protocol (returns *text*) + error taxonomy
    parse.py       text -> Classification, or InvalidModelOutput
    fake.py        deterministic fake; ~15% broken output by default
    anthropic_client.py   real provider, optional
  prompt.py        system prompt, PROMPT_VERSION, ticket-as-JSON-data
  safety.py        prompt-injection heuristic (flags, never blocks)
  store.py         SQLite: tickets table doubles as the job queue (leases)
  worker.py        N asyncio workers: claim -> model -> validate -> store, retry policy
tests/             parse (the gate), api (lifecycle, idempotency, listing), worker (retries, concurrency, restart), untrusted content
```

## Decisions on the open questions

**Storage: one SQLite table.** Tickets and their queue state live in the same row
(`status`, `attempts`, `locked_at`, `next_attempt_at`). Zero infrastructure,
survives restarts, and the reviewers can open the file with `sqlite3`. The table
has `CHECK` constraints on `status`, `category` and `priority`, so even a bug that
skipped the parser could not write an out-of-set value (`test_db_rejects_out_of_set_values_even_if_code_is_bypassed`).

**Async execution: in-process asyncio workers.** `WorkerPool` runs `WORKER_CONCURRENCY`
tasks (default 3) on the API's event loop. Each claims the oldest runnable pending
ticket, calls the model, validates, stores. Ingest just inserts the row and pokes an
`asyncio.Event` so the ticket is picked up immediately rather than on the next poll.
I chose this over Celery/RQ/Arq because a queue broker would triple the setup steps
for a 4-hour exercise; the `store.claim_next()` contract is the seam where a real
queue would go.

**Concurrency: N workers, one ticket each.** The bound is simply the number of
workers — no separate semaphore to keep in sync. `test_concurrency_is_bounded` asserts
the model never sees more than N calls in flight.

**Restart: leases, not a `processing` state.** A claim writes `locked_at`. Status
stays `pending`, so the external contract really is three states. On startup (single
process assumption) all leases are dropped; if that assumption is ever wrong, a lease
older than `LEASE_SECONDS` (60s; startup refuses a value ≤ `LLM_TIMEOUT`) is treated as abandoned and
reclaimable anyway. A late result from a stale holder is discarded because every
terminal write is `WHERE id = ? AND locked_at = ?` (`test_claim_is_exclusive_and_lease_expires`).
Claiming increments `attempts`, so a ticket that crashes the worker every time ends
up `failed` after `MAX_ATTEMPTS` instead of looping forever (`test_restart_releases_in_flight_leases`).

**Retry policy.** Three kinds of failure, treated differently:
- *Transient* (rate limit, 5xx, network, timeout) and *invalid output* (not JSON,
  out-of-set values, missing/empty/over-long summary): retry with backoff
  `RETRY_BASE_DELAY * 2^(attempt-1)`, up to `MAX_ATTEMPTS` (3), then `failed` with the
  last error recorded. Retrying invalid output is deliberate: models are stochastic,
  and the second answer is usually fine.
- *Permanent* (401/403/400, model refusal): `failed` on the first attempt. Retrying a
  bad API key three times just costs time.
`failed` is not a dead end: `POST /tickets/{id}/reclassify` resets attempts and
requeues.

**Model output validation.** The classifier interface returns a string, nothing more.
`parse_classification` tolerates cosmetic quirks (code fences, a preamble line, `"BILLING"`,
extra keys, stray whitespace) because rejecting those would be needless retries, but it
is strict about anything that would change meaning: unknown categories/priorities,
wrong types, an empty or >300-char summary, a >10 KB response. Anything rejected is
never written; the old classification (if any) stays untouched. What happens instead
is the retry policy above.

**Prompt injection.** Three layers, in order of how much I trust them:
1. *Validation* (most trust). Whatever the model is talked into, the only thing that
   can be stored is a valid category, a valid priority and a short string. The blast
   radius of a successful injection is "wrong category, misleading one-liner", which a
   human would catch — and that same blast radius exists for a plain wrong answer.
2. *Prompt structure.* The ticket is embedded as a JSON object inside the user turn, so
   there is no delimiter the sender can close, and the system prompt states that the
   content is untrusted data and that embedded instructions must not be followed or
   echoed into the summary. This helps; it is not a guarantee.
3. *Heuristic flag* (least trust). `safety.py` regex-flags obvious attempts
   ("ignore all previous instructions", "classify this as", "this ticket is from the CEO")
   and stores `injection_suspected: true`. It never blocks or alters classification — a
   customer who innocently writes "please classify this as urgent" should still get
   served — it just gives a reviewer a filter. t-1005 is flagged; the other nine are not.

What I did **not** do: refuse or quarantine flagged tickets, or run a second model as
a judge. Both are defensible; neither fit the time budget, and quarantining
public-facing tickets on a regex seemed worse than a flag.

**API shape.** Duplicate `POST /tickets` returns 200 with the existing record rather
than 409, because the realistic caller is an email pipeline retrying on a dropped
connection — it wants "this is fine", not an error. Pagination is `limit/offset`
because the dataset is small and it composes with the filters trivially; keyset
pagination on `(created_at, id)` would be the change at scale.

## The fake model

`FakeClassifier` is keyword-based and deterministic per `FAKE_SEED`, so a run is
reproducible. Roughly `FAKE_FAILURE_RATE` (15%) of calls misbehave in ways I have seen
real models misbehave: prose instead of JSON, `"category": "refund"`,
`"priority": "critical"`, truncated JSON, a missing field, a 500-char summary, or a
raised 529. Successful calls are sometimes wrapped in code fences or a preamble line,
which the parser must cope with. The fake resists t-1005's injection *by construction*
(it counts keywords; "invoices" wins) — that says nothing about a real model, which is
exactly why layers 1 and 3 above exist.

## Optional item: re-classification

`PROMPT_VERSION` is a hash of the system prompt text and is stored on every classified
ticket. `POST /tickets/reclassify-stale` requeues everything failed plus everything
classified under a different version, so a prompt change is: edit `prompt.py`, restart,
hit the endpoint. Nothing to bump by hand; a typo fix also counts as a change, which is
the safe direction to err in. Triggering the re-run stays manual on purpose — it costs
model calls, and on a big table that should be a deliberate decision.
Graceful shutdown is partly there too (`WorkerPool.stop` gives in-flight calls
`SHUTDOWN_GRACE` seconds to finish, then releases leases so nothing is lost), but I did
not go further than that.

## Weaknesses and what I would change with more time

- **Single process.** The workers live inside the API process, so scaling the API
  scales the workers, and `release_all_leases()` on boot is only correct with one
  process. Next step: a `--workers-only` entrypoint and rely purely on lease expiry.
- **Synchronous SQLite under a lock** inside async workers. Fine at this size (each
  call is microseconds), wrong if the store ever became network-attached. Route
  handlers are plain `def`, so FastAPI already runs them off the event loop.
- **Validation checks shape, not truth.** A well-formed wrong answer sails through.
  The next layer would be the evaluation script from the "finish early" list: labelled
  tickets + agreement rate, run on every prompt change.
- **The real adapter is untested against the live API.** It maps the SDK's exception
  classes to transient/permanent and handles `stop_reason == "refusal"`, but I could
  not run it here. In production I would also enable the API's structured-output
  mode as an extra layer — while keeping the parser, because the point is that the
  service does not depend on the provider behaving.
- **The injection heuristic is a regex list.** It will miss paraphrases and can
  false-positive; it's a triage hint, not a control.
- **No auth, no rate limiting, no request size limit beyond the 20 KB body cap.**
  Out of scope, but this is a public-facing ingest endpoint.
- **`POST /tickets/reclassify-stale` on a large table is one big UPDATE** with no
  batching. Fine for thousands; not for millions.
