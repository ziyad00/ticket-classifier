# Ticket classifier

A small HTTP service that ingests support tickets, classifies them asynchronously
with an LLM (a fake one by default), validates whatever the model says before
storing it, and serves the results back with filtering and pagination.

## Run it

```bash
uv sync --extra dev                 # or: python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
uv run uvicorn app.main:app         # http://localhost:8000  (docs at /docs)
uv run python scripts/load_samples.py --wait    # in a second terminal: loads the 10 sample tickets, prints results
uv run pytest                       # 53 tests, ~1.5s
uv run python scripts/evaluate.py   # agreement with data/labelled_tickets.json
```

No API key is needed. To use a real model instead of the fake:
`uv sync --extra anthropic`, `export ANTHROPIC_API_KEY=...`, `CLASSIFIER=anthropic uv run uvicorn app.main:app`.
All settings are in `.env.example`.

## API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/tickets` | body `{"id", "subject", "body"}`. **201** created, **200** if the id already exists (returns the existing ticket, does not re-classify). Rate-limited per client IP. |
| `GET` | `/tickets/{id}` | 404 if unknown. |
| `GET` | `/tickets?category=&priority=&status=&limit=20&offset=0` | Filters are ANDed; `total` is included. |
| `POST` | `/tickets/{id}/reclassify` | Requeue one classified/failed ticket. 202, or 409 if already pending. Admin. |
| `GET` | `/tickets/reclassify-stale` | Preview: how many tickets a re-run would touch (failed / stale, by prompt version) and a cost estimate. Changes nothing. |
| `POST` | `/tickets/reclassify-stale` | Body `{"confirm": true, "limit": 1000, "max_usd": 2.0}`. Without `confirm` it's a dry run (200 + preview). Requeues at most `limit` (oldest first) and reports `remaining`; call again to continue. `max_usd` refuses (409) if the estimate for this call is higher. Admin. |
| `GET` | `/health` | `{"status": "ok"}` — liveness only. |
| `GET` | `/stats` | queue counts, model calls today, whether workers are paused for budget. Admin. |

"Admin" endpoints are open when `ADMIN_TOKEN` is unset (local dev) and require
`Authorization: Bearer <token>` when it is. Errors are always
`{"error": {"code", "message", "details?"}}`; codes include `validation_error`,
`not_found`, `conflict`, `rate_limited`, `unauthorized`.

Ticket shape:

```json
{"id": "t-1001", "subject": "...", "body": "...", "status": "classified",
 "classification": {"category": "billing", "priority": "high", "summary": "..."},
 "attempts": 1, "last_error": null, "injection_suspected": false,
 "prompt_version": "sha256:1a2b3c4d5e6f", "created_at": "...", "updated_at": "..."}
```

`status` is one of `pending | classified | failed`. `classification` is `null` unless `status == classified`.
`summary` is model output: stored as plain text (angle brackets and control characters removed),
but consumers should still escape it when rendering.

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
  evaluate.py      agreement against the labelled set (used by scripts/evaluate.py)
  guard.py         input guardrail: RegexGuard (default) / ModelGuard (small model, yes/no)
  safety.py        the regex the RegexGuard and the fallback use
  ratelimit.py     per-IP sliding window for the POST endpoints
  store.py         SQLite: tickets table doubles as the job queue (leases)
  worker.py        N asyncio workers: claim -> model -> validate -> store, retry policy
data/              sample_tickets.json (the appendix), labelled_tickets.json (expected category/priority)
scripts/           load_samples.py (POST the samples, --wait for results), evaluate.py (agreement report)
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

**Spend control.** Every new ticket is a paid model call, so there are two brakes:
`INGEST_RATE_PER_MINUTE` limits how fast one client can create tickets (or trigger
re-runs), and `DAILY_MODEL_CALL_LIMIT` is a hard ceiling counted in the database per UTC
day — when it is reached the workers pause and tickets simply wait as `pending`;
`/stats` shows `paused_for_budget`. The rate limiter is in-memory and per process; a
multi-replica deployment would want it at the proxy instead.

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
wrong types, an empty or >300-char summary, a >10 KB response, pathological JSON
(deep nesting, giant numbers), and a summary that reads like an instruction — the model
echoing injected text. The summary is also reduced to plain text before storage: NFKC
normalised, control and invisible characters (RTL override, zero-width) removed, angle
brackets stripped, URLs replaced with `[link]`, since it will be shown in someone else's
UI. Anything rejected is never written; the old classification (if any) stays untouched.
What happens instead is the retry policy above.

**Prompt injection.** Three layers, in order of how much I trust them:
1. *Validation* (most trust). Whatever the model is talked into, the only thing that
   can be stored is a valid category, a valid priority and a short string. The blast
   radius of a successful injection is "wrong category, misleading one-liner", which a
   human would catch — and that same blast radius exists for a plain wrong answer.
2. *Prompt structure.* The ticket is embedded as a JSON object inside the user turn, so
   there is no delimiter the sender can close, and the system prompt states that the
   content is untrusted data and that embedded instructions must not be followed or
   echoed into the summary, with one worked example of an injected ticket and its
   correct output. With the real provider the request also constrains decoding to
   `CLASSIFICATION_SCHEMA`, so an out-of-set value cannot be generated at all. This
   helps; it is not a guarantee — `technical/high` is a legal value whatever the
   reason the model chose it.
3. *Input guard* (least trust). Before classifying, the worker asks a `Guard` whether the
   ticket contains instructions aimed at an AI and stores the answer as
   `injection_suspected`. The default `RegexGuard` matches obvious phrasings ("ignore all
   previous instructions", "classify this as", "this ticket is from the CEO"). `GUARD=model`
   swaps in `ModelGuard`: one cheap yes/no call to a small model (`GUARD_MODEL`,
   Haiku-class) with its own prompt and a boolean schema, which catches paraphrases and
   other languages; if that call fails the regex is used instead. Two models with two
   different prompts are harder to fool together than one. The flag never blocks or
   alters classification — a customer who innocently writes "please classify this as
   urgent" should still get served — it just gives a reviewer a filter. t-1005 is
   flagged; the other nine are not.

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
the safe direction to err in.

Triggering the re-run stays manual on purpose — it costs model calls, and on a big table
that should be a decision made with a number in front of you. So the endpoint is a dry
run unless you send `{"confirm": true}`, and the response (or `GET` on the same path)
tells you what you'd be paying for:

```json
{"dry_run": true,
 "affected": {"total": 10, "failed": 1, "stale": 9, "by_prompt_version": {"sha256:1a2b3c4d5e6f": 9}},
 "estimate": {"classifier": "anthropic", "model": "claude-opus-5", "model_calls": 10,
              "model_calls_worst_case": 30, "input_tokens": 3900, "output_tokens": 600,
              "usd": 0.0345, "usd_worst_case": 0.1035,
              "basis": "~4 chars/token, 60 output tokens/ticket, $5.0/M in, $25.0/M out; ..."}}
```

`max_usd` is a spend cap: `{"confirm": true, "max_usd": 0.05}` is refused with a 409 if
the estimate is above it. The estimate is deliberately crude (chars ÷ 4, list prices from
`PRICE_INPUT_PER_MTOK` / `PRICE_OUTPUT_PER_MTOK`); the worst case multiplies by
`MAX_ATTEMPTS`. With the fake classifier the cost is reported as $0.

A re-run is done in slices. `limit` (default 1000) is how many tickets one call requeues,
oldest first, and the store writes them in short transactions of 500 so a large table is
never locked for long. The response carries `remaining`; calling again continues from
where the last call stopped, because a requeued ticket is `pending` and no longer matches
the stale predicate. That makes the operation resumable after an interruption, cappable
per slice with `max_usd`, and pausable between slices — the workers drain each slice
before you decide on the next one.
Graceful shutdown is partly there too (`WorkerPool.stop` gives in-flight calls
`SHUTDOWN_GRACE` seconds to finish, then releases leases so nothing is lost), but I did
not go further than that.

## Second optional item: evaluation

`data/labelled_tickets.json` gives an expected category and priority for each of the
ten sample tickets (with a one-line reason each — t-1005 is labelled `billing/low`
because the real question is where to download invoices). `scripts/evaluate.py` runs
the classifier directly — no HTTP, no database — and prints per-ticket agreement plus
totals:

```
category agreement : 100.0%
priority agreement :  70.0%
both agree         :  70.0%
unclassified       : 0
rejected outputs   : 4   (model replies the parser refused, incl. retries)
```

`--min-agreement 0.8` makes it exit non-zero, so it can gate a prompt change in CI.
`CLASSIFIER=anthropic` runs it against the real model. The 70% above is the *fake*
scoring against my labels; it is a regression guard for the heuristics, not a claim
about a real model. The misses are instructive: the fake rates t-1005 `high` because the
subject says "URGENT" — the injected priority leaking through a keyword rule — which is
exactly the kind of thing this script exists to surface after a prompt change.

I built this second item because it is the natural companion to re-classification: change
the prompt, run the evaluation, then decide whether to pay for the re-run.
