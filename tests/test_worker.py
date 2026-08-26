"""Lifecycle, retries, concurrency, restart recovery, re-classification."""

import asyncio

from app.classifier import InvalidModelOutput
from app.config import Settings
from app.guard import RegexGuard
from app.prompt import PROMPT_VERSION
from app.store import TicketStore
from app.worker import WorkerPool
from tests.conftest import GOOD, PERMANENT, TRANSIENT, ScriptedClassifier, post_ticket, wait_for_status

BAD = 'Sure, this is a billing issue.'


async def test_invalid_output_is_retried_then_stored(make_client):
    client, _ = await make_client(ScriptedClassifier(BAD, TRANSIENT, GOOD))
    await post_ticket(client)
    data = await wait_for_status(client, "t-1", "classified", "failed")
    assert data["status"] == "classified"
    assert data["attempts"] == 3
    assert data["last_error"] is None


async def test_persistently_invalid_output_ends_failed_and_stores_nothing(make_client):
    client, app = await make_client(ScriptedClassifier('{"category": "refund", "priority": "high", "summary": "x"}'))
    await post_ticket(client)
    data = await wait_for_status(client, "t-1", "classified", "failed")
    assert data["status"] == "failed"
    assert data["attempts"] == 3
    assert "invalid output" in data["last_error"] and "category" in data["last_error"]
    assert data["classification"] is None
    row = app.state.ctx.store.get("t-1")
    assert row.category is None and row.priority is None and row.summary is None


async def test_permanent_model_error_fails_immediately(make_client):
    clf = ScriptedClassifier(PERMANENT)
    client, _ = await make_client(clf)
    await post_ticket(client)
    data = await wait_for_status(client, "t-1", "failed")
    assert data["attempts"] == 1 and clf.calls == 1
    assert data["last_error"].startswith("permanent:")


async def test_model_timeout_counts_as_attempt(make_client):
    clf = ScriptedClassifier(GOOD, latency=0.5)
    client, _ = await make_client(clf, llm_timeout=0.05, max_attempts=2)
    await post_ticket(client)
    data = await wait_for_status(client, "t-1", "failed")
    assert data["attempts"] == 2 and "timeout" in data["last_error"]


async def test_concurrency_is_bounded(make_client):
    clf = ScriptedClassifier(GOOD, latency=0.1)
    client, _ = await make_client(clf, worker_concurrency=2)
    for i in range(6):
        await post_ticket(client, f"t-{i}")
    for i in range(6):
        await wait_for_status(client, f"t-{i}", "classified", timeout=5)
    assert clf.max_active == 2


async def test_reclassify_failed_ticket(make_client):
    clf = ScriptedClassifier(PERMANENT, GOOD)
    client, _ = await make_client(clf)
    await post_ticket(client)
    await wait_for_status(client, "t-1", "failed")

    r = await client.post("/tickets/t-1/reclassify")
    assert r.status_code == 202 and r.json()["status"] == "pending"
    data = await wait_for_status(client, "t-1", "classified")
    assert data["attempts"] == 1 and data["classification"]["category"] == "billing"

    assert (await client.post("/tickets/nope/reclassify")).status_code == 404


async def test_reclassify_stale_prompt_versions(make_client):
    client, app = await make_client(ScriptedClassifier(GOOD))
    for i in range(3):
        await post_ticket(client, f"t-{i}")
    for i in range(3):
        await wait_for_status(client, f"t-{i}", "classified")
    store: TicketStore = app.state.ctx.store
    with store._lock:  # pretend t-1 was classified under an older prompt
        store._conn.execute("UPDATE tickets SET prompt_version = 'old' WHERE id = 't-1'")

    # Without confirm: a dry run that changes nothing.
    r = await client.post("/tickets/reclassify-stale")
    assert r.status_code == 200 and r.json()["dry_run"] is True
    assert r.json()["affected"] == {"total": 1, "failed": 0, "stale": 1, "by_prompt_version": {"old": 1}, "this_call": 1}
    assert (await client.get("/tickets/t-1")).json()["status"] == "classified"
    whole_run = (await client.get("/tickets/reclassify-stale")).json()
    assert whole_run["affected"]["total"] == 1 and whole_run["estimate"] == r.json()["estimate"]

    # A non-billable classifier estimates $0, so even a zero cap passes.
    r = await client.post("/tickets/reclassify-stale", json={"confirm": True, "max_usd": 0})
    assert r.status_code == 202
    data = await wait_for_status(client, "t-1", "classified")
    assert data["prompt_version"] == PROMPT_VERSION
    assert r.json()["requeued"] == 1 and r.json()["remaining"] == 0 and r.json()["estimate"]["usd"] == 0.0
    assert (await client.post("/tickets/reclassify-stale", json={"confirm": True})).json()["requeued"] == 0


async def test_reclassify_stale_in_batches_is_resumable(make_client):
    client, app = await make_client(ScriptedClassifier(GOOD), worker_concurrency=1)
    for i in range(5):
        await post_ticket(client, f"t-{i}")
    for i in range(5):
        await wait_for_status(client, f"t-{i}", "classified")
    store: TicketStore = app.state.ctx.store
    with store._lock:
        store._conn.execute("UPDATE tickets SET prompt_version = 'old'")

    preview = (await client.post("/tickets/reclassify-stale", json={"limit": 2})).json()
    assert preview["affected"]["total"] == 5 and preview["affected"]["this_call"] == 2
    assert preview["estimate"]["model_calls"] == 2

    r = (await client.post("/tickets/reclassify-stale", json={"confirm": True, "limit": 2})).json()
    assert (r["requeued"], r["remaining"]) == (2, 3)
    for i in range(2):  # oldest first
        await wait_for_status(client, f"t-{i}", "classified")
    assert (await client.get("/tickets/t-2")).json()["prompt_version"] == "old"

    r = (await client.post("/tickets/reclassify-stale", json={"confirm": True, "limit": 2})).json()
    assert (r["requeued"], r["remaining"]) == (2, 1)
    r = (await client.post("/tickets/reclassify-stale", json={"confirm": True, "limit": 2})).json()
    assert (r["requeued"], r["remaining"]) == (1, 0)


def test_requeue_stale_batches_short_transactions():
    store = TicketStore(":memory:")
    for i in range(7):
        store.insert_if_absent(f"t-{i}", "s", "b")
    with store._lock:
        store._conn.execute("UPDATE tickets SET status = 'failed'")
    assert store.requeue_stale("v", limit=100, batch_size=3) == 7
    assert store.count("pending") == 7
    assert store.requeue_stale("v", limit=100, batch_size=3) == 0


def test_cost_estimate_scales_and_respects_prices():
    from app.cost import estimate

    s = Settings(price_input_per_mtok=5.0, price_output_per_mtok=25.0, max_attempts=3)
    zero = estimate(0, 0, s, "anthropic", billable=True)
    assert zero["usd"] == 0.0 and zero["model_calls"] == 0
    ten = estimate(10, 4000, s, "anthropic", billable=True)
    assert ten["model_calls"] == 10 and ten["model_calls_worst_case"] == 30
    assert ten["usd"] > 0 and ten["usd_worst_case"] == round(ten["usd"] * 3, 4)
    assert estimate(10, 4000, s, "fake", billable=False)["usd"] == 0.0
    pricier = Settings(price_input_per_mtok=50.0, price_output_per_mtok=250.0)
    assert estimate(10, 4000, pricier, "anthropic", billable=True)["usd"] > ten["usd"]


async def test_reclassify_stale_refuses_when_over_budget(make_client):
    """With a paid classifier name, a cap below the estimate blocks the requeue."""
    clf = ScriptedClassifier(GOOD)
    clf.billable = True
    client, app = await make_client(clf)
    await post_ticket(client, "t-1")
    await wait_for_status(client, "t-1", "classified")
    store: TicketStore = app.state.ctx.store
    with store._lock:
        store._conn.execute("UPDATE tickets SET prompt_version = 'old' WHERE id = 't-1'")

    preview = (await client.get("/tickets/reclassify-stale")).json()
    assert preview["estimate"]["usd"] > 0 and preview["estimate"]["model"] == "claude-opus-5"

    r = await client.post("/tickets/reclassify-stale", json={"confirm": True, "max_usd": 0})
    assert r.status_code == 409 and "exceeds max_usd" in r.json()["error"]["message"]
    assert (await client.get("/tickets/t-1")).json()["prompt_version"] == "old", "nothing requeued"

    r = await client.post("/tickets/reclassify-stale", json={"confirm": True, "max_usd": 1.0})
    assert r.status_code == 202 and r.json()["requeued"] == 1


# ---- store-level: leases and restart ---------------------------------------


def test_claim_is_exclusive_and_lease_expires():
    store = TicketStore(":memory:", lease_seconds=0.05)
    store.insert_if_absent("a", "s", "b")
    first = store.claim_next()
    assert first is not None and first.attempts == 1
    assert store.claim_next() is None, "a leased ticket must not be handed out twice"

    import time

    time.sleep(0.06)
    again = store.claim_next()
    assert again is not None and again.id == "a" and again.attempts == 2, "an abandoned lease is reclaimable"
    assert not store.mark_classified("a", first.locked_at, _cls(), "v"), "the stale holder cannot write its result"
    assert store.mark_classified("a", again.locked_at, _cls(), "v")
    assert store.get("a").status == "classified"


def test_db_rejects_out_of_set_values_even_if_code_is_bypassed():
    import sqlite3

    import pytest

    store = TicketStore(":memory:")
    store.insert_if_absent("a", "s", "b")
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute("UPDATE tickets SET category = 'refund' WHERE id = 'a'")
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute("UPDATE tickets SET status = 'processing' WHERE id = 'a'")


async def test_restart_releases_in_flight_leases(tmp_path):
    """Simulate a crash mid-classification, then a fresh process picking the ticket up."""
    settings = Settings(db_path=str(tmp_path / "t.db"), worker_concurrency=1, retry_base_delay=0.01, poll_interval=0.02)
    store = TicketStore(settings.db_path)
    store.insert_if_absent("a", "s", "b")
    claimed = store.claim_next()
    assert claimed is not None
    store.close()  # "crash": lease left behind, status still pending

    store2 = TicketStore(settings.db_path)
    assert store2.get("a").locked_at is not None
    pool = WorkerPool(store2, ScriptedClassifier(GOOD), settings, RegexGuard())
    await pool.start()
    try:
        for _ in range(100):
            if store2.get("a").status == "classified":
                break
            await asyncio.sleep(0.02)
        row = store2.get("a")
        assert row.status == "classified"
        assert row.attempts == 2, "the interrupted attempt counts, so a crash loop cannot run forever"
    finally:
        await pool.stop()
        store2.close()


async def test_stop_waits_for_in_flight_ticket():
    settings = Settings(db_path=":memory:", worker_concurrency=1, poll_interval=0.02, shutdown_grace=2.0)
    store = TicketStore(":memory:")
    store.insert_if_absent("a", "s", "b")
    pool = WorkerPool(store, ScriptedClassifier(GOOD, latency=0.2), settings, RegexGuard())
    await pool.start()
    await asyncio.sleep(0.05)  # worker is now inside the model call
    assert pool.in_flight == 1
    await pool.stop()
    assert store.get("a").status == "classified"
    assert store.get("a").locked_at is None


def _cls():
    from app.models import Classification

    return Classification(category="other", priority="low", summary="x")


def test_settings_reject_lease_shorter_than_model_timeout():
    import pytest

    with pytest.raises(ValueError, match="LEASE_SECONDS"):
        Settings(lease_seconds=10, llm_timeout=30)
    Settings(lease_seconds=31, llm_timeout=30)  # boundary is fine
