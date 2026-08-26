"""Spend brakes, admin token, and plain-text summaries."""

from app.classifier import parse_classification
from tests.conftest import GOOD, ScriptedClassifier, post_ticket, wait_for_status


async def test_ingest_rate_limit_per_client(make_client):
    client, _ = await make_client(ScriptedClassifier(GOOD), ingest_rate_per_minute=3)
    for i in range(3):
        assert (await post_ticket(client, f"t-{i}")).status_code == 201
    r = await post_ticket(client, "t-3")
    assert r.status_code == 429 and r.json()["error"]["code"] == "rate_limited"
    assert (await client.get("/tickets/t-0")).status_code == 200, "reads are not rate limited"


async def test_daily_model_call_limit_pauses_workers(make_client):
    clf = ScriptedClassifier(GOOD)
    client, app = await make_client(clf, daily_model_call_limit=1, worker_concurrency=1)
    await post_ticket(client, "t-0")
    await post_ticket(client, "t-1")
    await wait_for_status(client, "t-0", "classified")
    import asyncio

    await asyncio.sleep(0.15)  # several poll intervals
    assert (await client.get("/tickets/t-1")).json()["status"] == "pending"
    assert clf.calls == 1
    stats = (await client.get("/stats")).json()
    assert stats["model_calls_today"] == 1 and stats["paused_for_budget"] is True and stats["pending"] == 1


async def test_admin_token_gates_stats_and_reclassify(make_client):
    client, _ = await make_client(ScriptedClassifier(GOOD), admin_token="s3cret")
    await post_ticket(client, "t-1")
    await wait_for_status(client, "t-1", "classified")

    assert (await client.get("/health")).json() == {"status": "ok"}
    for method, path in [("GET", "/stats"), ("GET", "/tickets/reclassify-stale"),
                         ("POST", "/tickets/reclassify-stale"), ("POST", "/tickets/t-1/reclassify")]:
        r = await client.request(method, path)
        assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized", path
        r = await client.request(method, path, headers={"Authorization": "Bearer s3cret"})
        assert r.status_code in (200, 202), path

    assert (await client.get("/tickets/t-1")).status_code == 200, "ticket reads are not admin-gated"


async def test_admin_endpoints_open_without_token(make_client):
    client, _ = await make_client(ScriptedClassifier(GOOD))
    assert (await client.get("/stats")).status_code == 200


def test_summary_is_reduced_to_plain_text():
    raw = '{"category": "other", "priority": "low", "summary": "Click <a href=x onclick=alert(1)>here</a>\\u0007 now"}'
    c = parse_classification(raw)
    assert c.summary == "Click a href=x onclick=alert(1)here/a now"
    assert "<" not in c.summary and "\x07" not in c.summary
