from tests.conftest import GOOD, SAMPLES, ScriptedClassifier, post_ticket, wait_for_status


async def test_create_then_read(make_client):
    client, _ = await make_client(ScriptedClassifier(GOOD))
    r = await post_ticket(client, "t-1")
    assert r.status_code == 201
    assert r.json()["status"] == "pending"
    assert r.json()["classification"] is None

    data = await wait_for_status(client, "t-1", "classified")
    assert data["classification"] == {
        "category": "billing",
        "priority": "high",
        "summary": "Customer was charged twice and wants a refund.",
    }
    assert data["attempts"] == 1
    assert data["prompt_version"]


async def test_duplicate_id_is_idempotent(make_client):
    clf = ScriptedClassifier(GOOD)
    client, _ = await make_client(clf)
    await post_ticket(client, "t-1", body="original")
    await wait_for_status(client, "t-1", "classified")

    r = await post_ticket(client, "t-1", body="different body, same id")
    assert r.status_code == 200
    assert r.json()["body"] == "original"
    assert r.json()["status"] == "classified"

    listing = (await client.get("/tickets")).json()
    assert listing["total"] == 1
    assert clf.calls == 1, "duplicate submission must not re-run classification"


async def test_classification_is_not_synchronous(make_client):
    clf = ScriptedClassifier(GOOD, latency=0.3)
    client, _ = await make_client(clf)
    r = await post_ticket(client, "t-1")
    assert r.status_code == 201 and r.json()["status"] == "pending"
    assert r.elapsed.total_seconds() < 0.2


async def test_get_unknown_404_error_shape(make_client):
    client, _ = await make_client(ScriptedClassifier(GOOD))
    r = await client.get("/tickets/nope")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


async def test_validation_errors(make_client):
    client, _ = await make_client(ScriptedClassifier(GOOD))
    r = await client.post("/tickets", json={"id": "t-1", "subject": "s"})  # no body
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"
    r = await client.post("/tickets", json={"id": "has spaces", "subject": "s", "body": "b"})
    assert r.status_code == 422
    r = await client.post("/tickets", json={"id": "t-2", "subject": "", "body": ""})
    assert r.status_code == 422
    r = await client.get("/tickets", params={"category": "refund"})
    assert r.status_code == 422


async def test_empty_subject_is_allowed(make_client):
    client, _ = await make_client(ScriptedClassifier(GOOD))
    r = await post_ticket(client, "t-1008", subject="", body="asdf")
    assert r.status_code == 201


async def test_list_filter_and_pagination(make_client):
    script = [
        '{"category": "billing", "priority": "high", "summary": "a"}',
        '{"category": "billing", "priority": "low", "summary": "b"}',
        '{"category": "technical", "priority": "high", "summary": "c"}',
    ]
    client, _ = await make_client(ScriptedClassifier(*script), worker_concurrency=1)
    for i in range(3):
        await post_ticket(client, f"t-{i}")
    for i in range(3):
        await wait_for_status(client, f"t-{i}", "classified")

    r = (await client.get("/tickets", params={"category": "billing"})).json()
    assert r["total"] == 2 and [t["id"] for t in r["items"]] == ["t-0", "t-1"]

    r = (await client.get("/tickets", params={"category": "billing", "priority": "high"})).json()
    assert [t["id"] for t in r["items"]] == ["t-0"]

    r = (await client.get("/tickets", params={"priority": "high"})).json()
    assert [t["id"] for t in r["items"]] == ["t-0", "t-2"]

    page1 = (await client.get("/tickets", params={"limit": 2, "offset": 0})).json()
    page2 = (await client.get("/tickets", params={"limit": 2, "offset": 2})).json()
    assert [t["id"] for t in page1["items"]] == ["t-0", "t-1"]
    assert [t["id"] for t in page2["items"]] == ["t-2"]
    assert page1["total"] == page2["total"] == 3

    assert (await client.get("/tickets", params={"limit": 0})).status_code == 422


async def test_sample_tickets_load_with_fake_classifier(make_client):
    """End to end with the shipped fake, failures included: everything must settle."""
    client, _ = await make_client(None, fake_failure_rate=0.3, worker_concurrency=3)
    for t in SAMPLES:
        assert (await client.post("/tickets", json=t)).status_code == 201
    for t in SAMPLES:
        data = await wait_for_status(client, t["id"], "classified", "failed", timeout=5)
        assert data["status"] in ("classified", "failed")
        if data["status"] == "classified":
            assert data["classification"]["category"] in ("billing", "technical", "account", "other")
        else:
            assert data["attempts"] == 3 and data["last_error"]
    health = (await client.get("/health")).json()
    assert health["pending"] == 0
