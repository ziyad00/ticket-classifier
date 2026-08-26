#!/usr/bin/env python3
"""Load data/sample_tickets.json into a running service and (optionally) wait for results.

    python scripts/load_samples.py [--url http://localhost:8000] [--wait]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "sample_tickets.json"


def request(method: str, url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--wait", action="store_true", help="poll until every ticket is classified or failed")
    args = ap.parse_args()

    tickets = json.loads(SAMPLES.read_text())
    for t in tickets:
        status, _ = request("POST", f"{args.url}/tickets", t)
        print(f"{t['id']}: {'created' if status == 201 else 'already existed' if status == 200 else f'HTTP {status}'}")

    if not args.wait:
        return 0

    ids = {t["id"] for t in tickets}
    deadline = time.time() + 60
    while time.time() < deadline:
        _, listing = request("GET", f"{args.url}/tickets?limit=100")
        rows = {r["id"]: r for r in listing["items"] if r["id"] in ids}
        if all(r["status"] != "pending" for r in rows.values()):
            break
        time.sleep(0.5)

    print(f"\n{'id':8} {'status':11} {'category':10} {'priority':8} {'att':>3}  summary / error")
    for t in tickets:
        r = rows[t["id"]]
        c = r["classification"] or {}
        tail = c.get("summary") or r["last_error"] or ""
        flag = " [injection?]" if r["injection_suspected"] else ""
        print(f"{r['id']:8} {r['status']:11} {c.get('category', '-'):10} {c.get('priority', '-'):8} {r['attempts']:>3}  {tail[:70]}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
