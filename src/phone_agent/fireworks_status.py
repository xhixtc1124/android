"""Read-only check for open Fireworks training sessions and trainer jobs.

Run before every launch: the account allows one serverless training run per
base model, and a session holds its slot until it expires or is deleted.
Prints resource names, states and times only; never the API key.
"""

from __future__ import annotations

import os
from collections import Counter

DONE_STATES = {"EXPIRED", "DELETED", "COMPLETED", "FAILED", "CANCELLED", "JOB_STATE_COMPLETED", "JOB_STATE_FAILED"}


def _list(fw, collection: str) -> list[dict]:
    items: list[dict] = []
    token = ""
    while True:
        query = f"pageSize=200&pageToken={token}" if token else "pageSize=200"
        # The SDK has no public list call for these collections; reuse its authenticated GET.
        resp = fw._get(f"/v1/accounts/{fw.account_id}/{collection}?{query}", timeout=30)
        if not resp.is_success:
            raise RuntimeError(f"listing {collection} failed: HTTP {resp.status_code}")
        data = resp.json()
        items += next((v for v in data.values() if isinstance(v, list)), [])
        token = data.get("nextPageToken", "")
        if not token:
            return items


def main() -> int:
    if not os.environ.get("FIREWORKS_API_KEY"):
        print("FIREWORKS_API_KEY is not set")
        return 1
    from fireworks.training.sdk import FireworksClient

    fw = FireworksClient(api_key=os.environ["FIREWORKS_API_KEY"])
    open_count = 0
    for collection in ("trainingSessions", "rlorTrainerJobs"):
        items = _list(fw, collection)
        states = Counter(str(it.get("state")) for it in items)
        print(f"{collection}: {len(items)} total, by state {dict(states)}")
        for it in items:
            if str(it.get("state")) in DONE_STATES:
                continue
            open_count += 1
            print(f"  OPEN {it.get('name', '').split('/')[-1]}  state={it.get('state')}  "
                  f"base={it.get('baseModel')}  created={it.get('createTime')}")
    print("no open sessions or jobs" if open_count == 0 else f"{open_count} open: ask before launching")
    return 0 if open_count == 0 else 3
