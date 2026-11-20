"""Tests for the training launcher's free-offer guard and row picking (no network)."""

import dataclasses
import json
from datetime import datetime

import pytest

from phone_agent.train import FREE_UNTIL, TRIALS, check_free, select_rows


def test_refuses_after_deadline():
    with pytest.raises(SystemExit):
        check_free(TRIALS["speed"], now=FREE_UNTIL)


def test_refuses_full_parameter():
    with pytest.raises(SystemExit):
        check_free(dataclasses.replace(TRIALS["speed"], lora_rank=0), now=datetime(2026, 11, 20, tzinfo=FREE_UNTIL.tzinfo))


def test_allows_lora_before_deadline():
    check_free(TRIALS["overfit"], now=datetime(2026, 11, 30, 23, 59, tzinfo=FREE_UNTIL.tzinfo))


def _records(n=200):
    kinds = ["tap", "scroll", "type", "open_app", "back", "done", "wait"]
    return [
        {"id": f"r{i}", "label": json.dumps({"action": kinds[i % len(kinds)]}), "history": [], "goal": "g" * (i % 5),
         "screen": [1080, 2400] if i else [1440, 3120]}
        for i in range(n)
    ]


def test_overfit_rows_cover_every_action():
    rows = select_rows(_records(), TRIALS["overfit"])
    assert len(rows) == 32 and len({r["id"] for r in rows}) == 32
    assert {json.loads(r["label"])["action"] for r in rows} == {"tap", "scroll", "type", "open_app", "back", "done", "wait"}


def test_speed_rows_are_seeded():
    trial = dataclasses.replace(TRIALS["speed"], rows=50)
    assert select_rows(_records(), trial) == select_rows(_records(), trial)
