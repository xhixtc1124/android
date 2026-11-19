"""AndroidControl: read Google's demos and translate their actions into ours.

One demo ("episode") = a goal + a series of screens + the human's action on each
screen. There is always one more screen than actions: the last screen shows the
result after the final action.

Google's action names -> ours:
    click        -> tap          (pixels -> 0-1000)
    long_press   -> long_press   (pixels -> 0-1000)
    input_text   -> type
    scroll       -> scroll       (same meaning: "down" = show what's further down)
    open_app     -> open_app
    navigate_back/home -> back/home
    wait         -> wait
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from phone_agent.actions import POINT_ACTIONS, SCROLL_DIRECTIONS, Action, to_norm
from phone_agent.tfrecord import iter_records, parse_example

SPLITS_FILE = "splits.json"
SUBSPLITS_FILE = "test_subsplits.json"
SHARD_GLOB = "android_control-*-of-00020"

_SIMPLE = {"navigate_back": "back", "navigate_home": "home", "wait": "wait"}
_POINT = {"click": "tap", "long_press": "long_press"}


@dataclass
class Episode:
    episode_id: int
    goal: str
    screenshots: list[bytes]  # PNG files, one per screen
    widths: list[int]
    heights: list[int]
    trees: list[bytes]  # accessibility tree per screen (see a11y.py)
    actions: list[dict]  # Google's action JSON, one per screen except the last
    step_instructions: list[str]  # human-written note per action (noisy)


def _text(values: list[bytes]) -> list[str]:
    return [v.decode("utf-8") for v in values]


def iter_episodes(path: str | Path) -> Iterator[Episode]:
    """Yield every demo in one shard file."""
    for raw in iter_records(path):
        ex = parse_example(raw)
        yield Episode(
            episode_id=int(ex["episode_id"][0]),
            goal=" ".join(_text(ex["goal"])[0].split()),  # tidy repeated spaces
            screenshots=ex["screenshots"],
            widths=ex["screenshot_widths"],
            heights=ex["screenshot_heights"],
            trees=ex.get("accessibility_trees", []),
            # A demo with zero actions has no "actions" field at all.
            actions=[json.loads(a) for a in ex.get("actions", [])],
            step_instructions=_text(ex.get("step_instructions", [])),
        )


def convert_action(raw: dict, width: int, height: int) -> Action:
    """Google's action dict -> our Action. Raises ValueError for anything unexpected."""
    kind = raw.get("action_type")
    if kind in _POINT:
        x, y = raw.get("x"), raw.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError(f"{kind} without numeric x/y: {raw}")
        nx, ny = to_norm(x, y, width, height)
        return Action(_POINT[kind], x=nx, y=ny)
    if kind in _SIMPLE:
        return Action(_SIMPLE[kind])
    if kind == "scroll":
        direction = str(raw.get("direction", "")).lower()
        if direction not in SCROLL_DIRECTIONS:
            raise ValueError(f"scroll with bad direction: {raw}")
        return Action("scroll", direction=direction)
    if kind == "input_text":
        if not isinstance(raw.get("text"), str):
            raise ValueError(f"input_text without text: {raw}")
        return Action("type", text=raw["text"])
    if kind == "open_app":
        if not isinstance(raw.get("app_name"), str) or not raw["app_name"].strip():
            raise ValueError(f"open_app without app_name: {raw}")
        return Action("open_app", app=raw["app_name"])
    raise ValueError(f"unknown action_type: {raw}")


def is_point(action: Action) -> bool:
    return action.action in POINT_ACTIONS


def load_splits(raw_dir: str | Path) -> tuple[dict[int, str], dict[int, list[str]]]:
    """episode_id -> 'train'/'validation'/'test', and test episode_id -> its test groups.

    Test groups: IDD (seen apps and tasks), app_unseen, task_unseen, category_unseen.
    They overlap: an episode from a new category is also from a new app.
    """
    raw_dir = Path(raw_dir)
    split_of: dict[int, str] = {}
    for split, ids in json.loads((raw_dir / SPLITS_FILE).read_text()).items():
        for episode_id in map(int, ids):
            if episode_id in split_of:
                raise ValueError(f"episode {episode_id} is in two splits")
            split_of[episode_id] = split
    groups: dict[int, list[str]] = {}
    for group, ids in json.loads((raw_dir / SUBSPLITS_FILE).read_text()).items():
        for episode_id in ids:
            groups.setdefault(int(episode_id), []).append(group)
    return split_of, {k: sorted(v) for k, v in groups.items()}
