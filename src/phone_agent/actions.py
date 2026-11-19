"""Action format shared by data conversion, scoring and the live harness.

The model answers with one JSON object per step, for example:
    {"action": "tap", "x": 512, "y": 731}

Coordinates use Muse Glimmer's native normalized scale: 0-1000 across (x) and
0-1000 down (y), relative to the screenshot. Labels therefore don't change when
screenshots are resized, and to_pixels() maps them onto any screen size.

Draft: confirm against AndroidControl on day 1 (scroll-direction meaning,
whether each demo's final screen exists for a "done" label).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

COORD_MAX = 1000
POINT_ACTIONS = {"tap", "long_press"}
SCROLL_DIRECTIONS = {"up", "down", "left", "right"}
DONE_STATUSES = {"complete", "infeasible"}
ACTION_TYPES = POINT_ACTIONS | {"type", "scroll", "back", "home", "open_app", "wait", "done"}


def to_norm(x_px: float, y_px: float, width: int, height: int) -> tuple[int, int]:
    """Screen pixels -> 0-1000 normalized coordinates (clamped to the screen)."""
    x = round(min(max(x_px, 0), width) / width * COORD_MAX)
    y = round(min(max(y_px, 0), height) / height * COORD_MAX)
    return x, y


def to_pixels(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    """0-1000 normalized coordinates -> screen pixels (clamped to the last pixel)."""
    return (
        min(round(x / COORD_MAX * width), width - 1),
        min(round(y / COORD_MAX * height), height - 1),
    )


@dataclass(frozen=True)
class Action:
    action: str
    x: int | None = None
    y: int | None = None
    text: str | None = None
    direction: str | None = None
    app: str | None = None
    status: str | None = None
    answer: str | None = None

    def to_json(self) -> str:
        """Compact JSON with only the fields this action uses; the training label format."""
        return json.dumps({k: v for k, v in asdict(self).items() if v is not None}, ensure_ascii=False, separators=(", ", ": "))


class ParseError(ValueError):
    """Why a model answer could not be used. `reason` is one of:
    no_json, not_object, unknown_action, missing_field, bad_value, out_of_range."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def _first_json_object(text: str) -> object:
    """Decode the first JSON object in text, tolerating code fences or prose around it.
    Applied identically to base and tuned answers."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
            return obj
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
    raise ParseError("no_json")


def _coord(obj: dict, key: str) -> int:
    if key not in obj:
        raise ParseError("missing_field", key)
    value = obj[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParseError("bad_value", f"{key}={value!r}")
    if not 0 <= value <= COORD_MAX:
        raise ParseError("out_of_range", f"{key}={value}")
    return round(value)


def _text(obj: dict, key: str) -> str:
    value = obj.get(key)
    if value is None:
        raise ParseError("missing_field", key)
    if not isinstance(value, str):
        raise ParseError("bad_value", f"{key}={value!r}")
    return value


def parse_action(text: str) -> Action:
    """Parse one model answer into an Action, or raise ParseError with a reason."""
    obj = _first_json_object(text)
    if not isinstance(obj, dict):
        raise ParseError("not_object")
    kind = obj.get("action")
    if kind not in ACTION_TYPES:
        raise ParseError("unknown_action", repr(kind))

    if kind in POINT_ACTIONS:
        return Action(kind, x=_coord(obj, "x"), y=_coord(obj, "y"))
    if kind == "type":
        return Action(kind, text=_text(obj, "text"))
    if kind == "scroll":
        direction = _text(obj, "direction").lower()
        if direction not in SCROLL_DIRECTIONS:
            raise ParseError("bad_value", f"direction={direction!r}")
        return Action(kind, direction=direction)
    if kind == "open_app":
        return Action(kind, app=_text(obj, "app"))
    if kind == "done":
        status = _text(obj, "status").lower()
        if status not in DONE_STATUSES:
            raise ParseError("bad_value", f"status={status!r}")
        answer = obj.get("answer")
        if answer is not None and not isinstance(answer, str):
            raise ParseError("bad_value", f"answer={answer!r}")
        return Action(kind, status=status, answer=answer)
    return Action(kind)  # back, home, wait
