"""Read Android accessibility trees: the list of on-screen elements and their boxes.

Android describes every screen to screen readers (for blind users) as a tree of
elements: buttons, text, input fields, lists. Each element has a box in screen pixels
and flags such as "clickable". AndroidControl saved this tree for every screenshot.

We use it for two things:
- scoring: a predicted tap counts as right if it lands inside the box of the element
  the human tapped;
- an optional "snap to the nearest button" add-on.

Field numbers below come from android_env's .proto files
(android_env/proto/a11y/*.proto at github.com/google-deepmind/android_env).
"""

from __future__ import annotations

from dataclasses import dataclass

from phone_agent.wire import iter_fields, to_signed

# AndroidAccessibilityForest
_FOREST_WINDOWS = 1
# AndroidAccessibilityWindowInfo
_WINDOW_LAYER, _WINDOW_TYPE, _WINDOW_TREE = 4, 6, 11
# AndroidAccessibilityTree
_TREE_NODES = 1
# AndroidAccessibilityNodeInfo
_NODE_BOUNDS, _NODE_CLASS, _NODE_DESC, _NODE_TEXT, _NODE_RESOURCE_ID = 2, 3, 4, 7, 10
_NODE_CLICKABLE, _NODE_EDITABLE, _NODE_LONG_CLICKABLE = 14, 15, 19
_NODE_SCROLLABLE, _NODE_VISIBLE = 21, 23
# ProtoRect
_RECT_LEFT, _RECT_TOP, _RECT_RIGHT, _RECT_BOTTOM = 1, 2, 3, 4

WINDOW_TYPES = {1: "application", 2: "keyboard", 3: "system", 4: "accessibility_overlay"}


@dataclass(frozen=True)
class Element:
    """One on-screen element. Box edges are in screenshot pixels."""

    left: int
    top: int
    right: int
    bottom: int
    clickable: bool = False
    long_clickable: bool = False
    editable: bool = False
    scrollable: bool = False
    visible: bool = False
    text: str = ""
    description: str = ""
    resource_id: str = ""
    class_name: str = ""
    window: str = "other"
    layer: int = 0

    @property
    def area(self) -> int:
        return max(0, self.right - self.left) * max(0, self.bottom - self.top)

    @property
    def interactive(self) -> bool:
        """Something a user can act on: tap, long press, or type into."""
        return self.clickable or self.long_clickable or self.editable

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom


def _text(value) -> str:
    return bytes(value).decode("utf-8", errors="replace")


def _rect(buf) -> tuple[int, int, int, int]:
    edges = {_RECT_LEFT: 0, _RECT_TOP: 0, _RECT_RIGHT: 0, _RECT_BOTTOM: 0}
    for field, _, value in iter_fields(buf):
        if field in edges:
            edges[field] = to_signed(value)
    return edges[_RECT_LEFT], edges[_RECT_TOP], edges[_RECT_RIGHT], edges[_RECT_BOTTOM]


def _node(buf, window: str, layer: int) -> Element:
    box = (0, 0, 0, 0)
    flags: dict[str, object] = {}
    for field, _, value in iter_fields(buf):
        if field == _NODE_BOUNDS:
            box = _rect(value)
        elif field == _NODE_CLICKABLE:
            flags["clickable"] = bool(value)
        elif field == _NODE_LONG_CLICKABLE:
            flags["long_clickable"] = bool(value)
        elif field == _NODE_EDITABLE:
            flags["editable"] = bool(value)
        elif field == _NODE_SCROLLABLE:
            flags["scrollable"] = bool(value)
        elif field == _NODE_VISIBLE:
            flags["visible"] = bool(value)
        elif field == _NODE_TEXT:
            flags["text"] = _text(value)
        elif field == _NODE_DESC:
            flags["description"] = _text(value)
        elif field == _NODE_RESOURCE_ID:
            flags["resource_id"] = _text(value)
        elif field == _NODE_CLASS:
            flags["class_name"] = _text(value)
    return Element(*box, window=window, layer=layer, **flags)


def parse_forest(data: bytes) -> list[Element]:
    """All elements in all windows (app, keyboard, status bar, ...) of one screen."""
    elements: list[Element] = []
    for field, _, window_buf in iter_fields(data):
        if field != _FOREST_WINDOWS:
            continue
        layer, window, trees = 0, "other", []
        for wfield, _, value in iter_fields(window_buf):
            if wfield == _WINDOW_LAYER:
                layer = to_signed(value)
            elif wfield == _WINDOW_TYPE:
                window = WINDOW_TYPES.get(value, "other")
            elif wfield == _WINDOW_TREE:
                trees.append(value)
        for tree in trees:
            for tfield, _, node_buf in iter_fields(tree):
                if tfield == _TREE_NODES:
                    elements.append(_node(node_buf, window, layer))
    return elements


def boxes_at(elements: list[Element], x: float, y: float) -> list[Element]:
    """Visible elements whose box contains the point, smallest first.

    The first one is usually the button that was actually pressed; the bigger ones
    are its containers (a list row, the whole page).
    """
    hits = [e for e in elements if e.visible and e.area > 0 and e.contains(x, y)]
    return sorted(hits, key=lambda e: e.area)
