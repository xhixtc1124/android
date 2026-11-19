"""What the model sees each step, and the training row built from it.

One step = one chat:
    system:    the rules (actions, coordinate scale, answer format)
    user:      [screenshot] + goal + the actions taken so far
    assistant: the next action as JSON           <- the only part that is trained

Training, the offline test and the live test all build prompts here, so the model
always sees exactly the same wording.
"""

from __future__ import annotations

import base64
from pathlib import Path

DEFAULT_MAX_HISTORY = 8

SYSTEM_PROMPT = """You operate an Android phone to complete the user's goal.
Each turn you get a screenshot of the current screen, the goal, and the actions taken so far.
Reply with exactly one next action as a JSON object and nothing else.

Coordinates: x and y are whole numbers from 0 to 1000. (0, 0) is the top-left corner of the screenshot and (1000, 1000) the bottom-right corner.

Actions:
{"action": "tap", "x": X, "y": Y}
{"action": "long_press", "x": X, "y": Y}
{"action": "type", "text": "..."}  types into the selected text field; it does not tap or press enter
{"action": "scroll", "direction": "up" | "down" | "left" | "right"}  "down" shows what is further down
{"action": "open_app", "app": "App name"}
{"action": "back"}
{"action": "home"}
{"action": "wait"}  wait for the screen to finish loading
{"action": "done", "status": "complete" | "infeasible"}  the goal is finished, or cannot be done; add "answer": "..." if the goal asks a question

Reasoning strength: low."""


def user_text(
    goal: str,
    history: list[str],
    step_instruction: str | None = None,
    max_history: int = DEFAULT_MAX_HISTORY,
) -> str:
    """The text half of the user turn. `history` is a list of action JSON strings, oldest first."""
    lines = [f"Goal: {goal}"]
    if step_instruction:
        lines.append(f"Current step: {step_instruction}")
    if not history:
        lines.append("Previous actions: none")
    else:
        shown = history[max(0, len(history) - max(0, max_history)) :]
        lines.append("Previous actions (oldest first):")
        if len(shown) < len(history):
            lines.append(f"({len(history) - len(shown)} earlier actions not shown)")
        lines.extend(shown)
    lines.append("What is the next action?")
    return "\n".join(lines)


def image_data_url(path: str | Path) -> str:
    """A JPEG file as a data URL: the image itself written out as text inside the row."""
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def prompt_messages(
    step: dict,
    image_url: str,
    *,
    use_step_instruction: bool = False,
    max_history: int = DEFAULT_MAX_HISTORY,
) -> list[dict]:
    """System + user messages for one step record (see convert.py for the record fields)."""
    text = user_text(
        step["goal"],
        step["history"],
        step.get("step_instruction") if use_step_instruction else None,
        max_history,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": text},
            ],
        },
    ]


def training_row(step: dict, image_url: str, **kwargs) -> dict:
    """A full chat with the human's action as the answer: one line of the training file."""
    messages = prompt_messages(step, image_url, **kwargs)
    messages.append({"role": "assistant", "content": step["label"]})
    return {"messages": messages}
