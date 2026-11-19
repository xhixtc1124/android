"""Tests for turning AndroidControl demos into step records and training rows."""

import io
import json

import pytest
from PIL import Image

from phone_agent import a11y, convert
from phone_agent.actions import Action, parse_action
from phone_agent.android_control import Episode, convert_action
from phone_agent.prompt import SYSTEM_PROMPT, training_row, user_text
from phone_agent.screens import PATCH_PX, image_tokens, model_grid, resize_for_model

from test_readers import _node, field


# --- action translation ---
@pytest.mark.parametrize(
    "raw, expected",
    [
        ({"action_type": "click", "x": 540, "y": 1200}, Action("tap", x=500, y=500)),
        ({"action_type": "long_press", "x": 0, "y": 2400}, Action("long_press", x=0, y=1000)),
        ({"action_type": "scroll", "direction": "Down"}, Action("scroll", direction="down")),
        ({"action_type": "input_text", "text": "Welsh cakes"}, Action("type", text="Welsh cakes")),
        ({"action_type": "open_app", "app_name": "Zoho Meeting"}, Action("open_app", app="Zoho Meeting")),
        ({"action_type": "navigate_back"}, Action("back")),
        ({"action_type": "navigate_home"}, Action("home")),
        ({"action_type": "wait"}, Action("wait")),
    ],
)
def test_convert_action(raw, expected):
    assert convert_action(raw, 1080, 2400) == expected


def test_convert_action_other_screen_size():
    # Same relative spot on a bigger phone -> same label.
    assert convert_action({"action_type": "click", "x": 720, "y": 1560}, 1440, 3120) == Action("tap", x=500, y=500)


@pytest.mark.parametrize(
    "raw",
    [
        {"action_type": "click", "x": "5", "y": 1},
        {"action_type": "scroll", "direction": "sideways"},
        {"action_type": "input_text"},
        {"action_type": "open_app", "app_name": " "},
        {"action_type": "teleport"},
    ],
)
def test_convert_action_rejects_bad_input(raw):
    with pytest.raises(ValueError):
        convert_action(raw, 1080, 2400)


# --- screenshot sizing ---
def test_model_grid_known_sizes():
    assert model_grid(1080, 2400, 1000) == (21, 47)  # 987 tokens
    assert model_grid(1440, 3120, 1000) == (21, 46)  # bigger phone, same budget
    assert image_tokens(1080, 2400, 1000) <= 1000


def test_model_grid_never_upscales():
    assert model_grid(280, 560, 10_000) == (10, 20)


def _png(width: int, height: int, color="white") -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def test_resize_gives_multiples_of_28():
    image = resize_for_model(_png(1080, 2400), 1000)
    assert image.mode == "RGB"
    assert image.size == (21 * PATCH_PX, 47 * PATCH_PX)


# --- prompt text ---
def test_user_text_history_cut():
    history = [json.dumps({"action": "wait"})] * 10
    text = user_text("Do it", history, max_history=3)
    assert "(7 earlier actions not shown)" in text
    assert text.count('{"action": "wait"}') == 3


def test_user_text_edge_cases():
    assert "Previous actions: none" in user_text("Do it", [])
    no_history = user_text("Do it", ['{"action": "back"}'], max_history=0)
    assert '{"action": "back"}' not in no_history and "(1 earlier actions not shown)" in no_history
    assert "Current step: tap OK" in user_text("Do it", [], step_instruction="tap OK")


def test_system_prompt_sets_low_reasoning():
    # The Glimmer renderer adds "Reasoning strength: high." unless the prompt already names one.
    assert "Reasoning strength: low." in SYSTEM_PROMPT


# --- whole demo -> step records ---
def _tree(*nodes: bytes) -> bytes:
    tree = b"".join(field(1, n) for n in nodes)
    return field(1, field(6, 1) + field(11, tree))


def _episode(n_actions: int = 2) -> Episode:
    button = _node((100, 100, 500, 300), visible=True, clickable=True)
    actions = [{"action_type": "click", "x": 300, "y": 200}, {"action_type": "input_text", "text": "hi"}]
    return Episode(
        episode_id=7,
        goal="Say hi",
        screenshots=[_png(1080, 2400)] * (n_actions + 1),
        widths=[1080] * (n_actions + 1),
        heights=[2400] * (n_actions + 1),
        trees=[_tree(button)] * (n_actions + 1),
        actions=actions[:n_actions],
        step_instructions=["tap the box", "type hi"][:n_actions],
    )


def test_episode_steps(tmp_path):
    records, counts = convert.episode_steps(_episode(), "train", [], tmp_path, 1000)

    assert [r["step"] for r in records] == [0, 1, 2]
    assert [parse_action(r["label"]).action for r in records] == ["tap", "type", "done"]
    assert [r["label_source"] for r in records] == ["human", "human", "end_of_demo"]
    assert records[2]["history"] == [records[0]["label"], records[1]["label"]]
    assert records[0]["target_boxes"] == [{"box": [93, 42, 463, 125], "interactive": True}]
    assert records[0]["interactive_boxes"] == [[93, 42, 463, 125]]
    assert (tmp_path / records[2]["image"]).exists()
    assert counts["episodes"] == 1 and counts["label_done"] == 1


def test_episode_without_actions_is_skipped(tmp_path):
    records, counts = convert.episode_steps(_episode(0), "train", [], tmp_path, 1000)
    assert records == [] and counts["skipped_episode_no_actions"] == 1


def test_training_row_shape(tmp_path):
    records, _ = convert.episode_steps(_episode(), "test", ["IDD"], tmp_path, 1000)
    row = training_row(records[1], "data:image/jpeg;base64,AAAA")
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["system", "user", "assistant"]
    user = row["messages"][1]["content"]
    assert user[0]["type"] == "image_url" and user[1]["type"] == "text"
    assert "tap the box" not in user[1]["text"]  # step notes are off by default
    assert row["messages"][2]["content"] == '{"action": "type", "text": "hi"}'
