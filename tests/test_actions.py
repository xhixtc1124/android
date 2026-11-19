import pytest

from phone_agent.actions import Action, ParseError, parse_action, to_norm, to_pixels

W, H = 1080, 2400  # AndroidControl / Pixel 6 screen size (to confirm on day 1)


def test_coordinates_round_trip_within_one_unit():
    for x_px, y_px in [(0, 0), (540, 1200), (1079, 2399), (123, 2001)]:
        x, y = to_norm(x_px, y_px, W, H)
        bx, by = to_pixels(x, y, W, H)
        assert abs(bx - x_px) <= W / 1000 and abs(by - y_px) <= H / 1000


def test_normalized_coords_are_resolution_independent():
    assert to_norm(540, 1200, 1080, 2400) == to_norm(720, 1600, 1440, 3200) == (500, 500)
    assert to_pixels(500, 500, 1440, 3200) == (720, 1600)


def test_to_norm_clamps_off_screen_points():
    assert to_norm(-5, 9999, W, H) == (0, 1000)
    assert to_pixels(1000, 1000, W, H) == (W - 1, H - 1)


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"action": "tap", "x": 512, "y": 731}', Action("tap", x=512, y=731)),
        ('{"action": "long_press", "x": 10.4, "y": 990}', Action("long_press", x=10, y=990)),
        ('{"action": "type", "text": "hello"}', Action("type", text="hello")),
        ('{"action": "scroll", "direction": "Down"}', Action("scroll", direction="down")),
        ('{"action": "open_app", "app": "Clock"}', Action("open_app", app="Clock")),
        ('{"action": "back"}', Action("back")),
        ('{"action": "done", "status": "complete"}', Action("done", status="complete")),
        ('{"action": "done", "status": "complete", "answer": "3"}', Action("done", status="complete", answer="3")),
    ],
)
def test_parse_valid_actions(text, expected):
    assert parse_action(text) == expected


def test_parse_tolerates_fences_and_prose_identically():
    text = 'Sure.\n```json\n{"action": "home"}\n```'
    assert parse_action(text) == Action("home")


@pytest.mark.parametrize(
    "text, reason",
    [
        ("tap the button", "no_json"),
        ('{"action": "tap", "x": 512}', "missing_field"),
        ('{"action": "tap", "x": 512, "y": 1970}', "out_of_range"),  # base-model failure seen in prior work
        ('{"action": "tap", "x": "512", "y": 7}', "bad_value"),
        ('{"action": "tap", "x": true, "y": 7}', "bad_value"),
        ('{"action": "swipe"}', "unknown_action"),
        ('{"action": "scroll", "direction": "sideways"}', "bad_value"),
        ('{"action": "done", "status": "maybe"}', "bad_value"),
    ],
)
def test_parse_reports_failure_reason(text, reason):
    with pytest.raises(ParseError) as err:
        parse_action(text)
    assert err.value.reason == reason


def test_label_json_round_trips():
    for action in [Action("tap", x=1, y=999), Action("type", text="café"), Action("done", status="infeasible")]:
        assert parse_action(action.to_json()) == action
    assert Action("tap", x=5, y=6).to_json() == '{"action": "tap", "x": 5, "y": 6}'
