"""Tests for the hand-written readers: protobuf (wire.py), TFRecord, accessibility trees."""

import gzip
import struct

import pytest

from phone_agent import a11y
from phone_agent.tfrecord import iter_records, parse_example
from phone_agent.wire import iter_fields, read_varint, to_signed


# --- tiny protobuf encoder, only for building test inputs ---
def varint(n: int) -> bytes:
    n &= (1 << 64) - 1  # negative numbers become 10-byte varints, like real protobuf
    out = bytearray()
    while True:
        byte, n = n & 0x7F, n >> 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def field(number: int, value) -> bytes:
    if isinstance(value, (bool, int)):
        return varint(number << 3) + varint(int(value))
    if isinstance(value, str):
        value = value.encode()
    return varint(number << 3 | 2) + varint(len(value)) + value


def test_varint_round_trip():
    for n in (0, 1, 127, 128, 300, 2**40):
        assert read_varint(varint(n), 0) == (n, len(varint(n)))


def test_negative_numbers():
    value, _ = read_varint(varint(-5), 0)
    assert to_signed(value) == -5


def test_iter_fields_mixed():
    msg = field(1, 42) + field(2, "hi") + field(3, b"\x00\x01")
    got = [(f, w, v if isinstance(v, int) else bytes(v)) for f, w, v in iter_fields(msg)]
    assert got == [(1, 0, 42), (2, 2, b"hi"), (3, 2, b"\x00\x01")]


def test_truncated_message_fails():
    with pytest.raises(ValueError):
        list(iter_fields(field(2, "hello")[:-2]))


def _feature_bytes(values: list[bytes]) -> bytes:
    return field(1, b"".join(field(1, v) for v in values))  # Feature.bytes_list


def _feature_ints(values: list[int]) -> bytes:
    packed = b"".join(varint(v) for v in values)
    return field(3, field(1, packed))  # Feature.int64_list, packed


def _feature_floats(values: list[float]) -> bytes:
    return field(2, field(1, struct.pack(f"<{len(values)}f", *values)))  # Feature.float_list


def _example(features: dict[str, bytes]) -> bytes:
    entries = b"".join(field(1, field(1, k) + field(2, v)) for k, v in features.items())
    return field(1, entries)


def test_tfrecord_and_example(tmp_path):
    examples = [
        _example({"goal": _feature_bytes([b"open the app"]), "ids": _feature_ints([7, -1]), "f": _feature_floats([0.5])}),
        _example({"goal": _feature_bytes([b"second"]), "empty": _feature_bytes([])}),
    ]
    path = tmp_path / "data.tfrecord"
    with gzip.open(path, "wb") as f:
        for data in examples:
            f.write(struct.pack("<Q", len(data)) + b"\0" * 4 + data + b"\0" * 4)

    rows = [parse_example(r) for r in iter_records(path)]
    assert rows[0] == {"goal": [b"open the app"], "ids": [7, -1], "f": [0.5]}
    assert rows[1] == {"goal": [b"second"], "empty": []}


def test_tfrecord_truncated_file_fails(tmp_path):
    path = tmp_path / "bad.tfrecord"
    with gzip.open(path, "wb") as f:
        f.write(struct.pack("<Q", 100) + b"\0" * 4 + b"short")
    with pytest.raises(ValueError):
        list(iter_records(path))


def _node(box, **flags) -> bytes:
    rect = b"".join(field(i + 1, v) for i, v in enumerate(box))
    numbers = {"clickable": 14, "editable": 15, "visible": 23, "text": 7}
    return field(2, rect) + b"".join(field(numbers[k], v) for k, v in flags.items())


def test_parse_forest_and_boxes_at():
    nodes = [
        _node((0, 0, 1080, 2400), visible=True),  # whole page
        _node((100, 100, 500, 300), visible=True, clickable=True),  # a row
        _node((120, 120, 300, 180), visible=True, text="Settings"),  # its label
        _node((-50, 0, 40, 40), visible=False, clickable=True),  # hidden, partly off-screen
    ]
    tree = b"".join(field(1, n) for n in nodes)
    window = field(4, 3) + field(6, 1) + field(11, tree)
    elements = a11y.parse_forest(field(1, window))

    assert len(elements) == 4
    assert elements[2].text == "Settings" and elements[2].window == "application" and elements[2].layer == 3
    assert elements[3].left == -50  # negative edges decode correctly

    hits = a11y.boxes_at(elements, 150, 150)
    assert [e.text for e in hits][0] == "Settings"  # smallest first
    assert [e.interactive for e in hits] == [False, True, False]
    assert a11y.boxes_at(elements, 10, 10)[0].area == 1080 * 2400  # hidden node ignored
