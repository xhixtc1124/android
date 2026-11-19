"""Tiny reader for Protocol Buffers ("protobuf"), Google's binary data format.

AndroidControl stores everything as protobuf messages. The official way to read them
needs TensorFlow and android_env installed. The format itself is simple, so we decode
it by hand and need no extra packages.

How protobuf bytes look: a message is a list of fields. Each field starts with a
"key" number that holds the field number and its wire type, followed by the value:
    wire type 0 = varint (a whole number of 1-10 bytes)
    wire type 1 = 8 fixed bytes
    wire type 2 = length + that many bytes (text, raw bytes, or a nested message)
    wire type 5 = 4 fixed bytes
We don't need the schema to split a message into fields; the schema only tells us what
each field number means (we copy those numbers from the .proto files into constants).
"""

from __future__ import annotations

from collections.abc import Iterator

VARINT, FIXED64, LENGTH, FIXED32 = 0, 1, 2, 5


def read_varint(buf: memoryview | bytes, pos: int) -> tuple[int, int]:
    """Read one varint starting at `pos`. Returns (value, position after it).

    Each byte gives 7 bits of the number; a high bit of 1 means "more bytes follow".
    """
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint longer than 10 bytes")


def to_signed(value: int) -> int:
    """Negative int32/int64 values are stored as huge unsigned 64-bit numbers; undo that."""
    return value - (1 << 64) if value >= (1 << 63) else value


def iter_fields(buf: memoryview | bytes) -> Iterator[tuple[int, int, int | memoryview]]:
    """Yield (field_number, wire_type, value) for every field in one message.

    Varints come back as int; everything else as a memoryview (a zero-copy slice).
    """
    view = memoryview(buf)
    pos = 0
    end = len(view)
    while pos < end:
        key, pos = read_varint(view, pos)
        field, wire_type = key >> 3, key & 7
        if wire_type == VARINT:
            value, pos = read_varint(view, pos)
            yield field, wire_type, value
            continue
        if wire_type == LENGTH:
            size, pos = read_varint(view, pos)
        elif wire_type == FIXED64:
            size = 8
        elif wire_type == FIXED32:
            size = 4
        else:
            raise ValueError(f"unsupported wire type {wire_type} for field {field}")
        if pos + size > end:
            raise ValueError(f"field {field} runs past the end of the message")
        yield field, wire_type, view[pos : pos + size]
        pos += size


def iter_packed_varints(buf: memoryview | bytes) -> Iterator[int]:
    """Repeated numbers are often "packed": one LENGTH field holding varints back to back."""
    view = memoryview(buf)
    pos = 0
    while pos < len(view):
        value, pos = read_varint(view, pos)
        yield value
