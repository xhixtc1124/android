"""Read TFRecord files without TensorFlow.

A TFRecord file is a list of records. Each record on disk is:
    8 bytes   length of the data (little-endian number)
    4 bytes   checksum of the length
    N bytes   the data: here, one `tf.train.Example` protobuf = one demo (episode)
    4 bytes   checksum of the data
AndroidControl's files are also gzip-compressed as a whole. Gzip has its own checksum,
so a damaged download still fails loudly even though we skip the per-record checksums
(those use CRC32C, which Python's standard library doesn't include).

A `tf.train.Example` is a dictionary: name -> list of bytes / floats / ints.
"""

from __future__ import annotations

import gzip
import struct
from collections.abc import Iterator
from pathlib import Path

from phone_agent.wire import FIXED32, LENGTH, iter_fields, iter_packed_varints, to_signed

# Field numbers from TensorFlow's example.proto / feature.proto.
_EXAMPLE_FEATURES = 1  # Example.features
_FEATURES_MAP = 1  # Features.feature (map<string, Feature>); each entry: key = 1, value = 2
_BYTES_LIST, _FLOAT_LIST, _INT64_LIST = 1, 2, 3  # which kind of list a Feature holds
_LIST_VALUES = 1  # BytesList.value / FloatList.value / Int64List.value


def iter_records(path: str | Path) -> Iterator[bytes]:
    """Yield the raw data of each record in a gzip-compressed TFRecord file."""
    with gzip.open(path, "rb") as f:
        while True:
            header = f.read(12)
            if not header:
                return
            if len(header) < 12:
                raise ValueError(f"{path}: file ends in the middle of a record header")
            (length,) = struct.unpack("<Q", header[:8])
            data = f.read(length)
            if len(data) < length or len(f.read(4)) < 4:
                raise ValueError(f"{path}: file ends in the middle of a record")
            yield data


def parse_example(data: bytes) -> dict[str, list]:
    """Decode one tf.train.Example into {name: [values]}."""
    out: dict[str, list] = {}
    for field, _, features in iter_fields(data):
        if field != _EXAMPLE_FEATURES:
            continue
        for map_field, _, entry in iter_fields(features):
            if map_field != _FEATURES_MAP:
                continue
            key, feature = None, None
            for entry_field, _, value in iter_fields(entry):
                if entry_field == 1:
                    key = bytes(value).decode("utf-8")
                elif entry_field == 2:
                    feature = value
            if key is not None:
                out[key] = _parse_feature(feature) if feature is not None else []
    return out


def _parse_feature(buf: memoryview) -> list:
    for kind, _, values in iter_fields(buf):
        if kind == _BYTES_LIST:
            return [bytes(v) for f, _, v in iter_fields(values) if f == _LIST_VALUES]
        if kind == _INT64_LIST:
            ints: list[int] = []
            for f, wire_type, v in iter_fields(values):
                if f != _LIST_VALUES:
                    continue
                if wire_type == LENGTH:  # packed
                    ints.extend(to_signed(x) for x in iter_packed_varints(v))
                else:
                    ints.append(to_signed(v))
            return ints
        if kind == _FLOAT_LIST:
            floats: list[float] = []
            for f, wire_type, v in iter_fields(values):
                if f != _LIST_VALUES:
                    continue
                if wire_type == LENGTH:  # packed
                    floats.extend(x for (x,) in struct.iter_unpack("<f", v))
                elif wire_type == FIXED32:
                    floats.append(struct.unpack("<f", v)[0])
            return floats
    return []
