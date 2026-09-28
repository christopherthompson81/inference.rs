"""Converts between the generated dataclasses and the JSON the engine takes and returns."""

import dataclasses
import json
import types as pytypes
import typing
from enum import Enum
from functools import cache

from . import types

TAG = "type"


class Mismatch(ValueError):
    """The JSON does not have the shape of the type it was read as."""


def to_data(value):
    """JSON-ready data for a dataclass, enum, list or dict; None fields are left out, as the server's defaults apply."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        wire = getattr(type(value), "_wire", {})
        fields = {
            wire.get(f.name, f.name): to_data(getattr(value, f.name))
            for f in dataclasses.fields(value)
            if getattr(value, f.name) is not None
        }
        external = getattr(type(value), "_external", None)
        return fields if external is None else {external: fields}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list | tuple):
        return [to_data(item) for item in value]
    if isinstance(value, dict):
        return {key: to_data(item) for key, item in value.items()}
    return value


def to_json(value) -> str:
    """A request as JSON: a str is taken to be JSON already; classes, dicts and lists of them are converted."""
    return value if isinstance(value, str) else json.dumps(to_data(value))


@cache
def _hints(cls):
    return typing.get_type_hints(cls, vars(types))


def _off_shape(strict: bool, data, message: str):
    """Leniently, JSON that does not fit is kept as it came (a newer server's field, say); strictly it is a mismatch."""
    if strict:
        raise Mismatch(message)
    return data


def from_data(annotation, data, strict: bool = False):
    """`data` read as `annotation`. Strict reading is for choosing a union's variant: any misfit is a mismatch."""
    if annotation is typing.Any or annotation is object:
        return data
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, pytypes.UnionType):
        return _union(typing.get_args(annotation), data, strict)
    if data is None:
        return None if annotation is type(None) else _off_shape(strict, None, f"null is not {annotation}")
    if origin is typing.Literal:
        allowed = typing.get_args(annotation)
        return data if data in allowed else _off_shape(strict, data, f"{data!r} is not one of {allowed}")
    if origin is list:
        if not isinstance(data, list):
            return _off_shape(strict, data, f"{type(data).__name__} is not a list")
        (item,) = typing.get_args(annotation) or (typing.Any,)
        return [from_data(item, value, strict) for value in data]
    if origin is dict:
        if not isinstance(data, dict):
            return _off_shape(strict, data, f"{type(data).__name__} is not an object")
        _, item = typing.get_args(annotation) or (str, typing.Any)
        return {key: from_data(item, value, strict) for key, value in data.items()}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            return annotation(data)
        except ValueError:
            return _off_shape(strict, data, f"{data!r} is not a {annotation.__name__}")
    if dataclasses.is_dataclass(annotation):
        return _dataclass(annotation, data, strict)
    return _scalar(annotation, data, strict)


def _scalar(annotation, data, strict: bool):
    if annotation is float and isinstance(data, int | float) and not isinstance(data, bool):
        return float(data)
    if annotation is int and isinstance(data, bool):
        return _off_shape(strict, data, "a boolean is not an integer")
    if isinstance(annotation, type) and not isinstance(data, annotation):
        return _off_shape(strict, data, f"{type(data).__name__} is not {annotation.__name__}")
    return data


def _dataclass(cls, data, strict: bool):
    external = getattr(cls, "_external", None)
    if external is not None:
        if not isinstance(data, dict) or list(data) != [external]:
            return _off_shape(strict, data, f"{cls.__name__} is written as {{{external!r}: ...}}")
        data = data[external]
    if not isinstance(data, dict):
        return _off_shape(strict, data, f"{type(data).__name__} is not a {cls.__name__}")
    wire = getattr(cls, "_wire", {})
    hints = _hints(cls)
    known = {wire.get(f.name, f.name): f for f in dataclasses.fields(cls)}
    if strict and not data.keys() <= known.keys():
        raise Mismatch(f"{sorted(data.keys() - known.keys())} are not fields of {cls.__name__}")
    values = {}
    for key, f in known.items():
        if key in data:
            values[f.name] = from_data(hints[f.name], data[key], strict)
        elif f.default is dataclasses.MISSING:
            # A server leaves some required fields out when they are empty.
            values[f.name] = _off_shape(strict, None, f"{cls.__name__} needs {key}")
    return cls(**values)


def _tagged(members, data):
    """The variant the data names: by its `type` tag, or by the one key of an externally tagged variant."""
    if isinstance(data, dict) and len(data) == 1:
        (key,) = data
        external = [member for member in members if getattr(member, "_external", None) == key]
        if len(external) == 1:
            return external[0]
    if not isinstance(data, dict) or not isinstance(data.get(TAG), str):
        return None
    matches = [
        member
        for member in members
        if dataclasses.is_dataclass(member)
        and any(f.name == TAG and f.default == data[TAG] for f in dataclasses.fields(member))
    ]
    return matches[0] if len(matches) == 1 else None


def _union(members, data, strict: bool):
    options = [member for member in members if member is not type(None)]
    if data is None:
        return None if len(options) < len(members) else _off_shape(strict, None, "null is not allowed")
    if len(options) == 1:
        return from_data(options[0], data, strict)
    tagged = _tagged(options, data)
    if tagged is not None:
        return from_data(tagged, data, strict)
    # An untagged union can only be told apart by which variant the data fits exactly.
    for member in options:
        try:
            return from_data(member, data, strict=True)
        except Mismatch:
            continue
    return _off_shape(strict, data, f"no variant of {members} fits")


def from_json(annotation, text: str):
    return from_data(annotation, json.loads(text))
