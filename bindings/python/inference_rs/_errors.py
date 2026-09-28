import json
from enum import IntEnum

from ._native import borrowed, lib


class Status(IntEnum):
    """Mirrors inference_status."""

    OK = 0
    INVALID_ARGUMENT = 1
    LOAD_FAILED = 2
    RUNTIME = 3
    OUT_OF_RANGE = 4
    NOT_AVAILABLE = 5
    INTERNAL = 6
    INVALID_REQUEST = 7
    UNAVAILABLE = 8
    NOT_FOUND = 9


class InferenceError(Exception):
    """A native call returned a non-OK status; `detail` is, for engine calls, the protocol's error JSON."""

    def __init__(self, status: int, detail: str, operation: str):
        self.status = Status(status) if status in Status._value2member_map_ else status
        self.detail = detail
        self.operation = operation
        description = borrowed(lib.inference_status_string(status))
        super().__init__(
            f"{operation}: {detail} ({description})"
            if detail
            else f"{operation}: {description}"
        )

    @property
    def code(self):
        """The error JSON's `code` (OpenAI envelope) or `type` (Anthropic envelope), if it has one."""
        try:
            error = json.loads(self.detail).get("error")
        except (ValueError, AttributeError):
            return None
        if not isinstance(error, dict):
            return None
        code = error.get("code")
        if isinstance(code, str):
            return code
        kind = error.get("type")
        return kind if isinstance(kind, str) else None


def check(status: int, operation: str) -> None:
    """Raises unless OK, reading the thread-local detail on the thread that made the call."""
    if status != Status.OK:
        raise InferenceError(status, borrowed(lib.inference_last_error()), operation)
