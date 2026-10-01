"""inference.rs over its C ABI (libinference_ffi), with typed requests and responses from `inference_rs.types`."""

from . import types
from ._callbacks import HostCallbacks, HostTool, HostToolCall, LogitsProcessor
from ._codec import from_data, from_json, to_data, to_json
from ._engine import (
    JsonEngine,
    MediaAttachment,
    SkillFile,
    Stream,
    StreamEvent,
    system_doctor,
    system_info,
    tune_model,
)
from ._errors import InferenceError, Status
from ._layout import (
    DEFAULT_THRESHOLD,
    LayoutDetection,
    LayoutImage,
    LayoutModel,
    PixelFormat,
)
from ._owned import Blob
from ._typed import Engine

__all__ = [
    "DEFAULT_THRESHOLD",
    "Blob",
    "Engine",
    "HostCallbacks",
    "HostTool",
    "HostToolCall",
    "InferenceError",
    "JsonEngine",
    "LayoutDetection",
    "LayoutImage",
    "LayoutModel",
    "LogitsProcessor",
    "MediaAttachment",
    "PixelFormat",
    "SkillFile",
    "Status",
    "Stream",
    "StreamEvent",
    "from_data",
    "from_json",
    "system_doctor",
    "system_info",
    "to_data",
    "to_json",
    "tune_model",
    "types",
]
