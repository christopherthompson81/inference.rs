"""inference.rs over its C ABI (libinference_ffi): JSON requests in, JSON responses out."""

from ._callbacks import HostCallbacks, HostTool, HostToolCall
from ._engine import (
    Engine,
    MediaAttachment,
    SkillFile,
    Stream,
    StreamEvent,
    system_doctor,
    system_info,
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

__all__ = [
    "DEFAULT_THRESHOLD",
    "Blob",
    "Engine",
    "HostCallbacks",
    "HostTool",
    "HostToolCall",
    "InferenceError",
    "LayoutDetection",
    "LayoutImage",
    "LayoutModel",
    "MediaAttachment",
    "PixelFormat",
    "SkillFile",
    "Status",
    "Stream",
    "StreamEvent",
    "system_doctor",
    "system_info",
]
