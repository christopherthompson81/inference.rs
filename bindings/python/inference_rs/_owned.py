import ctypes
from dataclasses import dataclass

from ._native import borrowed, lib


@dataclass(frozen=True)
class Blob:
    """Bytes the engine returned with their MIME type: generated speech, or a file's content."""

    data: bytes
    mime_type: str


def take_string(handle) -> str:
    try:
        return ctypes.string_at(
            lib.inference_string_data(handle), lib.inference_string_len(handle)
        ).decode("utf-8")
    finally:
        lib.inference_string_free(handle)


def take_blob(handle) -> Blob:
    try:
        size = lib.inference_blob_len(handle)
        data = ctypes.string_at(lib.inference_blob_data(handle), size) if size else b""
        return Blob(data, borrowed(lib.inference_blob_mime_type(handle)))
    finally:
        lib.inference_blob_free(handle)
