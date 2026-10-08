"""One-to-one ctypes declarations for inference.h; ownership and errors belong to the wrappers."""

import ctypes
import os
import sys
import threading
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    c_char_p,
    c_float,
    c_int32,
    c_int64,
    c_size_t,
    c_uint32,
    c_void_p,
)
from pathlib import Path

NATIVE_DIR_VARIABLE = "INFERENCE_NATIVE_DIR"
BUNDLED_DIR = "_lib"
CUDA_PATH_VARIABLE = "CUDA_PATH"
CUDA_DLL_DIRS = ("bin", "bin/x64")
# Release first: a consumer that built it has the fast one; tests fall back to the dev build.
PROFILES = ("release", "debug")

status = c_int32


class BackendConfig(Structure):
    _fields_ = [("backend", c_char_p), ("device", c_int32), ("threads", c_int32)]


class Image(Structure):
    _fields_ = [
        ("pixels", c_void_p),
        ("width", c_uint32),
        ("height", c_uint32),
        ("stride", c_uint32),
        ("format", c_int32),
    ]


class Media(Structure):
    _fields_ = [("data", c_void_p), ("len", c_size_t), ("mime_type", c_char_p)]


class SkillFile(Structure):
    _fields_ = [("path", c_char_p), ("data", c_void_p), ("len", c_size_t)]


TOOL_CALLBACK = CFUNCTYPE(None, c_void_p, c_char_p, c_void_p, c_size_t, c_void_p, c_size_t, c_void_p)
SEARCH_CALLBACK = CFUNCTYPE(None, c_void_p, c_void_p, c_size_t, c_void_p)
LOGITS_PROCESSOR_CALLBACK = CFUNCTYPE(c_int32, c_void_p, POINTER(c_float), c_size_t, POINTER(c_uint32), c_size_t)


class HostTool(Structure):
    _fields_ = [
        ("definition", c_void_p),
        ("definition_len", c_size_t),
        ("callback", TOOL_CALLBACK),
        ("user_data", c_void_p),
    ]


class HostCallbacks(Structure):
    _fields_ = [
        ("tools", POINTER(HostTool)),
        ("tool_count", c_size_t),
        ("search", SEARCH_CALLBACK),
        ("search_user_data", c_void_p),
    ]


out = POINTER(c_void_p)
# Inputs cross as a pointer and a length; c_char_p accepts bytes and never passes NULL for b"".
buffer = (c_char_p, c_size_t)

# BEGIN GENERATED from inference.h by bindings/scripts/generate_native.py
ABI_VERSION = (0 << 16) | (0 << 8) | 21

# name: (restype, argtypes)
SIGNATURES = {
    "inference_abi_version": (c_uint32, ()),
    "inference_build_version": (c_void_p, ()),
    "inference_last_error": (c_void_p, ()),
    "inference_status_string": (c_void_p, (c_int32,)),
    "inference_layout_model_load": (status, (c_char_p, POINTER(BackendConfig), out)),
    "inference_layout_model_free": (None, (c_void_p,)),
    "inference_layout_model_label_count": (c_size_t, (c_void_p,)),
    "inference_layout_model_label": (status, (c_void_p, c_size_t, out)),
    "inference_layout_detect": (status, (c_void_p, POINTER(Image), c_float, out)),
    "inference_layout_detect_batch": (status, (c_void_p, POINTER(Image), c_size_t, c_float, out)),
    "inference_layout_result_free": (None, (c_void_p,)),
    "inference_layout_result_count": (c_size_t, (c_void_p,)),
    "inference_layout_result_detection": (
        status,
        (c_void_p, c_size_t, POINTER(c_int32), out, POINTER(c_float), POINTER(c_float)),
    ),
    "inference_engine_load": (status, (*buffer, out)),
    "inference_engine_free": (None, (c_void_p,)),
    "inference_engine_for_owner": (status, (c_void_p, *buffer, out)),
    "inference_callback_result_set": (None, (c_void_p, *buffer)),
    "inference_callback_result_fail": (None, (c_void_p, c_char_p)),
    "inference_engine_load_with_callbacks": (status, (*buffer, POINTER(HostCallbacks), out)),
    "inference_engine_register_tool": (status, (c_void_p, POINTER(HostTool))),
    "inference_engine_unregister_tool": (status, (c_void_p, *buffer)),
    "inference_engine_register_logits_processor": (status, (c_void_p, *buffer, LOGITS_PROCESSOR_CALLBACK, c_void_p)),
    "inference_engine_unregister_logits_processor": (status, (c_void_p, *buffer)),
    "inference_chat": (status, (c_void_p, *buffer, out)),
    "inference_chat_with_media": (status, (c_void_p, *buffer, POINTER(Media), c_size_t, out)),
    "inference_chat_stream_open": (status, (c_void_p, *buffer, out)),
    "inference_chat_stream_open_with_media": (status, (c_void_p, *buffer, POINTER(Media), c_size_t, out)),
    "inference_stream_next": (status, (c_void_p, c_int64, out, POINTER(c_int32))),
    "inference_stream_cancel": (status, (c_void_p,)),
    "inference_stream_free": (None, (c_void_p,)),
    "inference_completion": (status, (c_void_p, *buffer, out)),
    "inference_completion_stream_open": (status, (c_void_p, *buffer, out)),
    "inference_embeddings": (status, (c_void_p, *buffer, out)),
    "inference_anthropic_messages": (status, (c_void_p, *buffer, out)),
    "inference_anthropic_messages_stream_open": (status, (c_void_p, *buffer, out)),
    "inference_anthropic_count_tokens": (status, (c_void_p, *buffer, out)),
    "inference_responses_create": (status, (c_void_p, *buffer, out)),
    "inference_responses_stream_open": (status, (c_void_p, *buffer, out)),
    "inference_responses_get": (status, (c_void_p, *buffer, out)),
    "inference_responses_delete": (status, (c_void_p, *buffer, out)),
    "inference_responses_cancel": (status, (c_void_p, *buffer, out)),
    "inference_models_list": (status, (c_void_p, out)),
    "inference_model_served": (status, (c_void_p, *buffer, out)),
    "inference_mcp_tools_list": (status, (c_void_p, out)),
    "inference_models_cache_stats": (status, (c_void_p, out)),
    "inference_models_speculative_stats": (status, (c_void_p, out)),
    "inference_model_unload": (status, (c_void_p, *buffer, out)),
    "inference_model_reload": (status, (c_void_p, *buffer, out)),
    "inference_model_status": (status, (c_void_p, *buffer, out)),
    "inference_model_add": (status, (c_void_p, *buffer, out)),
    "inference_model_remove": (status, (c_void_p, *buffer, out)),
    "inference_model_set_default": (status, (c_void_p, *buffer, out)),
    "inference_model_alias": (status, (c_void_p, *buffer, out)),
    "inference_lora_adapters_list": (status, (c_void_p, *buffer, out)),
    "inference_lora_adapter_load": (status, (c_void_p, *buffer, out)),
    "inference_lora_adapter_unload": (status, (c_void_p, *buffer, out)),
    "inference_image_generation": (status, (c_void_p, *buffer, out)),
    "inference_prompt_logits": (status, (c_void_p, *buffer, out, out)),
    "inference_speech_generation": (status, (c_void_p, *buffer, out)),
    "inference_approval_resolve": (status, (c_void_p, *buffer, *buffer, out)),
    "inference_file_upload": (status, (c_void_p, *buffer, c_char_p, c_char_p, c_char_p, out)),
    "inference_files_list": (status, (c_void_p, out)),
    "inference_file_get": (status, (c_void_p, *buffer, out)),
    "inference_file_delete": (status, (c_void_p, *buffer, out)),
    "inference_file_content": (status, (c_void_p, *buffer, out)),
    "inference_container_files_list": (status, (c_void_p, *buffer, out)),
    "inference_container_file_get": (status, (c_void_p, *buffer, *buffer, out)),
    "inference_container_file_content": (status, (c_void_p, *buffer, *buffer, out)),
    "inference_skill_upload": (status, (c_void_p, POINTER(SkillFile), c_size_t, out)),
    "inference_skill_version_upload": (status, (c_void_p, *buffer, POINTER(SkillFile), c_size_t, out)),
    "inference_skills_list": (status, (c_void_p, out)),
    "inference_skill_versions_list": (status, (c_void_p, *buffer, out)),
    "inference_re_isq": (status, (c_void_p, *buffer, out)),
    "inference_calibration_start": (status, (c_void_p, *buffer, out)),
    "inference_calibration_status": (status, (c_void_p, *buffer, out)),
    "inference_calibration_apply": (status, (c_void_p, *buffer, out)),
    "inference_sessions_list": (status, (c_void_p, out)),
    "inference_session_get": (status, (c_void_p, *buffer, out)),
    "inference_session_put": (status, (c_void_p, *buffer, *buffer, out)),
    "inference_session_delete": (status, (c_void_p, *buffer, out)),
    "inference_session_fork": (status, (c_void_p, *buffer, *buffer, out)),
    "inference_tokenize": (status, (c_void_p, *buffer, out)),
    "inference_detokenize": (status, (c_void_p, *buffer, out)),
    "inference_tokenize_chat": (status, (c_void_p, *buffer, out)),
    "inference_system_info": (status, (out,)),
    "inference_system_doctor": (status, (out,)),
    "inference_model_tune": (status, (*buffer, out)),
    "inference_blob_data": (c_void_p, (c_void_p,)),
    "inference_blob_len": (c_size_t, (c_void_p,)),
    "inference_blob_mime_type": (c_void_p, (c_void_p,)),
    "inference_blob_free": (None, (c_void_p,)),
    "inference_string_data": (c_void_p, (c_void_p,)),
    "inference_string_len": (c_size_t, (c_void_p,)),
    "inference_string_free": (None, (c_void_p,)),
}
# END GENERATED


def _file_name() -> str:
    if sys.platform == "win32":
        return "inference_ffi.dll"
    if sys.platform == "darwin":
        return "libinference_ffi.dylib"
    return "libinference_ffi.so"


def search_paths():
    """INFERENCE_NATIVE_DIR, then the library a wheel bundles, then the target/ of the checkout this package sits in."""
    configured = os.environ.get(NATIVE_DIR_VARIABLE)
    if configured:
        yield Path(configured) / _file_name()
    yield Path(__file__).resolve().parent / BUNDLED_DIR / _file_name()
    for parent in Path(__file__).resolve().parents:
        # The checkout's root; an installed package has none, and probing every parent would be guesswork.
        if (parent / "Cargo.toml").is_file():
            for profile in PROFILES:
                yield parent / "target" / profile / _file_name()
            return


def _load() -> ctypes.CDLL:
    if sys.platform == "win32" and os.environ.get(CUDA_PATH_VARIABLE):
        # A library loaded by path finds its own DLLs only in the directories added here, not on PATH.
        for directory in CUDA_DLL_DIRS:
            if (Path(os.environ[CUDA_PATH_VARIABLE]) / directory).is_dir():
                os.add_dll_directory(str(Path(os.environ[CUDA_PATH_VARIABLE]) / directory))
    for path in search_paths():
        if path.is_file():
            return ctypes.CDLL(str(path))
    return ctypes.CDLL(_file_name())


def _bind(library: ctypes.CDLL) -> ctypes.CDLL:
    for name, (restype, argtypes) in SIGNATURES.items():
        function = getattr(library, name)
        function.restype = restype
        function.argtypes = list(argtypes)
    return library


class _Library:
    """Loads and checks the library on first use, so importing the package never needs it."""

    _library = None
    _lock = threading.Lock()

    def __getattr__(self, name):
        with _Library._lock:
            if _Library._library is None:
                _Library._library = _checked()
        return getattr(_Library._library, name)


def _checked() -> ctypes.CDLL:
    library = _bind(_load())
    actual = library.inference_abi_version()
    if actual != ABI_VERSION:
        raise RuntimeError(
            f"libinference_ffi implements ABI {_describe(actual)}; this package needs {_describe(ABI_VERSION)}"
        )
    return library


def _describe(version: int) -> str:
    return f"{version >> 16}.{(version >> 8) & 0xFF}.{version & 0xFF}"


lib = _Library()


def borrowed(pointer) -> str:
    """Copies a borrowed NUL-terminated string; the ABI owns every const char * it returns."""
    return ctypes.string_at(pointer).decode("utf-8") if pointer else ""
