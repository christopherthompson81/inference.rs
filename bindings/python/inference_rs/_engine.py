import ctypes
import json
import weakref
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from . import _callbacks, _native
from ._errors import check
from ._handle import Handle, Lease, bytes_arg, c_string_arg, text_arg, timeout_ms
from ._native import lib
from ._owned import Blob, take_blob, take_string


@dataclass(frozen=True)
class MediaAttachment:
    """A buffer a request names by position: an image, audio or video URL of media://0 is the first."""

    data: bytes
    mime_type: str | None = None


@dataclass(frozen=True)
class SkillFile:
    """One file of a skill upload: its path within the skill (SKILL.md, scripts/run.py) and bytes."""

    path: str
    data: bytes


@dataclass(frozen=True)
class StreamEvent:
    """A stream event by protocol name; a failed request ends with an `error` event holding the error JSON."""

    name: str
    data: object


class Stream:
    """A streaming request's events; close it, or use `with`, to abandon it. An open stream keeps its engine."""

    def __init__(self, engine: Handle, pointer):
        def free(stream):
            lib.inference_stream_free(stream)
            engine.release()

        self._stream = Handle(pointer, free)
        self._finalizer = weakref.finalize(self, self._stream.close)
        self.done = False
        # Reads an event's data; the typed engine sets it to parse into the protocol's classes.
        self.parse = lambda name, data: data

    def next(self, timeout: float | None = None) -> StreamEvent | None:
        """The next event within `timeout` seconds; None on a timeout, or at the end (then `done` is True)."""
        if self.done:
            return None
        wait = timeout_ms(timeout)
        event = ctypes.c_void_p()
        done = ctypes.c_int32()
        with Lease(self._stream) as stream:
            check(
                lib.inference_stream_next(stream, wait, ctypes.byref(event), ctypes.byref(done)),
                "inference_stream_next",
            )
        if done.value:
            self.done = True
            return None
        if not event.value:
            return None
        envelope = json.loads(take_string(event))
        return StreamEvent(envelope["event"], self.parse(envelope["event"], envelope["data"]))

    def cancel(self):
        """Asks the request to stop; keep reading for its final event, which carries usage. Safe from any thread."""
        with Lease(self._stream) as stream:
            check(lib.inference_stream_cancel(stream), "inference_stream_cancel")

    def __iter__(self) -> Iterator[StreamEvent]:
        while (event := self.next()) is not None:
            yield event

    def close(self):
        self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class _Buffers:
    """Keeps the bytes and strings an array of native structs points at alive for one call."""

    def __init__(self):
        self._keep = []

    def bytes(self, data):
        # The bytes object itself stays alive here, so its buffer is passed without a copy.
        data = bytes_arg(data)
        pointer = ctypes.c_char_p(data)
        self._keep.append((data, pointer))
        return ctypes.cast(pointer, ctypes.c_void_p)

    def text(self, value, name: str):
        encoded = c_string_arg(value, name)
        self._keep.append(encoded)
        return encoded


class JsonEngine:
    """A loaded model serving JSON strings, as the HTTP server takes and returns them; see `Engine` for classes.

    Calls block and release the GIL, so several threads may share an engine. Close it, or use `with`. A host
    callback must not hold the last reference to its engine, which would then be freed on the engine's own thread.
    """

    def __init__(self, spec_json: str, callbacks: _callbacks.HostCallbacks | None = None):
        spec = text_arg(spec_json)
        engine = ctypes.c_void_p()
        if callbacks is None:
            check(
                lib.inference_engine_load(spec, len(spec), ctypes.byref(engine)),
                "inference_engine_load",
            )
            self._handle = Handle(engine.value, lib.inference_engine_free)
            self._finalizer = weakref.finalize(self, self._handle.close)
            return
        registration = _callbacks.Registration(callbacks)
        try:
            check(
                lib.inference_engine_load_with_callbacks(
                    spec,
                    len(spec),
                    ctypes.byref(registration.native),
                    ctypes.byref(engine),
                ),
                "inference_engine_load_with_callbacks",
            )
        except BaseException:
            _callbacks.unregister(registration.ids)
            raise
        ids = registration.ids
        self._handle = Handle(engine.value, lib.inference_engine_free, lambda: _callbacks.unregister(ids))
        self._finalizer = weakref.finalize(self, self._handle.close)

    @staticmethod
    def abi_version() -> int:
        """(major << 16) | (minor << 8) | patch of the ABI the library implements."""
        return lib.inference_abi_version()

    @staticmethod
    def build_version() -> str:
        return _native.borrowed(lib.inference_build_version())

    def close(self):
        self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # One input buffer in, an owned string out: most of the surface.
    def _call(self, name: str, value: str) -> str:
        data = text_arg(value)
        response = ctypes.c_void_p()
        with Lease(self._handle) as engine:
            check(
                getattr(lib, name)(engine, data, len(data), ctypes.byref(response)),
                name,
            )
        return take_string(response)

    def _stream(self, name: str, value: str, *extra) -> Stream:
        data = text_arg(value)
        stream = ctypes.c_void_p()
        with Lease(self._handle) as engine:
            check(
                getattr(lib, name)(engine, data, len(data), *extra, ctypes.byref(stream)),
                name,
            )
            try:
                self._handle.acquire()
            except ValueError:
                # Closed meanwhile: the stream must not outlive the engine's callback registrations.
                lib.inference_stream_free(stream)
                raise
        return Stream(self._handle, stream.value)

    def _get(self, name: str) -> str:
        response = ctypes.c_void_p()
        with Lease(self._handle) as engine:
            check(getattr(lib, name)(engine, ctypes.byref(response)), name)
        return take_string(response)

    def _blob(self, name: str, value: str) -> Blob:
        data = text_arg(value)
        blob = ctypes.c_void_p()
        with Lease(self._handle) as engine:
            check(getattr(lib, name)(engine, data, len(data), ctypes.byref(blob)), name)
        return take_blob(blob)

    @staticmethod
    def _media(media: Sequence[MediaAttachment], buffers: _Buffers):
        array = (_native.Media * max(len(media), 1))()
        for index, item in enumerate(media):
            array[index] = _native.Media(
                buffers.bytes(item.data),
                len(item.data),
                buffers.text(item.mime_type, "mime_type"),
            )
        return (array if media else None), len(media)

    @staticmethod
    def _skill_files(files: Sequence[SkillFile], buffers: _Buffers):
        array = (_native.SkillFile * max(len(files), 1))()
        for index, file in enumerate(files):
            array[index] = _native.SkillFile(
                buffers.text(file.path, "path"),
                buffers.bytes(file.data),
                len(file.data),
            )
        return (array if files else None), len(files)

    def chat(self, request_json: str, media: Sequence[MediaAttachment] = ()) -> str:
        data = text_arg(request_json)
        buffers = _Buffers()
        array, count = self._media(media, buffers)
        response = ctypes.c_void_p()
        with Lease(self._handle) as engine:
            check(
                lib.inference_chat_with_media(engine, data, len(data), array, count, ctypes.byref(response)),
                "inference_chat_with_media",
            )
        return take_string(response)

    def chat_stream(self, request_json: str, media: Sequence[MediaAttachment] = ()) -> Stream:
        buffers = _Buffers()
        array, count = self._media(media, buffers)
        return self._stream("inference_chat_stream_open_with_media", request_json, array, count)

    def completion(self, request_json: str) -> str:
        return self._call("inference_completion", request_json)

    def completion_stream(self, request_json: str) -> Stream:
        return self._stream("inference_completion_stream_open", request_json)

    def embeddings(self, request_json: str) -> str:
        return self._call("inference_embeddings", request_json)

    def anthropic_messages(self, request_json: str) -> str:
        """An Anthropic Messages request; failures carry the Anthropic error envelope."""
        return self._call("inference_anthropic_messages", request_json)

    def anthropic_messages_stream(self, request_json: str) -> Stream:
        return self._stream("inference_anthropic_messages_stream_open", request_json)

    def create_response(self, request_json: str) -> str:
        return self._call("inference_responses_create", request_json)

    def response_stream(self, request_json: str) -> Stream:
        return self._stream("inference_responses_stream_open", request_json)

    def get_response(self, response_id: str) -> str:
        return self._call("inference_responses_get", response_id)

    def delete_response(self, response_id: str) -> str:
        return self._call("inference_responses_delete", response_id)

    def cancel_response(self, response_id: str) -> str:
        return self._call("inference_responses_cancel", response_id)

    def list_models(self) -> str:
        return self._get("inference_models_list")

    def unload_model(self, request_json: str) -> str:
        return self._call("inference_model_unload", request_json)

    def reload_model(self, request_json: str) -> str:
        return self._call("inference_model_reload", request_json)

    def model_status(self, request_json: str) -> str:
        return self._call("inference_model_status", request_json)

    def list_lora_adapters(self, request_json: str = "{}") -> str:
        return self._call("inference_lora_adapters_list", request_json)

    def load_lora_adapter(self, request_json: str) -> str:
        return self._call("inference_lora_adapter_load", request_json)

    def unload_lora_adapter(self, request_json: str) -> str:
        return self._call("inference_lora_adapter_unload", request_json)

    def image_generation(self, request_json: str) -> str:
        return self._call("inference_image_generation", request_json)

    def speech_generation(self, request_json: str) -> Blob:
        """Speaks text; the blob's MIME type carries the sample rate and channel count."""
        return self._blob("inference_speech_generation", request_json)

    def _call2(self, name: str, first: str, second: str) -> str:
        first_data, second_data = text_arg(first), text_arg(second)
        response = ctypes.c_void_p()
        with Lease(self._handle) as engine:
            check(
                getattr(lib, name)(
                    engine,
                    first_data,
                    len(first_data),
                    second_data,
                    len(second_data),
                    ctypes.byref(response),
                ),
                name,
            )
        return take_string(response)

    def resolve_approval(self, approval_id: str, decision_json: str) -> str:
        """Answers the approval an agentic_tool_approval_required stream event named."""
        return self._call2("inference_approval_resolve", approval_id, decision_json)

    def upload_file(self, data: bytes, filename: str, purpose: str, mime_type: str | None = None) -> str:
        data = bytes_arg(data)
        response = ctypes.c_void_p()
        with Lease(self._handle) as engine:
            check(
                lib.inference_file_upload(
                    engine,
                    data,
                    len(data),
                    c_string_arg(filename, "filename"),
                    c_string_arg(mime_type, "mime_type"),
                    c_string_arg(purpose, "purpose"),
                    ctypes.byref(response),
                ),
                "inference_file_upload",
            )
        return take_string(response)

    def re_isq(self, request_json: str) -> str:
        return self._call("inference_re_isq", request_json)

    def calibration_start(self) -> str:
        return self._get("inference_calibration_start")

    def calibration_status(self) -> str:
        return self._get("inference_calibration_status")

    def cache_stats(self) -> str:
        """Each loaded model's cumulative prefix- and encoder-cache counters; diff two readings for a span."""
        return self._get("inference_models_cache_stats")

    def calibration_apply(self, request_json: str = "{}") -> str:
        return self._call("inference_calibration_apply", request_json)

    def list_sessions(self) -> str:
        return self._get("inference_sessions_list")

    def get_session(self, session_id: str) -> str:
        return self._call("inference_session_get", session_id)

    def put_session(self, session_id: str, session_json: str) -> str:
        return self._call2("inference_session_put", session_id, session_json)

    def delete_session(self, session_id: str) -> str:
        return self._call("inference_session_delete", session_id)

    def tokenize(self, request_json: str) -> str:
        return self._call("inference_tokenize", request_json)

    def detokenize(self, request_json: str) -> str:
        return self._call("inference_detokenize", request_json)

    def list_files(self) -> str:
        return self._get("inference_files_list")

    def get_file(self, file_id: str) -> str:
        return self._call("inference_file_get", file_id)

    def delete_file(self, file_id: str) -> str:
        return self._call("inference_file_delete", file_id)

    def file_content(self, file_id: str) -> Blob:
        return self._blob("inference_file_content", file_id)

    def upload_skill(self, files: Sequence[SkillFile]) -> str:
        buffers = _Buffers()
        array, count = self._skill_files(files, buffers)
        response = ctypes.c_void_p()
        with Lease(self._handle) as engine:
            check(
                lib.inference_skill_upload(engine, array, count, ctypes.byref(response)),
                "inference_skill_upload",
            )
        return take_string(response)

    def upload_skill_version(self, skill_id: str, files: Sequence[SkillFile]) -> str:
        buffers = _Buffers()
        array, count = self._skill_files(files, buffers)
        skill = text_arg(skill_id)
        response = ctypes.c_void_p()
        with Lease(self._handle) as engine:
            check(
                lib.inference_skill_version_upload(engine, skill, len(skill), array, count, ctypes.byref(response)),
                "inference_skill_version_upload",
            )
        return take_string(response)

    def list_skills(self) -> str:
        return self._get("inference_skills_list")

    def list_skill_versions(self, skill_id: str) -> str:
        return self._call("inference_skill_versions_list", skill_id)


def _report(name: str) -> str:
    response = ctypes.c_void_p()
    check(getattr(lib, name)(ctypes.byref(response)), name)
    return take_string(response)


def system_info() -> str:
    """Host, device and build information; needs no engine."""
    return _report("inference_system_info")


def system_doctor() -> str:
    """Environment diagnostics; needs no engine."""
    return _report("inference_system_doctor")
