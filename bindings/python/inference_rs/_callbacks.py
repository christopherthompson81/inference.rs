"""Host callbacks: Python functions the agent loop calls, found by the id the native side passes back."""

import ctypes
import itertools
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from . import _native
from ._native import lib

ENGINE_GONE = "the engine that registered this callback has been freed"


@dataclass(frozen=True)
class HostToolCall:
    name: str
    arguments_json: str
    session_id: str | None
    round: int | None


@dataclass(frozen=True)
class HostTool:
    """A host tool: its OpenAI function definition, and a handler whose return (or exception) the model sees.

    Handlers run on engine worker threads, possibly several at once, and must not call back into the engine.
    """

    definition_json: str
    handler: Callable[[HostToolCall], str]


@dataclass
class HostCallbacks:
    """Host functions fixed when an engine loads; `search` answers with a JSON array of {title, description, url, content}."""

    tools: list = field(default_factory=list)
    search: Callable[[str], str] | None = None


# An id, not a pointer to a Python object: requests still finishing may call a callback after the engine is freed,
# and a removed id fails that call instead of reaching a collected handler.
_handlers = {}
_lock = threading.Lock()
_ids = itertools.count(1)


def _register(handler) -> int:
    with _lock:
        handler_id = next(_ids)
        _handlers[handler_id] = handler
        return handler_id


def unregister(ids) -> None:
    with _lock:
        for handler_id in ids:
            _handlers.pop(handler_id, None)


def _find(user_data):
    with _lock:
        return _handlers.get(user_data or 0)


def _answer(result, text: str) -> None:
    data = text.encode("utf-8")
    lib.inference_callback_result_set(result, data, len(data))


def _fail(result, error) -> None:
    """Reports a failure; str() on an exception can itself raise, and nothing may raise through a C callback."""
    try:
        message = str(error).encode("utf-8", "replace")
    except BaseException:  # noqa: BLE001
        message = None
    lib.inference_callback_result_fail(result, message)


def _context(json_text: str):
    context = json.loads(json_text)
    session_id = context.get("session_id")
    round_number = context.get("round")
    return (
        session_id if isinstance(session_id, str) else None,
        round_number if isinstance(round_number, int) else None,
    )


@_native.TOOL_CALLBACK
def _tool(user_data, tool_name, arguments, arguments_len, context, context_len, result):
    try:
        handler = _find(user_data)
        if handler is None:
            _fail(result, ENGINE_GONE)
            return
        session_id, round_number = _context(
            ctypes.string_at(context, context_len).decode("utf-8")
        )
        call = HostToolCall(
            tool_name.decode("utf-8"),
            ctypes.string_at(arguments, arguments_len).decode("utf-8"),
            session_id,
            round_number,
        )
        _answer(result, handler(call))
    except BaseException as error:  # noqa: BLE001 - a handler's failure reaches the model, not the engine
        _fail(result, error)


@_native.SEARCH_CALLBACK
def _search(user_data, query, query_len, result):
    try:
        handler = _find(user_data)
        if handler is None:
            _fail(result, ENGINE_GONE)
            return
        _answer(result, handler(ctypes.string_at(query, query_len).decode("utf-8")))
    except BaseException as error:  # noqa: BLE001 - a handler's failure reaches the model, not the engine
        _fail(result, error)


class Registration:
    """The native description of `callbacks` for one load; its ids pass to the engine on success."""

    def __init__(self, callbacks: HostCallbacks):
        self.ids = []
        self._definitions = []
        tools = (_native.HostTool * max(len(callbacks.tools), 1))()
        try:
            for index, tool in enumerate(callbacks.tools):
                definition = ctypes.create_string_buffer(
                    tool.definition_json.encode("utf-8")
                )
                self._definitions.append(definition)
                handler_id = _register(tool.handler)
                self.ids.append(handler_id)
                tools[index] = _native.HostTool(
                    ctypes.cast(definition, ctypes.c_void_p),
                    len(definition) - 1,
                    _tool,
                    handler_id,
                )
            search_id = (
                _register(callbacks.search) if callbacks.search is not None else 0
            )
            if search_id:
                self.ids.append(search_id)
        except BaseException:
            unregister(self.ids)
            raise
        self._tools = tools
        self.native = _native.HostCallbacks(
            tools,
            len(callbacks.tools),
            _search if search_id else _native.SEARCH_CALLBACK(),
            search_id or None,
        )
