"""Checks that need no model: the callback trampolines, handle lifetimes, and argument validation."""

import ctypes
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inference_rs as ir
from inference_rs import _callbacks, _handle, _native


def call_tool(user_data, arguments=b"{}", context=b'{"session_id": "s", "round": 2}'):
    # A NULL result is allowed: inference_callback_result_set and _fail ignore it, so only the handler is observed.
    trampoline = _native.TOOL_CALLBACK(ctypes.cast(_callbacks._tool, ctypes.c_void_p).value)
    trampoline(user_data, b"lookup", arguments, len(arguments), context, len(context), None)


class Callbacks(unittest.TestCase):
    def setUp(self):
        # ctypes reports an exception escaping a callback here instead of raising it; any is a failure.
        self.escaped = []
        self.hook = sys.unraisablehook
        sys.unraisablehook = self.escaped.append

    def tearDown(self):
        sys.unraisablehook = self.hook
        self.assertEqual(self.escaped, [])

    def test_a_tool_call_reaches_its_handler_with_its_context(self):
        calls = []
        handler_id = _callbacks._register(lambda call: calls.append(call) or "ok")
        try:
            call_tool(handler_id, b'{"x": 1}')
        finally:
            _callbacks.unregister([handler_id])
        self.assertEqual(calls, [ir.HostToolCall("lookup", '{"x": 1}', "s", 2)])

    def test_unknown_ids_and_failing_handlers_do_not_raise_through_c(self):
        class Unprintable(Exception):
            def __str__(self):
                raise RuntimeError("no message")

        def fail(call):
            raise Unprintable

        handler_id = _callbacks._register(fail)
        try:
            call_tool(handler_id)
            call_tool(handler_id, context=b"not json")
        finally:
            _callbacks.unregister([handler_id])
        call_tool(handler_id)

    def test_a_search_call_reaches_its_handler(self):
        queries = []
        handler_id = _callbacks._register(lambda query: queries.append(query) or "[]")
        trampoline = _native.SEARCH_CALLBACK(ctypes.cast(_callbacks._search, ctypes.c_void_p).value)
        try:
            trampoline(handler_id, b"rust", 4, None)
        finally:
            _callbacks.unregister([handler_id])
        self.assertEqual(queries, ["rust"])


class Handles(unittest.TestCase):
    def test_a_handle_is_freed_after_its_last_reference(self):
        freed = []
        handle = _handle.Handle(7, freed.append, lambda: freed.append("after"))
        handle.acquire()
        handle.close()
        self.assertEqual(freed, [])
        with self.assertRaises(ValueError):
            handle.acquire()
        handle.release()
        self.assertEqual(freed, [7, "after"])
        handle.close()
        self.assertEqual(freed, [7, "after"])


class Arguments(unittest.TestCase):
    def test_a_short_pixel_buffer_is_refused(self):
        ir.LayoutImage(bytes(12), 2, 2, ir.PixelFormat.RGB8)
        ir.LayoutImage(bytes(10 + 6), 2, 2, ir.PixelFormat.RGB8, stride=10)
        with self.assertRaises(ValueError):
            ir.LayoutImage(bytes(11), 2, 2, ir.PixelFormat.RGB8)
        with self.assertRaises(ValueError):
            ir.LayoutImage(bytes(64), 2, 2, ir.PixelFormat.RGBA8, stride=4)

    def test_timeouts_round_up_and_clamp(self):
        self.assertEqual(_handle.timeout_ms(None), _handle.INFINITE)
        self.assertEqual(_handle.timeout_ms(float("inf")), _handle.INFINITE)
        self.assertEqual(_handle.timeout_ms(0), 0)
        self.assertEqual(_handle.timeout_ms(1e-7), 1)
        self.assertEqual(_handle.timeout_ms(1e30), _handle.MAX_TIMEOUT_MS)
        with self.assertRaises(ValueError):
            _handle.timeout_ms(-1)

    def test_c_strings_refuse_embedded_nul(self):
        self.assertEqual(_handle.c_string_arg(Path("a/b"), "path"), b"a/b")
        with self.assertRaises(ValueError):
            _handle.c_string_arg("a\0b", "name")

    def test_error_codes_tolerate_any_detail(self):
        for detail in ("", "[]", '{"error": 1}', '{"error": {"type": 5}}', "not json"):
            self.assertIsNone(ir.InferenceError(7, detail, "op").code)
        self.assertEqual(ir.InferenceError(7, json.dumps({"error": {"code": "c"}}), "op").code, "c")


if __name__ == "__main__":
    unittest.main()
