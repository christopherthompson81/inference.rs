import math
import os
import threading

# ctypes does not range-check c_int64, so longer waits are clamped rather than wrapped.
MAX_TIMEOUT_MS = 2**63 - 1
INFINITE = -1


class Handle:
    """A native object freed by `free` once its owner and every lease and dependent have let go."""

    def __init__(self, pointer, free, on_free=None):
        self._pointer = pointer
        self._free = free
        self._on_free = on_free
        self._references = 1
        self._closed = False
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            if self._closed:
                raise ValueError("the handle is closed")
            self._references += 1
            return self._pointer

    def release(self):
        with self._lock:
            self._references -= 1
            free = self._references == 0
        if free:
            self._free(self._pointer)
            if self._on_free is not None:
                self._on_free()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.release()


class Lease:
    """Holds a handle open for one call, even if another thread closes it meanwhile."""

    def __init__(self, handle: Handle):
        self._handle = handle

    def __enter__(self):
        return self._handle.acquire()

    def __exit__(self, *exc):
        self._handle.release()


def text_arg(value) -> bytes:
    """UTF-8 for a length-carrying input; str or path-like."""
    return os.fspath(value).encode("utf-8")


def c_string_arg(value, name: str):
    """UTF-8 for a NUL-terminated input, which would silently end at an embedded NUL."""
    if value is None:
        return None
    encoded = text_arg(value)
    if b"\0" in encoded:
        raise ValueError(f"{name} contains a NUL character")
    return encoded


def bytes_arg(data) -> bytes:
    """Bytes as the ABI takes them; bytes pass without a copy, other buffers are copied once."""
    return data if isinstance(data, bytes) else bytes(data)


def timeout_ms(timeout) -> int:
    if timeout is None or math.isinf(timeout):
        return INFINITE
    if timeout < 0:
        raise ValueError("timeout must not be negative")
    # Rounded up, so a sub-millisecond wait still waits rather than polling.
    return min(math.ceil(timeout * 1000), MAX_TIMEOUT_MS)
