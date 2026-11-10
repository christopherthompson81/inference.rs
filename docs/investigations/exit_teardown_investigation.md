# Process exit with engine work in flight

The transcription app (C# bindings over the C ABI) saw one segfault and one hang at process exit, after a stream's
`done`, in ~32 runs of a 35-minute recording; neither reproduced in 60 more. Its guess: an engine disposed while
an audio decode call or a stream thread was still alive.

## Run 1 - 2026-11-10 00:30

Question: which of those exits crash, on our side, and where?

Command: `scratchpad/drops/exitrace.py <device> <mode>` (Python bindings, bundle `libinference_ffi`, Parakeet with
VAD, a 10-minute recording), 10-12 runs a mode; exit codes. Modes: `stream` frees the engine three windows into a
stream (the stream itself left open) and exits; `decode` exits with an engine-free audio decode looping on another
thread; `done` exits after `done` with the decode running; `libcN` calls libc `exit` N ms after `done` with the
engine never freed (as a .NET process exits without running finalizers).

```
mode       CUDA (3090)                     Vulkan (3090)
stream     7 of 12 SIGSEGV                 7 of 10 SIGSEGV
decode     0 of 12                         -
both       6 of 12                         -
done       0 of 12                         -
libc0-1000 0 of 40                         0 of 30
```

Backtrace (gdb as parent, `stream` on CUDA): the main thread is in `exit`, in libcudart's exit handler tearing the
runtime down; an `inference-api` thread is inside `cublasGemmStridedBatchedEx` for the stream's next window and
faults. Freeing an engine is synchronous (`teardown.py`: VRAM 2088 -> 346 MiB, the context, when the free returns
after 47 ms), and an idle engine at exit is safe; the stream holds its own engine handle, so freeing the caller's
handle does not stop its work.

Implication: any GPU work still running when the process exits races the driver's exit handlers (CUDA and
Vulkan alike); the decode is innocent. The app's crash after `done` needs some GPU work in flight at its exit,
which its log may show (a second stream, a diarization stream for resumed words). Fix candidates: the caller
cancels its streams and frees the engine before exiting (safe today); and the library stops in-flight work
itself at exit, from an exit handler registered after the driver's (so it runs first) that cancels and waits.
