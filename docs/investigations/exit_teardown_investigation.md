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

## Run 2 - 2026-11-10 01:10

Change: a process-wide exit flag and in-flight count in `inference_tensor::utils`. Every model forward
(`in_precision` in the pipeline step, and CUDA's lookahead step) is counted; once exiting, a new forward parks its
thread rather than start. Parakeet's window loop and the diarizer's chunk loop stop at their next check. A libc
`atexit` handler, registered once a model's engine has loaded (so after the driver's, and run before it), sets the
flag and waits up to 5 s for the count to reach 0.

Command: `scripts/exit_race_check.sh <device> <10-minute recording> 15` (the `stream` mode of Run 1 as a script),
bundle library at the change.

```
             before (Run 1)       after
CUDA         7 of 12 faulted      0 of 15 (and 0 of 30 with the decode thread beside it)
Vulkan       7 of 10 faulted      0 of 15 (and 0 of 30)
```

An exit mid-stream takes 1.6 s in all (load included). A tiny-checkpoint test of the same exit passed with the
handler removed too (5 of 5 and 12 of 12 child runs): its small products never load cuBLAS, whose runtime's exit
handler is what faulted, so the check stays a script on the real checkpoint.
