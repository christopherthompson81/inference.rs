# inference_rs over ctypes

A pure-Python package for `libinference_ffi`, the inference.rs C ABI (`crates/inference-ffi/include/inference.h`).
It needs no compiled extension: `ctypes` loads the library, and the package checks its ABI version on first use,
refusing any other while the ABI is 0.0.x.

- `Engine` loads a model from a JSON spec and serves the requests the HTTP server does, as JSON in and JSON out:
  chat (with media attachments), completions, embeddings, Anthropic Messages, Responses, model and LoRA adapter
  management, image and speech generation, files, skills and agent approvals. Streaming calls return a `Stream` of
  `StreamEvent`s. Failures raise `InferenceError`, whose `detail` is the protocol's error JSON and `code` its code.
- `HostCallbacks` registers host tools and a search backend when an engine loads.
- `LayoutModel` runs PP-DocLayoutV3 document layout detection. `system_info()` and `system_doctor()` need no engine.

Engine calls release the GIL, so an engine can serve several threads at once.

## Building

```bash
cargo build --release -p inference-ffi   # or a debug build; add --features cuda for CUDA
pip install -e bindings/python
```

The library is found through `INFERENCE_NATIVE_DIR`, then `target/release` and `target/debug` of the checkout the
package sits in, then the platform's own search.

```python
import json
import inference_rs as ir

with ir.Engine(
    json.dumps({"model": {"Plain": {"model_id": "Qwen/Qwen3-0.6B"}}})
) as engine:
    request = {"model": "default", "messages": [{"role": "user", "content": "Hello"}]}
    print(
        json.loads(engine.chat(json.dumps(request)))["choices"][0]["message"]["content"]
    )
    with engine.chat_stream(json.dumps({**request, "stream": True})) as stream:
        for event in stream:
            print(event.data["choices"][0]["delta"].get("content") or "", end="")
```

## Tests

`scripts/local_ci.sh --bindings` runs `tests/` with `unittest`: every header entry point is declared with its
parameter count, at the header's ABI version and exported by the built library, and the engine test drives the tiny
random-weight test checkpoint.
