# inference_rs over ctypes

A pure-Python package for `libinference_ffi`, the inference.rs C ABI (`crates/inference-ffi/include/inference.h`).
It needs no compiled extension: `ctypes` loads the library, and the package checks its ABI version on first use,
refusing any other while the ABI is 0.0.x.

- `Engine` loads a model from its spec and serves the requests the HTTP server does: chat (with media attachments),
  completions, embeddings, Anthropic Messages, Responses, model and LoRA adapter management, image and speech
  generation, files, skills and agent approvals. Requests and responses are the dataclasses in `inference_rs.types`,
  generated from the server's OpenAPI document, so they match what the engine accepts (a request may also be its JSON
  string). Streaming calls return a `Stream` of `StreamEvent`s whose data is parsed the same way. Failures raise
  `InferenceError`, whose `detail` is the protocol's error JSON and `code` its code.
- `engine.json` (a `JsonEngine`) serves the same operations as JSON strings.
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
import inference_rs as ir
from inference_rs import types as t

with ir.Engine({"model": {"Plain": {"model_id": "Qwen/Qwen3-0.6B"}}}) as engine:
    request = t.ChatCompletionRequest(
        model="default", messages=[t.Message(role="user", content="Hello")]
    )
    print(engine.chat(request).choices[0].message.content)
    request.stream = True
    with engine.chat_stream(request) as stream:
        for event in stream:
            print(event.data.choices[0].delta.content or "", end="")
```

## Tests

`scripts/local_ci.sh --bindings` runs `tests/` with `unittest`: every header entry point is declared with its C types,
at the header's ABI version and exported by the built library; `types.py` matches the OpenAPI document; and the engine
tests drive the tiny random-weight test checkpoint through both engines.

`inference_rs/types.py` is generated: after `docs/openapi.json` changes, run
`python3 bindings/python/scripts/generate_types.py`.
