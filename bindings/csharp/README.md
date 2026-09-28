# InferenceRs for .NET

C# bindings for `libinference_ffi`, the inference.rs C ABI (`crates/inference-ffi/include/inference.h`).

- `InferenceEngine` loads a model from a JSON spec and serves the requests the HTTP server does, as JSON in and JSON
  out: chat (with media attachments), completions, embeddings, Anthropic Messages, Responses, model and LoRA adapter
  management, image and speech generation, files, skills, agent approvals and system reports. Streaming calls return
  an `EngineStream` of `StreamEvent`s. Failures throw `InferenceException`, whose `Detail` is the protocol's error JSON.
- `HostCallbacks` registers host tools and a search backend when an engine loads.
- `LayoutModel` runs PP-DocLayoutV3 document layout detection.

The bindings check the library's ABI version on first use and refuse any other: the ABI is unstable while it is
0.0.x.

## Building

```bash
cargo build --release -p inference-ffi   # or a debug build; add --features cuda for CUDA
dotnet build bindings/csharp/InferenceRs.slnx
```

The library is found through `INFERENCE_NATIVE_DIR`, then `target/release` and `target/debug` of the checkout the
bindings sit in, then the platform's own search.

```csharp
using InferenceRs;

using var engine = InferenceEngine.Load("""{"model": {"Plain": {"model_id": "Qwen/Qwen3-0.6B"}}}""");
var reply = engine.Chat("""{"model": "default", "messages": [{"role": "user", "content": "Hello"}]}""");
foreach (var streamEvent in engine.ChatStream("""{"model": "default", "messages": [{"role": "user", "content": "Hi"}]}"""))
{
    Console.WriteLine(streamEvent.Data);
}
```

## Tests

`scripts/local_ci.sh --bindings` builds the library, writes the tiny random-weight test checkpoint and runs
`tests/InferenceRs.BindingCoverage` (every header entry point is bound, at the header's ABI version) and
`tests/InferenceRs.EngineTest`.
