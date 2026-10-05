# inference — Blazing-Fast LLM Inference in Rust

The Rust SDK for [inference.rs](https://github.com/christopherthompson81/inference.rs), a high-performance
LLM inference engine supporting text, multimodal, speech, image generation, and embedding models.

[GitHub](https://github.com/christopherthompson81/inference.rs) | [Examples](https://github.com/christopherthompson81/inference.rs/tree/master/examples/rust)

## Quick Start

```rust
use inference::{IsqBits, ModelBuilder, TextMessages, TextMessageRole};

#[tokio::main]
async fn main() -> inference::error::Result<()> {
    let model = ModelBuilder::new("Qwen/Qwen3-4B")
        .with_auto_isq(IsqBits::Four)
        .build()
        .await?;

    let response = model.chat("What is Rust's ownership model?").await?;
    println!("{response}");
    Ok(())
}
```

## Capabilities

| Capability | Builder | Example |
|---|---|---|
| Any model (auto-detect) | `ModelBuilder` | `examples/rust/getting_started/text_generation/` |
| Text generation | `TextModelBuilder` | `examples/rust/getting_started/text_generation/` |
| Multimodal (image+text) | `MultimodalModelBuilder` | `examples/rust/getting_started/multimodal/` |
| GGUF quantized models | `GgufModelBuilder` | `examples/rust/getting_started/gguf/` |
| Image generation | `DiffusionModelBuilder` | `examples/rust/models/diffusion/` |
| Speech synthesis | `SpeechModelBuilder` | `examples/rust/models/speech/` |
| Embeddings | `EmbeddingModelBuilder` | `examples/rust/getting_started/embedding/` |
| Structured output | `Model::generate_structured` | `examples/rust/advanced/json_schema/` |
| Tool calling | `Tool`, `ToolChoice` | `examples/rust/advanced/tools/` |
| Agents (the engine's tool loop) | `with_tool`, `with_max_tool_rounds` | `examples/rust/advanced/agent/` |
| LoRA / X-LoRA | `LoraModelBuilder`, `XLoraModelBuilder` | `examples/rust/advanced/lora/` |
| AnyMoE | `AnyMoeModelBuilder` | `examples/rust/advanced/anymoe/` |
| MCP client | `McpClientConfig` | `examples/rust/advanced/mcp_client/` |

## Choosing a Request Type

| Type | Use When | Sampling |
|---|---|---|
| `TextMessages` | Simple text-only chat | Deterministic |
| `MultimodalMessages` | Prompt includes images or audio | Deterministic |
| `RequestBuilder` | Tools, logprobs, custom sampling, constraints, adapters, or web search | Configurable |

`TextMessages` and `MultimodalMessages` convert into `RequestBuilder` via `Into<RequestBuilder>`. Every request decodes greedily (top-k 1) unless `set_sampler_topk` raises it; temperature and top-p only matter then.

## Feature Flags

| Flag | Effect |
|---|---|
| `cuda` | CUDA GPU support |
| `cudnn` | cuDNN for candle's convolutions (requires `cuda`; slower than the default, not recommended) |
| `nccl` | Multi-GPU via NCCL (requires `cuda` and NCCL) |
| `metal` | Apple Metal GPU support |
| `accelerate` | Apple Accelerate framework |
| `mkl` | Intel MKL acceleration |

The default feature set (no flags) builds with pure Rust — no C compiler or system libraries required.

## License

MIT
