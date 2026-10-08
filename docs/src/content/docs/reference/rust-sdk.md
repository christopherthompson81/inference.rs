---
title: Rust SDK reference
description: The Model API surface of the inference crate, with signatures and links to runnable examples.
---

The `inference` crate is a Rust layer over the engine the server and the C ABI use. Every builder (`ModelBuilder`, `TextModelBuilder`, `GgufModelBuilder`, `EmbeddingModelBuilder`, ...) produces an engine spec and loads it into a `Model`. `Model` is a cheap handle: `Clone` shares the loaded engine. This page lists the surface; `cargo doc -p inference --open` builds the full rustdoc, and the [Rust examples](/examples/) are runnable.

`Model` dereferences to the engine (`inference::Engine`), so every engine operation is a method on it: model management, sessions, LoRA adapters, tokenization, files, MCP tools. Those take and return the same request and response types the HTTP API uses. The methods below are the Rust conveniences `Model` adds. A request goes to a specific model when it names one (`RequestBuilder::with_model`); otherwise it goes to the default model.

A builder's `into_spec()` returns the `EngineSpec` and callbacks it would load, for a caller that loads them itself or serves them over HTTP.

## Chat

```rust
async fn chat(&self, message: impl ToString) -> Result<String>
```
Quick one-shot: send a single user message, get the assistant's text reply.

```rust
async fn send_chat_request(&self, request: impl Into<ChatRequest>) -> Result<ChatCompletionResponse>
```
Generate non-streaming. Accepts `TextMessages`, `MultimodalMessages`, `RequestBuilder`, or a raw `ChatCompletionRequest`. Example: [text-generation](/examples/rust/getting-started/text-generation/).

```rust
async fn stream_chat_request(&self, request: impl Into<ChatRequest>) -> Result<ChatEventStream>
```
Generate streaming. The returned stream yields `ChatStreamEvent`s. Guide: [streaming](/guides/rust/streaming/).

## Scoring a prompt

```rust
async fn prompt_logits(&self, request: PromptLogitsRequest) -> Result<PromptLogits, ApiError>
```
One forward pass over a prompt (text or token ids): each token's log-probability, and with `LogitsOutput::Logits` the row-major logits. Example: [perplexity](/examples/rust/advanced/perplexity/).

## Reasoning

`ReasoningEffort::{Off, Low, Medium, High, XHigh}` is accepted by `TextMessages::with_reasoning_effort`, `MultimodalMessages::with_reasoning_effort` and `RequestBuilder::with_reasoning_effort`. `enable_thinking(bool)` is available on the same three. Leaving both out leaves the effort unspecified with thinking enabled; contradictory explicit controls return a request-validation error.

```rust
let messages = TextMessages::new()
    .add_message(TextMessageRole::User, "Solve this carefully.")
    .with_reasoning_effort(ReasoningEffort::High);
```

The effort is passed to the model's chat template; it does not change sampling parameters directly.

## Structured output

```rust
async fn generate_structured<T>(&self, request: impl Into<RequestBuilder>) -> Result<T>
where T: DeserializeOwned + JsonSchema
```
Constrains generation to the JSON schema derived from `T` (via `schemars`), then deserializes the reply into `T`. Example: [structured](/examples/rust/cookbook/structured/).

## Tools and agents

The engine runs the tool loop. Give the builder tools (a `#[tool]` function's `*_tool_with_callback()`, MCP servers, code execution, the shell), and a chat request runs every call the model makes, round after round, up to `max_tool_rounds`:

```rust
let model = ModelBuilder::new("Qwen/Qwen3-4B")
    .with_tool(get_weather_tool_with_callback())
    .with_code_execution(CodeExecutionConfig::default())
    .with_max_tool_rounds(6)
    .build()
    .await?;

let request = RequestBuilder::from(messages).with_code_execution();
let response = model.send_chat_request(request).await?;
```

The response's `agentic_tool_calls` records each call; a stream carries `AgenticToolCallProgress` events. Tools can also be registered after load with `model.register_tool(...)` and offered per request with `RequestBuilder::with_host_tool(name)`. `with_input_file(InputFile::from_text(...))` attaches request files, `with_shell_execution()` offers the shell, and `with_shell_skill(id)` mounts a skill uploaded to the engine's skill store. With `AgentPermission::Ask`, `with_agent_approval_callback` answers each approval in process. Guides: [build an agent](/guides/agents/build-an-agent/), [file inputs](/guides/agents/file-inputs/), [code execution](/guides/agents/enable-code-execution/), [shell execution](/guides/agents/enable-shell/), [OpenAI-compatible Skills](/guides/agents/skills/).

## Logits processors

`RequestBuilder::add_logits_processor(processor)` runs a processor on one request's logits each step, after the penalties; `inference::in_place(|logits, context| ...)` builds one from a closure over the `f32` logits. `model.register_logits_processor(name, processor)` makes one selectable by name from any request. Example: [logits processor](/examples/rust/advanced/logits-processor/).

## Embeddings

```rust
async fn generate_embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse>
```
One embedding per input, in order; build the request with `EmbeddingRequestBuilder`. Example: [embeddings](/examples/rust/advanced/embeddings/).

```rust
async fn generate_embedding(&self, text: impl Into<String>) -> Result<Vec<f32>>
```
Single-input convenience wrapper.

## Image and speech generation

```rust
async fn generate_image(&self, request: ImageGenerationRequest) -> Result<ImageGenerationResponse>
async fn generate_speech(&self, request: SpeechGenerationRequest) -> Result<SpeechAudio>
```
Diffusion image generation and text to speech, taking the same requests as `/v1/images/generations` and `/v1/audio/speech`. Examples: [diffusion](/examples/rust/models/diffusion/), [speech](/examples/rust/models/speech/).

## Quantization

Through the engine: `re_isq(ReIsqRequest)` requantizes a model loaded with [ISQ (in-situ quantization)](/reference/quantization-types/) to another type, and `calibration(CalibrationAction::{Start, Status, Apply}, model)` runs online calibration (the model must be loaded with ISQ). Guide: [online calibration](/guides/quantization/online-calibration/); example: [online-calibration](/examples/rust/quantization/online-calibration/).

## Tokenization

Through the engine: `tokenize(TokenizeRequest)` tokenizes text, `tokenize_chat(ChatCompletionRequest)` tokenizes a chat as its template renders it (tools, reasoning controls and generation prompt included), and `detokenize(DetokenizeRequest)` reverses either.

## Management

Through the engine: `models()`, `add_model(ModelSpec)`, `remove_model`, `unload_model`, `reload_model`, `model_status`, `set_default_model`, `add_model_alias`; sessions (`sessions`, `session`, `put_session`, `fork_session`, `delete_session`); LoRA adapters; files (`files`, `file_content`); `mcp_tools()`. A `MultiModelBuilder` loads several models into one engine. Example: [multi-model](/examples/rust/advanced/multi-model/); guide: [sessions](/guides/agents/persist-sessions/).
