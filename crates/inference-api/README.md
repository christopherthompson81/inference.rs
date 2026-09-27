# `inference-api`

The inference.rs engine surface: load an engine and run OpenAI-style operations on it (chat completions today), with
requests and responses as the OpenAI JSON types and no HTTP. The HTTP server (`inference-server-core`) is built on
it, and the C ABI (`inference-ffi`) is to expose it (#19).
