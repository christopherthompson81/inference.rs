//! Chat protocols a server offers on top of the engine: each translates its requests to the engine's chat
//! operation and the chat responses back. Server-level work, so not part of `inference_api::Engine` or the C ABI.

pub mod anthropic;
