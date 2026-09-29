//! The wire protocol shared by the engine, the HTTP server and the C ABI: request options, response bodies, tool
//! calls and their parsers, reasoning parsers and files. It builds without candle, so it compiles early.

pub mod files;
#[cfg(feature = "openai")]
pub mod openai;
pub mod reasoning_parsers;
pub mod request;
pub mod response;
#[cfg(feature = "openai")]
pub mod responses_types;
pub mod tools;

pub use inference_mcp::{AgentPermission, CodeExecutionPermission, Function, Tool, ToolType};
