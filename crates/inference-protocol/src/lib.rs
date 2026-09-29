//! The wire protocol shared by the engine, the HTTP server and the C ABI: request options, response bodies, tool
//! calls and their parsers, reasoning parsers and files. It builds without candle, so it compiles early.

pub mod files;
pub mod reasoning_parsers;
pub mod request;
pub mod response;
pub mod tools;

pub use inference_mcp::{Function, Tool, ToolType};
