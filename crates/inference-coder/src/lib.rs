//! A terminal coding agent for inference.rs. The harness owns the tool loop: it sends the conversation with its tool
//! definitions, runs the tool calls that come back on the user's machine (asking first where the permission mode says
//! to) and sends the results, over the engine in this process or any OpenAI-compatible server.

pub mod activity;
pub mod agent;
pub mod backend;
pub mod changes;
pub mod permission;
pub mod prompt;
pub mod session;
pub mod tools;
pub mod turn;

pub use agent::{
    Agent, AgentConfig, AgentUi, DEFAULT_MAX_REPLY_TOKENS, DEFAULT_MAX_ROUNDS, Interrupt, TurnEnd,
};
pub use backend::{ChatBackend, EngineBackend, HttpBackend};
pub use changes::Changes;
pub use permission::{Approval, PermissionMode, Permissions};
pub use session::{Session, SessionInfo};
pub use tools::{Tool, ToolContext, ToolKind, ToolOutcome, default_tools};
pub use turn::{ToolCall, Turn};
