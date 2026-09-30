//! Runs tool-calling chats (callbacks, MCP, code execution, files, web search) over an engine's request channel.

mod agentic_loop;
mod file_tools;
mod rag;
mod search;
mod tool_dispatch;

use std::{future::Future, pin::Pin, sync::Arc};

use inference_core::{AgentRunner, Engine, NormalRequest};

struct AgentLoop;

impl AgentRunner for AgentLoop {
    fn run(
        &self,
        engine: Arc<Engine>,
        request: NormalRequest,
    ) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        Box::pin(agentic_loop::agentic_loop(engine, request))
    }
}

/// The runner an engine builder installs so the engine can serve agentic requests.
pub fn runner() -> Arc<dyn AgentRunner> {
    Arc::new(AgentLoop)
}
