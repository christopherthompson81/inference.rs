//! The seam between the engine and the agent layer (`inference-agent`), which runs tool-calling chats over it.

use std::{future::Future, pin::Pin, sync::Arc};

use tokenizers::Tokenizer;
use tokio::sync::{
    Mutex,
    mpsc::{Receiver, Sender},
};
use tokio::task::JoinHandle;

use super::{Engine, IntervalLogger, agentic_session::AgenticSessionStore};
use crate::{
    EngineConfig, NormalRequest, Pipeline, SchedulerConfig, files::FileStore, get_mut_arcmutex,
    pipeline::Modalities, request::Request, search, tools::ToolCallbacksWithTools,
};

/// Default cap on tool-use rounds when the request doesn't set one.
pub const DEFAULT_MAX_TOOL_ROUNDS: usize = 256;

/// Set on inner probe requests so `handle_request` doesn't re-enter the loop. Distinct from `None` (unset).
pub const AGENTIC_LOOP_REENTRY_SENTINEL: Option<usize> = Some(0);

/// Whether this build registers the code-execution, shell and file tools.
pub const CODE_EXECUTION: bool = cfg!(feature = "code-execution");

/// Runs a chat that uses tools or web search, driving the engine through its request channel.
pub trait AgentRunner: Send + Sync {
    fn run(
        &self,
        engine: Arc<Engine>,
        request: NormalRequest,
    ) -> Pin<Box<dyn Future<Output = ()> + Send>>;
}

pub(crate) struct EngineParts {
    pub tx: Sender<Request>,
    pub rx: Receiver<Request>,
    pub pipeline: Arc<Mutex<dyn Pipeline>>,
    pub scheduler: SchedulerConfig,
    pub engine: EngineConfig,
    pub logger: Arc<IntervalLogger>,
    pub session_store: Arc<std::sync::Mutex<AgenticSessionStore>>,
    pub file_store: FileStore,
}

pub fn is_code_exec_tool(name: &str) -> bool {
    CODE_EXECUTION && inference_code_exec::code_exec_tool_called(name)
}

pub fn is_read_file_tool(name: &str) -> bool {
    CODE_EXECUTION && name == inference_code_exec::READ_FILE_TOOL_NAME
}

pub fn is_list_files_tool(name: &str) -> bool {
    CODE_EXECUTION && name == inference_code_exec::LIST_FILES_TOOL_NAME
}

pub fn is_surface_outputs_tool(name: &str) -> bool {
    CODE_EXECUTION && inference_code_exec::surface_outputs_tool_called(name)
}

pub fn is_shell_tool(name: &str) -> bool {
    CODE_EXECUTION && inference_code_exec::shell_tool_called(name)
}

pub fn registered_tool_active_for_request(
    name: &str,
    enable_code_execution: bool,
    enable_shell: bool,
) -> bool {
    if is_shell_tool(name) {
        enable_shell
    } else if is_code_exec_tool(name) {
        enable_code_execution
    } else {
        true
    }
}

impl Engine {
    pub(super) fn agent_runner(&self) -> Option<&Arc<dyn AgentRunner>> {
        self.agent_runner.as_ref()
    }

    /// The engine's own request channel, which inner agent requests go back through.
    pub fn request_sender(&self) -> &Sender<Request> {
        &self.tx
    }

    pub fn modalities(&self) -> Modalities {
        get_mut_arcmutex!(self.pipeline)
            .get_metadata()
            .modalities
            .clone()
    }

    pub fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        get_mut_arcmutex!(self.pipeline).tokenizer()
    }

    pub fn tool_callbacks(&self) -> &ToolCallbacksWithTools {
        &self.tool_callbacks
    }

    pub fn search_callback(&self) -> Option<&Arc<search::SearchCallback>> {
        self.search_callback.as_ref()
    }

    pub fn with_search_embedder<R>(
        &self,
        f: impl FnOnce(Option<&mut search::SearchEmbedder>) -> R,
    ) -> R {
        f(get_mut_arcmutex!(self.search_embedder).as_mut())
    }

    pub fn session_store(&self) -> &Arc<std::sync::Mutex<AgenticSessionStore>> {
        &self.session_store
    }

    pub fn file_store(&self) -> &FileStore {
        &self.file_store
    }

    /// Ties a task spawned for this engine to its lifetime: dropping the engine aborts it.
    pub fn track_task(&self, handle: JoinHandle<()>) {
        get_mut_arcmutex!(self.handles).push(handle);
    }
}
