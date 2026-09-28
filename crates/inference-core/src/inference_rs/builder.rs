use super::*;

/// The InferenceRsBuilder takes the pipeline and a scheduler method and constructs
/// an Engine and a InferenceRs instance. The Engine runs on a separate thread, and the InferenceRs
/// instance stays on the calling thread.
pub struct InferenceRsBuilder {
    pub(super) pipeline: Arc<tokio::sync::Mutex<dyn Pipeline>>,
    pub(super) method: SchedulerConfig,
    pub(super) model_id_override: Option<String>,
    pub(super) log: Option<String>,
    pub(super) no_kv_cache: Option<bool>,
    pub(super) no_prefix_cache: Option<bool>,
    pub(super) prefix_cache_n: Option<usize>,
    pub(super) disable_eos_stop: Option<bool>,
    pub(super) throughput_logging_enabled: bool,
    pub(super) search_embedding_model: Option<SearchEmbeddingModel>,
    pub(super) search_callback: Option<Arc<SearchCallback>>,
    pub(super) tool_callbacks: tools::ToolCallbacksWithTools,
    pub(super) mcp_client_config: Option<McpClientConfig>,
    pub(super) loader_config: Option<ModelLoaderConfig>,
    pub(super) code_exec_config: Option<CodeExecutionConfig>,
    pub(super) shell_config: Option<ShellConfig>,
    pub(super) defer_daemon_start: bool,
}

impl InferenceRsBuilder {
    /// Creates a new builder with the given pipeline, scheduler method, logging flag,
    /// and optional embedding model for web search. To override the search callback,
    /// use `.with_search_callback(...)` on the builder.
    pub fn new(
        pipeline: Arc<tokio::sync::Mutex<dyn Pipeline>>,
        method: SchedulerConfig,
        throughput_logging: bool,
        search_embedding_model: Option<SearchEmbeddingModel>,
    ) -> Self {
        Self {
            pipeline,
            method,
            model_id_override: None,
            log: None,
            no_kv_cache: None,
            no_prefix_cache: None,
            prefix_cache_n: None,
            disable_eos_stop: None,
            throughput_logging_enabled: throughput_logging,
            search_embedding_model,
            search_callback: None,
            tool_callbacks: HashMap::new(),
            mcp_client_config: None,
            loader_config: None,
            code_exec_config: None,
            shell_config: None,
            defer_daemon_start: false,
        }
    }

    /// Override the model ID used by InferenceRs. Defaults to the pipeline name.
    pub fn with_model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id_override = Some(model_id.into());
        self
    }

    /// Set the loader config for enabling model unload/reload support.
    /// Without this, models cannot be unloaded and reloaded.
    pub fn with_loader_config(mut self, loader_config: ModelLoaderConfig) -> Self {
        self.loader_config = Some(loader_config);
        self
    }
    pub fn with_log(mut self, log: String) -> Self {
        self.log = Some(log);
        self
    }
    pub fn with_opt_log(mut self, log: Option<String>) -> Self {
        self.log = log;
        self
    }
    pub fn with_no_kv_cache(mut self, no_kv_cache: bool) -> Self {
        self.no_kv_cache = Some(no_kv_cache);
        self
    }
    pub fn with_no_prefix_cache(mut self, no_prefix_cache: bool) -> Self {
        self.no_prefix_cache = Some(no_prefix_cache);
        self
    }
    pub fn with_prefix_cache_n(mut self, prefix_cache_n: usize) -> Self {
        self.prefix_cache_n = Some(prefix_cache_n);
        self
    }
    pub fn with_disable_eos_stop(mut self, disable_eos_stop: bool) -> Self {
        self.disable_eos_stop = Some(disable_eos_stop);
        self
    }

    /// Use a custom callback to gather search results.
    pub fn with_search_callback(mut self, search_callback: Arc<SearchCallback>) -> Self {
        self.search_callback = Some(search_callback);
        self
    }

    /// Register a custom callback for the specified tool name.
    pub fn with_tool_callback(
        mut self,
        name: impl Into<String>,
        tool_callback: Arc<ToolCallback>,
    ) -> Self {
        let name = name.into();
        // Wrap bare callback with a minimal tool definition.
        self.tool_callbacks.insert(
            name.clone(),
            ToolCallbackWithTool {
                callback: ToolCallbackKind::Text(tool_callback),
                tool: Tool {
                    tp: ToolType::Function,
                    function: Function {
                        description: None,
                        name,
                        parameters: None,
                        strict: None,
                    },
                },
            },
        );
        self
    }

    /// Register a custom callback with its associated Tool definition. The Tool will be
    /// automatically added to requests when tool callbacks are active.
    pub fn with_tool_callback_and_tool(
        mut self,
        name: impl Into<String>,
        tool_callback: Arc<ToolCallback>,
        tool: Tool,
    ) -> Self {
        let name = name.into();
        self.tool_callbacks.insert(
            name,
            ToolCallbackWithTool {
                callback: ToolCallbackKind::Text(tool_callback),
                tool,
            },
        );
        self
    }

    /// Register a pre-built tool callback with its Tool definition.
    pub fn with_tool_callback_with_tool(
        mut self,
        name: impl Into<String>,
        callback_with_tool: ToolCallbackWithTool,
    ) -> Self {
        self.tool_callbacks.insert(name.into(), callback_with_tool);
        self
    }

    /// Configure MCP client to connect to external MCP servers.
    pub fn with_mcp_client(mut self, config: McpClientConfig) -> Self {
        self.mcp_client_config = Some(config);
        self
    }

    /// Enable Python code execution. **Security**: lets the model run arbitrary code on the host with full network and filesystem access.
    pub fn with_code_execution(mut self, config: CodeExecutionConfig) -> Self {
        self.code_exec_config = Some(config);
        self
    }

    pub fn with_shell_execution(mut self, config: ShellConfig) -> Self {
        self.shell_config = Some(config);
        self
    }

    pub fn with_deferred_daemon_start(mut self, defer_daemon_start: bool) -> Self {
        self.defer_daemon_start = defer_daemon_start;
        self
    }

    pub async fn build(self) -> Arc<InferenceRs> {
        InferenceRs::new(self).await
    }
}
