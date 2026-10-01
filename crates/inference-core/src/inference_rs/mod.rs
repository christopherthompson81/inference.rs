//! The engine handle: models, their engine threads, and the requests routed to them.

use futures::future::BoxFuture;

use crate::*;

mod builder;
mod config;
mod lora;
mod models;
mod sessions;
#[cfg(test)]
mod tests;

pub use builder::*;
pub use config::*;

/// State preserved when a model is unloaded.
/// This contains all the information needed to reload the model on demand.
#[derive(Clone)]
pub struct UnloadedModelState {
    /// Configuration to recreate the loader
    pub loader_config: ModelLoaderConfig,
    /// Scheduler configuration
    pub scheduler_config: SchedulerConfig,
    /// Engine configuration
    pub engine_config: EngineConfig,
    /// MCP client configuration
    pub mcp_client_config: Option<McpClientConfig>,
    /// Model category (Text, Multimodal, etc.)
    pub category: ModelCategory,
    /// Model metadata configuration
    pub inference_config: InferenceRsConfig,
}

/// Internal structure to hold per-engine state
struct EngineInstance {
    sender: Sender<Request>,
    // Also delivered out of band through `ENGINE_INSTRUCTIONS`, since a full queue hides a queued Terminate.
    instruction_id: usize,
    engine_handler: Option<JoinHandle<()>>,
    reboot_state: RebootState,
    adapter_runtime: Option<Arc<DynamicLoraRuntime>>,
    config: InferenceRsConfig,
    category: ModelCategory,
    logger: Arc<IntervalLogger>,
    /// Shared with the engine so the SDK/HTTP layer can read/write sessions out of band.
    session_store: Arc<std::sync::Mutex<engine::agentic_session::AgenticSessionStore>>,
    /// Shared with the engine for fetch-by-id from the SDK/HTTP layer.
    pub(crate) file_store: files::FileStore,
}

impl Drop for EngineInstance {
    fn drop(&mut self) {
        engine::ENGINE_INSTRUCTIONS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.instruction_id);
        // The engine frees its own graphs on exit; this covers an engine that never ran its loop.
        if let Ok(pipeline) = self.reboot_state.pipeline.try_lock() {
            pipeline.cleanup_cuda_graphs();
        }
    }
}

impl EngineInstance {
    fn is_finished(&self) -> bool {
        self.engine_handler
            .as_ref()
            .is_none_or(JoinHandle::is_finished)
    }

    fn terminate(&self) {
        engine::ENGINE_INSTRUCTIONS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(
                self.instruction_id,
                Some(engine::EngineInstruction::Terminate),
            );
        let _ = self.sender.try_send(Request::Terminate);
    }

    fn join(&mut self) {
        if let Some(handle) = self.engine_handler.take()
            && handle.join().is_err()
        {
            warn!("Engine thread panicked during shutdown.");
        }
    }

    fn join_until(&mut self, deadline: Instant) {
        while !self.is_finished() {
            if Instant::now() >= deadline {
                warn!(
                    "Engine thread did not stop within {ENGINE_DROP_JOIN_TIMEOUT:?}; not waiting for it."
                );
                return;
            }
            std::thread::sleep(ENGINE_DROP_POLL_INTERVAL);
        }
        self.join();
    }
}

/// The InferenceRs struct handles sending requests to multiple engines.
/// It is the core multi-threaded component of inference.rs, and uses `mpsc`
/// `Sender` and `Receiver` primitives to send and receive requests to the
/// appropriate engine based on model ID.
///
/// ## Lock Ordering Convention
///
/// This struct uses multiple `RwLock`s. To prevent deadlocks, locks must be
/// acquired in this order:
/// 1. `reloading_models`
/// 2. `engines`
/// 3. `unloaded_models`
/// 4. `default_engine_id`
/// 5. `model_aliases`
///
/// Use scope-based lock management and explicit `drop()` calls.
pub struct InferenceRs {
    engines: RwLock<HashMap<String, EngineInstance>>,
    /// Models that have been unloaded but can be reloaded on demand
    unloaded_models: RwLock<HashMap<String, UnloadedModelState>>,
    /// Models currently being reloaded (to prevent concurrent reloads)
    reloading_models: RwLock<HashSet<String>>,
    default_engine_id: RwLock<Option<String>>,
    /// Alternate IDs that resolve to primary model IDs.
    model_aliases: RwLock<HashMap<String, String>>,
    log: Option<String>,
    id: String,
    creation_time: u64,
    next_request_id: Mutex<RefCell<usize>>,
}

#[derive(Clone)]
struct RebootState {
    pipeline: Arc<tokio::sync::Mutex<dyn Pipeline>>,
    method: SchedulerConfig,
    engine_config: EngineConfig,
    mcp_client_config: Option<McpClientConfig>,
    /// Optional loader config for reloading after unload
    loader_config: Option<ModelLoaderConfig>,
}

// Clears a model's reloading mark when its reload ends, however it ends.
struct ReloadingMark<'a> {
    reloading: &'a RwLock<HashSet<String>>,
    model_id: &'a str,
}

impl Drop for ReloadingMark<'_> {
    fn drop(&mut self) {
        if let Ok(mut reloading) = self.reloading.write() {
            reloading.remove(self.model_id);
        }
    }
}

/// Model status for loaded/unloaded state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStatus {
    Loaded,
    Unloaded,
    Reloading,
}

impl std::fmt::Display for ModelStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelStatus::Loaded => write!(f, "loaded"),
            ModelStatus::Unloaded => write!(f, "unloaded"),
            ModelStatus::Reloading => write!(f, "reloading"),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum InferenceRsError {
    #[error("engine state lock is poisoned")]
    EnginePoisoned,
    #[error("request engine is unavailable")]
    SenderPoisoned,
    /// The requested model was not found (neither loaded nor unloaded)
    #[error("model `{0}` was not found")]
    ModelNotFound(String),
    /// The model is currently being reloaded
    #[error("model `{0}` is being reloaded")]
    ModelReloading(String),
    /// Failed to reload the model
    #[error("failed to reload model: {0}")]
    ReloadFailed(String),
    /// Model does not have loader config for reloading
    #[error("model `{0}` has no loader configuration")]
    NoLoaderConfig(String),
    /// Model is already loaded
    #[error("model `{0}` is already loaded")]
    ModelAlreadyLoaded(String),
    /// Model is already unloaded
    #[error("model `{0}` is already unloaded")]
    ModelAlreadyUnloaded(String),
    #[error(transparent)]
    LoraAdapter(#[from] LoraAdapterError),
    /// Other error with a message.
    #[error("{0}")]
    Other(String),
}

impl Drop for InferenceRs {
    fn drop(&mut self) {
        // Engine threads still inside CUDA when the process exits race the context teardown and segfault, so wait.
        let engines = self
            .engines
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for engine in engines.values() {
            // try_send rather than blocking_send, which panics inside a runtime
            engine.terminate();
        }
        let deadline = Instant::now() + ENGINE_DROP_JOIN_TIMEOUT;
        for engine in engines.values_mut() {
            engine.join_until(deadline);
        }
    }
}

impl InferenceRs {
    fn prepare_request_dispatch(
        &self,
        request: &mut Request,
    ) -> Result<Sender<Request>, InferenceRsError> {
        if let Request::Normal(request) = &mut *request {
            request.mark_enqueued();
        }
        let requested_model = match &*request {
            Request::Normal(request) => request.model_id.clone(),
            _ => None,
        };
        self.get_sender(requested_model.as_deref())?;

        let model_id = self.resolve_alias_or_default(requested_model.as_deref())?;
        let engines = self
            .engines
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        let engine = engines
            .get(&model_id)
            .ok_or_else(|| InferenceRsError::ModelNotFound(model_id.clone()))?;
        if let Request::Normal(request) = request
            && let Some(selection) = request.adapter.as_mut()
        {
            let runtime = engine.adapter_runtime.as_ref().ok_or_else(|| {
                LoraAdapterError::RuntimeUnavailable {
                    model_id: model_id.clone(),
                }
            })?;
            selection.pin(runtime)?;
            if let Some(generation) = selection.resolved_generation() {
                debug!(model_id, %generation, "admitted LoRA adapter request");
            }
        }
        Ok(engine.sender.clone())
    }

    pub fn shutdown(self: Arc<Self>) -> BoxFuture<'static, Result<(), String>> {
        Box::pin(self.shutdown_inner())
    }

    async fn shutdown_inner(self: Arc<Self>) -> Result<(), String> {
        let mut this =
            Arc::try_unwrap(self).map_err(|_| "Cannot shutdown while InferenceRs is shared")?;
        let engines = this
            .engines
            .get_mut()
            .map_err(|_| "Failed to get mutable access to engines during shutdown")?;
        let mut engines = std::mem::take(engines);

        let senders = engines
            .values()
            .map(|engine| engine.sender.clone())
            .collect::<Vec<_>>();
        for sender in senders {
            let _ = sender.send(Request::Terminate).await;
        }

        for engine in engines.values_mut() {
            engine.join();
        }

        Ok(())
    }

    /// Create an engine instance with the given configuration
    fn create_engine_instance(reboot_state: RebootState) -> Result<EngineInstance, String> {
        let pipeline = reboot_state.pipeline.clone();
        let method = reboot_state.method.clone();
        let config = reboot_state.engine_config.clone();
        let (tx, rx) = channel(DEFAULT_ENGINE_REQUEST_QUEUE_CAPACITY);

        let pipeline_guard = pipeline.try_lock().unwrap();
        let category = pipeline_guard.category();
        let metadata = pipeline_guard.get_metadata();
        let kind = metadata.kind.clone();
        let device = pipeline_guard.device();
        let modalities = metadata.modalities.clone();
        let max_seq_len = match &category {
            ModelCategory::Diffusion | ModelCategory::Speech => None,
            _ => Some(metadata.max_seq_len),
        };
        let generation_defaults = pipeline_guard.generation_defaults();
        let encoder_cache_counters = pipeline_guard.encoder_cache_counters();
        let adapter_runtime = pipeline_guard.adapter_runtime();
        drop(pipeline_guard);

        // Warm cuTile before the engine starts capturing and serving CUDA work.
        #[cfg(feature = "cutile")]
        let warmup_device = device.clone();

        let logger = Arc::new(IntervalLogger::new(
            Duration::from_secs(5),
            encoder_cache_counters,
        ));
        let logger_for_engine = logger.clone();

        info!("Pipeline input modalities are {:?}", &modalities.input);
        info!("Pipeline output modalities are {:?}", &modalities.output);

        let inference_config = InferenceRsConfig {
            kind,
            device,
            category: category.clone(),
            modalities,
            max_seq_len,
            generation_defaults,
        };

        // Shared between engine and EngineInstance so the SDK/HTTP API
        // can access sessions without going through the request channel.
        let session_store = Arc::new(std::sync::Mutex::new(
            engine::agentic_session::AgenticSessionStore::new(),
        ));
        let session_store_for_engine = Arc::clone(&session_store);
        let file_store = files::FileStore::new();
        let file_store_for_engine = file_store.clone();

        let tx_for_engine = tx.clone();
        let instruction_id =
            engine::NEXT_ENGINE_INSTRUCTION_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Propagate Engine::new's outcome so a creation failure is a clean load error, not a zombie-engine panic.
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<Result<(), String>>(1);
        let engine_handler = thread::spawn(move || {
            candle_core::utils::init_global_threadpool();
            #[cfg(feature = "metal")]
            objc::rc::autoreleasepool(move || {
                let rt = build_engine_runtime();
                rt.block_on(async move {
                    file_store_for_engine.spawn_cleanup_task();
                    // cuTile warmup precedes graph capture.
                    #[cfg(feature = "cutile")]
                    if let Err(err) = inference_quant::cutile::warmup_moe_kernels(&warmup_device) {
                        warn!("Failed to warm up cuTile MoE kernels: {err}");
                    }
                    let engine = match Engine::new(engine::EngineParts {
                        tx: tx_for_engine,
                        rx,
                        pipeline,
                        scheduler: method,
                        engine: config,
                        logger: logger_for_engine,
                        session_store: session_store_for_engine,
                        file_store: file_store_for_engine,
                    }) {
                        Ok(engine) => {
                            let _ = ready_tx.send(Ok(()));
                            engine.with_instruction_id(instruction_id)
                        }
                        Err(e) => {
                            let _ = ready_tx.send(Err(format!("{e:#}")));
                            return;
                        }
                    };
                    Arc::new(engine).run().await;
                })
            });

            #[cfg(not(feature = "metal"))]
            {
                let rt = build_engine_runtime();
                rt.block_on(async move {
                    file_store_for_engine.spawn_cleanup_task();
                    // cuTile warmup precedes graph capture.
                    #[cfg(feature = "cutile")]
                    if let Err(err) = inference_quant::cutile::warmup_moe_kernels(&warmup_device) {
                        warn!("Failed to warm up cuTile MoE kernels: {err}");
                    }
                    let engine = match Engine::new(engine::EngineParts {
                        tx: tx_for_engine,
                        rx,
                        pipeline,
                        scheduler: method,
                        engine: config,
                        logger: logger_for_engine,
                        session_store: session_store_for_engine,
                        file_store: file_store_for_engine,
                    }) {
                        Ok(engine) => {
                            let _ = ready_tx.send(Ok(()));
                            engine.with_instruction_id(instruction_id)
                        }
                        Err(e) => {
                            let _ = ready_tx.send(Err(format!("{e:#}")));
                            return;
                        }
                    };
                    Arc::new(engine).run().await;
                })
            }
        });

        // Wait for the engine thread to report whether Engine::new succeeded
        // Propagate failures here instead of leaving a dead engine that looks loaded and then panics on the first request.
        match ready_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(format!("Engine creation failed: {e}")),
            Err(_) => return Err("Engine thread exited before reporting readiness".to_string()),
        }

        Ok(EngineInstance {
            instruction_id,
            sender: tx,
            engine_handler: Some(engine_handler),
            reboot_state,
            adapter_runtime,
            config: inference_config,
            category,
            logger,
            session_store,
            file_store,
        })
    }

    /// Initialize MCP and code-execution tool callbacks and merge them into `tool_callbacks`.
    /// Used by both `InferenceRsBuilder::new` and `add_model` so dynamically added models pick up
    /// the same external tools as the boot-time model.
    async fn init_external_tool_callbacks(
        #[cfg_attr(not(feature = "code-execution"), allow(unused_variables))] pipeline: &Arc<
            tokio::sync::Mutex<dyn Pipeline>,
        >,
        tool_callbacks: &mut tools::ToolCallbacksWithTools,
        mcp_client_config: Option<&McpClientConfig>,
        #[cfg_attr(not(feature = "code-execution"), allow(unused_variables))]
        code_exec_config: Option<&CodeExecutionConfig>,
        #[cfg_attr(not(feature = "code-execution"), allow(unused_variables))] shell_config: Option<
            &ShellConfig,
        >,
    ) {
        if let Some(config) = mcp_client_config {
            let mut mcp_client = McpClient::new(config.clone());
            let total_servers = config.servers.len();

            match mcp_client.initialize().await {
                Ok(()) => {
                    let mcp_callbacks_with_tools = mcp_client.get_tool_callbacks_with_tools();
                    let tools_count = mcp_callbacks_with_tools.len();

                    for (name, callback_with_tool) in mcp_callbacks_with_tools {
                        tool_callbacks.insert(name.clone(), callback_with_tool.clone());
                    }

                    if tools_count == 0 {
                        warn!(
                            "MCP client initialized but no tools were registered from {} servers",
                            total_servers
                        );
                    } else {
                        info!(
                            "MCP client initialized successfully with {} tools from {} servers",
                            tools_count, total_servers
                        );
                    }
                }
                Err(e) => {
                    warn!(
                        "Failed to initialize MCP client with {} configured servers: {}",
                        total_servers, e
                    );
                    warn!(
                        "Continuing without MCP functionality. Check your MCP configuration and server availability."
                    );
                }
            }
        }

        #[cfg(feature = "code-execution")]
        if let Some(code_exec_cfg) = code_exec_config {
            let exec_config = code_exec_cfg.clone();
            match inference_code_exec::CodeExecutionManager::new(exec_config).await {
                Ok(manager) => {
                    let input_modalities: Vec<inference_code_exec::InputModality> = {
                        let pipe = get_mut_arcmutex!(pipeline);
                        pipe.get_metadata()
                            .modalities
                            .input
                            .iter()
                            .filter_map(|m| match m {
                                pipeline::SupportedModality::Text => {
                                    Some(inference_code_exec::InputModality::Text)
                                }
                                pipeline::SupportedModality::Vision => {
                                    Some(inference_code_exec::InputModality::Vision)
                                }
                                pipeline::SupportedModality::Audio => {
                                    Some(inference_code_exec::InputModality::Audio)
                                }
                                pipeline::SupportedModality::Video => {
                                    Some(inference_code_exec::InputModality::Video)
                                }
                                _ => None,
                            })
                            .collect()
                    };
                    let effective = manager.effective_protection();
                    let network = manager.network_mode();
                    let callbacks = manager.get_tool_callbacks(&input_modalities);
                    let count = callbacks.len();
                    for (name, cb) in callbacks {
                        tool_callbacks.insert(name, cb);
                    }
                    warn!("============================================================");
                    warn!("  CODE EXECUTION IS ENABLED");
                    warn!("  The model can execute arbitrary Python code on this machine.");
                    if effective.any() {
                        let fs = if effective.fs_isolated {
                            "workdir + system libs only"
                        } else {
                            "NOT restricted"
                        };
                        let net = if effective.network_isolated {
                            match network {
                                Some(inference_sandbox::NetworkMode::None) => "denied",
                                Some(inference_sandbox::NetworkMode::Loopback) => "loopback only",
                                _ => "NOT restricted",
                            }
                        } else {
                            "NOT restricted"
                        };
                        warn!(
                            "  Sandbox: on. Filesystem: {fs}. Network: {net}. rlimits: {}.",
                            if effective.rlimits_applied {
                                "applied"
                            } else {
                                "not applied"
                            }
                        );
                        if !effective.fs_isolated || !effective.network_isolated {
                            warn!(
                                "  Some layers are inactive on this host. Use --sandbox on to make missing layers a hard error."
                            );
                        }
                    } else {
                        warn!("  Sandbox: OFF. Network and filesystem are NOT restricted.");
                        warn!(
                            "  Pass a sandbox_policy (or --sandbox on at the CLI) to enable isolation."
                        );
                    }
                    warn!(
                        "  See: https://github.com/christopherthompson81/inference.rs/blob/master/docs/src/content/docs/reference/sandbox.md"
                    );
                    warn!("============================================================");
                    info!("Code execution initialized with {count} tools");
                }
                Err(e) => {
                    warn!("Failed to initialize code execution: {e}");
                    warn!("Continuing without code execution functionality.");
                }
            }
        }

        #[cfg(feature = "code-execution")]
        if let Some(shell_cfg) = shell_config {
            let shell_config = shell_cfg.clone();
            match inference_code_exec::ShellManager::new(shell_config).await {
                Ok(manager) => {
                    let effective = manager.effective_protection();
                    let network = manager.network_mode();
                    let callbacks = manager.get_tool_callbacks();
                    let count = callbacks.len();
                    for (name, cb) in callbacks {
                        tool_callbacks.insert(name, cb);
                    }
                    warn!("============================================================");
                    warn!("  SHELL EXECUTION IS ENABLED");
                    warn!("  The model can execute arbitrary shell commands on this machine.");
                    if effective.any() {
                        let fs = if effective.fs_isolated {
                            "workdir + system libs only"
                        } else {
                            "NOT restricted"
                        };
                        let net = if effective.network_isolated {
                            match network {
                                Some(inference_sandbox::NetworkMode::None) => "denied",
                                Some(inference_sandbox::NetworkMode::Loopback) => "loopback only",
                                _ => "NOT restricted",
                            }
                        } else {
                            "NOT restricted"
                        };
                        warn!(
                            "  Sandbox: on. Filesystem: {fs}. Network: {net}. rlimits: {}.",
                            if effective.rlimits_applied {
                                "applied"
                            } else {
                                "not applied"
                            }
                        );
                    } else {
                        warn!("  Sandbox: OFF. Network and filesystem are NOT restricted.");
                    }
                    warn!(
                        "  See: https://github.com/christopherthompson81/inference.rs/blob/master/docs/src/content/docs/reference/sandbox.md"
                    );
                    warn!("============================================================");
                    info!("Shell execution initialized with {count} tool");
                }
                Err(e) => {
                    warn!("Failed to initialize shell execution: {e}");
                    warn!("Continuing without shell execution functionality.");
                }
            }
        }
    }

    async fn new(config: InferenceRsBuilder) -> Arc<Self> {
        info!("inference.rs version: {INFERENCE_RS_VERSION}");
        info!("git revision: {INFERENCE_RS_GIT_REVISION}");
        let InferenceRsBuilder {
            pipeline,
            method,
            model_id_override,
            log,
            no_kv_cache,
            no_prefix_cache,
            prefix_cache_n,
            disable_eos_stop,
            throughput_logging_enabled,
            search_embedding_model,
            search_callback,
            mut tool_callbacks,
            agent_runner,
            mcp_client_config,
            loader_config,
            #[cfg_attr(not(feature = "code-execution"), allow(unused_variables))]
            code_exec_config,
            #[cfg_attr(not(feature = "code-execution"), allow(unused_variables))]
            shell_config,
            defer_daemon_start,
        } = config;

        let device = get_mut_arcmutex!(pipeline).device();
        inference_quant::cublaslt::maybe_init_cublas_lt_wrapper(device.clone());
        #[cfg(feature = "cuda")]
        match cuda::preload::preload_candle_ptx(&device) {
            Ok(count) if count > 0 => info!("Preloaded {count} Candle CUDA PTX functions."),
            Ok(_) => {}
            Err(err) => warn!("Failed to preload Candle CUDA PTX functions: {err}"),
        }

        let no_kv_cache = no_kv_cache.unwrap_or(false);
        let no_prefix_cache = no_prefix_cache.unwrap_or(false);
        let prefix_cache_n = prefix_cache_n.unwrap_or(16);
        let disable_eos_stop = disable_eos_stop.unwrap_or(false);

        Self::init_external_tool_callbacks(
            &pipeline,
            &mut tool_callbacks,
            mcp_client_config.as_ref(),
            code_exec_config.as_ref(),
            shell_config.as_ref(),
        )
        .await;

        let engine_config = EngineConfig {
            no_kv_cache,
            no_prefix_cache,
            prefix_cache_n,
            disable_eos_stop,
            throughput_logging_enabled,
            search_embedding_model,
            search_callback,
            tool_callbacks,
            agent_runner,
        };
        let reboot_state = RebootState {
            pipeline: pipeline.clone(),
            method,
            engine_config,
            mcp_client_config: mcp_client_config.clone(),
            loader_config,
        };

        let pipeline_name = pipeline.lock().await.name();
        let engine_instance =
            Self::create_engine_instance(reboot_state).expect("Failed to create engine instance");

        let (id, alias_map) = match model_id_override {
            Some(override_id) => {
                let mut alias_map = HashMap::new();
                if override_id != pipeline_name {
                    alias_map.insert(pipeline_name.clone(), override_id.clone());
                }
                (override_id, alias_map)
            }
            None => (pipeline_name.clone(), HashMap::new()),
        };

        if distributed::is_daemon() && !defer_daemon_start {
            let request_sender = engine_instance.sender.clone();

            if cfg!(feature = "ring") {
                // Ring daemon replicator
                distributed::ring_daemon_replicator(request_sender);
            } else {
                // NCCL daemon replicator
                distributed::nccl_daemon_replicator(request_sender);
            }

            #[allow(clippy::empty_loop)]
            loop {}
        }

        // Determine if the current runtime is multi-threaded, as blocking operations are not allowed in single-threaded mode
        let is_multi_threaded = tokio::runtime::Handle::try_current()
            .is_ok_and(|h| h.runtime_flavor() != tokio::runtime::RuntimeFlavor::CurrentThread);

        // Do a dummy run; skip UQFF writes, whose CPU-resident model cannot serve requests.
        let loaded_for_uqff_write = get_mut_arcmutex!(pipeline)
            .get_metadata()
            .loaded_for_uqff_write;
        if !distributed::is_daemon()
            && is_multi_threaded
            && !loaded_for_uqff_write
            && matches!(
                engine_instance.category,
                ModelCategory::Text | ModelCategory::Multimodal { .. }
            )
        {
            let clone_sender = engine_instance.sender.clone();
            tokio::task::block_in_place(|| {
                let (tx, mut rx) = channel(1);
                let req = Request::Normal(Box::new(NormalRequest {
                    id: 0,
                    queued_at: None,
                    messages: RequestMessage::Completion {
                        text: "hello".to_string(),
                        echo_prompt: false,
                        best_of: None,
                    },
                    sampling_params: SamplingParams {
                        max_len: Some(1),
                        ..SamplingParams::deterministic()
                    },
                    seed: None,
                    response: tx,
                    return_logprobs: false,
                    is_streaming: false,
                    constraint: Constraint::None,
                    suffix: None,
                    tool_choice: None,
                    tools: None,
                    logits_processors: None,
                    host_tools: Vec::new(),
                    sequential_tool_calls: false,
                    return_raw_logits: false,
                    web_search_options: None,
                    enable_code_execution: false,
                    enable_shell: false,
                    shell_options: None,
                    code_execution_permission: None,
                    code_execution_approval_notifier: None,
                    agent_permission: None,
                    agent_approval_handler: None,
                    agent_approval_notifier: None,
                    max_tool_rounds: None,
                    tool_dispatch_url: None,
                    model_id: None,
                    adapter: None,
                    truncate_sequence: false,
                    session_id: None,
                    owner: None,
                    files: None,
                    input_files: Vec::new(),
                    cancellation: None,
                }));
                debug!("Beginning dummy run.");
                let start = Instant::now();
                clone_sender.blocking_send(req).unwrap();

                // Drain all responses from the channel until it's closed
                let mut received_any = false;
                while let Some(_resp) = rx.blocking_recv() {
                    received_any = true;
                }

                if received_any {
                    let end = Instant::now();
                    debug!(
                        "Dummy run completed in {}s.",
                        end.duration_since(start).as_secs_f64()
                    );
                } else {
                    warn!("Dummy run failed!");
                }
            });

            // Reset logger counters so the dummy run doesn't pollute stats
            engine_instance.logger.reset();
        }

        // Create engines map with the first engine
        let mut engines = HashMap::new();
        engines.insert(id.clone(), engine_instance);

        Arc::new(Self {
            engines: RwLock::new(engines),
            unloaded_models: RwLock::new(HashMap::new()),
            reloading_models: RwLock::new(HashSet::new()),
            default_engine_id: RwLock::new(Some(id.clone())),
            model_aliases: RwLock::new(alias_map),
            log,
            id,
            creation_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time travel has occurred!")
                .as_secs(),
            next_request_id: Mutex::new(RefCell::new(1)),
        })
    }

    /// Attempts to reboot a specific engine by model_id
    fn reboot_engine(&self, model_id: &str) -> Result<(), InferenceRsError> {
        let mut engines = self.engines.write().map_err(|_| {
            tracing::warn!("Couldn't get write lock on engines during reboot attempt");
            InferenceRsError::EnginePoisoned
        })?;

        if let Some(engine_instance) = engines.get(model_id) {
            if !engine_instance.is_finished() {
                tracing::info!("Engine {} already running, returning ok", model_id);
                return Ok(());
            }

            let new_engine_instance = Self::create_engine_instance(
                engine_instance.reboot_state.clone(),
            )
            .map_err(|e| {
                tracing::error!("Failed to create new engine instance: {}", e);
                InferenceRsError::EnginePoisoned
            })?;

            engines.insert(model_id.to_string(), new_engine_instance);
            tracing::info!("Successfully rebooted engine {}", model_id);
            Ok(())
        } else {
            Err(InferenceRsError::EnginePoisoned)
        }
    }

    fn engine_dead(&self, model_id: &str) -> Result<bool, InferenceRsError> {
        let engines = self.engines.read().map_err(|_| {
            tracing::warn!("Couldn't get read lock on engines!");
            InferenceRsError::EnginePoisoned
        })?;

        if let Some(engine_instance) = engines.get(model_id) {
            Ok(engine_instance.is_finished())
        } else {
            Err(InferenceRsError::EnginePoisoned)
        }
    }

    /// Get sender for a specific model. If model_id is None, uses default engine.
    /// If the model is unloaded, it will be automatically reloaded before returning the sender.
    pub fn get_sender(&self, model_id: Option<&str>) -> Result<Sender<Request>, InferenceRsError> {
        let resolved_model_id = self.resolve_alias_or_default(model_id)?;

        // Check if model is loaded
        let is_loaded = {
            let engines = self
                .engines
                .read()
                .map_err(|_| InferenceRsError::EnginePoisoned)?;
            engines.contains_key(&resolved_model_id)
        };

        if is_loaded {
            // Check if engine is dead and needs reboot
            if self.engine_dead(&resolved_model_id)? {
                tracing::warn!("Engine {} is dead, rebooting", resolved_model_id);
                self.reboot_engine(&resolved_model_id)?
            }

            let engines = self
                .engines
                .read()
                .map_err(|_| InferenceRsError::EnginePoisoned)?;
            if let Some(engine_instance) = engines.get(&resolved_model_id) {
                return Ok(engine_instance.sender.clone());
            }
        }

        // Check if model is unloaded - trigger auto-reload
        let is_unloaded = {
            let unloaded = self
                .unloaded_models
                .read()
                .map_err(|_| InferenceRsError::EnginePoisoned)?;
            unloaded.contains_key(&resolved_model_id)
        };

        if is_unloaded {
            tracing::info!(
                "Model {} is unloaded, triggering auto-reload",
                resolved_model_id
            );
            match self.reload_model_blocking(&resolved_model_id) {
                // another request reloaded it first
                Ok(()) | Err(InferenceRsError::ModelAlreadyLoaded(_)) => {}
                Err(error) => return Err(error),
            }

            // After reload, get the sender
            let engines = self
                .engines
                .read()
                .map_err(|_| InferenceRsError::EnginePoisoned)?;
            if let Some(engine_instance) = engines.get(&resolved_model_id) {
                return Ok(engine_instance.sender.clone());
            }
        }

        let is_reloading = self
            .reloading_models
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?
            .contains(&resolved_model_id);
        if is_reloading {
            return Err(InferenceRsError::ModelReloading(resolved_model_id));
        }

        Err(InferenceRsError::ModelNotFound(resolved_model_id))
    }

    pub fn get_id(&self) -> String {
        self.id.clone()
    }

    pub fn get_creation_time(&self) -> u64 {
        self.creation_time
    }

    /// Get the interval logger for a specific model. If model_id is None, uses default engine.
    pub fn get_logger(
        &self,
        model_id: Option<&str>,
    ) -> Result<Arc<IntervalLogger>, InferenceRsError> {
        let resolved_model_id = self.resolve_alias_or_default(model_id)?;

        let engines = self
            .engines
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        if let Some(engine_instance) = engines.get(&resolved_model_id) {
            Ok(engine_instance.logger.clone())
        } else {
            Err(InferenceRsError::EnginePoisoned)
        }
    }

    pub fn next_request_id(&self) -> usize {
        let l = self.next_request_id.lock().unwrap();
        let last = &mut *l.borrow_mut();
        let last_v = *last;
        *last += 1;
        last_v
    }

    /// Dispatch a request to the appropriate engine based on the model_id in the request
    pub fn send_request(&self, mut request: Request) -> Result<(), InferenceRsError> {
        let sender = self.prepare_request_dispatch(&mut request)?;
        sender
            .blocking_send(request)
            .map_err(|_| InferenceRsError::SenderPoisoned)
    }

    pub fn send_request_async<'a>(
        &'a self,
        request: Request,
    ) -> BoxFuture<'a, Result<(), InferenceRsError>> {
        Box::pin(self.send_request_async_inner(request))
    }

    async fn send_request_async_inner(&self, mut request: Request) -> Result<(), InferenceRsError> {
        let sender = self.prepare_request_dispatch(&mut request)?;
        sender
            .send(request)
            .await
            .map_err(|_| InferenceRsError::SenderPoisoned)
    }

    pub fn run_daemon_replicator_forever(self: Arc<Self>) -> ! {
        if cfg!(feature = "ring") {
            distributed::ring_daemon_replicator_inference(self);
        } else {
            distributed::nccl_daemon_replicator_inference(self);
        }

        #[allow(clippy::empty_loop)]
        loop {}
    }

    pub fn maybe_log_request(this: Arc<Self>, repr: String) {
        if let Some(file) = &this.log {
            let mut f = OpenOptions::new()
                .append(true)
                .create(true) // Optionally create the file if it doesn't already exist
                .open(file)
                .expect("Unable to open file");
            let time = chrono::offset::Local::now();
            f.write_all(format!("Request at {time}: {repr}\n\n").as_bytes())
                .expect("Unable to write data");
        }
    }

    pub fn maybe_log_response<T: Serialize>(this: Arc<Self>, resp: &T) {
        if let Some(file) = &this.log {
            let mut f = OpenOptions::new()
                .append(true)
                .create(true) // Optionally create the file if it doesn't already exist
                .open(file)
                .expect("Unable to open file");
            let time = chrono::offset::Local::now();
            let repr = serde_json::to_string(resp).expect("Serialization of response failed.");
            f.write_all(format!("Response at {time}: {repr}\n\n").as_bytes())
                .expect("Unable to write data");
        }
    }

    pub fn maybe_log_error(this: Arc<Self>, err: &dyn Error) {
        if let Some(file) = &this.log {
            let mut f = OpenOptions::new()
                .append(true)
                .create(true) // Optionally create the file if it doesn't already exist
                .open(file)
                .expect("Unable to open file");
            let time = chrono::offset::Local::now();
            f.write_all(format!("Error response at {time}: {err}\n\n").as_bytes())
                .expect("Unable to write data");
        }
    }

    /// Get the number of tools available for a specific model (including MCP tools)
    pub fn get_tools_count(&self, model_id: Option<&str>) -> Result<usize, String> {
        let resolved_model_id = self
            .resolve_alias_or_default(model_id)
            .map_err(|e| e.to_string())?;

        let engines = self
            .engines
            .read()
            .map_err(|_| "Failed to acquire read lock on engines")?;
        if let Some(engine_instance) = engines.get(&resolved_model_id) {
            Ok(engine_instance
                .reboot_state
                .engine_config
                .tool_callbacks
                .len())
        } else {
            Err(format!("Model {resolved_model_id} not found"))
        }
    }

    /// MCP-provided tools registered for `model_id`. Excludes built-ins (web search, code exec). Returns `(name, description)` per tool.
    pub fn list_mcp_tools(
        &self,
        model_id: Option<&str>,
    ) -> Result<Vec<(String, Option<String>)>, String> {
        let resolved_model_id = self
            .resolve_alias_or_default(model_id)
            .map_err(|e| e.to_string())?;

        let engines = self
            .engines
            .read()
            .map_err(|_| "Failed to acquire read lock on engines")?;
        let engine_instance = engines
            .get(&resolved_model_id)
            .ok_or_else(|| format!("Model {resolved_model_id} not found"))?;

        let mut tools: Vec<(String, Option<String>)> = engine_instance
            .reboot_state
            .engine_config
            .tool_callbacks
            .values()
            .filter(|cb| {
                let name = &cb.tool.function.name;
                // Exclude built-in tools; everything else came from MCP.
                !search::search_tool_called(name) && {
                    #[cfg(feature = "code-execution")]
                    {
                        !inference_code_exec::code_exec_tool_called(name)
                    }
                    #[cfg(not(feature = "code-execution"))]
                    {
                        true
                    }
                }
            })
            .map(|cb| {
                (
                    cb.tool.function.name.clone(),
                    cb.tool.function.description.clone(),
                )
            })
            .collect();
        tools.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(tools)
    }

    /// Check if MCP client is configured for a specific model
    pub fn has_mcp_client(&self, model_id: Option<&str>) -> Result<bool, String> {
        let resolved_model_id = self
            .resolve_alias_or_default(model_id)
            .map_err(|e| e.to_string())?;

        let engines = self
            .engines
            .read()
            .map_err(|_| "Failed to acquire read lock on engines")?;
        if let Some(engine_instance) = engines.get(&resolved_model_id) {
            Ok(engine_instance.reboot_state.mcp_client_config.is_some())
        } else {
            Err(format!("Model {resolved_model_id} not found"))
        }
    }

    /// Get config for a specific model
    pub fn config(&self, model_id: Option<&str>) -> Result<InferenceRsConfig, String> {
        let resolved_model_id = self
            .resolve_alias_or_default(model_id)
            .map_err(|e| e.to_string())?;

        let engines = self
            .engines
            .read()
            .map_err(|_| "Failed to acquire read lock on engines")?;
        if let Some(engine_instance) = engines.get(&resolved_model_id) {
            Ok(engine_instance.config.clone())
        } else {
            Err(format!("Model {resolved_model_id} not found"))
        }
    }
}
