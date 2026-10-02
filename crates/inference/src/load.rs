//! What every model builder takes besides its model: device, quantization, caches, scheduling and agent tools.

use std::sync::Arc;

use inference_api::engine::{
    AgenticSpec, CodeExecutionConfig, EngineCallbacks, EngineSpec, HfConfigOverrides, IsqType,
    McpClientConfig, MtpSpec, PagedCacheSpec, PagedCacheType, RuntimeSpec, SearchCallback,
    SearchEmbeddingModel, SearchSpec, ShellConfig, TokenSource, Tool, ToolCallbackKind,
    ToolCallbackWithTool,
};

const DEFAULT_PAGED_CONTEXT: usize = 4096;
const DEFAULT_PREFIX_CACHE_N: usize = 16;
const CPU: &str = "cpu";
const FIRST_DEVICE: usize = 0;

use crate::{Model, error::Result};

/// In-situ quantization at a bit width, resolved to the best type for the device it loads on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IsqBits {
    Two,
    Three,
    Four,
    Five,
    Six,
    Eight,
}

impl IsqBits {
    fn bits(self) -> u8 {
        match self {
            Self::Two => 2,
            Self::Three => 3,
            Self::Four => 4,
            Self::Five => 5,
            Self::Six => 6,
            Self::Eight => 8,
        }
    }
}

/// How much the paged-attention KV cache holds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MemoryGpuConfig {
    /// Room for this many tokens of context.
    ContextSize(usize),
    MbAmount(usize),
    /// This fraction of device memory, 0 to 1.
    Utilization(f32),
}

/// Paged attention's block size, cache size and cache type.
pub struct PagedAttentionMetaBuilder {
    block_size: Option<usize>,
    memory: MemoryGpuConfig,
    cache_type: PagedCacheType,
}

impl Default for PagedAttentionMetaBuilder {
    fn default() -> Self {
        Self {
            block_size: None,
            memory: MemoryGpuConfig::ContextSize(DEFAULT_PAGED_CONTEXT),
            cache_type: PagedCacheType::Auto,
        }
    }
}

impl PagedAttentionMetaBuilder {
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = Some(block_size);
        self
    }

    pub fn with_gpu_memory(mut self, memory: MemoryGpuConfig) -> Self {
        self.memory = memory;
        self
    }

    pub fn with_paged_cache_type(mut self, cache_type: PagedCacheType) -> Self {
        self.cache_type = cache_type;
        self
    }

    pub fn build(self) -> Result<PagedCacheSpec> {
        let mut spec = PagedCacheSpec {
            block_size: self.block_size,
            cache_type: self.cache_type,
            ..PagedCacheSpec::default()
        };
        match self.memory {
            MemoryGpuConfig::ContextSize(tokens) => spec.context_len = Some(tokens),
            MemoryGpuConfig::MbAmount(mb) => spec.memory_mb = Some(mb),
            MemoryGpuConfig::Utilization(fraction) => spec.memory_fraction = Some(fraction),
        }
        Ok(spec)
    }
}

pub use inference_api::sdk::ToolCallback;

/// The prompt length and batch size automatic device mapping plans for; neither is a limit.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AutoMapSizing {
    pub(crate) max_seq_len: usize,
    pub(crate) max_batch_size: usize,
}

impl Default for AutoMapSizing {
    fn default() -> Self {
        Self {
            max_seq_len: inference_api::engine::AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
            max_batch_size: inference_api::engine::AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
        }
    }
}

/// The options every builder shares, kept apart from the model they load.
#[derive(Default)]
pub(crate) struct LoadOptions {
    pub(crate) runtime: RuntimeSpec,
    pub(crate) agentic: AgenticSpec,
    pub(crate) tools: Vec<ToolCallbackWithTool>,
    pub(crate) search: Option<Arc<SearchCallback>>,
    pub(crate) with_logging: bool,
    pub(crate) auto_map: AutoMapSizing,
}

impl LoadOptions {
    pub(crate) fn new() -> Self {
        let mut options = Self::default();
        // The SDK's long-standing defaults: no throughput lines, a small prefix cache, paged attention only if asked.
        options.runtime.throughput_logging = Some(false);
        options.runtime.prefix_cache_n = Some(DEFAULT_PREFIX_CACHE_N);
        options.runtime.paged_attn = Some(false);
        options
    }

    /// The spec these options make once `models` names what it loads.
    pub(crate) fn spec_with(
        self,
        models: impl FnOnce(&mut EngineSpec),
    ) -> (EngineSpec, EngineCallbacks) {
        let mut spec = EngineSpec {
            runtime: self.runtime,
            agentic: self.agentic,
            adapters: inference_api::engine::AdapterSpec {
                runtime_updates: true,
                root: None,
            },
            ..EngineSpec::default()
        };
        models(&mut spec);
        let callbacks = EngineCallbacks {
            tools: self.tools,
            search: self.search,
        };
        (spec, callbacks)
    }

    pub(crate) fn spec(
        self,
        model: inference_api::engine::ModelSelected,
    ) -> (EngineSpec, EngineCallbacks) {
        self.spec_with(|spec| spec.model = Some(model))
    }

    pub(crate) async fn load(self, model: inference_api::engine::ModelSelected) -> Result<Model> {
        let with_logging = self.with_logging;
        let (spec, callbacks) = self.spec(model);
        load_engine(spec, callbacks, with_logging).await
    }
}

pub(crate) async fn load_engine(
    spec: EngineSpec,
    callbacks: EngineCallbacks,
    with_logging: bool,
) -> Result<Model> {
    if with_logging {
        crate::initialize_logging();
    }
    let engine = inference_api::Engine::load_with_callbacks(spec, callbacks).await?;
    Ok(engine.into())
}

pub(crate) fn device(device: String) -> String {
    if device == CPU || device.contains(':') {
        device
    } else {
        format!("{device}:{FIRST_DEVICE}")
    }
}

pub(crate) fn isq_bits(bits: IsqBits) -> String {
    bits.bits().to_string()
}

pub(crate) fn isq_type(isq: IsqType) -> String {
    isq.to_string()
}

pub(crate) fn token_source(source: &TokenSource) -> String {
    source.to_string()
}

pub(crate) fn text_tool(callback: Arc<ToolCallback>, tool: Tool) -> ToolCallbackWithTool {
    ToolCallbackWithTool {
        callback: ToolCallbackKind::Text(callback),
        tool,
    }
}

pub(crate) fn search(model: SearchEmbeddingModel) -> SearchSpec {
    SearchSpec {
        embedding_model: model,
    }
}

pub(crate) fn mtp(model: Option<String>, n_predict: Option<usize>) -> MtpSpec {
    MtpSpec {
        model,
        n_predict,
        ..MtpSpec::default()
    }
}

pub(crate) type Overrides = HfConfigOverrides;
pub(crate) type Mcp = McpClientConfig;
pub(crate) type CodeExecution = CodeExecutionConfig;
pub(crate) type Shell = ShellConfig;

/// The builder methods every model kind shares; each builder holds its options in `self.options`.
macro_rules! load_options_methods {
    () => {
        /// Loads on `device`: `cpu`, `cuda`, `cuda:N`, `metal` or `metal:N`; a bare name means its first device.
        pub fn with_device(mut self, device: impl Into<String>) -> Self {
            self.options.runtime.device = Some($crate::load::device(device.into()));
            self
        }

        pub fn with_force_cpu(self) -> Self {
            self.with_device("cpu")
        }

        /// The prompt length and batch size automatic device mapping plans memory for.
        pub fn with_auto_map_sizing(mut self, max_seq_len: usize, max_batch_size: usize) -> Self {
            self.options.auto_map.max_seq_len = max_seq_len;
            self.options.auto_map.max_batch_size = max_batch_size;
            self
        }

        /// Layers per device, as `ORD:NUM` entries or one count for device 0; unset maps them automatically.
        pub fn with_device_layers(mut self, layers: Vec<String>) -> Self {
            self.options.runtime.device_layers = Some(layers);
            self
        }

        pub fn with_token_source(mut self, source: $crate::TokenSource) -> Self {
            self.options.runtime.token_source = Some($crate::load::token_source(&source));
            self
        }

        pub fn with_hf_revision(mut self, revision: impl ToString) -> Self {
            self.options.runtime.hf_revision = Some(revision.to_string());
            self
        }

        pub fn with_isq(mut self, isq: $crate::IsqType) -> Self {
            self.options.runtime.isq = Some($crate::load::isq_type(isq));
            self
        }

        pub fn with_auto_isq(mut self, bits: $crate::IsqBits) -> Self {
            self.options.runtime.isq = Some($crate::load::isq_bits(bits));
            self
        }

        pub fn with_chat_template(mut self, chat_template: impl ToString) -> Self {
            self.options.runtime.chat_template = Some(chat_template.to_string());
            self
        }

        pub fn with_jinja_explicit(mut self, jinja: impl ToString) -> Self {
            self.options.runtime.jinja_explicit = Some(jinja.to_string());
            self
        }

        pub fn with_max_model_len(mut self, max_model_len: usize) -> Self {
            self.options.runtime.max_model_len = Some(max_model_len);
            self
        }

        pub fn with_hf_config_overrides(mut self, overrides: $crate::load::Overrides) -> Self {
            self.options.runtime.hf_config_overrides = Some(overrides);
            self
        }

        pub fn with_paged_attn(mut self, cache: $crate::PagedCacheSpec) -> Self {
            self.options.runtime.paged_attn = Some(true);
            self.options.runtime.paged_cache = cache;
            self
        }

        pub fn with_mtp_model(
            mut self,
            model: impl Into<String>,
            n_predict: Option<usize>,
        ) -> Self {
            self.options.runtime.mtp = Some($crate::load::mtp(Some(model.into()), n_predict));
            self
        }

        /// MTP drafting with the head built into the checkpoint.
        pub fn with_builtin_mtp(mut self, n_predict: Option<usize>) -> Self {
            self.options.runtime.mtp = Some($crate::load::mtp(None, n_predict));
            self
        }

        /// How MTP drafts its tokens; applies to the MTP set by `with_mtp_model` or `with_builtin_mtp` before it.
        pub fn with_mtp_draft_sampling(mut self, sampling: $crate::MtpDraftSampling) -> Self {
            if let Some(mtp) = self.options.runtime.mtp.as_mut() {
                mtp.draft_sampling = sampling;
            }
            self
        }

        pub fn with_max_num_seqs(mut self, max_num_seqs: usize) -> Self {
            self.options.runtime.max_seqs = Some(max_num_seqs);
            self
        }

        /// Prefix cache capacity in sequences; `None` or 0 turns it off.
        pub fn with_prefix_cache_n(mut self, n_seqs: Option<usize>) -> Self {
            self.options.runtime.prefix_cache_n = Some(n_seqs.unwrap_or(0));
            self
        }

        pub fn with_no_kv_cache(mut self) -> Self {
            self.options.runtime.no_kv_cache = true;
            self
        }

        pub fn with_throughput_logging(mut self) -> Self {
            self.options.runtime.throughput_logging = Some(true);
            self
        }

        pub fn with_logging(mut self) -> Self {
            self.options.with_logging = true;
            self
        }

        pub fn with_seed(mut self, seed: u64) -> Self {
            self.options.runtime.seed = Some(seed);
            self
        }

        /// Reranks web search results with this embedding model.
        pub fn with_search(mut self, model: $crate::SearchEmbeddingModel) -> Self {
            self.options.agentic.search = Some($crate::load::search(model));
            self
        }

        pub fn with_search_callback(
            mut self,
            callback: std::sync::Arc<$crate::SearchCallback>,
        ) -> Self {
            self.options.search = Some(callback);
            self
        }

        /// A host tool every chat request offers the model, answered by `callback`.
        pub fn with_tool_callback_and_tool(
            mut self,
            callback: std::sync::Arc<$crate::load::ToolCallback>,
            tool: $crate::Tool,
        ) -> Self {
            self.options
                .tools
                .push($crate::load::text_tool(callback, tool));
            self
        }

        /// A tool callback with its definition, as `#[tool]` generates.
        pub fn with_tool(mut self, tool: $crate::ToolCallbackWithTool) -> Self {
            self.options.tools.push(tool);
            self
        }

        pub fn with_mcp_client(mut self, config: $crate::load::Mcp) -> Self {
            self.options.agentic.mcp = Some(config);
            self
        }

        pub fn with_code_execution(mut self, config: $crate::load::CodeExecution) -> Self {
            self.options.agentic.code_execution = Some(config);
            self
        }

        pub fn with_shell(mut self, config: $crate::load::Shell) -> Self {
            self.options.agentic.shell = Some(config);
            self
        }

        pub fn with_max_tool_rounds(mut self, rounds: usize) -> Self {
            self.options.agentic.max_tool_rounds = Some(rounds);
            self
        }

        pub fn with_agent_permission(mut self, permission: $crate::AgentPermission) -> Self {
            self.options.agentic.agent_permission = Some(permission);
            self
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_device_name_means_its_first_device() {
        assert_eq!(device("cuda".into()), "cuda:0");
        assert_eq!(device("metal".into()), "metal:0");
        assert_eq!(device("cuda:1".into()), "cuda:1");
        assert_eq!(device("cpu".into()), "cpu");
    }

    #[test]
    fn a_builder_leaves_paged_attention_off_until_asked() {
        let options = LoadOptions::new();
        assert_eq!(options.runtime.paged_attn, Some(false));
        assert_eq!(options.runtime.throughput_logging, Some(false));
        let paged =
            crate::TextModelBuilder::new("model").with_paged_attn(PagedCacheSpec::default());
        assert_eq!(paged.options.runtime.paged_attn, Some(true));
    }
}
