use inference_quant::QuantizedConfig;

use crate::gdn::{GdnOutputGate, GdnStateDType, GdnVHeadLayout};
use crate::layers::{Activation, YarnRopeConfig};
use crate::paged_attention::AuxiliaryPagedLayers;
use crate::serde_default_fn;

// Re-export vision config from qwen3_vl
pub use crate::vision::qwen3_vl::VisionConfig;

serde_default_fn!(usize, default_full_attn_interval, 4);
serde_default_fn!(usize, default_conv_kernel, 4);
serde_default_fn!(f64, default_partial_rotary_factor, 0.25);
serde_default_fn!(f64, default_rope_theta, 10_000_000.0);
serde_default_fn!(f64, default_yarn_beta_fast, 32.0);
serde_default_fn!(f64, default_yarn_beta_slow, 1.0);
serde_default_fn!(bool, default_true, true);
serde_default_fn!(usize, default_ple_conv_kernel, 4);
serde_default_fn!(usize, default_ngram_size, 3);
serde_default_fn!(usize, default_heads_per_ngram, 8);
serde_default_fn!(u64, default_ngram_seed, 1234);
serde_default_fn!(u64, default_ngram_vocab_base, 20_000_000);
serde_default_fn!(usize, default_ngram_vocab_divisor, 128);

#[derive(Debug, Default, Clone, Copy, PartialEq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RopeType {
    #[default]
    #[serde(alias = "mrope")]
    Default,
    Yarn,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    // Qwen3.8-Flash-Next names its QSA layers `indexed_attention` in transformers
    #[serde(alias = "indexed_attention")]
    FullAttention,
    LinearAttention,
}

/// Qwen3.8-Flash-Next's additions to the Qwen3.5 MoE text model; all absent on Qwen3.5 and Qwen3.8 checkpoints.
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct FlashNextConfig {
    // hyper-connections: `hc_count` residual streams in place of every layer norm
    #[serde(default)]
    pub hc_count: Option<usize>,
    #[serde(default)]
    pub hc_lowrank: Option<usize>,
    #[serde(default)]
    pub output_gate_type: Option<GdnOutputGate>,
    // PLE: hashed n-gram embeddings into the streams before these layers, 1-based as in the checkpoint
    #[serde(default)]
    pub ple_layer_ids: Vec<usize>,
    #[serde(default)]
    pub ple_embed_dim: Option<usize>,
    #[serde(default = "default_ple_conv_kernel")]
    pub ple_conv_kernel_size: usize,
    #[serde(default = "default_ngram_size")]
    pub ngram_size: usize,
    #[serde(default = "default_heads_per_ngram")]
    pub heads_per_ngram: usize,
    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,
    #[serde(default = "default_ngram_seed")]
    pub seed: u64,
    #[serde(default = "default_ngram_vocab_base")]
    pub ngram_vocab_size_base: u64,
    #[serde(default = "default_ngram_vocab_divisor")]
    pub make_ngram_vocab_size_divisible_by: usize,
    // the hash constants a GGUF carries; absent, they are derived from the fields above as the reference does
    #[serde(default, rename = "_inference_ple_hash")]
    pub ple_hash: Option<PleHash>,
    // QSA: a block indexer selecting attention keys on the full-attention layers
    #[serde(default)]
    pub indexer_n_heads: Option<usize>,
    #[serde(default)]
    pub indexer_kv_heads: Option<usize>,
    #[serde(default)]
    pub indexer_head_dim: Option<usize>,
    #[serde(default)]
    pub indexer_budget: Option<usize>,
    #[serde(default)]
    pub indexer_compress_ratio: Option<usize>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
    One(u32),
    Many(Vec<u32>),
}

impl EosTokenId {
    // the reference resets its n-grams on the first
    fn primary(&self) -> Option<u32> {
        match self {
            Self::One(id) => Some(*id),
            Self::Many(ids) => ids.first().copied(),
        }
    }
}

/// PLE's n-gram hash: one odd multiplier per n-gram position, and each hash head's table size (a prime) and offset.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct PleHash {
    pub layer_multipliers: Vec<u64>,
    pub head_offsets: Vec<u64>,
    pub head_vocab_sizes: Vec<u64>,
}

/// The hyper-connection streams and their mixing rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyperConnections {
    pub streams: usize,
    pub lowrank: usize,
}

/// PLE geometry: which layer reads it, the n-gram heads and their embedding width, the conv and the EOS it resets on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PleConfig {
    pub layer: usize,
    pub embed_dim: usize,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub conv_kernel: usize,
    pub eos_token_id: u32,
}

impl PleConfig {
    pub fn ngram_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.ngram_heads()
    }

    // the conv's dilation is the n-gram size, so it reaches back (kernel - 1) * ngram_size positions
    pub fn conv_state_len(&self) -> usize {
        (self.conv_kernel - 1) * self.ngram_size
    }
}

/// The QSA indexer on the full-attention layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QsaConfig {
    pub n_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub budget: usize,
    pub compress_ratio: usize,
}

impl QsaConfig {
    /// The longest context over which QSA keeps every key: all complete blocks fit the budget, plus a partial tail.
    pub fn dense_context_len(&self) -> usize {
        self.budget + self.compress_ratio - 1
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParameters {
    #[serde(default, alias = "type")]
    pub rope_type: RopeType,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    pub mrope_section: Vec<usize>,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
    #[serde(default = "default_true")]
    pub mrope_interleaved: bool,
    #[serde(default)]
    pub factor: Option<f64>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    #[serde(default = "default_yarn_beta_fast")]
    pub beta_fast: f64,
    #[serde(default = "default_yarn_beta_slow")]
    pub beta_slow: f64,
    #[serde(default)]
    pub mscale: Option<f64>,
    #[serde(default)]
    pub mscale_all_dim: Option<f64>,
    #[serde(default)]
    pub attention_factor: Option<f64>,
    #[serde(default = "default_true")]
    pub truncate: bool,
}

impl RopeParameters {
    pub fn supported_max_position_embeddings(
        &self,
        declared_max_position_embeddings: usize,
    ) -> inference_tensor::Result<usize> {
        if self.rope_type != RopeType::Yarn {
            return Ok(declared_max_position_embeddings);
        }
        self.validate_scaling(declared_max_position_embeddings)?;
        let supported = self
            .original_max_position_embeddings
            .expect("validated YaRN original context length") as f64
            * self.factor.expect("validated YaRN factor");
        if supported > usize::MAX as f64 {
            inference_tensor::bail!("Qwen3.5 YaRN context length overflows usize");
        }
        Ok(supported.floor() as usize)
    }

    pub fn validate_scaling(&self, max_position_embeddings: usize) -> inference_tensor::Result<()> {
        if self.rope_type != RopeType::Yarn {
            return Ok(());
        }
        if !self.truncate {
            inference_tensor::bail!("Qwen3.5 YaRN truncate=false is not supported");
        }
        let factor = self
            .factor
            .ok_or_else(|| inference_tensor::Error::msg("Qwen3.5 YaRN requires a factor"))?;
        if !factor.is_finite() || factor < 1.0 {
            inference_tensor::bail!("Qwen3.5 YaRN factor must be finite and at least 1");
        }
        let original = self.original_max_position_embeddings.ok_or_else(|| {
            inference_tensor::Error::msg("Qwen3.5 YaRN requires original_max_position_embeddings")
        })?;
        if original == 0 {
            inference_tensor::bail!("Qwen3.5 YaRN original context length must be positive");
        }
        let supported = original as f64 * factor;
        if max_position_embeddings as f64 > supported {
            inference_tensor::bail!(
                "Qwen3.5 maximum position embeddings {max_position_embeddings} exceed the YaRN limit {supported:.0}"
            );
        }
        if !self.beta_fast.is_finite()
            || self.beta_fast <= 0.0
            || !self.beta_slow.is_finite()
            || self.beta_slow <= 0.0
        {
            inference_tensor::bail!("Qwen3.5 YaRN beta values must be finite and positive");
        }
        if self.beta_fast < self.beta_slow {
            inference_tensor::bail!("Qwen3.5 YaRN beta_fast must be at least beta_slow");
        }
        if self
            .mscale
            .is_some_and(|mscale| !mscale.is_finite() || mscale < 0.0)
            || self
                .mscale_all_dim
                .is_some_and(|mscale| !mscale.is_finite() || mscale < 0.0)
        {
            inference_tensor::bail!("Qwen3.5 YaRN mscale values must be finite and non-negative");
        }
        if self
            .attention_factor
            .is_some_and(|factor| !factor.is_finite() || factor <= 0.0)
        {
            inference_tensor::bail!("Qwen3.5 YaRN attention factor must be finite and positive");
        }
        Ok(())
    }

    pub fn yarn_rope_config(
        &self,
        max_position_embeddings: usize,
        head_dim: usize,
    ) -> inference_tensor::Result<Option<YarnRopeConfig>> {
        self.validate_scaling(max_position_embeddings)?;
        if self.rope_type != RopeType::Yarn {
            return Ok(None);
        }
        Ok(Some(YarnRopeConfig {
            base: self.rope_theta as f32,
            head_dim,
            max_position_embeddings,
            original_max_position_embeddings: self
                .original_max_position_embeddings
                .expect("validated YaRN original context length"),
            factor: self.factor.expect("validated YaRN factor") as f32,
            beta_fast: self.beta_fast as f32,
            beta_slow: self.beta_slow as f32,
            mscale: match (self.mscale, self.mscale_all_dim) {
                (Some(mscale), Some(mscale_all_dim)) if mscale != 0.0 && mscale_all_dim != 0.0 => {
                    mscale as f32
                }
                _ => 1.0,
            },
            mscale_all_dim: match (self.mscale, self.mscale_all_dim) {
                (Some(mscale), Some(mscale_all_dim)) if mscale != 0.0 && mscale_all_dim != 0.0 => {
                    mscale_all_dim as f32
                }
                _ => 0.0,
            },
            attention_factor: self.attention_factor.map(|x| x as f32),
        }))
    }
}

pub fn apply_max_model_len(config: &str, max_model_len: usize) -> inference_tensor::Result<String> {
    if max_model_len == 0 {
        inference_tensor::bail!("Qwen3.5 max_model_len must be positive");
    }
    let mut config: serde_json::Value = serde_json::from_str(config)
        .map_err(|err| inference_tensor::Error::msg(err.to_string()))?;
    let root = config
        .as_object_mut()
        .ok_or_else(|| inference_tensor::Error::msg("Qwen3.5 config must be a JSON object"))?;
    let text_config = if root.contains_key("text_config") {
        root.get_mut("text_config")
            .expect("checked text_config presence")
            .as_object_mut()
            .ok_or_else(|| {
                inference_tensor::Error::msg("Qwen3.5 text_config must be a JSON object")
            })?
    } else {
        root
    };
    let declared_max = text_config
        .get("max_position_embeddings")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            inference_tensor::Error::msg(
                "Qwen3.5 max_position_embeddings must be a positive integer",
            )
        })?;
    let rope_parameters: RopeParameters = serde_json::from_value(
        text_config
            .get("rope_parameters")
            .ok_or_else(|| inference_tensor::Error::msg("Qwen3.5 rope_parameters are required"))?
            .clone(),
    )
    .map_err(|err| inference_tensor::Error::msg(err.to_string()))?;
    let supported = rope_parameters.supported_max_position_embeddings(declared_max)?;
    if max_model_len > supported {
        inference_tensor::bail!(
            "Qwen3.5 max_model_len {max_model_len} exceeds the model-supported context length {supported}"
        );
    }
    text_config.insert(
        "max_position_embeddings".to_string(),
        serde_json::Value::from(max_model_len),
    );
    serde_json::to_string(&config).map_err(|err| inference_tensor::Error::msg(err.to_string()))
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TextConfig {
    pub head_dim: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    // MoE checkpoints omit it; every dense MLP needs it
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,
    // Hybrid attention fields
    #[serde(default = "default_full_attn_interval")]
    pub full_attention_interval: usize,
    #[serde(default)]
    pub layer_types: Option<Vec<LayerType>>,
    #[serde(default = "default_conv_kernel")]
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub mamba_ssm_dtype: GdnStateDType,
    // Sparse feed-forward; zero experts makes every layer a dense MLP
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
    // Other
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default, rename = "_inference_gdn_v_head_layout")]
    pub gdn_v_head_layout: GdnVHeadLayout,
    // Multi-token prediction head shipped in the checkpoint (`mtp.*` weights)
    #[serde(default)]
    pub mtp_num_hidden_layers: usize,
    #[serde(default)]
    pub mtp_use_dedicated_embeddings: bool,
    #[serde(flatten)]
    pub flash_next: FlashNextConfig,
}

impl TextConfig {
    pub fn validate(&self) -> inference_tensor::Result<()> {
        if self.max_position_embeddings == 0 {
            inference_tensor::bail!("Qwen3.5 maximum position embeddings must be positive");
        }
        if !self.rope_theta().is_finite() || self.rope_theta() <= 0.0 {
            inference_tensor::bail!("Qwen3.5 rope theta must be finite and positive");
        }
        if !self.partial_rotary_factor().is_finite()
            || self.partial_rotary_factor() <= 0.0
            || self.partial_rotary_factor() > 1.0
        {
            inference_tensor::bail!("Qwen3.5 partial rotary factor must be in (0, 1]");
        }
        if self.num_hidden_layers == 0 {
            inference_tensor::bail!("Qwen3.5 requires at least one hidden layer");
        }
        if self.full_attention_interval == 0
            || self.full_attention_interval > self.num_hidden_layers
        {
            inference_tensor::bail!(
                "Qwen3.5 full_attention_interval {} is invalid for {} layers",
                self.full_attention_interval,
                self.num_hidden_layers
            );
        }
        if let Some(layer_types) = &self.layer_types
            && (layer_types.len() != self.num_hidden_layers
                || !layer_types.contains(&LayerType::FullAttention))
        {
            inference_tensor::bail!(
                "Qwen3.5 layer_types must list {} layers with at least one full_attention entry",
                self.num_hidden_layers
            );
        }
        if self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            inference_tensor::bail!(
                "Qwen3.5 has incompatible attention head counts: {} query and {} KV",
                self.num_attention_heads,
                self.num_key_value_heads
            );
        }
        if self.linear_num_key_heads == 0
            || self.linear_num_value_heads == 0
            || !self
                .linear_num_value_heads
                .is_multiple_of(self.linear_num_key_heads)
        {
            inference_tensor::bail!(
                "Qwen3.5 has incompatible GDN head counts: {} key and {} value",
                self.linear_num_key_heads,
                self.linear_num_value_heads
            );
        }
        if self.linear_key_head_dim == 0
            || self.linear_value_head_dim == 0
            || self.linear_conv_kernel_dim == 0
        {
            inference_tensor::bail!("Qwen3.5 GDN dimensions must be non-zero");
        }
        let rot_dim = self.rot_dim();
        if rot_dim == 0 || rot_dim > self.head_dim || !rot_dim.is_multiple_of(2) {
            inference_tensor::bail!(
                "Qwen3.5 rotary dimension {rot_dim} must be positive, even, and no larger than head_dim {}",
                self.head_dim
            );
        }
        if self.mrope_section().len() != 3 || self.mrope_section().contains(&0) {
            inference_tensor::bail!("Qwen3.5 MRoPE requires three non-zero sections");
        }
        if !self.rope_parameters.mrope_interleaved {
            inference_tensor::bail!("Qwen3.5 requires interleaved MRoPE");
        }
        let section_width = self
            .mrope_section()
            .iter()
            .try_fold(0usize, |sum, section| sum.checked_add(*section))
            .ok_or_else(|| inference_tensor::Error::msg("Qwen3.5 MRoPE section width overflow"))?;
        if section_width.checked_mul(2) != Some(rot_dim) {
            inference_tensor::bail!(
                "Qwen3.5 MRoPE sections span {} dimensions, expected {rot_dim}",
                section_width.saturating_mul(2)
            );
        }
        self.validate_feed_forward()?;
        self.validate_rope_scaling()?;
        self.validate_flash_next()?;
        Ok(())
    }

    fn validate_flash_next(&self) -> inference_tensor::Result<()> {
        let fields = &self.flash_next;
        if fields.hc_count.is_some() != fields.hc_lowrank.is_some()
            || fields.hc_count == Some(0)
            || fields.hc_lowrank == Some(0)
        {
            inference_tensor::bail!(
                "Qwen3.8-Flash-Next needs a nonzero hc_count and hc_lowrank together"
            );
        }
        let qsa = [
            fields.indexer_n_heads,
            fields.indexer_kv_heads,
            fields.indexer_head_dim,
            fields.indexer_budget,
            fields.indexer_compress_ratio,
        ];
        if qsa.iter().any(Option::is_some) && qsa.iter().any(|field| field.is_none_or(|v| v == 0)) {
            inference_tensor::bail!(
                "Qwen3.8-Flash-Next's QSA indexer needs every indexer_* field, nonzero"
            );
        }
        if let Some(qsa) = self.qsa()
            && (qsa.kv_heads != 1 || qsa.compress_ratio < 2 || qsa.budget % qsa.compress_ratio != 0)
        {
            inference_tensor::bail!(
                "QSA takes one indexer key head and a block size above 1 that divides the budget, got {qsa:?}"
            );
        }
        if let Some(ple) = self.ple()? {
            if self.hyper_connections().is_none() {
                inference_tensor::bail!(
                    "PLE writes into hyper-connection streams, which this config lacks"
                );
            }
            if self.layer_types().get(ple.layer) != Some(&LayerType::LinearAttention) {
                inference_tensor::bail!(
                    "PLE layer {} must be a linear-attention layer",
                    ple.layer + 1
                );
            }
            if ple.ngram_size < 2
                || ple.heads_per_ngram == 0
                || ple.conv_kernel == 0
                || !ple.embed_dim.is_multiple_of(ple.ngram_heads())
            {
                inference_tensor::bail!(
                    "PLE embedding width {} must split evenly over {} n-gram heads",
                    ple.embed_dim,
                    ple.ngram_heads()
                );
            }
        }
        Ok(())
    }

    pub fn hyper_connections(&self) -> Option<HyperConnections> {
        Some(HyperConnections {
            streams: self.flash_next.hc_count?,
            lowrank: self.flash_next.hc_lowrank?,
        })
    }

    /// The residual's width: `hc_count` streams of `hidden_size`, or one.
    pub fn residual_size(&self) -> usize {
        self.hyper_connections().map_or(1, |hc| hc.streams) * self.hidden_size
    }

    pub fn ple(&self) -> inference_tensor::Result<Option<PleConfig>> {
        let fields = &self.flash_next;
        match fields.ple_layer_ids.as_slice() {
            [] => Ok(None),
            [layer] if (1..=self.num_hidden_layers).contains(layer) => Ok(Some(PleConfig {
                layer: layer - 1,
                embed_dim: fields.ple_embed_dim.unwrap_or(self.hidden_size),
                ngram_size: fields.ngram_size,
                heads_per_ngram: fields.heads_per_ngram,
                conv_kernel: fields.ple_conv_kernel_size,
                eos_token_id: fields
                    .eos_token_id
                    .as_ref()
                    .and_then(EosTokenId::primary)
                    .ok_or_else(|| {
                        inference_tensor::Error::msg(
                            "PLE resets its n-grams on EOS, which the config lacks",
                        )
                    })?,
            })),
            layers => inference_tensor::bail!(
                "PLE on layers {layers:?} is unsupported: only one layer, in 1..={}",
                self.num_hidden_layers
            ),
        }
    }

    pub fn qsa(&self) -> Option<QsaConfig> {
        let fields = &self.flash_next;
        Some(QsaConfig {
            n_heads: fields.indexer_n_heads?,
            kv_heads: fields.indexer_kv_heads?,
            head_dim: fields.indexer_head_dim?,
            budget: fields.indexer_budget?,
            compress_ratio: fields.indexer_compress_ratio?,
        })
    }

    fn validate_feed_forward(&self) -> inference_tensor::Result<()> {
        if !self.is_moe() {
            self.dense_intermediate_size()?;
            return Ok(());
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_experts {
            inference_tensor::bail!(
                "Qwen3.5 has invalid MoE routing: {} of {} experts",
                self.num_experts_per_tok,
                self.num_experts
            );
        }
        if self.moe_intermediate_size == 0 || self.shared_expert_intermediate_size == 0 {
            inference_tensor::bail!(
                "Qwen3.5 MoE needs nonzero moe_intermediate_size and shared_expert_intermediate_size"
            );
        }
        if !self.mlp_only_layers.is_empty() {
            inference_tensor::bail!(
                "Qwen3.5 MoE `mlp_only_layers` is not implemented yet in inference.rs."
            );
        }
        Ok(())
    }

    /// Every decoder layer is sparse when the config names experts.
    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }

    pub fn dense_intermediate_size(&self) -> inference_tensor::Result<usize> {
        match self.intermediate_size {
            Some(size) if size > 0 => Ok(size),
            _ => inference_tensor::bail!("Qwen3.5 dense MLP needs a positive intermediate_size"),
        }
    }

    /// Rejects a checkpoint whose experts do not match the loader it was given.
    pub fn check_experts(&self, moe: bool) -> inference_tensor::Result<()> {
        match (moe, self.is_moe()) {
            (false, true) => inference_tensor::bail!(
                "text_config names {} experts: load it as qwen3_5moe",
                self.num_experts
            ),
            (true, false) => {
                inference_tensor::bail!("Qwen3.5 MoE needs nonzero num_experts in text_config")
            }
            _ => Ok(()),
        }
    }

    fn validate_rope_scaling(&self) -> inference_tensor::Result<()> {
        self.rope_parameters
            .validate_scaling(self.max_position_embeddings)
    }

    pub fn yarn_rope_config(&self) -> inference_tensor::Result<Option<YarnRopeConfig>> {
        self.rope_parameters
            .yarn_rope_config(self.max_position_embeddings, self.rot_dim())
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters.rope_theta
    }

    pub fn partial_rotary_factor(&self) -> f64 {
        self.rope_parameters.partial_rotary_factor
    }

    pub fn mrope_section(&self) -> &[usize] {
        &self.rope_parameters.mrope_section
    }

    pub fn layer_types(&self) -> Vec<LayerType> {
        if let Some(layer_types) = &self.layer_types {
            return layer_types.clone();
        }
        (0..self.num_hidden_layers)
            .map(|i| {
                if (i + 1) % self.full_attention_interval == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect()
    }

    pub fn linear_key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    pub fn linear_value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    pub fn linear_conv_dim(&self) -> usize {
        2 * self.linear_key_dim() + self.linear_value_dim()
    }

    /// One sequence's recurrent state over the linear-attention layers, as the hybrid cache keeps it (conv rows in the
    /// activation dtype, taken as BF16 since they are a small share).
    pub fn recurrent_state_bytes(&self) -> usize {
        let linear = self
            .layer_types()
            .into_iter()
            .filter(|ty| *ty == LayerType::LinearAttention)
            .count();
        let state = self.linear_num_value_heads
            * self.linear_key_head_dim
            * self.linear_value_head_dim
            * self.mamba_ssm_dtype.dtype().size_in_bytes();
        let conv = self.linear_conv_dim()
            * self.linear_conv_kernel_dim
            * inference_tensor::DType::BF16.size_in_bytes();
        // PLE keeps its conv's last positions of every stream channel, and a one-value placeholder state
        let ple = self.ple().ok().flatten().map_or(0, |ple| {
            self.residual_size()
                * ple.conv_state_len()
                * inference_tensor::DType::BF16.size_in_bytes()
                + inference_tensor::DType::F32.size_in_bytes()
        });
        linear * (state + conv) + ple
    }

    /// The longest context the model runs exactly: without QSA's sparse selection (paged attention on CUDA), only
    /// as far as QSA keeps every key.
    pub fn supported_context_len(&self, sparse_attention: bool) -> usize {
        match self.qsa() {
            Some(qsa) if !sparse_attention => {
                qsa.dense_context_len().min(self.max_position_embeddings)
            }
            _ => self.max_position_embeddings,
        }
    }

    /// QSA's indexer keys: one auxiliary paged layer per full-attention layer of the main stack, after the model's own.
    pub fn qsa_auxiliary_layers(&self, mtp: bool) -> Option<AuxiliaryPagedLayers> {
        let qsa = self.qsa()?;
        Some(AuxiliaryPagedLayers {
            // the main stack's full-attention layers, then the MTP blocks' (each one QSA layer)
            count: self.full_attention_layers() + self.mtp_layers(mtp),
            num_attn_heads: qsa.n_heads,
            num_kv_heads: qsa.kv_heads,
            head_dim: qsa.head_dim,
        })
    }

    /// The paged cache layer holding `layer_idx`'s indexer keys.
    fn full_attention_layers(&self) -> usize {
        self.layer_types()
            .iter()
            .filter(|ty| **ty == LayerType::FullAttention)
            .count()
    }

    /// The paged layer of the first MTP block's indexer keys, after the main stack's.
    pub fn mtp_qsa_cache_layer(&self) -> usize {
        self.num_hidden_layers + self.mtp_num_hidden_layers + self.full_attention_layers()
    }

    pub fn qsa_cache_layer(&self, layer_idx: usize, mtp: bool) -> usize {
        let rank = self.layer_types()[..layer_idx]
            .iter()
            .filter(|ty| **ty == LayerType::FullAttention)
            .count();
        self.num_hidden_layers + self.mtp_layers(mtp) + rank
    }

    pub fn rot_dim(&self) -> usize {
        (self.head_dim as f64 * self.partial_rotary_factor()) as usize
    }

    /// Absolute layer index of the first MTP block; its paged KV cache lives past the main stack.
    pub fn mtp_layer_idx(&self) -> usize {
        self.num_hidden_layers
    }

    pub fn mtp_layers(&self, mtp: bool) -> usize {
        if mtp { self.mtp_num_hidden_layers } else { 0 }
    }

    /// Paged-KV mask over the main stack plus any MTP blocks appended after it.
    pub fn paged_kv_layers(&self, mtp: bool) -> Vec<bool> {
        let mut layers = self
            .layer_types()
            .into_iter()
            .map(|ty| ty == LayerType::FullAttention)
            .collect::<Vec<_>>();
        layers.extend(std::iter::repeat_n(true, self.mtp_layers(mtp)));
        layers
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub tie_word_embeddings: bool,
    /// Top-level quantization_config takes precedence
    pub quantization_config: Option<QuantizedConfig>,
    /// Injected by the loader when the built-in MTP head should be loaded (see `MTP_CONFIG_KEY`).
    #[serde(default, rename = "_inference_mtp")]
    pub mtp: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_config(
        max_position_embeddings: usize,
        rope_parameters: serde_json::Value,
    ) -> TextConfig {
        serde_json::from_value(serde_json::json!({
            "head_dim": 64,
            "vocab_size": 32,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 8,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_act": "silu",
            "max_position_embeddings": max_position_embeddings,
            "rms_norm_eps": 1e-6,
            "rope_parameters": rope_parameters,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4
        }))
        .unwrap()
    }

    fn moe_text_config(patch: serde_json::Value) -> serde_json::Value {
        let mut cfg = serde_json::json!({
            "head_dim": 64,
            "vocab_size": 32,
            "hidden_size": 128,
            "num_hidden_layers": 8,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_act": "silu",
            "max_position_embeddings": 1024,
            "rms_norm_eps": 1e-6,
            "rope_parameters": { "mrope_section": [4, 2, 2] },
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 64,
            "num_experts": 4,
            "num_experts_per_tok": 2
        });
        let (serde_json::Value::Object(cfg_map), serde_json::Value::Object(patch)) =
            (&mut cfg, patch)
        else {
            unreachable!("configs are JSON objects")
        };
        cfg_map.extend(patch);
        cfg
    }

    fn validation_error(cfg: serde_json::Value) -> String {
        serde_json::from_value::<TextConfig>(cfg)
            .unwrap()
            .validate()
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn moe_config_needs_no_dense_intermediate_size() {
        let cfg: TextConfig =
            serde_json::from_value(moe_text_config(serde_json::json!({}))).unwrap();
        cfg.validate().unwrap();
        assert!(cfg.is_moe());
        assert!(cfg.norm_topk_prob);
        cfg.check_experts(true).unwrap();
        assert!(
            cfg.check_experts(false)
                .unwrap_err()
                .to_string()
                .contains("load it as qwen3_5moe")
        );
    }

    #[test]
    fn dense_config_without_intermediate_size_is_rejected() {
        let error = validation_error(moe_text_config(serde_json::json!({ "num_experts": 0 })));
        assert!(error.contains("positive intermediate_size"), "{error}");
        let cfg: TextConfig =
            serde_json::from_value(moe_text_config(serde_json::json!({ "num_experts": 0 })))
                .unwrap();
        assert!(
            cfg.check_experts(true)
                .unwrap_err()
                .to_string()
                .contains("nonzero num_experts")
        );
    }

    #[test]
    fn moe_config_rejects_inconsistent_experts() {
        for patch in [
            serde_json::json!({ "num_experts_per_tok": 0 }),
            serde_json::json!({ "num_experts_per_tok": 5 }),
        ] {
            assert!(validation_error(moe_text_config(patch)).contains("invalid MoE routing"));
        }
        for patch in [
            serde_json::json!({ "moe_intermediate_size": 0 }),
            serde_json::json!({ "shared_expert_intermediate_size": 0 }),
        ] {
            assert!(
                validation_error(moe_text_config(patch)).contains("nonzero moe_intermediate_size")
            );
        }
        assert!(
            validation_error(moe_text_config(
                serde_json::json!({ "mlp_only_layers": [0] })
            ))
            .contains("mlp_only_layers")
        );
    }

    #[test]
    fn gdn_head_layout_defaults_to_grouped_and_accepts_tiled_override() {
        use crate::gdn::GdnConfig;
        let default: TextConfig =
            serde_json::from_value(moe_text_config(serde_json::json!({}))).unwrap();
        assert_eq!(default.v_head_layout(), GdnVHeadLayout::Grouped);
        let tiled: TextConfig = serde_json::from_value(moe_text_config(
            serde_json::json!({ "_inference_gdn_v_head_layout": "tiled" }),
        ))
        .unwrap();
        assert_eq!(tiled.v_head_layout(), GdnVHeadLayout::Tiled);
    }

    #[test]
    fn mtp_layers_extend_the_paged_kv_mask() {
        let cfg: TextConfig = serde_json::from_str(
            r#"{
                "head_dim": 64, "vocab_size": 32, "hidden_size": 128, "intermediate_size": 256,
                "num_hidden_layers": 8, "num_attention_heads": 4, "num_key_value_heads": 2,
                "hidden_act": "silu", "max_position_embeddings": 1024, "rms_norm_eps": 1e-6,
                "rope_parameters": { "mrope_section": [8, 4, 4] },
                "linear_key_head_dim": 16, "linear_value_head_dim": 16,
                "linear_num_key_heads": 2, "linear_num_value_heads": 4,
                "mtp_num_hidden_layers": 1
            }"#,
        )
        .unwrap();
        let base = cfg.paged_kv_layers(false);
        assert_eq!(base.len(), 8);
        assert_eq!(base.iter().filter(|x| **x).count(), 2);
        let with_mtp = cfg.paged_kv_layers(true);
        assert_eq!(with_mtp.len(), 9);
        assert!(with_mtp[8]);
        assert_eq!(cfg.mtp_layer_idx(), 8);
        assert_eq!(cfg.mtp_layers(true), 1);
        assert_eq!(cfg.mtp_layers(false), 0);
    }

    #[test]
    fn native_rope_remains_unscaled() {
        let cfg = text_config(
            262_144,
            serde_json::json!({
                "rope_type": "default",
                "rope_theta": 10_000_000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [4, 2, 2]
            }),
        );
        cfg.validate().unwrap();
        assert!(cfg.yarn_rope_config().unwrap().is_none());
        assert_eq!(
            cfg.rope_parameters
                .supported_max_position_embeddings(cfg.max_position_embeddings)
                .unwrap(),
            262_144
        );
    }

    #[test]
    fn legacy_mrope_type_is_canonicalized_to_default() {
        let cfg = text_config(
            262_144,
            serde_json::json!({
                "rope_type": "mrope",
                "mrope_section": [4, 2, 2]
            }),
        );
        cfg.validate().unwrap();
        assert_eq!(cfg.rope_parameters.rope_type, RopeType::Default);
        assert!(cfg.yarn_rope_config().unwrap().is_none());
    }

    #[test]
    fn yarn_factor_four_accepts_qwen_ultra_long_limit() {
        let cfg = text_config(
            1_010_000,
            serde_json::json!({
                "rope_type": "yarn",
                "rope_theta": 10_000_000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [4, 2, 2],
                "factor": 4.0,
                "original_max_position_embeddings": 262144
            }),
        );
        cfg.validate().unwrap();
        let yarn = cfg.yarn_rope_config().unwrap().unwrap();
        assert_eq!(yarn.original_max_position_embeddings, 262_144);
        assert_eq!(yarn.factor, 4.0);
        assert_eq!(yarn.max_position_embeddings, 1_010_000);
        assert!((yarn.mscale - 1.0).abs() < f32::EPSILON);
        assert!((yarn.mscale_all_dim - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn yarn_rejects_length_above_derived_limit() {
        let cfg = text_config(
            1_048_577,
            serde_json::json!({
                "rope_type": "yarn",
                "mrope_section": [4, 2, 2],
                "factor": 4.0,
                "original_max_position_embeddings": 262144
            }),
        );
        let error = cfg.validate().unwrap_err().to_string();
        assert!(error.contains("exceed the YaRN limit"));
    }

    #[test]
    fn yarn_rejects_unimplemented_non_truncating_correction_range() {
        let cfg = text_config(
            524_288,
            serde_json::json!({
                "rope_type": "yarn",
                "mrope_section": [4, 2, 2],
                "factor": 2.0,
                "original_max_position_embeddings": 262144,
                "truncate": false
            }),
        );
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("truncate=false")
        );
    }

    #[test]
    fn yarn_rejects_reversed_beta_range() {
        let cfg = text_config(
            524_288,
            serde_json::json!({
                "rope_type": "yarn",
                "mrope_section": [4, 2, 2],
                "factor": 2.0,
                "original_max_position_embeddings": 262144,
                "beta_fast": 1.0,
                "beta_slow": 32.0
            }),
        );
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("beta_fast")
        );
    }

    #[test]
    fn yarn_zero_mscale_uses_hf_attention_default() {
        let cfg = text_config(
            524_288,
            serde_json::json!({
                "rope_type": "yarn",
                "mrope_section": [4, 2, 2],
                "factor": 2.0,
                "original_max_position_embeddings": 262144,
                "mscale": 0.0,
                "mscale_all_dim": 1.0
            }),
        );
        let yarn = cfg.yarn_rope_config().unwrap().unwrap();
        assert_eq!(yarn.mscale, 1.0);
        assert_eq!(yarn.mscale_all_dim, 0.0);
    }

    #[test]
    fn runtime_limit_handles_nested_and_text_configs() {
        let text = serde_json::json!({
            "max_position_embeddings": 262144,
            "rope_parameters": {
                "rope_type": "yarn",
                "mrope_section": [4, 2, 2],
                "factor": 4.0,
                "original_max_position_embeddings": 262144
            }
        });
        let nested = apply_max_model_len(
            &serde_json::json!({ "text_config": text.clone(), "vision_config": {} }).to_string(),
            1_010_000,
        )
        .unwrap();
        let nested: serde_json::Value = serde_json::from_str(&nested).unwrap();
        assert_eq!(nested["text_config"]["max_position_embeddings"], 1_010_000);
        assert_eq!(nested["vision_config"], serde_json::json!({}));

        let standalone = apply_max_model_len(&text.to_string(), 524_288).unwrap();
        let standalone: serde_json::Value = serde_json::from_str(&standalone).unwrap();
        assert_eq!(standalone["max_position_embeddings"], 524_288);
        assert!(apply_max_model_len(&text.to_string(), 1_048_577).is_err());
    }
}
