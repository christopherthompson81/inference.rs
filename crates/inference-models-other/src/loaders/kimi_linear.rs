use super::deepseek_family::*;
use super::*;

use crate::kimi_linear::Config;

// the checkpoint's module names: Mixtral's experts under `block_sparse_moe`, the KDA projections beside MLA's
const ISQ: [&str; 7] = [
    LM_HEAD,
    r"layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj|kv_a_proj_with_mqa|kv_b_proj|k_b_proj|v_b_proj)\.weight$",
    r"layers\.(\d+)\.self_attn\.(f_a_proj|f_b_proj|g_a_proj|g_b_proj|b_proj)\.weight$",
    r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight$",
    r"layers\.(\d+)\.block_sparse_moe\.experts\.(w1|w2|w3)\.weight$",
    r"layers\.(\d+)\.block_sparse_moe\.shared_experts\.(gate_proj|up_proj|down_proj)\.weight$",
    r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$",
];
const MOQE: [&str; 2] = [
    r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight$",
    r"layers\.(\d+)\.block_sparse_moe\.experts\.(w1|w2|w3)\.weight$",
];

fn spec(cfg: &Config) -> FamilyLoaderSpec {
    FamilyLoaderSpec {
        hidden_size: cfg.hidden_size,
        vocab_size: cfg.vocab_size,
        intermediate_size: cfg.intermediate_size,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        max_position_embeddings: cfg.model_max_length,
        tie_word_embeddings: cfg.tie_word_embeddings,
        num_kv_heads: cfg.num_attention_heads,
        k_head_dim: cfg.qk_rope_head_dim + cfg.qk_nope_head_dim,
        v_head_dim: cfg.v_head_dim,
        attention: AttentionSizing::Mla {
            q_lora_rank: cfg.q_lora_rank,
            q_head_dim: cfg.qk_rope_head_dim + cfg.qk_nope_head_dim,
            kv_lora_rank: cfg.kv_lora_rank,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
            v_head_dim: cfg.v_head_dim,
            attention_bias: false,
        },
        moe: Some(MoeSizing {
            n_routed_experts: cfg.num_experts,
            moe_intermediate_size: cfg.moe_intermediate_size,
            first_k_dense_replace: cfg.first_k_dense_replace,
            moe_layer_freq: Some(cfg.moe_layer_freq),
            shared_intermediate: (cfg.num_shared_experts > 0)
                .then_some(cfg.moe_intermediate_size * cfg.num_shared_experts),
            correction_bias: true,
        }),
        isq_head: Vec::new(),
    }
}

// a KDA layer's q, k, v and o projections, the decay and gate LoRAs, beta's projection, its convs and vectors
fn kda_elems(cfg: &Config, weight_pack_factor: usize) -> usize {
    let kda = cfg.kda();
    let (h, w, n) = (cfg.hidden_size, kda.width(), kda.head_dim);
    let projections = (4 * h * w + 2 * (h * n + n * w) + h * kda.heads) / weight_pack_factor;
    projections + kda.conv_dim() * kda.conv_kernel + w + kda.heads + n
}

/// `NormalLoader` for Kimi-Linear (`KimiLinearForCausalLM`).
pub struct KimiLinearLoader;

impl NormalModelLoader for KimiLinearLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(crate::kimi_linear::Model::new(
            &Config::from_json(config)?,
            vb,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(Config::from_json(config)?))
    }
}

impl IsqModelLoader for KimiLinearLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        FamilyLoaderSpec::promoted_isq_predicates()
    }
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&ISQ)
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&MOQE)
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for KimiLinearLoader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        spec(&Config::from_json(config)?).non_mapped_size_in_bytes(
            dtype,
            weight_pack_factor,
            quantization,
        )
    }
    // the family's MLA and MoE sizing, with a KDA layer's own attention in place of MLA's
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = Config::from_json(config)?;
        let spec = spec(&cfg);
        let (mla, kda) = (
            spec.attention_elems(weight_pack_factor),
            kda_elems(&cfg, weight_pack_factor),
        );
        Ok((0..cfg.num_hidden_layers)
            .map(|layer| {
                let attention = if cfg.is_full_attention(layer) {
                    mla
                } else {
                    kda
                };
                (2 * cfg.hidden_size + attention + spec.mlp_elems(layer, weight_pack_factor))
                    * dtype.size_in_bytes()
            })
            .collect())
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Ok(Config::from_json(config)?.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = Config::from_json(config)?;
        let paged = (0..cfg.num_hidden_layers)
            .map(|layer| cfg.is_full_attention(layer))
            .collect();
        Ok(Box::new(
            inference_nn::paged_attention::HybridPagedKvCacheConfig::new(
                spec(&cfg).metadata(),
                paged,
            )
            .with_recurrent_state_bytes(cfg.recurrent_state_bytes(DType::F32)),
        ))
    }
}
