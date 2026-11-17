use inference_nn::vision::sam::SamConfig;
use inference_quant::QuantizedConfig;
use inference_tensor::Result;

use crate::{
    decoder::{DecoderSpec, MoeRouting, MoeSpec, RopeKind, SharedExpertSpec},
    layers::Activation,
    moe::ExpertProjNames,
    serde_default_fn,
};

serde_default_fn!(f64, rms_norm_eps, 1e-6);
serde_default_fn!(f32, rope_theta, 10_000.0);
serde_default_fn!(f64, routed_scaling_factor, 1.0);
serde_default_fn!(String, topk_method, GREEDY.to_string());
serde_default_fn!(usize, moe_layer_freq, 1);
serde_default_fn!(usize, max_tiles, DEFAULT_MAX_TILES);
serde_default_fn!(u32, image_token_id, IMAGE_TOKEN_ID);

const GREEDY: &str = "greedy";
// the reference's `<image>` id, which its config leaves out
pub const IMAGE_TOKEN_ID: u32 = 128_815;
const MLP: &str = "mlp";
const SHARED_EXPERTS: &str = "shared_experts";
// `dynamic_preprocess`'s tile counts
pub const MIN_TILES: usize = 2;
pub const DEFAULT_MAX_TILES: usize = 9;

/// DeepSeek-V2 without MLA (`use_mla: false`): Llama attention, softmax top-k experts beside shared ones.
#[derive(Debug, Clone, inference_json::FromJson)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub n_routed_experts: usize,
    #[serde(default)]
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub first_k_dense_replace: usize,
    #[serde(default = "moe_layer_freq")]
    pub moe_layer_freq: usize,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default = "routed_scaling_factor")]
    pub routed_scaling_factor: f64,
    #[serde(default = "topk_method")]
    pub topk_method: String,
    #[serde(default)]
    pub use_mla: bool,
    pub max_position_embeddings: usize,
    #[serde(default = "rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(alias = "quantization")]
    pub quantization_config: Option<QuantizedConfig>,
}

impl TextConfig {
    fn is_dense(&self, layer: usize) -> bool {
        layer < self.first_k_dense_replace || !layer.is_multiple_of(self.moe_layer_freq)
    }

    pub fn decoder_spec(&self) -> Result<DecoderSpec> {
        if self.use_mla {
            inference_tensor::bail!(
                "DeepSeek-OCR's text decoder with MLA (`use_mla`) is not supported"
            );
        }
        if self.topk_method != GREEDY {
            inference_tensor::bail!(
                "DeepSeek-OCR's text decoder routes greedily; `topk_method` {} is not supported",
                self.topk_method
            );
        }
        // without renormalization V2's gate leaves the scale out, which is 1 in every config
        if self.norm_topk_prob && self.routed_scaling_factor != 1.0 {
            inference_tensor::bail!(
                "DeepSeek-OCR's text decoder with a routed scale is not supported"
            );
        }
        let layers = self.num_hidden_layers;
        Ok(DecoderSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.hidden_size / self.num_attention_heads,
            hidden_act: Activation::Silu,
            rms_norm_eps: self.rms_norm_eps,
            rope: RopeKind::Default {
                theta: self.rope_theta,
            },
            max_position_embeddings: self.max_position_embeddings,
            layer_windows: vec![None; layers],
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
            moe: Some(MoeSpec {
                num_experts: self.n_routed_experts,
                intermediate_size: self.moe_intermediate_size,
                routing: MoeRouting::TopK {
                    k: self.num_experts_per_tok,
                    renormalize: self.norm_topk_prob,
                },
                quantized_router: false,
                expert_names: ExpertProjNames::DEFAULT,
                name: MLP,
                dense_layers: (0..layers).filter(|&layer| self.is_dense(layer)).collect(),
                cuda_decode_graphs: true,
                shared_expert: (self.n_shared_experts > 0).then_some(SharedExpertSpec {
                    intermediate_size: self.moe_intermediate_size * self.n_shared_experts,
                    name: SHARED_EXPERTS,
                    gated: false,
                }),
            }),
            ..Default::default()
        })
    }
}

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct ClipWidth {
    pub heads: usize,
    pub image_size: usize,
    pub layers: usize,
    pub patch_size: usize,
    pub width: usize,
}

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct SamWidth {
    pub downsample_channels: Vec<usize>,
    pub global_attn_indexes: Vec<usize>,
    pub heads: usize,
    pub layers: usize,
    pub width: usize,
}

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct VisionWidths {
    #[serde(rename = "clip-l-14-224")]
    pub clip: ClipWidth,
    pub sam_vit_b: SamWidth,
}

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct VisionConfig {
    /// The global view's side.
    pub image_size: usize,
    pub width: VisionWidths,
}

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct ProjectorConfig {
    pub input_dim: usize,
    pub n_embed: usize,
}

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct Config {
    pub language_config: TextConfig,
    pub vision_config: VisionConfig,
    pub projector_config: ProjectorConfig,
    /// `dynamic_preprocess`'s largest tile count (Unlimited-OCR tiles further).
    #[serde(default = "max_tiles")]
    pub max_tiles: usize,
    #[serde(default = "image_token_id")]
    pub image_token_id: u32,
}

// a local tile's side, and the SAM patch and neck strides that leave a tile's token grid
pub const TILE_SIZE: usize = 640;
const SAM_STRIDE: usize = 16 * 4;
// the gray the global view is padded with: `int(0.5 * 255)`
pub const PAD_GRAY: u8 = 127;

impl Config {
    pub fn sam_config(&self) -> Result<SamConfig> {
        let sam = &self.vision_config.width.sam_vit_b;
        let [mid, out] = sam.downsample_channels[..] else {
            inference_tensor::bail!(
                "DeepSeek-OCR's SAM takes two downsample widths, not {:?}",
                sam.downsample_channels
            );
        };
        let mut cfg = SamConfig::vit_b(sam.global_attn_indexes.clone(), [mid, out]);
        cfg.embed_dim = sam.width;
        cfg.depth = sam.layers;
        cfg.num_heads = sam.heads;
        cfg.image_size = self.vision_config.image_size;
        Ok(cfg)
    }

    /// The token grid side of a view `size` pixels across.
    pub fn grid(size: usize) -> usize {
        size / SAM_STRIDE
    }

    /// An image's tokens: each grid row of the tiles and of the global view ends in a newline, then one separator.
    pub fn image_token_count(&self, tiles: (usize, usize)) -> usize {
        let global = Self::grid(self.vision_config.image_size);
        let tile = Self::grid(TILE_SIZE);
        let (cols, rows) = tiles;
        let local = if cols * rows > 1 {
            rows * tile * (cols * tile + 1)
        } else {
            0
        };
        local + global * (global + 1) + 1
    }
}
