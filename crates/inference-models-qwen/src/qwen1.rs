//! Qwen 1 (`QWenLMHeadModel`): a Llama-style decoder under `transformer.h`, with a fused biased `c_attn`.

use inference_quant::QuantizedConfig;
use inference_tensor::Result;

use crate::{
    decoder::{DecoderSpec, ModuleNames, NormNames, RopeKind},
    layers::Activation,
    moe::ExpertProjNames,
    serde_default_fn,
};

serde_default_fn!(f64, rotary_pct_default, 1.0);
serde_default_fn!(bool, no_bias_default, true);

/// ChatML with the system message Qwen's `make_context` defaults to; the checkpoints ship no template.
pub const CHAT_TEMPLATE: &str = "{%- if messages and messages[0]['role'] == 'system' %}{%- set system = messages[0]['content'] %}{%- set turns = messages[1:] %}{%- else %}{%- set system = 'You are a helpful assistant.' %}{%- set turns = messages %}{%- endif %}{{ '<|im_start|>system\n' + system + '<|im_end|>\n' }}{%- for message in turns %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{%- endfor %}{%- if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{%- endif %}";

const O_PROJ: &str = "c_proj";
const NORM_NAMES: NormNames = NormNames {
    input: Some("ln_1"),
    pre_ffn: Some("ln_2"),
    post_attn: None,
    post_ffn: None,
    last: "ln_f",
};
// a1 = w1(x), a2 = w2(x), c_proj(a1 * silu(a2)): w2 is the gate
const MODULE_NAMES: ModuleNames = ModuleNames {
    model: "transformer",
    embed_tokens: "wte",
    layers: "h",
    self_attn: "attn",
    qkv_proj: "c_attn",
    mlp: ExpertProjNames {
        gate: "w2",
        up: "w1",
        down: "c_proj",
    },
};

#[derive(Debug, Clone, Default, serde::Serialize, inference_json::FromJson)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    /// Twice the MLP's width: each of `w1` and `w2` is half of it.
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub kv_channels: usize,
    pub layer_norm_epsilon: f64,
    pub rotary_emb_base: f64,
    #[serde(default = "rotary_pct_default")]
    pub rotary_pct: f64,
    /// The trained context; past it the reference switches on dynamic NTK and logn scaling, so it is the limit here.
    pub seq_length: usize,
    #[serde(default = "no_bias_default")]
    pub no_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub quantization_config: Option<QuantizedConfig>,
}

impl Config {
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn decoder_spec(&self) -> Result<DecoderSpec> {
        if !self.no_bias {
            inference_tensor::bail!(
                "Qwen 1 with biased output and MLP projections (`no_bias: false`) is not supported"
            );
        }
        let theta = self.rotary_emb_base as f32;
        let rope = if self.rotary_pct < 1.0 {
            RopeKind::Partial {
                theta,
                rotary_dim: (self.kv_channels as f64 * self.rotary_pct) as usize,
                is_gpt_neox: Some(true),
            }
        } else {
            RopeKind::Default { theta }
        };
        Ok(DecoderSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size / 2,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_attention_heads,
            head_dim: self.kv_channels,
            hidden_act: Activation::Silu,
            rms_norm_eps: self.layer_norm_epsilon,
            rope,
            max_position_embeddings: self.seq_length,
            qkv_bias: true,
            fused_qkv: true,
            o_proj_name: Some(O_PROJ),
            norm_names: NORM_NAMES,
            module_names: MODULE_NAMES,
            layer_windows: vec![None; self.num_hidden_layers],
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
            ..Default::default()
        })
    }
}

pub type Model = crate::decoder::CausalLm;
