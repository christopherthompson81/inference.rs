//! Nomic BERT (nomic-embed-text): a post-norm BERT encoder with rotary positions, a SwiGLU or GELU FFN, and in
//! nomic-bert-moe an ungated GELU MoE every `moe_every_n_layers`-th layer.

use std::sync::Arc;

use inference_quant::{QuantMethod, ShardedVarBuilder};
use inference_tensor::nn::{Embedding, LayerNorm, Module};
use inference_tensor::{D, DType, Device, Result, Tensor};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, FlashParams},
    device_map::DeviceMapper,
    layers::{dense_embedding, layer_norm},
    model::{EmbeddingModel, IsqModel, NormalLoadingMetadata},
    paged_attention::AttentionImplementation,
    rope_encoder::{QkvLayout, RopeAttentionShape, RopeSelfAttention, RopeStyle, rotary},
    serde_default_fn,
    utils::unvarbuilder::UnVarBuilder,
};

pub const SWIGLU: &str = "swiglu";
const GELU: &str = "gelu";

serde_default_fn!(usize, type_vocab_size, 2);
serde_default_fn!(f64, layer_norm_epsilon, 1e-12);
serde_default_fn!(f32, rotary_emb_base, 10_000.0);
serde_default_fn!(f64, rotary_emb_fraction, 1.0);
serde_default_fn!(String, activation_function, SWIGLU.to_string());

// Nomic configs carry GPT-2 names (`n_embd`), and newer ones the BERT names beside them
#[derive(Debug, Clone, inference_json::FromJson)]
pub struct Config {
    pub vocab_size: usize,
    hidden_size: Option<usize>,
    n_embd: Option<usize>,
    num_hidden_layers: Option<usize>,
    n_layer: Option<usize>,
    num_attention_heads: Option<usize>,
    n_head: Option<usize>,
    intermediate_size: Option<usize>,
    n_inner: Option<usize>,
    /// The longest input: `n_positions`, which rotary positions allow past the trained length.
    #[serde(rename = "n_positions")]
    pub max_positions: usize,
    #[serde(default = "type_vocab_size")]
    pub type_vocab_size: usize,
    #[serde(default = "layer_norm_epsilon")]
    pub layer_norm_epsilon: f64,
    #[serde(default = "rotary_emb_base")]
    pub rotary_emb_base: f32,
    #[serde(default = "rotary_emb_fraction")]
    pub rotary_emb_fraction: f64,
    #[serde(default)]
    pub rotary_emb_interleaved: bool,
    #[serde(default)]
    pub qkv_proj_bias: bool,
    #[serde(default)]
    pub mlp_fc1_bias: bool,
    #[serde(default)]
    pub mlp_fc2_bias: bool,
    #[serde(default)]
    pub prenorm: bool,
    #[serde(default = "activation_function")]
    pub activation_function: String,
    /// nomic-bert-moe: layer `i` is MoE when `i % moe_every_n_layers == 1`.
    #[serde(default)]
    pub moe_every_n_layers: usize,
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub moe_top_k: usize,
    /// Only `false`, whose top-k weights stay the softmax's own, is supported.
    #[serde(default)]
    pub moe_normalize_expert_weights: bool,
    #[serde(default)]
    pub num_shared_experts: usize,
}

impl Config {
    pub fn hidden_size(&self) -> usize {
        self.hidden_size.or(self.n_embd).unwrap_or_default()
    }

    pub fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers.or(self.n_layer).unwrap_or_default()
    }

    pub fn num_attention_heads(&self) -> usize {
        self.num_attention_heads.or(self.n_head).unwrap_or_default()
    }

    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size.or(self.n_inner).unwrap_or_default()
    }

    pub fn is_moe_layer(&self, layer: usize) -> bool {
        self.moe_every_n_layers > 0 && layer % self.moe_every_n_layers == 1
    }

    fn validate(&self) -> Result<()> {
        if [
            self.hidden_size(),
            self.num_hidden_layers(),
            self.num_attention_heads(),
        ]
        .contains(&0)
        {
            inference_tensor::bail!(
                "Nomic BERT config needs its hidden size, layer count and head count"
            );
        }
        let unsupported = [
            (self.rotary_emb_fraction != 1.0, "partial rotary"),
            (self.rotary_emb_interleaved, "interleaved rotary"),
            (self.prenorm, "pre-norm layers"),
            (
                self.activation_function != SWIGLU && self.activation_function != GELU,
                "an FFN other than SwiGLU or GELU",
            ),
            (
                self.moe_every_n_layers > 0 && self.activation_function != GELU,
                "gated MoE experts",
            ),
            (
                self.moe_normalize_expert_weights,
                "renormalized expert weights",
            ),
            (self.num_shared_experts > 0, "shared experts"),
            (
                self.mlp_fc1_bias != self.mlp_fc2_bias,
                "one biased FFN projection",
            ),
        ];
        match unsupported.iter().find(|(set, _)| *set) {
            Some((_, what)) => inference_tensor::bail!("Nomic BERT with {what} is not supported"),
            None => Ok(()),
        }
    }
}

/// Experts without a gate, each `gelu(x @ w1_e^T) @ w2_e`, run densely: every expert over every token, weighted by
/// the router's softmax kept at the top-k and zero elsewhere, which sums exactly the top-k experts' outputs.
struct DenseMoe {
    router: Arc<dyn QuantMethod>,
    // both `[experts * intermediate, hidden]`, as megablocks stores them
    w1: Tensor,
    w2: Tensor,
    bias: Tensor,
    num_experts: usize,
    top_k: usize,
}

impl DenseMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, hidden) = xs.dims3()?;
        let tokens = xs.reshape((b_sz * seq_len, hidden))?;
        let probs = inference_tensor::nn::ops::softmax_last_dim(
            &self.router.forward(&tokens)?.to_dtype(DType::F32)?,
        )?;
        // the k-th largest probability per token, by masking out each maximum in turn; ties at it keep every one
        let mut remaining = probs.clone();
        let mut kth = remaining.max_keepdim(D::Minus1)?;
        for _ in 1..self.top_k {
            let taken = remaining.broadcast_ge(&kth)?;
            remaining = taken.where_cond(
                &remaining.zeros_like()?.affine(1., f64::NEG_INFINITY)?,
                &remaining,
            )?;
            kth = remaining.max_keepdim(D::Minus1)?;
        }
        let route =
            (probs.broadcast_ge(&kth)?.to_dtype(DType::F32)? * probs)?.to_dtype(xs.dtype())?;
        let act = tokens.matmul(&self.w1.t()?)?.gelu_erf()?;
        let intermediate = act.dim(1)? / self.num_experts;
        let weighted = act
            .reshape((b_sz * seq_len, self.num_experts, intermediate))?
            .broadcast_mul(&route.unsqueeze(D::Minus1)?)?
            .reshape((b_sz * seq_len, self.num_experts * intermediate))?;
        weighted
            .matmul(&self.w2)?
            .broadcast_add(&self.bias)?
            .reshape((b_sz, seq_len, hidden))
    }
}

enum Ffn {
    SwiGlu {
        up: Arc<dyn QuantMethod>,
        gate: Arc<dyn QuantMethod>,
        down: Arc<dyn QuantMethod>,
    },
    Gelu {
        fc1: Arc<dyn QuantMethod>,
        fc2: Arc<dyn QuantMethod>,
    },
    Moe(DenseMoe),
}

impl Ffn {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::SwiGlu { up, gate, down } => {
                down.forward(&(up.forward(xs)? * gate.forward(xs)?.silu()?)?)
            }
            Self::Gelu { fc1, fc2 } => fc2.forward(&fc1.forward(xs)?.gelu_erf()?),
            Self::Moe(moe) => moe.forward(xs),
        }
    }
}

struct Layer {
    attn: RopeSelfAttention,
    norm1: LayerNorm,
    ffn: Ffn,
    norm2: LayerNorm,
}

impl Layer {
    fn forward(&self, xs: &Tensor, mask: &AttentionMask) -> Result<Tensor> {
        let xs = self.norm1.forward(&(self.attn.forward(xs, mask)? + xs)?)?;
        self.norm2.forward(&(self.ffn.forward(&xs)? + xs)?)
    }
}

pub struct Model {
    word: Embedding,
    token_type: Embedding,
    emb_ln: LayerNorm,
    layers: Vec<Layer>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    device: Device,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if !matches!(attention_mechanism, AttentionImplementation::Eager) {
            inference_tensor::bail!("Expected AttentionImplementation::Eager");
        }
        cfg.validate()?;
        let NormalLoadingMetadata {
            mapper,
            loading_isq,
            real_device,
            ..
        } = normal_loading_metadata;
        let (hidden, eps) = (cfg.hidden_size(), cfg.layer_norm_epsilon);
        let vb_e = mapper.set_nm_device(vb.pp("embeddings"), false);
        let head_dim = hidden / cfg.num_attention_heads();
        let base_rotary = rotary(
            cfg.rotary_emb_base,
            head_dim,
            cfg.max_positions,
            &real_device,
            vb.dtype(),
        )?;
        let vb_l = vb.pp("encoder").pp("layers");
        let layers = (0..cfg.num_hidden_layers())
            .map(|i| {
                let vb = vb_l.pp(i);
                let place = |vb: ShardedVarBuilder| mapper.set_device(i, vb, loading_isq);
                let norm = |name| layer_norm(hidden, eps, mapper.set_device(i, vb.pp(name), false));
                let layer_rotary = match mapper.device_for(i, false) {
                    Some(device) if !device.same_device(&real_device) => rotary(
                        cfg.rotary_emb_base,
                        head_dim,
                        cfg.max_positions,
                        device,
                        vb.dtype(),
                    )?,
                    _ => base_rotary.clone(),
                };
                let linear =
                    |i, o, bias, vb| inference_quant::linear_b(i, o, bias, &None, place(vb));
                let vb_mlp = vb.pp("mlp");
                let inter = cfg.intermediate_size();
                let ffn = if cfg.is_moe_layer(i) {
                    // the experts stay unquantized: ISQ quantizes per output row, megablocks stacks experts on rows
                    let vb_experts = mapper.set_device(i, vb_mlp.pp("experts"), false);
                    let stack = (cfg.num_experts * inter, hidden);
                    Ffn::Moe(DenseMoe {
                        router: inference_quant::linear_b(
                            hidden,
                            cfg.num_experts,
                            false,
                            &None,
                            mapper.set_device(i, vb_mlp.pp("router").pp("layer"), false),
                        )?,
                        w1: vb_experts.pp("mlp").get(stack, "w1")?,
                        w2: vb_experts.pp("mlp").get(stack, "w2")?,
                        bias: vb_experts.get(hidden, "bias")?,
                        num_experts: cfg.num_experts,
                        top_k: cfg.moe_top_k,
                    })
                } else if cfg.activation_function == GELU {
                    Ffn::Gelu {
                        fc1: linear(hidden, inter, cfg.mlp_fc1_bias, vb_mlp.pp("fc1"))?,
                        fc2: linear(inter, hidden, cfg.mlp_fc2_bias, vb_mlp.pp("fc2"))?,
                    }
                } else {
                    Ffn::SwiGlu {
                        up: linear(hidden, inter, cfg.mlp_fc1_bias, vb_mlp.pp("fc11"))?,
                        gate: linear(hidden, inter, cfg.mlp_fc1_bias, vb_mlp.pp("fc12"))?,
                        down: linear(inter, hidden, cfg.mlp_fc2_bias, vb_mlp.pp("fc2"))?,
                    }
                };
                Ok(Layer {
                    attn: RopeSelfAttention::new(
                        &RopeAttentionShape {
                            hidden,
                            num_heads: cfg.num_attention_heads(),
                            bias: cfg.qkv_proj_bias,
                            layout: QkvLayout::Stacked,
                            style: RopeStyle::NeoX,
                        },
                        layer_rotary,
                        place(vb.pp("attn").pp("Wqkv")),
                        place(vb.pp("attn").pp("out_proj")),
                    )?,
                    norm1: norm("norm1")?,
                    ffn,
                    norm2: norm("norm2")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            word: dense_embedding(cfg.vocab_size, hidden, vb_e.pp("word_embeddings"), &None)?,
            token_type: dense_embedding(
                cfg.type_vocab_size,
                hidden,
                vb_e.pp("token_type_embeddings"),
                &None,
            )?,
            emb_ln: layer_norm(hidden, eps, mapper.set_nm_device(vb.pp("emb_ln"), false))?,
            layers,
            mapper,
            device: real_device,
        })
    }
}

impl IsqModel for Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_e = uvb.pp("embeddings");
        uvb_e.pp("word_embeddings").add(&self.word);
        uvb_e.pp("token_type_embeddings").add(&self.token_type);
        uvb.pp("emb_ln").add(&self.emb_ln);
        for (i, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb.pp("encoder").pp("layers").pp(i);
            uvb_l.pp("norm1").add(&layer.norm1);
            uvb_l.pp("norm2").add(&layer.norm2);
            if let Ffn::Moe(moe) = &layer.ffn {
                let uvb_mlp = uvb_l.pp("mlp");
                uvb_mlp.pp("router").pp("layer").add(&moe.router);
                let uvb_experts = uvb_mlp.pp("experts");
                uvb_experts.add_tensor("bias", moe.bias.clone());
                uvb_experts.pp("mlp").add_tensor("w1", moe.w1.clone());
                uvb_experts.pp("mlp").add_tensor("w2", moe.w2.clone());
            }
        }
        uvb.to_safetensors()
    }
}

impl EmbeddingModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        extras: &crate::embedding::EmbeddingExtras,
        _flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let xs = self.word.forward(input_ids)?;
        // a one-row table (nomic-bert-moe) has a single segment whatever the tokenizer numbers it
        let xs = match extras.token_type_ids.as_ref() {
            Some(types) if self.token_type.embeddings().dim(0)? > 1 => {
                (xs + self.token_type.forward(types)?)?
            }
            _ => xs.broadcast_add(&self.token_type.embeddings().narrow(0, 0, 1)?)?,
        };
        let mut xs = self.emb_ln.forward(&xs)?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&self.mapper.map(xs, i)?, &AttentionMask::None)?;
        }
        xs.to_device(&self.device)
    }
    fn device(&self) -> &Device {
        &self.device
    }
}

impl AnyMoeBaseModelMixin for Model {}
