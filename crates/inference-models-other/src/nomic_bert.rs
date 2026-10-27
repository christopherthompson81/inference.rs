//! Nomic BERT (nomic-embed-text): a post-norm BERT encoder with rotary positions and a SwiGLU FFN.

use std::sync::Arc;

use inference_quant::{QuantMethod, ShardedVarBuilder};
use inference_tensor::nn::{Embedding, LayerNorm, Module};
use inference_tensor::{Device, Result, Tensor};
use serde::Deserialize;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, FlashParams},
    device_map::DeviceMapper,
    layers::{dense_embedding, layer_norm},
    model::{EmbeddingModel, IsqModel, NormalLoadingMetadata},
    paged_attention::AttentionImplementation,
    rope_encoder::{RopeSelfAttention, rotary},
    serde_default_fn,
    utils::unvarbuilder::UnVarBuilder,
};

const SWIGLU: &str = "swiglu";

serde_default_fn!(usize, type_vocab_size, 2);
serde_default_fn!(f64, layer_norm_epsilon, 1e-12);
serde_default_fn!(f32, rotary_emb_base, 10_000.0);
serde_default_fn!(f64, rotary_emb_fraction, 1.0);
serde_default_fn!(String, activation_function, SWIGLU.to_string());

// Nomic configs carry GPT-2 names (`n_embd`), and newer ones the BERT names beside them
#[derive(Debug, Clone, Deserialize)]
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
    /// Set on nomic-bert-moe checkpoints, which this encoder does not run.
    #[serde(default)]
    pub moe_every_n_layers: usize,
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
            (self.activation_function != SWIGLU, "a non-SwiGLU FFN"),
            (self.moe_every_n_layers > 0, "MoE layers"),
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

struct Layer {
    attn: RopeSelfAttention,
    norm1: LayerNorm,
    up: Arc<dyn QuantMethod>,
    gate: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
    norm2: LayerNorm,
}

impl Layer {
    fn forward(&self, xs: &Tensor, mask: &AttentionMask) -> Result<Tensor> {
        let xs = self.norm1.forward(&(self.attn.forward(xs, mask)? + xs)?)?;
        let ffn = self
            .down
            .forward(&(self.up.forward(&xs)? * self.gate.forward(&xs)?.silu()?)?)?;
        self.norm2.forward(&(ffn + xs)?)
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
                Ok(Layer {
                    attn: RopeSelfAttention::new(
                        hidden,
                        cfg.num_attention_heads(),
                        cfg.qkv_proj_bias,
                        layer_rotary,
                        place(vb.pp("attn").pp("Wqkv")),
                        place(vb.pp("attn").pp("out_proj")),
                    )?,
                    norm1: norm("norm1")?,
                    up: linear(
                        hidden,
                        cfg.intermediate_size(),
                        cfg.mlp_fc1_bias,
                        vb_mlp.pp("fc11"),
                    )?,
                    gate: linear(
                        hidden,
                        cfg.intermediate_size(),
                        cfg.mlp_fc1_bias,
                        vb_mlp.pp("fc12"),
                    )?,
                    down: linear(
                        cfg.intermediate_size(),
                        hidden,
                        cfg.mlp_fc2_bias,
                        vb_mlp.pp("fc2"),
                    )?,
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
        }
        uvb.to_safetensors()
    }
}

impl EmbeddingModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
        _flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let xs = self.word.forward(input_ids)?;
        let xs = match token_type_ids {
            Some(types) => (xs + self.token_type.forward(types)?)?,
            None => xs.broadcast_add(&self.token_type.embeddings().narrow(0, 0, 1)?)?,
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
