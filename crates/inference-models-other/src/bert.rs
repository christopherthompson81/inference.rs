//! BERT-family encoders (BERT, RoBERTa, XLM-R, CamemBERT): learned positions, post-LayerNorm and a GELU FFN.
#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use inference_quant::{QuantMethod, ShardedVarBuilder};
use inference_tensor::nn::{Embedding, LayerNorm, Module};
use inference_tensor::{Device, Result, Tensor};
use serde::Deserialize;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, FlashParams, SdpaParams},
    device_map::DeviceMapper,
    layers::{Activation, Sdpa, dense_embedding, layer_norm},
    model::{EmbeddingModel, IsqModel, NormalLoadingMetadata},
    paged_attention::AttentionImplementation,
    serde_default_fn,
    utils::unvarbuilder::UnVarBuilder,
};

const WORD_EMBEDDINGS: &str = "embeddings.word_embeddings.weight";
// RoBERTa-style models number positions from `pad_token_id + 1`
const ROBERTA_MODEL_TYPES: &[&str] = &["roberta", "xlm-roberta", "camembert"];
const ABSOLUTE_POSITIONS: &str = "absolute";

serde_default_fn!(usize, type_vocab_size, 2);
serde_default_fn!(f64, layer_norm_eps, 1e-12);
serde_default_fn!(
    String,
    position_embedding_type,
    ABSOLUTE_POSITIONS.to_string()
);

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    #[serde(default = "type_vocab_size")]
    pub type_vocab_size: usize,
    #[serde(default = "layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default)]
    pub pad_token_id: usize,
    #[serde(default = "position_embedding_type")]
    pub position_embedding_type: String,
    pub model_type: String,
}

impl Config {
    fn roberta_style(&self) -> bool {
        ROBERTA_MODEL_TYPES.contains(&self.model_type.as_str())
    }

    /// The first position id a sequence takes.
    pub fn position_offset(&self) -> usize {
        if self.roberta_style() {
            self.pad_token_id + 1
        } else {
            0
        }
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_position_embeddings - self.position_offset()
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    // `*ForMaskedLM` / `*ForSequenceClassification` checkpoints keep the encoder under the base model prefix
    fn base_prefix(&self) -> &'static str {
        if self.roberta_style() {
            "roberta"
        } else {
            "bert"
        }
    }
}

struct Embeddings {
    word: Embedding,
    position: Embedding,
    token_type: Embedding,
    norm: LayerNorm,
    position_offset: usize,
}

impl Embeddings {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let positions = self
            .position
            .embeddings()
            .narrow(0, self.position_offset, seq_len)?;
        // a single segment: every token is type 0
        let token_type = self.token_type.embeddings().narrow(0, 0, 1)?;
        let xs = self
            .word
            .forward(input_ids)?
            .broadcast_add(&positions.broadcast_add(&token_type)?)?;
        self.norm.forward(&xs)
    }
}

struct Attention {
    query: Arc<dyn QuantMethod>,
    key: Arc<dyn QuantMethod>,
    value: Arc<dyn QuantMethod>,
    output: Arc<dyn QuantMethod>,
    norm: LayerNorm,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, hidden) = xs.dims3()?;
        let heads = |proj: &Arc<dyn QuantMethod>| {
            proj.forward(xs)?
                .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)
        };
        let (q, k, v) = (heads(&self.query)?, heads(&self.key)?, heads(&self.value)?);
        let params = SdpaParams {
            n_kv_groups: 1,
            sliding_window: None,
            softcap: None,
            softmax_scale: 1.0 / (self.head_dim as f32).sqrt(),
            sinks: None,
            chunk: None,
        };
        // batches hold equal-length prompts, so there is no padding to mask
        let attn = Sdpa.run_attention(
            &q,
            &k,
            &v,
            &AttentionMask::None,
            Some(&FlashParams::empty(false)),
            &params,
        )?;
        let attn = attn.transpose(1, 2)?.reshape((b_sz, seq_len, hidden))?;
        self.norm.forward(&(self.output.forward(&attn)? + xs)?)
    }
}

struct Layer {
    attention: Attention,
    intermediate: Arc<dyn QuantMethod>,
    output: Arc<dyn QuantMethod>,
    norm: LayerNorm,
    act: Activation,
}

impl Layer {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        layer_idx: usize,
        mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let place = |vb: ShardedVarBuilder| mapper.set_device(layer_idx, vb, loading_isq);
        let norm = |vb: ShardedVarBuilder| {
            layer_norm(
                hidden,
                cfg.layer_norm_eps,
                mapper.set_device(layer_idx, vb.pp("LayerNorm"), false),
            )
        };
        let vb_attn = vb.pp("attention");
        let vb_self = vb_attn.pp("self");
        let linear = |i, o, vb| inference_quant::linear(i, o, &None, place(vb));
        Ok(Self {
            attention: Attention {
                query: linear(hidden, hidden, vb_self.pp("query"))?,
                key: linear(hidden, hidden, vb_self.pp("key"))?,
                value: linear(hidden, hidden, vb_self.pp("value"))?,
                output: linear(hidden, hidden, vb_attn.pp("output").pp("dense"))?,
                norm: norm(vb_attn.pp("output"))?,
                num_heads: cfg.num_attention_heads,
                head_dim: cfg.head_dim(),
            },
            intermediate: linear(
                hidden,
                cfg.intermediate_size,
                vb.pp("intermediate").pp("dense"),
            )?,
            output: linear(cfg.intermediate_size, hidden, vb.pp("output").pp("dense"))?,
            norm: norm(vb.pp("output"))?,
            act: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.attention.forward(xs)?;
        let ffn = self
            .output
            .forward(&self.act.forward(&self.intermediate.forward(&xs)?)?)?;
        self.norm.forward(&(ffn + xs)?)
    }
}

/// The encoder's last hidden states, which the sentence-transformers modules pool.
pub struct Model {
    embeddings: Embeddings,
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
        if cfg.position_embedding_type != ABSOLUTE_POSITIONS {
            inference_tensor::bail!(
                "BERT position_embedding_type `{}` is not supported",
                cfg.position_embedding_type
            );
        }
        let vb = if vb.contains_tensor(WORD_EMBEDDINGS) {
            vb
        } else {
            vb.pp(cfg.base_prefix())
        };
        let NormalLoadingMetadata {
            mapper,
            loading_isq,
            real_device,
            ..
        } = normal_loading_metadata;
        let vb_e = mapper.set_nm_device(vb.pp("embeddings"), false);
        let hidden = cfg.hidden_size;
        let embeddings = Embeddings {
            word: dense_embedding(cfg.vocab_size, hidden, vb_e.pp("word_embeddings"), &None)?,
            position: dense_embedding(
                cfg.max_position_embeddings,
                hidden,
                vb_e.pp("position_embeddings"),
                &None,
            )?,
            token_type: dense_embedding(
                cfg.type_vocab_size,
                hidden,
                vb_e.pp("token_type_embeddings"),
                &None,
            )?,
            norm: layer_norm(hidden, cfg.layer_norm_eps, vb_e.pp("LayerNorm"))?,
            position_offset: cfg.position_offset(),
        };
        let vb_l = vb.pp("encoder").pp("layer");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Layer::new(cfg, vb_l.pp(i), i, &*mapper, loading_isq))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embeddings,
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
        uvb_e.pp("word_embeddings").add(&self.embeddings.word);
        uvb_e
            .pp("position_embeddings")
            .add(&self.embeddings.position);
        uvb_e
            .pp("token_type_embeddings")
            .add(&self.embeddings.token_type);
        uvb_e.pp("LayerNorm").add(&self.embeddings.norm);
        for (i, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb.pp("encoder").pp("layer").pp(i);
            uvb_l
                .pp("attention")
                .pp("output")
                .pp("LayerNorm")
                .add(&layer.attention.norm);
            uvb_l.pp("output").pp("LayerNorm").add(&layer.norm);
        }
        uvb.to_safetensors()
    }
}

impl EmbeddingModel for Model {
    fn forward(&self, input_ids: &Tensor, _flash_params: &FlashParams) -> Result<Tensor> {
        let mut xs = self.embeddings.forward(input_ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&self.mapper.map(xs, i)?)?;
        }
        xs.to_device(&self.device)
    }
    fn device(&self) -> &Device {
        &self.device
    }
}

impl AnyMoeBaseModelMixin for Model {}
