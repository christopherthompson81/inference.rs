//! BERT-family encoders (BERT, RoBERTa, XLM-R, CamemBERT, DistilBERT, Jina BERT v2): learned or ALiBi positions,
//! post-LayerNorm and a GELU or GLU FFN, with the `*ForSequenceClassification` heads that cross-encoder rerankers add.
#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use inference_quant::{QuantMethod, ShardedVarBuilder};
use inference_tensor::nn::{Embedding, LayerNorm, Module};
use inference_tensor::{D, DType, Device, IndexOp, Result, Tensor};
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
// BERT's block under other names, with no segment embeddings
const DISTILBERT_MODEL_TYPE: &str = "distilbert";
const ABSOLUTE_POSITIONS: &str = "absolute";
// Jina BERT v2: no position table, a per-head linear bias on |i - j| instead
const ALIBI_POSITIONS: &str = "alibi";
pub const ORIGINAL_FFN: &str = "original";
const CLASSIFIER_SUFFIX: &str = "ForSequenceClassification";
const PRE_CLASSIFIER: &str = "pre_classifier";
const PRE_CLASSIFIER_WEIGHT: &str = "pre_classifier.weight";
// transformers' PretrainedConfig default when a config lists no labels
const DEFAULT_NUM_LABELS: usize = 2;

serde_default_fn!(usize, type_vocab_size, 2);
serde_default_fn!(f64, layer_norm_eps, 1e-12);
serde_default_fn!(
    String,
    position_embedding_type,
    ABSOLUTE_POSITIONS.to_string()
);
serde_default_fn!(String, feed_forward_type, ORIGINAL_FFN.to_string());

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    // DistilBERT's config keys
    #[serde(alias = "dim")]
    pub hidden_size: usize,
    #[serde(alias = "n_layers")]
    pub num_hidden_layers: usize,
    #[serde(alias = "n_heads")]
    pub num_attention_heads: usize,
    #[serde(alias = "hidden_dim")]
    pub intermediate_size: usize,
    #[serde(alias = "activation")]
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
    // Jina v3's config names no model type
    #[serde(default)]
    pub model_type: String,
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub id2label: Option<std::collections::HashMap<String, String>>,
    /// Jina BERT v2's `geglu` / `reglu`: one fused gate-and-up projection with a LayerNorm inside the MLP.
    #[serde(default = "feed_forward_type")]
    pub feed_forward_type: String,
}

impl Config {
    fn roberta_style(&self) -> bool {
        ROBERTA_MODEL_TYPES.contains(&self.model_type.as_str())
    }

    fn distilbert(&self) -> bool {
        self.model_type == DISTILBERT_MODEL_TYPE
    }

    /// The segment-embedding rows; DistilBERT has none.
    pub fn token_type_rows(&self) -> usize {
        if self.distilbert() {
            0
        } else {
            self.type_vocab_size
        }
    }

    fn names(&self) -> &'static LayerNames {
        if self.distilbert() {
            &DISTILBERT_NAMES
        } else {
            &BERT_NAMES
        }
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

    /// A cross-encoder: a classification head over the first token, scoring rather than embedding.
    pub fn is_classifier(&self) -> bool {
        self.architectures
            .iter()
            .any(|arch| arch.ends_with(CLASSIFIER_SUFFIX))
    }

    pub fn num_labels(&self) -> usize {
        self.id2label
            .as_ref()
            .map_or(DEFAULT_NUM_LABELS, |labels| labels.len())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn alibi(&self) -> bool {
        self.position_embedding_type == ALIBI_POSITIONS
    }

    /// The activation of a GLU FFN; `None` for BERT's own intermediate / output pair.
    pub fn glu_activation(&self) -> Result<Option<Activation>> {
        Ok(match self.feed_forward_type.as_str() {
            ORIGINAL_FFN => None,
            "geglu" => Some(Activation::Gelu),
            "reglu" => Some(Activation::Relu),
            other => inference_tensor::bail!("BERT feed_forward_type `{other}` is not supported"),
        })
    }

    // `*ForMaskedLM` / `*ForSequenceClassification` checkpoints keep the encoder under the base model prefix
    fn base_prefix(&self) -> &'static str {
        if self.roberta_style() {
            "roberta"
        } else if self.distilbert() {
            "distilbert"
        } else {
            "bert"
        }
    }
}

/// Where an encoder's layers and their weights sit, under the base prefix.
struct LayerNames {
    layers: &'static str,
    query: &'static str,
    key: &'static str,
    value: &'static str,
    attention_output: &'static str,
    attention_norm: &'static str,
    intermediate: &'static str,
    output: &'static str,
    output_norm: &'static str,
}

const BERT_NAMES: LayerNames = LayerNames {
    layers: "encoder.layer",
    query: "attention.self.query",
    key: "attention.self.key",
    value: "attention.self.value",
    attention_output: "attention.output.dense",
    attention_norm: "attention.output.LayerNorm",
    intermediate: "intermediate.dense",
    output: "output.dense",
    output_norm: "output.LayerNorm",
};

const DISTILBERT_NAMES: LayerNames = LayerNames {
    layers: "transformer.layer",
    query: "attention.q_lin",
    key: "attention.k_lin",
    value: "attention.v_lin",
    attention_output: "attention.out_lin",
    attention_norm: "sa_layer_norm",
    intermediate: "ffn.lin1",
    output: "ffn.lin2",
    output_norm: "output_layer_norm",
};

/// ALiBi's per-head slopes (Press et al.): a geometric series for a power-of-two head count, with the odd terms of
/// the next power's series filling the rest.
fn alibi_slopes(heads: usize) -> Vec<f32> {
    let power_of_two = |n: usize| -> Vec<f32> {
        let start = 2f32.powf(-8.0 / n as f32);
        std::iter::successors(Some(start), |slope| Some(slope * start))
            .take(n)
            .collect()
    };
    let closest = 1 << heads.ilog2();
    let mut slopes = power_of_two(closest);
    slopes.extend(
        power_of_two(2 * closest)
            .into_iter()
            .step_by(2)
            .take(heads - closest),
    );
    slopes
}

struct Embeddings {
    word: Embedding,
    // absent with ALiBi
    position: Option<Embedding>,
    // absent in DistilBERT
    token_type: Option<Embedding>,
    norm: LayerNorm,
    position_offset: usize,
}

impl Embeddings {
    fn forward(&self, input_ids: &Tensor, token_type_ids: Option<&Tensor>) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let xs = self.word.forward(input_ids)?;
        let xs = match &self.position {
            Some(position) => xs.broadcast_add(&position.embeddings().narrow(
                0,
                self.position_offset,
                seq_len,
            )?)?,
            None => xs,
        };
        // a one-row table (RoBERTa, XLM-R) has a single segment whatever the tokenizer numbers it
        let xs = match (&self.token_type, token_type_ids) {
            (None, _) => xs,
            (Some(table), Some(types)) if table.embeddings().dim(0)? > 1 => {
                (xs + table.forward(types)?)?
            }
            (Some(table), _) => xs.broadcast_add(&table.embeddings().narrow(0, 0, 1)?)?,
        };
        self.norm.forward(&xs)
    }
}

/// A sequence-classification head over the first token's state.
enum Head {
    /// BERT: the pooler's tanh dense layer, then `classifier`.
    Pooled {
        pooler: Arc<dyn QuantMethod>,
        classifier: Arc<dyn QuantMethod>,
    },
    /// RoBERTa / XLM-R: `classifier.dense`, tanh, then `classifier.out_proj`.
    Projected {
        dense: Arc<dyn QuantMethod>,
        out_proj: Arc<dyn QuantMethod>,
    },
    /// DistilBERT: `pre_classifier`, ReLU, then `classifier`.
    PreClassified {
        pre_classifier: Arc<dyn QuantMethod>,
        classifier: Arc<dyn QuantMethod>,
    },
}

impl Head {
    fn new(cfg: &Config, root: &ShardedVarBuilder, encoder: &ShardedVarBuilder) -> Result<Self> {
        let (hidden, labels) = (cfg.hidden_size, cfg.num_labels());
        let linear = |i, o, vb| inference_quant::linear(i, o, &None, vb);
        let classifier = root.pp("classifier");
        // by what the checkpoint holds: an XLM-R reranker GGUF names its model type `bert`
        Ok(if classifier.contains_tensor("dense.weight") {
            Self::Projected {
                dense: linear(hidden, hidden, classifier.pp("dense"))?,
                out_proj: linear(hidden, labels, classifier.pp("out_proj"))?,
            }
        } else if root.contains_tensor(PRE_CLASSIFIER_WEIGHT) {
            Self::PreClassified {
                pre_classifier: linear(hidden, hidden, root.pp(PRE_CLASSIFIER))?,
                classifier: linear(hidden, labels, classifier)?,
            }
        } else {
            Self::Pooled {
                pooler: linear(hidden, hidden, encoder.pp("pooler").pp("dense"))?,
                classifier: linear(hidden, labels, classifier)?,
            }
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let first = xs.i((.., 0, ..))?.contiguous()?;
        match self {
            Self::Pooled { pooler, classifier } => {
                classifier.forward(&pooler.forward(&first)?.tanh()?)
            }
            Self::Projected { dense, out_proj } => {
                out_proj.forward(&dense.forward(&first)?.tanh()?)
            }
            Self::PreClassified {
                pre_classifier,
                classifier,
            } => classifier.forward(&pre_classifier.forward(&first)?.relu()?),
        }
    }

    fn add_residual(&self, root: &UnVarBuilder, encoder: &UnVarBuilder) {
        match self {
            Self::Pooled { pooler, classifier } => {
                encoder.pp("pooler").pp("dense").add(pooler);
                root.pp("classifier").add(classifier);
            }
            Self::Projected { dense, out_proj } => {
                root.pp("classifier").pp("dense").add(dense);
                root.pp("classifier").pp("out_proj").add(out_proj);
            }
            Self::PreClassified {
                pre_classifier,
                classifier,
            } => {
                root.pp(PRE_CLASSIFIER).add(pre_classifier);
                root.pp("classifier").add(classifier);
            }
        }
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
    fn forward(&self, xs: &Tensor, mask: &AttentionMask) -> Result<Tensor> {
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
        // batches hold equal-length prompts, so only ALiBi masks
        let attn =
            Sdpa.run_attention(&q, &k, &v, mask, Some(&FlashParams::empty(false)), &params)?;
        let attn = attn.transpose(1, 2)?.reshape((b_sz, seq_len, hidden))?;
        self.norm.forward(&(self.output.forward(&attn)? + xs)?)
    }
}

enum Ffn {
    /// BERT's `intermediate.dense`, the activation, `output.dense`.
    Plain {
        intermediate: Arc<dyn QuantMethod>,
        output: Arc<dyn QuantMethod>,
    },
    /// Jina's `mlp.gated_layers` (activated half, then linear half) into `mlp.wo`.
    Glu {
        gated: Arc<dyn QuantMethod>,
        wo: Arc<dyn QuantMethod>,
    },
}

struct Layer {
    attention: Attention,
    ffn: Ffn,
    // `output.LayerNorm`, or `mlp.layernorm` with a GLU FFN
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
        let names = cfg.names();
        let place = |vb: ShardedVarBuilder| mapper.set_device(layer_idx, vb, loading_isq);
        let norm = |vb: ShardedVarBuilder| {
            layer_norm(
                hidden,
                cfg.layer_norm_eps,
                mapper.set_device(layer_idx, vb, false),
            )
        };
        let linear = |i, o, name| inference_quant::linear(i, o, &None, place(vb.pp(name)));
        let inter = cfg.intermediate_size;
        let glu = cfg.glu_activation()?;
        Ok(Self {
            attention: Attention {
                query: linear(hidden, hidden, names.query)?,
                key: linear(hidden, hidden, names.key)?,
                value: linear(hidden, hidden, names.value)?,
                output: linear(hidden, hidden, names.attention_output)?,
                norm: norm(vb.pp(names.attention_norm))?,
                num_heads: cfg.num_attention_heads,
                head_dim: cfg.head_dim(),
            },
            ffn: match glu {
                None => Ffn::Plain {
                    intermediate: linear(hidden, inter, names.intermediate)?,
                    output: linear(inter, hidden, names.output)?,
                },
                Some(_) => Ffn::Glu {
                    gated: inference_quant::linear_b(
                        hidden,
                        2 * inter,
                        false,
                        &None,
                        place(vb.pp("mlp").pp("gated_layers")),
                    )?,
                    wo: linear(inter, hidden, "mlp.wo")?,
                },
            },
            norm: match glu {
                None => norm(vb.pp(names.output_norm))?,
                Some(_) => layer_norm(
                    hidden,
                    cfg.layer_norm_eps,
                    mapper.set_device(layer_idx, vb.pp("mlp").pp("layernorm"), false),
                )?,
            },
            act: glu.unwrap_or(cfg.hidden_act),
        })
    }

    fn forward(&self, xs: &Tensor, mask: &AttentionMask) -> Result<Tensor> {
        let xs = self.attention.forward(xs, mask)?;
        let ffn = match &self.ffn {
            Ffn::Plain {
                intermediate,
                output,
            } => output.forward(&self.act.forward(&intermediate.forward(&xs)?)?)?,
            Ffn::Glu { gated, wo } => {
                let gated = gated.forward(&xs)?;
                let half = gated.dim(D::Minus1)? / 2;
                let activated = self.act.forward(&gated.narrow(D::Minus1, 0, half)?)?;
                wo.forward(&(activated * gated.narrow(D::Minus1, half, half)?)?)?
            }
        };
        self.norm.forward(&(ffn + xs)?)
    }
}

/// The encoder's last hidden states, which the sentence-transformers modules pool, or a classifier's logits.
pub struct Model {
    embeddings: Embeddings,
    layers: Vec<Layer>,
    head: Option<Head>,
    // `[heads, 1, 1]`, negated: each head's bias is slope * -|i - j|
    alibi_slopes: Option<Tensor>,
    // the base model prefix the encoder's weights sit under, kept for the UQFF residual
    prefix: Option<&'static str>,
    names: &'static LayerNames,
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
        if cfg.position_embedding_type != ABSOLUTE_POSITIONS && !cfg.alibi() {
            inference_tensor::bail!(
                "BERT position_embedding_type `{}` is not supported",
                cfg.position_embedding_type
            );
        }
        let root = vb.clone();
        let prefix = (!vb.contains_tensor(WORD_EMBEDDINGS)).then(|| cfg.base_prefix());
        let vb = match prefix {
            Some(prefix) => vb.pp(prefix),
            None => vb,
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
            position: (!cfg.alibi())
                .then(|| {
                    dense_embedding(
                        cfg.max_position_embeddings,
                        hidden,
                        vb_e.pp("position_embeddings"),
                        &None,
                    )
                })
                .transpose()?,
            token_type: (cfg.token_type_rows() > 0)
                .then(|| {
                    dense_embedding(
                        cfg.token_type_rows(),
                        hidden,
                        vb_e.pp("token_type_embeddings"),
                        &None,
                    )
                })
                .transpose()?,
            norm: layer_norm(hidden, cfg.layer_norm_eps, vb_e.pp("LayerNorm"))?,
            position_offset: cfg.position_offset(),
        };
        let vb_l = vb.pp(cfg.names().layers);
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Layer::new(cfg, vb_l.pp(i), i, &*mapper, loading_isq))
            .collect::<Result<Vec<_>>>()?;
        let head = cfg
            .is_classifier()
            .then(|| {
                Head::new(
                    cfg,
                    &mapper.set_nm_device(root, false),
                    &mapper.set_nm_device(vb.clone(), false),
                )
            })
            .transpose()?;
        let alibi_slopes = cfg
            .alibi()
            .then(|| {
                let slopes = alibi_slopes(cfg.num_attention_heads);
                let heads = slopes.len();
                Tensor::from_vec(
                    slopes.into_iter().map(|s| -s).collect(),
                    (heads, 1, 1),
                    &real_device,
                )
            })
            .transpose()?;
        Ok(Self {
            embeddings,
            layers,
            head,
            alibi_slopes,
            prefix,
            names: cfg.names(),
            mapper,
            device: real_device,
        })
    }
}

impl Model {
    fn add_encoder_residual(&self, uvb: &UnVarBuilder) {
        let uvb_e = uvb.pp("embeddings");
        uvb_e.pp("word_embeddings").add(&self.embeddings.word);
        if let Some(position) = &self.embeddings.position {
            uvb_e.pp("position_embeddings").add(position);
        }
        if let Some(token_type) = &self.embeddings.token_type {
            uvb_e.pp("token_type_embeddings").add(token_type);
        }
        uvb_e.pp("LayerNorm").add(&self.embeddings.norm);
        for (i, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb.pp(self.names.layers).pp(i);
            uvb_l
                .pp(self.names.attention_norm)
                .add(&layer.attention.norm);
            match layer.ffn {
                Ffn::Plain { .. } => uvb_l.pp(self.names.output_norm).add(&layer.norm),
                Ffn::Glu { .. } => uvb_l.pp("mlp").pp("layernorm").add(&layer.norm),
            }
        }
    }
}

impl IsqModel for Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let root = UnVarBuilder::new();
        let encoder = self.prefix.map(|prefix| root.pp(prefix));
        let encoder = encoder.as_ref().unwrap_or(&root);
        self.add_encoder_residual(encoder);
        if let Some(head) = &self.head {
            head.add_residual(&root, encoder);
        }
        root.to_safetensors()
    }
}

/// `(1, heads, seq, seq)`: each head's `-slope * |i - j|`, built on the slopes' device at the start of a forward.
fn alibi_bias(slopes: &Tensor, seq_len: usize, dtype: DType) -> Result<Tensor> {
    let positions = Tensor::arange(0f32, seq_len as f32, slopes.device())?;
    let distance = positions
        .unsqueeze(1)?
        .broadcast_sub(&positions.unsqueeze(0)?)?
        .abs()?;
    slopes
        .broadcast_mul(&distance.unsqueeze(0)?)?
        .unsqueeze(0)?
        .to_dtype(dtype)
}

impl EmbeddingModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        extras: &crate::embedding::EmbeddingExtras,
        _flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut xs = self
            .embeddings
            .forward(input_ids, extras.token_type_ids.as_ref())?;
        let alibi = self
            .alibi_slopes
            .as_ref()
            .map(|slopes| alibi_bias(slopes, input_ids.dim(1)?, xs.dtype()))
            .transpose()?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            let mask = match &alibi {
                Some(bias) => AttentionMask::Custom(bias.to_device(xs.device())?),
                None => AttentionMask::None,
            };
            xs = layer.forward(&xs, &mask)?;
        }
        let xs = xs.to_device(&self.device)?;
        match &self.head {
            Some(head) => head.forward(&xs),
            None => Ok(xs),
        }
    }
    fn device(&self) -> &Device {
        &self.device
    }
}

impl AnyMoeBaseModelMixin for Model {}

#[cfg(test)]
mod tests {
    use super::alibi_slopes;

    #[test]
    fn alibi_slopes_follow_the_power_of_two_series_and_fill_with_the_next_ones_odd_terms() {
        let slopes = alibi_slopes(12);
        // 8 heads: 2^-1 .. 2^-8; the other 4: 2^-0.5, 2^-1.5, 2^-2.5, 2^-3.5
        let expected = [
            -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -0.5, -1.5, -2.5, -3.5,
        ]
        .map(|exp: f32| 2f32.powf(exp));
        for (slope, want) in slopes.iter().zip(expected) {
            assert!((slope - want).abs() < 1e-6, "{slopes:?}");
        }
    }
}
