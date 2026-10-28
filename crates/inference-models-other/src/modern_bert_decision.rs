//! A ModernBERT decision model (laya, Julia-1): the encoder's states plus a question-type embedding, a few pre-norm
//! transformer blocks (torch's `TransformerEncoderLayer`, `norm_first`, ReLU), and a scorer over each option marker.
#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use inference_quant::{QuantMethod, QuantMethodConfig, ShardedVarBuilder, UnquantLinear};
use inference_tensor::nn::{Embedding, LayerNorm, LayerNormConfig, Module};
use inference_tensor::{D, DType, Device, Result, Tensor};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, FlashParams, SdpaParams},
    decision::{DecisionRows, DecisionSpec, QuestionKind},
    layers::{Sdpa, dense_embedding, layer_norm},
    model::{EmbeddingModel, IsqModel, NormalLoadingMetadata},
    modern_bert,
    paged_attention::AttentionImplementation,
    utils::unvarbuilder::UnVarBuilder,
};

const ENCODER: &str = "encoder";
// torch's MultiheadAttention in these heads splits the width into heads of 64
const HEAD_DIM: usize = 64;
const QUESTION_TYPES: usize = 3;
// torch's LayerNorm default
const NORM_EPS: f64 = 1e-5;

fn norm(size: usize, vb: ShardedVarBuilder) -> Result<LayerNorm> {
    layer_norm(
        size,
        LayerNormConfig {
            eps: NORM_EPS,
            remove_mean: true,
            affine: true,
        },
        vb,
    )
}

fn linear(i: usize, o: usize, vb: ShardedVarBuilder) -> Result<Arc<dyn QuantMethod>> {
    inference_quant::linear(i, o, &None, vb)
}

/// torch's pre-norm `TransformerEncoderLayer`: `x + attn(norm1(x))`, then `x + linear2(relu(linear1(norm2(x))))`.
pub(crate) struct Block {
    norm1: LayerNorm,
    in_proj: Arc<dyn QuantMethod>,
    out_proj: Arc<dyn QuantMethod>,
    norm2: LayerNorm,
    linear1: Arc<dyn QuantMethod>,
    linear2: Arc<dyn QuantMethod>,
    heads: usize,
}

impl Block {
    fn new(width: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let attn = vb.pp("self_attn");
        Ok(Self {
            norm1: norm(width, vb.pp("norm1"))?,
            // torch keeps q, k and v as one `in_proj_weight` / `in_proj_bias`, outside ISQ's names
            in_proj: Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                inference_tensor::nn::Linear::new(
                    attn.get((3 * width, width), "in_proj_weight")?,
                    Some(attn.get(3 * width, "in_proj_bias")?),
                ),
            ))?),
            out_proj: linear(width, width, attn.pp("out_proj"))?,
            norm2: norm(width, vb.pp("norm2"))?,
            linear1: linear(width, 4 * width, vb.pp("linear1"))?,
            linear2: linear(4 * width, width, vb.pp("linear2"))?,
            heads: (width / HEAD_DIM).max(1),
        })
    }

    /// A block from its parts, under whatever names a checkpoint gives them.
    pub(crate) fn from_parts(
        norms: (LayerNorm, LayerNorm),
        attention: (Arc<dyn QuantMethod>, Arc<dyn QuantMethod>),
        mlp: (Arc<dyn QuantMethod>, Arc<dyn QuantMethod>),
        width: usize,
    ) -> Self {
        Self {
            norm1: norms.0,
            in_proj: attention.0,
            out_proj: attention.1,
            norm2: norms.1,
            linear1: mlp.0,
            linear2: mlp.1,
            heads: (width / HEAD_DIM).max(1),
        }
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_masked(xs, &AttentionMask::None)
    }

    /// As `forward`, under an attention mask (d1-omni's media and text read only their own kind).
    pub(crate) fn forward_masked(&self, xs: &Tensor, mask: &AttentionMask) -> Result<Tensor> {
        let (b_sz, seq_len, width) = xs.dims3()?;
        let head_dim = width / self.heads;
        let qkv = self.in_proj.forward(&self.norm1.forward(xs)?)?;
        let heads = |i: usize| {
            qkv.narrow(D::Minus1, i * width, width)?
                .reshape((b_sz, seq_len, self.heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let params = SdpaParams {
            n_kv_groups: 1,
            sliding_window: None,
            softcap: None,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
            sinks: None,
            chunk: None,
        };
        // batches hold equal-length prompts, so no padding is masked
        let attn = Sdpa.run_attention(
            &heads(0)?,
            &heads(1)?,
            &heads(2)?,
            mask,
            Some(&FlashParams::empty(false)),
            &params,
        )?;
        let attn = attn.transpose(1, 2)?.reshape((b_sz, seq_len, width))?;
        let xs = (xs + self.out_proj.forward(&attn)?)?;
        let ff = self
            .linear2
            .forward(&self.linear1.forward(&self.norm2.forward(&xs)?)?.relu()?)?;
        xs + ff
    }
}

/// The encoder's states at each option's marker, scored to one logit an option and divided by the checkpoint's
/// temperature for the question's type and option count: `[batch, most options]`.
pub struct Model {
    encoder: modern_bert::Model,
    type_emb: Embedding,
    blocks: Vec<Block>,
    scorer_norm: LayerNorm,
    scorer_hidden: Arc<dyn QuantMethod>,
    scorer_out: Arc<dyn QuantMethod>,
    spec: DecisionSpec,
    device: Device,
}

impl Model {
    pub fn new(
        cfg: &modern_bert::Config,
        decision: &DecisionSpec,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let width = cfg.hidden_size;
        let device = normal_loading_metadata.real_device.clone();
        let head = normal_loading_metadata
            .mapper
            .set_nm_device(vb.clone(), false);
        let blocks = (0..decision.head_layers)
            .map(|i| Block::new(width, head.pp("head").pp("layers").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let scorer = head.pp("scorer");
        Ok(Self {
            type_emb: dense_embedding(QUESTION_TYPES, width, head.pp("type_emb"), &None)?,
            blocks,
            // torch's Sequential(LayerNorm, Linear, GELU, Linear)
            scorer_norm: norm(width, scorer.pp(0))?,
            scorer_hidden: linear(width, width, scorer.pp(1))?,
            scorer_out: linear(width, 1, scorer.pp(3))?,
            encoder: modern_bert::Model::new(
                cfg,
                vb.pp(ENCODER),
                normal_loading_metadata,
                attention_mechanism,
            )?,
            spec: decision.clone(),
            device,
        })
    }

    pub(crate) fn markers(xs: &Tensor, rows: &DecisionRows) -> Result<Tensor> {
        let width = xs.dim(D::Minus1)?;
        let most = rows.markers.iter().map(Vec::len).max().unwrap_or(0);
        let picked = rows
            .markers
            .iter()
            .enumerate()
            .map(|(row, markers)| {
                let index = Tensor::new(markers.as_slice(), xs.device())?;
                let states = xs.get(row)?.index_select(&index, 0)?;
                // rows with fewer options pad to the batch's most; the caller reads only its own
                match most - markers.len() {
                    0 => Ok(states),
                    pad => Tensor::cat(
                        &[
                            states,
                            Tensor::zeros((pad, width), xs.dtype(), xs.device())?,
                        ],
                        0,
                    ),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::stack(&picked, 0)
    }
}

impl IsqModel for Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let root = UnVarBuilder::new();
        root.pp("type_emb").add(&self.type_emb);
        for (i, block) in self.blocks.iter().enumerate() {
            let uvb = root.pp("head").pp("layers").pp(i);
            uvb.pp("norm1").add(&block.norm1);
            uvb.pp("norm2").add(&block.norm2);
        }
        root.pp("scorer").pp(0).add(&self.scorer_norm);
        let mut tensors = root.to_safetensors();
        tensors.extend(
            self.encoder
                .residual_tensors()
                .into_iter()
                .map(|(name, tensor)| (format!("{ENCODER}.{name}"), tensor)),
        );
        tensors
    }
}

impl EmbeddingModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        extras: &crate::embedding::EmbeddingExtras,
        _flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let Some(rows) = &extras.decision else {
            inference_tensor::bail!("a decision model answers decision questions only");
        };
        let xs = self.encoder.hidden_states(input_ids)?;
        let kinds = Tensor::new(rows.kinds.as_slice(), &self.device)?;
        let mut xs = xs.broadcast_add(&self.type_emb.forward(&kinds)?.unsqueeze(1)?)?;
        for block in &self.blocks {
            xs = block.forward(&xs)?;
        }
        let picked = Self::markers(&xs, rows)?;
        let hidden = self
            .scorer_hidden
            .forward(&self.scorer_norm.forward(&picked)?)?
            .gelu_erf()?;
        let logits = self
            .scorer_out
            .forward(&hidden)?
            .squeeze(D::Minus1)?
            .to_dtype(DType::F32)?;
        let temperatures = rows
            .kinds
            .iter()
            .zip(&rows.markers)
            .map(|(&kind, markers)| match QuestionKind::from_index(kind) {
                Some(kind) => Ok(self.spec.temperature(kind, markers.len())),
                None => inference_tensor::bail!("no decision question type {kind}"),
            })
            .collect::<Result<Vec<f32>>>()?;
        let temperatures = Tensor::from_vec(temperatures, (rows.kinds.len(), 1), &self.device)?;
        logits.broadcast_div(&temperatures)
    }
    fn device(&self) -> &Device {
        &self.device
    }
}

impl AnyMoeBaseModelMixin for Model {}
