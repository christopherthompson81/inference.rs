//! The vision tower Qwen2-VL and Qwen2.5-VL share; they differ in their norms and MLP, and Qwen2.5-VL attends in windows.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::attention::AttentionMask;
use std::sync::Arc;

use inference_quant::{ColumnParallelLayer, QuantMethod, RowParallelLayer, ShardedVarBuilder};
use inference_tensor::nn::{LayerNorm, Linear, Module};
use inference_tensor::{D, DType, Device, IndexOp, Result, Tensor};

use crate::{
    layers::{self, Activation, Conv3dConfig, Conv3dNoBias, MatMul, RmsNorm, layer_norm},
    ops::RepeatInterleaveOp,
    utils::unvarbuilder::{ToTensors, UnVarBuilder},
};

const NORM_EPS: f64 = 1e-6;
const ROPE_THETA: f32 = 10000.;
// The tiny test checkpoints draw their weights in load order, so each part loads its tensors as it always has.

/// The tower's geometry, as each model's vision config names it.
#[derive(Clone, Copy)]
pub(crate) struct TowerShape {
    pub in_channels: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
}

#[derive(Clone, Copy)]
pub(crate) enum NormKind {
    Layer,
    Rms,
}

pub(crate) enum VisionNorm {
    Layer(LayerNorm),
    Rms(RmsNorm),
}

impl VisionNorm {
    fn new(kind: NormKind, dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(match kind {
            NormKind::Layer => Self::Layer(layer_norm(dim, NORM_EPS, vb)?),
            NormKind::Rms => Self::Rms(RmsNorm::new(dim, NORM_EPS, vb)?),
        })
    }
}

impl Module for VisionNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Layer(norm) => norm.forward(xs),
            Self::Rms(norm) => norm.forward(xs),
        }
    }
}

impl ToTensors for VisionNorm {
    fn to_tensors(&self) -> std::collections::HashMap<String, Tensor> {
        match self {
            Self::Layer(norm) => norm.to_tensors(),
            Self::Rms(norm) => norm.to_tensors(),
        }
    }
}

pub(crate) struct PatchEmbed {
    proj: Conv3dNoBias,
    shape: TowerShape,
}

// https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L272
impl PatchEmbed {
    pub fn new(shape: TowerShape, vb: ShardedVarBuilder) -> Result<Self> {
        if shape.temporal_patch_size != 2 {
            inference_tensor::bail!("Only support temporal patch size of 2");
        }
        Ok(Self {
            proj: Conv3dNoBias::new(
                shape.in_channels,
                shape.hidden_size,
                [
                    shape.temporal_patch_size,
                    shape.patch_size,
                    shape.patch_size,
                ],
                Conv3dConfig {
                    stride: shape.patch_size,
                    ..Default::default()
                },
                vb.pp("proj"),
            )?,
            shape,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let s = &self.shape;
        let xs = xs.reshape((
            (),
            s.in_channels,
            s.temporal_patch_size,
            s.patch_size,
            s.patch_size,
        ))?;
        xs.apply(&self.proj)?.reshape(((), s.hidden_size))
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let weight = self
            .proj
            .weight()
            .expect("Qwen-VL patch embedding weight reconstruction should succeed");
        uvb.pp("proj").add_tensor("weight", weight);
        uvb.to_safetensors()
    }
}

pub(crate) enum VisionMlp {
    // https://github.com/huggingface/transformers/blob/a769ed45e17c44fd17b85c025863c4e4f2f73634/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L314
    Plain {
        fc1: Arc<dyn QuantMethod>,
        fc2: Arc<dyn QuantMethod>,
        act: Activation,
    },
    // https://github.com/huggingface/transformers/blob/6a1ab634b6886b6560b0502e7a305c8cd881732e/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L75
    Gated {
        gate_proj: Arc<dyn QuantMethod>,
        up_proj: Arc<dyn QuantMethod>,
        down_proj: Arc<dyn QuantMethod>,
        act: Activation,
    },
}

/// Which MLP a block has, with its hidden width and activation.
#[derive(Clone, Copy)]
pub(crate) enum MlpKind {
    Plain { hidden_dim: usize, act: Activation },
    Gated { hidden_dim: usize, act: Activation },
}

impl VisionMlp {
    fn new(
        kind: MlpKind,
        dim: usize,
        vb: ShardedVarBuilder,
        comm: &Arc<inference_quant::Comm>,
    ) -> Result<Self> {
        let column = |hidden_dim, name| {
            ColumnParallelLayer::new(dim, hidden_dim, &None, true, comm, vb.pp(name))
        };
        Ok(match kind {
            MlpKind::Plain { hidden_dim, act } => Self::Plain {
                fc1: column(hidden_dim, "fc1")?,
                fc2: ColumnParallelLayer::new(hidden_dim, dim, &None, true, comm, vb.pp("fc2"))?,
                act,
            },
            MlpKind::Gated { hidden_dim, act } => Self::Gated {
                gate_proj: column(hidden_dim, "gate_proj")?,
                up_proj: column(hidden_dim, "up_proj")?,
                down_proj: RowParallelLayer::new(
                    hidden_dim,
                    dim,
                    &None,
                    true,
                    comm,
                    vb.pp("down_proj"),
                )?,
                act,
            },
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.unsqueeze(0)?;
        match self {
            Self::Plain { fc1, fc2, act } => fc2.forward(&act.forward(&fc1.forward(&xs)?)?)?,
            Self::Gated {
                gate_proj,
                up_proj,
                down_proj,
                act,
            } => {
                let lhs = gate_proj.forward(&xs)?;
                let rhs = up_proj.forward(&xs)?;
                down_proj.forward(&crate::ops::mul_and_act(&lhs, &rhs, *act)?)?
            }
        }
        .squeeze(0)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        match self {
            Self::Plain { fc1, fc2, .. } => {
                uvb.pp("fc1").add(fc1);
                uvb.pp("fc2").add(fc2);
            }
            Self::Gated {
                gate_proj,
                up_proj,
                down_proj,
                ..
            } => {
                uvb.pp("gate_proj").add(gate_proj);
                uvb.pp("up_proj").add(up_proj);
                uvb.pp("down_proj").add(down_proj);
            }
        }
        uvb.to_safetensors()
    }
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

/// `freqs` is `(seq_len, head_dim)`.
fn apply_rotary_pos_emb_vision(xs: &Tensor, freqs: &Tensor) -> Result<Tensor> {
    let cos = freqs.cos()?.unsqueeze(D::Minus2)?.to_dtype(xs.dtype())?;
    let sin = freqs.sin()?.unsqueeze(D::Minus2)?.to_dtype(xs.dtype())?;

    xs.broadcast_mul(&cos)? + rotate_half(xs)?.broadcast_mul(&sin)
}

// https://github.com/huggingface/transformers/blob/a769ed45e17c44fd17b85c025863c4e4f2f73634/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L325
struct VisionAttention {
    qkv: Arc<dyn QuantMethod>,
    proj: Arc<dyn QuantMethod>,
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn new(dim: usize, num_heads: usize, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            qkv: inference_quant::linear(dim, dim * 3, &None, vb.pp("qkv"))?,
            proj: inference_quant::linear(dim, dim, &None, vb.pp("proj"))?,
            num_heads,
            head_dim: dim / num_heads,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        rotary_pos_emb: &Tensor,
    ) -> Result<Tensor> {
        let seq_len = xs.dim(0)?;
        let (mut q, mut k, mut v) = {
            let qkv = self
                .qkv
                .forward(&xs.unsqueeze(0)?)?
                .reshape((seq_len, 3, self.num_heads, ()))?
                .permute((1, 0, 2, 3))?
                .chunk(3, 0)?;
            (qkv[0].squeeze(0)?, qkv[1].squeeze(0)?, qkv[2].squeeze(0)?)
        };

        q = apply_rotary_pos_emb_vision(&q.unsqueeze(0)?, rotary_pos_emb)?
            .squeeze(0)?
            .to_dtype(q.dtype())?;
        k = apply_rotary_pos_emb_vision(&k.unsqueeze(0)?, rotary_pos_emb)?
            .squeeze(0)?
            .to_dtype(q.dtype())?;

        q = q.transpose(0, 1)?.contiguous()?;
        k = k.transpose(0, 1)?.contiguous()?;
        v = v.transpose(0, 1)?.contiguous()?;

        let att = {
            let mut att =
                (MatMul.matmul(&q, &k.transpose(1, 2)?)? / (self.head_dim as f64).sqrt())?;
            att = match attention_mask {
                AttentionMask::Custom(m) => att.broadcast_add(m)?,
                _ => att,
            };
            att = inference_tensor::nn::ops::softmax_last_dim(&att)?;
            MatMul
                .matmul(&att, &v)?
                .transpose(0, 1)?
                .reshape((seq_len, ()))?
                .to_dtype(xs.dtype())?
        };

        self.proj.forward(&att.unsqueeze(0)?)?.squeeze(0)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("qkv").add(&self.qkv);
        uvb.pp("proj").add(&self.proj);
        uvb.to_safetensors()
    }
}

// https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L418
pub(crate) struct VisionBlock {
    norm1: VisionNorm,
    norm2: VisionNorm,
    mlp: VisionMlp,
    attn: VisionAttention,
}

impl VisionBlock {
    pub fn new(
        norm: NormKind,
        shape: TowerShape,
        mlp: MlpKind,
        vb: ShardedVarBuilder,
        comm: &Arc<inference_quant::Comm>,
    ) -> Result<Self> {
        Ok(Self {
            norm1: VisionNorm::new(norm, shape.hidden_size, vb.pp("norm1"))?,
            norm2: VisionNorm::new(norm, shape.hidden_size, vb.pp("norm2"))?,
            mlp: VisionMlp::new(mlp, shape.hidden_size, vb.pp("mlp"), comm)?,
            attn: VisionAttention::new(shape.hidden_size, shape.num_heads, vb.pp("attn"))?,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        rotary_pos_emb: &Tensor,
    ) -> Result<Tensor> {
        let attention_mask = match attention_mask {
            Some(mask) => AttentionMask::Custom(mask.clone()),
            None => AttentionMask::None,
        };
        let xs = (xs
            + self
                .attn
                .forward(&self.norm1.forward(xs)?, &attention_mask, rotary_pos_emb)?)?;
        &xs + self.mlp.forward(&self.norm2.forward(&xs)?)?
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("norm1").add(&self.norm1);
        uvb.pp("norm2").add(&self.norm2);
        uvb.pp("attn").extend(self.attn.residual_tensors());
        uvb.pp("mlp").extend(self.mlp.residual_tensors());
        uvb.to_safetensors()
    }
}

pub(crate) struct PatchMerger {
    ln_q: VisionNorm,
    mlp0: Linear,
    mlp2: Linear,
    hidden_size: usize,
}

impl PatchMerger {
    pub fn new(
        norm: NormKind,
        dim: usize,
        context_dim: usize,
        spatial_merge_size: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let hidden_size = context_dim * spatial_merge_size.pow(2);
        let mlp0 = layers::linear(hidden_size, hidden_size, vb.pp("mlp.0"))?;
        let mlp2 = layers::linear(hidden_size, dim, vb.pp("mlp.2"))?;
        Ok(Self {
            ln_q: VisionNorm::new(norm, context_dim, vb.pp("ln_q"))?,
            mlp0,
            mlp2,
            hidden_size,
        })
    }

    /// The dtype the tower computes in.
    pub fn dtype(&self) -> DType {
        self.mlp0.weight().dtype()
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.unsqueeze(0)?
            .apply(&self.ln_q)?
            .reshape(((), self.hidden_size))?
            .apply(&self.mlp0)?
            .gelu()?
            .apply(&self.mlp2)?
            .squeeze(0)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("ln_q").add(&self.ln_q);
        uvb.pp("mlp").pp(0).add(&self.mlp0);
        uvb.pp("mlp").pp(2).add(&self.mlp2);
        uvb.to_safetensors()
    }
}

/// Each patch's 2D rotary angles, the merge windows' patches kept together.
pub(crate) struct VisionRotaryEmbedding {
    inv_freq: Tensor,
    spatial_merge_size: usize,
}

impl VisionRotaryEmbedding {
    pub fn new(shape: TowerShape, device: &Device) -> Result<Self> {
        let dim = shape.hidden_size / shape.num_heads / 2;
        let inv_freq = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / ROPE_THETA.powf(i as f32 / dim as f32))
            .collect::<Vec<_>>();
        let inv_freq_len = inv_freq.len();
        Ok(Self {
            inv_freq: Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?,
            spatial_merge_size: shape.spatial_merge_size,
        })
    }

    /// `(patches, head_dim / 2)` angles over every item of `grid_thw`.
    pub fn angles(&self, grid_thw: &Tensor, device: &Device) -> Result<Tensor> {
        let merge = self.spatial_merge_size;
        let mut pos_ids = Vec::new();
        for i_thw in grid_thw.to_vec2::<u32>()? {
            let (t, h, w) = (i_thw[0], i_thw[1], i_thw[2]);
            let windows = |ids: Tensor| -> Result<Tensor> {
                ids.reshape((h as usize / merge, merge, w as usize / merge, merge))?
                    .permute((0, 2, 1, 3))?
                    .flatten_all()
            };
            let hpos_ids = windows(
                Tensor::arange(0, h, device)?
                    .unsqueeze(1)?
                    .repeat((1, w as usize))?,
            )?;
            let wpos_ids = windows(
                Tensor::arange(0, w, device)?
                    .unsqueeze(0)?
                    .repeat((h as usize, 1))?,
            )?;
            pos_ids.push(Tensor::stack(&[hpos_ids, wpos_ids], D::Minus1)?.repeat((t as usize, 1))?);
        }
        let pos_ids = Tensor::cat(&pos_ids, 0)?;
        let max_grid_size = grid_thw.i((.., 1..))?.max(0)?.max(0)?.to_scalar::<u32>()?;
        let seq = Tensor::arange(0f32, max_grid_size as f32, self.inv_freq.device())?
            .unsqueeze(D::Minus1)?;
        seq.broadcast_matmul(&self.inv_freq)?
            .index_select(&pos_ids.flatten_all()?, 0)?
            .reshape((pos_ids.dim(0)?, pos_ids.dim(1)?, ()))?
            .flatten_from(1)
    }
}

/// Where each item's patches start and end in the tower's sequence, `0` first.
pub(crate) fn item_cu_seqlens(grid_thw: &Tensor) -> Result<Vec<usize>> {
    let grid_thw = grid_thw.to_device(&Device::Cpu)?;
    Ok((grid_thw.i((.., 1))? * grid_thw.i((.., 2))?)?
        .repeat_interleave_flat(grid_thw.i((.., 0))?.to_vec1::<u32>()?)?
        .to_dtype(DType::F32)?
        .cumsum(0)?
        .to_dtype(DType::U32)?
        .pad_with_zeros(0, 1, 0)?
        .to_vec1::<u32>()?
        .into_iter()
        .map(|len| len as usize)
        .collect())
}

/// Attention within each `cu_seqlens` span only; none when one span covers the sequence.
pub(crate) fn block_diagonal_mask(
    cu_seqlens: &[usize],
    seq_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Option<Tensor>> {
    if cu_seqlens == [0, seq_len] {
        return Ok(None);
    }
    let mut mask = Tensor::full(f32::MIN, (1, seq_len, seq_len), device)?.to_dtype(dtype)?;
    for span in cu_seqlens.windows(2) {
        let (a, b) = (span[0], span[1]);
        mask = mask.slice_assign(
            &[0..1, a..b, a..b],
            &Tensor::zeros((1, b - a, b - a), dtype, device)?,
        )?;
    }
    Ok(Some(mask))
}

pub(crate) fn tower_residual_tensors(
    patch_embed: &PatchEmbed,
    blocks: &[VisionBlock],
    merger: &PatchMerger,
) -> Vec<(String, Tensor)> {
    let uvb = UnVarBuilder::new();
    uvb.pp("patch_embed").extend(patch_embed.residual_tensors());
    let uvb_blocks = uvb.pp("blocks");
    for (layer_idx, block) in blocks.iter().enumerate() {
        uvb_blocks.pp(layer_idx).extend(block.residual_tensors());
    }
    uvb.pp("merger").extend(merger.residual_tensors());
    uvb.to_safetensors()
}
