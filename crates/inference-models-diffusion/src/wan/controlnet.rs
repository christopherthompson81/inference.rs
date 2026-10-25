//! WorldStereo's ControlNet: point-cloud render latents, the render mask and per-pixel camera rays through 20 small
//! attention blocks, one residual per main transformer block.

use inference_quant::ShardedVarBuilder;
use inference_tensor::{DType, Result, Tensor};
use serde::Deserialize;

use super::ops::{
    ACT_DTYPE, AttentionKind, LayerNorm, Linear, PatchConv, RmsNorm, RopeTables, gated_residual,
    gelu_tanh, heads, self_attention,
};
use super::transformer::StepInputs;

// Noise latents plus the 4-channel first-frame mask: the ControlNet's share of the main input
const LATENT_PREFIX_CHANNELS: usize = 20;
const NORM_EPS: f64 = 1e-5;
const QK_NORM_EPS: f64 = 1e-6;
const GROUP_NORM_EPS: f64 = 1e-5;
// GroupNorm groups are mid_channels / 8
const CHANNELS_PER_GROUP: usize = 8;

#[derive(Debug, Clone, Deserialize)]
pub struct ControlNetConfig {
    pub conv_out_dim: usize,
    pub dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub add_channels: usize,
    pub mid_channels: usize,
    pub camera_embedding_dim: usize,
}

// SiLU(temb) -> shift, scale, gate; LayerNorm with affine terms
struct AdaNorm {
    linear: Linear,
    norm: LayerNorm,
}

impl AdaNorm {
    fn load(time_dim: usize, dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            linear: Linear::load(time_dim, 3 * dim, vb.pp("linear"))?,
            norm: LayerNorm::load(dim, NORM_EPS, vb.pp("norm"))?,
        })
    }

    // the modulated input and the gate
    fn forward(&self, xs: &Tensor, temb: &Tensor) -> Result<(Tensor, Tensor)> {
        let m = self.linear.forward(&temb.silu()?)?.to_dtype(DType::F32)?;
        let dim = m.dim(1)? / 3;
        let (shift, scale, gate) = (
            m.narrow(1, 0, dim)?,
            m.narrow(1, dim, dim)?,
            m.narrow(1, 2 * dim, dim)?,
        );
        Ok((self.norm.forward_act(xs, Some((&shift, &scale)))?, gate))
    }
}

struct ControlBlock {
    norm1: AdaNorm,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    norm2: AdaNorm,
    ffn_up: Linear,
    ffn_down: Linear,
    heads: usize,
}

impl ControlBlock {
    fn load(cfg: &ControlNetConfig, time_dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let (d, attn) = (cfg.dim, vb.pp("self_attn"));
        Ok(Self {
            norm1: AdaNorm::load(time_dim, d, vb.pp("norm1"))?,
            to_q: Linear::load(d, d, attn.pp("to_q"))?,
            to_k: Linear::load(d, d, attn.pp("to_k"))?,
            to_v: Linear::load(d, d, attn.pp("to_v"))?,
            to_out: Linear::load(d, d, attn.pp("to_out").pp(0))?,
            norm_q: RmsNorm::load(d, QK_NORM_EPS, attn.pp("norm_q"))?,
            norm_k: RmsNorm::load(d, QK_NORM_EPS, attn.pp("norm_k"))?,
            norm2: AdaNorm::load(time_dim, d, vb.pp("norm2"))?,
            ffn_up: Linear::load(d, cfg.ffn_dim, vb.pp("ffn").pp(0))?,
            ffn_down: Linear::load(cfg.ffn_dim, d, vb.pp("ffn").pp(2))?,
            heads: cfg.num_heads,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        temb: &Tensor,
        rope: &RopeTables,
        kind: AttentionKind,
    ) -> Result<Tensor> {
        let (normed, gate) = self.norm1.forward(xs, temb)?;
        let q = self
            .norm_q
            .heads(&self.to_q.forward(&normed)?, self.heads, Some(rope))?;
        let k = self
            .norm_k
            .heads(&self.to_k.forward(&normed)?, self.heads, Some(rope))?;
        let v = heads(&self.to_v.forward(&normed)?, self.heads)?;
        let xs = gated_residual(
            xs,
            &self.to_out.forward(&self_attention(&q, &k, &v, kind)?)?,
            &gate,
        )?;
        let (normed, gate) = self.norm2.forward(&xs, temb)?;
        let ff = self
            .ffn_down
            .forward(&gelu_tanh(&self.ffn_up.forward(&normed)?)?)?;
        gated_residual(&xs, &ff, &gate)
    }
}

pub(crate) struct ControlNet {
    patch_embedding: PatchConv,
    mask_proj: PatchConv,
    group_norm_weight: Tensor,
    group_norm_bias: Tensor,
    mask_zero_proj: PatchConv,
    proj_in: Linear,
    blocks: Vec<ControlBlock>,
    proj_out: Vec<Linear>,
    head_dim: usize,
    groups: usize,
}

impl ControlNet {
    pub(crate) fn load(
        cfg: &ControlNetConfig,
        time_dim: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let mask = vb.pp("controlnet_mask_embedding");
        Ok(Self {
            patch_embedding: PatchConv::load(&vb.pp("controlnet_patch_embedding"))?,
            mask_proj: PatchConv::load(&mask.pp("mask_proj").pp(0))?,
            group_norm_weight: mask
                .pp("mask_proj")
                .pp(1)
                .get(cfg.mid_channels, "weight")?
                .to_dtype(DType::F32)?,
            group_norm_bias: mask
                .pp("mask_proj")
                .pp(1)
                .get(cfg.mid_channels, "bias")?
                .to_dtype(DType::F32)?,
            mask_zero_proj: PatchConv::load(&mask.pp("mask_zero_proj"))?,
            proj_in: Linear::load(cfg.conv_out_dim, cfg.dim, vb.pp("proj_in"))?,
            blocks: (0..cfg.num_layers)
                .map(|i| ControlBlock::load(cfg, time_dim, vb.pp("controlnet_blocks").pp(i)))
                .collect::<Result<_>>()?,
            proj_out: (0..cfg.num_layers)
                .map(|i| Linear::load(cfg.dim, time_dim, vb.pp("proj_out").pp(i)))
                .collect::<Result<_>>()?,
            head_dim: cfg.dim / cfg.num_heads,
            groups: cfg.mid_channels / CHANNELS_PER_GROUP,
        })
    }

    // GroupNorm over (channels in the group, every frame, every pixel), as on the 5D `[b, c, f, h, w]` tensor
    fn group_norm(&self, tokens: &Tensor) -> Result<Tensor> {
        let (n, c) = tokens.dims2()?;
        let g = self.groups;
        let x = tokens.to_dtype(DType::F32)?.reshape((n, g, c / g))?;
        let mean = x.mean_keepdim(2)?.mean_keepdim(0)?;
        let centered = x.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(2)?.mean_keepdim(0)?;
        centered
            .broadcast_div(&(var + GROUP_NORM_EPS)?.sqrt()?)?
            .reshape((n, c))?
            .broadcast_mul(&self.group_norm_weight)?
            .broadcast_add(&self.group_norm_bias)?
            .to_dtype(ACT_DTYPE)
    }

    /// The state the ControlNet blocks run on, for a latent grid of `f x h x w`; the blocks then step it one at a time
    /// beside the main blocks, so only one residual exists at once.
    pub(crate) fn start<'a>(
        &'a self,
        inputs: &StepInputs,
        temb: &'a Tensor,
        (f, h, w): (usize, usize, usize),
        attention: AttentionKind,
    ) -> Result<ControlState<'a>> {
        let input = Tensor::cat(
            &[
                &inputs.latents.narrow(0, 0, LATENT_PREFIX_CHANNELS)?,
                &inputs.render_latent.to_dtype(inputs.latents.dtype())?,
            ],
            0,
        )?;
        let tokens = self.patch_embedding.forward(&input)?;
        // the 8x8 patch conv leaves one cell per latent pixel, laid out frame, row, column
        let cells = self
            .group_norm(&self.mask_proj.forward(inputs.render_conditions)?)?
            .silu()?;
        let c = cells.dim(1)?;
        let cells = cells.reshape((f, h, w, c))?.permute((3, 0, 1, 2))?;
        let tokens = (tokens + self.mask_zero_proj.forward(&cells)?)?;
        Ok(ControlState {
            net: self,
            xs: self.proj_in.forward(&tokens.unsqueeze(0)?)?,
            temb,
            rope: RopeTables::grid(self.head_dim, (f, h / 2, w / 2), inputs.latents.device())?,
            next: 0,
            attention,
        })
    }
}

/// The ControlNet partway through its blocks.
pub(crate) struct ControlState<'a> {
    net: &'a ControlNet,
    xs: Tensor,
    temb: &'a Tensor,
    rope: RopeTables,
    next: usize,
    attention: AttentionKind,
}

impl ControlState<'_> {
    /// The next block's residual `[1, N, main dim]`, or None once every block has run.
    pub(crate) fn step(&mut self) -> Result<Option<Tensor>> {
        let Some(block) = self.net.blocks.get(self.next) else {
            return Ok(None);
        };
        self.xs = block.forward(&self.xs, self.temb, &self.rope, self.attention)?;
        let residual = self.net.proj_out[self.next].forward(&self.xs)?;
        self.next += 1;
        Ok(Some(residual))
    }
}
