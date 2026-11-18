//! The denoiser (`LatentSeqMMFlowModel`): the object latent (8192 tokens), the image tokens and the camera (1 token)
//! each refined on their own, then joint blocks over all of them; each block's rotary positions are predicted from the
//! tokens themselves (`RePo3DRotaryEmbedding`). Its blocks run in fp16 as shipped, the rest in f32.

use inference_tensor::nn::VarBuilder;
use inference_tensor::{D, DType, Result, Tensor};

use super::layers::{
    BLOCK_EPS, DEFAULT_EPS, FeedForward, HeadRmsNorm, Norm32, ScalarEmbedder, attention, plain_norm,
};
use crate::worldmirror::layers::AmpLinear;

#[derive(Clone, Copy, Debug)]
pub(crate) struct DitSpec {
    pub latent_tokens: usize,
    pub latent_channels: usize,
    pub camera_channels: usize,
    pub channels: usize,
    pub image_channels: usize,
    pub vae_channels: usize,
    pub blocks: usize,
    pub refiner_blocks: usize,
    pub heads: usize,
    pub mlp_ratio: usize,
}

/// `speridlabs/idol`'s `config.json` denoiser.
pub(crate) const IDOL_DIT: DitSpec = DitSpec {
    latent_tokens: 8192,
    latent_channels: 16,
    camera_channels: 5,
    channels: 1024,
    image_channels: 1280,
    vae_channels: 128,
    blocks: 24,
    refiner_blocks: 2,
    heads: 16,
    mlp_ratio: 4,
};

// the rotary layers' hidden width, as a share of the model's
const REPO_HIDDEN_DIVISOR: usize = 8;
// latent positions' sinusoid: 2^0 .. 2^15 octaves, then fractional ones up to 2^16
const POSITION_OCTAVES: usize = 16;
const TIME_MAX_PERIOD: f64 = 10000.0;
const MOD_CHUNKS: usize = 6;

/// `RePo3DRotaryEmbedding`: each token's features predict a 3D offset per head, turned into rotary angles (f32).
struct LearnedRope {
    norm: Norm32,
    gate: AmpLinear,
    content: AmpLinear,
    out: AmpLinear,
    // per head-dim pair: its axis' frequency times pi, `[1, 1, 1, head_dim / 2]`, and which axis (0..3) it reads
    freqs: Tensor,
    axis_index: Tensor,
    heads: usize,
}

impl LearnedRope {
    fn load(spec: DitSpec, vb: VarBuilder) -> Result<Self> {
        let c = spec.channels;
        let hidden = c / REPO_HIDDEN_DIVISOR;
        let head_dim = c / spec.heads;
        let d01 = 2 * (head_dim / 6);
        let dims = [d01, d01, head_dim - 2 * d01];
        let mut freqs = Vec::new();
        let mut axis = Vec::new();
        for (i, d) in dims.iter().enumerate() {
            freqs.push(vb.get(d / 2, &format!("freqs_{i}"))?);
            axis.extend(std::iter::repeat_n(i as u32, d / 2));
        }
        let pairs = head_dim / 2;
        Ok(Self {
            norm: Norm32::load(c, true, DEFAULT_EPS, vb.pp("norm"))?,
            gate: AmpLinear::load(c, hidden, false, DType::F32, vb.pp("gate_map"))?,
            content: AmpLinear::load(c, hidden, false, DType::F32, vb.pp("content_map"))?,
            out: AmpLinear::load(
                hidden,
                3 * spec.heads,
                false,
                DType::F32,
                vb.pp("final_map"),
            )?,
            freqs: (Tensor::cat(&freqs, 0)? * std::f64::consts::PI)?.reshape((1, 1, 1, pairs))?,
            axis_index: Tensor::from_vec(axis, pairs, vb.device())?,
            heads: spec.heads,
        })
    }

    /// `(cos, sin)` `[b, l, heads, head_dim / 2]` for tokens `[b, l, c]`.
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, l, _) = xs.dims3()?;
        let h = self.norm.forward(&xs.to_dtype(DType::F32)?)?;
        let delta = self
            .out
            .forward(&(self.gate.forward(&h)?.silu()? * self.content.forward(&h)?)?)?
            .reshape((b, l, self.heads, 3))?;
        // each pair's axis offset (the reference's clamp_mul is x * f in the forward pass)
        let per_pair = delta.index_select(&self.axis_index, D::Minus1)?;
        let angles = per_pair.broadcast_mul(&self.freqs)?;
        Ok((angles.cos()?, angles.sin()?))
    }
}

// rotates adjacent pairs of `[b, l, heads, d]` by `(cos, sin)` `[b, l, heads, d / 2]`, in f32
fn rotate(xs: &Tensor, (cos, sin): &(Tensor, Tensor)) -> Result<Tensor> {
    let (b, l, h, d) = xs.dims4()?;
    let pairs = xs.to_dtype(DType::F32)?.reshape((b, l, h, d / 2, 2))?;
    let x0 = pairs.narrow(4, 0, 1)?.squeeze(4)?;
    let x1 = pairs.narrow(4, 1, 1)?.squeeze(4)?;
    let r0 = ((&x0 * cos)? - (&x1 * sin)?)?;
    let r1 = ((&x0 * sin)? + (&x1 * cos)?)?;
    Tensor::stack(&[r0, r1], 4)?
        .reshape((b, l, h, d))?
        .to_dtype(xs.dtype())
}

struct Block {
    norm1: Norm32,
    norm2: Norm32,
    qkv: AmpLinear,
    q_norm: HeadRmsNorm,
    k_norm: HeadRmsNorm,
    out: AmpLinear,
    mlp: FeedForward,
    // `[1, 6c]` added to the shared time modulation; none for the image-token refiner
    shift_table: Option<Tensor>,
    heads: usize,
}

impl Block {
    fn load(spec: DitSpec, modulated: bool, dtype: DType, vb: VarBuilder) -> Result<Self> {
        let c = spec.channels;
        let head_dim = c / spec.heads;
        let attn = vb.pp("attn");
        Ok(Self {
            norm1: Norm32::load(c, !modulated, BLOCK_EPS, vb.pp("norm1"))?,
            norm2: Norm32::load(c, !modulated, BLOCK_EPS, vb.pp("norm2"))?,
            qkv: AmpLinear::load(c, 3 * c, true, dtype, attn.pp("qkv"))?,
            q_norm: HeadRmsNorm::load(spec.heads, head_dim, attn.pp("q_norm"))?,
            k_norm: HeadRmsNorm::load(spec.heads, head_dim, attn.pp("k_norm"))?,
            out: AmpLinear::load(c, c, true, dtype, attn.pp("out"))?,
            mlp: FeedForward::load(c, spec.mlp_ratio * c, dtype, vb.pp("mlp.mlp"))?,
            shift_table: modulated
                .then(|| vb.get((1, MOD_CHUNKS * c), "shift_table"))
                .transpose()?,
            heads: spec.heads,
        })
    }

    fn attend(&self, xs: &Tensor, rope: &(Tensor, Tensor)) -> Result<Tensor> {
        let (b, l, c) = xs.dims3()?;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, l, 3, self.heads, c / self.heads))?;
        let part = |i: usize| qkv.narrow(2, i, 1)?.squeeze(2);
        let q = self.q_norm.forward(&rotate(&part(0)?, rope)?)?;
        let k = self.k_norm.forward(&rotate(&part(1)?, rope)?)?;
        let v = part(2)?.contiguous()?;
        self.out
            .forward(&attention(&q, &k, &v)?.reshape((b, l, c))?)
    }

    /// `mod_` the shared time modulation `[b, 6c]` in the stream's dtype; ignored by the unmodulated refiner.
    fn forward(&self, xs: &Tensor, mod_: &Tensor, rope: &(Tensor, Tensor)) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if crate::trellis2::dit_cuda::applies(xs) {
            return self.forward_fused(xs, mod_, rope);
        }
        let Some(shift_table) = &self.shift_table else {
            let xs = (xs + self.attend(&self.norm1.forward(xs)?, rope)?)?;
            return &xs + self.mlp.forward(&self.norm2.forward(&xs)?)?;
        };
        let m = mod_.broadcast_add(&shift_table.to_dtype(mod_.dtype())?)?;
        let chunk = |i: usize| m.chunk(MOD_CHUNKS, 1)?[i].unsqueeze(1);
        let (shift_msa, scale_msa, gate_msa) = (chunk(0)?, chunk(1)?, chunk(2)?);
        let (shift_mlp, scale_mlp, gate_mlp) = (chunk(3)?, chunk(4)?, chunk(5)?);
        let h = self.norm1.forward(xs)?;
        let h = h
            .broadcast_mul(&(scale_msa + 1.0)?)?
            .broadcast_add(&shift_msa)?;
        let xs = (xs + self.attend(&h, rope)?.broadcast_mul(&gate_msa)?)?;
        let h = self.norm2.forward(&xs)?;
        let h = h
            .broadcast_mul(&(scale_mlp + 1.0)?)?
            .broadcast_add(&shift_mlp)?;
        &xs + self.mlp.forward(&h)?.broadcast_mul(&gate_mlp)?
    }
}

#[cfg(feature = "cuda")]
impl Block {
    // `forward` with its norms, modulation, rotations and residuals fused, the same arithmetic
    fn forward_fused(&self, xs: &Tensor, mod_: &Tensor, rope: &(Tensor, Tensor)) -> Result<Tensor> {
        use super::layers::NORMALIZE_EPS;
        use crate::trellis2::dit::{sdpa, unbiased};
        use crate::trellis2::dit_cuda::{Rotation, bias_gelu, heads, ln_mod, residual};
        // chunks of the modulation: shift, scale and gate for attention, then for the MLP
        const SHIFT_MSA: usize = 0;
        const SCALE_MSA: usize = 1;
        const GATE_MSA: usize = 2;
        const SHIFT_MLP: usize = 3;
        const SCALE_MLP: usize = 4;
        const GATE_MLP: usize = 5;
        let (b, n, c) = xs.dims3()?;
        let rotation = Some(Rotation {
            cos: &rope.0,
            sin: &rope.1,
            per_head: true,
            first: true,
        });
        let head_eps = NORMALIZE_EPS as f32;
        let attend = |h: &Tensor| -> Result<(Tensor, &Tensor)> {
            let (qkv, qkv_bias) = unbiased(&self.qkv, h)?;
            let qkv = qkv.reshape((b, n, 3, self.heads, ()))?;
            let q = heads(
                &qkv,
                qkv_bias,
                0,
                Some(self.q_norm.gain()),
                rotation,
                head_eps,
            )?;
            let k = heads(
                &qkv,
                qkv_bias,
                1,
                Some(self.k_norm.gain()),
                rotation,
                head_eps,
            )?;
            let v = heads(&qkv, qkv_bias, 2, None, None, head_eps)?;
            let attn = sdpa(&q, &k, &v)?;
            unbiased(&self.out, &attn.transpose(1, 2)?.reshape((b, n, c))?)
        };
        let (up, down) = self.mlp.linears();
        let mlp = |h: &Tensor| -> Result<(Tensor, &Tensor)> {
            let (h, up_bias) = unbiased(up, h)?;
            unbiased(down, &bias_gelu(&h, up_bias)?)
        };
        let (eps1, eps2) = (self.norm1.eps(), self.norm2.eps());
        let Some(shift_table) = &self.shift_table else {
            let (y, bias) = attend(&ln_mod(xs, self.norm1.affine(), None, eps1)?)?;
            let xs = residual(xs, &y, bias, None)?;
            let (y, bias) = mlp(&ln_mod(&xs, self.norm2.affine(), None, eps2)?)?;
            return residual(&xs, &y, bias, None);
        };
        let m = mod_.broadcast_add(&shift_table.to_dtype(mod_.dtype())?)?;
        let (y, bias) = attend(&ln_mod(xs, None, Some((&m, SCALE_MSA, SHIFT_MSA)), eps1)?)?;
        let xs = residual(xs, &y, bias, Some((&m, GATE_MSA)))?;
        let (y, bias) = mlp(&ln_mod(&xs, None, Some((&m, SCALE_MLP, SHIFT_MLP)), eps2)?)?;
        residual(&xs, &y, bias, Some((&m, GATE_MLP)))
    }
}

/// The denoiser's velocity, split as its input is.
pub(crate) struct Velocity {
    pub latent: Tensor,
    pub camera: Tensor,
}

pub(crate) struct Dit {
    t_embedder: ScalarEmbedder,
    ada_ln: AmpLinear,
    input_layer: AmpLinear,
    cond_embedder: AmpLinear,
    cond_embedder2: AmpLinear,
    // the latent tokens' fixed positions, embedded once (`[1, latent_tokens, c]`, f32)
    position_embedding: Tensor,
    noise_ropes: Vec<LearnedRope>,
    context_ropes: Vec<LearnedRope>,
    ropes: Vec<LearnedRope>,
    noise_refiner: Vec<Block>,
    context_refiner: Vec<Block>,
    cam_refiner: (AmpLinear, AmpLinear),
    blocks: Vec<Block>,
    shift_table: Tensor,
    out_layer: AmpLinear,
    cam_out_layer: AmpLinear,
    spec: DitSpec,
    dtype: DType,
}

impl Dit {
    /// `dtype` the blocks' (`F16` as shipped).
    pub(crate) fn load(spec: DitSpec, dtype: DType, vb: VarBuilder) -> Result<Self> {
        let c = spec.channels;
        let f32_linear =
            |i: usize, o: usize, name: &str| AmpLinear::load(i, o, true, DType::F32, vb.pp(name));
        let ropes = |name: &str, n: usize| -> Result<Vec<LearnedRope>> {
            (0..n)
                .map(|i| LearnedRope::load(spec, vb.pp(name).pp(i)))
                .collect()
        };
        let blocks = |name: &str, n: usize, modulated: bool| -> Result<Vec<Block>> {
            (0..n)
                .map(|i| Block::load(spec, modulated, dtype, vb.pp(name).pp(i)))
                .collect()
        };
        let positions = vb.get((1, spec.latent_tokens, 3), "pos_pe")?;
        Ok(Self {
            t_embedder: ScalarEmbedder::load(c, TIME_MAX_PERIOD, false, vb.pp("t_embedder"))?,
            ada_ln: AmpLinear::load(
                c,
                MOD_CHUNKS * c,
                true,
                DType::F32,
                vb.pp("adaLN_modulation.1"),
            )?,
            input_layer: f32_linear(spec.latent_channels, c, "input_layer")?,
            cond_embedder: f32_linear(spec.image_channels, c, "cond_embedder")?,
            cond_embedder2: f32_linear(spec.vae_channels, c, "cond_embedder2")?,
            position_embedding: position_embedding(&positions, c)?,
            noise_ropes: ropes("noise_repo_layers", spec.refiner_blocks)?,
            context_ropes: ropes("context_repo_layers", spec.refiner_blocks)?,
            ropes: ropes("repo_layers", spec.blocks)?,
            noise_refiner: blocks("noise_refiner", spec.refiner_blocks, true)?,
            context_refiner: blocks("context_refiner", spec.refiner_blocks, false)?,
            cam_refiner: (
                AmpLinear::load(
                    spec.camera_channels,
                    c,
                    true,
                    DType::F32,
                    vb.pp("cam_refiner.mlp.0"),
                )?,
                AmpLinear::load(c, c, true, DType::F32, vb.pp("cam_refiner.mlp.2"))?,
            ),
            blocks: blocks("blocks", spec.blocks, true)?,
            shift_table: vb.get((1, 2, c), "shift_table")?,
            out_layer: f32_linear(c, spec.latent_channels, "out_layer")?,
            cam_out_layer: f32_linear(c, spec.camera_channels, "cam_out_layer")?,
            spec,
            dtype,
        })
    }

    /// The image tokens embedded and refined once per image: `feature1` `[b, n, 1280]`, `feature2` `[b, n, 128]`.
    pub(crate) fn condition(&self, feature1: &Tensor, feature2: &Tensor) -> Result<Tensor> {
        let mut h = (self.cond_embedder.forward(feature1)?
            + self.cond_embedder2.forward(feature2)?)?
        .to_dtype(self.dtype)?;
        for (rope, block) in self.context_ropes.iter().zip(&self.context_refiner) {
            h = block.forward(&h, &h, &rope.forward(&h)?)?;
        }
        Ok(h)
    }

    /// The velocity at time `t` (`[b]`, in [0, 1000]) of `latent` `[b, 8192, 16]` and `camera` `[b, 1, 5]` (f32),
    /// given `condition` from `condition`.
    pub(crate) fn forward(
        &self,
        latent: &Tensor,
        camera: &Tensor,
        t: &Tensor,
        condition: &Tensor,
    ) -> Result<Velocity> {
        let (b, n_latent, _) = latent.dims3()?;
        let t_emb = self.t_embedder.forward(t)?;
        let t_mod = self.ada_ln.forward(&t_emb.silu()?)?.to_dtype(self.dtype)?;

        let mut h_x = self.input_layer.forward(latent)?.to_dtype(self.dtype)?;
        h_x = h_x.broadcast_add(&self.position_embedding.to_dtype(self.dtype)?)?;
        for (rope, block) in self.noise_ropes.iter().zip(&self.noise_refiner) {
            h_x = block.forward(&h_x, &t_mod, &rope.forward(&h_x)?)?;
        }
        let h_cam = self
            .cam_refiner
            .1
            .forward(&self.cam_refiner.0.forward(camera)?.gelu()?)?
            .to_dtype(self.dtype)?;
        let condition = if condition.dim(0)? == b {
            condition.clone()
        } else {
            condition
                .broadcast_as((b, condition.dim(1)?, self.spec.channels))?
                .contiguous()?
        };
        let mut h = Tensor::cat(&[&h_x, &condition, &h_cam], 1)?;
        for (rope, block) in self.ropes.iter().zip(&self.blocks) {
            h = block.forward(&h, &t_mod, &rope.forward(&h)?)?;
        }
        let total = h.dim(1)?;
        let shift_scale = self.shift_table.broadcast_add(&t_emb.unsqueeze(1)?)?;
        let shift = shift_scale.narrow(1, 0, 1)?;
        let scale = (shift_scale.narrow(1, 1, 1)? + 1.0)?;
        let head = |h: Tensor, layer: &AmpLinear| -> Result<Tensor> {
            let h = plain_norm(&h, DEFAULT_EPS)?
                .broadcast_mul(&scale)?
                .broadcast_add(&shift)?;
            layer.forward(&h)
        };
        Ok(Velocity {
            latent: head(h.narrow(1, 0, n_latent)?, &self.out_layer)?,
            camera: head(h.narrow(1, total - 1, 1)?, &self.cam_out_layer)?,
        })
    }
}

// `PcdAbsolutePositionEmbedder`: per coordinate, sin then cos of 2 pi x f over its frequencies, zero-padded to `c`
fn position_embedding(positions: &Tensor, channels: usize) -> Result<Tensor> {
    let freq_dim = channels / 3 / 2;
    let extra = freq_dim.saturating_sub(POSITION_OCTAVES);
    let freqs: Vec<f32> = (0..POSITION_OCTAVES)
        .map(|i| i as f32)
        .chain((0..extra).map(|i| i as f32 / extra as f32 * POSITION_OCTAVES as f32))
        .take(freq_dim)
        .map(|e| 2f32.powf(e))
        .collect();
    let (b, l, _) = positions.dims3()?;
    let freqs = Tensor::from_vec(freqs, (1, 1, 1, freq_dim), positions.device())?;
    let angles = (positions.unsqueeze(3)?.broadcast_mul(&freqs)? * (2.0 * std::f64::consts::PI))?;
    let embed =
        Tensor::cat(&[angles.sin()?, angles.cos()?], 3)?.reshape((b, l, 3 * 2 * freq_dim))?;
    let pad = channels - 3 * 2 * freq_dim;
    if pad == 0 {
        return Ok(embed);
    }
    Tensor::cat(
        &[
            embed,
            Tensor::zeros((b, l, pad), DType::F32, positions.device())?,
        ],
        2,
    )
}
