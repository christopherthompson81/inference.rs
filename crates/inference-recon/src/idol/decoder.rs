//! The two decoders that read Idol's latent: the octree decoder splits a budget of anchor points among each cell's 8
//! children level by level (`OctreeProbabilityFixedlenDecoder`), and the Gaussian decoder turns each anchor into 32
//! Gaussians (`GaussianFixedlenDecoder`). Their blocks run in fp16 as shipped, the rest in f32.

use inference_tensor::nn::VarBuilder;
use inference_tensor::{DType, Result, Tensor};
use rand::Rng;
use rand::rngs::StdRng;

use super::layers::{
    BLOCK_EPS, DEFAULT_EPS, FeedForward, HeadRmsNorm, Norm32, ScalarEmbedder, attention, plain_norm,
};
use crate::worldmirror::layers::AmpLinear;

#[derive(Clone, Copy, Debug)]
pub(crate) struct DecoderSpec {
    pub channels: usize,
    pub latent_channels: usize,
    pub blocks: usize,
    pub heads: usize,
    pub mlp_ratio: usize,
}

/// `speridlabs/idol`'s `config.json` decoder (octree).
pub(crate) const IDOL_OCTREE: DecoderSpec = DecoderSpec {
    channels: 1024,
    latent_channels: 16,
    blocks: 4,
    heads: 16,
    mlp_ratio: 4,
};

/// `speridlabs/idol`'s `config.json` decoder_gs.
pub(crate) const IDOL_GAUSSIANS: DecoderSpec = DecoderSpec {
    channels: 1024,
    latent_channels: 16,
    blocks: 16,
    heads: 16,
    mlp_ratio: 4,
};

// the query points' sinusoid spans 2^0 .. 2^10
const POINT_OCTAVES: f64 = 10.0;
const LEVEL_MAX_PERIOD: f64 = 1024.0;
const MOD_CHUNKS: usize = 6;
const CHILDREN: usize = 8;
/// Gaussians per anchor.
pub(crate) const GAUSSIANS_PER_ANCHOR: usize = 32;

// `PcdAbsolutePositionEmbedderv2`: per coordinate, sin then cos of pi x f over 2^linspace(0, 10), zero-padded
fn point_embedding(points: &Tensor, channels: usize) -> Result<Tensor> {
    let freq_dim = channels / 3 / 2;
    let freqs: Vec<f32> = (0..freq_dim)
        .map(|i| {
            let e = if freq_dim > 1 {
                POINT_OCTAVES * i as f64 / (freq_dim - 1) as f64
            } else {
                0.0
            };
            2f32.powf(e as f32)
        })
        .collect();
    let (b, l, _) = points.dims3()?;
    let freqs = Tensor::from_vec(freqs, (1, 1, 1, freq_dim), points.device())?;
    let angles = (points.unsqueeze(3)?.broadcast_mul(&freqs)? * std::f64::consts::PI)?;
    let embed = Tensor::cat(&[angles.sin()?, angles.cos()?], 3)?.reshape((b, l, 6 * freq_dim))?;
    let pad = channels - 6 * freq_dim;
    if pad == 0 {
        return Ok(embed);
    }
    Tensor::cat(
        &[
            embed,
            Tensor::zeros((b, l, pad), DType::F32, points.device())?,
        ],
        2,
    )
}

/// Self or cross attention with RMS-normed queries and keys (`MultiHeadAttention`).
struct Attention {
    qkv: Option<AmpLinear>,
    q: Option<AmpLinear>,
    kv: Option<AmpLinear>,
    q_norm: HeadRmsNorm,
    k_norm: HeadRmsNorm,
    out: AmpLinear,
    heads: usize,
}

impl Attention {
    fn load(spec: DecoderSpec, cross: bool, dtype: DType, vb: VarBuilder) -> Result<Self> {
        let c = spec.channels;
        let head_dim = c / spec.heads;
        Ok(Self {
            qkv: (!cross)
                .then(|| AmpLinear::load(c, 3 * c, true, dtype, vb.pp("to_qkv")))
                .transpose()?,
            q: cross
                .then(|| AmpLinear::load(c, c, true, dtype, vb.pp("to_q")))
                .transpose()?,
            kv: cross
                .then(|| AmpLinear::load(spec.latent_channels, 2 * c, true, dtype, vb.pp("to_kv")))
                .transpose()?,
            q_norm: HeadRmsNorm::load(spec.heads, head_dim, vb.pp("q_rms_norm"))?,
            k_norm: HeadRmsNorm::load(spec.heads, head_dim, vb.pp("k_rms_norm"))?,
            out: AmpLinear::load(c, c, true, dtype, vb.pp("to_out"))?,
            heads: spec.heads,
        })
    }

    fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let (b, l, c) = xs.dims3()?;
        let d = c / self.heads;
        let (q, k, v) = match (&self.qkv, context) {
            (Some(qkv), None) => {
                let qkv = qkv.forward(xs)?.reshape((b, l, 3, self.heads, d))?;
                let part = |i: usize| qkv.narrow(2, i, 1)?.squeeze(2);
                (part(0)?, part(1)?, part(2)?)
            }
            (None, Some(context)) => {
                let (q_layer, kv_layer) = (
                    self.q.as_ref().expect("cross"),
                    self.kv.as_ref().expect("cross"),
                );
                let s = context.dim(1)?;
                let kv = kv_layer
                    .forward(context)?
                    .reshape((b, s, 2, self.heads, d))?;
                let part = |i: usize| kv.narrow(2, i, 1)?.squeeze(2);
                (
                    q_layer.forward(xs)?.reshape((b, l, self.heads, d))?,
                    part(0)?,
                    part(1)?,
                )
            }
            _ => inference_tensor::bail!("an attention layer called with the wrong context"),
        };
        let h = attention(
            &self.q_norm.forward(&q)?,
            &self.k_norm.forward(&k)?,
            &v.contiguous()?,
        )?;
        self.out.forward(&h.reshape((b, l, c))?)
    }
}

/// A decoder block: the octree's (cross attention + FFN, modulated by the level) or the Gaussian decoder's (self
/// attention, cross attention, FFN).
struct Block {
    norm1: Norm32,
    norm2: Norm32,
    norm3: Option<Norm32>,
    self_attn: Option<Attention>,
    cross_attn: Attention,
    mlp: FeedForward,
}

impl Block {
    fn load(spec: DecoderSpec, modulated: bool, dtype: DType, vb: VarBuilder) -> Result<Self> {
        let c = spec.channels;
        Ok(Self {
            norm1: Norm32::load(c, false, BLOCK_EPS, vb.pp("norm1"))?,
            norm2: Norm32::load(c, !modulated, BLOCK_EPS, vb.pp("norm2"))?,
            norm3: (!modulated)
                .then(|| Norm32::load(c, false, BLOCK_EPS, vb.pp("norm3")))
                .transpose()?,
            self_attn: (!modulated)
                .then(|| Attention::load(spec, false, dtype, vb.pp("self_attn")))
                .transpose()?,
            cross_attn: Attention::load(spec, true, dtype, vb.pp("cross_attn"))?,
            mlp: FeedForward::load(c, spec.mlp_ratio * c, dtype, vb.pp("mlp.mlp"))?,
        })
    }

    fn forward(&self, xs: &Tensor, mod_: Option<&Tensor>, context: &Tensor) -> Result<Tensor> {
        let Some(m) = mod_ else {
            let self_attn = self
                .self_attn
                .as_ref()
                .expect("unmodulated blocks attend to themselves");
            let xs = (xs + self_attn.forward(&self.norm1.forward(xs)?, None)?)?;
            let xs = (&xs
                + self
                    .cross_attn
                    .forward(&self.norm2.forward(&xs)?, Some(context))?)?;
            let norm3 = self
                .norm3
                .as_ref()
                .expect("unmodulated blocks have a third norm");
            return &xs + self.mlp.forward(&norm3.forward(&xs)?)?;
        };
        let chunk = |i: usize| m.chunk(MOD_CHUNKS, 1)?[i].unsqueeze(1);
        let (shift_msa, scale_msa, gate_msa) = (chunk(0)?, chunk(1)?, chunk(2)?);
        let (shift_mlp, scale_mlp, gate_mlp) = (chunk(3)?, chunk(4)?, chunk(5)?);
        let h = self
            .norm1
            .forward(xs)?
            .broadcast_mul(&(scale_msa + 1.0)?)?
            .broadcast_add(&shift_msa)?;
        let xs = (xs
            + self
                .cross_attn
                .forward(&h, Some(context))?
                .broadcast_mul(&gate_msa)?)?;
        let h = self
            .norm2
            .forward(&xs)?
            .broadcast_mul(&(scale_mlp + 1.0)?)?
            .broadcast_add(&shift_mlp)?;
        &xs + self.mlp.forward(&h)?.broadcast_mul(&gate_mlp)?
    }
}

// the stem both decoders share: a point's linear projection plus its sinusoid, through one more linear
struct PointStem {
    in_proj: AmpLinear,
    input_layer: AmpLinear,
    channels: usize,
}

impl PointStem {
    fn load(spec: DecoderSpec, vb: &VarBuilder) -> Result<Self> {
        let c = spec.channels;
        Ok(Self {
            in_proj: AmpLinear::load(3, c, true, DType::F32, vb.pp("in_proj"))?,
            input_layer: AmpLinear::load(c, c, true, DType::F32, vb.pp("input_layer"))?,
            channels: c,
        })
    }

    fn forward(&self, points: &Tensor) -> Result<Tensor> {
        let h = (self.in_proj.forward(points)? + point_embedding(points, self.channels)?)?;
        self.input_layer.forward(&h)
    }
}

pub(crate) struct OctreeDecoder {
    stem: PointStem,
    l_embedder: ScalarEmbedder,
    ada_ln: AmpLinear,
    blocks: Vec<Block>,
    out_proj: AmpLinear,
    dtype: DType,
}

impl OctreeDecoder {
    pub(crate) fn load(spec: DecoderSpec, dtype: DType, vb: VarBuilder) -> Result<Self> {
        let c = spec.channels;
        Ok(Self {
            stem: PointStem::load(spec, &vb)?,
            l_embedder: ScalarEmbedder::load(c, LEVEL_MAX_PERIOD, true, vb.pp("l_embedder"))?,
            ada_ln: AmpLinear::load(
                c,
                MOD_CHUNKS * c,
                true,
                DType::F32,
                vb.pp("adaLN_modulation.1"),
            )?,
            blocks: (0..spec.blocks)
                .map(|i| Block::load(spec, true, dtype, vb.pp("blocks").pp(i)))
                .collect::<Result<_>>()?,
            out_proj: AmpLinear::load(c, CHILDREN, true, DType::F32, vb.pp("out_proj"))?,
            dtype,
        })
    }

    /// The logits `[b, l, 8]` of how each cell's points split among its children, for cell centres `[b, l, 3]` in
    /// [0, 1] at octree resolution `resolution` (cells per side), reading `latent` `[b, 8192, 16]`.
    pub(crate) fn forward(
        &self,
        centres: &Tensor,
        resolution: usize,
        latent: &Tensor,
    ) -> Result<Tensor> {
        let b = centres.dim(0)?;
        let h = self.stem.forward(centres)?;
        let level = Tensor::full(resolution as f32, b, centres.device())?;
        let m = self
            .ada_ln
            .forward(&self.l_embedder.forward(&level)?.silu()?)?
            .to_dtype(self.dtype)?;
        let context = latent.to_dtype(self.dtype)?;
        let mut out = h.to_dtype(self.dtype)?;
        for block in &self.blocks {
            out = block.forward(&out, Some(&m), &context)?;
        }
        self.out_proj.forward(&plain_norm(&out, DEFAULT_EPS)?)
    }

    /// `points` anchor positions `[1, points, 3]` in [0, 1]^3 for one latent, descending the octree to `level`:
    /// each cell's points split among its children by systematic resampling, then each lands uniformly in its leaf.
    pub(crate) fn sample(
        &self,
        latent: &Tensor,
        points: usize,
        level: u32,
        rng: &mut StdRng,
    ) -> Result<Tensor> {
        let device = latent.device();
        let mut cells: Vec<([u32; 3], usize)> = vec![([0, 0, 0], points)];
        for l in 1..=level {
            let parent_res = 1u32 << (l - 1);
            let centres: Vec<f32> = cells
                .iter()
                .flat_map(|(c, _)| c.map(|v| (v as f32 + 0.5) / parent_res as f32))
                .collect();
            let centres = Tensor::from_vec(centres, (1, cells.len(), 3), device)?;
            let probs = inference_tensor::nn::ops::softmax_last_dim(&self.forward(
                &centres,
                1 << l,
                latent,
            )?)?
            .squeeze(0)?
            .to_vec2::<f32>()?;
            let mut next = Vec::with_capacity(cells.len() * CHILDREN);
            for ((coord, count), p) in cells.iter().zip(&probs) {
                let split = systematic_split(p, *count, rng);
                for (child, n) in split.into_iter().enumerate() {
                    if n > 0 {
                        // children in x-fastest order: (i, j, k) = (child & 1, child >> 1 & 1, child >> 2)
                        let offset = [child as u32 & 1, (child as u32 >> 1) & 1, child as u32 >> 2];
                        next.push(([0, 1, 2].map(|a| coord[a] * 2 + offset[a]), n));
                    }
                }
            }
            cells = next;
        }
        let res = (1u32 << level) as f32;
        let mut anchors = Vec::with_capacity(points * 3);
        for (coord, count) in &cells {
            for _ in 0..*count {
                anchors.extend(coord.map(|v| (v as f32 + rng.random::<f32>()) / res));
            }
        }
        Tensor::from_vec(anchors, (1, points, 3), device)
    }
}

// `systematic_sample`: `count` items among the bins of `probs` at stratified points `(u0 + i) / count`
fn systematic_split(probs: &[f32], count: usize, rng: &mut StdRng) -> [usize; CHILDREN] {
    let mut out = [0usize; CHILDREN];
    if count == 0 {
        return out;
    }
    let total: f32 = probs.iter().map(|p| p.max(0.0)).sum();
    let mut cdf = [0f32; CHILDREN];
    let mut acc = 0f32;
    for (i, p) in probs.iter().enumerate() {
        acc += if total > 0.0 {
            p.max(0.0) / total
        } else {
            1.0 / CHILDREN as f32
        };
        cdf[i] = acc.min(1.0 - 1e-12);
    }
    let u0 = rng.random::<f32>() / count as f32;
    for i in 0..count {
        let u = (u0 + i as f32 / count as f32).min(1.0 - 1e-12);
        let bin = cdf.iter().position(|&c| c >= u).unwrap_or(CHILDREN - 1);
        out[bin] += 1;
    }
    out
}

/// The Gaussian decoder's settings from `config.json` (`representation_config`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct GaussianSettings {
    pub rotation_lr: f32,
    pub perturb_size: f32,
    pub filter_kernel: f32,
    pub scaling_bias: f32,
    pub opacity_bias: f32,
}

pub(crate) const IDOL_GAUSSIAN_SETTINGS: GaussianSettings = GaussianSettings {
    rotation_lr: 0.1,
    perturb_size: 1.5,
    filter_kernel: 0.0009,
    scaling_bias: 0.004,
    opacity_bias: 0.1,
};

/// Activated Gaussians: the object is z-up inside [-0.5, 0.5]^3.
pub struct Gaussians {
    pub xyz: Vec<[f32; 3]>,
    pub dc: Vec<[f32; 3]>,
    pub scaling: Vec<[f32; 3]>,
    /// unit quaternions, w x y z
    pub rotation: Vec<[f32; 4]>,
    pub opacity: Vec<f32>,
}

// per anchor and Gaussian: offset 3, colour 3, scale 3, rotation 4, opacity 1, offset scale 1
const FIELDS: [usize; 6] = [3, 3, 3, 4, 1, 1];

pub(crate) struct GaussianDecoder {
    stem: PointStem,
    blocks: Vec<Block>,
    out_proj: AmpLinear,
    offset_pattern: Vec<[f32; 3]>,
    base_offset_scale: f32,
    settings: GaussianSettings,
    dtype: DType,
}

impl GaussianDecoder {
    pub(crate) fn load(
        spec: DecoderSpec,
        settings: GaussianSettings,
        dtype: DType,
        vb: VarBuilder,
    ) -> Result<Self> {
        let c = spec.channels;
        let outputs = GAUSSIANS_PER_ANCHOR * FIELDS.iter().sum::<usize>();
        let pattern = vb
            .get((GAUSSIANS_PER_ANCHOR, 3), "points_offset_perturbation")?
            .to_vec2::<f32>()?;
        Ok(Self {
            stem: PointStem::load(spec, &vb)?,
            blocks: (0..spec.blocks)
                .map(|i| Block::load(spec, false, dtype, vb.pp("blocks").pp(i)))
                .collect::<Result<_>>()?,
            out_proj: AmpLinear::load(c, outputs, true, DType::F32, vb.pp("out_proj"))?,
            offset_pattern: pattern.into_iter().map(|r| [r[0], r[1], r[2]]).collect(),
            base_offset_scale: vb.get((), "base_offset_scale")?.to_scalar::<f32>()?,
            settings,
            dtype,
        })
    }

    /// Per anchor `[b, l, 3]` its raw outputs `[b, l, 480]`, reading `latent` `[b, 8192, 16]`.
    pub(crate) fn forward(&self, anchors: &Tensor, latent: &Tensor) -> Result<Tensor> {
        let context = latent.to_dtype(self.dtype)?;
        let mut out = self.stem.forward(anchors)?.to_dtype(self.dtype)?;
        for block in &self.blocks {
            out = block.forward(&out, None, &context)?;
        }
        self.out_proj.forward(&plain_norm(&out, DEFAULT_EPS)?)
    }

    /// The Gaussians of anchors `[1, l, 3]` from their raw outputs `[1, l, 480]`.
    pub(crate) fn gaussians(&self, anchors: &Tensor, features: &Tensor) -> Result<Gaussians> {
        let anchors = anchors.squeeze(0)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let features = features
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        let s = self.settings;
        let g = GAUSSIANS_PER_ANCHOR;
        let mut starts = [0usize; 6];
        for i in 1..6 {
            starts[i] = starts[i - 1] + g * FIELDS[i - 1];
        }
        let softplus = |x: f32| if x > 20.0 { x } else { x.exp().ln_1p() };
        let scaling_offset = s.scaling_bias + (-(-s.scaling_bias).exp_m1()).ln();
        let opacity_offset = (s.opacity_bias / (1.0 - s.opacity_bias)).ln();
        let total = anchors.len() * g;
        let mut out = Gaussians {
            xyz: Vec::with_capacity(total),
            dc: Vec::with_capacity(total),
            scaling: Vec::with_capacity(total),
            rotation: Vec::with_capacity(total),
            opacity: Vec::with_capacity(total),
        };
        for (anchor, f) in anchors.iter().zip(&features) {
            for i in 0..g {
                let field = |k: usize, j: usize| f[starts[k] + i * FIELDS[k] + j];
                let offset_scale = softplus(field(5, 0) + self.base_offset_scale);
                out.xyz.push([0, 1, 2].map(|a| {
                    let o = (field(0, a) + self.offset_pattern[i][a]).tanh()
                        * 0.5
                        * s.perturb_size
                        * offset_scale;
                    anchor[a] + o - 0.5
                }));
                out.dc.push([0, 1, 2].map(|a| field(1, a)));
                out.scaling.push([0, 1, 2].map(|a| {
                    let v = softplus(field(2, a) + scaling_offset);
                    (v * v + s.filter_kernel * s.filter_kernel).sqrt()
                }));
                let q = [0, 1, 2, 3]
                    .map(|a| field(3, a) * s.rotation_lr + if a == 0 { 1.0 } else { 0.0 });
                let norm = q.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
                out.rotation.push(q.map(|v| v / norm));
                out.opacity
                    .push(1.0 / (1.0 + (-(field(4, 0) + opacity_offset)).exp()));
            }
        }
        Ok(out)
    }
}

/// `count` anchors' worth of Gaussians in batches of `batch` anchors, so the decoder's activations stay bounded.
pub(crate) fn decode_in_batches(
    decoder: &GaussianDecoder,
    anchors: &Tensor,
    latent: &Tensor,
    batch: usize,
) -> Result<Gaussians> {
    let l = anchors.dim(1)?;
    let mut all: Option<Gaussians> = None;
    for start in (0..l).step_by(batch) {
        let part = anchors.narrow(1, start, batch.min(l - start))?;
        let g = decoder.gaussians(&part, &decoder.forward(&part, latent)?)?;
        match &mut all {
            None => all = Some(g),
            Some(a) => {
                a.xyz.extend(g.xyz);
                a.dc.extend(g.dc);
                a.scaling.extend(g.scaling);
                a.rotation.extend(g.rotation);
                a.opacity.extend(g.opacity);
            }
        }
    }
    all.ok_or_else(|| inference_tensor::Error::msg("no anchors to decode"))
}
