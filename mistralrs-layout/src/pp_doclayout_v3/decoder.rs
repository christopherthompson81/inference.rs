use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::{LayerNorm, Linear, VarBuilder};

use super::config::PPDocLayoutV3Config;
use super::encoder::{Mlp, SelfAttention};

/// Flattened multi-level feature memory layout.
#[derive(Debug, Clone)]
pub struct LevelGeom {
    pub shapes: Vec<(usize, usize)>,
    pub starts: Vec<usize>,
    pub total: usize,
}

impl LevelGeom {
    pub fn new(shapes: Vec<(usize, usize)>) -> Self {
        let mut starts = Vec::with_capacity(shapes.len());
        let mut total = 0;
        for (h, w) in &shapes {
            starts.push(total);
            total += h * w;
        }
        Self {
            shapes,
            starts,
            total,
        }
    }
}

/// Per-forward invariants shared by every decoder layer.
pub struct DecodeCtx<'a> {
    /// `(b, S, d_model)` flattened encoder memory.
    pub memory: &'a Tensor,
    pub geom: &'a LevelGeom,
}

/// Deformable DETR multi-scale deformable attention with 4-d (box) reference points.
struct MsDeformAttn {
    sampling_offsets: Linear,
    attention_weights: Linear,
    value_proj: Linear,
    output_proj: Linear,
    heads: usize,
    levels: usize,
    points: usize,
}

impl MsDeformAttn {
    fn new(cfg: &PPDocLayoutV3Config, vb: VarBuilder) -> Result<Self> {
        let (d, h, l, p) = (
            cfg.d_model,
            cfg.decoder_attention_heads,
            cfg.num_feature_levels,
            cfg.decoder_n_points,
        );
        Ok(Self {
            sampling_offsets: candle_nn::linear(d, h * l * p * 2, vb.pp("sampling_offsets"))?,
            attention_weights: candle_nn::linear(d, h * l * p, vb.pp("attention_weights"))?,
            value_proj: candle_nn::linear(d, d, vb.pp("value_proj"))?,
            output_proj: candle_nn::linear(d, d, vb.pp("output_proj"))?,
            heads: h,
            levels: l,
            points: p,
        })
    }

    /// `query`: `(b, q, d)` with position embedding already added; `ref_boxes`: `(b, q, 4)` cxcywh in [0, 1].
    fn forward(&self, query: &Tensor, ref_boxes: &Tensor, ctx: &DecodeCtx) -> Result<Tensor> {
        let (b, q, d) = query.dims3()?;
        let (h, l, p) = (self.heads, self.levels, self.points);
        let hd = d / h;
        let s = ctx.geom.total;

        let value = self
            .value_proj
            .forward(ctx.memory)?
            .reshape((b, s, h, hd))?;

        let offsets = self
            .sampling_offsets
            .forward(query)?
            .reshape((b, q, h, l, p, 2))?;
        let attn = self
            .attention_weights
            .forward(query)?
            .reshape((b, q, h, l * p))?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?.reshape((b, q, h, l, p))?;

        let ref_xy = ref_boxes
            .narrow(D::Minus1, 0, 2)?
            .reshape((b, q, 1, 1, 1, 2))?;
        let ref_wh = ref_boxes
            .narrow(D::Minus1, 2, 2)?
            .reshape((b, q, 1, 1, 1, 2))?;
        let step = (ref_wh * (0.5 / p as f64))?;
        let loc = ref_xy.broadcast_add(&offsets.broadcast_mul(&step)?)?;
        let out = if query.device().is_metal() {
            sample_ops(&value, &loc, &attn, ctx.geom)?
        } else {
            crate::msda::ms_deform_attn(&value, &loc, &attn, &ctx.geom.shapes)?
        };
        self.output_proj.forward(&out)
    }
}

/// Tensor-op sampler for backends without the fused kernel.
fn sample_ops(value: &Tensor, loc: &Tensor, attn: &Tensor, geom: &LevelGeom) -> Result<Tensor> {
    let (b, s, h, hd) = value.dims4()?;
    let &[_, q, _, l, p, _] = loc.dims() else {
        candle_core::bail!("sampling locations must be rank 6");
    };
    let d = h * hd;
    let bh_offset = Tensor::arange(0u32, (b * h) as u32, value.device())?
        .affine(s as f64, 0.)?
        .reshape((b * h, 1))?;
    let value = value
        .transpose(1, 2)?
        .contiguous()?
        .reshape((b * h * s, hd))?;
    // (b, h, q, l, p, 2)
    let loc = loc.permute((0, 2, 1, 3, 4, 5))?;
    let attn = attn.permute((0, 2, 1, 3, 4))?;

    let mut idx_l = Vec::with_capacity(l);
    let mut w_l = Vec::with_capacity(l);
    for (lvl, &(lh, lw)) in geom.shapes.iter().enumerate() {
        let loc_l = loc.narrow(3, lvl, 1)?;
        // grid_sample(align_corners=False): pixel = loc * size - 0.5
        let x = loc_l.narrow(5, 0, 1)?.squeeze(5)?.affine(lw as f64, -0.5)?;
        let y = loc_l.narrow(5, 1, 1)?.squeeze(5)?.affine(lh as f64, -0.5)?;
        let x0 = x.floor()?;
        let y0 = y.floor()?;
        let fx = (&x - &x0)?;
        let fy = (&y - &y0)?;
        let mut idx_c = Vec::with_capacity(4);
        let mut w_c = Vec::with_capacity(4);
        for (dx, dy) in [(0., 0.), (1., 0.), (0., 1.), (1., 1.)] {
            let xc = (&x0 + dx)?;
            let yc = (&y0 + dy)?;
            let wx = if dx == 0. {
                fx.affine(-1., 1.)?
            } else {
                fx.clone()
            };
            let wy = if dy == 0. {
                fy.affine(-1., 1.)?
            } else {
                fy.clone()
            };
            // zero padding: out-of-bounds taps get zero weight and a clamped (harmless) index
            let valid = xc
                .ge(0.)?
                .mul(&xc.le((lw - 1) as f64)?)?
                .mul(&yc.ge(0.)?)?
                .mul(&yc.le((lh - 1) as f64)?)?
                .to_dtype(DType::F32)?;
            let xi = xc.clamp(0., (lw - 1) as f64)?;
            let yi = yc.clamp(0., (lh - 1) as f64)?;
            idx_c.push((yi.affine(lw as f64, geom.starts[lvl] as f64)? + xi)?);
            w_c.push(wx.mul(&wy)?.mul(&valid)?);
        }
        idx_l.push(Tensor::stack(&idx_c, D::Minus1)?);
        w_l.push(Tensor::stack(&w_c, D::Minus1)?);
    }
    // (b, h, q, l, p, 4)
    let idx = Tensor::cat(&idx_l, 3)?;
    let w = Tensor::cat(&w_l, 3)?.broadcast_mul(&attn.unsqueeze(D::Minus1)?)?;
    let k = l * p * 4;

    // per-head indices are < S and exact in f32; the cross-head offset is added in u32
    let idx = idx
        .reshape((b * h, q * k))?
        .to_dtype(DType::U32)?
        .broadcast_add(&bh_offset)?
        .flatten_all()?;
    let sampled = value.index_select(&idx, 0)?.reshape((b * h * q, k, hd))?;
    let w = w.reshape((b * h * q, 1, k))?;
    w.matmul(&sampled)?
        .reshape((b, h, q, hd))?
        .transpose(1, 2)?
        .reshape((b, q, d))
}

pub struct DecoderLayer {
    self_attn: SelfAttention,
    self_attn_norm: LayerNorm,
    encoder_attn: MsDeformAttn,
    encoder_attn_norm: LayerNorm,
    mlp: Mlp,
    final_norm: LayerNorm,
}

impl DecoderLayer {
    pub fn new(cfg: &PPDocLayoutV3Config, vb: VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        let eps = cfg.layer_norm_eps;
        Ok(Self {
            self_attn: SelfAttention::new(d, cfg.decoder_attention_heads, vb.pp("self_attn"))?,
            self_attn_norm: candle_nn::layer_norm(d, eps, vb.pp("self_attn_layer_norm"))?,
            encoder_attn: MsDeformAttn::new(cfg, vb.pp("encoder_attn"))?,
            encoder_attn_norm: candle_nn::layer_norm(d, eps, vb.pp("encoder_attn_layer_norm"))?,
            mlp: Mlp::new(d, cfg.decoder_ffn_dim, cfg.decoder_activation_function, &vb)?,
            final_norm: candle_nn::layer_norm(d, eps, vb.pp("final_layer_norm"))?,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        pos: &Tensor,
        ref_boxes: &Tensor,
        ctx: &DecodeCtx,
    ) -> Result<Tensor> {
        let xs = self
            .self_attn_norm
            .forward(&(xs + self.self_attn.forward(xs, Some(pos))?)?)?;
        let cross = self.encoder_attn.forward(&(&xs + pos)?, ref_boxes, ctx)?;
        let xs = self.encoder_attn_norm.forward(&(xs + cross)?)?;
        self.final_norm.forward(&(&xs + self.mlp.forward(&xs)?)?)
    }
}

const INV_SIGMOID_EPS: f64 = 1e-5;

pub fn inverse_sigmoid(xs: &Tensor) -> Result<Tensor> {
    let xs = xs.clamp(0., 1.)?;
    let x1 = xs.clamp(INV_SIGMOID_EPS, 1.)?;
    let x2 = xs.affine(-1., 1.)?.clamp(INV_SIGMOID_EPS, 1.)?;
    (x1 / x2)?.log()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn fused_sampler_matches_tensor_ops() -> Result<()> {
        let cpu = Device::Cpu;
        let geom = LevelGeom::new(vec![(4, 5), (3, 3)]);
        let (b, q, h, d, l, p) = (2, 3, 2, 4, 2, 3);
        let value = Tensor::randn(0f32, 1., (b, geom.total, h, d), &cpu)?;
        // outside [0, 1] on purpose so zero-padded taps are exercised
        let loc = Tensor::rand(-0.2f32, 1.2, (b, q, h, l, p, 2), &cpu)?;
        let attn = Tensor::rand(0f32, 1., (b, q, h, l, p), &cpu)?;
        let want = sample_ops(&value, &loc, &attn, &geom)?;
        let devs = std::iter::once(Ok(cpu.clone()))
            .chain(cfg!(feature = "cuda").then(|| Device::new_cuda(0)))
            .collect::<Result<Vec<_>>>()?;
        for dev in devs {
            let got = crate::msda::ms_deform_attn(
                &value.to_device(&dev)?,
                &loc.to_device(&dev)?,
                &attn.to_device(&dev)?,
                &geom.shapes,
            )?
            .to_device(&cpu)?;
            let err = (got - &want)?
                .abs()?
                .flatten_all()?
                .max(D::Minus1)?
                .to_scalar::<f32>()?;
            assert!(err < 1e-5, "{dev:?} err={err}");
        }
        Ok(())
    }
}
