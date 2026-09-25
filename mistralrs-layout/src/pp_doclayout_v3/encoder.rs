use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{Activation, LayerNorm, Linear, VarBuilder};

use super::config::PPDocLayoutV3Config;
use crate::layers::{ConvNorm, ConvNormSpec, RTDETR_CONV};

const CSP_BLOCKS: usize = 3;

/// Multi-head self-attention where the position embedding is added to queries and keys only.
pub struct SelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    pub fn new(hidden: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q: candle_nn::linear(hidden, hidden, vb.pp("q_proj"))?,
            k: candle_nn::linear(hidden, hidden, vb.pp("k_proj"))?,
            v: candle_nn::linear(hidden, hidden, vb.pp("v_proj"))?,
            o: candle_nn::linear(hidden, hidden, vb.pp("out_proj"))?,
            heads,
            head_dim: hidden / heads,
        })
    }

    pub fn forward(&self, xs: &Tensor, pos: Option<&Tensor>) -> Result<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let qk_in = match pos {
            Some(p) => xs.broadcast_add(p)?,
            None => xs.clone(),
        };
        let split = |t: Tensor| {
            t.reshape((b, n, self.heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = split(self.q.forward(&qk_in)?)?;
        let k = split(self.k.forward(&qk_in)?)?;
        let v = split(self.v.forward(xs)?)?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = (q.matmul(&k.t()?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, n, c))?;
        self.o.forward(&out)
    }
}

pub struct Mlp {
    fc1: Linear,
    fc2: Linear,
    act: Activation,
}

impl Mlp {
    pub fn new(hidden: usize, ffn: usize, act: Activation, vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: candle_nn::linear(hidden, ffn, vb.pp("fc1"))?,
            fc2: candle_nn::linear(ffn, hidden, vb.pp("fc2"))?,
            act,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.act.forward(&self.fc1.forward(xs)?)?)
    }
}

/// Post-norm transformer layer (`normalize_before=False` in every released config).
struct EncoderLayer {
    self_attn: SelfAttention,
    self_attn_norm: LayerNorm,
    mlp: Mlp,
    final_norm: LayerNorm,
}

impl EncoderLayer {
    fn new(cfg: &PPDocLayoutV3Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.encoder_hidden_dim;
        Ok(Self {
            self_attn: SelfAttention::new(h, cfg.encoder_attention_heads, vb.pp("self_attn"))?,
            self_attn_norm: candle_nn::layer_norm(
                h,
                cfg.layer_norm_eps,
                vb.pp("self_attn_layer_norm"),
            )?,
            mlp: Mlp::new(h, cfg.encoder_ffn_dim, cfg.encoder_activation_function, &vb)?,
            final_norm: candle_nn::layer_norm(h, cfg.layer_norm_eps, vb.pp("final_layer_norm"))?,
        })
    }

    fn forward(&self, xs: &Tensor, pos: &Tensor) -> Result<Tensor> {
        let xs = self
            .self_attn_norm
            .forward(&(xs + self.self_attn.forward(xs, Some(pos))?)?)?;
        self.final_norm.forward(&(&xs + self.mlp.forward(&xs)?)?)
    }
}

/// `[sin_h | cos_h | sin_w | cos_w]` per position, row-major; computed in f64 like the reference.
pub fn sine_pos_embed_2d(
    h: usize,
    w: usize,
    dim: usize,
    temperature: f64,
    dev: &Device,
) -> Result<Tensor> {
    let pos_dim = dim / 4;
    let omega: Vec<f64> = (0..pos_dim)
        .map(|i| 1.0 / temperature.powf(i as f64 / pos_dim as f64))
        .collect();
    let mut data = Vec::with_capacity(h * w * dim);
    for y in 0..h {
        for x in 0..w {
            let (y, x) = (y as f64, x as f64);
            data.extend(omega.iter().map(|o| (y * o).sin() as f32));
            data.extend(omega.iter().map(|o| (y * o).cos() as f32));
            data.extend(omega.iter().map(|o| (x * o).sin() as f32));
            data.extend(omega.iter().map(|o| (x * o).cos() as f32));
        }
    }
    Tensor::from_vec(data, (1, h * w, dim), dev)
}

struct Aifi {
    layers: Vec<EncoderLayer>,
    pos: Tensor,
}

impl Aifi {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = xs.dims4()?;
        let mut hs = xs.flatten_from(2)?.transpose(1, 2)?;
        for l in &self.layers {
            hs = l.forward(&hs, &self.pos)?;
        }
        hs.transpose(1, 2)?.reshape((b, c, h, w))
    }
}

struct RepVggBlock {
    conv1: ConvNorm,
    conv2: ConvNorm,
    act: Activation,
}

impl Module for RepVggBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.act
            .forward(&(self.conv1.forward(xs)? + self.conv2.forward(xs)?)?)
    }
}

struct CspRepLayer {
    conv1: ConvNorm,
    conv2: ConvNorm,
    bottlenecks: Vec<RepVggBlock>,
}

impl CspRepLayer {
    fn new(cfg: &PPDocLayoutV3Config, vb: VarBuilder) -> Result<Self> {
        let out_c = cfg.encoder_hidden_dim;
        let in_c = out_c * 2;
        let hidden = (out_c as f64 * cfg.hidden_expansion) as usize;
        if hidden != out_c {
            candle_core::bail!("hidden_expansion != 1.0 (conv3) is not supported");
        }
        let act = cfg.activation_function;
        let bottlenecks = (0..CSP_BLOCKS)
            .map(|i| {
                let vb = vb.pp("bottlenecks").pp(i);
                Ok(RepVggBlock {
                    conv1: ConvNormSpec::new(hidden, hidden, 3)
                        .names(RTDETR_CONV)
                        .eps(cfg.batch_norm_eps)
                        .load(vb.pp("conv1"))?,
                    conv2: ConvNormSpec::new(hidden, hidden, 1)
                        .names(RTDETR_CONV)
                        .eps(cfg.batch_norm_eps)
                        .load(vb.pp("conv2"))?,
                    act,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv1: ConvNormSpec::new(in_c, hidden, 1)
                .act(act)
                .names(RTDETR_CONV)
                .eps(cfg.batch_norm_eps)
                .load(vb.pp("conv1"))?,
            conv2: ConvNormSpec::new(in_c, hidden, 1)
                .act(act)
                .names(RTDETR_CONV)
                .eps(cfg.batch_norm_eps)
                .load(vb.pp("conv2"))?,
            bottlenecks,
        })
    }
}

impl Module for CspRepLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h1 = self.conv1.forward(xs)?;
        for b in &self.bottlenecks {
            h1 = b.forward(&h1)?;
        }
        h1 + self.conv2.forward(xs)?
    }
}

enum ScaleLayer {
    Conv(ConvNorm),
    Up2x,
}

/// Mask-feature FPN: per-level conv(+2x bilinear) heads summed at the finest stride.
struct MaskFeatFpn {
    scale_heads: Vec<Vec<ScaleLayer>>,
    output_conv: ConvNorm,
}

impl MaskFeatFpn {
    fn new(
        in_c: usize,
        strides: &[usize],
        feat_c: usize,
        out_c: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        if strides.windows(2).any(|w| w[0] > w[1]) {
            candle_core::bail!("mask feature FPN expects ascending feat_strides");
        }
        let base = strides[0];
        let mut scale_heads = Vec::with_capacity(strides.len());
        for (i, &s) in strides.iter().enumerate() {
            let head_len = ((s as f64).log2() - (base as f64).log2()).max(1.0) as usize;
            let vbh = vb.pp("scale_heads").pp(i).pp("layers");
            let mut layers = Vec::new();
            for k in 0..head_len {
                let ic = if k == 0 { in_c } else { feat_c };
                let idx = layers.len();
                layers.push(ScaleLayer::Conv(
                    ConvNormSpec::new(ic, feat_c, 3)
                        .act(Activation::Silu)
                        .load(vbh.pp(idx))?,
                ));
                if s != base {
                    layers.push(ScaleLayer::Up2x);
                }
            }
            scale_heads.push(layers);
        }
        Ok(Self {
            scale_heads,
            output_conv: ConvNormSpec::new(feat_c, out_c, 3)
                .act(Activation::Silu)
                .load(vb.pp("output_conv"))?,
        })
    }

    fn head(&self, i: usize, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for l in &self.scale_heads[i] {
            xs = match l {
                ScaleLayer::Conv(c) => c.forward(&xs)?,
                ScaleLayer::Up2x => {
                    let (_, _, h, w) = xs.dims4()?;
                    xs.upsample_bilinear2d(h * 2, w * 2, false)?
                }
            };
        }
        Ok(xs)
    }

    fn forward(&self, feats: &[Tensor]) -> Result<Tensor> {
        let mut out = self.head(0, &feats[0])?;
        let (_, _, h, w) = out.dims4()?;
        for (i, f) in feats.iter().enumerate().skip(1) {
            let mut y = self.head(i, f)?;
            if y.dims4()?.2 != h || y.dims4()?.3 != w {
                y = y.upsample_bilinear2d(h, w, false)?;
            }
            out = (out + y)?;
        }
        self.output_conv.forward(&out)
    }
}

pub struct EncoderOutput {
    /// PAN outputs, finest stride first.
    pub feats: Vec<Tensor>,
    /// `(b, num_prototypes, H/4, W/4)`
    pub mask_feat: Tensor,
}

pub struct HybridEncoder {
    aifi: Vec<(usize, Aifi)>,
    lateral_convs: Vec<ConvNorm>,
    fpn_blocks: Vec<CspRepLayer>,
    downsample_convs: Vec<ConvNorm>,
    pan_blocks: Vec<CspRepLayer>,
    mask_feature_head: MaskFeatFpn,
    mask_lateral: ConvNorm,
    mask_out_conv: ConvNorm,
    mask_out_proj: candle_nn::Conv2d,
}

impl HybridEncoder {
    /// `aifi_hw` is the spatial size of each `encode_proj_layers` level, used to precompute its position embedding.
    pub fn new(
        cfg: &PPDocLayoutV3Config,
        aifi_hw: &[(usize, usize)],
        vb: VarBuilder,
    ) -> Result<Self> {
        let h = cfg.encoder_hidden_dim;
        let act = cfg.activation_function;
        let aifi = cfg
            .encode_proj_layers
            .iter()
            .zip(aifi_hw)
            .enumerate()
            .map(|(i, (&lvl, &(ah, aw)))| {
                let vb = vb.pp("encoder").pp(i).pp("layers");
                let layers = (0..cfg.encoder_layers)
                    .map(|j| EncoderLayer::new(cfg, vb.pp(j)))
                    .collect::<Result<Vec<_>>>()?;
                let pos =
                    sine_pos_embed_2d(ah, aw, h, cfg.positional_encoding_temperature, vb.device())?;
                Ok((lvl, Aifi { layers, pos }))
            })
            .collect::<Result<Vec<_>>>()?;
        let n = cfg.encoder_in_channels.len() - 1;
        let mut lateral_convs = Vec::with_capacity(n);
        let mut fpn_blocks = Vec::with_capacity(n);
        let mut downsample_convs = Vec::with_capacity(n);
        let mut pan_blocks = Vec::with_capacity(n);
        for i in 0..n {
            lateral_convs.push(
                ConvNormSpec::new(h, h, 1)
                    .act(act)
                    .names(RTDETR_CONV)
                    .eps(cfg.batch_norm_eps)
                    .load(vb.pp("lateral_convs").pp(i))?,
            );
            fpn_blocks.push(CspRepLayer::new(cfg, vb.pp("fpn_blocks").pp(i))?);
            downsample_convs.push(
                ConvNormSpec::new(h, h, 3)
                    .stride(2)
                    .act(act)
                    .names(RTDETR_CONV)
                    .eps(cfg.batch_norm_eps)
                    .load(vb.pp("downsample_convs").pp(i))?,
            );
            pan_blocks.push(CspRepLayer::new(cfg, vb.pp("pan_blocks").pp(i))?);
        }
        let [feat_c, mask_c] = cfg.mask_feature_channels[..] else {
            candle_core::bail!("mask_feature_channels must have two entries");
        };
        let vbo = vb.pp("encoder_mask_output");
        Ok(Self {
            aifi,
            lateral_convs,
            fpn_blocks,
            downsample_convs,
            pan_blocks,
            mask_feature_head: MaskFeatFpn::new(
                h,
                &cfg.feat_strides,
                feat_c,
                mask_c,
                vb.pp("mask_feature_head"),
            )?,
            mask_lateral: ConvNormSpec::new(cfg.x4_feat_dim, mask_c, 3)
                .act(Activation::Silu)
                .load(vb.pp("encoder_mask_lateral"))?,
            mask_out_conv: ConvNormSpec::new(mask_c, mask_c, 3)
                .act(Activation::Silu)
                .load(vbo.pp("base_conv"))?,
            mask_out_proj: candle_nn::conv2d(
                mask_c,
                cfg.num_prototypes,
                1,
                Default::default(),
                vbo.pp("conv"),
            )?,
        })
    }

    pub fn forward(&self, mut feats: Vec<Tensor>, x4_feat: &Tensor) -> Result<EncoderOutput> {
        for (lvl, aifi) in &self.aifi {
            feats[*lvl] = aifi.forward(&feats[*lvl])?;
        }
        let n = self.lateral_convs.len();

        // top-down: `fpn` ends up coarsest first, like the reference before its reverse()
        let mut fpn = vec![feats[n].clone()];
        for i in 0..n {
            let top = self.lateral_convs[i].forward(fpn.last().unwrap())?;
            *fpn.last_mut().unwrap() = top.clone();
            let (_, _, th, tw) = top.dims4()?;
            let up = top.upsample_nearest2d(th * 2, tw * 2)?;
            let fused = Tensor::cat(&[&up, &feats[n - i - 1]], 1)?;
            fpn.push(self.fpn_blocks[i].forward(&fused)?);
        }
        fpn.reverse();

        let mut pan = vec![fpn[0].clone()];
        for i in 0..n {
            let down = self.downsample_convs[i].forward(pan.last().unwrap())?;
            let fused = Tensor::cat(&[&down, &fpn[i + 1]], 1)?;
            pan.push(self.pan_blocks[i].forward(&fused)?);
        }

        let mask_feat = self.mask_feature_head.forward(&pan)?;
        let (_, _, mh, mw) = mask_feat.dims4()?;
        let mask_feat = mask_feat.upsample_bilinear2d(mh * 2, mw * 2, false)?;
        let mask_feat = (mask_feat + self.mask_lateral.forward(x4_feat)?)?;
        let mask_feat = self
            .mask_out_proj
            .forward(&self.mask_out_conv.forward(&mask_feat)?)?;
        Ok(EncoderOutput {
            feats: pan,
            mask_feat,
        })
    }
}
