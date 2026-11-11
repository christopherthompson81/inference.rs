//! The 12 Hz codec's decoder: split RVQ dequantization, a sliding-window transformer, ConvNeXt upsampling and a
//! SnakeBeta vocoder to 24 kHz.

use inference_nn::attention::{FlashParams, SdpaParams};
use inference_nn::layers::masker::{CausalMaskConfig, CausalMasker, NotACache};
use inference_nn::layers::{RmsNorm, RotaryEmbedding, Sdpa, apply_rotary_preselected_qk};
use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{Embedding, LayerNorm, Linear, Module};
use inference_tensor::{D, DType, Device, Result, Tensor};

use super::config::CodecDecoderConfig;
use crate::layers::Activation;

const CLUSTER_USAGE_EPS: f64 = 1e-5;
const SNAKE_EPS: f64 = 1e-9;
const CONVNEXT_KERNEL: usize = 7;
const CONVNEXT_EXPANSION: usize = 4;
const CONVNEXT_LN_EPS: f64 = 1e-6;
const RESIDUAL_KERNEL: usize = 7;
const RESIDUAL_DILATIONS: [usize; 3] = [1, 3, 9];
const PRE_CONV_KERNEL: usize = 3;
const DECODER_KERNEL: usize = 7;
/// Frames decoded at once, and the frames of left context each later chunk is decoded with (the reference's).
pub const CHUNK_FRAMES: usize = 300;
pub const CONTEXT_FRAMES: usize = 25;

/// A conv padded on the left only, so each output sees its input's past.
struct CausalConv {
    weight: Tensor,
    bias: Tensor,
    stride: usize,
    dilation: usize,
    groups: usize,
    // the dilated kernel's span less the stride
    padding: usize,
    span: usize,
}

impl CausalConv {
    fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        dilation: usize,
        groups: usize,
        vb: &ShardedVarBuilder,
    ) -> Result<Self> {
        let stride = 1;
        let span = (kernel - 1) * dilation + 1;
        Ok(Self {
            weight: vb.get((out_ch, in_ch / groups, kernel), "weight")?,
            bias: vb.get(out_ch, "bias")?,
            stride,
            dilation,
            groups,
            padding: span - stride,
            span,
        })
    }
}

impl Module for CausalConv {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let len = xs.dim(D::Minus1)?;
        // the reference's right padding to a whole last frame
        let frames = (len + self.padding - self.span).div_ceil(self.stride) + 1;
        let ideal = (frames - 1) * self.stride + self.span - self.padding;
        let xs = xs.pad_with_zeros(D::Minus1, self.padding, ideal - len)?;
        let ys = xs.conv1d(&self.weight, 0, self.stride, self.dilation, self.groups)?;
        ys.broadcast_add(&self.bias.reshape((1, (), 1))?)
    }
}

/// A transposed conv trimmed on the right, so it stays causal.
struct CausalTransConv {
    weight: Tensor,
    bias: Tensor,
    stride: usize,
    trim: usize,
}

impl CausalTransConv {
    fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        vb: &ShardedVarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            weight: vb.get((in_ch, out_ch, kernel), "weight")?,
            bias: vb.get(out_ch, "bias")?,
            stride,
            trim: kernel - stride,
        })
    }
}

impl Module for CausalTransConv {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = xs.conv_transpose1d(&self.weight, 0, 0, self.stride, 1, 1)?;
        let ys = ys.broadcast_add(&self.bias.reshape((1, (), 1))?)?;
        let len = ys.dim(D::Minus1)?;
        ys.narrow(D::Minus1, 0, len - self.trim)
    }
}

/// `x + sin(x * e^alpha)^2 / e^beta`, per channel.
struct SnakeBeta {
    alpha: Tensor,
    inv_beta: Tensor,
}

impl SnakeBeta {
    fn new(channels: usize, vb: &ShardedVarBuilder) -> Result<Self> {
        let alpha = vb
            .get(channels, "alpha")?
            .exp()?
            .reshape((1, channels, 1))?;
        let beta = vb.get(channels, "beta")?.exp()?;
        let inv_beta = ((beta + SNAKE_EPS)?.recip()?).reshape((1, channels, 1))?;
        Ok(Self { alpha, inv_beta })
    }
}

impl Module for SnakeBeta {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let wave = xs.broadcast_mul(&self.alpha)?.sin()?.sqr()?;
        xs + wave.broadcast_mul(&self.inv_beta)?
    }
}

struct ConvNeXt {
    dwconv: CausalConv,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Tensor,
}

impl ConvNeXt {
    fn new(dim: usize, vb: &ShardedVarBuilder) -> Result<Self> {
        let hidden = CONVNEXT_EXPANSION * dim;
        let norm = vb.pp("norm");
        Ok(Self {
            dwconv: CausalConv::new(
                dim,
                dim,
                CONVNEXT_KERNEL,
                1,
                dim,
                &vb.pp("dwconv").pp("conv"),
            )?,
            norm: LayerNorm::new(
                norm.get(dim, "weight")?,
                norm.get(dim, "bias")?,
                CONVNEXT_LN_EPS,
            ),
            pwconv1: linear(dim, hidden, true, &vb.pp("pwconv1"))?,
            pwconv2: linear(hidden, dim, true, &vb.pp("pwconv2"))?,
            gamma: vb.get(dim, "gamma")?,
        })
    }
}

impl Module for ConvNeXt {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = self.dwconv.forward(xs)?.transpose(1, 2)?;
        let ys = self.pwconv1.forward(&self.norm.forward(&ys)?)?.gelu_erf()?;
        let ys = self.pwconv2.forward(&ys)?.broadcast_mul(&self.gamma)?;
        xs + ys.transpose(1, 2)?
    }
}

struct ResidualUnit {
    act1: SnakeBeta,
    conv1: CausalConv,
    act2: SnakeBeta,
    conv2: CausalConv,
}

impl ResidualUnit {
    fn new(dim: usize, dilation: usize, vb: &ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            act1: SnakeBeta::new(dim, &vb.pp("act1"))?,
            conv1: CausalConv::new(
                dim,
                dim,
                RESIDUAL_KERNEL,
                dilation,
                1,
                &vb.pp("conv1").pp("conv"),
            )?,
            act2: SnakeBeta::new(dim, &vb.pp("act2"))?,
            conv2: CausalConv::new(dim, dim, 1, 1, 1, &vb.pp("conv2").pp("conv"))?,
        })
    }
}

impl Module for ResidualUnit {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = self.conv1.forward(&self.act1.forward(xs)?)?;
        let ys = self.conv2.forward(&self.act2.forward(&ys)?)?;
        ys + xs
    }
}

struct DecoderBlock {
    act: SnakeBeta,
    upsample: CausalTransConv,
    units: Vec<ResidualUnit>,
}

impl DecoderBlock {
    fn new(in_dim: usize, out_dim: usize, rate: usize, vb: &ShardedVarBuilder) -> Result<Self> {
        let vb = vb.pp("block");
        let units = RESIDUAL_DILATIONS
            .iter()
            .enumerate()
            .map(|(i, &dilation)| ResidualUnit::new(out_dim, dilation, &vb.pp(i + 2)))
            .collect::<Result<_>>()?;
        Ok(Self {
            act: SnakeBeta::new(in_dim, &vb.pp(0))?,
            upsample: CausalTransConv::new(in_dim, out_dim, 2 * rate, rate, &vb.pp(1).pp("conv"))?,
            units,
        })
    }
}

impl Module for DecoderBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.upsample.forward(&self.act.forward(xs)?)?;
        for unit in &self.units {
            xs = unit.forward(&xs)?;
        }
        Ok(xs)
    }
}

fn linear(in_dim: usize, out_dim: usize, bias: bool, vb: &ShardedVarBuilder) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    let bias = bias.then(|| vb.get(out_dim, "bias")).transpose()?;
    Ok(Linear::new(weight, bias))
}

struct TransformerLayer {
    input_norm: RmsNorm,
    post_norm: RmsNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
    attn_scale: Tensor,
    mlp_scale: Tensor,
}

struct Transformer {
    input_proj: Linear,
    output_proj: Linear,
    layers: Vec<TransformerLayer>,
    norm: RmsNorm,
    rotary: RotaryEmbedding,
    heads: usize,
    head_dim: usize,
    window: usize,
    act: Activation,
}

impl Transformer {
    fn new(cfg: &CodecDecoderConfig, vb: &ShardedVarBuilder, device: &Device) -> Result<Self> {
        let (hidden, inner) = (cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim);
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let vb = vb.pp("layers").pp(i);
                let attn = vb.pp("self_attn");
                let mlp = vb.pp("mlp");
                Ok(TransformerLayer {
                    input_norm: RmsNorm::new(hidden, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
                    post_norm: RmsNorm::new(
                        hidden,
                        cfg.rms_norm_eps,
                        vb.pp("post_attention_layernorm"),
                    )?,
                    q: linear(hidden, inner, false, &attn.pp("q_proj"))?,
                    k: linear(hidden, kv, false, &attn.pp("k_proj"))?,
                    v: linear(hidden, kv, false, &attn.pp("v_proj"))?,
                    o: linear(inner, hidden, false, &attn.pp("o_proj"))?,
                    gate: linear(hidden, cfg.intermediate_size, false, &mlp.pp("gate_proj"))?,
                    up: linear(hidden, cfg.intermediate_size, false, &mlp.pp("up_proj"))?,
                    down: linear(cfg.intermediate_size, hidden, false, &mlp.pp("down_proj"))?,
                    attn_scale: vb.pp("self_attn_layer_scale").get(hidden, "scale")?,
                    mlp_scale: vb.pp("mlp_layer_scale").get(hidden, "scale")?,
                })
            })
            .collect::<Result<_>>()?;
        #[allow(clippy::cast_possible_truncation)]
        let theta = cfg.rope_theta as f32;
        Ok(Self {
            input_proj: linear(cfg.latent_dim, hidden, true, &vb.pp("input_proj"))?,
            output_proj: linear(hidden, cfg.latent_dim, true, &vb.pp("output_proj"))?,
            layers,
            norm: RmsNorm::new(hidden, cfg.rms_norm_eps, vb.pp("norm"))?,
            rotary: RotaryEmbedding::new(
                theta,
                cfg.head_dim,
                cfg.max_position_embeddings,
                device,
                true,
                vb.dtype(),
            )?,
            heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim,
            window: cfg.sliding_window,
            act: cfg.hidden_act,
        })
    }

    // `(1, frames, latent)` in and out
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, seq, _) = xs.dims3()?;
        let shape = Tensor::zeros((b, seq), DType::U32, xs.device())?;
        let mask = CausalMasker.make_causal_mask(
            &shape,
            &NotACache,
            xs.dtype(),
            &CausalMaskConfig {
                sliding_window: Some(self.window),
                force_custom: true,
            },
        )?;
        let (cos, sin) = self.rotary.get_cos_sin()?;
        let (cos, sin) = (cos.narrow(0, 0, seq)?, sin.narrow(0, 0, seq)?);
        #[allow(clippy::cast_precision_loss)]
        let params = SdpaParams {
            n_kv_groups: 1,
            sliding_window: None,
            softcap: None,
            softmax_scale: 1.0 / (self.head_dim as f32).sqrt(),
            sinks: None,
            chunk: None,
        };
        let heads = |t: Tensor| -> Result<Tensor> {
            t.reshape((b, seq, (), self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let mut xs = self.input_proj.forward(xs)?;
        for layer in &self.layers {
            let h = layer.input_norm.forward(&xs)?;
            let (q, k) = apply_rotary_preselected_qk(
                &heads(layer.q.forward(&h)?)?,
                &heads(layer.k.forward(&h)?)?,
                &cos,
                &sin,
                true,
            )?;
            let v = heads(layer.v.forward(&h)?)?;
            let attn =
                Sdpa.run_attention(&q, &k, &v, &mask, Some(&FlashParams::empty(false)), &params)?;
            let attn = attn
                .transpose(1, 2)?
                .reshape((b, seq, self.heads * self.head_dim))?;
            xs = (xs + layer.o.forward(&attn)?.broadcast_mul(&layer.attn_scale)?)?;
            let h = layer.post_norm.forward(&xs)?;
            let gated = (self.act.forward(&layer.gate.forward(&h)?)? * layer.up.forward(&h)?)?;
            xs = (xs
                + layer
                    .down
                    .forward(&gated)?
                    .broadcast_mul(&layer.mlp_scale)?)?;
        }
        self.output_proj.forward(&self.norm.forward(&xs)?)
    }
}

/// One residual quantizer half: codebooks summed, then a 1x1 conv out.
struct Rvq {
    codebooks: Vec<Embedding>,
    // (out, in, 1), applied as a matmul over the codebook width
    output_proj: Tensor,
}

impl Rvq {
    fn new(count: usize, cfg: &CodecDecoderConfig, vb: &ShardedVarBuilder) -> Result<Self> {
        let dim = cfg.codebook_dim / 2;
        let codebooks = (0..count)
            .map(|i| {
                let vb = vb.pp("vq").pp("layers").pp(i).pp("_codebook");
                let usage = vb.get(cfg.codebook_size, "cluster_usage")?;
                let sum = vb.get((cfg.codebook_size, dim), "embedding_sum")?;
                let table = sum.broadcast_div(
                    &usage
                        .clamp(CLUSTER_USAGE_EPS, f64::INFINITY)?
                        .unsqueeze(1)?,
                )?;
                Ok(Embedding::new(table, dim))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            codebooks,
            output_proj: vb
                .get((cfg.codebook_dim, dim, 1), "output_proj.weight")?
                .squeeze(2)?,
        })
    }

    // codes `(count, frames)` to `(1, codebook_dim, frames)`
    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let mut sum: Option<Tensor> = None;
        for (i, book) in self.codebooks.iter().enumerate() {
            let e = book.forward(&codes.get(i)?)?;
            sum = Some(match sum {
                Some(s) => (s + e)?,
                None => e,
            });
        }
        let sum = sum.expect("a codebook");
        self.output_proj.matmul(&sum.t()?)?.unsqueeze(0)
    }
}

pub struct CodecDecoder {
    first: Rvq,
    rest: Rvq,
    semantic: usize,
    pre_conv: CausalConv,
    transformer: Transformer,
    upsample: Vec<(CausalTransConv, ConvNeXt)>,
    decoder_in: CausalConv,
    blocks: Vec<DecoderBlock>,
    out_act: SnakeBeta,
    out_conv: CausalConv,
    /// Samples per frame.
    pub upsample_rate: usize,
}

impl CodecDecoder {
    /// `vb` is the codec checkpoint's `decoder` prefix.
    pub fn new(cfg: &CodecDecoderConfig, vb: &ShardedVarBuilder, device: &Device) -> Result<Self> {
        let q = vb.pp("quantizer");
        let upsample = cfg
            .upsampling_ratios
            .iter()
            .enumerate()
            .map(|(i, &factor)| {
                let vb = vb.pp("upsample").pp(i);
                Ok((
                    CausalTransConv::new(
                        cfg.latent_dim,
                        cfg.latent_dim,
                        factor,
                        factor,
                        &vb.pp(0).pp("conv"),
                    )?,
                    ConvNeXt::new(cfg.latent_dim, &vb.pp(1))?,
                ))
            })
            .collect::<Result<_>>()?;
        let dvb = vb.pp("decoder");
        let blocks = cfg
            .upsample_rates
            .iter()
            .enumerate()
            .map(|(i, &rate)| {
                DecoderBlock::new(
                    cfg.decoder_dim >> i,
                    cfg.decoder_dim >> (i + 1),
                    rate,
                    &dvb.pp(i + 1),
                )
            })
            .collect::<Result<_>>()?;
        let out_dim = cfg.decoder_dim >> cfg.upsample_rates.len();
        let last = cfg.upsample_rates.len() + 1;
        Ok(Self {
            first: Rvq::new(cfg.num_semantic_quantizers, cfg, &q.pp("rvq_first"))?,
            rest: Rvq::new(
                cfg.num_quantizers - cfg.num_semantic_quantizers,
                cfg,
                &q.pp("rvq_rest"),
            )?,
            semantic: cfg.num_semantic_quantizers,
            pre_conv: CausalConv::new(
                cfg.codebook_dim,
                cfg.latent_dim,
                PRE_CONV_KERNEL,
                1,
                1,
                &vb.pp("pre_conv").pp("conv"),
            )?,
            transformer: Transformer::new(cfg, &vb.pp("pre_transformer"), device)?,
            upsample,
            decoder_in: CausalConv::new(
                cfg.latent_dim,
                cfg.decoder_dim,
                DECODER_KERNEL,
                1,
                1,
                &dvb.pp(0).pp("conv"),
            )?,
            blocks,
            out_act: SnakeBeta::new(out_dim, &dvb.pp(last))?,
            out_conv: CausalConv::new(
                out_dim,
                1,
                DECODER_KERNEL,
                1,
                1,
                &dvb.pp(last + 1).pp("conv"),
            )?,
            upsample_rate: cfg
                .upsample_rates
                .iter()
                .chain(&cfg.upsampling_ratios)
                .product(),
        })
    }

    // codes `(quantizers, frames)` to samples `(frames * upsample_rate)`
    fn decode_chunk(&self, codes: &Tensor) -> Result<Tensor> {
        let quantizers = codes.dim(0)?;
        let hidden = (self.first.decode(&codes.narrow(0, 0, self.semantic)?)?
            + self
                .rest
                .decode(&codes.narrow(0, self.semantic, quantizers - self.semantic)?)?)?;
        let hidden = self.pre_conv.forward(&hidden)?.transpose(1, 2)?;
        let mut xs = self.transformer.forward(&hidden)?.transpose(1, 2)?;
        for (conv, block) in &self.upsample {
            xs = block.forward(&conv.forward(&xs)?)?;
        }
        let mut wav = self.decoder_in.forward(&xs)?;
        for block in &self.blocks {
            wav = block.forward(&wav)?;
        }
        let wav = self.out_conv.forward(&self.out_act.forward(&wav)?)?;
        wav.clamp(-1f32, 1f32)?.flatten_all()
    }

    /// Codes `(quantizers, frames)` (u32) to 24 kHz samples, in the reference's chunks with left context.
    pub fn decode(&self, codes: &Tensor) -> Result<Vec<f32>> {
        let frames = codes.dim(1)?;
        let mut out = Vec::with_capacity(frames * self.upsample_rate);
        let mut start = 0;
        while start < frames {
            let end = (start + CHUNK_FRAMES).min(frames);
            let context = if start > CONTEXT_FRAMES {
                CONTEXT_FRAMES
            } else {
                start
            };
            let wav =
                self.decode_chunk(&codes.narrow(1, start - context, end - start + context)?)?;
            let skip = context * self.upsample_rate;
            out.extend(
                wav.narrow(0, skip, wav.dim(0)? - skip)?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?,
            );
            start = end;
        }
        Ok(out)
    }
}
