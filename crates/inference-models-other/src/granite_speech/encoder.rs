//! Granite Speech's CTC Conformer encoder: block-local attention with Shaw relative positions, and a mid-stack CTC
//! head whose softmax feeds back into the residual.
#![allow(clippy::cast_precision_loss)]

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{Embedding, LayerNorm, Linear, Module};
use inference_tensor::{D, Device, Result, Tensor};

use super::config::EncoderConfig;
use crate::layers::{layer_norm, linear_b, linear_no_bias};
use crate::utils::unvarbuilder::UnVarBuilder;

// torch's LayerNorm and BatchNorm1d defaults
const NORM_EPS: f64 = 1e-5;
// transformers fills masked relative-position scores with the dtype's most negative finite value
const MASKED: f32 = -f32::MAX;

struct FeedForward {
    pre_norm: LayerNorm,
    up_proj: Linear,
    down_proj: Linear,
}

impl FeedForward {
    fn new(cfg: &EncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let inner = cfg.hidden_dim * cfg.feedforward_mult;
        Ok(Self {
            pre_norm: layer_norm(cfg.hidden_dim, NORM_EPS, vb.pp("pre_norm"))?,
            up_proj: linear_b(cfg.hidden_dim, inner, true, vb.pp("up_proj"))?,
            down_proj: linear_b(inner, cfg.hidden_dim, true, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.up_proj.forward(&self.pre_norm.forward(xs)?)?.silu()?;
        self.down_proj.forward(&xs)
    }
}

struct Attention {
    pre_norm: LayerNorm,
    to_q: Linear,
    to_kv: Linear,
    to_out: Linear,
    rel_pos_emb: Embedding,
    /// `clamp(i - j, -context, context) + max_pos_emb` for every query `i` and key `j` of a block.
    distances: Tensor,
    heads: usize,
    head_dim: usize,
    context: usize,
}

impl Attention {
    fn new(cfg: &EncoderConfig, distances: &Tensor, vb: ShardedVarBuilder) -> Result<Self> {
        let inner = cfg.dim_head * cfg.num_heads;
        let rel_pos_emb = vb
            .pp("rel_pos_emb")
            .get((2 * cfg.max_pos_emb + 1, cfg.dim_head), "weight")?;
        Ok(Self {
            pre_norm: layer_norm(cfg.hidden_dim, NORM_EPS, vb.pp("pre_norm"))?,
            to_q: linear_no_bias(cfg.hidden_dim, inner, vb.pp("to_q"))?,
            to_kv: linear_no_bias(cfg.hidden_dim, 2 * inner, vb.pp("to_kv"))?,
            to_out: linear_b(inner, cfg.hidden_dim, true, vb.pp("to_out"))?,
            rel_pos_emb: Embedding::new(rel_pos_emb, cfg.dim_head),
            distances: distances.to_device(vb.device())?,
            heads: cfg.num_heads,
            head_dim: cfg.dim_head,
            context: cfg.context_size,
        })
    }

    /// The last block's padding: a query or key past the `valid` frames scores the most negative value.
    fn padding_mask(&self, valid: usize, device: &Device) -> Result<Tensor> {
        let c = self.context;
        let mask = (0..c * c)
            .map(|i| {
                if i / c < valid && i % c < valid {
                    0f32
                } else {
                    MASKED
                }
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(mask, (c, c), device)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, frames, _) = xs.dims3()?;
        let c = self.context;
        let blocks = frames.div_ceil(c);
        let remainder = frames % c;
        let xs = self.pre_norm.forward(xs)?;
        let xs = if remainder > 0 {
            xs.pad_with_zeros(1, 0, c - remainder)?
        } else {
            xs
        };
        let (bn, h, d) = (batch * blocks, self.heads, self.head_dim);
        let split = |t: Tensor| -> Result<Tensor> {
            t.reshape((bn, c, h, d))?.transpose(1, 2)?.contiguous()
        };
        let q = split(self.to_q.forward(&xs)?)?;
        let kv = self.to_kv.forward(&xs)?.chunk(2, D::Minus1)?;
        let (k, v) = (split(kv[0].clone())?, split(kv[1].clone())?);
        let scale = (d as f64).powf(-0.5);

        // pos[b, h, i, j] = q[b, h, i] . rel[i, j]
        let rel = self
            .rel_pos_emb
            .forward(&self.distances)?
            .to_dtype(q.dtype())?
            .transpose(1, 2)?
            .contiguous()?;
        let pos = q
            .permute((2, 0, 1, 3))?
            .reshape((c, bn * h, d))?
            .contiguous()?
            .matmul(&rel)?
            .reshape((c, bn, h, c))?
            .permute((1, 2, 0, 3))?;
        let mut pos = (pos * scale)?;
        if remainder > 0 {
            let mask = self
                .padding_mask(remainder, q.device())?
                .to_dtype(pos.dtype())?;
            let zeros = Tensor::zeros((blocks - 1, c, c), pos.dtype(), q.device())?;
            let per_block = Tensor::cat(&[zeros, mask.unsqueeze(0)?], 0)?
                .unsqueeze(1)?
                .repeat((batch, 1, 1, 1))?;
            pos = pos.broadcast_add(&per_block)?;
        }
        let scores = (q.matmul(&k.t()?)? * scale)?.add(&pos)?;
        let weights = inference_tensor::nn::ops::softmax_last_dim(&scores)?;
        let out = weights
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch, blocks * c, h * d))?
            .narrow(1, 0, frames)?;
        self.to_out.forward(&out)
    }
}

struct ConvModule {
    norm: LayerNorm,
    up_conv: Linear,
    depth_conv: Tensor,
    // BatchNorm1d at inference, folded: `x * scale + shift`
    bn_scale: Tensor,
    bn_shift: Tensor,
    down_conv: Linear,
    padding: usize,
    // kept for the residual export
    batch_norm: [Tensor; 4],
}

impl ConvModule {
    fn new(cfg: &EncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let (hidden, inner) = (cfg.hidden_dim, cfg.hidden_dim * cfg.conv_expansion_factor);
        let pointwise = |vb: ShardedVarBuilder, out: usize, inp: usize| -> Result<Linear> {
            let weight = vb.get((out, inp, 1), "weight")?.squeeze(2)?;
            Ok(Linear::new(weight, Some(vb.get(out, "bias")?)))
        };
        let bn = vb.pp("batch_norm");
        let weight = bn.get(inner, "weight")?;
        let bias = bn.get(inner, "bias")?;
        // llama.cpp's GGUF stores the statistics folded in: its weight and bias are the scale and shift
        let (mean, var) = if crate::layers::contains_tensor_or_weight_source(&bn, "running_mean") {
            (
                bn.get(inner, "running_mean")?,
                bn.get(inner, "running_var")?,
            )
        } else {
            let zeros = weight.zeros_like()?;
            (zeros.clone(), (zeros + (1.0 - NORM_EPS))?)
        };
        let bn_scale = weight.broadcast_div(&(var.clone() + NORM_EPS)?.sqrt()?)?;
        let bn_shift = (&bias - mean.mul(&bn_scale)?)?;
        if cfg.conv_kernel_size.is_multiple_of(2) {
            inference_tensor::bail!("Granite Speech's depthwise convolution takes an odd kernel");
        }
        Ok(Self {
            norm: layer_norm(hidden, NORM_EPS, vb.pp("norm"))?,
            up_conv: pointwise(vb.pp("up_conv"), 2 * inner, hidden)?,
            depth_conv: vb
                .pp("depth_conv")
                .pp("conv")
                .get((inner, 1, cfg.conv_kernel_size), "weight")?,
            bn_scale: bn_scale.reshape((1, inner, 1))?,
            bn_shift: bn_shift.reshape((1, inner, 1))?,
            down_conv: pointwise(vb.pp("down_conv"), hidden, inner)?,
            padding: cfg.conv_kernel_size / 2,
            batch_norm: [weight, bias, mean, var],
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.up_conv.forward(&self.norm.forward(xs)?)?;
        let halves = xs.chunk(2, D::Minus1)?;
        let xs = halves[0].mul(&inference_tensor::nn::ops::sigmoid(&halves[1])?)?;
        let channels = xs.dim(D::Minus1)?;
        let xs = xs.transpose(1, 2)?.contiguous()?.conv1d(
            &self.depth_conv,
            self.padding,
            1,
            1,
            channels,
        )?;
        let xs = xs
            .broadcast_mul(&self.bn_scale)?
            .broadcast_add(&self.bn_shift)?
            .silu()?;
        self.down_conv.forward(&xs.transpose(1, 2)?.contiguous()?)
    }
}

struct Block {
    ff1: FeedForward,
    attn: Attention,
    conv: ConvModule,
    ff2: FeedForward,
    post_norm: LayerNorm,
}

impl Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = ((self.ff1.forward(xs)? * 0.5)? + xs)?;
        let xs = (self.attn.forward(&xs)? + &xs)?;
        let xs = (self.conv.forward(&xs)? + &xs)?;
        let xs = ((self.ff2.forward(&xs)? * 0.5)? + &xs)?;
        self.post_norm.forward(&xs)
    }
}

pub struct Encoder {
    input_linear: Linear,
    layers: Vec<Block>,
    out: Linear,
    out_mid: Linear,
    cat_hidden_layers: Vec<usize>,
}

impl Encoder {
    pub fn new(cfg: &EncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let c = cfg.context_size as i64;
        let max = cfg.max_pos_emb as i64;
        let distances = (0..c * c)
            .map(|i| {
                u32::try_from((i / c - i % c).clamp(-c, c) + max).expect("a non-negative distance")
            })
            .collect::<Vec<_>>();
        let distances = Tensor::from_vec(
            distances,
            (cfg.context_size, cfg.context_size),
            &Device::Cpu,
        )?;
        let layers_vb = vb.pp("layers");
        let layers = (0..cfg.num_layers)
            .map(|i| {
                let vb = layers_vb.pp(i);
                Ok(Block {
                    ff1: FeedForward::new(cfg, vb.pp("ff1"))?,
                    attn: Attention::new(cfg, &distances, vb.pp("attn"))?,
                    conv: ConvModule::new(cfg, vb.pp("conv"))?,
                    ff2: FeedForward::new(cfg, vb.pp("ff2"))?,
                    post_norm: layer_norm(cfg.hidden_dim, NORM_EPS, vb.pp("post_norm"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            input_linear: linear_b(cfg.input_dim, cfg.hidden_dim, true, vb.pp("input_linear"))?,
            layers,
            out: linear_b(cfg.hidden_dim, cfg.output_dim, true, vb.pp("out"))?,
            out_mid: linear_b(cfg.output_dim, cfg.hidden_dim, true, vb.pp("out_mid"))?,
            cat_hidden_layers: cfg.cat_hidden_layers.clone().unwrap_or_default(),
        })
    }

    /// `(batch, frames, input)` stacked log-mel features to `(batch, frames, hidden * (1 + cat layers))`.
    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let mut xs = self.input_linear.forward(features)?;
        let mut exported = Vec::new();
        if self.cat_hidden_layers.contains(&0) {
            exported.push(xs.clone());
        }
        let mid = self.layers.len() / 2;
        for (i, layer) in self.layers.iter().enumerate() {
            let idx = i + 1;
            xs = layer.forward(&xs)?;
            if self.cat_hidden_layers.contains(&idx) {
                exported.push(xs.clone());
            }
            if idx == mid {
                let ctc = inference_tensor::nn::ops::softmax_last_dim(&self.out.forward(&xs)?)?;
                xs = (&xs + self.out_mid.forward(&ctc)?)?;
            }
        }
        if exported.is_empty() {
            return Ok(xs);
        }
        exported.push(xs);
        Tensor::cat(&exported, D::Minus1)
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("input_linear").add(&self.input_linear);
        uvb.pp("out").add(&self.out);
        uvb.pp("out_mid").add(&self.out_mid);
        for (i, layer) in self.layers.iter().enumerate() {
            let l = uvb.pp("layers").pp(i);
            for (name, ff) in [("ff1", &layer.ff1), ("ff2", &layer.ff2)] {
                l.pp(name).pp("pre_norm").add(&ff.pre_norm);
                l.pp(name).pp("up_proj").add(&ff.up_proj);
                l.pp(name).pp("down_proj").add(&ff.down_proj);
            }
            let attn = l.pp("attn");
            attn.pp("pre_norm").add(&layer.attn.pre_norm);
            attn.pp("to_q").add(&layer.attn.to_q);
            attn.pp("to_kv").add(&layer.attn.to_kv);
            attn.pp("to_out").add(&layer.attn.to_out);
            attn.pp("rel_pos_emb")
                .add_tensor("weight", layer.attn.rel_pos_emb.embeddings().clone());
            let conv = l.pp("conv");
            conv.pp("norm").add(&layer.conv.norm);
            for (name, linear) in [
                ("up_conv", &layer.conv.up_conv),
                ("down_conv", &layer.conv.down_conv),
            ] {
                conv.pp(name)
                    .add_tensor("weight", linear.weight().unsqueeze(2).expect("a 2D weight"));
                if let Some(bias) = linear.bias() {
                    conv.pp(name).add_tensor("bias", bias.clone());
                }
            }
            conv.pp("depth_conv")
                .pp("conv")
                .add_tensor("weight", layer.conv.depth_conv.clone());
            let [weight, bias, mean, var] = layer.conv.batch_norm.clone();
            let bn = conv.pp("batch_norm");
            bn.add_tensor("weight", weight);
            bn.add_tensor("bias", bias);
            bn.add_tensor("running_mean", mean);
            bn.add_tensor("running_var", var);
            l.pp("post_norm").add(&layer.post_norm);
        }
        uvb.to_safetensors()
    }
}
