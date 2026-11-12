//! VibeVoice's speech-tokenizer encoder over raw 24 kHz audio: causal convs (left padding only), ConvNeXt-1d blocks
//! (RMSNorm, depthwise causal conv, a GELU FFN, each branch scaled by a learned gamma), one latent per hop.

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::Module;
use inference_tensor::{Result, Tensor};

use super::config::EncoderConfig;
use crate::layers::{Activation, RmsNorm, linear_b};

/// The causal context each conv carries from one chunk of audio into the next.
pub struct EncoderCache(Vec<Option<Tensor>>);

impl EncoderCache {
    pub fn new() -> Self {
        Self(Vec::new())
    }
}

impl Default for EncoderCache {
    fn default() -> Self {
        Self::new()
    }
}

struct CausalConv {
    weight: Tensor,
    bias: Tensor,
    stride: usize,
    groups: usize,
    left_pad: usize,
    // this conv's slot in the cache
    slot: usize,
}

impl CausalConv {
    fn new(
        (in_ch, out_ch, kernel, stride, groups): (usize, usize, usize, usize, usize),
        slot: &mut usize,
        vb: &ShardedVarBuilder,
    ) -> Result<Self> {
        let conv = Self {
            weight: vb.get((out_ch, in_ch / groups, kernel), "weight")?,
            bias: vb.get(out_ch, "bias")?,
            stride,
            groups,
            left_pad: kernel - stride,
            slot: *slot,
        };
        *slot += 1;
        Ok(conv)
    }

    // `xs` `(1, c, t)` after the previous chunk's last `left_pad` samples (zeros before the first)
    fn forward(&self, xs: &Tensor, cache: &mut EncoderCache) -> Result<Tensor> {
        if cache.0.len() <= self.slot {
            cache.0.resize(self.slot + 1, None);
        }
        let (b, c, _) = xs.dims3()?;
        let past = match cache.0[self.slot].take() {
            Some(past) => past,
            None => Tensor::zeros((b, c, self.left_pad), xs.dtype(), xs.device())?,
        };
        let padded = Tensor::cat(&[&past, xs], 2)?;
        let len = padded.dim(2)?;
        cache.0[self.slot] = Some(
            padded
                .narrow(2, len - self.left_pad, self.left_pad)?
                .contiguous()?,
        );
        padded
            .conv1d(&self.weight, 0, self.stride, 1, self.groups)?
            .broadcast_add(&self.bias.reshape((1, (), 1))?)
    }
}

struct ConvNextBlock {
    norm: RmsNorm,
    mixer: CausalConv,
    gamma: Tensor,
    ffn_norm: RmsNorm,
    linear1: inference_tensor::nn::Linear,
    linear2: inference_tensor::nn::Linear,
    act: Activation,
    ffn_gamma: Tensor,
}

impl ConvNextBlock {
    fn new(
        cfg: &EncoderConfig,
        width: usize,
        slot: &mut usize,
        vb: &ShardedVarBuilder,
    ) -> Result<Self> {
        let inner = cfg.ffn_expansion * width;
        let ffn = vb.pp("ffn");
        Ok(Self {
            norm: RmsNorm::new(width, cfg.rms_norm_eps, vb.pp("norm"))?,
            mixer: CausalConv::new(
                (width, width, cfg.kernel_size, 1, width),
                slot,
                &vb.pp("mixer").pp("conv"),
            )?,
            gamma: vb.get(width, "gamma")?.reshape((1, (), 1))?,
            ffn_norm: RmsNorm::new(width, cfg.rms_norm_eps, vb.pp("ffn_norm"))?,
            linear1: linear_b(width, inner, true, ffn.pp("linear1"))?,
            linear2: linear_b(inner, width, true, ffn.pp("linear2"))?,
            act: cfg.hidden_act,
            ffn_gamma: vb.get(width, "ffn_gamma")?.reshape((1, (), 1))?,
        })
    }

    // `xs` is `(1, c, t)`; the norms and the FFN run over channels
    fn forward(&self, xs: &Tensor, cache: &mut EncoderCache) -> Result<Tensor> {
        let normed = self.norm.forward(&xs.transpose(1, 2)?)?.transpose(1, 2)?;
        let mixed = self.mixer.forward(&normed.contiguous()?, cache)?;
        let xs = (xs + mixed.broadcast_mul(&self.gamma)?)?;
        let normed = self.ffn_norm.forward(&xs.transpose(1, 2)?)?;
        let ffn = self
            .linear2
            .forward(&self.act.forward(&self.linear1.forward(&normed)?)?)?
            .transpose(1, 2)?;
        xs + ffn.broadcast_mul(&self.ffn_gamma)?
    }
}

struct Stage {
    conv: CausalConv,
    blocks: Vec<ConvNextBlock>,
}

impl Stage {
    fn forward(&self, xs: &Tensor, cache: &mut EncoderCache) -> Result<Tensor> {
        let mut xs = self.conv.forward(xs, cache)?;
        for block in &self.blocks {
            xs = block.forward(&xs, cache)?;
        }
        Ok(xs)
    }
}

pub struct SpeechEncoder {
    stages: Vec<Stage>,
    head: CausalConv,
}

impl SpeechEncoder {
    pub fn new(cfg: &EncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let mut slot = 0;
        let blocks = |width: usize, count: usize, slot: &mut usize, vb: &ShardedVarBuilder| {
            (0..count)
                .map(|i| ConvNextBlock::new(cfg, width, slot, &vb.pp("stage").pp(i)))
                .collect::<Result<Vec<_>>>()
        };
        let stem = vb.pp("stem");
        let mut stages = vec![Stage {
            conv: CausalConv::new(
                (cfg.channels, cfg.num_filters, cfg.kernel_size, 1, 1),
                &mut slot,
                &stem.pp("conv").pp("conv"),
            )?,
            blocks: blocks(cfg.num_filters, cfg.depths[0], &mut slot, &stem)?,
        }];
        for (i, &ratio) in cfg.downsampling_ratios.iter().enumerate() {
            let layer = vb.pp("conv_layers").pp(i);
            let (from, to) = (cfg.num_filters << i, cfg.num_filters << (i + 1));
            stages.push(Stage {
                conv: CausalConv::new(
                    (from, to, 2 * ratio, ratio, 1),
                    &mut slot,
                    &layer.pp("conv").pp("conv"),
                )?,
                blocks: blocks(to, cfg.depths[i + 1], &mut slot, &layer)?,
            });
        }
        let last = cfg.num_filters << cfg.downsampling_ratios.len();
        let head = CausalConv::new(
            (last, cfg.hidden_size, cfg.kernel_size, 1, 1),
            &mut slot,
            &vb.pp("head").pp("conv"),
        )?;
        Ok(Self { stages, head })
    }

    /// The latents `(1, samples / hop, hidden)` of `(1, 1, samples)` audio after what `cache` has seen.
    pub fn forward(&self, xs: &Tensor, cache: &mut EncoderCache) -> Result<Tensor> {
        let mut xs = xs.clone();
        for stage in &self.stages {
            xs = stage.forward(&xs, cache)?;
        }
        self.head.forward(&xs, cache)?.transpose(1, 2)
    }
}
