//! VibeVoice's speech-tokenizer encoder over raw 24 kHz audio: causal convs (left padding only), ConvNeXt-1d blocks
//! (RMSNorm, depthwise causal conv, a GELU FFN, each branch scaled by a learned gamma), one latent per hop.

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::Module;
use inference_tensor::{Result, Tensor};

use super::config::EncoderConfig;
use crate::layers::{Activation, RmsNorm, linear_b};

/// Where a checkpoint keeps the encoder's parts: transformers' port, or Microsoft's own layout (the streaming
/// checkpoints), whose convs sit one wrapper deeper and whose stem is its first downsampling layer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EncoderNames {
    #[default]
    Transformers,
    Upstream,
}

impl EncoderNames {
    // stage `i`'s conv (0 the stem) and the prefix of its blocks
    fn stage(self, vb: &ShardedVarBuilder, i: usize) -> (ShardedVarBuilder, ShardedVarBuilder) {
        match (self, i) {
            (Self::Transformers, 0) => {
                let stem = vb.pp("stem");
                (stem.pp("conv").pp("conv"), stem.pp("stage"))
            }
            (Self::Transformers, i) => {
                let layer = vb.pp("conv_layers").pp(i - 1);
                (layer.pp("conv").pp("conv"), layer.pp("stage"))
            }
            (Self::Upstream, i) => (
                vb.pp("downsample_layers").pp(i).pp(0).pp("conv").pp("conv"),
                vb.pp("stages").pp(i),
            ),
        }
    }

    fn head(self, vb: &ShardedVarBuilder) -> ShardedVarBuilder {
        match self {
            Self::Transformers => vb.pp("head").pp("conv"),
            Self::Upstream => vb.pp("head").pp("conv").pp("conv"),
        }
    }

    fn mixer(self, vb: &ShardedVarBuilder) -> ShardedVarBuilder {
        match self {
            Self::Transformers => vb.pp("mixer").pp("conv"),
            Self::Upstream => vb.pp("mixer").pp("conv").pp("conv").pp("conv"),
        }
    }
}

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
        names: EncoderNames,
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
                &names.mixer(vb),
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
    pub fn new(cfg: &EncoderConfig, names: EncoderNames, vb: ShardedVarBuilder) -> Result<Self> {
        let mut slot = 0;
        let mut stages = Vec::with_capacity(cfg.downsampling_ratios.len() + 1);
        for i in 0..=cfg.downsampling_ratios.len() {
            let (conv, blocks) = names.stage(&vb, i);
            // the stem keeps the width; each downsampling conv doubles it
            let (shape, width) = match i {
                0 => (
                    (cfg.channels, cfg.num_filters, cfg.kernel_size, 1, 1),
                    cfg.num_filters,
                ),
                i => {
                    let ratio = cfg.downsampling_ratios[i - 1];
                    let (from, to) = (cfg.num_filters << (i - 1), cfg.num_filters << i);
                    ((from, to, 2 * ratio, ratio, 1), to)
                }
            };
            stages.push(Stage {
                conv: CausalConv::new(shape, &mut slot, &conv)?,
                blocks: (0..cfg.depths[i])
                    .map(|j| ConvNextBlock::new(cfg, names, width, &mut slot, &blocks.pp(j)))
                    .collect::<Result<_>>()?,
            });
        }
        let last = cfg.num_filters << cfg.downsampling_ratios.len();
        let head = CausalConv::new(
            (last, cfg.hidden_size, cfg.kernel_size, 1, 1),
            &mut slot,
            &names.head(&vb),
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
