use candle_core::{Module, Result, Tensor};
use candle_nn::{Activation, VarBuilder};

use crate::layers::{ConvNorm, ConvNormSpec};

const ACT: Activation = Activation::Relu;
const STEM_CHANNELS: [usize; 3] = [3, 32, 48];
const LAYERS_PER_BLOCK: usize = 6;

struct StageSpec {
    in_c: usize,
    mid_c: usize,
    out_c: usize,
    blocks: usize,
    downsample: bool,
    light: bool,
    kernel: usize,
}

/// HGNetV2 "L" as instantiated by the HF config defaults (the checkpoint only records `arch: L`).
const STAGES: [StageSpec; 4] = [
    StageSpec {
        in_c: 48,
        mid_c: 48,
        out_c: 128,
        blocks: 1,
        downsample: false,
        light: false,
        kernel: 3,
    },
    StageSpec {
        in_c: 128,
        mid_c: 96,
        out_c: 512,
        blocks: 1,
        downsample: true,
        light: false,
        kernel: 3,
    },
    StageSpec {
        in_c: 512,
        mid_c: 192,
        out_c: 1024,
        blocks: 3,
        downsample: true,
        light: true,
        kernel: 5,
    },
    StageSpec {
        in_c: 1024,
        mid_c: 384,
        out_c: 2048,
        blocks: 1,
        downsample: true,
        light: true,
        kernel: 5,
    },
];

struct Embeddings {
    stem1: ConvNorm,
    stem2a: ConvNorm,
    stem2b: ConvNorm,
    stem3: ConvNorm,
    stem4: ConvNorm,
}

impl Embeddings {
    fn new(vb: VarBuilder) -> Result<Self> {
        let [c0, c1, c2] = STEM_CHANNELS;
        Ok(Self {
            stem1: ConvNormSpec::new(c0, c1, 3)
                .stride(2)
                .act(ACT)
                .load(vb.pp("stem1"))?,
            stem2a: ConvNormSpec::new(c1, c1 / 2, 2)
                .act(ACT)
                .load(vb.pp("stem2a"))?,
            stem2b: ConvNormSpec::new(c1 / 2, c1, 2)
                .act(ACT)
                .load(vb.pp("stem2b"))?,
            stem3: ConvNormSpec::new(c1 * 2, c1, 3)
                .stride(2)
                .act(ACT)
                .load(vb.pp("stem3"))?,
            stem4: ConvNormSpec::new(c1, c2, 1).act(ACT).load(vb.pp("stem4"))?,
        })
    }
}

fn pad_br(xs: &Tensor) -> Result<Tensor> {
    xs.pad_with_zeros(3, 0, 1)?.pad_with_zeros(2, 0, 1)
}

impl Module for Embeddings {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let emb = pad_br(&self.stem1.forward(xs)?)?;
        let s2 = self.stem2b.forward(&pad_br(&self.stem2a.forward(&emb)?)?)?;
        // ceil_mode is a no-op for a stride-1 pool
        let pooled = emb.max_pool2d_with_stride(2, 1)?;
        let emb = Tensor::cat(&[pooled, s2], 1)?;
        self.stem4.forward(&self.stem3.forward(&emb)?)
    }
}

enum Layer {
    Plain(ConvNorm),
    Light(ConvNorm, ConvNorm),
}

impl Module for Layer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Plain(c) => c.forward(xs),
            Self::Light(c1, c2) => c2.forward(&c1.forward(xs)?),
        }
    }
}

struct BasicLayer {
    layers: Vec<Layer>,
    squeeze: ConvNorm,
    excite: ConvNorm,
    residual: bool,
}

impl BasicLayer {
    fn new(in_c: usize, s: &StageSpec, residual: bool, vb: VarBuilder) -> Result<Self> {
        let vbl = vb.pp("layers");
        let mut layers = Vec::with_capacity(LAYERS_PER_BLOCK);
        for i in 0..LAYERS_PER_BLOCK {
            let li = if i == 0 { in_c } else { s.mid_c };
            let vb = vbl.pp(i);
            layers.push(if s.light {
                Layer::Light(
                    ConvNormSpec::new(li, s.mid_c, 1).load(vb.pp("conv1"))?,
                    ConvNormSpec::new(s.mid_c, s.mid_c, s.kernel)
                        .groups(s.mid_c)
                        .act(ACT)
                        .load(vb.pp("conv2"))?,
                )
            } else {
                Layer::Plain(ConvNormSpec::new(li, s.mid_c, s.kernel).act(ACT).load(vb)?)
            });
        }
        let total = in_c + LAYERS_PER_BLOCK * s.mid_c;
        let vba = vb.pp("aggregation");
        Ok(Self {
            layers,
            squeeze: ConvNormSpec::new(total, s.out_c / 2, 1)
                .act(ACT)
                .load(vba.pp(0))?,
            excite: ConvNormSpec::new(s.out_c / 2, s.out_c, 1)
                .act(ACT)
                .load(vba.pp(1))?,
            residual,
        })
    }
}

impl Module for BasicLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut outs = Vec::with_capacity(self.layers.len() + 1);
        outs.push(xs.clone());
        let mut h = xs.clone();
        for l in &self.layers {
            h = l.forward(&h)?;
            outs.push(h.clone());
        }
        let agg = self
            .excite
            .forward(&self.squeeze.forward(&Tensor::cat(&outs, 1)?)?)?;
        if self.residual {
            agg + xs
        } else {
            Ok(agg)
        }
    }
}

struct Stage {
    downsample: Option<ConvNorm>,
    blocks: Vec<BasicLayer>,
}

impl Stage {
    fn new(s: &StageSpec, vb: VarBuilder) -> Result<Self> {
        let downsample = if s.downsample {
            Some(
                ConvNormSpec::new(s.in_c, s.in_c, 3)
                    .stride(2)
                    .groups(s.in_c)
                    .load(vb.pp("downsample"))?,
            )
        } else {
            None
        };
        let blocks = (0..s.blocks)
            .map(|i| {
                let in_c = if i == 0 { s.in_c } else { s.out_c };
                BasicLayer::new(in_c, s, i != 0, vb.pp("blocks").pp(i))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { downsample, blocks })
    }
}

impl Module for Stage {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = match &self.downsample {
            Some(d) => d.forward(xs)?,
            None => xs.clone(),
        };
        for b in &self.blocks {
            xs = b.forward(&xs)?;
        }
        Ok(xs)
    }
}

pub struct HGNetV2Backbone {
    embedder: Embeddings,
    stages: Vec<Stage>,
}

impl HGNetV2Backbone {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let stages = STAGES
            .iter()
            .enumerate()
            .map(|(i, s)| Stage::new(s, vb.pp("encoder").pp("stages").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embedder: Embeddings::new(vb.pp("embedder"))?,
            stages,
        })
    }

    pub fn out_channels() -> [usize; 4] {
        STAGES.map(|s| s.out_c)
    }

    /// Returns every stage output (strides 4, 8, 16, 32).
    pub fn forward(&self, xs: &Tensor) -> Result<Vec<Tensor>> {
        let mut h = self.embedder.forward(xs)?;
        let mut feats = Vec::with_capacity(self.stages.len());
        for s in &self.stages {
            h = s.forward(&h)?;
            feats.push(h.clone());
        }
        Ok(feats)
    }
}
