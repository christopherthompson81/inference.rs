use inference_tensor::nn::{
    Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear, Module, VarBuilder, embedding, layer_norm,
    linear, ops,
};
use inference_tensor::{D, Result, Tensor};

use super::istftnet::AdainResBlk1d;
use super::lstm::BiLstm;
use crate::weight_norm::{conv1d, conv1d_weight_norm};

const NORM_EPS: f64 = 1e-5;
const TEXT_SLOPE: f64 = 0.2;

/// Layer norm over channels without affine, then a style-driven affine; on (1, T, channels).
#[derive(Debug, Clone)]
struct AdaLayerNorm {
    fc: Linear,
    channels: usize,
}

impl AdaLayerNorm {
    fn new(style_dim: usize, channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc: linear(style_dim, 2 * channels, vb.pp("fc"))?,
            channels,
        })
    }

    fn forward(&self, xs: &Tensor, s: &Tensor) -> Result<Tensor> {
        let h = self.fc.forward(s)?.unsqueeze(1)?;
        let gamma = (h.narrow(2, 0, self.channels)? + 1.)?;
        let beta = h.narrow(2, self.channels, self.channels)?;
        let mean = xs.mean_keepdim(D::Minus1)?;
        let centered = xs.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = centered.broadcast_div(&(var + NORM_EPS)?.sqrt()?)?;
        normed.broadcast_mul(&gamma)?.broadcast_add(&beta)
    }
}

/// The predictor's text encoder: BiLSTMs over the BERT features with the style appended, each followed by AdaLayerNorm.
#[derive(Debug, Clone)]
struct DurationEncoder {
    blocks: Vec<(BiLstm, AdaLayerNorm)>,
}

impl DurationEncoder {
    fn new(style_dim: usize, d_model: usize, layers: usize, vb: VarBuilder) -> Result<Self> {
        let blocks = (0..layers)
            .map(|i| {
                Ok((
                    BiLstm::new(d_model + style_dim, d_model / 2, vb.pp(2 * i))?,
                    AdaLayerNorm::new(style_dim, d_model, vb.pp(2 * i + 1))?,
                ))
            })
            .collect::<Result<_>>()?;
        Ok(Self { blocks })
    }

    /// `xs` is (1, T, d_model); returns (1, T, d_model + style_dim).
    fn forward(&self, xs: &Tensor, s: &Tensor) -> Result<Tensor> {
        let t = xs.dim(1)?;
        let style = s.unsqueeze(1)?.broadcast_as((1, t, s.dim(1)?))?;
        let mut xs = Tensor::cat(&[xs, &style], 2)?;
        for (lstm, norm) in &self.blocks {
            xs = Tensor::cat(&[&norm.forward(&lstm.forward(&xs)?, s)?, &style], 2)?;
        }
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
pub struct ProsodyPredictor {
    text_encoder: DurationEncoder,
    lstm: BiLstm,
    duration_proj: Linear,
    shared: BiLstm,
    f0: Vec<AdainResBlk1d>,
    n: Vec<AdainResBlk1d>,
    f0_proj: Conv1d,
    n_proj: Conv1d,
}

impl ProsodyPredictor {
    pub fn new(
        style_dim: usize,
        d_hid: usize,
        layers: usize,
        max_dur: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let stack = |name: &str| -> Result<Vec<AdainResBlk1d>> {
            let vb = vb.pp(name);
            Ok(vec![
                AdainResBlk1d::new(d_hid, d_hid, style_dim, false, vb.pp(0))?,
                AdainResBlk1d::new(d_hid, d_hid / 2, style_dim, true, vb.pp(1))?,
                AdainResBlk1d::new(d_hid / 2, d_hid / 2, style_dim, false, vb.pp(2))?,
            ])
        };
        Ok(Self {
            text_encoder: DurationEncoder::new(
                style_dim,
                d_hid,
                layers,
                vb.pp("text_encoder.lstms"),
            )?,
            lstm: BiLstm::new(d_hid + style_dim, d_hid / 2, vb.pp("lstm"))?,
            duration_proj: linear(d_hid, max_dur, vb.pp("duration_proj.linear_layer"))?,
            shared: BiLstm::new(d_hid + style_dim, d_hid / 2, vb.pp("shared"))?,
            f0: stack("F0")?,
            n: stack("N")?,
            f0_proj: conv1d(d_hid / 2, 1, 1, Default::default(), vb.pp("F0_proj"))?,
            n_proj: conv1d(d_hid / 2, 1, 1, Default::default(), vb.pp("N_proj"))?,
        })
    }

    /// Returns the duration features `d` (1, T, d_hid + style_dim) and the per-token durations before rounding.
    pub fn durations(&self, d_en: &Tensor, s: &Tensor, speed: f64) -> Result<(Tensor, Vec<f32>)> {
        let d = self.text_encoder.forward(d_en, s)?;
        let logits = self.duration_proj.forward(&self.lstm.forward(&d)?)?;
        let speed = Tensor::new(speed as f32, logits.device())?;
        let duration = ops::sigmoid(&logits)?
            .sum(D::Minus1)?
            .broadcast_div(&speed)?;
        Ok((d, duration.flatten_all()?.to_vec1::<f32>()?))
    }

    /// `en` is (1, frames, d_hid + style_dim); returns F0 and N curves, each (1, 2 * frames).
    pub fn f0_n(&self, en: &Tensor, s: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = self.shared.forward(en)?.transpose(1, 2)?.contiguous()?;
        let run = |blocks: &[AdainResBlk1d], proj: &Conv1d| -> Result<Tensor> {
            let mut h = x.clone();
            for block in blocks {
                h = block.forward(&h, s)?;
            }
            proj.forward(&h)?.squeeze(1)
        };
        Ok((run(&self.f0, &self.f0_proj)?, run(&self.n, &self.n_proj)?))
    }
}

#[derive(Debug, Clone)]
pub struct TextEncoder {
    embedding: Embedding,
    cnn: Vec<(Conv1d, LayerNorm)>,
    lstm: BiLstm,
}

impl TextEncoder {
    pub fn new(
        channels: usize,
        kernel: usize,
        depth: usize,
        n_symbols: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv = Conv1dConfig {
            padding: (kernel - 1) / 2,
            ..Default::default()
        };
        let cnn = (0..depth)
            .map(|i| {
                let vb = vb.pp("cnn").pp(i);
                Ok((
                    conv1d_weight_norm(channels, channels, kernel, true, conv, vb.pp(0))?,
                    layer_norm(channels, NORM_EPS, vb.pp(1))?,
                ))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            embedding: embedding(n_symbols, channels, vb.pp("embedding"))?,
            cnn,
            lstm: BiLstm::new(channels, channels / 2, vb.pp("lstm"))?,
        })
    }

    /// `ids` is (1, T); returns (1, channels, T).
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let mut xs = self.embedding.forward(ids)?.transpose(1, 2)?.contiguous()?;
        for (conv, norm) in &self.cnn {
            let h = norm.forward(&conv.forward(&xs)?.transpose(1, 2)?)?;
            xs = ops::leaky_relu(&h, TEXT_SLOPE)?
                .transpose(1, 2)?
                .contiguous()?;
        }
        self.lstm
            .forward(&xs.transpose(1, 2)?.contiguous()?)?
            .transpose(1, 2)?
            .contiguous()
    }
}
