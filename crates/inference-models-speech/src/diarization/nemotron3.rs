//! Nemotron-3 Diarization from the transformers layout (`config.json`, `processor_config.json`, `model.safetensors`):
//! stacked log-mel frames through a RoPE transformer, a subpixel upsampler and an 8-speaker head, run in chunks
//! behind the speaker cache.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use inference_audio::nemo::{NemoMel, NemoMelConfig};
use inference_quant::QuantMethod;
use inference_tensor::nn::{LayerNorm, Linear, Module, VarBuilder, layer_norm, linear, ops};
use inference_tensor::{D, DType, Device, Error, Result, Tensor};

use super::cache::{CacheConfig, Silence, SpeakerCache};
use super::{DEFAULT_THRESHOLD, Diarization, speaker_segments};

const LAYER_NORM_EPS: f64 = 1e-5;
const UPSAMPLE_KERNEL: usize = 3;

fn msg(e: impl std::fmt::Display) -> Error {
    Error::Msg(e.to_string())
}

#[derive(Debug, Clone, inference_json::FromJson)]
struct RopeParameters {
    rope_theta: f64,
}

#[derive(Debug, Clone, inference_json::FromJson)]
struct AudioConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_mel_bins: usize,
    max_position_embeddings: usize,
    subsampling_factor: usize,
    rope_parameters: RopeParameters,
}

#[derive(Debug, Clone, inference_json::FromJson)]
struct HeadConfig {
    audio_hidden_size: usize,
    hidden_size: usize,
    num_speakers: usize,
}

#[derive(Debug, Clone, inference_json::FromJson)]
struct StreamingConfig {
    speaker_cache_length: usize,
    speaker_cache_silence_frames_per_speaker: usize,
    prediction_score_threshold: f32,
    latest_frames_score_boost: f32,
    min_positive_scores_rate: f64,
    strong_boost_rate: f64,
    weak_boost_rate: f64,
}

/// A Nemotron-3 Diarization `config.json`: the encoder, the head, and the offline chunking around the speaker cache.
#[derive(Debug, Clone, inference_json::FromJson)]
pub struct Nemotron3Config {
    audio_config: AudioConfig,
    head_config: HeadConfig,
    streaming_config: StreamingConfig,
    chunk_length: usize,
    chunk_right_context: usize,
    fifo_length: usize,
    speaker_cache_update_period: usize,
}

#[derive(Debug, Clone, inference_json::FromJson)]
struct FeatureConfig {
    feature_size: usize,
    sampling_rate: u32,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    #[serde(default)]
    preemphasis: Option<f32>,
}

#[derive(Debug, Clone, inference_json::FromJson)]
struct ProcessorConfig {
    feature_extractor: FeatureConfig,
}

/// The files of a Nemotron-3 Diarization checkpoint.
#[derive(Debug, Clone)]
pub struct Nemotron3Files {
    pub config: PathBuf,
    pub processor_config: PathBuf,
    pub weights: Vec<PathBuf>,
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let text =
        std::fs::read_to_string(path).map_err(|e| msg(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| msg(format!("{}: {e}", path.display())))
}

/// A NeMo `TransformerEncoder` in transformers' layout: `subsampling_factor` stacked log-mel frames projected, pre-norm
/// layers with half-split RoPE over the first `rotary_dim` of each head (q and k normalized per head first, with
/// `qk_norm`), a final norm.
#[derive(Debug, Clone)]
pub struct NemoTransformerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_mel_bins: usize,
    pub subsampling_factor: usize,
    pub max_positions: usize,
    pub rope_theta: f64,
    pub rotary_dim: usize,
    pub qk_norm: bool,
}

impl NemoTransformerConfig {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

impl From<&AudioConfig> for NemoTransformerConfig {
    fn from(a: &AudioConfig) -> Self {
        Self {
            hidden_size: a.hidden_size,
            intermediate_size: a.intermediate_size,
            num_hidden_layers: a.num_hidden_layers,
            num_attention_heads: a.num_attention_heads,
            num_mel_bins: a.num_mel_bins,
            subsampling_factor: a.subsampling_factor,
            max_positions: a.max_position_embeddings,
            rope_theta: a.rope_parameters.rope_theta,
            rotary_dim: a.hidden_size / a.num_attention_heads,
            qk_norm: false,
        }
    }
}

/// A projection of [`NemoTransformer`]'s layers: dense, or a quantized one kept as a GGUF stores it.
pub enum NemoProjection {
    Dense(Linear),
    Quantized(Arc<dyn QuantMethod>),
}

impl NemoProjection {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(linear) => linear.forward(xs),
            Self::Quantized(layer) => layer.forward(xs),
        }
    }
}

/// The query, key and value projections: one with rows `[query | key | value]`, or three.
pub enum NemoQkv {
    Fused(NemoProjection),
    Split([NemoProjection; 3]),
}

/// One layer's projections, as [`NemoTransformer::with_projections`] takes them.
pub struct NemoLayerProjections {
    pub qkv: NemoQkv,
    pub o: NemoProjection,
    pub fc1: NemoProjection,
    pub fc2: NemoProjection,
}

struct Layer {
    norm1: LayerNorm,
    projections: NemoLayerProjections,
    q_norm: Option<LayerNorm>,
    k_norm: Option<LayerNorm>,
    norm2: LayerNorm,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let half = xs.dim(D::Minus1)? / 2;
    Tensor::cat(
        &[
            &xs.narrow(D::Minus1, half, half)?.neg()?,
            &xs.narrow(D::Minus1, 0, half)?,
        ],
        D::Minus1,
    )
}

impl Layer {
    // `projections` from the caller, or dense ones from `vb`
    fn new(
        cfg: &NemoTransformerConfig,
        vb: VarBuilder,
        projections: Option<NemoLayerProjections>,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let attn = vb.pp("self_attn");
        let projections = match projections {
            Some(projections) => projections,
            None => {
                let weight = |name: &str| attn.pp(name).get((h, h), "weight");
                // one [query | key | value] product: three of 512 columns ran a fifth slower a FLOP than one of 1536
                let qkv = Tensor::cat(
                    &[weight("q_proj")?, weight("k_proj")?, weight("v_proj")?],
                    0,
                )?;
                let dense = |linear: Result<Linear>| linear.map(NemoProjection::Dense);
                NemoLayerProjections {
                    qkv: NemoQkv::Fused(NemoProjection::Dense(Linear::new(qkv, None))),
                    o: dense(linear(h, h, attn.pp("o_proj")))?,
                    fc1: dense(linear(h, cfg.intermediate_size, vb.pp("mlp").pp("fc1")))?,
                    fc2: dense(linear(cfg.intermediate_size, h, vb.pp("mlp").pp("fc2")))?,
                }
            }
        };
        let head_norm = |name: &str| {
            cfg.qk_norm
                .then(|| layer_norm(cfg.head_dim(), LAYER_NORM_EPS, attn.pp(name)))
                .transpose()
        };
        Ok(Self {
            norm1: layer_norm(h, LAYER_NORM_EPS, vb.pp("layer_norm1"))?,
            projections,
            q_norm: head_norm("q_norm")?,
            k_norm: head_norm("k_norm")?,
            norm2: layer_norm(h, LAYER_NORM_EPS, vb.pp("layer_norm2"))?,
            heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
            rotary_dim: cfg.rotary_dim,
        })
    }

    fn heads(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        xs.reshape((b, t, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    // RoPE over the first `rotary_dim` of each head, the rest passed through
    fn rope(&self, x: Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let rotated = |x: &Tensor| -> Result<Tensor> {
            x.broadcast_mul(cos)? + rotate_half(x)?.broadcast_mul(sin)?
        };
        if self.rotary_dim == self.head_dim {
            return rotated(&x);
        }
        let pass = self.head_dim - self.rotary_dim;
        Tensor::cat(
            &[
                &rotated(&x.narrow(D::Minus1, 0, self.rotary_dim)?)?,
                &x.narrow(D::Minus1, self.rotary_dim, pass)?,
            ],
            D::Minus1,
        )
    }

    /// `cos` and `sin` are `(T, rotary_dim)`.
    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let normed = self.norm1.forward(xs)?;
        let head_normed = |x: Tensor, norm: &Option<LayerNorm>| match norm {
            Some(norm) => norm.forward(&x),
            None => Ok(x),
        };
        let p = &self.projections;
        let [q, k, v] = match &p.qkv {
            NemoQkv::Fused(qkv) => {
                let qkv = qkv.forward(&normed)?;
                let width = self.heads * self.head_dim;
                [0, 1, 2].map(|i| qkv.narrow(D::Minus1, i * width, width))
            }
            NemoQkv::Split(split) => [0, 1, 2].map(|i| split[i].forward(&normed)),
        };
        let q = head_normed(self.heads(&q?)?, &self.q_norm)?;
        let k = head_normed(self.heads(&k?)?, &self.k_norm)?;
        let q = self.rope(q, cos, sin)?;
        let k = self.rope(k, cos, sin)?;
        let v = self.heads(&v?)?;
        let scale = (self.head_dim as f64).powf(-0.5);
        // the CPU scales the queries, a tenth the values of the (heads, T, T) scores (2^-3 at head 64: the same
        // bits), and its matmul reads the transposed keys in place; a GPU keeps its scaled scores (its f16 matmul
        // inputs round small queries) and contiguous keys
        let scores = if q.device().is_cpu() {
            (q * scale)?.matmul(&k.t()?)?
        } else {
            (q.matmul(&k.t()?.contiguous()?)? * scale)?
        };
        let attn = ops::softmax_last_dim(&scores)?.matmul(&v)?;
        let attn = attn
            .transpose(1, 2)?
            .reshape((b, t, self.heads * self.head_dim))?;
        let xs = (xs + p.o.forward(&attn)?)?;
        let mlp = p
            .fc2
            .forward(&p.fc1.forward(&self.norm2.forward(&xs)?)?.gelu_erf()?)?;
        xs + mlp
    }
}

/// The encoder [`NemoTransformerConfig`] describes, its weights under transformers' names (`embedder.projection`,
/// `input_layer_norm`, `layers.N`, `layer_norm`).
pub struct NemoTransformer {
    // absent for an encoder over another encoder's states, which `forward` takes directly
    embedder: Option<Linear>,
    input_norm: LayerNorm,
    layers: Vec<Layer>,
    final_norm: LayerNorm,
    // (max positions, rotary_dim) RoPE tables, built once so a step slices rather than uploads
    cos: Tensor,
    sin: Tensor,
    config: NemoTransformerConfig,
    device: Device,
}

impl NemoTransformer {
    pub fn new(config: NemoTransformerConfig, vb: VarBuilder) -> Result<Self> {
        Self::build(config, vb, None)
    }

    /// [`Self::new`] with each layer's projections given (quantized ones, say), the rest from `vb`.
    pub fn with_projections(
        config: NemoTransformerConfig,
        vb: VarBuilder,
        projections: Vec<NemoLayerProjections>,
    ) -> Result<Self> {
        if projections.len() != config.num_hidden_layers {
            inference_tensor::bail!(
                "{} layers' projections for {} layers",
                projections.len(),
                config.num_hidden_layers
            );
        }
        Self::build(config, vb, Some(projections))
    }

    fn build(
        config: NemoTransformerConfig,
        vb: VarBuilder,
        projections: Option<Vec<NemoLayerProjections>>,
    ) -> Result<Self> {
        let c = &config;
        let mut projections = projections.map(Vec::into_iter);
        let layers = (0..c.num_hidden_layers)
            .map(|i| {
                let given = projections.as_mut().and_then(Iterator::next);
                Layer::new(c, vb.pp("layers").pp(i), given)
            })
            .collect::<Result<Vec<_>>>()?;
        let stacked = c.subsampling_factor * c.num_mel_bins;
        let projection = vb.pp("embedder").pp("projection");
        let embedder = projection
            .contains_tensor("weight")
            .then(|| -> Result<Linear> {
                Ok(Linear::new(
                    projection.get((c.hidden_size, stacked), "weight")?,
                    None,
                ))
            })
            .transpose()?;
        let (cos, sin) = rope_tables(
            c.max_positions,
            c.rotary_dim,
            c.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;
        Ok(Self {
            embedder,
            input_norm: layer_norm(c.hidden_size, LAYER_NORM_EPS, vb.pp("input_layer_norm"))?,
            layers,
            final_norm: layer_norm(c.hidden_size, LAYER_NORM_EPS, vb.pp("layer_norm"))?,
            cos,
            sin,
            device: vb.device().clone(),
            config,
        })
    }

    pub fn config(&self) -> &NemoTransformerConfig {
        &self.config
    }

    /// `(1, count, hidden)` for groups `first..first + count` of `(frames, mels)` row-major `features`: each
    /// `subsampling_factor` frames stacked, past the end zero.
    pub fn embed(&self, features: &[f32], first: usize, count: usize) -> Result<Tensor> {
        let c = &self.config;
        let row = c.subsampling_factor * c.num_mel_bins;
        let begin = (first * row).min(features.len());
        let mut stacked = features[begin..((first + count) * row).min(features.len())].to_vec();
        stacked.resize(count * row, 0.);
        let xs =
            Tensor::from_vec(stacked, (1, count, row), &self.device)?.to_dtype(self.cos.dtype())?;
        self.embedder
            .as_ref()
            .ok_or_else(|| msg("this encoder has no frame embedder"))?
            .forward(&xs)
    }

    /// The layers and the final norm over `(1, T, hidden)` embeddings, positions from 0.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let t = xs.dim(1)?;
        let cos = self.cos.narrow(0, 0, t)?;
        let sin = self.sin.narrow(0, 0, t)?;
        let mut hs = self.input_norm.forward(xs)?;
        for layer in &self.layers {
            hs = layer.forward(&hs, &cos, &sin)?;
        }
        self.final_norm.forward(&hs)
    }
}

pub struct Nemotron3Diarizer {
    mel: NemoMel,
    encoder: NemoTransformer,
    proj: Linear,
    upsample: Tensor,
    upsample_bias: Tensor,
    dense: Linear,
    out_proj: Linear,
    silence: Tensor,
    config: Nemotron3Config,
    frame_seconds: f64,
    device: Device,
}

impl Nemotron3Diarizer {
    pub fn load(files: &Nemotron3Files, device: &Device, dtype: DType) -> Result<Self> {
        // SAFETY: the weights are read-only for the model's life; a file changed underneath it is undefined
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files.weights, dtype, device)? };
        Self::from_configs(&files.config, &files.processor_config, vb)
    }

    /// The model `config` and `processor_config` describe, its weights from `vb`.
    pub fn from_configs(config: &Path, processor_config: &Path, vb: VarBuilder) -> Result<Self> {
        let config: Nemotron3Config = read_json(config)?;
        let processor: ProcessorConfig = read_json(processor_config)?;
        Self::new(config, &processor.feature_extractor, vb)
    }

    fn new(config: Nemotron3Config, fe: &FeatureConfig, vb: VarBuilder) -> Result<Self> {
        let mel = NemoMelConfig {
            sample_rate: fe.sampling_rate,
            n_fft: fe.n_fft,
            win_length: fe.win_length,
            hop_length: fe.hop_length,
            n_mels: fe.feature_size,
            preemphasis: fe.preemphasis,
            normalize: false,
            bf16_constants: false,
        };
        Self::with_features(config, mel, vb)
    }

    /// The diarizer a `config.json`'s text describes over features `mel` computes (another model's front end, as
    /// Nemotron 3.5 Transcribe's speaker branch reads its own), its weights from `vb`.
    pub fn from_config_json(config: &str, mel: NemoMelConfig, vb: VarBuilder) -> Result<Self> {
        let config: Nemotron3Config = serde_json::from_str(config).map_err(msg)?;
        Self::with_features(config, mel, vb)
    }

    fn with_features(config: Nemotron3Config, fe: NemoMelConfig, vb: VarBuilder) -> Result<Self> {
        let a = &config.audio_config;
        let h = &config.head_config;
        if fe.n_mels != a.num_mel_bins {
            return Err(msg(format!(
                "the features have {} mel bins but the encoder takes {}",
                fe.n_mels, a.num_mel_bins
            )));
        }
        let frame_seconds = fe.hop_length as f64 / f64::from(fe.sample_rate);
        let mel = NemoMel::new(fe);
        let tower = vb.pp("model").pp("audio_tower");
        let encoder = NemoTransformer::new(a.into(), tower)?;
        let up = vb.pp("model").pp("upsampler").pp("conv");
        let channels = h.hidden_size * a.subsampling_factor;
        let classifier = vb.pp("classifier");
        Ok(Self {
            mel,
            encoder,
            proj: linear(
                h.audio_hidden_size,
                h.hidden_size,
                vb.pp("model").pp("proj"),
            )?,
            upsample: up.get((channels, h.hidden_size, UPSAMPLE_KERNEL), "weight")?,
            upsample_bias: up.get(channels, "bias")?,
            dense: linear(h.hidden_size, h.hidden_size, classifier.pp("dense"))?,
            out_proj: linear(h.hidden_size, h.num_speakers, classifier.pp("out_proj"))?,
            silence: vb.get(a.hidden_size, "silence_embeds")?,
            frame_seconds,
            device: vb.device().clone(),
            config,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.mel.config().sample_rate
    }

    pub fn subsampling_factor(&self) -> usize {
        self.config.audio_config.subsampling_factor
    }

    pub fn num_speakers(&self) -> usize {
        self.config.head_config.num_speakers
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn cache_config(&self) -> CacheConfig {
        let s = &self.config.streaming_config;
        let speakers = self.num_speakers();
        let budget =
            (s.speaker_cache_length / speakers - s.speaker_cache_silence_frames_per_speaker) as f64;
        CacheConfig {
            cache_length: s.speaker_cache_length,
            // offline runs take the top-level sizes, as transformers' forward does without a cache of its own
            fifo_length: self.config.fifo_length,
            update_period: self.config.speaker_cache_update_period,
            silence_frames: s.speaker_cache_silence_frames_per_speaker,
            score_threshold: s.prediction_score_threshold,
            latest_boost: s.latest_frames_score_boost,
            num_speakers: speakers,
            min_positive_scores: (budget * s.min_positive_scores_rate).floor() as usize,
            strong_boosted: (budget * s.strong_boost_rate).floor() as usize,
            weak_boosted: (budget * s.weak_boost_rate).floor() as usize,
        }
    }

    /// `(frames, mels)` log-mel features of mono `pcm` at the model's rate, as rows, and the frame count.
    pub fn features(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        self.mel.features(pcm).map_err(msg)
    }

    // one step: the encoder over `(1, T, hidden)`, then the head, to `(T * subsampling, speakers)` logits
    fn step(&self, xs: &Tensor) -> Result<Tensor> {
        let t = xs.dim(1)?;
        let hs = self.proj.forward(&self.encoder.forward(xs)?)?;
        let up = hs
            .transpose(1, 2)?
            .conv1d(&self.upsample, UPSAMPLE_KERNEL / 2, 1, 1, 1)?
            .broadcast_add(&self.upsample_bias.reshape((1, (), 1))?)?
            .transpose(1, 2)?;
        let factor = self.config.audio_config.subsampling_factor;
        let up = up.reshape((t * factor, self.config.head_config.hidden_size))?;
        self.out_proj
            .forward(&self.dense.forward(&up.relu()?)?.relu()?)
    }

    /// `(frames, speakers)` speech logits of mono `pcm` at the model's rate, chunk by chunk behind the cache.
    pub fn logits(&self, pcm: &[f32]) -> Result<Tensor> {
        let (features, frames) = self.features(pcm)?;
        self.logits_of_features(&features, frames)?
            .narrow(0, 0, frames)
    }

    /// As [`Self::logits`], from `frames` rows of [`Self::features`] (or features computed alike), for every frame
    /// of the whole encoder steps they fill (the last one's zero-padded frames too).
    pub fn logits_of_features(&self, features: &[f32], frames: usize) -> Result<Tensor> {
        let factor = self.config.audio_config.subsampling_factor;
        let total = frames.div_ceil(factor);
        let (chunk, context) = (self.config.chunk_length, self.config.chunk_right_context);
        let mut cache =
            SpeakerCache::new(self.cache_config(), Silence::Learned(self.silence.clone()));
        let mut out = Vec::new();
        for start in (0..total).step_by(chunk) {
            let frames_here = chunk.min(total - start);
            let seen = (frames_here + context).min(total - start);
            let chunk_embeds = self.encoder.embed(features, start, seen)?;
            let cached = cache.embeds()?;
            let cached_len = cached.as_ref().map_or(Ok(0), |c| c.dim(0))?;
            let input = match cached {
                Some(c) => Tensor::cat(&[&c.unsqueeze(0)?, &chunk_embeds], 1)?,
                None => chunk_embeds,
            };
            let logits = self.step(&input)?;
            // the cache scores encoder frames: each one's 10 ms sigmoids, averaged
            let probs = ops::sigmoid(&logits.to_dtype(DType::F32)?)?
                .reshape((logits.dim(0)? / factor, factor, self.num_speakers()))?
                .mean(1)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            cache.update(&input.squeeze(0)?, &probs, 0, frames_here)?;
            out.push(logits.narrow(0, cached_len * factor, frames_here * factor)?);
        }
        Tensor::cat(&out, 0)
    }

    /// Who speaks when in mono `pcm` at `sample_rate`, resampled to the model's rate first.
    pub fn diarize(
        &self,
        pcm: &[f32],
        sample_rate: u32,
        threshold: Option<f32>,
    ) -> Result<Diarization> {
        let pcm =
            inference_audio::mel::resample(pcm, sample_rate, self.sample_rate()).map_err(msg)?;
        let probabilities = ops::sigmoid(&self.logits(&pcm)?)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let s = self.num_speakers();
        let segments = speaker_segments(
            &probabilities,
            s,
            self.frame_seconds,
            threshold.unwrap_or(DEFAULT_THRESHOLD),
        );
        Ok(Diarization {
            duration: pcm.len() as f64 / f64::from(self.sample_rate()),
            frame_seconds: self.frame_seconds,
            num_speakers: s,
            probabilities,
            segments,
        })
    }
}

// half-split RoPE: cos and sin of position * inv_freq, the frequencies repeated across both halves
fn rope_tables(
    positions: usize,
    dim: usize,
    theta: f64,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = dim / 2;
    let mut cos = Vec::with_capacity(positions * dim);
    let mut sin = Vec::with_capacity(positions * dim);
    for p in 0..positions {
        let row: Vec<f32> = (0..half)
            .map(|i| p as f32 * (1.0 / theta.powf(2.0 * i as f64 / dim as f64)) as f32)
            .collect();
        for _ in 0..2 {
            cos.extend(row.iter().map(|a| a.cos()));
            sin.extend(row.iter().map(|a| a.sin()));
        }
    }
    let table = |v: Vec<f32>| Tensor::from_vec(v, (positions, dim), device)?.to_dtype(dtype);
    Ok((table(cos)?, table(sin)?))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use inference_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};
    use inference_tensor::nn::{Linear, VarBuilder};
    use inference_tensor::quantized::{GgmlDType, QTensor};
    use inference_tensor::{DType, Device, Result, Tensor};

    use super::{
        NemoLayerProjections, NemoProjection, NemoQkv, NemoTransformer, NemoTransformerConfig,
    };

    const HIDDEN: usize = 64;
    const INTERMEDIATE: usize = 128;
    const LAYERS: usize = 2;
    const MELS: usize = 8;
    const STACK: usize = 2;
    const FRAMES: usize = 10;
    // Q8_0 weights against their dequantized copies, through two layers
    const Q8_TOLERANCE: f32 = 2e-2;

    fn config() -> NemoTransformerConfig {
        NemoTransformerConfig {
            hidden_size: HIDDEN,
            intermediate_size: INTERMEDIATE,
            num_hidden_layers: LAYERS,
            num_attention_heads: 4,
            num_mel_bins: MELS,
            subsampling_factor: STACK,
            max_positions: 64,
            rope_theta: 10_000.0,
            rotary_dim: HIDDEN / 4,
            qk_norm: false,
        }
    }

    fn weights() -> Result<HashMap<String, Tensor>> {
        let dev = Device::Cpu;
        let r = |shape: &[usize]| Tensor::randn(0f32, 0.1, shape, &dev);
        let mut w = HashMap::new();
        w.insert(
            "embedder.projection.weight".into(),
            r(&[HIDDEN, STACK * MELS])?,
        );
        for norm in ["input_layer_norm", "layer_norm"] {
            w.insert(
                format!("{norm}.weight"),
                Tensor::ones(HIDDEN, DType::F32, &dev)?,
            );
            w.insert(
                format!("{norm}.bias"),
                Tensor::zeros(HIDDEN, DType::F32, &dev)?,
            );
        }
        for l in 0..LAYERS {
            let p = |name: &str| format!("layers.{l}.{name}");
            for norm in ["layer_norm1", "layer_norm2"] {
                w.insert(
                    p(&format!("{norm}.weight")),
                    Tensor::ones(HIDDEN, DType::F32, &dev)?,
                );
                w.insert(
                    p(&format!("{norm}.bias")),
                    Tensor::zeros(HIDDEN, DType::F32, &dev)?,
                );
            }
            for q in ["q_proj", "k_proj", "v_proj"] {
                w.insert(p(&format!("self_attn.{q}.weight")), r(&[HIDDEN, HIDDEN])?);
            }
            w.insert(p("self_attn.o_proj.weight"), r(&[HIDDEN, HIDDEN])?);
            w.insert(p("self_attn.o_proj.bias"), r(&[HIDDEN])?);
            w.insert(p("mlp.fc1.weight"), r(&[INTERMEDIATE, HIDDEN])?);
            w.insert(p("mlp.fc1.bias"), r(&[INTERMEDIATE])?);
            w.insert(p("mlp.fc2.weight"), r(&[HIDDEN, INTERMEDIATE])?);
            w.insert(p("mlp.fc2.bias"), r(&[HIDDEN])?);
        }
        Ok(w)
    }

    fn run(encoder: &NemoTransformer) -> Result<Vec<f32>> {
        let features: Vec<f32> = (0..FRAMES * STACK * MELS)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
            .collect();
        encoder
            .forward(&encoder.embed(&features, 0, FRAMES)?)?
            .flatten_all()?
            .to_vec1::<f32>()
    }

    // each layer's projections from the weights, `make` turning a weight and bias into one
    fn projections(
        w: &HashMap<String, Tensor>,
        make: &dyn Fn(Tensor, Option<Tensor>) -> Result<NemoProjection>,
    ) -> Result<Vec<NemoLayerProjections>> {
        (0..LAYERS)
            .map(|l| {
                let get = |name: &str| w[&format!("layers.{l}.{name}")].clone();
                let qkv = Tensor::cat(
                    &["q_proj", "k_proj", "v_proj"].map(|q| get(&format!("self_attn.{q}.weight"))),
                    0,
                )?;
                Ok(NemoLayerProjections {
                    qkv: NemoQkv::Fused(make(qkv, None)?),
                    o: make(
                        get("self_attn.o_proj.weight"),
                        Some(get("self_attn.o_proj.bias")),
                    )?,
                    fc1: make(get("mlp.fc1.weight"), Some(get("mlp.fc1.bias")))?,
                    fc2: make(get("mlp.fc2.weight"), Some(get("mlp.fc2.bias")))?,
                })
            })
            .collect()
    }

    // a fused qkv (rows [q | k | v]) scores as the three projections it was cut from
    #[test]
    fn a_fused_qkv_encoder_matches_the_split_one() -> Result<()> {
        let w = weights()?;
        let vb = || VarBuilder::from_tensors(w.clone(), DType::F32, &Device::Cpu);
        let split = run(&NemoTransformer::new(config(), vb())?)?;
        let dense = |w: Tensor, b: Option<Tensor>| Ok(NemoProjection::Dense(Linear::new(w, b)));
        let fused = run(&NemoTransformer::with_projections(
            config(),
            vb(),
            projections(&w, &dense)?,
        )?)?;
        for (a, b) in fused.iter().zip(&split) {
            assert!((a - b).abs() <= 1e-5 * b.abs().max(1.0), "{a} vs {b}");
        }
        Ok(())
    }

    // Q8_0 projections stay within rounding of the dense encoder on their dequantized weights
    #[test]
    fn a_q8_0_encoder_scores_as_its_dequantized_weights() -> Result<()> {
        let w = weights()?;
        let vb = || VarBuilder::from_tensors(w.clone(), DType::F32, &Device::Cpu);
        let quantized = |w: Tensor, b: Option<Tensor>| -> Result<NemoProjection> {
            let layer = GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(QTensor::quantize(&w, GgmlDType::Q8_0)?),
                b,
            })?;
            Ok(NemoProjection::Quantized(Arc::new(layer)))
        };
        let dequantized = |w: Tensor, b: Option<Tensor>| -> Result<NemoProjection> {
            let w = QTensor::quantize(&w, GgmlDType::Q8_0)?.dequantize(&Device::Cpu)?;
            Ok(NemoProjection::Dense(Linear::new(w, b)))
        };
        let q8 = run(&NemoTransformer::with_projections(
            config(),
            vb(),
            projections(&w, &quantized)?,
        )?)?;
        let dense = run(&NemoTransformer::with_projections(
            config(),
            vb(),
            projections(&w, &dequantized)?,
        )?)?;
        let scale = dense.iter().fold(0f32, |m, v| m.max(v.abs()));
        let error = q8
            .iter()
            .zip(&dense)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            error <= Q8_TOLERANCE * scale,
            "max error {error} at scale {scale}"
        );
        Ok(())
    }
}
