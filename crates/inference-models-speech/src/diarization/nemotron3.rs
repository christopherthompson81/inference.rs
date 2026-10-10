//! Nemotron-3 Diarization from the transformers layout (`config.json`, `processor_config.json`, `model.safetensors`):
//! stacked log-mel frames through a RoPE transformer, a subpixel upsampler and an 8-speaker head, run in chunks
//! behind the speaker cache.

use std::path::{Path, PathBuf};

use inference_audio::nemo::{NemoMel, NemoMelConfig};
use inference_tensor::nn::{LayerNorm, Linear, Module, VarBuilder, layer_norm, linear, ops};
use inference_tensor::{D, DType, Device, Error, Result, Tensor};
use serde::Deserialize;

use super::cache::{CacheConfig, Silence, SpeakerCache};
use super::{DEFAULT_THRESHOLD, Diarization, speaker_segments};

const LAYER_NORM_EPS: f64 = 1e-5;
const UPSAMPLE_KERNEL: usize = 3;

fn msg(e: impl std::fmt::Display) -> Error {
    Error::Msg(e.to_string())
}

#[derive(Debug, Clone, Deserialize)]
struct RopeParameters {
    rope_theta: f64,
}

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
struct HeadConfig {
    audio_hidden_size: usize,
    hidden_size: usize,
    num_speakers: usize,
}

#[derive(Debug, Clone, Deserialize)]
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
#[derive(Debug, Clone, Deserialize)]
pub struct Nemotron3Config {
    audio_config: AudioConfig,
    head_config: HeadConfig,
    streaming_config: StreamingConfig,
    chunk_length: usize,
    chunk_right_context: usize,
    fifo_length: usize,
    speaker_cache_update_period: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct FeatureConfig {
    feature_size: usize,
    sampling_rate: u32,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    #[serde(default)]
    preemphasis: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
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

struct Layer {
    norm1: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    heads: usize,
    head_dim: usize,
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
    fn new(cfg: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let attn = vb.pp("self_attn");
        let no_bias = |name: &str| -> Result<Linear> {
            Ok(Linear::new(attn.pp(name).get((h, h), "weight")?, None))
        };
        Ok(Self {
            norm1: layer_norm(h, LAYER_NORM_EPS, vb.pp("layer_norm1"))?,
            q: no_bias("q_proj")?,
            k: no_bias("k_proj")?,
            v: no_bias("v_proj")?,
            o: linear(h, h, attn.pp("o_proj"))?,
            norm2: layer_norm(h, LAYER_NORM_EPS, vb.pp("layer_norm2"))?,
            fc1: linear(h, cfg.intermediate_size, vb.pp("mlp").pp("fc1"))?,
            fc2: linear(cfg.intermediate_size, h, vb.pp("mlp").pp("fc2"))?,
            heads: cfg.num_attention_heads,
            head_dim: h / cfg.num_attention_heads,
        })
    }

    fn heads(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        xs.reshape((b, t, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// `cos` and `sin` are `(T, head_dim)`.
    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let normed = self.norm1.forward(xs)?;
        let rope = |x: Tensor| -> Result<Tensor> {
            x.broadcast_mul(cos)? + rotate_half(&x)?.broadcast_mul(sin)?
        };
        let q = rope(self.heads(&self.q.forward(&normed)?)?)?;
        let k = rope(self.heads(&self.k.forward(&normed)?)?)?;
        let v = self.heads(&self.v.forward(&normed)?)?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let scores = (q.matmul(&k.t()?.contiguous()?)? * scale)?;
        let attn = ops::softmax_last_dim(&scores)?.matmul(&v)?;
        let attn = attn
            .transpose(1, 2)?
            .reshape((b, t, self.heads * self.head_dim))?;
        let xs = (xs + self.o.forward(&attn)?)?;
        let mlp = self
            .fc2
            .forward(&self.fc1.forward(&self.norm2.forward(&xs)?)?.gelu_erf()?)?;
        xs + mlp
    }
}

pub struct Nemotron3Diarizer {
    mel: NemoMel,
    embedder: Linear,
    input_norm: LayerNorm,
    layers: Vec<Layer>,
    final_norm: LayerNorm,
    proj: Linear,
    upsample: Tensor,
    upsample_bias: Tensor,
    dense: Linear,
    out_proj: Linear,
    silence: Tensor,
    // (max positions, head_dim) RoPE tables, built once so a step slices rather than uploads
    cos: Tensor,
    sin: Tensor,
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
        let a = &config.audio_config;
        let h = &config.head_config;
        if fe.feature_size != a.num_mel_bins {
            return Err(msg(format!(
                "the features have {} mel bins but the encoder takes {}",
                fe.feature_size, a.num_mel_bins
            )));
        }
        let mel = NemoMel::new(NemoMelConfig {
            sample_rate: fe.sampling_rate,
            n_fft: fe.n_fft,
            win_length: fe.win_length,
            hop_length: fe.hop_length,
            n_mels: fe.feature_size,
            preemphasis: fe.preemphasis,
            normalize: false,
        });
        let tower = vb.pp("model").pp("audio_tower");
        let layers = (0..a.num_hidden_layers)
            .map(|i| Layer::new(a, tower.pp("layers").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let stacked = a.subsampling_factor * a.num_mel_bins;
        let embedder = Linear::new(
            tower
                .pp("embedder")
                .pp("projection")
                .get((a.hidden_size, stacked), "weight")?,
            None,
        );
        let up = vb.pp("model").pp("upsampler").pp("conv");
        let channels = h.hidden_size * a.subsampling_factor;
        let head_dim = a.hidden_size / a.num_attention_heads;
        let (cos, sin) = rope_tables(
            a.max_position_embeddings,
            head_dim,
            a.rope_parameters.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;
        let classifier = vb.pp("classifier");
        Ok(Self {
            mel,
            embedder,
            input_norm: layer_norm(a.hidden_size, LAYER_NORM_EPS, tower.pp("input_layer_norm"))?,
            layers,
            final_norm: layer_norm(a.hidden_size, LAYER_NORM_EPS, tower.pp("layer_norm"))?,
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
            cos,
            sin,
            frame_seconds: fe.hop_length as f64 / f64::from(fe.sampling_rate),
            device: vb.device().clone(),
            config,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.mel.config().sample_rate
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

    // (1, count, hidden) for groups `first..first + count`: each `subsampling` frames stacked, past the end zero
    fn embed(&self, features: &[f32], first: usize, count: usize) -> Result<Tensor> {
        let a = &self.config.audio_config;
        let row = a.subsampling_factor * a.num_mel_bins;
        let begin = (first * row).min(features.len());
        let mut stacked = features[begin..((first + count) * row).min(features.len())].to_vec();
        stacked.resize(count * row, 0.);
        let xs =
            Tensor::from_vec(stacked, (1, count, row), &self.device)?.to_dtype(self.cos.dtype())?;
        self.embedder.forward(&xs)
    }

    // one step: the transformer over `(1, T, hidden)`, then the head, to `(T * subsampling, speakers)` logits
    fn step(&self, xs: &Tensor) -> Result<Tensor> {
        let t = xs.dim(1)?;
        let cos = self.cos.narrow(0, 0, t)?;
        let sin = self.sin.narrow(0, 0, t)?;
        let mut hs = self.input_norm.forward(xs)?;
        for layer in &self.layers {
            hs = layer.forward(&hs, &cos, &sin)?;
        }
        let hs = self.proj.forward(&self.final_norm.forward(&hs)?)?;
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
        let factor = self.config.audio_config.subsampling_factor;
        let total = frames.div_ceil(factor);
        let (chunk, context) = (self.config.chunk_length, self.config.chunk_right_context);
        let mut cache =
            SpeakerCache::new(self.cache_config(), Silence::Learned(self.silence.clone()));
        let mut out = Vec::new();
        for start in (0..total).step_by(chunk) {
            let frames_here = chunk.min(total - start);
            let seen = (frames_here + context).min(total - start);
            let chunk_embeds = self.embed(&features, start, seen)?;
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
        Tensor::cat(&out, 0)?.narrow(0, 0, frames)
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
