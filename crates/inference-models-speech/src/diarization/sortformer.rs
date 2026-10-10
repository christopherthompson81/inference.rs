//! Streaming Sortformer v2 from its `.nemo`: a FastConformer (Parakeet's encoder) over each chunk's own mel slice
//! behind the speaker cache, a post-norm transformer and a sigmoid speaker head, at the encoder's 80 ms frames.

use std::path::Path;

use inference_audio::nemo::NemoMel;
use inference_tensor::nn::{LayerNorm, Linear, Module, VarBuilder, layer_norm, linear, ops};
use inference_tensor::{DType, Device, Error, Result, Tensor};
use serde::Deserialize;

use super::cache::{CacheConfig, Silence, SpeakerCache};
use super::{DEFAULT_THRESHOLD, Diarization, speaker_segments};
use crate::nemo::{NemoArchive, NemoEncoderConfig, NemoPreprocessorConfig, parakeet_encoder_name};
use crate::parakeet::{Encoder, relative_positions};

// the class a Sortformer `.nemo` restores to, by any module path; Nemotron-3's does too, over a transformer encoder
const SORTFORMER_CLASS: &str = ".SortformerEncLabelModel";
const ACTIVATION: &str = "relu";
const FASTCONFORMER_TARGET: &str = "ConformerEncoder";
const LAYER_NORM_EPS: f64 = 1e-5;

fn msg(e: impl std::fmt::Display) -> Error {
    Error::Msg(e.to_string())
}

#[derive(Debug, Clone, Deserialize)]
struct TransformerConfig {
    num_layers: usize,
    hidden_size: usize,
    inner_size: usize,
    num_attention_heads: usize,
    hidden_act: String,
    pre_ln: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct ModulesConfig {
    num_spks: usize,
    fc_d_model: usize,
    tf_d_model: usize,
    spkcache_len: usize,
    fifo_len: usize,
    chunk_len: usize,
    spkcache_update_period: usize,
    chunk_left_context: usize,
    chunk_right_context: usize,
    spkcache_sil_frames_per_spk: usize,
    pred_score_threshold: f32,
    scores_boost_latest: f32,
    sil_threshold: f32,
    strong_boost_rate: f64,
    weak_boost_rate: f64,
    min_pos_scores_rate: f64,
}

/// The parts of a Sortformer `model_config.yaml` inference reads.
#[derive(Debug, Clone, Deserialize)]
pub struct SortformerConfig {
    preprocessor: NemoPreprocessorConfig,
    encoder: NemoEncoderConfig,
    transformer_encoder: TransformerConfig,
    sortformer_modules: ModulesConfig,
}

#[derive(Deserialize)]
struct ModuleTarget {
    #[serde(rename = "_target_")]
    target: Option<String>,
}

#[derive(Deserialize)]
struct Probe {
    target: Option<String>,
    encoder: Option<ModuleTarget>,
}

/// Whether `archive` is a Streaming Sortformer: a diarization model over a FastConformer.
pub fn is_sortformer(archive: &NemoArchive) -> Result<bool> {
    let probe: Probe = archive.config()?;
    let encoder = probe.encoder.and_then(|e| e.target);
    Ok(probe.target.is_some_and(|t| t.ends_with(SORTFORMER_CLASS))
        && encoder.is_some_and(|e| e.ends_with(FASTCONFORMER_TARGET)))
}

impl SortformerConfig {
    /// A `model_config.yaml`'s text.
    pub fn from_yaml(text: &str) -> Result<Self> {
        serde_saphyr::from_str(text).map_err(msg)
    }
}

// NeMo's TransformerEncoderBlock with pre_ln false: attention, residual, norm, then feed-forward, residual, norm
struct PostNormLayer {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    norm1: LayerNorm,
    dense_in: Linear,
    dense_out: Linear,
    norm2: LayerNorm,
    heads: usize,
    head_dim: usize,
}

impl PostNormLayer {
    fn new(c: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        let h = c.hidden_size;
        let attn = vb.pp("first_sub_layer");
        let ff = vb.pp("second_sub_layer");
        Ok(Self {
            query: linear(h, h, attn.pp("query_net"))?,
            key: linear(h, h, attn.pp("key_net"))?,
            value: linear(h, h, attn.pp("value_net"))?,
            out: linear(h, h, attn.pp("out_projection"))?,
            norm1: layer_norm(h, LAYER_NORM_EPS, vb.pp("layer_norm_1"))?,
            dense_in: linear(h, c.inner_size, ff.pp("dense_in"))?,
            dense_out: linear(c.inner_size, h, ff.pp("dense_out"))?,
            norm2: layer_norm(h, LAYER_NORM_EPS, vb.pp("layer_norm_2"))?,
            heads: c.num_attention_heads,
            head_dim: h / c.num_attention_heads,
        })
    }

    fn heads(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        xs.reshape((b, t, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, h) = xs.dims3()?;
        // NeMo divides queries and keys each by the fourth root of the head size
        let scale = (self.head_dim as f64).powf(-0.25);
        let q = (self.heads(&self.query.forward(xs)?)? * scale)?;
        let k = (self.heads(&self.key.forward(xs)?)? * scale)?;
        let v = self.heads(&self.value.forward(xs)?)?;
        let attn = ops::softmax_last_dim(&q.matmul(&k.t()?)?)?.matmul(&v)?;
        let attn = attn.transpose(1, 2)?.reshape((b, t, h))?;
        let xs = self.norm1.forward(&(self.out.forward(&attn)? + xs)?)?;
        let ff = self
            .dense_out
            .forward(&self.dense_in.forward(&xs)?.relu()?)?;
        self.norm2.forward(&(ff + xs)?)
    }
}

pub struct SortformerDiarizer {
    mel: NemoMel,
    encoder: Encoder,
    proj: Linear,
    layers: Vec<PostNormLayer>,
    hidden: Linear,
    to_speakers: Linear,
    // relative positions for the longest step, which a shorter one takes the middle of
    positions: Tensor,
    longest_step: usize,
    config: SortformerConfig,
    subsampling: usize,
    frame_seconds: f64,
    device: Device,
}

impl SortformerDiarizer {
    pub fn load(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let archive = NemoArchive::open(path)?;
        if !is_sortformer(&archive)? {
            return Err(msg(format!(
                "`{}` is not a Streaming Sortformer `.nemo`",
                path.display()
            )));
        }
        let config: SortformerConfig = archive.config()?;
        let vb = archive.var_builder(parakeet_encoder_name, dtype, device)?;
        Self::new(config, vb)
    }

    pub fn new(config: SortformerConfig, vb: VarBuilder) -> Result<Self> {
        let p = &config.preprocessor;
        let t = &config.transformer_encoder;
        let m = &config.sortformer_modules;
        if t.pre_ln || t.hidden_act != ACTIVATION {
            return Err(msg(format!(
                "a transformer with pre_ln {} and `{}` is not Sortformer's",
                t.pre_ln, t.hidden_act
            )));
        }
        if p.features != config.encoder.feat_in {
            return Err(msg(format!(
                "the features have {} mel bins but the encoder takes {}",
                p.features, config.encoder.feat_in
            )));
        }
        let hop_length = p.hop_length();
        let mel = NemoMel::new(p.mel()?);
        let encoder = Encoder::new(&config.encoder.parakeet()?, vb.pp("encoder"))?;
        let modules = vb.pp("sortformer_modules");
        let layers = (0..t.num_layers)
            .map(|i| PostNormLayer::new(t, vb.pp("transformer_encoder").pp("layers").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let longest_step = m.spkcache_len
            + m.fifo_len
            + m.chunk_left_context
            + m.chunk_len
            + m.chunk_right_context;
        let positions =
            relative_positions(longest_step, encoder.hidden_size(), vb.dtype(), vb.device())?;
        let subsampling = config.encoder.subsampling_factor;
        Ok(Self {
            mel,
            proj: linear(m.fc_d_model, m.tf_d_model, modules.pp("encoder_proj"))?,
            hidden: linear(
                m.tf_d_model,
                m.tf_d_model,
                modules.pp("first_hidden_to_hidden"),
            )?,
            to_speakers: linear(
                m.tf_d_model,
                m.num_spks,
                modules.pp("single_hidden_to_spks"),
            )?,
            encoder,
            layers,
            positions,
            longest_step,
            subsampling,
            frame_seconds: (hop_length * subsampling) as f64 / f64::from(p.sample_rate),
            device: vb.device().clone(),
            config,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.preprocessor.sample_rate
    }

    pub fn num_speakers(&self) -> usize {
        self.config.sortformer_modules.num_spks
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn mel_bins(&self) -> usize {
        self.config.preprocessor.features
    }

    /// `(frames, mels)` row-major log-mel features of mono `pcm` at the model's rate, and the frame count.
    pub fn features(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        self.mel.features(pcm).map_err(msg)
    }

    fn cache(&self) -> Result<SpeakerCache> {
        let m = &self.config.sortformer_modules;
        let speakers = m.num_spks;
        let budget = (m.spkcache_len / speakers - m.spkcache_sil_frames_per_spk) as f64;
        let config = CacheConfig {
            cache_length: m.spkcache_len,
            fifo_length: m.fifo_len,
            update_period: m.spkcache_update_period,
            silence_frames: m.spkcache_sil_frames_per_spk,
            score_threshold: m.pred_score_threshold,
            latest_boost: m.scores_boost_latest,
            num_speakers: speakers,
            min_positive_scores: (budget * m.min_pos_scores_rate).floor() as usize,
            strong_boosted: (budget * m.strong_boost_rate).floor() as usize,
            weak_boosted: (budget * m.weak_boost_rate).floor() as usize,
        };
        let silence = Silence::Popped {
            threshold: m.sil_threshold,
            mean: Tensor::zeros(m.fc_d_model, DType::F32, &self.device)?,
            frames: 0,
        };
        Ok(SpeakerCache::new(config, silence))
    }

    // one step over `(1, T, fc_d_model)` pre-encode frames to `(T, speakers)` speech probabilities
    fn step(&self, xs: &Tensor) -> Result<Vec<f32>> {
        let t = xs.dim(1)?;
        let positions = self.positions.narrow(1, self.longest_step - t, 2 * t - 1)?;
        let mut hs = self.proj.forward(&self.encoder.encode(xs, &positions)?)?;
        for layer in &self.layers {
            hs = layer.forward(&hs)?;
        }
        let logits = self
            .to_speakers
            .forward(&self.hidden.forward(&hs.relu()?)?.relu()?)?;
        ops::sigmoid(&logits.to_dtype(DType::F32)?)?
            .flatten_all()?
            .to_vec1::<f32>()
    }

    /// `(frames, speakers)` row-major speech probabilities of mono `pcm` at the model's rate, as NeMo's
    /// `forward_streaming` runs it: each chunk's mel slice with its context, behind the cache.
    pub fn probabilities(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        let m = &self.config.sortformer_modules;
        let (features, frames) = self.features(pcm)?;
        let mels = self.config.preprocessor.features;
        let sub = self.subsampling;
        let s = m.num_spks;
        let mut cache = self.cache()?;
        let mut out = Vec::with_capacity(frames.div_ceil(sub) * s);
        let mut start = 0;
        while start < frames {
            let left = (m.chunk_left_context * sub).min(start);
            let end = (start + m.chunk_len * sub).min(frames);
            let right = (m.chunk_right_context * sub).min(frames - end);
            let slice = &features[(start - left) * mels..(end + right) * mels];
            let xs = Tensor::from_slice(slice, (1, slice.len() / mels, mels), &self.device)?
                .to_dtype(self.positions.dtype())?;
            let chunk = self.encoder.subsample(&xs)?.squeeze(0)?;
            let (left, right) = (left.div_ceil(sub), right.div_ceil(sub));
            let chunk_frames = chunk.dim(0)? - left - right;
            let (input, cached) = match cache.embeds()? {
                Some(cached) => (Tensor::cat(&[&cached, &chunk], 0)?, cached.dim(0)?),
                None => (chunk, 0),
            };
            let probs = self.step(&input.unsqueeze(0)?)?;
            let first = cached + left;
            out.extend_from_slice(&probs[first * s..(first + chunk_frames) * s]);
            cache.update(&input, &probs, left, chunk_frames)?;
            start = end;
        }
        Ok(out)
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
        let probabilities = self.probabilities(&pcm)?;
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
