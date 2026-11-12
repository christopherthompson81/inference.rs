//! Nemotron 3.5 Transcribe: a Transformer speech encoder and the Nemotron-3 Diarization model on one log-mel, the
//! thresholded speaker activities selecting fixed speaker embeddings added to the encoder states, a linear projector,
//! and a Nemotron-H decoder whose `<|audio|>` positions take the projected audio. Read from NVIDIA's NeMo checkpoint.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub mod inputs_processor;
pub mod transcription;
pub mod word_timestamps;

use std::collections::HashMap;

use inference_audio::nemo::NemoMelConfig;
use inference_models_speech::diarization::{
    NemoLayerProjections, NemoProjection, NemoQkv, NemoTransformer, NemoTransformerConfig,
    Nemotron3Diarizer,
};
use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{LayerNorm, Linear, Module, VarBuilder, layer_norm, linear};
use inference_tensor::{DType, Device, Result, Tensor};
use serde::Serialize;

use crate::granite::GraniteMoeHybrid;
use crate::serde_default_fn;
use crate::{
    amoe::AnyMoeBaseModelMixin,
    kv_cache::EitherCache,
    media_inputs::media::place_token_features,
    model::{IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata, NormalModel},
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
};

pub const MODEL: &str = "Nemotron 3.5 Transcribe";
// NeMo's prefixes
pub const LLM: &str = "llm";
const ASR: &str = "perception.encoder.asr_encoder";
const DIARIZATION: &str = "perception.encoder.diarization_model";
const ASR_NORM: &str = "perception.encoder.asr_norm";
const SPEAKER_NORM: &str = "perception.encoder.diar_norm";
const PROJECTOR: &str = "perception.proj";
// NVIDIA's word timestamp head, beside `perception`
const WORD_TIMESTAMPS: &str = "ctc_timestamp.decoder";
// frames of the head whose full-vocabulary logits are held at once
const LOGIT_ROWS: usize = 256;
const LAYER_NORM_EPS: f64 = 1e-5;
// the encoder runs on chunks of this many frames, each seeing this many more on either side
const CHUNK_FRAMES: usize = 500;
const CHUNK_CONTEXT: usize = 50;
/// The most positions one encoder run sees: a chunk and its context on both sides.
pub const ENCODER_WINDOW: usize = CHUNK_FRAMES + 2 * CHUNK_CONTEXT;
/// The speaker branch's parameters (its 99.3 M diarization model) and the projector's, for sizing.
pub const SPEAKER_BRANCH_PARAMS: usize = 103_000_000;
const SPEAKER_TABLE_BASE: f64 = 10_000.0;
// NeMo `TransformerEncoder` blocks to transformers' names
const BLOCK_RENAMES: [(&str, &str); 8] = [
    ("norm1", "layer_norm1"),
    ("norm2", "layer_norm2"),
    ("attn.out_proj", "self_attn.o_proj"),
    ("attn.q_norm", "self_attn.q_norm"),
    ("attn.k_norm", "self_attn.k_norm"),
    ("ffn.net.0", "mlp.fc1"),
    ("ffn.net.3", "mlp.fc2"),
    ("attn.w_qkv", "self_attn.qkv"),
];
// a block's projections, which the speech encoder takes as linears of their own (quantized when the GGUF is)
const BLOCK_PROJECTIONS: [&str; 4] = ["attn.w_qkv", "attn.out_proj", "ffn.net.0", "ffn.net.3"];
const ENCODER_RENAMES: [(&str, &str); 3] = [
    ("pre_encode.proj", "embedder.projection"),
    ("embed_norm", "input_layer_norm"),
    ("final_norm", "layer_norm"),
];
// the diarization head's NeMo modules, to transformers' names
const DIARIZATION_HEAD: [(&str, &str); 4] = [
    ("sortformer_modules.encoder_proj", "model.proj"),
    (
        "sortformer_modules.subpixel_upsample",
        "model.upsampler.conv",
    ),
    (
        "sortformer_modules.first_hidden_to_hidden",
        "classifier.dense",
    ),
    (
        "sortformer_modules.single_hidden_to_spks",
        "classifier.out_proj",
    ),
];
const SILENCE: (&str, &str) = ("sortformer_modules.learnable_sil_emb", "silence_embeds");

serde_default_fn!(f32, default_activity_threshold, 0.5);
serde_default_fn!(f64, default_partial_rotary, 1.0);

#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
    #[serde(default = "default_partial_rotary")]
    pub partial_rotary_factor: f64,
}

#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct AsrConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_mel_bins: usize,
    pub max_position_embeddings: usize,
    pub subsampling_factor: usize,
    pub rope_parameters: RopeParameters,
}

#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct AudioConfig {
    pub asr_config: AsrConfig,
    #[serde(default = "default_activity_threshold")]
    pub speaker_activity_threshold: f32,
}

/// The word timestamp head: a transformer over the speech encoder's states, upsampled, and a CTC classifier.
#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct WordTimestampConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_parameters: RopeParameters,
    pub residual_scale: f64,
    pub upsampling_factor: usize,
    pub vocab_size: usize,
    pub blank_token_id: u32,
}

impl WordTimestampConfig {
    fn encoder(&self) -> NemoTransformerConfig {
        let head_dim = self.hidden_size / self.num_attention_heads;
        NemoTransformerConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            // it reads encoder states, not log-mel frames
            num_mel_bins: 0,
            subsampling_factor: 1,
            max_positions: self.max_position_embeddings,
            rope_theta: self.rope_parameters.rope_theta,
            rotary_dim: (head_dim as f64 * self.rope_parameters.partial_rotary_factor) as usize,
            qk_norm: true,
        }
    }
}

#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct Config {
    pub text_config: crate::nemotron_h::Config,
    pub audio_config: AudioConfig,
    pub audio_token_id: u32,
    #[serde(default)]
    pub word_timestamp_config: Option<WordTimestampConfig>,
}

impl Config {
    pub fn encoder(&self) -> NemoTransformerConfig {
        let a = &self.audio_config.asr_config;
        let head_dim = a.hidden_size / a.num_attention_heads;
        NemoTransformerConfig {
            hidden_size: a.hidden_size,
            intermediate_size: a.intermediate_size,
            num_hidden_layers: a.num_hidden_layers,
            num_attention_heads: a.num_attention_heads,
            num_mel_bins: a.num_mel_bins,
            subsampling_factor: a.subsampling_factor,
            max_positions: a.max_position_embeddings,
            rope_theta: a.rope_parameters.rope_theta,
            rotary_dim: (head_dim as f64 * a.rope_parameters.partial_rotary_factor) as usize,
            qk_norm: true,
        }
    }
}

/// The diarization model's own config, inside the checkpoint's.
fn diarization_config(config: &str) -> Result<String> {
    let value: serde_json::Value =
        serde_json::from_str(config).map_err(inference_tensor::Error::msg)?;
    let diarization = &value["audio_config"]["diarization_config"];
    if diarization.is_null() {
        inference_tensor::bail!("{MODEL} config has no `audio_config.diarization_config`");
    }
    Ok(diarization.to_string())
}

/// The log-mel Transcribe's processor computes (NeMo's, its filters and window rounded to bfloat16).
pub fn mel_config(num_mel_bins: usize) -> NemoMelConfig {
    NemoMelConfig {
        sample_rate: inputs_processor::SAMPLE_RATE,
        n_fft: inputs_processor::N_FFT,
        win_length: inputs_processor::WIN_LENGTH,
        hop_length: inputs_processor::HOP_LENGTH,
        n_mels: num_mel_bins,
        preemphasis: Some(inputs_processor::PREEMPHASIS),
        normalize: false,
        bf16_constants: true,
        reflect_pad: false,
    }
}

/// One clip's `(frames, mels)` log-mel: as computed (the speaker branch reads it) and normalized per feature over its
/// frames (the speech encoder reads that).
pub struct AudioClip {
    pub raw: Vec<f32>,
    pub normalized: Vec<f32>,
    pub frames: usize,
}

pub struct NemotronTranscribeSpecificArgs {
    /// The clips whose placeholders this step holds, in prompt order.
    pub audio: Vec<AudioClip>,
}

fn tower_tensor(vb: &ShardedVarBuilder, name: &str, dtype: DType) -> Result<Tensor> {
    vb.get_unchecked_dtype(name, dtype)
}

// a quantized GGUF weight stays as it is stored; a dense one (or one loaded elsewhere) comes back in the tower's dtype,
// a kernel-1 convolution's trailing dimension dropped
fn tower_projection(
    vb: &ShardedVarBuilder,
    [in_dim, out_dim]: [usize; 2],
    dtype: DType,
    device: &Device,
) -> Result<NemoProjection> {
    let bias = || {
        vb.contains_tensor("bias")
            .then(|| tower_tensor(vb, "bias", dtype))
            .transpose()
    };
    let dense = |w: Tensor, b: Option<Tensor>| -> Result<NemoProjection> {
        let w = if w.rank() == 3 { w.squeeze(2)? } else { w };
        let to = |t: Tensor| t.to_dtype(dtype)?.to_device(device);
        Ok(NemoProjection::Dense(Linear::new(
            to(w)?,
            b.map(to).transpose()?,
        )))
    };
    // a checkpoint's tensors are all dense
    if vb.weight_source().is_none() {
        return dense(tower_tensor(vb, "weight", dtype)?, bias()?);
    }
    let layer = inference_quant::linear_b(
        in_dim,
        out_dim,
        vb.contains_tensor("bias"),
        &None,
        vb.clone(),
    )?;
    match layer.unquant_weight_bias() {
        Some((w, b)) => dense(w, b),
        None if !layer.dtype_and_device().1.same_device(device) => {
            dense(layer.dequantize_w()?, bias()?)
        }
        None => Ok(NemoProjection::Quantized(layer)),
    }
}

// where an encoder's tensors go: `ours` before each renamed one, its blocks' `skip` modules left out
struct EncoderNames<'a> {
    ours: &'a str,
    skip: &'a [&'a str],
}

/// A NeMo `TransformerEncoder`'s tensors under transformers' names, its fused qkv split in three.
fn encoder_weights(
    vb: &ShardedVarBuilder,
    layers: usize,
    names: EncoderNames<'_>,
    dtype: DType,
    weights: &mut HashMap<String, Tensor>,
) -> Result<()> {
    let join = |name: &str| match names.ours {
        "" => name.to_string(),
        prefix => format!("{prefix}.{name}"),
    };
    for (theirs, renamed) in ENCODER_RENAMES {
        for suffix in ["weight", "bias"] {
            let name = format!("{theirs}.{suffix}");
            if vb.contains_tensor(&name) {
                weights.insert(
                    join(&format!("{renamed}.{suffix}")),
                    tower_tensor(vb, &name, dtype)?,
                );
            }
        }
    }
    for layer in 0..layers {
        let block = vb.pp("layers").pp(layer);
        for (theirs, renamed) in BLOCK_RENAMES {
            if names.skip.contains(&theirs) {
                continue;
            }
            for suffix in ["weight", "bias"] {
                let name = format!("{theirs}.{suffix}");
                if !block.contains_tensor(&name) {
                    continue;
                }
                let tensor = tower_tensor(&block, &name, dtype)?;
                let ours = |module: &str| join(&format!("layers.{layer}.{module}.{suffix}"));
                if renamed == "self_attn.qkv" {
                    // rows `[query | key | value]`
                    let rows = tensor.dim(0)? / 3;
                    for (i, projection) in ["q_proj", "k_proj", "v_proj"].into_iter().enumerate() {
                        weights.insert(
                            ours(&format!("self_attn.{projection}")),
                            tensor.narrow(0, i * rows, rows)?,
                        );
                    }
                } else {
                    weights.insert(ours(renamed), tensor);
                }
            }
        }
    }
    Ok(())
}

fn diarization_weights(
    vb: &ShardedVarBuilder,
    layers: usize,
    dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    let mut weights = HashMap::new();
    encoder_weights(
        &vb.pp("encoder"),
        layers,
        EncoderNames {
            ours: "model.audio_tower",
            skip: &[],
        },
        dtype,
        &mut weights,
    )?;
    for (theirs, ours) in DIARIZATION_HEAD {
        for suffix in ["weight", "bias"] {
            weights.insert(
                format!("{ours}.{suffix}"),
                tower_tensor(vb, &format!("{theirs}.{suffix}"), dtype)?,
            );
        }
    }
    weights.insert(SILENCE.1.to_string(), tower_tensor(vb, SILENCE.0, dtype)?);
    Ok(weights)
}

/// `(speakers, hidden)` sinusoids: speaker `p`'s row `sin`/`cos` of `p * 10000^(-2i / hidden)` interleaved.
fn speaker_table(speakers: usize, hidden: usize, device: &Device) -> Result<Tensor> {
    let mut table = vec![0f32; speakers * hidden];
    for p in 0..speakers {
        for i in (0..hidden).step_by(2) {
            let angle = p as f64 * (-(SPEAKER_TABLE_BASE.ln() / hidden as f64) * i as f64).exp();
            table[p * hidden + i] = angle.sin() as f32;
            if i + 1 < hidden {
                table[p * hidden + i + 1] = angle.cos() as f32;
            }
        }
    }
    Tensor::from_vec(table, (speakers, hidden), device)
}

/// The word timestamp head: the speech encoder's states before `asr_norm`, upsampled by a transposed convolution, a
/// small transformer with a scaled residual, and a classifier over the alignment tokenizer's pieces and the blank. Its
/// matrices run as its file stores them: quantized from a GGUF that holds them quantized.
struct WordTimestampHead {
    // the transposed convolution's taps: head frame `j` of each encoder step is tap `j` of the step
    taps: Vec<NemoProjection>,
    encoder: NemoTransformer,
    residual_scale: f64,
    classifier: NemoProjection,
}

impl WordTimestampHead {
    fn new(
        cfg: &WordTimestampConfig,
        vb: &ShardedVarBuilder,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let vb = vb.pp(WORD_TIMESTAMPS);
        let (hidden, inter) = (cfg.hidden_size, cfg.intermediate_size);
        let matrix =
            |vb: &ShardedVarBuilder, dims: [usize; 2]| tower_projection(vb, dims, dtype, device);
        // `(in, out, kernel)`, kernel and stride the upsampling factor
        let upsample = vb.pp("upsample");
        let weight = tower_tensor(&upsample, "weight", dtype)?;
        let bias = tower_tensor(&upsample, "bias", dtype)?.to_device(device)?;
        let taps = (0..cfg.upsampling_factor)
            .map(|j| {
                let tap = weight.narrow(2, j, 1)?.squeeze(2)?.t()?.contiguous()?;
                Ok(NemoProjection::Dense(Linear::new(
                    tap.to_device(device)?,
                    Some(bias.clone()),
                )))
            })
            .collect::<Result<Vec<_>>>()?;
        let transformer = vb.pp("transformer");
        let mut weights = HashMap::new();
        encoder_weights(
            &transformer,
            cfg.num_hidden_layers,
            EncoderNames {
                ours: "",
                skip: &BLOCK_PROJECTIONS,
            },
            dtype,
            &mut weights,
        )?;
        let projections = (0..cfg.num_hidden_layers)
            .map(|layer| {
                let block = transformer.pp("layers").pp(layer);
                let [qkv, o, fc1, fc2] = BLOCK_PROJECTIONS;
                Ok(NemoLayerProjections {
                    qkv: NemoQkv::Fused(matrix(&block.pp(qkv), [hidden, 3 * hidden])?),
                    o: matrix(&block.pp(o), [hidden, hidden])?,
                    fc1: matrix(&block.pp(fc1), [hidden, inter])?,
                    fc2: matrix(&block.pp(fc2), [inter, hidden])?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let encoder = NemoTransformer::with_projections(
            cfg.encoder(),
            VarBuilder::from_tensors(weights, dtype, device),
            projections,
        )?;
        Ok(Self {
            taps,
            encoder,
            residual_scale: cfg.residual_scale,
            classifier: matrix(&vb.pp("decoder_layers").pp(0), [hidden, cfg.vocab_size])?,
        })
    }

    // one encoder window's `(steps, hidden)` states to its `(steps * upsampling, classes)` log-probabilities
    fn log_probs(&self, states: &Tensor, classes: &Tensor) -> Result<Tensor> {
        let xs = states.unsqueeze(0)?;
        let (_, steps, hidden) = xs.dims3()?;
        let taps = self
            .taps
            .iter()
            .map(|tap| tap.forward(&xs))
            .collect::<Result<Vec<_>>>()?;
        let frames = steps * taps.len();
        let upsampled = Tensor::stack(&taps, 2)?.reshape((1, frames, hidden))?;
        let hidden = (&upsampled + (self.encoder.forward(&upsampled)? * self.residual_scale)?)?;
        // the full vocabulary's logits only a block of frames at a time
        let mut rows = Vec::new();
        for first in (0..frames).step_by(LOGIT_ROWS) {
            let block = hidden.narrow(1, first, LOGIT_ROWS.min(frames - first))?;
            let logits = self
                .classifier
                .forward(&block)?
                .squeeze(0)?
                .to_dtype(DType::F32)?;
            let max = logits.max_keepdim(inference_tensor::D::Minus1)?;
            let normalizer = (logits
                .broadcast_sub(&max)?
                .exp()?
                .sum_keepdim(inference_tensor::D::Minus1)?
                .log()?
                + max)?;
            rows.push(
                logits
                    .index_select(classes, 1)?
                    .broadcast_sub(&normalizer)?,
            );
        }
        Tensor::cat(&rows, 0)
    }
}

/// The speech encoder and the speaker branch beside it, and the projector: in f32 on the CPU, as transformers keeps
/// them, and in the model's dtype on an accelerator, where the f32 copy would take 1.4 GiB more.
struct AudioTower {
    encoder: NemoTransformer,
    diarizer: Nemotron3Diarizer,
    asr_norm: LayerNorm,
    speaker_norm: LayerNorm,
    speaker_table: Tensor,
    projector: Linear,
    activity_threshold: f32,
    subsampling: usize,
    // absent from a checkpoint or GGUF without it
    word_timestamps: Option<(WordTimestampHead, usize)>,
}

// the windows the encoders run on: `(first, start, end, last)`, steps `start..end` of each, seen with `first..last`
fn encoder_windows(steps: usize) -> impl Iterator<Item = (usize, usize, usize, usize)> {
    (0..steps).step_by(CHUNK_FRAMES).map(move |start| {
        let end = (start + CHUNK_FRAMES).min(steps);
        let first = start.saturating_sub(CHUNK_CONTEXT);
        (first, start, end, (end + CHUNK_CONTEXT).min(steps))
    })
}

impl AudioTower {
    fn new(cfg: &Config, config: &str, vb: &ShardedVarBuilder, device: &Device) -> Result<Self> {
        let encoder_cfg = cfg.encoder();
        let dtype = if device.is_cpu() {
            DType::F32
        } else {
            vb.dtype()
        };
        let mut weights = HashMap::new();
        encoder_weights(
            &vb.pp(ASR),
            encoder_cfg.num_hidden_layers,
            EncoderNames {
                ours: "",
                skip: &BLOCK_PROJECTIONS,
            },
            dtype,
            &mut weights,
        )?;
        let (h, inter) = (encoder_cfg.hidden_size, encoder_cfg.intermediate_size);
        let projections = (0..encoder_cfg.num_hidden_layers)
            .map(|layer| {
                let block = vb.pp(ASR).pp("layers").pp(layer);
                let [qkv, o, fc1, fc2] = BLOCK_PROJECTIONS;
                let projection = |name: &str, in_dim: usize, out_dim: usize| {
                    tower_projection(&block.pp(name), [in_dim, out_dim], dtype, device)
                };
                Ok(NemoLayerProjections {
                    qkv: NemoQkv::Fused(projection(qkv, h, 3 * h)?),
                    o: projection(o, h, h)?,
                    fc1: projection(fc1, h, inter)?,
                    fc2: projection(fc2, inter, h)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let encoder = NemoTransformer::with_projections(
            encoder_cfg.clone(),
            VarBuilder::from_tensors(weights, dtype, device),
            projections,
        )?;
        let diarization = diarization_config(config)?;
        let layers = serde_json::from_str::<serde_json::Value>(&diarization)
            .map_err(inference_tensor::Error::msg)?["audio_config"]["num_hidden_layers"]
            .as_u64()
            .ok_or_else(|| inference_tensor::Error::msg("diarization config has no layer count"))?
            as usize;
        let diarizer = Nemotron3Diarizer::from_config_json(
            &diarization,
            mel_config(encoder_cfg.num_mel_bins),
            VarBuilder::from_tensors(
                diarization_weights(&vb.pp(DIARIZATION), layers, dtype)?,
                dtype,
                device,
            ),
        )?;
        let speakers = diarizer.num_speakers();
        let hidden = encoder_cfg.hidden_size;
        let norm = |prefix: &str, size: usize| -> Result<LayerNorm> {
            let vb = vb.pp(prefix);
            let mut map = HashMap::new();
            for suffix in ["weight", "bias"] {
                map.insert(suffix.to_string(), tower_tensor(&vb, suffix, dtype)?);
            }
            layer_norm(
                size,
                LAYER_NORM_EPS,
                VarBuilder::from_tensors(map, dtype, device),
            )
        };
        let projector = {
            let p = vb.pp(PROJECTOR);
            let mut map = HashMap::new();
            for suffix in ["weight", "bias"] {
                map.insert(suffix.to_string(), tower_tensor(&p, suffix, dtype)?);
            }
            linear(
                hidden,
                cfg.text_config.hidden_size,
                VarBuilder::from_tensors(map, dtype, device),
            )?
        };
        let word_timestamps = match &cfg.word_timestamp_config {
            Some(head) if vb.contains_tensor(&format!("{WORD_TIMESTAMPS}.upsample.weight")) => {
                Some((
                    WordTimestampHead::new(head, vb, dtype, device)?,
                    head.upsampling_factor,
                ))
            }
            _ => None,
        };
        Ok(Self {
            word_timestamps,
            asr_norm: norm(ASR_NORM, hidden)?,
            speaker_norm: norm(SPEAKER_NORM, speakers)?,
            speaker_table: speaker_table(speakers, hidden, device)?.to_dtype(dtype)?,
            projector,
            activity_threshold: cfg.audio_config.speaker_activity_threshold,
            subsampling: encoder_cfg.subsampling_factor,
            encoder,
            diarizer,
        })
    }

    // the encoder over chunks of `CHUNK_FRAMES` steps, each run with `CHUNK_CONTEXT` more on either side
    fn encode(&self, clip: &AudioClip, steps: usize) -> Result<Tensor> {
        let mut out = Vec::new();
        for (first, start, end, last) in encoder_windows(steps) {
            let xs = self.encoder.embed(&clip.normalized, first, last - first)?;
            let hs = self.encoder.forward(&xs)?;
            out.push(hs.narrow(1, start - first, end - start)?);
        }
        Tensor::cat(&out, 1)?.squeeze(0)
    }

    /// `(steps, hidden)` projected features, one per `subsampling` frames of the clip.
    fn forward(&self, clip: &AudioClip) -> Result<Tensor> {
        let steps = clip.frames.div_ceil(self.subsampling);
        let hidden = self.encode(clip, steps)?;
        let logits = self.diarizer.logits_of_features(&clip.raw, clip.frames)?;
        let speakers = self.diarizer.num_speakers();
        // each step's sigmoids averaged over its frames, then thresholded
        let probabilities = inference_tensor::nn::ops::sigmoid(&logits.to_dtype(DType::F32)?)?
            .narrow(0, 0, steps * self.subsampling)?
            .reshape((steps, self.subsampling, speakers))?
            .mean(1)?;
        let activity = probabilities
            .gt(f64::from(self.activity_threshold))?
            .to_dtype(self.speaker_table.dtype())?;
        let speaker_states = self
            .speaker_norm
            .forward(&activity)?
            .matmul(&self.speaker_table)?;
        let hidden = (self.asr_norm.forward(&hidden)? + speaker_states)?;
        self.projector.forward(&hidden)
    }

    /// The word timestamp head's log-probabilities over `classes` (`frames x classes.len()`, the head's frames) and the
    /// diarization model's logits, for [`word_timestamps::align`]; `None` without the head.
    fn word_timestamp_inputs(
        &self,
        clip: &AudioClip,
        classes: &[u32],
    ) -> Result<
        Option<(
            word_timestamps::LogProbs,
            word_timestamps::SpeakerLogits,
            usize,
        )>,
    > {
        let Some((head, upsampling)) = &self.word_timestamps else {
            return Ok(None);
        };
        let steps = clip.frames.div_ceil(self.subsampling);
        let states = self.encode(clip, steps)?;
        let ids = Tensor::new(classes, states.device())?;
        let mut rows = Vec::new();
        for (first, start, end, last) in encoder_windows(steps) {
            let window = head.log_probs(&states.narrow(0, first, last - first)?, &ids)?;
            rows.push(window.narrow(
                0,
                (start - first) * upsampling,
                (end - start) * upsampling,
            )?);
        }
        let log_probs = word_timestamps::LogProbs {
            frames: steps * upsampling,
            columns: classes.to_vec(),
            values: Tensor::cat(&rows, 0)?.flatten_all()?.to_vec1::<f32>()?,
        };
        let logits = self.diarizer.logits_of_features(&clip.raw, clip.frames)?;
        let speakers = logits.dim(1)?;
        let speaker_logits = word_timestamps::SpeakerLogits {
            speakers,
            values: logits
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
        };
        Ok(Some((log_probs, speaker_logits, *upsampling)))
    }
}

pub struct NemotronTranscribeModel {
    text: GraniteMoeHybrid,
    audio: AudioTower,
    audio_token: u32,
    vocab_size: usize,
    mel: inference_audio::nemo::NemoMel,
    // the word timestamp head's vocabulary and its blank, when the loader gave the tokenizer
    alignment: Option<(tokenizers::Tokenizer, u32)>,
}

impl NemotronTranscribeModel {
    pub fn new(
        cfg: &Config,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let device = normal_loading_metadata.real_device.clone();
        let audio = AudioTower::new(cfg, config, &vb, &device)?;
        let llm = vb.pp(LLM);
        let mut layout = cfg.text_config.layout()?;
        (layout.embeddings, layout.final_norm) = crate::loaders::NEMO_DECODER_ENDS;
        let text = GraniteMoeHybrid::new_with_layout(
            &cfg.text_config.granite()?,
            layout,
            llm.pp("model"),
            llm.pp("lm_head"),
            true,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        let tokenizer = serde_json::from_str::<serde_json::Value>(config)
            .map_err(inference_tensor::Error::msg)?
            .get(inference_nn::loaders::ALIGNMENT_TOKENIZER_KEY)
            .map(|json| {
                inference_nn::utils::tokenizer::tokenizer_from_bytes(json.to_string().as_bytes())
            })
            .transpose()
            .map_err(inference_tensor::Error::msg)?;
        let alignment = tokenizer.zip(
            cfg.word_timestamp_config
                .as_ref()
                .map(|head| head.blank_token_id),
        );
        Ok(Self {
            text,
            audio,
            audio_token: cfg.audio_token_id,
            vocab_size: cfg.text_config.vocab_size,
            mel: inference_audio::nemo::NemoMel::new(mel_config(
                cfg.audio_config.asr_config.num_mel_bins,
            )),
            alignment,
        })
    }

    /// `transcript`'s words timed on `audio` and attributed to its `<spk:N>` tags (a transcript without tags aligns as
    /// tag 0); `None` when the checkpoint has no word timestamp head or its loader gave no alignment tokenizer.
    pub fn word_timestamps(
        &self,
        audio: &inference_audio::AudioInput,
        transcript: &str,
        language: Option<&str>,
    ) -> anyhow::Result<Option<word_timestamps::WordTimestamps>> {
        let Some((tokenizer, blank)) = &self.alignment else {
            return Ok(None);
        };
        let mono = audio.to_mono();
        let pcm = if audio.sample_rate == inputs_processor::SAMPLE_RATE {
            mono
        } else {
            inference_audio::mel::resample(&mono, audio.sample_rate, inputs_processor::SAMPLE_RATE)?
                .into_owned()
        };
        let clip = inputs_processor::audio_clip(&self.mel, &pcm)?;
        let streams =
            word_timestamps::tokenize_transcript(tokenizer, transcript, language, *blank)?;
        let classes = word_timestamps::classes(&streams, *blank);
        let Some((log_probs, speakers, upsampling)) =
            self.audio.word_timestamp_inputs(&clip, &classes)?
        else {
            return Ok(None);
        };
        let timing = word_timestamps::Timing {
            spectrogram_frames: clip.frames,
            subsampling: self.audio.subsampling,
            upsampling,
            hop_seconds: inputs_processor::HOP_LENGTH as f64
                / f64::from(inputs_processor::SAMPLE_RATE),
        };
        // the speaker branch places a tagged transcript's speakers; an untagged one is one speaker everywhere
        let speakers = word_timestamps::has_speaker_tags(transcript).then_some(&speakers);
        word_timestamps::align(&streams, &log_probs, speakers, timing).map(Some)
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        args: NemotronTranscribeSpecificArgs,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        // the placeholder lies past the vocabulary (NVIDIA's tokenizer appends it): it is looked up as token 0
        let looked_up = if (self.audio_token as usize) < self.vocab_size {
            input_ids.clone()
        } else {
            let placeholder = input_ids.eq(self.audio_token)?;
            placeholder.where_cond(&input_ids.zeros_like()?, input_ids)?
        };
        let mut embeds = self.text.token_embeddings(&looked_up)?;
        if !args.audio.is_empty() {
            let projected = args
                .audio
                .iter()
                .map(|clip| self.audio.forward(clip))
                .collect::<Result<Vec<_>>>()?;
            let projected = Tensor::cat(&projected, 0)?.to_device(embeds.device())?;
            embeds = place_token_features(&embeds, input_ids, self.audio_token, &projected, MODEL)?;
        }
        self.text
            .forward_embeds(input_ids, embeds, ctx, &|_, x| Ok(x))
    }
}

impl IsqModel for NemotronTranscribeModel {
    // the audio tower stays as the checkpoint has it
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = crate::utils::unvarbuilder::UnVarBuilder::new();
        self.text.residual_tensors_m(uvb.pp(LLM).pp("model"))
    }
}

impl crate::speculative::SpeculativeTargetMixin for NemotronTranscribeModel {}
impl crate::model::BlockDiffusionMixin for NemotronTranscribeModel {}
impl AnyMoeBaseModelMixin for NemotronTranscribeModel {}

impl MultimodalModel for NemotronTranscribeModel {
    fn word_timestamps(
        &self,
        audio: &inference_audio::AudioInput,
        transcript: &str,
        language: Option<&str>,
    ) -> anyhow::Result<Option<Vec<inference_nn::loaders::AlignedWord>>> {
        let Some(timestamps) =
            NemotronTranscribeModel::word_timestamps(self, audio, transcript, language)?
        else {
            return Ok(None);
        };
        if !timestamps.failed.is_empty() {
            tracing::warn!(
                "{MODEL}: the words of speaker tags {:?} did not fit on their speakers' frames",
                timestamps.failed
            );
        }
        Ok(Some(
            timestamps
                .words
                .into_iter()
                .map(|word| inference_nn::loaders::AlignedWord {
                    text: word.text,
                    start: word.start,
                    end: word.end,
                    speaker: word.tag,
                })
                .collect(),
        ))
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        _pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn std::any::Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let args = *model_specific_args
            .downcast::<NemotronTranscribeSpecificArgs>()
            .expect("Cannot downcast into `NemotronTranscribeSpecificArgs`");
        self.forward_inner(input_ids, args, ctx)
    }

    fn device(&self) -> &Device {
        NormalModel::device(&self.text)
    }

    fn cache(&self) -> &EitherCache {
        NormalModel::cache(&self.text)
    }

    fn max_seq_len(&self) -> usize {
        NormalModel::max_seq_len(&self.text)
    }

    fn config(&self) -> &ModelConfigMetadata {
        NormalModel::config(&self.text)
    }

    fn model_config(
        &self,
    ) -> std::sync::Arc<dyn inference_nn::paged_attention::ModelConfigLike + Send + Sync> {
        self.text.paged_kv_config()
    }

    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs(&self) -> bool {
        NormalModel::supports_cuda_decode_graphs(&self.text)
    }

    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn std::any::Any> {
        Box::new(NemotronTranscribeSpecificArgs { audio: Vec::new() })
    }

    fn supports_packed_prefill(&self) -> bool {
        false
    }

    fn supports_mixed_media_batches(&self) -> bool {
        true
    }
}
