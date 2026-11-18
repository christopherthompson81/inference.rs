#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use std::{any::Any, sync::Arc};

use inference_audio::nemo::NemoMel;
use inference_tensor::Device;
use tokenizers::Tokenizer;

use crate::device_map::DeviceMapper;
use crate::media_inputs::{
    audio_prompt::{
        AudioPlaceholder, AudioPrompt, audio_step_output, expand_audio_placeholders,
        plan_audio_prompts, step_audio_clips,
    },
    processor::{
        InputProcessorOutput, InputsHost, MediaSequence, MultimodalInputsProcessor, TextInputs,
    },
};
use crate::paged_attention::PagedAttentionMeta;

use super::{AudioClip, Config, MODEL, NemotronTranscribeSpecificArgs, mel_config};

pub const AUDIO_TOKEN: &str = "<|audio|>";
// the processor's `NemotronAsrStreamingFeatureExtractor` (`processor_config.json`)
pub const SAMPLE_RATE: u32 = 16_000;
pub const N_FFT: usize = 512;
pub const WIN_LENGTH: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const PREEMPHASIS: f32 = 0.97;
// transformers adds this to each feature's (unbiased) standard deviation
const NORMALIZE_EPSILON: f64 = 1e-5;

/// `features` (`frames` rows of `mels`) normalized per mel bin to zero mean and unit variance over the frames.
fn normalize_per_feature(features: &[f32], frames: usize, mels: usize) -> Vec<f32> {
    let mut out = features.to_vec();
    for m in 0..mels {
        let column = || (0..frames).map(|t| f64::from(features[t * mels + m]));
        let mean = column().sum::<f64>() / frames.max(1) as f64;
        let variance =
            column().map(|v| (v - mean).powi(2)).sum::<f64>() / (frames.max(2) - 1) as f64;
        let scale = 1.0 / (variance.sqrt() + NORMALIZE_EPSILON);
        for t in 0..frames {
            out[t * mels + m] = ((f64::from(features[t * mels + m]) - mean) * scale) as f32;
        }
    }
    out
}

/// A 16 kHz mono clip's log-mel, as the model reads it.
pub fn audio_clip(mel: &NemoMel, pcm: &[f32]) -> anyhow::Result<AudioClip> {
    let (raw, frames) = mel.features(pcm)?;
    let normalized = normalize_per_feature(&raw, frames, mel.config().n_mels);
    Ok(AudioClip {
        raw,
        normalized,
        frames,
    })
}

pub struct NemotronTranscribeInputsProcessor {
    mel: NemoMel,
    subsampling: usize,
    audio_token: u32,
}

impl NemotronTranscribeInputsProcessor {
    pub fn new(config: &Config) -> Self {
        let asr = &config.audio_config.asr_config;
        Self {
            mel: NemoMel::new(mel_config(asr.num_mel_bins)),
            subsampling: asr.subsampling_factor,
            audio_token: config.audio_token_id,
        }
    }

    /// The tokens a clip of `samples` fills: one per `subsampling` of its `floor(samples / hop)` frames, rounded up.
    pub fn clip_tokens(&self, samples: usize) -> usize {
        (samples / HOP_LENGTH).div_ceil(self.subsampling)
    }

    fn clip(&self, pcm: &[f32]) -> anyhow::Result<AudioClip> {
        audio_clip(&self.mel, pcm)
    }
}

impl AudioPrompt for NemotronTranscribeInputsProcessor {
    fn model(&self) -> &'static str {
        MODEL
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn expand_prompt(
        &self,
        tokenizer: &Tokenizer,
        seq: &mut dyn MediaSequence,
        clips: &[Vec<f32>],
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        let counts = clips
            .iter()
            .map(|clip| self.clip_tokens(clip.len()))
            .collect::<Vec<_>>();
        let placeholder = AudioPlaceholder {
            model: MODEL,
            text: AUDIO_TOKEN,
            token: self.audio_token,
        };
        expand_audio_placeholders(tokenizer, seq, placeholder, &counts, paged_attn_metadata)
    }
}

impl MultimodalInputsProcessor for NemotronTranscribeInputsProcessor {
    fn prepare_for_paged_prompt_planning(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut dyn MediaSequence],
        _device: &Device,
        _other_config: Option<Arc<dyn Any>>,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        plan_audio_prompts(self, tokenizer, input_seqs, paged_attn_metadata)
    }

    fn process_inputs(
        &self,
        host: &dyn InputsHost,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut dyn MediaSequence],
        is_prompt: bool,
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        sliding_window: Option<usize>,
        _other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> anyhow::Result<InputProcessorOutput> {
        let clips = step_audio_clips(
            self,
            tokenizer,
            input_seqs,
            is_prompt,
            no_kv_cache,
            paged_attn_metadata.as_mut(),
        )?;
        let audio = clips
            .iter()
            .map(|clip| self.clip(clip))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let text = TextInputs {
            device,
            last_n_context_len,
            return_raw_logits,
            paged_attn_metadata: paged_attn_metadata.as_mut(),
            mapper,
            sliding_window,
        };
        audio_step_output(
            host,
            input_seqs,
            is_prompt,
            no_kv_cache,
            text,
            Box::new(NemotronTranscribeSpecificArgs { audio }),
        )
    }
}
