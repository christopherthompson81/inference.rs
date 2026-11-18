use std::{any::Any, sync::Arc};

use inference_audio::whisper::{LogMelMax, WhisperLogMel};
use inference_tensor::{Device, Tensor};
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

use super::{AudioClip, MODEL, PooledAudioSpec, Qwen2AudioSpecificArgs, audio_token_count};

pub const AUDIO_BOS: &str = "<|audio_bos|>";
pub const AUDIO_TOKEN: &str = "<|AUDIO|>";
pub const AUDIO_EOS: &str = "<|audio_eos|>";
/// Audio Flamingo 3's placeholder, which its template writes for each audio part.
pub const SOUND_TOKEN: &str = "<sound>";
pub const SAMPLE_RATE: u32 = 16_000;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
// each 30 s window is its own feature extraction
const WINDOW_SAMPLES: usize = 30 * SAMPLE_RATE as usize;
const WINDOW_FRAMES: usize = WINDOW_SAMPLES / HOP_LENGTH;

/// A clip's 30 s windows, as sample ranges, up to `max_windows`; the rest is cut, as the processors do.
fn windows(samples: usize, max_windows: usize) -> Vec<std::ops::Range<usize>> {
    let count = samples.div_ceil(WINDOW_SAMPLES).clamp(1, max_windows);
    let end = samples.min(count * WINDOW_SAMPLES);
    (0..count)
        .map(|i| i * WINDOW_SAMPLES..((i + 1) * WINDOW_SAMPLES).min(end))
        .collect()
}

/// The encoder positions a window's real frames reach: its mel frames (its hops, rounded up) through the stride-2
/// conv.
pub fn window_positions(samples: usize) -> usize {
    let frames = samples.min(WINDOW_SAMPLES).div_ceil(HOP_LENGTH).max(1);
    (frames - 1) / 2 + 1
}

/// The tokens a clip of `samples` fills over its windows.
pub fn clip_tokens(samples: usize, max_windows: usize) -> usize {
    windows(samples, max_windows)
        .into_iter()
        .map(|range| audio_token_count(window_positions(range.len())))
        .sum()
}

pub struct Qwen2AudioInputsProcessor {
    spec: PooledAudioSpec,
    placeholder: &'static str,
}

impl Qwen2AudioInputsProcessor {
    pub fn new(spec: PooledAudioSpec, placeholder: &'static str) -> Self {
        Self { spec, placeholder }
    }

    fn log_mel(&self) -> WhisperLogMel {
        WhisperLogMel {
            sample_rate: SAMPLE_RATE,
            n_fft: N_FFT,
            hop_length: HOP_LENGTH,
            n_mels: self.spec.audio.num_mel_bins,
        }
    }

    // each window padded to 30 s and its log-mel peaked on its own, as the feature extractor sees them one by one
    fn clip_features(&self, clip: &[f32], device: &Device) -> anyhow::Result<AudioClip> {
        let ranges = windows(clip.len(), self.spec.max_windows);
        let n_mels = self.log_mel().n_mels;
        let mut mel = vec![0.0f32; ranges.len() * n_mels * WINDOW_FRAMES];
        let mut positions = Vec::with_capacity(ranges.len());
        for (window, range) in ranges.into_iter().enumerate() {
            positions.push(window_positions(range.len()));
            let mut padded = clip[range].to_vec();
            padded.resize(WINDOW_SAMPLES, 0.0);
            let frames = self.log_mel().frames(&padded, LogMelMax::Clip);
            for (t, frame) in frames.iter().enumerate().take(WINDOW_FRAMES) {
                for (m, &value) in frame.iter().enumerate() {
                    mel[(window * n_mels + m) * WINDOW_FRAMES + t] = value;
                }
            }
        }
        Ok(AudioClip {
            mel: Tensor::from_vec(mel, (positions.len(), n_mels, WINDOW_FRAMES), device)?,
            positions,
        })
    }
}

impl AudioPrompt for Qwen2AudioInputsProcessor {
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
            .map(|clip| clip_tokens(clip.len(), self.spec.max_windows))
            .collect::<Vec<_>>();
        let placeholder = AudioPlaceholder {
            model: MODEL,
            text: self.placeholder,
            token: self.spec.audio_token,
        };
        expand_audio_placeholders(tokenizer, seq, placeholder, &counts, paged_attn_metadata)
    }
}

impl MultimodalInputsProcessor for Qwen2AudioInputsProcessor {
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
            .map(|clip| self.clip_features(clip, device))
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
            Box::new(Qwen2AudioSpecificArgs { audio }),
        )
    }
}
