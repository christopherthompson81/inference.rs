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

use super::{MODEL, VoxtralChatConfig, VoxtralChatSpecificArgs};

pub const BEGIN_AUDIO: &str = "[BEGIN_AUDIO]";
pub const AUDIO_TOKEN: &str = "[AUDIO]";
pub const SAMPLE_RATE: u32 = 16_000;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
// the encoder reads 30 s at a time; a clip is zero-padded to whole chunks
const CHUNK_SAMPLES: usize = 30 * SAMPLE_RATE as usize;
const CHUNK_FRAMES: usize = CHUNK_SAMPLES / HOP_LENGTH;

/// A clip's 30 s chunks.
pub fn chunk_count(samples: usize) -> usize {
    samples.div_ceil(CHUNK_SAMPLES).max(1)
}

pub struct VoxtralChatInputsProcessor {
    config: VoxtralChatConfig,
}

impl VoxtralChatInputsProcessor {
    pub fn new(config: VoxtralChatConfig) -> Self {
        Self { config }
    }

    fn log_mel(&self) -> WhisperLogMel {
        WhisperLogMel {
            sample_rate: SAMPLE_RATE,
            n_fft: N_FFT,
            hop_length: HOP_LENGTH,
            n_mels: self.config.audio_config.num_mel_bins,
        }
    }

    /// The `[AUDIO]` tokens one chunk fills: its encoder positions, regrouped by the projector's stacking.
    pub fn tokens_per_chunk(config: &VoxtralChatConfig) -> usize {
        let ac = &config.audio_config;
        ac.max_source_positions * ac.hidden_size / config.stacked_width()
    }

    // one log-mel over the padded clip (one peak for every chunk), cut into (chunks, n_mels, frames)
    fn clip_features(&self, clip: &[f32], device: &Device) -> anyhow::Result<Tensor> {
        let chunks = chunk_count(clip.len());
        let mut padded = clip.to_vec();
        padded.resize(chunks * CHUNK_SAMPLES, 0.0);
        let frames = self.log_mel().frames(&padded, LogMelMax::Clip);
        let n_mels = self.log_mel().n_mels;
        let mut mel = vec![0.0f32; chunks * n_mels * CHUNK_FRAMES];
        for (t, frame) in frames.iter().enumerate().take(chunks * CHUNK_FRAMES) {
            let (chunk, t) = (t / CHUNK_FRAMES, t % CHUNK_FRAMES);
            for (m, &value) in frame.iter().enumerate() {
                mel[(chunk * n_mels + m) * CHUNK_FRAMES + t] = value;
            }
        }
        Ok(Tensor::from_vec(
            mel,
            (chunks, n_mels, CHUNK_FRAMES),
            device,
        )?)
    }
}

impl AudioPrompt for VoxtralChatInputsProcessor {
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
        let per_chunk = Self::tokens_per_chunk(&self.config);
        let counts = clips
            .iter()
            .map(|clip| chunk_count(clip.len()) * per_chunk)
            .collect::<Vec<_>>();
        let placeholder = AudioPlaceholder {
            model: MODEL,
            text: AUDIO_TOKEN,
            token: self.config.audio_token_id,
        };
        expand_audio_placeholders(tokenizer, seq, placeholder, &counts, paged_attn_metadata)
    }
}

impl MultimodalInputsProcessor for VoxtralChatInputsProcessor {
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
            Box::new(VoxtralChatSpecificArgs { audio }),
        )
    }
}
