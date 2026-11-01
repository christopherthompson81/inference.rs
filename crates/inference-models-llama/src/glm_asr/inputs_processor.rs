use std::{any::Any, sync::Arc};

use inference_audio::whisper::{LogMelMax, WhisperLogMel};
use inference_tensor::{Device, Tensor};
use tokenizers::Tokenizer;

use crate::device_map::DeviceMapper;
use crate::media_inputs::{
    audio_prompt::{AudioPlaceholder, active_clip_range, expand_audio_placeholders, mono_clips},
    processor::{
        InputProcessorOutput, InputsHost, MediaSequence, ModelInputs, MultimodalInputsProcessor,
        TextInputs,
    },
};
use crate::paged_attention::PagedAttentionMeta;

use super::{AudioClip, GlmAsrConfig, GlmAsrSpecificArgs, MODEL};

pub const AUDIO_BEGIN: &str = "<|begin_of_audio|>";
pub const AUDIO_TOKEN: &str = "<|pad|>";
pub const AUDIO_END: &str = "<|end_of_audio|>";
pub const SAMPLE_RATE: u32 = 16_000;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
// each 30 s window is its own feature extraction; the processor reads at most 655 s, 21 whole windows
const CHUNK_SAMPLES: usize = 30 * SAMPLE_RATE as usize;
const CHUNK_FRAMES: usize = CHUNK_SAMPLES / HOP_LENGTH;
const MAX_CHUNKS: usize = 21;
// four encoder positions make one token
const MERGE: usize = 4;

/// A clip's 30 s windows, as sample ranges, up to the processor's limit.
fn chunks(samples: usize) -> Vec<std::ops::Range<usize>> {
    let count = samples.div_ceil(CHUNK_SAMPLES).clamp(1, MAX_CHUNKS);
    let end = samples.min(count * CHUNK_SAMPLES);
    (0..count)
        .map(|i| i * CHUNK_SAMPLES..((i + 1) * CHUNK_SAMPLES).min(end))
        .collect()
}

/// The tokens a window of `samples` keeps: its frames (its hops, rounded up) through the stride-2 conv, merged by four.
pub fn chunk_tokens(samples: usize) -> usize {
    let positions = (samples.div_ceil(HOP_LENGTH).max(1) - 1) / 2 + 1;
    if positions < MERGE {
        0
    } else {
        (positions - MERGE) / MERGE + 1
    }
}

pub struct GlmAsrInputsProcessor {
    config: GlmAsrConfig,
}

impl GlmAsrInputsProcessor {
    pub fn new(config: GlmAsrConfig) -> Self {
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

    /// The tokens a clip of `samples` fills, over all its windows.
    pub fn clip_tokens(samples: usize) -> usize {
        chunks(samples)
            .into_iter()
            .map(|range| chunk_tokens(range.len()))
            .sum()
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
            .map(|clip| Self::clip_tokens(clip.len()))
            .collect::<Vec<_>>();
        let placeholder = AudioPlaceholder {
            model: MODEL,
            text: AUDIO_TOKEN,
            token: self.config.audio_token_id,
        };
        expand_audio_placeholders(tokenizer, seq, placeholder, &counts, paged_attn_metadata)
    }

    // each window padded to 30 s and its log-mel peaked on its own, as the feature extractor sees them one by one
    fn clip_features(&self, clip: &[f32], device: &Device) -> anyhow::Result<AudioClip> {
        let windows = chunks(clip.len());
        let n_mels = self.log_mel().n_mels;
        let mut mel = vec![0.0f32; windows.len() * n_mels * CHUNK_FRAMES];
        let mut tokens = Vec::with_capacity(windows.len());
        for (chunk, range) in windows.into_iter().enumerate() {
            tokens.push(chunk_tokens(range.len()));
            let mut padded = clip[range].to_vec();
            padded.resize(CHUNK_SAMPLES, 0.0);
            let frames = self.log_mel().frames(&padded, LogMelMax::Clip);
            for (t, frame) in frames.iter().enumerate().take(CHUNK_FRAMES) {
                for (m, &value) in frame.iter().enumerate() {
                    mel[(chunk * n_mels + m) * CHUNK_FRAMES + t] = value;
                }
            }
        }
        Ok(AudioClip {
            mel: Tensor::from_vec(mel, (tokens.len(), n_mels, CHUNK_FRAMES), device)?,
            tokens,
        })
    }
}

impl MultimodalInputsProcessor for GlmAsrInputsProcessor {
    fn prepare_for_paged_prompt_planning(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut dyn MediaSequence],
        _device: &Device,
        _other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        let Some(tokenizer) = tokenizer else {
            anyhow::bail!("{MODEL} audio processor requires a tokenizer");
        };
        for seq in input_seqs {
            if !seq.has_audios() || seq.multimodal().has_changed_prompt {
                continue;
            }
            let clips = mono_clips(&**seq, SAMPLE_RATE)?;
            self.expand_prompt(
                &tokenizer,
                &mut **seq,
                &clips,
                paged_attn_metadata.as_deref_mut(),
            )?;
        }
        Ok(())
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
        if no_kv_cache {
            anyhow::bail!("{MODEL} must have a kv cache.");
        }
        let Some(tokenizer) = tokenizer else {
            anyhow::bail!("{MODEL} audio processor requires a tokenizer");
        };
        let mut audio = Vec::new();
        if is_prompt {
            for seq in input_seqs.iter_mut().filter(|seq| seq.has_audios()) {
                let clips = mono_clips(&**seq, SAMPLE_RATE)?;
                self.expand_prompt(&tokenizer, &mut **seq, &clips, paged_attn_metadata.as_mut())?;
                let range = active_clip_range(&**seq, clips.len(), MODEL)?;
                for clip in &clips[range] {
                    audio.push(self.clip_features(clip, device)?);
                }
            }
        }

        let toks = input_seqs
            .iter()
            .map(|seq| seq.get_toks())
            .collect::<Vec<_>>()
            .into();
        let text = TextInputs {
            device,
            last_n_context_len,
            return_raw_logits,
            paged_attn_metadata: paged_attn_metadata.as_mut(),
            mapper,
            sliding_window,
        };
        let inference_nn::media_inputs::processor::InnerInputProcessorOutput {
            inputs:
                inference_nn::media_inputs::processor::InputMetadata {
                    input,
                    positions,
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta,
                },
            seq_indices,
        } = host.step_inputs(toks, input_seqs, text, is_prompt, no_kv_cache)?;

        Ok(InputProcessorOutput {
            inputs: Box::new(ModelInputs {
                input_ids: input,
                seqlen_offsets: positions,
                context_lens,
                position_ids,
                pixel_values: None,
                model_specific_args: Box::new(GlmAsrSpecificArgs { audio }),
                paged_attn_meta,
                flash_meta,
                recurrent_batch_kind: if is_prompt {
                    crate::gdn::RecurrentBatchKind::Prefill
                } else {
                    crate::gdn::RecurrentBatchKind::Decode
                },
            }),
            seq_indices,
        })
    }
}
