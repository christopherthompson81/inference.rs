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

use super::{AudioChunk, MODEL, UltravoxConfig, UltravoxSpecificArgs};

/// The run of audio tokens is marked with an unused Llama special token: the reference writes EOS, which the turn
/// ends carry too, and placement goes by token.
pub const AUDIO_TOKEN: &str = "<|reserved_special_token_0|>";
pub const SAMPLE_RATE: u32 = 16_000;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
// the encoder reads 3000 frames (30 s) at a time; later chunks are zero-padded to it
const CHUNK_FRAMES: usize = 3000;
// the stride-2 conv
const ENCODER_DOWNSAMPLE: usize = 2;

/// The chunks of a clip of `samples`: each one's real frames as the reference counts them (the waveform mask sliced
/// every hop, so `ceil(samples / hop)` in all), as `(first frame, frames)`.
fn chunk_lengths(samples: usize) -> Vec<(usize, usize)> {
    let frames = samples.div_ceil(HOP_LENGTH).max(1);
    (0..frames)
        .step_by(CHUNK_FRAMES)
        .map(|offset| (offset, (frames - offset).min(CHUNK_FRAMES)))
        .collect()
}

pub struct UltravoxInputsProcessor {
    config: UltravoxConfig,
}

impl UltravoxInputsProcessor {
    pub fn new(config: UltravoxConfig) -> Self {
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

    /// The tokens a chunk of `frames` fills: one per `stack_factor` encoder positions, rounded up.
    pub fn chunk_tokens(&self, frames: usize) -> usize {
        frames.div_ceil(ENCODER_DOWNSAMPLE * self.config.stack_factor)
    }

    fn clip_tokens(&self, samples: usize) -> usize {
        chunk_lengths(samples)
            .into_iter()
            .map(|(_, frames)| self.chunk_tokens(frames))
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
            .map(|clip| self.clip_tokens(clip.len()))
            .collect::<Vec<_>>();
        let placeholder = AudioPlaceholder {
            model: MODEL,
            text: AUDIO_TOKEN,
            token: self.config.audio_token_id,
        };
        expand_audio_placeholders(tokenizer, seq, placeholder, &counts, paged_attn_metadata)
    }

    // one log-mel over the whole clip (one peak, no 30 s padding), cut into chunks; the first is read as it is, later
    // ones zero-padded to a whole chunk with their padded tail masked
    fn clip_chunks(&self, clip: &[f32], device: &Device) -> anyhow::Result<Vec<AudioChunk>> {
        let frames = self.log_mel().frames(clip, LogMelMax::Clip);
        let n_mels = self.log_mel().n_mels;
        chunk_lengths(clip.len())
            .into_iter()
            .map(|(offset, real)| {
                let end = (offset + CHUNK_FRAMES).min(frames.len());
                let width = if offset == 0 {
                    end - offset
                } else {
                    CHUNK_FRAMES
                };
                let mut mel = vec![0.0f32; n_mels * width];
                for (t, frame) in frames[offset.min(end)..end].iter().enumerate() {
                    for (m, &value) in frame.iter().enumerate() {
                        mel[m * width + t] = value;
                    }
                }
                Ok(AudioChunk {
                    mel: Tensor::from_vec(mel, (1, n_mels, width), device)?,
                    keys: (real - 1) / ENCODER_DOWNSAMPLE + 1,
                    tokens: self.chunk_tokens(real),
                })
            })
            .collect()
    }
}

impl MultimodalInputsProcessor for UltravoxInputsProcessor {
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
                    audio.extend(self.clip_chunks(clip, device)?);
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
                model_specific_args: Box::new(UltravoxSpecificArgs { audio }),
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
