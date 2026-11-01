#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

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

use super::{AudioClip, MERaLiON2Config, MERaLiON2SpecificArgs, MODEL};

pub const AUDIO_TOKEN: &str = "<SpeechHere>";
pub const SAMPLE_RATE: u32 = 16_000;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
// the processor cuts each clip into 30 s chunks, at most 300 s of them, each its own padded feature extraction
const CHUNK_SAMPLES: usize = 30 * SAMPLE_RATE as usize;
const CHUNK_FRAMES: usize = CHUNK_SAMPLES / HOP_LENGTH;
const MAX_CHUNKS: usize = 10;
/// Every chunk fills this many tokens, whatever its length (`fixed_speech_embeds_length`).
pub const CHUNK_TOKENS: usize = 100;
// the feature extractor's zero-mean unit-variance normalization
const NORM_EPS: f64 = 1e-7;

/// The text a turn holds: the instruction in the template MERaLiON-2 was trained on, then one placeholder per clip.
pub fn instruction(text: &str, clips: usize) -> String {
    format!(
        "Instruction: {text} \nFollow the text instruction based on the following audio: {}",
        AUDIO_TOKEN.repeat(clips)
    )
}

/// A clip's chunks as sample ranges: one more than its whole chunks (the processor's `len // chunk + 1`, so a clip of
/// whole chunks ends in an empty one), up to the limit.
fn chunks(samples: usize) -> Vec<std::ops::Range<usize>> {
    let count = (samples / CHUNK_SAMPLES + 1).min(MAX_CHUNKS);
    (0..count)
        .map(|i| (i * CHUNK_SAMPLES).min(samples)..((i + 1) * CHUNK_SAMPLES).min(samples))
        .collect()
}

pub fn clip_tokens(samples: usize) -> usize {
    chunks(samples).len() * CHUNK_TOKENS
}

/// `samples` at zero mean and unit variance over themselves, zero-padded to a whole chunk; an empty chunk stays silent.
fn normalized_chunk(samples: &[f32]) -> Vec<f32> {
    let mut chunk = vec![0.0f32; CHUNK_SAMPLES];
    if samples.is_empty() {
        return chunk;
    }
    let n = samples.len() as f64;
    let mean = samples.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let var = samples
        .iter()
        .map(|&x| (f64::from(x) - mean).powi(2))
        .sum::<f64>()
        / n;
    let scale = (var + NORM_EPS).sqrt();
    for (out, &x) in chunk.iter_mut().zip(samples) {
        *out = ((f64::from(x) - mean) / scale) as f32;
    }
    chunk
}

pub struct MERaLiON2InputsProcessor {
    config: MERaLiON2Config,
}

impl MERaLiON2InputsProcessor {
    pub fn new(config: MERaLiON2Config) -> Self {
        Self { config }
    }

    fn log_mel(&self) -> WhisperLogMel {
        WhisperLogMel {
            sample_rate: SAMPLE_RATE,
            n_fft: N_FFT,
            hop_length: HOP_LENGTH,
            n_mels: self.config.speech_config.num_mel_bins,
        }
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
            .map(|clip| clip_tokens(clip.len()))
            .collect::<Vec<_>>();
        let placeholder = AudioPlaceholder {
            model: MODEL,
            text: AUDIO_TOKEN,
            token: self.config.speech_token_index,
        };
        expand_audio_placeholders(tokenizer, seq, placeholder, &counts, paged_attn_metadata)
    }

    // each chunk normalized and its log-mel peaked on its own, as the feature extractor sees them one by one
    fn clip_features(&self, clip: &[f32], device: &Device) -> anyhow::Result<AudioClip> {
        let ranges = chunks(clip.len());
        let n_mels = self.log_mel().n_mels;
        let mut mel = vec![0.0f32; ranges.len() * n_mels * CHUNK_FRAMES];
        for (chunk, range) in ranges.iter().enumerate() {
            let frames = self
                .log_mel()
                .frames(&normalized_chunk(&clip[range.clone()]), LogMelMax::Clip);
            for (t, frame) in frames.iter().enumerate().take(CHUNK_FRAMES) {
                for (m, &value) in frame.iter().enumerate() {
                    mel[(chunk * n_mels + m) * CHUNK_FRAMES + t] = value;
                }
            }
        }
        Ok(AudioClip {
            mel: Tensor::from_vec(mel, (ranges.len(), n_mels, CHUNK_FRAMES), device)?,
        })
    }
}

impl MultimodalInputsProcessor for MERaLiON2InputsProcessor {
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
                model_specific_args: Box::new(MERaLiON2SpecificArgs { audio }),
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

#[cfg(test)]
mod tests {
    use super::*;

    // `len // chunk + 1`, capped at ten: a clip of whole chunks ends in an empty one, as the processor counts it
    #[test]
    fn every_clip_takes_one_more_chunk_than_it_fills() {
        assert_eq!(clip_tokens(1), CHUNK_TOKENS);
        assert_eq!(clip_tokens(CHUNK_SAMPLES - 1), CHUNK_TOKENS);
        assert_eq!(clip_tokens(CHUNK_SAMPLES), 2 * CHUNK_TOKENS);
        assert_eq!(chunks(CHUNK_SAMPLES)[1], CHUNK_SAMPLES..CHUNK_SAMPLES);
        assert_eq!(clip_tokens(20 * CHUNK_SAMPLES), MAX_CHUNKS * CHUNK_TOKENS);
    }
}
