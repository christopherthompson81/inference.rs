use std::{any::Any, sync::Arc};

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

use super::{MODEL, VibeVoiceAsrSpecificArgs};

pub const AUDIO_BOS: &str = "<|object_ref_start|>";
pub const AUDIO_TOKEN: &str = "<|box_start|>";
pub const AUDIO_EOS: &str = "<|object_ref_end|>";
/// The template's stand-in for the clip's length, which the processor writes in seconds (`%.2f`).
pub const AUDIO_DURATION: &str = "<|AUDIO_DURATION|>";
pub const SAMPLE_RATE: u32 = 24_000;
// the feature extractor's loudness target and the guard in its division
const TARGET_DB_FS: f64 = -25.0;
const NORMALIZE_EPS: f64 = 1e-6;

/// The embeddings a clip of `samples` fills: one per hop, the last partial one included.
pub fn clip_tokens(samples: usize, hop: usize) -> usize {
    samples.div_ceil(hop).max(1)
}

/// The feature extractor's clip: scaled to -25 dB FS RMS, then down to a peak of 1 where it passes it, and zero-padded
/// to a whole number of hops.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn prepare_clip(clip: &[f32], hop: usize) -> Vec<f32> {
    let rms =
        (clip.iter().map(|&s| f64::from(s).powi(2)).sum::<f64>() / clip.len().max(1) as f64).sqrt();
    let gain = 10f64.powf(TARGET_DB_FS / 20.0) / (rms + NORMALIZE_EPS);
    let mut out: Vec<f32> = clip.iter().map(|&s| (f64::from(s) * gain) as f32).collect();
    let peak = out.iter().fold(0f32, |m, s| m.max(s.abs()));
    if peak > 1.0 {
        let scale = f64::from(peak) + NORMALIZE_EPS;
        out.iter_mut()
            .for_each(|s| *s = (f64::from(*s) / scale) as f32);
    }
    out.resize(clip_tokens(clip.len(), hop) * hop, 0.0);
    out
}

// each duration placeholder, in order, as its clip's seconds
#[allow(clippy::cast_precision_loss)]
fn fill_durations(seq: &mut dyn MediaSequence, clips: &[Vec<f32>]) {
    if seq.multimodal().has_changed_prompt {
        return;
    }
    let mut prompt = seq.get_initial_prompt().to_string();
    for clip in clips {
        let Some(at) = prompt.find(AUDIO_DURATION) else {
            break;
        };
        let seconds = format!("{:.2}", clip.len() as f64 / f64::from(SAMPLE_RATE));
        prompt.replace_range(at..at + AUDIO_DURATION.len(), &seconds);
    }
    seq.set_initial_prompt(prompt);
}

pub struct VibeVoiceAsrInputsProcessor {
    audio_token: u32,
    hop: usize,
}

impl VibeVoiceAsrInputsProcessor {
    pub fn new(audio_token: u32, hop: usize) -> Self {
        Self { audio_token, hop }
    }

    fn expand_prompt(
        &self,
        tokenizer: &Tokenizer,
        seq: &mut dyn MediaSequence,
        clips: &[Vec<f32>],
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        fill_durations(seq, clips);
        let counts = clips
            .iter()
            .map(|clip| clip_tokens(clip.len(), self.hop))
            .collect::<Vec<_>>();
        let placeholder = AudioPlaceholder {
            model: MODEL,
            text: AUDIO_TOKEN,
            token: self.audio_token,
        };
        expand_audio_placeholders(tokenizer, seq, placeholder, &counts, paged_attn_metadata)
    }
}

impl MultimodalInputsProcessor for VibeVoiceAsrInputsProcessor {
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
                    let prepared = prepare_clip(clip, self.hop);
                    let len = prepared.len();
                    audio.push(Tensor::from_vec(prepared, (1, 1, len), device)?);
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
                model_specific_args: Box::new(VibeVoiceAsrSpecificArgs { audio }),
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

    #[test]
    fn a_clip_is_normalized_and_padded_to_whole_hops() {
        let clip = vec![0.5f32; 4000];
        let prepared = prepare_clip(&clip, 3200);
        assert_eq!(prepared.len(), 6400);
        // -25 dB FS of a constant is its value
        assert!((f64::from(prepared[0]) - 10f64.powf(-1.25)).abs() < 1e-5);
        assert_eq!(prepared[5000], 0.0);
        assert_eq!(clip_tokens(4000, 3200), 2);
    }
}
