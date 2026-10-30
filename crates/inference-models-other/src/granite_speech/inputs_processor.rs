use std::{any::Any, sync::Arc};

use inference_audio::AudioInput;
use inference_tensor::{Device, Tensor};
use tokenizers::Tokenizer;

use crate::device_map::DeviceMapper;
use crate::media_inputs::{
    media::build_mm_features_from_ranges,
    processor::{
        InputProcessorOutput, InputsHost, InputsProcessorValidationError, MediaSequence,
        ModelInputs, MultimodalInputsProcessor, TextInputs,
    },
};
use crate::paged_attention::{PagedAttentionMeta, block_hash::MultimodalKind};

use super::features::{SAMPLE_RATE, audio_token_count, log_mel_features};
use super::{Config, GraniteSpeechSpecificArgs};

pub const AUDIO_TOKEN: &str = "<|audio|>";

pub struct GraniteSpeechInputsProcessor {
    config: Config,
}

fn mono_16k(audio: &AudioInput) -> anyhow::Result<Vec<f32>> {
    let mono = audio.to_mono();
    if audio.sample_rate == SAMPLE_RATE {
        return Ok(mono);
    }
    inference_audio::mel::resample(&mono, audio.sample_rate, SAMPLE_RATE)
}

/// One range per clip of the audio token's runs, in order; adjacent clips share a run.
fn audio_ranges(ids: &[u32], token: u32, counts: &[usize]) -> anyhow::Result<Vec<(usize, usize)>> {
    let mut ranges = Vec::with_capacity(counts.len());
    let mut counts = counts.iter();
    let mut position = 0;
    while position < ids.len() {
        if ids[position] != token {
            position += 1;
            continue;
        }
        let Some(&count) = counts.next() else {
            anyhow::bail!("Granite Speech prompt has more audio tokens than its clips");
        };
        if ids.len() < position + count
            || ids[position..position + count]
                .iter()
                .any(|&id| id != token)
        {
            anyhow::bail!("Granite Speech prompt cuts a clip's {count} tokens short");
        }
        ranges.push((position, count));
        position += count;
    }
    if counts.next().is_some() {
        anyhow::bail!("Granite Speech prompt has fewer audio tokens than its clips");
    }
    Ok(ranges)
}

impl GraniteSpeechInputsProcessor {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    fn clips(seq: &dyn MediaSequence) -> anyhow::Result<Vec<Vec<f32>>> {
        seq.clone_audios()
            .unwrap_or_default()
            .iter()
            .map(mono_16k)
            .collect()
    }

    fn maybe_expand_prompt(
        &self,
        tokenizer: &Tokenizer,
        seq: &mut dyn MediaSequence,
        clips: &[Vec<f32>],
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        if seq.multimodal().has_changed_prompt {
            return Ok(());
        }
        let prompt = seq.get_initial_prompt();
        let placeholders = prompt.matches(AUDIO_TOKEN).count();
        if placeholders != clips.len() {
            return Err(InputsProcessorValidationError(format!(
                "The number of `{AUDIO_TOKEN}` tokens ({placeholders}) must match the number of audio clips ({})",
                clips.len()
            ))
            .into());
        }
        let counts = clips
            .iter()
            .map(|clip| {
                audio_token_count(
                    clip.len(),
                    self.config.window_size,
                    self.config.downsample_rate,
                )
            })
            .collect::<Vec<_>>();
        let mut parts = prompt.split(AUDIO_TOKEN);
        let mut expanded = parts.next().unwrap_or_default().to_string();
        for (count, part) in counts.iter().zip(parts) {
            expanded.push_str(&AUDIO_TOKEN.repeat(*count));
            expanded.push_str(part);
        }
        seq.set_initial_prompt(expanded.clone());
        let ids = tokenizer
            .encode_fast(expanded, false)
            .map_err(|err| anyhow::anyhow!(err.to_string()))?
            .get_ids()
            .to_vec();
        if seq.mm_features().is_empty() {
            let hashes = seq
                .audio_hashes()
                .ok_or_else(|| {
                    anyhow::Error::msg("Granite Speech clips are missing content hashes")
                })?
                .to_vec();
            let ranges = audio_ranges(&ids, self.config.audio_token_index, &counts)?;
            seq.set_mm_features(build_mm_features_from_ranges(
                &ranges,
                &hashes,
                MultimodalKind::Audio,
            ));
        }
        seq.set_toks_and_reallocate(ids, paged_attn_metadata);
        seq.multimodal_mut().has_changed_prompt = true;
        Ok(())
    }
}

impl MultimodalInputsProcessor for GraniteSpeechInputsProcessor {
    fn prepare_for_paged_prompt_planning(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut dyn MediaSequence],
        _device: &Device,
        _other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        let Some(tokenizer) = tokenizer else {
            anyhow::bail!("Granite Speech audio processor requires a tokenizer");
        };
        for seq in input_seqs {
            if !seq.has_audios() || seq.multimodal().has_changed_prompt {
                continue;
            }
            let clips = Self::clips(&**seq)?;
            self.maybe_expand_prompt(
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
            anyhow::bail!("Granite Speech must have a kv cache.");
        }
        let Some(tokenizer) = tokenizer else {
            anyhow::bail!("Granite Speech audio processor requires a tokenizer");
        };
        let mut audio_features = Vec::new();
        if is_prompt {
            for seq in input_seqs.iter_mut().filter(|seq| seq.has_audios()) {
                let clips = Self::clips(&**seq)?;
                self.maybe_expand_prompt(
                    &tokenizer,
                    &mut **seq,
                    &clips,
                    paged_attn_metadata.as_mut(),
                )?;
                let range = if seq.is_chunked_prefill_view() {
                    seq.active_local_multimodal_item_range(MultimodalKind::Audio, clips.len())
                        .ok_or_else(|| {
                            anyhow::Error::msg(
                                "Granite Speech audio chunk is missing its active clip range",
                            )
                        })?
                } else {
                    seq.count_prefix_cached_mm_items_by_kind(MultimodalKind::Audio)..clips.len()
                };
                for clip in &clips[range] {
                    let (features, frames) = log_mel_features(clip);
                    let width = features.len() / frames.max(1);
                    audio_features.push(Tensor::from_vec(features, (1, frames, width), device)?);
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
                model_specific_args: Box::new(GraniteSpeechSpecificArgs { audio_features }),
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
    use super::audio_ranges;

    #[test]
    fn adjacent_clips_split_their_shared_run() {
        let ids = [1, 9, 9, 9, 9, 2];
        assert_eq!(
            audio_ranges(&ids, 9, &[3, 1]).unwrap(),
            vec![(1, 3), (4, 1)]
        );
        assert!(audio_ranges(&ids, 9, &[3]).is_err());
        assert!(audio_ranges(&ids, 9, &[3, 2]).is_err());
    }
}
