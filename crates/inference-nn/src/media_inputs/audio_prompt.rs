//! Expanding an audio-in model's per-clip placeholder into the run of tokens its encoder output fills.

use std::{any::Any, sync::Arc};

use inference_audio::AudioInput;
use tokenizers::Tokenizer;

use super::media::build_mm_features_from_ranges;
use super::processor::{
    InputProcessorOutput, InputsHost, InputsProcessorValidationError, MediaSequence, TextInputs,
    TextOnlyInputs,
};
use crate::gdn::RecurrentBatchKind;
use crate::paged_attention::{PagedAttentionMeta, block_hash::MultimodalKind};

/// What an audio language model adds to the input-processor steps every such model shares.
pub trait AudioPrompt {
    /// The model name errors carry.
    fn model(&self) -> &'static str;
    /// The rate clips are resampled to.
    fn sample_rate(&self) -> u32;
    /// Expands `seq`'s audio placeholders into the token runs `clips` fill.
    fn expand_prompt(
        &self,
        tokenizer: &Tokenizer,
        seq: &mut dyn MediaSequence,
        clips: &[Vec<f32>],
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()>;
}

fn require_tokenizer(
    prompt: &dyn AudioPrompt,
    tokenizer: Option<Arc<Tokenizer>>,
) -> anyhow::Result<Arc<Tokenizer>> {
    tokenizer
        .ok_or_else(|| anyhow::anyhow!("{} audio processor requires a tokenizer", prompt.model()))
}

/// `prepare_for_paged_prompt_planning` for an audio prompt: each new prompt's placeholders expanded for its clips.
pub fn plan_audio_prompts(
    prompt: &dyn AudioPrompt,
    tokenizer: Option<Arc<Tokenizer>>,
    input_seqs: &mut [&mut dyn MediaSequence],
    mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
) -> anyhow::Result<()> {
    let tokenizer = require_tokenizer(prompt, tokenizer)?;
    for seq in input_seqs {
        if !seq.has_audios() || seq.multimodal().has_changed_prompt {
            continue;
        }
        let clips = mono_clips(&**seq, prompt.sample_rate())?;
        prompt.expand_prompt(
            &tokenizer,
            &mut **seq,
            &clips,
            paged_attn_metadata.as_deref_mut(),
        )?;
    }
    Ok(())
}

/// A step's audio: on a prompt step, each sequence's placeholders expanded and the clips the step encodes, in order.
pub fn step_audio_clips(
    prompt: &dyn AudioPrompt,
    tokenizer: Option<Arc<Tokenizer>>,
    input_seqs: &mut [&mut dyn MediaSequence],
    is_prompt: bool,
    no_kv_cache: bool,
    mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
) -> anyhow::Result<Vec<Vec<f32>>> {
    if no_kv_cache {
        anyhow::bail!("{} must have a kv cache.", prompt.model());
    }
    let tokenizer = require_tokenizer(prompt, tokenizer)?;
    let mut active = Vec::new();
    if is_prompt {
        for seq in input_seqs.iter_mut().filter(|seq| seq.has_audios()) {
            let mut clips = mono_clips(&**seq, prompt.sample_rate())?;
            prompt.expand_prompt(
                &tokenizer,
                &mut **seq,
                &clips,
                paged_attn_metadata.as_deref_mut(),
            )?;
            let range = active_clip_range(&**seq, clips.len(), prompt.model())?;
            active.extend(clips.drain(range));
        }
    }
    Ok(active)
}

/// The step's inputs around the encoded audio its model reads from `model_specific_args`.
pub fn audio_step_output(
    host: &dyn InputsHost,
    input_seqs: &mut [&mut dyn MediaSequence],
    is_prompt: bool,
    no_kv_cache: bool,
    text: TextInputs<'_>,
    model_specific_args: Box<dyn Any>,
) -> anyhow::Result<InputProcessorOutput> {
    let toks = input_seqs
        .iter()
        .map(|seq| seq.get_toks())
        .collect::<Vec<_>>()
        .into();
    let step = host.step_inputs(toks, input_seqs, text, is_prompt, no_kv_cache)?;
    let inputs = step.inputs;
    Ok(TextOnlyInputs {
        input_ids: inputs.input,
        seqlen_offsets: inputs.positions,
        context_lens: inputs.context_lens,
        position_ids: inputs.position_ids,
        paged_attn_meta: inputs.paged_attn_meta,
        flash_meta: inputs.flash_meta,
        recurrent_batch_kind: if is_prompt {
            RecurrentBatchKind::Prefill
        } else {
            RecurrentBatchKind::Decode
        },
        seq_indices: step.seq_indices,
    }
    .into_output(model_specific_args))
}

/// The placeholder a prompt names each clip with, and the token it tokenizes to.
#[derive(Clone, Copy)]
pub struct AudioPlaceholder {
    pub model: &'static str,
    pub text: &'static str,
    pub token: u32,
}

/// A sequence's clips as mono samples at `sample_rate`.
pub fn mono_clips(seq: &dyn MediaSequence, sample_rate: u32) -> anyhow::Result<Vec<Vec<f32>>> {
    seq.clone_audios()
        .unwrap_or_default()
        .iter()
        .map(|audio: &AudioInput| {
            let mono = audio.to_mono();
            if audio.sample_rate == sample_rate {
                Ok(mono)
            } else {
                inference_audio::mel::resample(&mono, audio.sample_rate, sample_rate)
                    .map(std::borrow::Cow::into_owned)
            }
        })
        .collect()
}

/// One range per clip of the placeholder token's runs, in order; adjacent clips share a run.
pub fn audio_ranges(
    ids: &[u32],
    placeholder: AudioPlaceholder,
    counts: &[usize],
) -> anyhow::Result<Vec<(usize, usize)>> {
    let model = placeholder.model;
    let mut ranges = Vec::with_capacity(counts.len());
    let mut counts = counts.iter();
    let mut position = 0;
    while position < ids.len() {
        if ids[position] != placeholder.token {
            position += 1;
            continue;
        }
        let Some(&count) = counts.next() else {
            anyhow::bail!("{model} prompt has more audio tokens than its clips");
        };
        if ids.len() < position + count
            || ids[position..position + count]
                .iter()
                .any(|&id| id != placeholder.token)
        {
            anyhow::bail!("{model} prompt cuts a clip's {count} tokens short");
        }
        ranges.push((position, count));
        position += count;
    }
    if counts.next().is_some() {
        anyhow::bail!("{model} prompt has fewer audio tokens than its clips");
    }
    Ok(ranges)
}

/// Once per sequence: repeats each clip's placeholder `counts[clip]` times, retokenizes the prompt and records the
/// clips' token ranges for the prefix cache.
pub fn expand_audio_placeholders(
    tokenizer: &Tokenizer,
    seq: &mut dyn MediaSequence,
    placeholder: AudioPlaceholder,
    counts: &[usize],
    paged_attn_metadata: Option<&mut PagedAttentionMeta>,
) -> anyhow::Result<()> {
    if seq.multimodal().has_changed_prompt {
        return Ok(());
    }
    let prompt = seq.get_initial_prompt();
    let placeholders = prompt.matches(placeholder.text).count();
    if placeholders != counts.len() {
        return Err(InputsProcessorValidationError(format!(
            "The number of `{}` tokens ({placeholders}) must match the number of audio clips ({})",
            placeholder.text,
            counts.len()
        ))
        .into());
    }
    let mut parts = prompt.split(placeholder.text);
    let mut expanded = parts.next().unwrap_or_default().to_string();
    for (count, part) in counts.iter().zip(parts) {
        expanded.push_str(&placeholder.text.repeat(*count));
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
                anyhow::anyhow!("{} clips are missing content hashes", placeholder.model)
            })?
            .to_vec();
        let ranges = audio_ranges(&ids, placeholder, counts)?;
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

/// The clips of `seq` this prompt step encodes: the active chunk's, or those past the prefix cache's hits.
pub fn active_clip_range(
    seq: &dyn MediaSequence,
    clips: usize,
    model: &str,
) -> anyhow::Result<std::ops::Range<usize>> {
    if seq.is_chunked_prefill_view() {
        seq.active_local_multimodal_item_range(MultimodalKind::Audio, clips)
            .ok_or_else(|| anyhow::anyhow!("{model} audio chunk is missing its active clip range"))
    } else {
        Ok(seq.count_prefix_cached_mm_items_by_kind(MultimodalKind::Audio)..clips)
    }
}

#[cfg(test)]
mod tests {
    use super::{AudioPlaceholder, audio_ranges};

    const PLACEHOLDER: AudioPlaceholder = AudioPlaceholder {
        model: "test",
        text: "<audio>",
        token: 9,
    };

    #[test]
    fn adjacent_clips_split_their_shared_run() {
        let ids = [1, 9, 9, 9, 9, 2];
        assert_eq!(
            audio_ranges(&ids, PLACEHOLDER, &[3, 1]).unwrap(),
            vec![(1, 3), (4, 1)]
        );
        assert!(audio_ranges(&ids, PLACEHOLDER, &[3]).is_err());
        assert!(audio_ranges(&ids, PLACEHOLDER, &[3, 2]).is_err());
    }
}
