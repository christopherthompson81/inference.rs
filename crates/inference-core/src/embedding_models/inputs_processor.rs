#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::attention::FlashParams;
use crate::attention::flash_params::make_flash_params;
use std::{any::Any, fmt::Debug, sync::Arc};

use anyhow::Result;
use inference_tensor::{Device, Tensor, WithDType};
use tokenizers::Tokenizer;

use crate::paged_attention::PagedAttentionMeta;
use crate::{
    device_map::DeviceMapper,
    pipeline::{
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    sequence::Sequence,
};
use inference_nn::decision::DecisionRows;
use inference_nn::embedding::{EmbeddingExtras, TaskRows};

fn _make_tensor_with_pad<D: WithDType>(
    x: Vec<Vec<D>>,
    max_len: usize,
    pad: D,
    device: &Device,
) -> Result<Tensor> {
    let mut padded_x = Vec::new();
    for mut x_i in x {
        assert!(x_i.len() <= max_len);
        x_i.extend([pad].repeat(max_len - x_i.len()));
        let shape = (x_i.len(),);
        padded_x.push(Tensor::from_vec(x_i, shape, device)?);
    }
    Tensor::cat(&padded_x[..], 0).map_err(anyhow::Error::msg)
}

pub struct InputMetadata {
    pub input: Tensor,
    pub flash_meta: FlashParams,
}

pub struct InnerInputProcessorOutput {
    pub inputs: InputMetadata,
    pub seq_indices: Vec<usize>,
}

// chunk_offset_toks is the number of tokens by which the tokens are offset,
// chunk_offset_toks / prompt_chunksize = number of batches
#[allow(clippy::too_many_arguments)]
pub fn make_prompt_chunk<T: WithDType + Debug>(
    chunk_offset_toks: usize,
    toks: Vec<&[T]>,
    device: &Device,
    mapper: Option<&dyn DeviceMapper>,
    has_causal_attention: bool,
    sliding_window: Option<usize>,
) -> Result<InputMetadata> {
    let max_len = toks
        .iter()
        .map(|seq| seq.len())
        .max()
        .expect("No sequences");
    let padding_tok = T::zero();
    // Pad each sequence by the padding token to the max len.
    let mut seqs_tensors = Vec::new();
    let flash_attn = crate::using_flash_attn();
    let mut seqlens_q = if flash_attn { vec![0] } else { Vec::new() };
    let mut seqlens_k = if flash_attn { vec![0] } else { Vec::new() };
    for ctxt in toks {
        let mut ctxt = ctxt.to_vec();
        ctxt.extend(std::iter::repeat_n(
            padding_tok,
            max_len.saturating_sub(ctxt.len()),
        ));

        if flash_attn {
            seqlens_q.push(ctxt.len() as u32);
            seqlens_k.push((ctxt.len() + chunk_offset_toks) as u32);
        }

        seqs_tensors.push(Tensor::new(ctxt, device).unwrap().unsqueeze(0).unwrap());
    }

    let flash_meta = if flash_attn {
        make_flash_params(
            device,
            mapper,
            &seqlens_q,
            &seqlens_k,
            sliding_window,
            has_causal_attention,
            false,
        )?
    } else {
        FlashParams::empty(has_causal_attention)
    };

    let input = Tensor::cat(&seqs_tensors, 0).unwrap();

    Ok(InputMetadata { input, flash_meta })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn get_prompt_input<T: WithDType + std::fmt::Debug>(
    toks: Vec<&[T]>,
    input_seqs: &[&mut Sequence],
    device: &Device,
    mapper: Option<&dyn DeviceMapper>,
    has_causal_attention: bool,
    sliding_window: Option<usize>,
) -> Result<InnerInputProcessorOutput> {
    let offset = input_seqs[0].token_offset();
    make_prompt_chunk(
        offset,
        toks,
        device,
        mapper,
        has_causal_attention,
        sliding_window,
    )
    .map(|inputs| InnerInputProcessorOutput {
        inputs,
        seq_indices: (0..input_seqs.len()).collect(),
    })
}

#[derive(Clone)]
pub struct ModelInputs {
    pub input_ids: Tensor,
    pub extras: EmbeddingExtras,
    pub flash_meta: FlashParams,
    /// Each row's task, on the host, for the adapters a LoRA runtime applies.
    pub tasks: Vec<Option<usize>>,
}

/// Each row's task adapter; `None` when no sequence named a task.
fn task_rows(input_seqs: &[&mut Sequence], device: &Device) -> Result<Option<TaskRows>> {
    if input_seqs.iter().all(|seq| seq.embedding_task().is_none()) {
        return Ok(None);
    }
    let tasks: Vec<Option<usize>> = input_seqs.iter().map(|seq| seq.embedding_task()).collect();
    // a row with no task reads adapter 0 and scales its delta by 0
    let ids = tasks
        .iter()
        .map(|t| t.unwrap_or(0) as u32)
        .collect::<Vec<_>>();
    let active = tasks
        .iter()
        .map(|t| if t.is_some() { 1f32 } else { 0. })
        .collect::<Vec<_>>();
    Ok(Some(TaskRows {
        ids: Tensor::from_vec(ids, tasks.len(), device)?,
        active: Tensor::from_vec(active, tasks.len(), device)?,
    }))
}

/// Each row's question type and option markers; `None` when no sequence is a decision question.
fn decision_rows(input_seqs: &[&mut Sequence]) -> Result<Option<DecisionRows>> {
    if input_seqs.iter().all(|seq| seq.decision_input().is_none()) {
        return Ok(None);
    }
    let mut rows = DecisionRows {
        kinds: Vec::with_capacity(input_seqs.len()),
        markers: Vec::with_capacity(input_seqs.len()),
    };
    for seq in input_seqs {
        let Some(input) = seq.decision_input() else {
            anyhow::bail!("a decision batch holds a sequence without a question");
        };
        rows.kinds.push(input.kind);
        rows.markers.push(input.markers.clone());
    }
    Ok(Some(rows))
}

/// The batch's segment ids, padded and truncated like its tokens; `None` when no sequence is a pair.
fn token_type_ids(input_seqs: &[&mut Sequence], device: &Device) -> Result<Option<Tensor>> {
    if input_seqs.iter().all(|seq| seq.token_type_ids().is_none()) {
        return Ok(None);
    }
    let max_len = input_seqs
        .iter()
        .map(|seq| seq.get_toks().len())
        .max()
        .unwrap_or(0);
    let mut types = Vec::with_capacity(input_seqs.len() * max_len);
    for seq in input_seqs {
        let seq_types = seq.token_type_ids().unwrap_or_default();
        let kept = seq_types.len().min(seq.get_toks().len());
        types.extend_from_slice(&seq_types[..kept]);
        types.extend(std::iter::repeat_n(0, max_len - kept));
    }
    Ok(Some(Tensor::from_vec(
        types,
        (input_seqs.len(), max_len),
        device,
    )?))
}

pub struct EmbeddingInputsProcessor {
    pub has_causal_attention: bool,
}

impl InputsProcessor for EmbeddingInputsProcessor {
    fn process_inputs(
        &self,
        _: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        device: &Device,
        _no_kv_cache: bool,
        _last_n_context_len: Option<(usize, usize)>,
        _return_raw_logits: bool,
        _sliding_window: Option<usize>,
        _: Option<Arc<dyn Any>>,
        _paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputProcessorOutput> {
        assert!(is_prompt);

        let metadata = get_prompt_input(
            input_seqs
                .iter()
                .map(|seq| seq.get_toks())
                .collect::<Vec<_>>(),
            input_seqs,
            device,
            mapper,
            self.has_causal_attention,
            None,
        )?;
        let InnerInputProcessorOutput {
            inputs:
                InputMetadata {
                    input: input_ids,
                    flash_meta,
                },
            seq_indices,
        } = metadata;
        let extras = EmbeddingExtras {
            token_type_ids: token_type_ids(input_seqs, device)?,
            tasks: task_rows(input_seqs, device)?,
            decision: decision_rows(input_seqs)?,
        };
        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids,
            extras,
            flash_meta,
            tasks: input_seqs.iter().map(|seq| seq.embedding_task()).collect(),
        });
        Ok(InputProcessorOutput {
            inputs,
            seq_indices,
        })
    }

    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Embedding
    }
}

pub struct EmbeddingProcessor {
    pub has_causal_attention: bool,
}

impl Processor for EmbeddingProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(EmbeddingInputsProcessor {
            has_causal_attention: self.has_causal_attention,
        })
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }
    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}
