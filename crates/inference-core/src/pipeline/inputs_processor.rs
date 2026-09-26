#![allow(clippy::cast_possible_truncation)]

use std::{any::Any, sync::Arc};

use crate::paged_attention::PagedAttentionMeta;
use anyhow::Result;
use candle_core::Device;
use tokenizers::Tokenizer;

use crate::{device_map::DeviceMapper, sequence::Sequence};

#[derive(PartialEq)]
pub enum InputsProcessorType {
    Text,
    Vision,
    Embedding,
}

pub struct InputProcessorOutput {
    pub inputs: Box<dyn Any>,
    pub seq_indices: Vec<usize>,
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(crate) struct InputsProcessorValidationError(pub(crate) String);

pub(crate) fn is_inputs_processor_validation_error(error: &anyhow::Error) -> bool {
    error.chain().any(|source| {
        source
            .downcast_ref::<InputsProcessorValidationError>()
            .is_some()
    })
}

#[cfg(test)]
mod validation_error_tests {
    use super::*;

    #[test]
    fn detects_validation_errors_through_context() {
        let error = anyhow::Error::new(InputsProcessorValidationError("bad input".to_string()))
            .context("planning failed");

        assert!(is_inputs_processor_validation_error(&error));
        assert!(!is_inputs_processor_validation_error(&anyhow::anyhow!(
            "internal failure"
        )));
    }
}

/// Processor: Prepare inputs for the model (potentially preparing the images if applicable)
pub trait InputsProcessor {
    fn prepare_for_paged_prompt_planning(
        &self,
        _tokenizer: Option<Arc<Tokenizer>>,
        _input_seqs: &mut [&mut Sequence],
        _device: &Device,
        _other_config: Option<Arc<dyn Any>>,
        _paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> Result<()> {
        Ok(())
    }

    /// This should also enable matmul via f16 if prompt and the sequence length is greater than 32.
    /// Otherwise, matmul via f16 is disabled.
    ///
    /// This should return a type which can be downcasted to the proper type as used in `forward_inputs`
    #[allow(clippy::too_many_arguments)]
    fn process_inputs(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        is_xlora: bool,
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        sliding_window: Option<usize>,
        other_config: Option<Arc<dyn Any>>,
        paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputProcessorOutput>;

    fn get_type(&self) -> InputsProcessorType;
}

// ========================= Test models input processor

pub mod text_models_inputs_processor {
    use std::{any::Any, collections::HashMap, fmt::Debug, sync::Arc};

    use anyhow::Result;
    use candle_core::{Device, Tensor, WithDType};
    use tokenizers::Tokenizer;

    use crate::{
        attention::{
            flash_params::{make_flash_params, packed_rope_positions},
            FlashParams,
        },
        device_map::DeviceMapper,
        flashinfer::{
            decode_split_capacity_pages as flashinfer_decode_split_capacity_pages,
            decode_split_pages as flashinfer_decode_split_pages, flashinfer_metadata,
            flashinfer_paged_kv, flashinfer_tile_plan, flashinfer_view,
            make_paged_kv_decode_tensors, make_paged_kv_tensors,
        },
        gdn::RecurrentBatchKind,
        get_mut_arcmutex,
        paged_attention::{
            block_aligned_sliding_window_start,
            block_hash::{noncausal_mm_ranges, MultimodalAttentionPolicy},
            block_table_rows::BlockTableSnapshot,
            input_metadata::{_make_tensor_with_pad, DecodePagedRows},
            AttentionBackendKind, PagedAttentionInputMetadata, PagedAttentionMeta, _PAD_SLOT_ID,
        },
        pipeline::recurrent_batch_kind_for_input,
        sequence::Sequence,
        AdapterLease,
    };

    use super::{InputProcessorOutput, InputsProcessor, InputsProcessorType};

    pub(crate) trait NoncausalMmContext {
        fn set_noncausal_mm_context(&mut self, input_seqs: &[&mut Sequence]);
        fn set_noncausal_mm_context_views(
            &mut self,
            input_seqs: &[&mut Sequence],
            include_full_attention: bool,
        );
    }

    impl NoncausalMmContext for PagedAttentionMeta {
        fn set_noncausal_mm_context(&mut self, input_seqs: &[&mut Sequence]) {
            self.set_noncausal_mm_context_views(input_seqs, true);
        }

        fn set_noncausal_mm_context_views(
            &mut self,
            input_seqs: &[&mut Sequence],
            include_full_attention: bool,
        ) {
            self.mm_prefix_ranges_by_seq_id.clear();
            self.full_mm_prefix_ranges_by_seq_id.clear();
            for seq in input_seqs {
                let full_ranges = noncausal_mm_ranges(seq.mm_features(), None);
                if !full_ranges.is_empty() {
                    if include_full_attention {
                        self.full_mm_prefix_ranges_by_seq_id
                            .insert(*seq.id(), full_ranges.clone());
                    }
                    self.mm_prefix_ranges_by_seq_id
                        .insert(*seq.id(), full_ranges);
                }
            }
            self.has_noncausal_mm_context = !self.mm_prefix_ranges_by_seq_id.is_empty()
                || !self.full_mm_prefix_ranges_by_seq_id.is_empty();
        }
    }

    pub struct InputMetadata {
        pub input: Tensor,
        pub positions: Vec<usize>,
        pub context_lens: Vec<(usize, usize)>, // (start index, len)
        pub position_ids: Vec<usize>,
        pub paged_attn_meta: Option<PagedAttentionInputMetadata>, // For paged attention
        pub flash_meta: FlashParams,
    }

    pub struct InnerInputProcessorOutput {
        pub inputs: InputMetadata,
        pub seq_indices: Vec<usize>,
    }

    // chunk_offset_toks is the number of tokens by which the tokens are offset,
    // chunk_offset_toks / prompt_chunksize = number of batches
    //
    // prefix_cache_lens: when provided, indicates how many tokens per sequence are already
    // cached in the paged KV cache. Only new (non-cached) tokens will be included in the
    // input tensor, and slot_mappings will only cover new token slots. Block tables still
    // cover the entire context so that context_attention_fwd can read cached blocks.
    #[allow(clippy::too_many_arguments)]
    pub fn make_prompt_chunk<T: WithDType + Debug>(
        chunk_offset_toks: usize,
        toks: Vec<&[T]>,
        seq_ids: &[usize],
        device: &Device,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
        prefix_cache_lens: Option<&[usize]>,
        sliding_window: Option<usize>,
        allow_packed_prefill: bool,
    ) -> Result<InputMetadata> {
        // Determine effective tokens per sequence after prefix cache trimming
        let effective_lens: Vec<usize> = toks
            .iter()
            .enumerate()
            .map(|(i, seq)| {
                let cached = prefix_cache_lens.map_or(0, |lens| lens[i]);
                seq.len().saturating_sub(cached)
            })
            .collect();
        let max_len = *effective_lens.iter().max().expect("No sequences");
        let padding_tok = T::zero();
        let has_any_cache_hit = prefix_cache_lens.is_some_and(|lens| lens.iter().any(|&l| l > 0));
        let prompt_chunk_causal = paged_attn_metadata.as_ref().is_none_or(|metadata| {
            metadata.prompt_chunk_attention_policy == MultimodalAttentionPolicy::Causal
        });
        let all_model_devices_cuda = mapper.is_none_or(|mapper| {
            mapper
                .get_unique_devices()
                .iter()
                .all(|device| device.is_cuda())
        });
        let packed_prefill = allow_packed_prefill
            && effective_lens.len() > 1
            && effective_lens.iter().any(|len| *len != max_len)
            && device.is_cuda()
            && all_model_devices_cuda
            && crate::using_flash_attn()
            && !return_raw_logits
            && last_n_context_len.is_none()
            && chunk_offset_toks == 0
            && !has_any_cache_hit
            && paged_attn_metadata.as_ref().is_some_and(|metadata| {
                metadata.enable_packed_prefill
                    && metadata.is_final_prompt_chunk
                    && metadata.prompt_chunk_attention_policy == MultimodalAttentionPolicy::Causal
            });
        if packed_prefill {
            tracing::debug!(
                sequences = effective_lens.len(),
                tokens = effective_lens.iter().sum::<usize>(),
                padded_tokens = effective_lens.len() * max_len,
                "Using packed prompt prefill"
            );
        }
        let mut seqs_tensors = Vec::new();
        let mut seqlen_offsets = Vec::new();
        let mut context_lens = Vec::new();
        let mut position_ids = Vec::new();
        let mut slot_mappings = Vec::new();
        let mut block_tables = Vec::new();
        let mut full_block_tables = Vec::new();
        let mut paged_attn_context_lens = Vec::new();
        let mut full_paged_attn_context_lens = Vec::new();
        let flash_attn = crate::using_flash_attn();
        let mut seqlens_q = if flash_attn { vec![0] } else { Vec::new() };
        let mut seqlens_k = if flash_attn { vec![0] } else { Vec::new() };
        let mut num_cached_tokens_vec: Vec<usize> = Vec::new();
        let mut query_lens_vec: Vec<usize> = Vec::new();
        for (seq_idx, (seq_id, ctxt)) in seq_ids.iter().zip(&toks).enumerate() {
            let cached = prefix_cache_lens.map_or(0, |lens| lens[seq_idx]);
            let full_prompt_len = ctxt.len();
            // The new (non-cached) tokens to process
            let new_toks = &ctxt[cached..];
            let new_len = new_toks.len();

            let offset = last_n_context_len.unwrap_or_default();
            // seqlen_offset includes cached prefix so position IDs are correct
            seqlen_offsets.push(offset.1 + chunk_offset_toks + cached);

            position_ids.push(new_len + chunk_offset_toks + cached);
            let mut input_toks = new_toks.to_vec();
            if !packed_prefill {
                input_toks.extend(std::iter::repeat_n(
                    padding_tok,
                    max_len.saturating_sub(input_toks.len()),
                ));
            }
            // If we are returning raw logits, we want to not trim the logits at all.
            if return_raw_logits {
                if last_n_context_len.is_some() {
                    anyhow::bail!("`return_raw_logits` is incompatible with `last_n_context_len`");
                }

                context_lens.push((0, input_toks.len()));
            } else {
                context_lens.push((
                    new_len.saturating_sub(last_n_context_len.map(|(a, _)| a).unwrap_or(1)),
                    last_n_context_len.map(|(a, _)| a).unwrap_or(1),
                ));
            }

            if flash_attn {
                seqlens_q.push(input_toks.len() as u32);
                seqlens_k.push((input_toks.len() + chunk_offset_toks + cached) as u32);
            }

            seqs_tensors.push(Tensor::new(input_toks, device)?.unsqueeze(0)?);

            if has_any_cache_hit {
                num_cached_tokens_vec.push(cached);
                query_lens_vec.push(new_len);
            }

            if let Some(paged_attn_metadata) = &mut paged_attn_metadata {
                let kv_mgr = get_mut_arcmutex!(paged_attn_metadata.kv_cache_manager);
                let block_ids = kv_mgr.get_block_ids(*seq_id);

                if block_ids.is_none() {
                    // Will be None during profiling.
                    slot_mappings.push([_PAD_SLOT_ID].repeat(new_len));
                    continue;
                }
                let table: Vec<usize> = block_ids.unwrap().to_vec();
                drop(kv_mgr);

                // Slot mappings only for new tokens (cached tokens are already in cache)
                let slot_start = cached + chunk_offset_toks;
                let slot_end = full_prompt_len + chunk_offset_toks;
                let mut slot_mapping = Vec::new();
                let mut ctxt_len = Vec::new();
                for i in slot_start..slot_end {
                    ctxt_len.push(i);

                    let block_number = if i / paged_attn_metadata.block_size >= table.len() {
                        panic!(
                            "Block table is too small (prompt)! i={} block_size={} table_len={}",
                            i,
                            paged_attn_metadata.block_size,
                            table.len()
                        );
                    } else {
                        table.get(i / paged_attn_metadata.block_size).unwrap()
                    };
                    let block_offset = i % paged_attn_metadata.block_size;
                    // Use checked arithmetic to prevent overflow
                    let slot = block_number
                        .checked_mul(paged_attn_metadata.block_size)
                        .and_then(|v| v.checked_add(block_offset))
                        .expect("Slot calculation overflowed");
                    slot_mapping.push(
                        slot.try_into()
                            .expect("Slot value too large for target integer type"),
                    );
                }
                slot_mappings.push(slot_mapping);
                let full_context_len = chunk_offset_toks + cached + new_len;
                full_block_tables.push(table.clone());
                full_paged_attn_context_lens.push(full_context_len);

                if let Some(sliding_window) = paged_attn_metadata.sliding_window {
                    let mut block_aligned_start = block_aligned_sliding_window_start(
                        full_context_len,
                        new_len,
                        sliding_window,
                        paged_attn_metadata.block_size,
                    );
                    if let Some(mm_start) = paged_attn_metadata
                        .mm_prefix_ranges_by_seq_id
                        .get(seq_id)
                        .into_iter()
                        .flatten()
                        .filter(|&&(start, end)| start < slot_end && slot_start < end)
                        .map(|&(start, _)| start)
                        .min()
                    {
                        block_aligned_start = block_aligned_start.min(
                            mm_start / paged_attn_metadata.block_size
                                * paged_attn_metadata.block_size,
                        );
                    }
                    let paged_context_len = full_context_len - block_aligned_start;
                    let slide_idx = block_aligned_start / paged_attn_metadata.block_size;
                    let needed_blocks = paged_context_len.div_ceil(paged_attn_metadata.block_size);
                    let slide_end = (slide_idx + needed_blocks).min(table.len());
                    block_tables.push(table.get(slide_idx..slide_end).unwrap_or(&[]).to_vec());
                    paged_attn_context_lens.push((0..paged_context_len).collect());
                } else {
                    block_tables.push(table.clone());
                    paged_attn_context_lens.push(ctxt_len);
                }
            }
        }

        let flash_meta = if flash_attn {
            make_flash_params(
                device,
                mapper,
                &seqlens_q,
                &seqlens_k,
                sliding_window,
                prompt_chunk_causal,
                packed_prefill,
            )?
        } else {
            FlashParams::empty(prompt_chunk_causal)
        };

        let input_concat_dim = if packed_prefill { 1 } else { 0 };
        let input = Tensor::cat(&seqs_tensors, input_concat_dim).unwrap();

        let paged_attn_meta = if let Some(paged_attn_metadata) = &paged_attn_metadata {
            // Create paged attention tensors on CPU first (see comment above about CUDA contexts)
            let prefill_query_lens = slot_mappings.iter().map(Vec::len).collect::<Vec<_>>();
            let slot_mappings = if packed_prefill {
                let slots = slot_mappings.into_iter().flatten().collect::<Vec<_>>();
                let slot_count = slots.len();
                Tensor::from_vec(slots, (slot_count,), &Device::Cpu)?
            } else {
                let max_slot_mapping_len = slot_mappings.iter().map(Vec::len).max().unwrap();
                _make_tensor_with_pad(
                    slot_mappings,
                    max_slot_mapping_len,
                    _PAD_SLOT_ID,
                    &Device::Cpu,
                )?
            };

            let max_block_table_len = block_tables.iter().map(|x| x.len()).max().unwrap();
            let block_size = paged_attn_metadata.block_size;
            let full_context_lens_for_fi = if has_any_cache_hit {
                num_cached_tokens_vec
                    .iter()
                    .zip(query_lens_vec.iter())
                    .map(|(cached, query_len)| cached + query_len)
                    .collect::<Vec<_>>()
            } else {
                prefill_query_lens.clone()
            };
            let paged_context_lens_for_fi = if sliding_window.is_some() {
                paged_attn_context_lens
                    .iter()
                    .map(Vec::len)
                    .collect::<Vec<_>>()
            } else {
                full_context_lens_for_fi.clone()
            };
            let (paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len) =
                make_paged_kv_tensors(
                    &block_tables,
                    &paged_context_lens_for_fi,
                    block_size,
                    block_tables.len() * max_block_table_len,
                )?;
            let decode_split_pages = flashinfer_decode_split_pages(
                block_size,
                block_tables.len(),
                paged_attn_metadata.prefill_key_value_heads,
                paged_context_lens_for_fi.iter().copied().max().unwrap_or(0),
            );
            let tiles_per_row = max_block_table_len
                .max(1)
                .div_ceil(flashinfer_decode_split_capacity_pages(block_size));
            let (request_indices, kv_tile_indices, o_indptr, kv_chunk_size, block_valid_mask) =
                make_paged_kv_decode_tensors(
                    &block_tables,
                    &paged_context_lens_for_fi,
                    block_size,
                    Some(decode_split_pages),
                    block_tables.len() * tiles_per_row,
                )?;
            let block_tables = _make_tensor_with_pad(
                block_tables
                    .iter()
                    .map(|x| x.iter().map(|x| *x as u32).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                max_block_table_len,
                0,
                &Device::Cpu,
            )?;
            let block_tables = block_tables.reshape(((), max_block_table_len))?;

            let max_context_len = paged_attn_context_lens
                .iter()
                .map(|x| x.len())
                .max()
                .unwrap();

            let context_lens = _make_tensor_with_pad(
                paged_attn_context_lens
                    .iter()
                    .map(|x| x.iter().map(|x| *x as u32).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                max_context_len,
                0,
                &Device::Cpu,
            )?
            .reshape(((),))?;
            let packed_rope_positions = if packed_prefill {
                let positions = packed_rope_positions(&seqlen_offsets, &prefill_query_lens)?;
                let position_count = positions.len();
                Some(Tensor::from_vec(
                    positions,
                    (position_count,),
                    &Device::Cpu,
                )?)
            } else {
                None
            };

            // For device mapping, make a copy of each tensor for each device
            let devices = mapper.unwrap().get_unique_devices();
            let mut slot_mappings_map = HashMap::new();
            let mut rope_positions_map = HashMap::new();
            let mut block_tables_map = HashMap::new();
            let mut context_lens_map = HashMap::new();
            let mut mm_prefix_ranges_map = HashMap::new();
            let mut full_mm_prefix_ranges_map = HashMap::new();
            let mut full_block_tables_map = HashMap::new();
            let mut full_context_lens_map = HashMap::new();
            let mut paged_kv_indptr_map = HashMap::new();
            let mut paged_kv_indices_map = HashMap::new();
            let mut paged_kv_last_page_len_map = HashMap::new();
            let mut request_indices_map = HashMap::new();
            let mut kv_tile_indices_map = HashMap::new();
            let mut o_indptr_map = HashMap::new();
            let mut kv_chunk_size_map = HashMap::new();
            let mut block_valid_mask_map = HashMap::new();
            let mut full_paged_kv_indptr_map = HashMap::new();
            let mut full_paged_kv_indices_map = HashMap::new();
            let mut full_paged_kv_last_page_len_map = HashMap::new();
            let mut full_request_indices_map = HashMap::new();
            let mut full_kv_tile_indices_map = HashMap::new();
            let mut full_o_indptr_map = HashMap::new();
            let mut full_kv_chunk_size_map = HashMap::new();
            let mut full_block_valid_mask_map = HashMap::new();

            let (
                full_block_tables_tensor,
                full_context_lens_tensor,
                full_max_context_len,
                full_paged_kv_tensors,
                full_decode_tensors,
            ) = if sliding_window.is_some() {
                let full_max_block_table_len =
                    full_block_tables.iter().map(|x| x.len()).max().unwrap_or(1);
                let full_paged_kv_tensors = Some(make_paged_kv_tensors(
                    &full_block_tables,
                    &full_paged_attn_context_lens,
                    block_size,
                    full_block_tables.len() * full_max_block_table_len,
                )?);
                let full_decode_split_pages = flashinfer_decode_split_pages(
                    block_size,
                    full_block_tables.len(),
                    paged_attn_metadata.prefill_key_value_heads,
                    full_context_lens_for_fi.iter().copied().max().unwrap_or(0),
                );
                let full_tiles_per_row = full_max_block_table_len
                    .max(1)
                    .div_ceil(flashinfer_decode_split_capacity_pages(block_size));
                let full_decode_tensors = Some(make_paged_kv_decode_tensors(
                    &full_block_tables,
                    &full_paged_attn_context_lens,
                    block_size,
                    Some(full_decode_split_pages),
                    full_block_tables.len() * full_tiles_per_row,
                )?);
                let full_block_tables_tensor = _make_tensor_with_pad(
                    full_block_tables
                        .iter()
                        .map(|x| x.iter().map(|x| *x as u32).collect::<Vec<_>>())
                        .collect::<Vec<_>>(),
                    full_max_block_table_len,
                    0,
                    &Device::Cpu,
                )?
                .reshape(((), full_max_block_table_len))?;
                let full_context_lens_tensor = Tensor::from_vec(
                    full_paged_attn_context_lens
                        .iter()
                        .map(|x| *x as u32)
                        .collect::<Vec<_>>(),
                    (full_paged_attn_context_lens.len(),),
                    &Device::Cpu,
                )?;
                let full_max_context_len = full_paged_attn_context_lens.iter().copied().max();
                (
                    Some(full_block_tables_tensor),
                    Some(full_context_lens_tensor),
                    full_max_context_len,
                    full_paged_kv_tensors,
                    full_decode_tensors,
                )
            } else {
                (None, None, None, None, None)
            };
            let kv_window_starts = full_paged_attn_context_lens
                .iter()
                .zip(paged_context_lens_for_fi.iter())
                .map(|(full_len, paged_len)| full_len.saturating_sub(*paged_len))
                .collect::<Vec<_>>();
            let mm_prefix_ranges_tensor = crate::paged_attention::mm_prefix::make_ranges_tensor(
                seq_ids,
                &paged_attn_metadata.mm_prefix_ranges_by_seq_id,
                &kv_window_starts,
                &paged_context_lens_for_fi,
                &prefill_query_lens,
            )?;
            let full_kv_window_starts = vec![0; seq_ids.len()];
            let full_mm_prefix_ranges_tensor = if sliding_window.is_some() {
                crate::paged_attention::mm_prefix::make_ranges_tensor(
                    seq_ids,
                    &paged_attn_metadata.full_mm_prefix_ranges_by_seq_id,
                    &full_kv_window_starts,
                    &full_paged_attn_context_lens,
                    &prefill_query_lens,
                )?
            } else {
                None
            };

            for device in devices {
                slot_mappings_map
                    .insert(device.location(), slot_mappings.clone().to_device(&device)?);
                if let Some(positions) = &packed_rope_positions {
                    rope_positions_map
                        .insert(device.location(), positions.clone().to_device(&device)?);
                }
                block_tables_map
                    .insert(device.location(), block_tables.clone().to_device(&device)?);
                context_lens_map
                    .insert(device.location(), context_lens.clone().to_device(&device)?);
                if let Some(mm_prefix_ranges_tensor) = &mm_prefix_ranges_tensor {
                    mm_prefix_ranges_map.insert(
                        device.location(),
                        mm_prefix_ranges_tensor.clone().to_device(&device)?,
                    );
                }
                if let Some(full_mm_prefix_ranges_tensor) = &full_mm_prefix_ranges_tensor {
                    full_mm_prefix_ranges_map.insert(
                        device.location(),
                        full_mm_prefix_ranges_tensor.clone().to_device(&device)?,
                    );
                }
                paged_kv_indptr_map.insert(
                    device.location(),
                    paged_kv_indptr.clone().to_device(&device)?,
                );
                paged_kv_indices_map.insert(
                    device.location(),
                    paged_kv_indices.clone().to_device(&device)?,
                );
                paged_kv_last_page_len_map.insert(
                    device.location(),
                    paged_kv_last_page_len.clone().to_device(&device)?,
                );
                request_indices_map.insert(
                    device.location(),
                    request_indices.clone().to_device(&device)?,
                );
                kv_tile_indices_map.insert(
                    device.location(),
                    kv_tile_indices.clone().to_device(&device)?,
                );
                o_indptr_map.insert(device.location(), o_indptr.clone().to_device(&device)?);
                kv_chunk_size_map
                    .insert(device.location(), kv_chunk_size.clone().to_device(&device)?);
                block_valid_mask_map.insert(
                    device.location(),
                    block_valid_mask.clone().to_device(&device)?,
                );
                if let Some(full_block_tables_tensor) = &full_block_tables_tensor {
                    full_block_tables_map.insert(
                        device.location(),
                        full_block_tables_tensor.clone().to_device(&device)?,
                    );
                }
                if let Some(full_context_lens_tensor) = &full_context_lens_tensor {
                    full_context_lens_map.insert(
                        device.location(),
                        full_context_lens_tensor.clone().to_device(&device)?,
                    );
                }
                if let Some((indptr, indices, last_page_len)) = &full_paged_kv_tensors {
                    full_paged_kv_indptr_map
                        .insert(device.location(), indptr.clone().to_device(&device)?);
                    full_paged_kv_indices_map
                        .insert(device.location(), indices.clone().to_device(&device)?);
                    full_paged_kv_last_page_len_map
                        .insert(device.location(), last_page_len.clone().to_device(&device)?);
                }
                if let Some((req, kv, o, chunk, valid)) = &full_decode_tensors {
                    full_request_indices_map
                        .insert(device.location(), req.clone().to_device(&device)?);
                    full_kv_tile_indices_map
                        .insert(device.location(), kv.clone().to_device(&device)?);
                    full_o_indptr_map.insert(device.location(), o.clone().to_device(&device)?);
                    full_kv_chunk_size_map
                        .insert(device.location(), chunk.clone().to_device(&device)?);
                    full_block_valid_mask_map
                        .insert(device.location(), valid.clone().to_device(&device)?);
                }
            }

            let prompt_chunk_attention_policy = paged_attn_metadata.prompt_chunk_attention_policy;
            let sliding_flashinfer_view = if sliding_window.is_some() {
                Some(flashinfer_view(
                    Some(block_tables_map.clone()),
                    Some(context_lens_map.clone()),
                    Some(max_context_len),
                    flashinfer_paged_kv(
                        paged_kv_indptr_map.clone(),
                        paged_kv_indices_map.clone(),
                        paged_kv_last_page_len_map.clone(),
                    ),
                    flashinfer_tile_plan(
                        request_indices_map.clone(),
                        kv_tile_indices_map.clone(),
                        o_indptr_map.clone(),
                        kv_chunk_size_map.clone(),
                        block_valid_mask_map.clone(),
                    ),
                ))
            } else {
                None
            };
            let logical_flashinfer_view = if sliding_window.is_some() {
                flashinfer_view(
                    Some(full_block_tables_map.clone()),
                    Some(full_context_lens_map.clone()),
                    full_max_context_len,
                    flashinfer_paged_kv(
                        full_paged_kv_indptr_map.clone(),
                        full_paged_kv_indices_map.clone(),
                        full_paged_kv_last_page_len_map.clone(),
                    ),
                    flashinfer_tile_plan(
                        full_request_indices_map.clone(),
                        full_kv_tile_indices_map.clone(),
                        full_o_indptr_map.clone(),
                        full_kv_chunk_size_map.clone(),
                        full_block_valid_mask_map.clone(),
                    ),
                )
            } else {
                flashinfer_view(
                    Some(block_tables_map.clone()),
                    Some(context_lens_map.clone()),
                    Some(max_context_len),
                    flashinfer_paged_kv(
                        paged_kv_indptr_map.clone(),
                        paged_kv_indices_map.clone(),
                        paged_kv_last_page_len_map.clone(),
                    ),
                    flashinfer_tile_plan(
                        request_indices_map.clone(),
                        kv_tile_indices_map.clone(),
                        o_indptr_map.clone(),
                        kv_chunk_size_map.clone(),
                        block_valid_mask_map.clone(),
                    ),
                )
            };
            let flashinfer = Some(flashinfer_metadata(
                logical_flashinfer_view,
                sliding_flashinfer_view,
            ));

            Some(PagedAttentionInputMetadata {
                slot_mappings: slot_mappings_map,
                block_tables: Some(block_tables_map),
                context_lens: Some(context_lens_map),
                block_size: Some(block_size),
                paged_context_lens_cpu: Some(paged_context_lens_for_fi.clone()),
                full_paged_context_lens_cpu: Some(full_paged_attn_context_lens.clone()),
                max_context_len: Some(max_context_len),
                full_block_tables: if full_block_tables_map.is_empty() {
                    None
                } else {
                    Some(full_block_tables_map)
                },
                full_context_lens: if full_context_lens_map.is_empty() {
                    None
                } else {
                    Some(full_context_lens_map)
                },
                full_max_context_len,
                is_first_prompt_chunk: chunk_offset_toks == 0 && !has_any_cache_hit,
                is_final_prompt_chunk: paged_attn_metadata.is_final_prompt_chunk,
                needs_logits: paged_attn_metadata.needs_logits,
                prompt_chunk_attention_policy,
                // Keep the slow path local to chunks whose query rows overlap a noncausal range.
                has_noncausal_mm_context: mm_prefix_ranges_tensor.is_some()
                    || full_mm_prefix_ranges_tensor.is_some(),
                prefix_gather_workspace_limit: paged_attn_metadata.prefix_gather_workspace_limit,
                mm_prefix_ranges: if mm_prefix_ranges_map.is_empty() {
                    None
                } else {
                    Some(mm_prefix_ranges_map)
                },
                full_mm_prefix_ranges: if full_mm_prefix_ranges_map.is_empty() {
                    None
                } else {
                    Some(full_mm_prefix_ranges_map)
                },
                prefill_attention_heads: paged_attn_metadata.prefill_attention_heads,
                prefill_key_value_heads: paged_attn_metadata.prefill_key_value_heads,
                prefill_head_dim: paged_attn_metadata.prefill_head_dim,
                flashinfer,
                rope_positions: if rope_positions_map.is_empty() {
                    None
                } else {
                    Some(rope_positions_map)
                },
                num_cached_tokens: if has_any_cache_hit {
                    Some(num_cached_tokens_vec.clone())
                } else {
                    None
                },
                // Always set: saves a per-layer GPU->CPU slot-mapping sync in forward_prefix.
                query_lens: Some(prefill_query_lens.clone()),
                cu_seqlens_q: if has_any_cache_hit {
                    // Cumulative query lengths for Sdpa varlen: [0, q0, q0+q1, ...]
                    let mut cu_q = vec![0u32];
                    for &ql in &query_lens_vec {
                        cu_q.push(cu_q.last().unwrap() + ql as u32);
                    }
                    let cu_q_t = Tensor::new(&cu_q[..], &Device::Cpu)?;
                    let devices = mapper.unwrap().get_unique_devices();
                    let mut map = HashMap::new();
                    for device in &devices {
                        map.insert(device.location(), cu_q_t.to_device(device)?);
                    }
                    Some(map)
                } else {
                    None
                },
                cu_seqlens_kv: if has_any_cache_hit {
                    // Cumulative KV lengths: [0, c0+q0, c0+q0+c1+q1, ...]
                    // U32 to match flash-attn varlen expectations
                    let mut cu_kv = vec![0u32];
                    for (&nc, &ql) in num_cached_tokens_vec.iter().zip(query_lens_vec.iter()) {
                        cu_kv.push(cu_kv.last().unwrap() + (nc + ql) as u32);
                    }
                    let cu_kv_t = Tensor::new(&cu_kv[..], &Device::Cpu)?;
                    let devices = mapper.unwrap().get_unique_devices();
                    let mut map = HashMap::new();
                    for device in &devices {
                        map.insert(device.location(), cu_kv_t.to_device(device)?);
                    }
                    Some(map)
                } else {
                    None
                },
                decode_rows: None,
            })
        } else {
            None
        };

        Ok(InputMetadata {
            input,
            positions: seqlen_offsets,
            context_lens,
            position_ids,
            paged_attn_meta,
            flash_meta,
        })
    }

    fn completion_input_tensor<T: WithDType>(
        host_tokens: Vec<T>,
        batch: usize,
        host_width: usize,
        staged_device_rows: &[Tensor],
        device: &Device,
    ) -> Result<Tensor> {
        let host = Tensor::from_vec(host_tokens, (batch, host_width), device)?;
        if staged_device_rows.is_empty() {
            return Ok(host);
        }
        let staged_device_rows = staged_device_rows
            .iter()
            .map(|tokens| tokens.to_device(device)?.to_dtype(T::DTYPE))
            .collect::<candle_core::Result<Vec<_>>>()?;
        #[cfg(feature = "cuda")]
        if T::DTYPE == candle_core::DType::U32 && device.is_cuda() {
            return crate::cuda::input_packing::pack_completion_input(&host, &staged_device_rows)
                .map_err(anyhow::Error::msg);
        }
        let staged = Tensor::stack(&staged_device_rows, 0)?;
        Ok(Tensor::cat(&[&host, &staged], 1)?)
    }

    fn make_completion_chunk<T: WithDType + From<u32> + Clone + std::fmt::Debug>(
        toks: Vec<&[T]>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
        sliding_window: Option<usize>,
        decode_window: usize,
    ) -> Result<InputMetadata> {
        // Pad each sequence by the padding token to the max len.
        let flash_attn = crate::using_flash_attn();
        let mut input_tokens = Vec::new();
        let mut input_width = None;
        let mut seqlen_offsets = Vec::new();
        let mut context_lens = Vec::new();
        let mut position_ids = Vec::new();

        let mut slot_mappings = Vec::new();
        let mut paged_attn_context_lens = Vec::new();
        let mut full_paged_attn_context_lens = Vec::new();
        let mut seqlens_q = if flash_attn { vec![0] } else { Vec::new() };
        let mut seqlens_k = if flash_attn { vec![0] } else { Vec::new() };
        // Staged speculative tokens are appended to the decode input only when
        // the whole batch has the same fixed proposal width. The generic
        // verifier keeps the target forward rectangular in this first batched
        // implementation; mixed staged/no-staged batches fall back to a normal
        // one-token decode and the driver clears the stale staged proposals.
        let use_staged_speculative =
            crate::speculative::staging::staged_batch_width(input_seqs).is_some();
        let use_device_staged = use_staged_speculative
            && input_seqs
                .iter()
                .any(|seq| seq.active_staged_speculative_tokens().as_device().is_some());
        let sequence_block_tables = paged_attn_metadata.as_ref().map(|paged_attn_metadata| {
            let kv_mgr = get_mut_arcmutex!(paged_attn_metadata.kv_cache_manager);
            input_seqs
                .iter()
                .map(|seq| {
                    Arc::<[usize]>::from(
                        kv_mgr
                            .get_block_ids(*seq.id())
                            .expect("Sequence must have allocated blocks for completion"),
                    )
                })
                .collect::<Vec<_>>()
        });
        let mut host_input_width = None;
        for (seq_idx, (seq, ctxt)) in input_seqs.iter().zip(toks).enumerate() {
            let staged_speculative = if use_staged_speculative && !use_device_staged {
                seq.active_staged_speculative_tokens()
                    .as_host()
                    .expect("host-backed speculative batch changed storage kind")
            } else {
                &[]
            };
            let start_pos = ctxt.len().saturating_sub(decode_window);
            let mut ctxt = ctxt[start_pos..].to_vec();
            ctxt.extend(staged_speculative.iter().copied().map(T::from));
            let host_width = ctxt.len();
            let query_len = host_width
                + if use_device_staged {
                    seq.active_staged_speculative_len()
                } else {
                    0
                };
            let effective_context_len = start_pos + query_len;
            seqlen_offsets.push(start_pos);
            context_lens.push((0, query_len));
            position_ids.push(effective_context_len);

            if flash_attn {
                seqlens_q.push(query_len as u32);
                seqlens_k.push(effective_context_len as u32);
            }

            match input_width {
                Some(width) if width != query_len => {
                    anyhow::bail!("completion input rows must have one query width")
                }
                None => input_width = Some(query_len),
                Some(_) => {}
            }
            match host_input_width {
                Some(width) if width != host_width => {
                    anyhow::bail!("completion input host rows must have one query width")
                }
                None => host_input_width = Some(host_width),
                Some(_) => {}
            }
            input_tokens.extend(ctxt);

            if let Some(paged_attn_metadata) = &mut paged_attn_metadata {
                let table = &sequence_block_tables
                    .as_ref()
                    .expect("paged block tables were snapshotted")[seq_idx];

                let block_start = start_pos - seq.token_offset();
                let block_end = block_start + query_len;
                let mut slot_mapping = Vec::with_capacity(query_len);
                for block_pos in block_start..block_end {
                    let block_number = if block_pos / paged_attn_metadata.block_size >= table.len()
                    {
                        panic!("Block table is too small (completion)! block_pos={} block_size={} table_len={}", block_pos, paged_attn_metadata.block_size, table.len());
                    } else {
                        table
                            .get(block_pos / paged_attn_metadata.block_size)
                            .unwrap()
                    };
                    let block_offset = block_pos % paged_attn_metadata.block_size;
                    // Use checked arithmetic to prevent overflow
                    let slot = block_number
                        .checked_mul(paged_attn_metadata.block_size)
                        .and_then(|v| v.checked_add(block_offset))
                        .expect("Slot calculation overflowed");
                    let slot = slot
                        .try_into()
                        .expect("Slot value too large for target integer type");
                    slot_mapping.push(slot);
                }
                slot_mappings.push(slot_mapping);

                for row in 0..query_len {
                    let full_context_len = start_pos + row + 1;

                    full_paged_attn_context_lens.push(full_context_len);

                    let paged_attn_context_len = if let Some(sliding_window) =
                        paged_attn_metadata.sliding_window
                    {
                        let window_start = full_context_len.saturating_sub(sliding_window);
                        let block_aligned_start = (window_start / paged_attn_metadata.block_size)
                            * paged_attn_metadata.block_size;
                        full_context_len - block_aligned_start
                    } else {
                        full_context_len
                    };
                    paged_attn_context_lens.push(paged_attn_context_len);
                }
            }
        }

        let paged_single_token_decode = paged_attn_metadata.is_some()
            && context_lens.iter().all(|&(_, query_len)| query_len == 1);
        let flash_meta = if flash_attn && !paged_single_token_decode {
            make_flash_params(
                device,
                mapper,
                &seqlens_q,
                &seqlens_k,
                sliding_window,
                true,
                false,
            )?
        } else {
            FlashParams::empty(true)
        };

        let paged_attn_meta = if let Some(paged_attn_input) = &paged_attn_metadata {
            let query_len = context_lens.first().map_or(1, |(_, q)| *q);
            let block_tables = BlockTableSnapshot::from_sequence_tables(
                sequence_block_tables.expect("paged block tables were snapshotted"),
                query_len,
            );
            let rows = Arc::new(DecodePagedRows {
                slot_mappings,
                block_tables,
                context_lens: paged_attn_context_lens,
                full_context_lens: full_paged_attn_context_lens,
                query_len,
                block_size: paged_attn_input.block_size,
                use_standard_metadata: paged_attn_input.attention_backend
                    == AttentionBackendKind::Standard,
                max_paged_context_len: paged_attn_input.max_paged_context_len,
                sliding_window: paged_attn_input.sliding_window,
                decode_window,
                devices: mapper.unwrap().get_unique_devices(),
                num_kv_heads: paged_attn_input.prefill_key_value_heads,
            });
            Some(rows.build()?)
        } else {
            None
        };

        let staged_device_rows = if use_device_staged {
            input_seqs
                .iter()
                .map(|seq| match seq.active_staged_speculative_tokens() {
                    crate::speculative::SpeculativeTokens::Host(tokens) => {
                        Tensor::new(tokens.as_slice(), device)
                    }
                    crate::speculative::SpeculativeTokens::Device(tokens) => Ok(tokens.clone()),
                })
                .collect::<candle_core::Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let input = completion_input_tensor(
            input_tokens,
            input_seqs.len(),
            host_input_width.unwrap_or_default(),
            &staged_device_rows,
            device,
        )?;
        if input.dims() != [input_seqs.len(), input_width.unwrap_or_default()] {
            anyhow::bail!("completion input tensor shape changed while staging proposals");
        }

        Ok(InputMetadata {
            input,
            positions: seqlen_offsets,
            context_lens,
            position_ids,
            paged_attn_meta,
            flash_meta,
        })
    }

    #[cfg(feature = "models-gemma")]
    #[allow(clippy::too_many_arguments)]
    fn make_completion_prefill_chunk<T: WithDType + std::fmt::Debug>(
        toks: Vec<&[T]>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
        sliding_window: Option<usize>,
        decode_window: usize,
    ) -> Result<InputMetadata> {
        let prefix_cache_lens = toks
            .iter()
            .map(|ctxt| ctxt.len().saturating_sub(decode_window))
            .collect::<Vec<_>>();
        make_prompt_chunk(
            0,
            toks,
            &input_seqs.iter().map(|seq| *seq.id()).collect::<Vec<_>>(),
            device,
            last_n_context_len,
            return_raw_logits,
            paged_attn_metadata,
            mapper,
            Some(&prefix_cache_lens),
            sliding_window,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_prompt_input<T: WithDType + std::fmt::Debug>(
        toks: Vec<&[T]>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
        sliding_window: Option<usize>,
    ) -> Result<InnerInputProcessorOutput> {
        let offset = input_seqs[0].token_offset();
        // Collect prefix cache lens when paged attention is in use
        let prefix_cache_lens: Vec<usize> =
            input_seqs.iter().map(|s| s.prefix_cache_len()).collect();
        let has_paged_attn = paged_attn_metadata.is_some();
        make_prompt_chunk(
            offset,
            toks,
            &input_seqs.iter().map(|s| *s.id()).collect::<Vec<_>>(),
            device,
            last_n_context_len,
            return_raw_logits,
            paged_attn_metadata,
            mapper,
            if has_paged_attn {
                Some(&prefix_cache_lens)
            } else {
                None
            },
            sliding_window,
            true,
        )
        .map(|inputs| InnerInputProcessorOutput {
            inputs,
            seq_indices: (0..input_seqs.len()).collect(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_completion_input<T: WithDType + std::fmt::Debug + From<u32> + Clone>(
        toks: Vec<&[T]>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
        sliding_window: Option<usize>,
    ) -> Result<InnerInputProcessorOutput> {
        if no_kv_cache {
            return get_prompt_input(
                toks,
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata,
                mapper,
                None,
            );
        }

        make_completion_chunk(
            toks,
            input_seqs,
            device,
            paged_attn_metadata,
            mapper,
            sliding_window,
            1,
        )
        .map(|inputs| InnerInputProcessorOutput {
            inputs,
            seq_indices: (0..input_seqs.len()).collect(),
        })
    }

    /// `get_completion_input` for models that consume more than one new token per decode step
    /// (e.g. block diffusion, where each step feeds the last committed canvas to the encoder).
    #[cfg(feature = "models-gemma")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_completion_input_windowed<
        T: WithDType + std::fmt::Debug + From<u32> + Clone,
    >(
        toks: Vec<&[T]>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
        sliding_window: Option<usize>,
        decode_window: usize,
    ) -> Result<InnerInputProcessorOutput> {
        if no_kv_cache {
            return get_prompt_input(
                toks,
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata,
                mapper,
                None,
            );
        }

        let inputs = if paged_attn_metadata.is_some() {
            make_completion_prefill_chunk(
                toks,
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata,
                mapper,
                sliding_window,
                decode_window,
            )
        } else {
            make_completion_chunk(
                toks,
                input_seqs,
                device,
                paged_attn_metadata,
                mapper,
                sliding_window,
                decode_window,
            )
        }?;
        Ok(InnerInputProcessorOutput {
            inputs,
            seq_indices: (0..input_seqs.len()).collect(),
        })
    }

    #[derive(Clone)]
    pub struct ModelInputs {
        pub input_ids: Tensor,
        pub input_ids_full: Option<Tensor>,
        pub seqlen_offsets: Vec<usize>,
        pub seqlen_offsets_full: Option<Vec<usize>>,
        pub context_lens: Vec<(usize, usize)>,
        pub position_ids: Vec<usize>,
        pub paged_attn_meta: Option<PagedAttentionInputMetadata>,
        pub flash_meta: FlashParams,
        pub flash_meta_full: Option<FlashParams>,
        pub recurrent_batch_kind: RecurrentBatchKind,
        pub adapter_leases: Arc<[Option<AdapterLease>]>,
    }

    fn adapter_leases(
        input_seqs: &[&mut Sequence],
        seq_indices: &[usize],
    ) -> Arc<[Option<AdapterLease>]> {
        seq_indices
            .iter()
            .map(|&index| input_seqs[index].adapter_lease().cloned())
            .collect::<Vec<_>>()
            .into()
    }

    pub struct TextInputsProcessor;

    impl InputsProcessor for TextInputsProcessor {
        fn process_inputs(
            &self,
            _: Option<Arc<Tokenizer>>,
            input_seqs: &mut [&mut Sequence],
            is_prompt: bool,
            is_xlora: bool,
            device: &Device,
            no_kv_cache: bool,
            last_n_context_len: Option<(usize, usize)>,
            return_raw_logits: bool,
            sliding_window: Option<usize>,
            _: Option<Arc<dyn Any>>,
            mut paged_attn_metadata: Option<PagedAttentionMeta>,
            mapper: Option<&dyn DeviceMapper>,
        ) -> Result<InputProcessorOutput> {
            let flash_sliding_window = if no_kv_cache { None } else { sliding_window };
            if is_xlora && !is_prompt {
                let prompt = get_prompt_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    last_n_context_len,
                    return_raw_logits,
                    paged_attn_metadata.as_mut(),
                    mapper,
                    flash_sliding_window,
                )?;
                let completion = get_completion_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    no_kv_cache,
                    last_n_context_len,
                    return_raw_logits,
                    paged_attn_metadata.as_mut(),
                    mapper,
                    flash_sliding_window,
                )?;
                let InnerInputProcessorOutput {
                    inputs:
                        InputMetadata {
                            input: input_ids_full,
                            positions: seqlen_offsets_full,
                            context_lens: _,
                            position_ids,
                            paged_attn_meta: _,
                            flash_meta: flash_meta_full,
                        },
                    seq_indices,
                } = prompt;
                let InnerInputProcessorOutput {
                    inputs:
                        InputMetadata {
                            input: input_ids,
                            positions: seqlen_offsets,
                            context_lens,
                            position_ids: _,
                            paged_attn_meta,
                            flash_meta,
                        },
                    seq_indices: _,
                } = completion;
                let adapter_leases = adapter_leases(input_seqs, &seq_indices);
                let inputs: Box<dyn Any> = Box::new(ModelInputs {
                    input_ids,
                    input_ids_full: Some(input_ids_full),
                    seqlen_offsets,
                    seqlen_offsets_full: Some(seqlen_offsets_full),
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta,
                    flash_meta_full: Some(flash_meta_full),
                    recurrent_batch_kind: RecurrentBatchKind::Decode,
                    adapter_leases,
                });
                Ok(InputProcessorOutput {
                    inputs,
                    seq_indices,
                })
            } else if is_xlora && is_prompt {
                let metadata = get_prompt_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    last_n_context_len,
                    return_raw_logits,
                    paged_attn_metadata.as_mut(),
                    mapper,
                    flash_sliding_window,
                )?;
                let InnerInputProcessorOutput {
                    inputs:
                        InputMetadata {
                            input: input_ids,
                            positions: seqlen_offsets,
                            context_lens,
                            position_ids,
                            paged_attn_meta,
                            flash_meta,
                        },
                    seq_indices,
                } = metadata;
                let adapter_leases = adapter_leases(input_seqs, &seq_indices);
                let inputs: Box<dyn Any> = Box::new(ModelInputs {
                    input_ids: input_ids.clone(),
                    input_ids_full: Some(input_ids),
                    seqlen_offsets: seqlen_offsets.clone(),
                    seqlen_offsets_full: Some(seqlen_offsets),
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta: flash_meta.clone(),
                    flash_meta_full: Some(flash_meta),
                    recurrent_batch_kind: RecurrentBatchKind::Prefill,
                    adapter_leases,
                });
                Ok(InputProcessorOutput {
                    inputs,
                    seq_indices,
                })
            } else if is_prompt {
                let metadata = get_prompt_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    last_n_context_len,
                    return_raw_logits,
                    paged_attn_metadata.as_mut(),
                    mapper,
                    flash_sliding_window,
                )?;
                let InnerInputProcessorOutput {
                    inputs:
                        InputMetadata {
                            input: input_ids,
                            positions: seqlen_offsets,
                            context_lens,
                            position_ids,
                            paged_attn_meta,
                            flash_meta,
                        },
                    seq_indices,
                } = metadata;
                let adapter_leases = adapter_leases(input_seqs, &seq_indices);
                let inputs: Box<dyn Any> = Box::new(ModelInputs {
                    input_ids,
                    input_ids_full: None,
                    seqlen_offsets,
                    seqlen_offsets_full: None,
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta,
                    flash_meta_full: None,
                    recurrent_batch_kind: RecurrentBatchKind::Prefill,
                    adapter_leases,
                });
                Ok(InputProcessorOutput {
                    inputs,
                    seq_indices,
                })
            } else {
                let recurrent_batch_kind = recurrent_batch_kind_for_input(
                    false,
                    crate::speculative::staging::staged_batch_width(input_seqs).is_some(),
                );
                let metadata = get_completion_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    no_kv_cache,
                    last_n_context_len,
                    return_raw_logits,
                    paged_attn_metadata.as_mut(),
                    mapper,
                    flash_sliding_window,
                )?;
                let InnerInputProcessorOutput {
                    inputs:
                        InputMetadata {
                            input: input_ids,
                            positions: seqlen_offsets,
                            context_lens,
                            position_ids,
                            paged_attn_meta,
                            flash_meta,
                        },
                    seq_indices,
                } = metadata;
                let adapter_leases = adapter_leases(input_seqs, &seq_indices);
                let inputs: Box<dyn Any> = Box::new(ModelInputs {
                    input_ids,
                    input_ids_full: None,
                    seqlen_offsets,
                    seqlen_offsets_full: None,
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta,
                    flash_meta_full: None,
                    recurrent_batch_kind,
                    adapter_leases,
                });
                Ok(InputProcessorOutput {
                    inputs,
                    seq_indices,
                })
            }
        }

        fn get_type(&self) -> InputsProcessorType {
            InputsProcessorType::Text
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::attention::flash_params::sliding_k_lengths;
        use crate::paged_attention::block_table_rows::BlockTableRows;
        use crate::paged_attention::input_metadata::{
            cuda_graph_block_table_len_with_cap, PagedDecodeMetadataRequirements,
        };

        fn assert_zero_padded_rows(rows: &[Vec<u32>], prefixes: &[&[u32]]) {
            assert_eq!(rows.len(), prefixes.len());
            for (row, prefix) in rows.iter().zip(prefixes) {
                assert_eq!(&row[..prefix.len()], *prefix);
                assert!(row[prefix.len()..].iter().all(|&value| value == 0));
            }
        }

        #[test]
        fn completion_input_keeps_staged_rows_device_backed() {
            let staged = vec![
                Tensor::from_vec(vec![10u32, 11], 2, &Device::Cpu).unwrap(),
                Tensor::from_vec(vec![20u32, 21], 2, &Device::Cpu).unwrap(),
            ];
            let input =
                completion_input_tensor(vec![7u32, 8], 2, 1, &staged, &Device::Cpu).unwrap();
            assert_eq!(
                input.to_vec2::<u32>().unwrap(),
                vec![vec![7, 10, 11], vec![8, 20, 21]]
            );
        }

        #[test]
        fn cuda_graph_context_buckets_track_live_rows() {
            const CACHE_CAPACITY: usize = 1_604_288;
            assert_eq!(
                cuda_graph_block_table_len_with_cap(4, 32, true, 128, Some(CACHE_CAPACITY)),
                16
            );
            assert_eq!(
                cuda_graph_block_table_len_with_cap(16, 32, true, 512, Some(CACHE_CAPACITY)),
                16
            );
            assert_eq!(
                cuda_graph_block_table_len_with_cap(17, 32, true, 513, Some(CACHE_CAPACITY)),
                32
            );
            assert_eq!(
                cuda_graph_block_table_len_with_cap(33, 32, true, 1025, Some(CACHE_CAPACITY)),
                64
            );
        }

        #[test]
        fn flashinfer_decode_metadata_ignores_aggregate_cache_capacity() {
            const CACHE_CAPACITY: usize = 1_604_288;
            let table = vec![1, 2, 3, 4];
            let metadata = Arc::new(DecodePagedRows {
                slot_mappings: vec![vec![127]],
                block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![table], 1),
                context_lens: vec![128],
                full_context_lens: vec![128],
                query_len: 1,
                block_size: 32,
                use_standard_metadata: false,
                max_paged_context_len: CACHE_CAPACITY,
                sliding_window: None,
                decode_window: 1,
                devices: vec![Device::Cpu],
                num_kv_heads: 4,
            })
            .build()
            .unwrap();
            let view = &metadata.flashinfer.unwrap().views.logical;
            assert_eq!(view.paged_kv.indices[&Device::Cpu.location()].dims(), &[64]);
            assert_eq!(
                view.tile_plan.request_indices[&Device::Cpu.location()].dims(),
                &[8]
            );
        }

        #[test]
        fn fa3_graph_update_builds_csr_without_fallback_metadata() {
            let table = vec![1, 2, 3, 4];
            let rows = Arc::new(DecodePagedRows {
                slot_mappings: vec![vec![127]],
                block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![table], 1),
                context_lens: vec![128],
                full_context_lens: vec![128],
                query_len: 1,
                block_size: 32,
                use_standard_metadata: false,
                max_paged_context_len: 1_604_288,
                sliding_window: None,
                decode_window: 1,
                devices: vec![Device::Cpu],
                num_kv_heads: 4,
            });
            let metadata = rows
                .build_graph_update(PagedDecodeMetadataRequirements::graph(
                    false, false, true, false,
                ))
                .unwrap();
            assert!(metadata.block_tables.is_none());
            assert!(metadata.context_lens.is_none());
            let view = &metadata.flashinfer.unwrap().views.logical;
            assert!(!view.paged_kv.indices.is_empty());
            assert!(view.tile_plan.request_indices.is_empty());
            assert!(view.tile_plan.kv_tile_indices.is_empty());
            assert!(view.tile_plan.o_indptr.is_empty());
            assert!(view.tile_plan.kv_chunk_size.is_empty());
            assert!(view.tile_plan.block_valid_mask.is_empty());
        }

        #[test]
        fn padded_decode_rows_alias_row_zero_without_kv_writes() {
            let rows = DecodePagedRows {
                slot_mappings: vec![vec![40, 41], vec![72, 73]],
                block_tables: BlockTableSnapshot::from_owned_sequence_tables(
                    vec![vec![1, 2], vec![3]],
                    2,
                ),
                context_lens: vec![40, 41, 8, 9],
                full_context_lens: vec![40, 41, 8, 9],
                query_len: 2,
                block_size: 32,
                use_standard_metadata: true,
                max_paged_context_len: 1024,
                sliding_window: None,
                decode_window: 1,
                devices: vec![Device::Cpu],
                num_kv_heads: 4,
            };
            let padded = rows.padded(4);
            assert_eq!(padded.batch_size(), 4);
            assert_eq!(padded.slot_mappings[2], vec![_PAD_SLOT_ID, _PAD_SLOT_ID]);
            assert_eq!(padded.slot_mappings[3], vec![_PAD_SLOT_ID, _PAD_SLOT_ID]);
            assert_eq!(padded.block_tables.len(), 8);
            assert_eq!(padded.block_tables.unique_table_count(), 2);
            assert_eq!(
                &padded.materialized_block_tables()[4..],
                &[vec![1, 2], vec![1, 2], vec![1, 2], vec![1, 2]]
            );
            assert_eq!(&padded.context_lens[4..], &[40, 41, 40, 41]);
            assert_eq!(&padded.full_context_lens[4..], &[40, 41, 40, 41]);
            let metadata = Arc::new(padded).build().unwrap();
            assert_eq!(metadata.slot_mappings[&Device::Cpu.location()].dims(), &[8]);
            assert_eq!(
                metadata.paged_context_lens_cpu.as_deref(),
                Some(&[40, 41, 8, 9, 40, 41, 40, 41][..])
            );
            assert!(metadata.decode_rows.is_some());
        }

        #[test]
        fn unsliding_decode_rows_share_canonical_tables_and_metadata() {
            let rows = Arc::new(DecodePagedRows {
                slot_mappings: vec![vec![39, 40], vec![71, 72]],
                block_tables: BlockTableSnapshot::from_owned_sequence_tables(
                    vec![vec![10, 11, 12], vec![20, 21, 22]],
                    2,
                ),
                context_lens: vec![9, 10, 7, 8],
                full_context_lens: vec![9, 10, 7, 8],
                query_len: 2,
                block_size: 4,
                use_standard_metadata: true,
                max_paged_context_len: 128,
                sliding_window: None,
                decode_window: 1,
                devices: vec![Device::Cpu],
                num_kv_heads: 1,
            });

            assert_eq!(rows.block_tables.unique_table_count(), 2);
            assert_eq!(
                rows.materialized_block_tables(),
                vec![
                    vec![10, 11, 12],
                    vec![10, 11, 12],
                    vec![20, 21, 22],
                    vec![20, 21, 22],
                ]
            );

            let metadata = rows.build_materialized().unwrap();
            let location = Device::Cpu.location();
            assert_eq!(
                metadata.block_tables.as_ref().unwrap()[&location].id(),
                metadata.full_block_tables.as_ref().unwrap()[&location].id()
            );
            let flashinfer = metadata.flashinfer.unwrap();
            assert!(flashinfer.views.sliding.is_none());
            let block_tables = flashinfer.views.logical.block_tables.unwrap()[&location]
                .to_vec2::<u32>()
                .unwrap();
            assert_zero_padded_rows(
                &block_tables,
                &[&[10, 11, 12], &[10, 11, 12], &[20, 21, 22], &[20, 21, 22]],
            );
        }

        #[test]
        fn sliding_decode_rows_materialize_exact_logical_and_windowed_tables() {
            let rows = Arc::new(DecodePagedRows {
                slot_mappings: vec![vec![47, 48], vec![91, 92]],
                block_tables: BlockTableSnapshot::from_owned_sequence_tables(
                    vec![vec![10, 11, 12, 13], vec![20, 21, 22]],
                    2,
                ),
                context_lens: vec![5, 6, 4, 5],
                full_context_lens: vec![9, 10, 4, 5],
                query_len: 2,
                block_size: 4,
                use_standard_metadata: true,
                max_paged_context_len: 128,
                sliding_window: Some(4),
                decode_window: 1,
                devices: vec![Device::Cpu],
                num_kv_heads: 1,
            });

            assert_eq!(
                rows.materialized_block_tables(),
                vec![vec![11, 12], vec![11, 12], vec![20], vec![20, 21]]
            );
            let metadata = rows.build_materialized().unwrap();
            let location = Device::Cpu.location();
            assert_eq!(
                metadata.block_tables.unwrap()[&location]
                    .to_vec2::<u32>()
                    .unwrap(),
                vec![vec![11, 12], vec![11, 12], vec![20, 0], vec![20, 21]]
            );
            let full_block_tables = metadata.full_block_tables.unwrap()[&location]
                .to_vec2::<u32>()
                .unwrap();
            assert_zero_padded_rows(
                &full_block_tables,
                &[
                    &[10, 11, 12, 13],
                    &[10, 11, 12, 13],
                    &[20, 21, 22],
                    &[20, 21, 22],
                ],
            );
        }

        #[test]
        fn ragged_prompt_selects_each_last_real_token() {
            let short = [1u32, 2];
            let long = [3u32, 4, 5, 6];
            let input = make_prompt_chunk(
                0,
                vec![short.as_slice(), long.as_slice()],
                &[0, 1],
                &Device::Cpu,
                None,
                false,
                None,
                None,
                None,
                None,
                false,
            )
            .unwrap();

            assert_eq!(input.input.dims(), &[2, 4]);
            assert_eq!(input.context_lens, vec![(1, 1), (3, 1)]);
        }

        #[test]
        fn packed_rope_positions_preserve_logical_offsets() {
            let positions = packed_rope_positions(&[0, 16, 32], &[3, 1, 4]).unwrap();

            assert_eq!(positions, vec![0, 1, 2, 16, 32, 33, 34, 35]);
        }

        #[test]
        fn packed_rope_positions_reject_mismatched_metadata() {
            let error = packed_rope_positions(&[0, 16], &[3]).unwrap_err();

            assert!(error.to_string().contains("2 offsets for 1 queries"));
        }

        #[test]
        fn packed_flash_params_preserve_ragged_boundaries() {
            let params = make_flash_params(
                &Device::Cpu,
                None,
                &[0, 3, 1, 4],
                &[0, 3, 1, 4],
                None,
                true,
                true,
            )
            .unwrap();
            let cumulative = params.cumulative_seqlens_q[&Device::Cpu.location()]
                .to_vec1::<u32>()
                .unwrap();

            assert_eq!(params.max_q, 4);
            assert_eq!(cumulative, vec![0, 3, 4, 8]);
        }

        #[test]
        fn sliding_single_token_uses_the_retained_window() {
            assert_eq!(
                sliding_k_lengths(&[0, 1], &[0, 101], 4).unwrap(),
                vec![0, 4]
            );
        }

        #[test]
        fn sliding_query_longer_than_the_window_keeps_the_full_query() {
            assert_eq!(sliding_k_lengths(&[0, 8], &[0, 8], 4).unwrap(), vec![0, 8]);
        }

        #[test]
        fn sliding_cached_multi_token_append_keeps_retained_and_new_tokens() {
            assert_eq!(
                sliding_k_lengths(&[0, 3], &[0, 103], 4).unwrap(),
                vec![0, 7]
            );
        }

        #[test]
        fn fresh_packed_sliding_metadata_preserves_logical_boundaries() {
            let params = make_flash_params(
                &Device::Cpu,
                None,
                &[0, 3, 1, 6],
                &[0, 3, 1, 6],
                Some(4),
                true,
                true,
            )
            .unwrap();
            let sliding = params.sliding_k.unwrap();

            assert_eq!(sliding.max, 6);
            assert_eq!(
                sliding.cumulative_seqlens[&Device::Cpu.location()]
                    .to_vec1::<u32>()
                    .unwrap(),
                vec![0, 3, 4, 10]
            );
        }

        #[test]
        fn sliding_metadata_rejects_inconsistent_lengths() {
            assert!(sliding_k_lengths(&[0, 2], &[0], 4).is_err());
            assert!(sliding_k_lengths(&[0, 3], &[0, 2], 4).is_err());
        }
    }
}
