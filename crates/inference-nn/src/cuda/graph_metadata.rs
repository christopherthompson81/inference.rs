//! The paged attention metadata a captured decode step reads, held in buffers each replay overwrites in place.

use std::collections::HashMap;

use inference_tensor::{DType, Device, DeviceLocation, Tensor, Var};

use crate::cuda::graph_capture::CudaGraphHostStaging;
use crate::flashinfer::{
    Fa3DecodeState, FlashInferMetadata, FlashInferPagedAttentionView,
    FlashInferPagedAttentionViews, FlashInferPagedKv, FlashInferTilePlan, make_fa3_decode_state,
};
use crate::model::decode_positions_tensor;
use crate::paged_attention::input_metadata::PagedDecodeMetadataRequirements;
use crate::paged_attention::{ModelConfigLike, PagedAttentionInputMetadata};

pub type CudaGraphVarMap = HashMap<DeviceLocation, Var>;

pub struct CudaDecodeGraphMetadataInput<'a> {
    pub metadata: &'a PagedAttentionInputMetadata,
    pub seqlen_offsets: &'a [usize],
    pub position_ids: &'a [usize],
    pub seq_len: usize,
    pub kv_cache: &'a [(Tensor, Tensor)],
    pub model_metadata: Option<&'a (dyn ModelConfigLike + Send + Sync)>,
    pub activation_dtype: DType,
}

pub struct CudaDecodeGraphMetadataBuffers {
    requirements: PagedDecodeMetadataRequirements,
    flashinfer_views_alias: bool,
    slot_mappings: CudaGraphVarMap,
    block_tables: Option<CudaGraphVarMap>,
    context_lens: Option<CudaGraphVarMap>,
    full_block_tables: Option<CudaGraphVarMap>,
    full_context_lens: Option<CudaGraphVarMap>,
    paged_kv_indptr: Option<CudaGraphVarMap>,
    paged_kv_indices: Option<CudaGraphVarMap>,
    paged_kv_last_page_len: Option<CudaGraphVarMap>,
    full_paged_kv_indptr: Option<CudaGraphVarMap>,
    full_paged_kv_indices: Option<CudaGraphVarMap>,
    full_paged_kv_last_page_len: Option<CudaGraphVarMap>,
    paged_kv_request_indices: Option<CudaGraphVarMap>,
    paged_kv_tile_indices: Option<CudaGraphVarMap>,
    paged_kv_o_indptr: Option<CudaGraphVarMap>,
    paged_kv_chunk_size: Option<CudaGraphVarMap>,
    paged_kv_block_valid_mask: Option<CudaGraphVarMap>,
    full_paged_kv_request_indices: Option<CudaGraphVarMap>,
    full_paged_kv_tile_indices: Option<CudaGraphVarMap>,
    full_paged_kv_o_indptr: Option<CudaGraphVarMap>,
    full_paged_kv_chunk_size: Option<CudaGraphVarMap>,
    full_paged_kv_block_valid_mask: Option<CudaGraphVarMap>,
    fa3_decode: Option<Fa3DecodeState>,
    rope_positions: CudaGraphVarMap,
}

impl CudaDecodeGraphMetadataBuffers {
    pub fn new(
        input: CudaDecodeGraphMetadataInput<'_>,
    ) -> inference_tensor::Result<(Self, PagedAttentionInputMetadata)> {
        let CudaDecodeGraphMetadataInput {
            metadata,
            seqlen_offsets,
            position_ids,
            seq_len,
            kv_cache,
            model_metadata,
            activation_dtype,
        } = input;
        let slot_mappings = var_map_from_tensor_map(&metadata.slot_mappings)?;
        if seqlen_offsets.len() != position_ids.len() {
            inference_tensor::bail!(
                "CUDA graph decode has {} KV offsets but {} position ends",
                seqlen_offsets.len(),
                position_ids.len()
            );
        }
        let rope_positions =
            rope_positions_var_map(&metadata.slot_mappings, position_ids, seq_len)?;
        let fa3_decode = metadata
            .flashinfer
            .as_ref()
            .map(|flashinfer| {
                make_fa3_decode_state(
                    flashinfer,
                    seqlen_offsets.len(),
                    seq_len,
                    kv_cache,
                    model_metadata,
                    activation_dtype,
                )
            })
            .transpose()?
            .flatten();
        let flashinfer_views_alias = flashinfer_views_alias(metadata);
        let requirements = PagedDecodeMetadataRequirements::graph(
            metadata.block_tables.is_some(),
            metadata.context_lens.is_some(),
            metadata.flashinfer.is_some(),
            metadata.flashinfer.is_some(),
        );
        let mut buffers = Self {
            requirements,
            flashinfer_views_alias,
            slot_mappings,
            block_tables: option_var_map_from_tensor_map(metadata.block_tables.as_ref())?,
            context_lens: option_var_map_from_tensor_map(metadata.context_lens.as_ref())?,
            full_block_tables: option_var_map_from_tensor_map_if_distinct(
                metadata.full_block_tables.as_ref(),
                flashinfer_views_alias,
            )?,
            full_context_lens: option_var_map_from_tensor_map_if_distinct(
                metadata.full_context_lens.as_ref(),
                flashinfer_views_alias,
            )?,
            paged_kv_indptr: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.indptr),
            )?,
            paged_kv_indices: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.indices),
            )?,
            paged_kv_last_page_len: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.last_page_len),
            )?,
            full_paged_kv_indptr: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.paged_kv.indptr),
                flashinfer_views_alias,
            )?,
            full_paged_kv_indices: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.paged_kv.indices),
                flashinfer_views_alias,
            )?,
            full_paged_kv_last_page_len: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.paged_kv.last_page_len),
                flashinfer_views_alias,
            )?,
            paged_kv_request_indices: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.request_indices),
            )?,
            paged_kv_tile_indices: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.kv_tile_indices),
            )?,
            paged_kv_o_indptr: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.o_indptr),
            )?,
            paged_kv_chunk_size: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.kv_chunk_size),
            )?,
            paged_kv_block_valid_mask: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.block_valid_mask),
            )?,
            full_paged_kv_request_indices: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.request_indices),
                flashinfer_views_alias,
            )?,
            full_paged_kv_tile_indices: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.kv_tile_indices),
                flashinfer_views_alias,
            )?,
            full_paged_kv_o_indptr: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.o_indptr),
                flashinfer_views_alias,
            )?,
            full_paged_kv_chunk_size: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.kv_chunk_size),
                flashinfer_views_alias,
            )?,
            full_paged_kv_block_valid_mask: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.block_valid_mask),
                flashinfer_views_alias,
            )?,
            fa3_decode,
            rope_positions,
        };
        if flashinfer_views_alias {
            buffers.full_block_tables = buffers.block_tables.clone();
            buffers.full_context_lens = buffers.context_lens.clone();
            buffers.full_paged_kv_indptr = buffers.paged_kv_indptr.clone();
            buffers.full_paged_kv_indices = buffers.paged_kv_indices.clone();
            buffers.full_paged_kv_last_page_len = buffers.paged_kv_last_page_len.clone();
            buffers.full_paged_kv_request_indices = buffers.paged_kv_request_indices.clone();
            buffers.full_paged_kv_tile_indices = buffers.paged_kv_tile_indices.clone();
            buffers.full_paged_kv_o_indptr = buffers.paged_kv_o_indptr.clone();
            buffers.full_paged_kv_chunk_size = buffers.paged_kv_chunk_size.clone();
            buffers.full_paged_kv_block_valid_mask = buffers.paged_kv_block_valid_mask.clone();
        }
        let metadata = buffers.metadata_from(metadata);
        Ok((buffers, metadata))
    }

    pub fn finish_capture(&mut self, metadata: &PagedAttentionInputMetadata) {
        let tile_plan_used = metadata
            .flashinfer
            .as_ref()
            .is_some_and(FlashInferMetadata::decode_tile_plan_was_used);
        self.requirements = PagedDecodeMetadataRequirements::graph(
            self.block_tables.is_some(),
            self.context_lens.is_some(),
            self.fa3_decode.is_some() || tile_plan_used,
            tile_plan_used,
        );
    }

    pub fn copy_from(
        &mut self,
        metadata: &PagedAttentionInputMetadata,
        position_ids: &[usize],
        seq_len: usize,
        host_staging: &mut CudaGraphHostStaging,
    ) -> inference_tensor::Result<()> {
        let graph_update = if metadata.has_host_staged_decode_tensors() {
            Some(
                metadata
                    .decode_rows
                    .as_ref()
                    .expect("host-staged decode metadata requires source rows")
                    .build_graph_update(self.requirements)
                    .map_err(inference_tensor::Error::msg)?,
            )
        } else {
            None
        };
        let metadata = graph_update.as_ref().unwrap_or(metadata);
        copy_var_map(
            &self.slot_mappings,
            &metadata.slot_mappings,
            "slot_mappings",
            host_staging,
        )?;
        if self.requirements.context_lens {
            copy_option_var_map(
                &self.context_lens,
                metadata.context_lens.as_ref(),
                "context_lens",
                host_staging,
            )?;
            if !self.flashinfer_views_alias {
                copy_option_var_map(
                    &self.full_context_lens,
                    metadata.full_context_lens.as_ref(),
                    "full_context_lens",
                    host_staging,
                )?;
            }
        }
        if self.requirements.block_tables {
            copy_option_var_map(
                &self.block_tables,
                metadata.block_tables.as_ref(),
                "block_tables",
                host_staging,
            )?;
            if !self.flashinfer_views_alias {
                copy_option_var_map(
                    &self.full_block_tables,
                    metadata.full_block_tables.as_ref(),
                    "full_block_tables",
                    host_staging,
                )?;
            }
        }
        if self.requirements.flashinfer_paged_kv {
            copy_option_var_map(
                &self.paged_kv_last_page_len,
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.last_page_len),
                "paged_kv_last_page_len",
                host_staging,
            )?;
            copy_option_var_map(
                &self.paged_kv_indptr,
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.indptr),
                "paged_kv_indptr",
                host_staging,
            )?;
            copy_option_var_map(
                &self.paged_kv_indices,
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.indices),
                "paged_kv_indices",
                host_staging,
            )?;
            if !self.flashinfer_views_alias {
                copy_option_var_map(
                    &self.full_paged_kv_last_page_len,
                    flashinfer_full_view(metadata).map(|view| &view.paged_kv.last_page_len),
                    "full_paged_kv_last_page_len",
                    host_staging,
                )?;
                copy_option_var_map(
                    &self.full_paged_kv_indptr,
                    flashinfer_full_view(metadata).map(|view| &view.paged_kv.indptr),
                    "full_paged_kv_indptr",
                    host_staging,
                )?;
                copy_option_var_map(
                    &self.full_paged_kv_indices,
                    flashinfer_full_view(metadata).map(|view| &view.paged_kv.indices),
                    "full_paged_kv_indices",
                    host_staging,
                )?;
            }
        }
        if self.requirements.flashinfer_tile_plan {
            copy_flashinfer_tile_plan(
                metadata,
                false,
                FlashInferTilePlanVars {
                    request_indices: &self.paged_kv_request_indices,
                    kv_tile_indices: &self.paged_kv_tile_indices,
                    o_indptr: &self.paged_kv_o_indptr,
                    kv_chunk_size: &self.paged_kv_chunk_size,
                    block_valid_mask: &self.paged_kv_block_valid_mask,
                },
                host_staging,
            )?;
            if !self.flashinfer_views_alias {
                copy_flashinfer_tile_plan(
                    metadata,
                    true,
                    FlashInferTilePlanVars {
                        request_indices: &self.full_paged_kv_request_indices,
                        kv_tile_indices: &self.full_paged_kv_tile_indices,
                        o_indptr: &self.full_paged_kv_o_indptr,
                        kv_chunk_size: &self.full_paged_kv_chunk_size,
                        block_valid_mask: &self.full_paged_kv_block_valid_mask,
                    },
                    host_staging,
                )?;
            }
        }
        copy_rope_positions(&self.rope_positions, position_ids, seq_len, host_staging)?;
        Ok(())
    }

    fn flashinfer_metadata_from(
        &self,
        metadata: &PagedAttentionInputMetadata,
    ) -> Option<FlashInferMetadata> {
        let original = metadata.flashinfer.as_ref()?;
        let logical = FlashInferPagedAttentionView {
            block_tables: option_tensor_map_from_var_map(&self.full_block_tables),
            context_lens: option_tensor_map_from_var_map(&self.full_context_lens),
            paged_kv: flashinfer_paged_kv_from_vars(
                &self.full_paged_kv_indptr,
                &self.full_paged_kv_indices,
                &self.full_paged_kv_last_page_len,
            )?,
            tile_plan: flashinfer_tile_plan_from_vars(
                &self.full_paged_kv_request_indices,
                &self.full_paged_kv_tile_indices,
                &self.full_paged_kv_o_indptr,
                &self.full_paged_kv_chunk_size,
                &self.full_paged_kv_block_valid_mask,
            )?,
        };
        let sliding = if original.views.sliding.is_some() {
            Some(FlashInferPagedAttentionView {
                block_tables: option_tensor_map_from_var_map(&self.block_tables),
                context_lens: option_tensor_map_from_var_map(&self.context_lens),
                paged_kv: flashinfer_paged_kv_from_vars(
                    &self.paged_kv_indptr,
                    &self.paged_kv_indices,
                    &self.paged_kv_last_page_len,
                )?,
                tile_plan: flashinfer_tile_plan_from_vars(
                    &self.paged_kv_request_indices,
                    &self.paged_kv_tile_indices,
                    &self.paged_kv_o_indptr,
                    &self.paged_kv_chunk_size,
                    &self.paged_kv_block_valid_mask,
                )?,
            })
        } else {
            None
        };

        Some(
            FlashInferMetadata {
                views: FlashInferPagedAttentionViews { logical, sliding },
                fa3_decode: self.fa3_decode.clone(),
                decode_tile_plan_used: None,
            }
            .track_decode_tile_plan(),
        )
    }

    pub fn metadata_from(
        &self,
        metadata: &PagedAttentionInputMetadata,
    ) -> PagedAttentionInputMetadata {
        PagedAttentionInputMetadata {
            block_tables: option_tensor_map_from_var_map(&self.block_tables),
            context_lens: option_tensor_map_from_var_map(&self.context_lens),
            block_size: metadata.block_size,
            paged_context_lens_cpu: metadata.paged_context_lens_cpu.clone(),
            full_paged_context_lens_cpu: metadata.full_paged_context_lens_cpu.clone(),
            slot_mappings: tensor_map_from_var_map(&self.slot_mappings),
            max_context_len: metadata.max_context_len,
            full_block_tables: option_tensor_map_from_var_map(&self.full_block_tables),
            full_context_lens: option_tensor_map_from_var_map(&self.full_context_lens),
            full_max_context_len: metadata.full_max_context_len,
            is_first_prompt_chunk: metadata.is_first_prompt_chunk,
            is_final_prompt_chunk: metadata.is_final_prompt_chunk,
            needs_logits: metadata.needs_logits,
            prompt_chunk_attention_policy: metadata.prompt_chunk_attention_policy,
            has_noncausal_mm_context: metadata.has_noncausal_mm_context,
            prefix_gather_workspace_limit: metadata.prefix_gather_workspace_limit,
            mm_prefix_ranges: metadata.mm_prefix_ranges.clone(),
            full_mm_prefix_ranges: metadata.full_mm_prefix_ranges.clone(),
            prefill_attention_heads: metadata.prefill_attention_heads,
            prefill_key_value_heads: metadata.prefill_key_value_heads,
            prefill_head_dim: metadata.prefill_head_dim,
            flashinfer: self.flashinfer_metadata_from(metadata),
            rope_positions: Some(tensor_map_from_var_map(&self.rope_positions)),
            num_cached_tokens: metadata.num_cached_tokens.clone(),
            query_lens: metadata.query_lens.clone(),
            cu_seqlens_q: metadata.cu_seqlens_q.clone(),
            cu_seqlens_kv: metadata.cu_seqlens_kv.clone(),
            decode_rows: metadata.decode_rows.clone(),
        }
    }
}

pub fn flashinfer_paged_view(
    metadata: &PagedAttentionInputMetadata,
) -> Option<&FlashInferPagedAttentionView> {
    let views = &metadata.flashinfer.as_ref()?.views;
    Some(views.sliding.as_ref().unwrap_or(&views.logical))
}

pub fn flashinfer_full_view(
    metadata: &PagedAttentionInputMetadata,
) -> Option<&FlashInferPagedAttentionView> {
    Some(&metadata.flashinfer.as_ref()?.views.logical)
}

pub fn flashinfer_views_alias(metadata: &PagedAttentionInputMetadata) -> bool {
    metadata
        .flashinfer
        .as_ref()
        .is_some_and(|flashinfer| flashinfer.views.sliding.is_none())
}

fn flashinfer_paged_kv_from_vars(
    indptr: &Option<CudaGraphVarMap>,
    indices: &Option<CudaGraphVarMap>,
    last_page_len: &Option<CudaGraphVarMap>,
) -> Option<FlashInferPagedKv> {
    Some(FlashInferPagedKv {
        indptr: option_tensor_map_from_var_map(indptr)?,
        indices: option_tensor_map_from_var_map(indices)?,
        last_page_len: option_tensor_map_from_var_map(last_page_len)?,
    })
}

fn flashinfer_tile_plan_from_vars(
    request_indices: &Option<CudaGraphVarMap>,
    kv_tile_indices: &Option<CudaGraphVarMap>,
    o_indptr: &Option<CudaGraphVarMap>,
    kv_chunk_size: &Option<CudaGraphVarMap>,
    block_valid_mask: &Option<CudaGraphVarMap>,
) -> Option<FlashInferTilePlan> {
    Some(FlashInferTilePlan {
        request_indices: option_tensor_map_from_var_map(request_indices)?,
        kv_tile_indices: option_tensor_map_from_var_map(kv_tile_indices)?,
        o_indptr: option_tensor_map_from_var_map(o_indptr)?,
        kv_chunk_size: option_tensor_map_from_var_map(kv_chunk_size)?,
        block_valid_mask: option_tensor_map_from_var_map(block_valid_mask)?,
    })
}

fn var_map_from_tensor_map(
    map: &HashMap<DeviceLocation, Tensor>,
) -> inference_tensor::Result<CudaGraphVarMap> {
    map.iter()
        .map(|(location, tensor)| Ok((*location, Var::from_tensor(tensor)?)))
        .collect()
}

fn option_var_map_from_tensor_map(
    map: Option<&HashMap<DeviceLocation, Tensor>>,
) -> inference_tensor::Result<Option<CudaGraphVarMap>> {
    map.map(var_map_from_tensor_map).transpose()
}

fn option_var_map_from_tensor_map_if_distinct(
    map: Option<&HashMap<DeviceLocation, Tensor>>,
    aliases_existing: bool,
) -> inference_tensor::Result<Option<CudaGraphVarMap>> {
    if aliases_existing {
        Ok(None)
    } else {
        option_var_map_from_tensor_map(map)
    }
}

fn tensor_map_from_var_map(map: &CudaGraphVarMap) -> HashMap<DeviceLocation, Tensor> {
    map.iter()
        .map(|(location, var)| (*location, var.as_detached_tensor()))
        .collect()
}

fn option_tensor_map_from_var_map(
    map: &Option<CudaGraphVarMap>,
) -> Option<HashMap<DeviceLocation, Tensor>> {
    map.as_ref().map(tensor_map_from_var_map)
}

fn copy_var_map(
    dst: &CudaGraphVarMap,
    src: &HashMap<DeviceLocation, Tensor>,
    name: &'static str,
    host_staging: &mut CudaGraphHostStaging,
) -> inference_tensor::Result<()> {
    if dst.len() != src.len() {
        inference_tensor::bail!("{name} device count changed during CUDA graph replay");
    }
    for (location, dst) in dst {
        let src = src
            .get(location)
            .ok_or_else(|| inference_tensor::Error::msg(format!("{name} missing {location:?}")))?;
        if src.device().is_cpu() && dst.device().is_cuda() {
            host_staging.copy_from(name, *location, src, dst)?;
        } else {
            dst.set(src)?;
        }
    }
    Ok(())
}

fn copy_option_var_map(
    dst: &Option<CudaGraphVarMap>,
    src: Option<&HashMap<DeviceLocation, Tensor>>,
    name: &'static str,
    host_staging: &mut CudaGraphHostStaging,
) -> inference_tensor::Result<()> {
    match (dst, src) {
        (Some(dst), Some(src)) => copy_var_map(dst, src, name, host_staging),
        (None, None) => Ok(()),
        _ => inference_tensor::bail!("{name} changed optional state during CUDA graph replay"),
    }
}

struct FlashInferTilePlanVars<'a> {
    request_indices: &'a Option<CudaGraphVarMap>,
    kv_tile_indices: &'a Option<CudaGraphVarMap>,
    o_indptr: &'a Option<CudaGraphVarMap>,
    kv_chunk_size: &'a Option<CudaGraphVarMap>,
    block_valid_mask: &'a Option<CudaGraphVarMap>,
}

fn copy_flashinfer_tile_plan(
    metadata: &PagedAttentionInputMetadata,
    full: bool,
    vars: FlashInferTilePlanVars<'_>,
    host_staging: &mut CudaGraphHostStaging,
) -> inference_tensor::Result<()> {
    let view = if full {
        flashinfer_full_view(metadata)
    } else {
        flashinfer_paged_view(metadata)
    };
    copy_option_var_map(
        vars.request_indices,
        view.map(|view| &view.tile_plan.request_indices),
        if full {
            "full_paged_kv_request_indices"
        } else {
            "paged_kv_request_indices"
        },
        host_staging,
    )?;
    copy_option_var_map(
        vars.kv_tile_indices,
        view.map(|view| &view.tile_plan.kv_tile_indices),
        if full {
            "full_paged_kv_tile_indices"
        } else {
            "paged_kv_tile_indices"
        },
        host_staging,
    )?;
    copy_option_var_map(
        vars.o_indptr,
        view.map(|view| &view.tile_plan.o_indptr),
        if full {
            "full_paged_kv_o_indptr"
        } else {
            "paged_kv_o_indptr"
        },
        host_staging,
    )?;
    copy_option_var_map(
        vars.kv_chunk_size,
        view.map(|view| &view.tile_plan.kv_chunk_size),
        if full {
            "full_paged_kv_chunk_size"
        } else {
            "paged_kv_chunk_size"
        },
        host_staging,
    )?;
    copy_option_var_map(
        vars.block_valid_mask,
        view.map(|view| &view.tile_plan.block_valid_mask),
        if full {
            "full_paged_kv_block_valid_mask"
        } else {
            "paged_kv_block_valid_mask"
        },
        host_staging,
    )
}

fn rope_positions_var_map(
    slot_mappings: &HashMap<DeviceLocation, Tensor>,
    position_ids: &[usize],
    seq_len: usize,
) -> inference_tensor::Result<CudaGraphVarMap> {
    slot_mappings
        .iter()
        .map(|(location, tensor)| {
            let positions = decode_positions_tensor(position_ids, seq_len, tensor.device())?;
            Ok((*location, Var::from_tensor(&positions)?))
        })
        .collect()
}

fn copy_rope_positions(
    dst: &CudaGraphVarMap,
    position_ids: &[usize],
    seq_len: usize,
    host_staging: &mut CudaGraphHostStaging,
) -> inference_tensor::Result<()> {
    let positions = decode_positions_tensor(position_ids, seq_len, &Device::Cpu)?;
    for (location, dst) in dst {
        if dst.device().is_cuda() {
            host_staging.copy_from("rope_positions", *location, &positions, dst)?;
        } else {
            dst.set(&positions)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::paged_attention::block_table_rows::BlockTableSnapshot;
    use crate::paged_attention::input_metadata::DecodePagedRows;

    #[test]
    fn graph_rope_positions_expand_adjusted_ends_for_verification() {
        let table = vec![1, 2];
        let metadata = Arc::new(DecodePagedRows {
            slot_mappings: vec![vec![37, 38, 39], vec![37, 38, 39]],
            block_tables: BlockTableSnapshot::from_owned_sequence_tables(
                vec![table.clone(), table],
                3,
            ),
            context_lens: vec![38, 39, 40, 38, 39, 40],
            full_context_lens: vec![38, 39, 40, 38, 39, 40],
            query_len: 3,
            block_size: 32,
            use_standard_metadata: false,
            max_paged_context_len: 128,
            sliding_window: None,
            decode_window: 1,
            devices: vec![Device::Cpu],
            num_kv_heads: 4,
        })
        .build_materialized()
        .unwrap();

        let (buffers, _) = CudaDecodeGraphMetadataBuffers::new(CudaDecodeGraphMetadataInput {
            metadata: &metadata,
            seqlen_offsets: &[97, 97],
            position_ids: &[100, 52],
            seq_len: 3,
            kv_cache: &[],
            model_metadata: None,
            activation_dtype: DType::F32,
        })
        .unwrap();
        let positions = buffers.rope_positions[&Device::Cpu.location()]
            .as_detached_tensor()
            .to_vec1::<u32>()
            .unwrap();

        assert_eq!(positions, vec![97, 98, 99, 49, 50, 51]);
    }

    #[test]
    fn graph_buffers_share_unsliding_flashinfer_metadata() {
        let table = vec![1, 2, 3, 4];
        let metadata = Arc::new(DecodePagedRows {
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
        })
        .build()
        .unwrap();
        let (buffers, _) = CudaDecodeGraphMetadataBuffers::new(CudaDecodeGraphMetadataInput {
            metadata: &metadata,
            seqlen_offsets: &[127],
            position_ids: &[128],
            seq_len: 1,
            kv_cache: &[],
            model_metadata: None,
            activation_dtype: DType::F32,
        })
        .unwrap();
        assert!(buffers.flashinfer_views_alias);
        let location = Device::Cpu.location();
        assert_eq!(
            buffers.paged_kv_indices.as_ref().unwrap()[&location].id(),
            buffers.full_paged_kv_indices.as_ref().unwrap()[&location].id()
        );
        assert_eq!(
            buffers.paged_kv_request_indices.as_ref().unwrap()[&location].id(),
            buffers.full_paged_kv_request_indices.as_ref().unwrap()[&location].id()
        );
    }

    #[test]
    fn graph_metadata_replay_updates_persistent_cuda_buffers() -> anyhow::Result<()> {
        crate::skip_without_cuda!();
        let device = Device::new_cuda(0)?;
        let rows = |table: Vec<usize>, context_len: usize| -> anyhow::Result<_> {
            Ok(Arc::new(DecodePagedRows {
                slot_mappings: vec![vec![i64::try_from(context_len - 1)?]],
                block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![table], 1),
                context_lens: vec![context_len],
                full_context_lens: vec![context_len],
                query_len: 1,
                block_size: 32,
                use_standard_metadata: false,
                max_paged_context_len: 1_604_288,
                sliding_window: None,
                decode_window: 1,
                devices: vec![device.clone()],
                num_kv_heads: 4,
            }))
        };
        let initial = rows(vec![1, 2, 3, 4], 128)?.build_materialized()?;
        let updated = rows(vec![5, 6, 7, 8, 9, 10, 11, 12], 256)?.build()?;
        assert!(updated.has_host_staged_decode_tensors());

        let (mut buffers, _) = CudaDecodeGraphMetadataBuffers::new(CudaDecodeGraphMetadataInput {
            metadata: &initial,
            seqlen_offsets: &[127],
            position_ids: &[128],
            seq_len: 1,
            kv_cache: &[],
            model_metadata: None,
            activation_dtype: DType::F32,
        })?;
        let location = device.location();
        let mut host_staging = CudaGraphHostStaging::new(device.as_cuda_device()?.cuda_stream())?;
        host_staging.update(|host_staging| buffers.copy_from(&updated, &[256], 1, host_staging))?;
        host_staging.update(|host_staging| buffers.copy_from(&updated, &[256], 1, host_staging))?;
        device.synchronize()?;

        let indices = buffers.paged_kv_indices.as_ref().unwrap()[&location]
            .as_detached_tensor()
            .to_device(&Device::Cpu)?
            .to_vec1::<i32>()?;
        assert_eq!(&indices[..8], &[5, 6, 7, 8, 9, 10, 11, 12]);
        let positions = buffers.rope_positions[&location]
            .as_detached_tensor()
            .to_device(&Device::Cpu)?
            .to_vec1::<u32>()?;
        assert_eq!(positions, vec![255]);
        assert!(host_staging.staged_buffer_count() > 1);
        assert_eq!(host_staging.pending_completion_count(), 1);
        Ok(())
    }
}
