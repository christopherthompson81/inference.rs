use crate::attention::FlashParams;
use crate::attention::flash_params::make_flash_params;
use crate::paged_attention::PagedAttentionInputMetadata;
use crate::paged_attention::input_metadata::DecodePagedRows;
use std::{
    collections::HashMap,
    fmt,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use candle_core::cuda_backend::cudarc::driver::{CudaStream, sys};
use candle_core::{DType, Device, DeviceLocation, Tensor, Var};

use crate::gdn::RecurrentBatchKind;
#[cfg(target_family = "unix")]
use crate::paged_attention::plan::DecodePlan;
use crate::{
    flashinfer::{
        Fa3DecodeState, FlashInferMetadata, FlashInferPagedAttentionView,
        FlashInferPagedAttentionViews, FlashInferPagedKv, FlashInferTilePlan,
        make_fa3_decode_state,
    },
    paged_attention::{
        AttentionBackendKind, ModelConfigLike, block_table_rows::BlockTableSnapshot,
    },
};

use crate::cuda::phase_timer::CudaPhaseTimer;
use crate::device_map::DeviceMapper;
use crate::kv_cache::HybridCache;
use crate::model::decode_positions_tensor;
use crate::paged_attention::_PAD_SLOT_ID;
use crate::paged_attention::input_metadata::{
    DecodePagedRowsGraphKey, PagedDecodeMetadataRequirements,
};
use crate::pipeline::DecodeGraphPrecaptureCtx;
use crate::speculative::SpeculativeGraphState;
pub(crate) use inference_nn::cuda::graph_capture::*;

const TARGET_CUDA_DECODE_GRAPH_CACHE_DEFAULT_CAPACITY: usize = 64;
const TARGET_CUDA_DECODE_GRAPH_CACHE_MAX_CAPACITY: usize = 96;
const TARGET_CUDA_DECODE_GRAPH_CACHE_CAPACITY_QUANTUM: usize = 16;
const CUDA_GRAPH_SPEC_STATE_BUDGET_FLOOR_PERCENT: usize = 4;
const CUDA_GRAPH_SPEC_STATE_BUDGET_CEILING_PERCENT: usize = 8;
// C128 buckets plus one extra max-batch context graph require just under ten largest entries.
const CUDA_GRAPH_SPEC_STATE_WORKING_SET_MULTIPLIER: usize = 10;
const CUDA_GRAPH_SPEC_STATE_BUDGET_BYTES_ENV: &str =
    "INFERENCE_RS_CUDA_GRAPH_SPEC_STATE_BUDGET_BYTES";
static NEXT_CUDA_DECODE_GRAPH_GENERATION: AtomicU64 = AtomicU64::new(1);
pub(crate) fn target_cuda_graph_cache_capacity(
    startup_shapes: usize,
    runtime_batch_shapes: usize,
) -> usize {
    let working_set = startup_shapes.saturating_add(runtime_batch_shapes);
    let rounded = working_set
        .div_ceil(TARGET_CUDA_DECODE_GRAPH_CACHE_CAPACITY_QUANTUM)
        .saturating_mul(TARGET_CUDA_DECODE_GRAPH_CACHE_CAPACITY_QUANTUM);
    rounded.clamp(
        TARGET_CUDA_DECODE_GRAPH_CACHE_DEFAULT_CAPACITY,
        TARGET_CUDA_DECODE_GRAPH_CACHE_MAX_CAPACITY,
    )
}

pub(crate) fn cuda_graph_startup_capture_allowed(q_len: usize) -> bool {
    q_len > 0
}

pub(crate) fn prepare_fa3_decode_schedules(
    metadata: &PagedAttentionInputMetadata,
) -> candle_core::Result<()> {
    let Some(flashinfer) = metadata.flashinfer.as_ref() else {
        return Ok(());
    };
    flashinfer.for_each_fa3_decode_schedule(|prepare| {
        inference_paged_attn::fa3_prepare_decode_metadata(
            inference_paged_attn::Fa3DecodeMetadata {
                paged_kv_indptr: prepare.paged_kv_indptr,
                paged_kv_indices: prepare.paged_kv_indices,
                paged_kv_last_page_len: prepare.paged_kv_last_page_len,
                page_table: &prepare.buffers.page_table,
                seqused_k: &prepare.buffers.seqused_k,
                cu_seqlens_q: &prepare.buffers.cu_seqlens_q,
                scheduler_metadata: &prepare.buffers.scheduler_metadata,
            },
            prepare.buffers.schedule(prepare.key)?,
        )
    })
}

/// One decode step, padded up to its graph batch bucket. Pad rows alias row 0 for reads and skip
/// their KV and recurrent writes, so the model can run them and drop their outputs.
#[derive(Clone)]
pub(crate) struct CudaGraphDecodeStep {
    pub(crate) input_ids: Tensor,
    pub(crate) seqlen_offsets: Vec<usize>,
    pub(crate) context_lens: Vec<(usize, usize)>,
    pub(crate) position_ids: Vec<usize>,
    pub(crate) metadata: PagedAttentionInputMetadata,
    pub(crate) state_indices: Option<Vec<u32>>,
    pub(crate) real_batch: usize,
}

pub(crate) struct CudaGraphDecodeStepInputs<'a> {
    pub(crate) input_ids: &'a Tensor,
    pub(crate) seqlen_offsets: &'a [usize],
    pub(crate) context_lens: &'a [(usize, usize)],
    pub(crate) position_ids: &'a [usize],
    pub(crate) metadata: &'a PagedAttentionInputMetadata,
    pub(crate) state_indices: Option<&'a [u32]>,
    pub(crate) pad_slot: Option<u32>,
}

impl CudaGraphDecodeStep {
    /// Returns None when the step can't be padded (no host rows to rebuild the metadata from, or a
    /// hybrid batch without a pad slot).
    pub(crate) fn padded(
        inputs: CudaGraphDecodeStepInputs<'_>,
        batch: usize,
    ) -> candle_core::Result<Option<Self>> {
        let CudaGraphDecodeStepInputs {
            input_ids,
            seqlen_offsets,
            context_lens,
            position_ids,
            metadata,
            state_indices,
            pad_slot,
        } = inputs;
        let real_batch = input_ids.dim(0)?;
        if real_batch == batch {
            return Ok(Some(Self {
                input_ids: input_ids.clone(),
                seqlen_offsets: seqlen_offsets.to_vec(),
                context_lens: context_lens.to_vec(),
                position_ids: position_ids.to_vec(),
                metadata: metadata.clone(),
                state_indices: state_indices.map(<[u32]>::to_vec),
                real_batch,
            }));
        }
        let Some(rows) = metadata.decode_rows.as_ref() else {
            return Ok(None);
        };
        let state_indices = match (state_indices, pad_slot) {
            (Some(slots), Some(pad_slot)) => {
                let mut padded = slots.to_vec();
                padded.resize(batch, pad_slot);
                Some(padded)
            }
            (Some(_), None) => return Ok(None),
            (None, _) => None,
        };
        let pad = batch - real_batch;
        let (_, q_len) = input_ids.dims2()?;
        let input_ids = if input_ids.dtype() == DType::U32 && input_ids.device().is_cuda() {
            crate::cuda::input_packing::pad_decode_input(input_ids, batch)?
        } else {
            let pad_ids = input_ids.narrow(0, 0, 1)?.repeat((pad, 1))?;
            Tensor::cat(&[input_ids, &pad_ids], 0)?
        };
        let mut seqlen_offsets = seqlen_offsets.to_vec();
        seqlen_offsets.resize(batch, seqlen_offsets[0]);
        let mut context_lens = context_lens.to_vec();
        context_lens.resize(batch, context_lens[0]);
        let mut position_ids = position_ids.to_vec();
        position_ids.resize(batch, position_ids[0]);
        let rows = Arc::new(rows.padded(batch));
        if rows.query_len != q_len {
            candle_core::bail!(
                "CUDA graph decode rows cover {} query tokens but the input has {q_len}",
                rows.query_len
            );
        }
        let metadata = rows.build().map_err(candle_core::Error::msg)?;
        Ok(Some(Self {
            input_ids,
            seqlen_offsets,
            context_lens,
            position_ids,
            metadata,
            state_indices,
            real_batch,
        }))
    }

    pub(crate) fn batch(&self) -> usize {
        self.seqlen_offsets.len()
    }

    /// Drops the pad rows from a `[batch, ...]` or `[batch * q, ...]` output.
    pub(crate) fn narrow_rows(&self, tensor: &Tensor) -> candle_core::Result<Tensor> {
        let batch = self.batch();
        if batch == self.real_batch {
            return Ok(tensor.clone());
        }
        let rows = tensor.dim(0)? / batch * self.real_batch;
        tensor.narrow(0, 0, rows)
    }

    fn one_token_continuation(&self, input_ids: Tensor) -> candle_core::Result<Option<Self>> {
        let (batch, q_len) = input_ids.dims2()?;
        let Some(rows) = self.metadata.decode_rows.as_ref() else {
            return Ok(None);
        };
        if q_len != 1
            || rows.query_len != 1
            || rows.decode_window != 1
            || batch != self.batch()
            || rows.batch_size() != batch
            || self.real_batch == 0
            || self.real_batch > batch
            || self.seqlen_offsets.len() != batch
            || self.context_lens.len() != batch
            || self.position_ids.len() != batch
        {
            return Ok(None);
        }

        let mut slot_mappings = Vec::with_capacity(self.real_batch);
        let mut block_tables = Vec::with_capacity(self.real_batch);
        let mut context_lens = Vec::with_capacity(self.real_batch);
        let mut full_context_lens = Vec::with_capacity(self.real_batch);
        for row in 0..self.real_batch {
            let Some(&current_slot) = rows.slot_mappings[row].first() else {
                return Ok(None);
            };
            let Ok(current_slot) = usize::try_from(current_slot) else {
                return Ok(None);
            };
            let full_table = rows.full_block_table(row);
            let current_block = current_slot / rows.block_size;
            let Some(current_block_idx) =
                full_table.iter().position(|&block| block == current_block)
            else {
                return Ok(None);
            };
            let current_block_offset = current_slot % rows.block_size;
            let next_slot = if current_block_offset + 1 < rows.block_size {
                current_slot + 1
            } else {
                let next_block_idx = current_block_idx + 1;
                let Some(&next_block) = full_table.get(next_block_idx) else {
                    return Ok(None);
                };
                let Some(next_slot) = next_block.checked_mul(rows.block_size) else {
                    return Ok(None);
                };
                next_slot
            };
            let Ok(next_slot) = i64::try_from(next_slot) else {
                return Ok(None);
            };
            let Some(next_full_context_len) = rows.full_context_lens[row].checked_add(1) else {
                return Ok(None);
            };

            let paged_context_len = match rows.sliding_window {
                Some(window) => {
                    let window_start = next_full_context_len.saturating_sub(window);
                    let block_aligned_start = window_start / rows.block_size * rows.block_size;
                    next_full_context_len - block_aligned_start
                }
                None => next_full_context_len,
            };
            slot_mappings.push(vec![next_slot]);
            block_tables.push(
                rows.block_tables
                    .table_arc(rows.block_tables.row_table_index(row)),
            );
            context_lens.push(paged_context_len);
            full_context_lens.push(next_full_context_len);
        }

        let rows = Arc::new(
            DecodePagedRows {
                slot_mappings,
                block_tables: BlockTableSnapshot::from_sequence_tables(block_tables, 1),
                context_lens,
                full_context_lens,
                query_len: 1,
                block_size: rows.block_size,
                use_standard_metadata: rows.use_standard_metadata,
                max_paged_context_len: rows.max_paged_context_len,
                sliding_window: rows.sliding_window,
                decode_window: rows.decode_window,
                devices: rows.devices.clone(),
                num_kv_heads: rows.num_kv_heads,
            }
            .padded(batch),
        );
        let metadata = rows.build_graph_staged().map_err(candle_core::Error::msg)?;
        let Some(mut seqlen_offsets) = self.seqlen_offsets[..self.real_batch]
            .iter()
            .map(|offset| offset.checked_add(1))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        seqlen_offsets.resize(batch, seqlen_offsets[0]);
        let mut context_lens = self.context_lens[..self.real_batch].to_vec();
        context_lens.resize(batch, context_lens[0]);
        let Some(mut position_ids) = self.position_ids[..self.real_batch]
            .iter()
            .map(|position| position.checked_add(1))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        position_ids.resize(batch, position_ids[0]);
        Ok(Some(Self {
            input_ids,
            seqlen_offsets,
            context_lens,
            position_ids,
            metadata,
            state_indices: self.state_indices.clone(),
            real_batch: self.real_batch,
        }))
    }
}

/// A fabricated batch-1 decode step (token 0 at position 0 over one block, no KV writes) that the
/// precapture pads up to every bucket.
pub(crate) struct CudaGraphPrecaptureInputs {
    pub(crate) input_ids: Tensor,
    pub(crate) seqlen_offsets: Vec<usize>,
    pub(crate) context_lens: Vec<(usize, usize)>,
    pub(crate) position_ids: Vec<usize>,
    pub(crate) metadata: PagedAttentionInputMetadata,
    pub(crate) flash_meta: FlashParams,
}

impl CudaGraphPrecaptureInputs {
    pub(crate) fn new(
        ctx: &DecodeGraphPrecaptureCtx,
        q_len: usize,
        device: &Device,
        mapper: Option<&dyn DeviceMapper>,
    ) -> candle_core::Result<Self> {
        let devices = mapper
            .map(|mapper| mapper.get_unique_devices())
            .unwrap_or_else(|| vec![device.clone()]);
        let rows = Arc::new(DecodePagedRows {
            slot_mappings: vec![vec![_PAD_SLOT_ID; q_len]],
            block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![vec![0]], q_len),
            context_lens: vec![1; q_len],
            full_context_lens: vec![1; q_len],
            query_len: q_len,
            block_size: ctx.block_size,
            use_standard_metadata: ctx.attention_backend == AttentionBackendKind::Standard,
            max_paged_context_len: ctx.max_paged_context_len,
            sliding_window: ctx.sliding_window,
            decode_window: 1,
            devices,
            num_kv_heads: ctx.num_kv_heads,
        });
        let metadata = rows.build_materialized().map_err(candle_core::Error::msg)?;
        let q_len_u32 = u32::try_from(q_len).map_err(candle_core::Error::wrap)?;
        let flash_meta = if crate::using_flash_attn() {
            make_flash_params(
                device,
                mapper,
                &[0, q_len_u32],
                &[0, q_len_u32],
                ctx.sliding_window,
                true,
                false,
            )
            .map_err(candle_core::Error::msg)?
        } else {
            FlashParams::empty(true)
        };
        Ok(Self {
            input_ids: Tensor::zeros((1, q_len), DType::U32, device)?,
            seqlen_offsets: vec![0],
            context_lens: vec![(0, q_len)],
            position_ids: vec![q_len],
            metadata,
            flash_meta,
        })
    }

    pub(crate) fn step_inputs<'a>(
        &'a self,
        state_indices: Option<&'a [u32]>,
        pad_slot: Option<u32>,
    ) -> CudaGraphDecodeStepInputs<'a> {
        CudaGraphDecodeStepInputs {
            input_ids: &self.input_ids,
            seqlen_offsets: &self.seqlen_offsets,
            context_lens: &self.context_lens,
            position_ids: &self.position_ids,
            metadata: &self.metadata,
            state_indices,
            pad_slot,
        }
    }
}

pub(crate) struct HybridGraphSlots {
    pub(crate) real: Vec<u32>,
    pub(crate) storage_generation: u64,
}

/// The batch's live recurrent slots after reserving graph capacity.
pub(crate) fn hybrid_graph_slots(
    cache: &mut HybridCache,
) -> candle_core::Result<Option<HybridGraphSlots>> {
    let Some(real) = cache.state_indices_host().map(<[u32]>::to_vec) else {
        return Ok(None);
    };
    cache.graph_pad_slot()?;
    Ok(Some(HybridGraphSlots {
        real,
        storage_generation: cache.recurrent_storage_generation(),
    }))
}

/// Points the hybrid cache's state indices at fresh `Var` buffers holding `host`, one per recurrent
/// device, so a captured forward reads slots the replay can overwrite.
pub(crate) fn install_hybrid_graph_state_indices(
    cache: &mut HybridCache,
    host: &[u32],
) -> candle_core::Result<CudaGraphVarMap> {
    let mut vars = CudaGraphVarMap::new();
    let mut tensors = Vec::new();
    for device in cache.recurrent_devices() {
        let var = Var::from_tensor(&Tensor::from_vec(host.to_vec(), (host.len(),), &device)?)?;
        tensors.push((device.clone(), var.as_detached_tensor()));
        vars.insert(device.location(), var);
    }
    cache.set_state_indices_tensors(host.to_vec(), tensors);
    Ok(vars)
}

fn copy_state_indices(
    dst: &CudaGraphVarMap,
    host: &[u32],
    host_staging: &mut CudaGraphHostStaging,
) -> candle_core::Result<()> {
    for (location, var) in dst {
        host_staging.copy_from_u32_slice("state_indices", *location, host, var)?;
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CudaDecodeGraphKey {
    device: DeviceLocation,
    input_shape: Vec<usize>,
    input_dtype: DType,
    recurrent_batch_kind: RecurrentBatchKind,
    tensors: Vec<CudaGraphTensorKey>,
    decode_rows: Option<DecodePagedRowsGraphKey>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CudaGraphTensorKey {
    name: &'static str,
    location: DeviceLocation,
    shape: Vec<usize>,
    dtype: DType,
}

type CudaGraphVarMap = HashMap<DeviceLocation, Var>;

pub(crate) struct CudaDecodeGraphCaptureCtx<'a> {
    pub(crate) key: CudaDecodeGraphKey,
    pub(crate) input_ids: &'a Tensor,
    pub(crate) seqlen_offsets: &'a [usize],
    pub(crate) position_ids: &'a [usize],
    pub(crate) kv_cache: &'a [(Tensor, Tensor)],
    pub(crate) metadata: &'a PagedAttentionInputMetadata,
    pub(crate) model_metadata: Option<&'a (dyn ModelConfigLike + Send + Sync)>,
    pub(crate) activation_dtype: DType,
    pub(crate) warmup_logits: &'a Tensor,
    pub(crate) state_indices: Option<CudaGraphVarMap>,
    pub(crate) real_batch: usize,
}

struct CudaDecodeGraphMetadataInput<'a> {
    metadata: &'a PagedAttentionInputMetadata,
    seqlen_offsets: &'a [usize],
    position_ids: &'a [usize],
    seq_len: usize,
    kv_cache: &'a [(Tensor, Tensor)],
    model_metadata: Option<&'a (dyn ModelConfigLike + Send + Sync)>,
    activation_dtype: DType,
}

pub(crate) struct CudaDecodeGraphMetadataBuffers {
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

impl CudaDecodeGraphKey {
    fn has_same_spec_state_shape(&self, other: &Self) -> bool {
        self.device == other.device
            && self.input_shape == other.input_shape
            && self.input_dtype == other.input_dtype
            && self.recurrent_batch_kind == other.recurrent_batch_kind
    }

    pub(crate) fn new(
        input_ids: &Tensor,
        metadata: &PagedAttentionInputMetadata,
        recurrent_batch_kind: RecurrentBatchKind,
    ) -> candle_core::Result<Self> {
        let decode_rows = metadata.decode_rows.as_ref().map(|rows| rows.graph_key());
        let mut tensors = Vec::new();
        if decode_rows.is_none() {
            push_graph_tensor_keys("slot_mappings", Some(&metadata.slot_mappings), &mut tensors);
            push_graph_tensor_keys("block_tables", metadata.block_tables.as_ref(), &mut tensors);
            push_graph_tensor_keys("context_lens", metadata.context_lens.as_ref(), &mut tensors);
            push_graph_tensor_keys(
                "full_block_tables",
                metadata.full_block_tables.as_ref(),
                &mut tensors,
            );
            push_graph_tensor_keys(
                "full_context_lens",
                metadata.full_context_lens.as_ref(),
                &mut tensors,
            );
            push_flashinfer_graph_tensor_keys(metadata, &mut tensors);
            if flashinfer_views_alias(metadata) {
                tensors.retain(|tensor| !tensor.name.starts_with("full_"));
            }
        }
        tensors.sort_by(|a, b| {
            a.name.cmp(b.name).then_with(|| {
                device_location_sort_key(&a.location).cmp(&device_location_sort_key(&b.location))
            })
        });

        Ok(Self {
            device: input_ids.device().location(),
            input_shape: input_ids.dims().to_vec(),
            input_dtype: input_ids.dtype(),
            recurrent_batch_kind,
            tensors,
            decode_rows,
        })
    }
}

impl CudaDecodeGraphMetadataBuffers {
    fn new(
        input: CudaDecodeGraphMetadataInput<'_>,
    ) -> candle_core::Result<(Self, PagedAttentionInputMetadata)> {
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
            candle_core::bail!(
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

    fn finish_capture(&mut self, metadata: &PagedAttentionInputMetadata) {
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

    fn copy_from(
        &mut self,
        metadata: &PagedAttentionInputMetadata,
        position_ids: &[usize],
        seq_len: usize,
        host_staging: &mut CudaGraphHostStaging,
    ) -> candle_core::Result<()> {
        let graph_update = if metadata.has_host_staged_decode_tensors() {
            Some(
                metadata
                    .decode_rows
                    .as_ref()
                    .expect("host-staged decode metadata requires source rows")
                    .build_graph_update(self.requirements)
                    .map_err(candle_core::Error::msg)?,
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

    fn metadata_from(&self, metadata: &PagedAttentionInputMetadata) -> PagedAttentionInputMetadata {
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

#[derive(Clone, Default)]
pub(crate) struct CudaGraphSpecStateUsage {
    bytes: HashMap<DeviceLocation, usize>,
    device_totals: HashMap<DeviceLocation, usize>,
}

impl CudaGraphSpecStateUsage {
    fn from_state(state: &dyn SpeculativeGraphState) -> candle_core::Result<Self> {
        let mut usage = Self::default();
        for tensor in state.tensors() {
            let location = tensor.device().location();
            let bytes = tensor
                .elem_count()
                .saturating_mul(tensor.dtype().size_in_bytes());
            usage
                .bytes
                .entry(location)
                .and_modify(|total| *total = total.saturating_add(bytes))
                .or_insert(bytes);
            if let std::collections::hash_map::Entry::Vacant(entry) =
                usage.device_totals.entry(location)
            {
                let Device::Cuda(device) = tensor.device() else {
                    candle_core::bail!("CUDA graph speculative state expected CUDA tensors");
                };
                let (_, total) = device
                    .cuda_stream()
                    .context()
                    .mem_get_info()
                    .map_err(candle_core::Error::wrap)?;
                entry.insert(total);
            }
        }
        Ok(usage)
    }

    fn total_bytes(&self) -> usize {
        self.bytes
            .values()
            .fold(0usize, |total, bytes| total.saturating_add(*bytes))
    }
}

fn default_spec_state_budget(total: usize, largest_entry: usize) -> usize {
    let floor = total.saturating_mul(CUDA_GRAPH_SPEC_STATE_BUDGET_FLOOR_PERCENT) / 100;
    let ceiling = total.saturating_mul(CUDA_GRAPH_SPEC_STATE_BUDGET_CEILING_PERCENT) / 100;
    let working_set = largest_entry.saturating_mul(CUDA_GRAPH_SPEC_STATE_WORKING_SET_MULTIPLIER);
    floor.max(working_set.min(ceiling)).max(largest_entry)
}

fn configured_spec_state_budget(total: usize, largest_entry: usize) -> usize {
    std::env::var(CUDA_GRAPH_SPEC_STATE_BUDGET_BYTES_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or_else(|| default_spec_state_budget(total, largest_entry))
}

fn spec_state_eviction_plan(
    existing: &[CudaGraphSpecStateUsage],
    incoming: &CudaGraphSpecStateUsage,
    budgets: &HashMap<DeviceLocation, usize>,
) -> Vec<usize> {
    let mut totals = incoming.bytes.clone();
    for usage in existing {
        for (location, bytes) in &usage.bytes {
            totals
                .entry(*location)
                .and_modify(|total| *total = total.saturating_add(*bytes))
                .or_insert(*bytes);
        }
    }

    let mut retained = vec![true; existing.len()];
    let mut evictions = Vec::new();
    loop {
        let over_budget = totals
            .iter()
            .filter_map(|(location, bytes)| {
                (*bytes > budgets.get(location).copied().unwrap_or(usize::MAX)).then_some(*location)
            })
            .collect::<Vec<_>>();
        if over_budget.is_empty() {
            break;
        }
        let Some((idx, usage)) = existing.iter().enumerate().find(|(idx, usage)| {
            retained[*idx]
                && over_budget
                    .iter()
                    .any(|location| usage.bytes.get(location).is_some_and(|bytes| *bytes > 0))
        }) else {
            break;
        };
        retained[idx] = false;
        evictions.push(idx);
        for (location, bytes) in &usage.bytes {
            totals
                .entry(*location)
                .and_modify(|total| *total = total.saturating_sub(*bytes));
        }
    }
    evictions
}

pub(crate) struct CudaDecodeGraphEntry {
    generation: u64,
    replay_epoch: u64,
    key: CudaDecodeGraphKey,
    host_staging: CudaGraphHostStaging,
    input_ids: Var,
    metadata_buffers: CudaDecodeGraphMetadataBuffers,
    state_indices: Option<CudaGraphVarMap>,
    _metadata: PagedAttentionInputMetadata,
    logits: Tensor,
    // Proposer-facing graph outputs refreshed by replay.
    spec_state: Option<Arc<dyn SpeculativeGraphState>>,
    spec_state_usage: CudaGraphSpecStateUsage,
    // Must stay last so graph-backed tensors enqueue their frees before the graph exec is destroyed.
    graph: CudaGraphHandle,
}

pub struct CudaDecodeGraphLaunch {
    generation: u64,
    replay_epoch: u64,
    key: CudaDecodeGraphKey,
    input_ids: Tensor,
    graph_stream: Arc<CudaStream>,
    real_batch: usize,
    source: CudaGraphDecodeStep,
}

impl CudaDecodeGraphLaunch {
    pub(crate) fn resident_input(&self) -> &Tensor {
        &self.input_ids
    }

    pub(crate) fn graph_stream(&self) -> &Arc<CudaStream> {
        &self.graph_stream
    }

    #[cfg(test)]
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn real_batch(&self) -> usize {
        self.real_batch
    }

    fn one_token_continuation(&self) -> candle_core::Result<Option<CudaGraphDecodeStep>> {
        let Some(continuation) = self.source.one_token_continuation(self.input_ids.clone())? else {
            return Ok(None);
        };
        let key = CudaDecodeGraphKey::new(
            &continuation.input_ids,
            &continuation.metadata,
            self.key.recurrent_batch_kind,
        )?;
        Ok((key == self.key).then_some(continuation))
    }

    fn matches(&self, entry: &CudaDecodeGraphEntry) -> bool {
        cuda_graph_replay_version_matches(
            entry.generation,
            entry.replay_epoch,
            self.generation,
            self.replay_epoch,
        ) && self.key == entry.key
    }
}

fn cuda_graph_replay_version_matches(
    entry_generation: u64,
    entry_replay_epoch: u64,
    launch_generation: u64,
    launch_replay_epoch: u64,
) -> bool {
    entry_generation == launch_generation && entry_replay_epoch == launch_replay_epoch
}

impl fmt::Debug for CudaDecodeGraphLaunch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaDecodeGraphLaunch")
            .field("generation", &self.generation)
            .field("replay_epoch", &self.replay_epoch)
            .field("input_shape", &self.input_ids.shape())
            .field("input_device", self.input_ids.device())
            .field("real_batch", &self.real_batch)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy)]
pub(crate) enum CudaDecodeGraphReplayInput<'a> {
    Host,
    Resident(&'a CudaDecodeGraphLaunch),
}

impl CudaDecodeGraphEntry {
    pub(crate) fn with_spec_state(
        mut self,
        spec_state: Option<Box<dyn SpeculativeGraphState>>,
        usage: Option<CudaGraphSpecStateUsage>,
    ) -> Self {
        assert_eq!(spec_state.is_some(), usage.is_some());
        self.spec_state = spec_state.map(Arc::from);
        self.spec_state_usage = usage.unwrap_or_default();
        self
    }

    fn launch(
        &self,
        step: &CudaGraphDecodeStep,
        replay_epoch: u64,
    ) -> candle_core::Result<Option<CudaDecodeGraphLaunch>> {
        let input_ids = self.input_ids.as_detached_tensor();
        let (batch, q_len) = input_ids.dims2()?;
        if input_ids.dtype() != DType::U32
            || q_len != 1
            || !input_ids.is_contiguous()
            || self.spec_state.is_some()
        {
            return Ok(None);
        }
        if step.real_batch > batch {
            candle_core::bail!(
                "CUDA graph resident input has batch capacity {batch}, smaller than {} live rows",
                step.real_batch
            );
        }
        Ok(Some(CudaDecodeGraphLaunch {
            generation: self.generation,
            replay_epoch,
            key: self.key.clone(),
            input_ids,
            graph_stream: self.graph.stream().clone(),
            real_batch: step.real_batch,
            source: step.clone(),
        }))
    }

    fn release(self) -> (Arc<CudaStream>, candle_core::Result<()>) {
        let Self {
            generation: _,
            replay_epoch,
            key: _,
            host_staging,
            input_ids,
            metadata_buffers,
            state_indices,
            _metadata,
            logits,
            spec_state,
            spec_state_usage: _,
            graph,
        } = self;
        let stream = graph.stream().clone();
        let mut release_result = stream
            .synchronize()
            .map_err(candle_core::Error::wrap)
            .map_err(|err| err.context("CUDA graph entry release wait failed"));
        drop_cuda_graph_entry_resource(host_staging, &stream, "host staging", &mut release_result);
        drop_cuda_graph_entry_output(
            spec_state,
            &stream,
            replay_epoch,
            "speculative state",
            &mut release_result,
        );
        drop_cuda_graph_entry_output(logits, &stream, replay_epoch, "logits", &mut release_result);
        drop_cuda_graph_entry_resource(
            _metadata,
            &stream,
            "paged-attention metadata",
            &mut release_result,
        );
        drop_cuda_graph_entry_resource(
            state_indices,
            &stream,
            "state indices",
            &mut release_result,
        );
        drop_cuda_graph_entry_resource(
            metadata_buffers,
            &stream,
            "metadata buffers",
            &mut release_result,
        );
        drop_cuda_graph_entry_resource(input_ids, &stream, "input ids", &mut release_result);
        let storage_result = stream
            .synchronize()
            .map_err(candle_core::Error::wrap)
            .map_err(|err| err.context("CUDA graph entry storage release failed"));
        if release_result.is_ok() {
            release_result = storage_result;
        }
        drop(graph);
        (stream, release_result)
    }
}

fn unmaterialized_graph_output_drop_error(replay_epoch: u64, error: sys::CUresult) -> bool {
    replay_epoch == 0 && error == sys::CUresult::CUDA_ERROR_INVALID_VALUE
}

fn drop_cuda_graph_entry_output<T>(
    output: T,
    stream: &Arc<CudaStream>,
    replay_epoch: u64,
    name: &'static str,
    release_result: &mut candle_core::Result<()>,
) {
    drop(output);
    if let Err(err) = stream.context().check_err() {
        if unmaterialized_graph_output_drop_error(replay_epoch, err.0) {
            return;
        }
        if release_result.is_ok() {
            *release_result = Err(candle_core::Error::wrap(err)
                .context(format!("CUDA graph entry {name} release failed")));
        }
    }
}

fn drop_cuda_graph_entry_resource<T>(
    resource: T,
    stream: &Arc<CudaStream>,
    name: &'static str,
    release_result: &mut candle_core::Result<()>,
) {
    drop(resource);
    if let Err(err) = stream.context().check_err()
        && release_result.is_ok()
    {
        *release_result = Err(candle_core::Error::wrap(err)
            .context(format!("CUDA graph entry {name} release failed")));
    }
}

pub(crate) struct CudaDecodeGraphReplay {
    pub(crate) logits: Tensor,
    pub(crate) spec_state: Option<Arc<dyn SpeculativeGraphState>>,
    pub(crate) launch: Option<CudaDecodeGraphLaunch>,
}

pub(crate) struct CudaDecodeGraphState {
    entries: Vec<CudaDecodeGraphEntry>,
    spec_state_budgets: HashMap<DeviceLocation, usize>,
    capacity: usize,
    disabled: bool,
    suspended: bool,
    eager_retry_blocked: bool,
    recurrent_storage_generation: Option<u64>,
}

impl Default for CudaDecodeGraphState {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            spec_state_budgets: HashMap::new(),
            capacity: TARGET_CUDA_DECODE_GRAPH_CACHE_DEFAULT_CAPACITY,
            disabled: false,
            suspended: false,
            eager_retry_blocked: false,
            recurrent_storage_generation: None,
        }
    }
}

impl Drop for CudaDecodeGraphState {
    fn drop(&mut self) {
        record_cuda_graph_resident_entries(CudaGraphComponent::Target, 0);
    }
}

impl CudaDecodeGraphState {
    pub(crate) fn ensure_capacity(&mut self, capacity: usize) {
        self.capacity = self.capacity.max(capacity);
    }

    pub(crate) fn disabled(&self) -> bool {
        self.disabled || self.suspended
    }

    pub(crate) fn disable(&mut self) {
        self.disabled = true;
        self.clear();
    }

    pub(crate) fn take_eager_retry_allowed(&mut self) -> bool {
        !std::mem::take(&mut self.eager_retry_blocked)
    }

    pub(crate) fn block_eager_retry(&mut self) {
        self.eager_retry_blocked = true;
    }

    pub(crate) fn clear(&mut self) {
        self.eager_retry_blocked = false;
        let entries = std::mem::take(&mut self.entries);
        record_cuda_graph_resident_entries(CudaGraphComponent::Target, 0);
        release_cuda_graph_entries(entries);
    }

    pub(crate) fn evict_lru_for_memory_pressure(&mut self, max_entries: usize) -> usize {
        let entries = drain_lru_entries(&mut self.entries, max_entries);
        let evicted = entries.len();
        if evicted == 0 {
            return 0;
        }
        record_cuda_graph_resident_entries(CudaGraphComponent::Target, self.entries.len());
        release_cuda_graph_entries(entries);
        record_cuda_graph_evictions(
            CudaGraphComponent::Target,
            CudaGraphEvictionReason::MemoryPressure,
            evicted,
        );
        evicted
    }

    pub(crate) fn observe_recurrent_storage_generation(&mut self, generation: u64) {
        let previous = self.recurrent_storage_generation.replace(generation);
        if previous.is_some_and(|previous| previous != generation) {
            self.clear();
        }
    }

    pub(crate) fn suspend(&mut self) {
        self.suspended = true;
        self.clear();
    }

    pub(crate) fn resume(&mut self) {
        self.suspended = false;
        self.clear();
    }

    pub(crate) fn contains(&self, key: &CudaDecodeGraphKey) -> bool {
        self.entries.iter().any(|entry| entry.key == *key)
    }

    pub(crate) fn replay(
        &mut self,
        key: &CudaDecodeGraphKey,
        step: &CudaGraphDecodeStep,
        input: CudaDecodeGraphReplayInput<'_>,
    ) -> candle_core::Result<Option<CudaDecodeGraphReplay>> {
        let Some(pos) = self.entries.iter().position(|entry| entry.key == *key) else {
            return Ok(None);
        };
        let mut entry = self.entries.remove(pos);
        if let CudaDecodeGraphReplayInput::Resident(launch) = input
            && (!launch.matches(&entry) || launch.real_batch != step.real_batch)
        {
            self.entries.push(entry);
            return Ok(None);
        }
        let graph_event =
            CudaGraphEventGuard::new(CudaGraphComponent::Target, CudaGraphEvent::Replay);
        let prelaunch = (|| -> candle_core::Result<_> {
            match input {
                CudaDecodeGraphReplayInput::Host => {
                    entry.input_ids.set(&step.input_ids).map_err(|err| {
                        err.context(format!(
                            "CUDA graph input update failed for generation {}, replay epoch {}, {} live rows, key {:?}",
                            entry.generation, entry.replay_epoch, step.real_batch, entry.key
                        ))
                    })?
                }
                CudaDecodeGraphReplayInput::Resident(_) => {}
            }
            let replay_epoch = entry
                .replay_epoch
                .checked_add(1)
                .expect("CUDA decode graph replay epoch overflow");
            let spec_state = match &entry.spec_state {
                Some(state) if step.real_batch == step.input_ids.dim(0)? => Some(state.clone()),
                Some(state) => Some(Arc::from(state.for_real_batch(step.real_batch)?)),
                None => None,
            };
            let replay = CudaDecodeGraphReplay {
                logits: step.narrow_rows(&entry.logits)?,
                spec_state,
                launch: entry.launch(step, replay_epoch)?,
            };
            let (_, seq_len) = step.input_ids.dims2()?;
            let metadata_buffers = &mut entry.metadata_buffers;
            let state_indices = &entry.state_indices;
            entry
                .host_staging
                .update(|host_staging| {
                    metadata_buffers.copy_from(
                        &step.metadata,
                        &step.position_ids,
                        seq_len,
                        host_staging,
                    )?;
                    match (state_indices, &step.state_indices) {
                        (Some(dst), Some(host)) => copy_state_indices(dst, host, host_staging),
                        (None, None) => Ok(()),
                        _ => candle_core::bail!(
                            "hybrid state indices changed optional state during CUDA graph replay"
                        ),
                    }
                })
                .map_err(|err| {
                    err.context(format!(
                        "CUDA graph metadata update failed for generation {}, replay epoch {}, {} live rows, key {:?}",
                        entry.generation, entry.replay_epoch, step.real_batch, entry.key
                    ))
                })?;
            entry.host_staging.order_before_graph().map_err(|err| {
                err.context(format!(
                    "CUDA graph metadata ordering failed for generation {}, replay epoch {}, {} live rows, key {:?}",
                    entry.generation, entry.replay_epoch, step.real_batch, entry.key
                ))
            })?;
            Ok(Some((replay_epoch, replay)))
        })();
        let (replay_epoch, mut replay) = match prelaunch {
            Ok(Some(prelaunch)) => prelaunch,
            Ok(None) => {
                self.entries.push(entry);
                return Ok(None);
            }
            Err(err) => {
                self.entries.push(entry);
                return Err(err);
            }
        };
        let (_, timing_rows) = step.input_ids.dims2()?;
        let phase_timer = CudaPhaseTimer::start(entry.graph.stream())?;
        if let Err(err) = entry.graph.launch().map_err(|err| {
            err.context(format!(
                "CUDA graph replay launch failed for generation {}, replay epoch {}, {} live rows, key {:?}",
                entry.generation, entry.replay_epoch, step.real_batch, entry.key
            ))
        }) {
            self.disabled = true;
            self.block_eager_retry();
            self.entries.push(entry);
            return Err(err);
        }
        if let Some(timer) = phase_timer {
            timer.finish("target_graph", step.real_batch, timing_rows)?;
        }
        if let Err(record_err) = entry.host_staging.record_graph_complete().map_err(|err| {
            err.context(format!(
                "CUDA graph completion recording failed for generation {}, replay epoch {}, {} live rows, key {:?}",
                entry.generation, entry.replay_epoch, step.real_batch, entry.key
            ))
        }) {
            let synchronize_result = entry
                .graph
                .stream()
                .synchronize()
                .map_err(candle_core::Error::wrap)
                .map_err(|err| err.context("CUDA graph replay recovery synchronization failed"));
            entry.replay_epoch = replay_epoch;
            replay.launch = None;
            self.disabled = true;
            self.block_eager_retry();
            self.entries.push(entry);
            return match synchronize_result {
                Ok(()) => {
                    tracing::warn!(
                        "CUDA decode graphs retired after completion recording error: {record_err:?}"
                    );
                    record_cuda_graph_dispatch(
                        CudaGraphComponent::Target,
                        CudaGraphDispatchMode::Replay,
                        CudaGraphDispatchReason::CacheHit,
                    );
                    Ok(Some(replay))
                }
                Err(synchronize_err) => {
                    tracing::warn!(
                        "CUDA decode graph completion recording and recovery synchronization failed: {record_err:?}; {synchronize_err:?}"
                    );
                    Err(candle_core::Error::msg(format!(
                        "{record_err}; CUDA graph state may have advanced and recovery failed: {synchronize_err}"
                    )))
                }
            };
        }
        entry.replay_epoch = replay_epoch;
        self.entries.push(entry);
        graph_event.success();
        record_cuda_graph_dispatch(
            CudaGraphComponent::Target,
            CudaGraphDispatchMode::Replay,
            CudaGraphDispatchReason::CacheHit,
        );
        Ok(Some(replay))
    }

    pub(crate) fn replay_one_token(
        &mut self,
        launch: CudaDecodeGraphLaunch,
    ) -> candle_core::Result<Option<CudaDecodeGraphReplay>> {
        let Some(step) = launch.one_token_continuation()? else {
            return Ok(None);
        };
        self.replay(
            &launch.key,
            &step,
            CudaDecodeGraphReplayInput::Resident(&launch),
        )
    }

    pub(crate) fn prepare_spec_state_admission(
        &mut self,
        spec_state: &dyn SpeculativeGraphState,
    ) -> candle_core::Result<CudaGraphSpecStateUsage> {
        let usage = CudaGraphSpecStateUsage::from_state(spec_state)?;
        self.evict_for_spec_state(&usage);
        Ok(usage)
    }

    pub(crate) fn prepare_spec_state_admission_for_key(&mut self, key: &CudaDecodeGraphKey) {
        let usage = self
            .entries
            .iter()
            .rev()
            .find(|entry| entry.key.has_same_spec_state_shape(key))
            .map(|entry| entry.spec_state_usage.clone());
        if let Some(usage) = usage {
            self.evict_for_spec_state(&usage);
        }
    }

    pub(crate) fn insert(&mut self, mut entry: CudaDecodeGraphEntry) {
        self.evict_for_spec_state(&entry.spec_state_usage);
        if let Some(evicted) = take_cuda_graph_capacity_eviction(&mut self.entries, self.capacity) {
            record_cuda_graph_evictions(
                CudaGraphComponent::Target,
                CudaGraphEvictionReason::Capacity,
                1,
            );
            release_cuda_graph_entries(vec![evicted]);
        }
        entry.generation = self.allocate_generation();
        self.entries.push(entry);
        record_cuda_graph_resident_entries(CudaGraphComponent::Target, self.entries.len());
    }

    fn evict_for_spec_state(&mut self, incoming: &CudaGraphSpecStateUsage) {
        for (location, total) in &incoming.device_totals {
            let incoming_bytes = incoming.bytes.get(location).copied().unwrap_or(0);
            let configured = configured_spec_state_budget(*total, incoming_bytes);
            self.spec_state_budgets
                .entry(*location)
                .and_modify(|budget| *budget = (*budget).max(configured))
                .or_insert(configured);
        }
        let usages = self
            .entries
            .iter()
            .map(|entry| entry.spec_state_usage.clone())
            .collect::<Vec<_>>();
        let mut evictions = spec_state_eviction_plan(&usages, incoming, &self.spec_state_budgets);
        if !evictions.is_empty() {
            let evicted = evictions.len();
            tracing::debug!(
                entries = evicted,
                incoming_bytes = incoming.total_bytes(),
                "Evicting CUDA graphs to stay within the speculative state budget"
            );
            evictions.sort_unstable();
            let mut entries = Vec::with_capacity(evictions.len());
            for idx in evictions.into_iter().rev() {
                entries.push(self.entries.remove(idx));
            }
            record_cuda_graph_resident_entries(CudaGraphComponent::Target, self.entries.len());
            release_cuda_graph_entries(entries);
            record_cuda_graph_evictions(
                CudaGraphComponent::Target,
                CudaGraphEvictionReason::SpecStateBudget,
                evicted,
            );
        }
    }

    fn allocate_generation(&mut self) -> u64 {
        NEXT_CUDA_DECODE_GRAPH_GENERATION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |generation| {
                generation.checked_add(1)
            })
            .expect("CUDA decode graph generation overflow")
    }
}

fn drain_lru_entries<T>(entries: &mut Vec<T>, max_entries: usize) -> Vec<T> {
    let count = max_entries.min(entries.len());
    entries.drain(..count).collect()
}

fn release_cuda_graph_entries(entries: Vec<CudaDecodeGraphEntry>) {
    let mut streams = Vec::new();
    for entry in entries {
        let (stream, release_result) = entry.release();
        if let Err(err) = release_result {
            tracing::warn!("Failed to release CUDA graph entry storage: {err:?}");
        }
        if !streams.iter().any(|known: &Arc<CudaStream>| {
            known.context().cu_device() == stream.context().cu_device()
        }) {
            streams.push(stream);
        }
    }
    for stream in streams {
        if let Err(err) = trim_cuda_graph_memory(&stream) {
            tracing::warn!("Failed to trim released CUDA graph memory: {err:?}");
        }
    }
}

pub(crate) fn capture_cuda_decode_graph<F>(
    ctx: CudaDecodeGraphCaptureCtx<'_>,
    forward: F,
) -> candle_core::Result<CudaDecodeGraphEntry>
where
    F: FnOnce(&Tensor, &PagedAttentionInputMetadata) -> candle_core::Result<Tensor>,
{
    let CudaDecodeGraphCaptureCtx {
        key,
        input_ids,
        seqlen_offsets,
        position_ids,
        kv_cache,
        metadata,
        model_metadata,
        activation_dtype,
        warmup_logits,
        state_indices,
        real_batch,
    } = ctx;
    let materialized_metadata = metadata
        .materialize_decode_tensors()
        .map_err(candle_core::Error::msg)?;
    let metadata = &materialized_metadata;
    let (batch, seq_len) = input_ids.dims2()?;
    let input_ids = Var::from_tensor(input_ids)?;
    let (mut metadata_buffers, metadata) =
        CudaDecodeGraphMetadataBuffers::new(CudaDecodeGraphMetadataInput {
            metadata,
            seqlen_offsets,
            position_ids,
            seq_len,
            kv_cache,
            model_metadata,
            activation_dtype,
        })?;
    let graph_input_ids = input_ids.as_detached_tensor();
    let Device::Cuda(cuda_device) = graph_input_ids.device() else {
        candle_core::bail!("CUDA graph decode expected CUDA input ids");
    };
    graph_input_ids.device().synchronize()?;
    let stream = cuda_device.cuda_stream();
    let _memory_pool_guard = prepare_cuda_graph_memory_pool(&stream)?;
    let restore_event_tracking = disable_event_tracking_for_capture(&stream);
    let _htod_cache_guard = cuda_device.enable_cuda_graph_htod_cache();

    if let Err(err) = stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
    {
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        return Err(
            candle_core::Error::msg(err.to_string()).context("CUDA graph begin capture failed")
        );
    }

    if let Err(err) = prepare_fa3_decode_schedules(&metadata) {
        end_cuda_capture_discard(&stream);
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        return Err(err.context("FA3 decode preparation capture failed"));
    }

    let logits = match forward(&graph_input_ids, &metadata) {
        Ok(logits) => logits,
        Err(err) => {
            end_cuda_capture_discard(&stream);
            restore_event_tracking_after_capture(&stream, restore_event_tracking);
            return Err(err.context("CUDA graph captured forward failed"));
        }
    };
    if logits.shape() != warmup_logits.shape()
        || logits.dtype() != warmup_logits.dtype()
        || logits.device().location() != warmup_logits.device().location()
        || !logits.is_contiguous()
    {
        end_cuda_capture_discard(&stream);
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        return Err(candle_core::Error::msg(
            "captured CUDA graph logits do not match the contiguous warmup output",
        ));
    }

    let graph = match CudaGraphHandle::end_capture(&stream) {
        Ok(Some(graph)) => graph,
        Ok(None) => {
            restore_event_tracking_after_capture(&stream, restore_event_tracking);
            return Err(candle_core::Error::msg(
                "CUDA graph capture returned no graph",
            ));
        }
        Err(err) => {
            restore_event_tracking_after_capture(&stream, restore_event_tracking);
            return Err(err);
        }
    };
    restore_event_tracking_after_capture(&stream, restore_event_tracking);

    graph.upload()?;
    metadata_buffers.finish_capture(&metadata);
    let host_staging = CudaGraphHostStaging::new(graph.stream().clone())?;
    tracing::debug!(
        "Captured CUDA decode graph: batch bucket {batch} ({real_batch} live rows), {seq_len} query tokens"
    );

    Ok(CudaDecodeGraphEntry {
        generation: 0,
        replay_epoch: 0,
        key,
        host_staging,
        input_ids,
        metadata_buffers,
        state_indices,
        _metadata: metadata,
        logits,
        spec_state: None,
        spec_state_usage: CudaGraphSpecStateUsage::default(),
        graph,
    })
}

pub(crate) fn cuda_decode_graph_batch_kind_supported(kind: RecurrentBatchKind) -> bool {
    matches!(
        kind,
        RecurrentBatchKind::Decode | RecurrentBatchKind::SpeculativeDecode
    )
}

pub(crate) fn cuda_decode_graph_supported_for_model(
    model_metadata: Option<&(dyn ModelConfigLike + Send + Sync)>,
) -> bool {
    let Some(metadata) = model_metadata else {
        return false;
    };
    #[cfg(target_family = "unix")]
    {
        (0..metadata.num_layers()).all(|layer_idx| {
            !DecodePlan::requires_host_context_lengths(
                metadata.attention_backend_kind_for_layer(layer_idx),
                metadata.k_head_dim_for_layer(layer_idx),
            )
        })
    }
    #[cfg(not(target_family = "unix"))]
    {
        (0..metadata.num_layers()).all(|layer_idx| {
            !matches!(
                metadata.attention_backend_kind_for_layer(layer_idx),
                AttentionBackendKind::FlashInfer
            )
        })
    }
}

fn device_location_sort_key(location: &DeviceLocation) -> (u8, usize) {
    match location {
        DeviceLocation::Cpu => (0, 0),
        DeviceLocation::Cuda { gpu_id } => (1, *gpu_id),
        DeviceLocation::Metal { gpu_id } => (2, *gpu_id),
    }
}

fn push_graph_tensor_keys(
    name: &'static str,
    map: Option<&HashMap<DeviceLocation, Tensor>>,
    keys: &mut Vec<CudaGraphTensorKey>,
) {
    if let Some(map) = map {
        keys.extend(map.iter().map(|(location, tensor)| CudaGraphTensorKey {
            name,
            location: *location,
            shape: tensor.dims().to_vec(),
            dtype: tensor.dtype(),
        }));
    }
}

fn push_flashinfer_graph_tensor_keys(
    metadata: &PagedAttentionInputMetadata,
    keys: &mut Vec<CudaGraphTensorKey>,
) {
    let paged = flashinfer_paged_view(metadata);
    let full = flashinfer_full_view(metadata);
    push_graph_tensor_keys(
        "paged_kv_indptr",
        paged.map(|view| &view.paged_kv.indptr),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_indices",
        paged.map(|view| &view.paged_kv.indices),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_last_page_len",
        paged.map(|view| &view.paged_kv.last_page_len),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_indptr",
        full.map(|view| &view.paged_kv.indptr),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_indices",
        full.map(|view| &view.paged_kv.indices),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_last_page_len",
        full.map(|view| &view.paged_kv.last_page_len),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_request_indices",
        paged.map(|view| &view.tile_plan.request_indices),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_tile_indices",
        paged.map(|view| &view.tile_plan.kv_tile_indices),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_o_indptr",
        paged.map(|view| &view.tile_plan.o_indptr),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_chunk_size",
        paged.map(|view| &view.tile_plan.kv_chunk_size),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_block_valid_mask",
        paged.map(|view| &view.tile_plan.block_valid_mask),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_request_indices",
        full.map(|view| &view.tile_plan.request_indices),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_tile_indices",
        full.map(|view| &view.tile_plan.kv_tile_indices),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_o_indptr",
        full.map(|view| &view.tile_plan.o_indptr),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_chunk_size",
        full.map(|view| &view.tile_plan.kv_chunk_size),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_block_valid_mask",
        full.map(|view| &view.tile_plan.block_valid_mask),
        keys,
    );
}

fn flashinfer_paged_view(
    metadata: &PagedAttentionInputMetadata,
) -> Option<&FlashInferPagedAttentionView> {
    let views = &metadata.flashinfer.as_ref()?.views;
    Some(views.sliding.as_ref().unwrap_or(&views.logical))
}

fn flashinfer_full_view(
    metadata: &PagedAttentionInputMetadata,
) -> Option<&FlashInferPagedAttentionView> {
    Some(&metadata.flashinfer.as_ref()?.views.logical)
}

fn flashinfer_views_alias(metadata: &PagedAttentionInputMetadata) -> bool {
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
) -> candle_core::Result<CudaGraphVarMap> {
    map.iter()
        .map(|(location, tensor)| Ok((*location, Var::from_tensor(tensor)?)))
        .collect()
}

fn option_var_map_from_tensor_map(
    map: Option<&HashMap<DeviceLocation, Tensor>>,
) -> candle_core::Result<Option<CudaGraphVarMap>> {
    map.map(var_map_from_tensor_map).transpose()
}

fn option_var_map_from_tensor_map_if_distinct(
    map: Option<&HashMap<DeviceLocation, Tensor>>,
    aliases_existing: bool,
) -> candle_core::Result<Option<CudaGraphVarMap>> {
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
) -> candle_core::Result<()> {
    if dst.len() != src.len() {
        candle_core::bail!("{name} device count changed during CUDA graph replay");
    }
    for (location, dst) in dst {
        let src = src
            .get(location)
            .ok_or_else(|| candle_core::Error::msg(format!("{name} missing {location:?}")))?;
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
) -> candle_core::Result<()> {
    match (dst, src) {
        (Some(dst), Some(src)) => copy_var_map(dst, src, name, host_staging),
        (None, None) => Ok(()),
        _ => candle_core::bail!("{name} changed optional state during CUDA graph replay"),
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
) -> candle_core::Result<()> {
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
) -> candle_core::Result<CudaGraphVarMap> {
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
) -> candle_core::Result<()> {
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

/// Drops a pipeline's captured decode graphs, and the recurrent pad slot they held.
pub(crate) fn clear_decode_graphs(
    graphs: &std::sync::Mutex<CudaDecodeGraphState>,
    cache: &crate::pipeline::EitherCache,
) {
    graphs.lock().expect("CUDA graph mutex poisoned").clear();
    if cache.is_hybrid()
        && let Err(err) = cache.hybrid().release_graph_pad_slot()
    {
        tracing::error!("Failed to release CUDA graph recurrent pad slot: {err}");
    }
}

/// Frees up to `max_entries` captured graphs: decode graphs first, then the model's speculative ones.
pub(crate) fn reclaim_decode_graphs(
    graphs: &std::sync::Mutex<CudaDecodeGraphState>,
    model: &dyn inference_nn::speculative::SpeculativeTargetMixin,
    max_entries: usize,
) -> usize {
    reclaim_cuda_graph_entries(
        max_entries,
        |limit| {
            graphs
                .lock()
                .expect("CUDA graph mutex poisoned")
                .evict_lru_for_memory_pressure(limit)
        },
        |limit| model.evict_speculative_cuda_graphs(limit),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec_usage(bytes: &[(usize, usize)]) -> CudaGraphSpecStateUsage {
        CudaGraphSpecStateUsage {
            bytes: bytes
                .iter()
                .map(|(gpu_id, bytes)| (DeviceLocation::Cuda { gpu_id: *gpu_id }, *bytes))
                .collect(),
            device_totals: HashMap::new(),
        }
    }

    #[test]
    fn capacity_eviction_removes_one_lru_entry_and_preserves_the_bound() {
        let mut entries = vec![10, 20, 30];
        assert_eq!(take_cuda_graph_capacity_eviction(&mut entries, 4), None);
        entries.push(40);
        assert_eq!(take_cuda_graph_capacity_eviction(&mut entries, 4), Some(10));
        entries.push(50);
        assert_eq!(entries, vec![20, 30, 40, 50]);
        assert_eq!(entries.len(), 4);
    }

    #[test]
    fn target_cache_retains_64_entries_before_lru_eviction() {
        let capacity = TARGET_CUDA_DECODE_GRAPH_CACHE_DEFAULT_CAPACITY;
        assert_eq!(capacity, 64);

        let mut entries = (0..capacity - 1).collect::<Vec<_>>();
        assert_eq!(
            take_cuda_graph_capacity_eviction(&mut entries, capacity),
            None
        );
        entries.push(capacity - 1);
        assert_eq!(entries.len(), capacity);

        assert_eq!(
            take_cuda_graph_capacity_eviction(&mut entries, capacity),
            Some(0),
        );
        entries.push(capacity);

        assert_eq!(entries.len(), capacity);
        assert_eq!(entries.first(), Some(&1));
        assert_eq!(entries.last(), Some(&capacity));
    }

    #[test]
    fn target_cache_capacity_tracks_large_speculative_working_sets() {
        assert_eq!(target_cuda_graph_cache_capacity(23, 23), 64);
        assert_eq!(target_cuda_graph_cache_capacity(38, 15), 64);
        assert_eq!(target_cuda_graph_cache_capacity(54, 23), 80);
        assert_eq!(target_cuda_graph_cache_capacity(61, 30), 96);
        assert_eq!(target_cuda_graph_cache_capacity(80, 23), 96);
        assert_eq!(target_cuda_graph_cache_capacity(usize::MAX, usize::MAX), 96);
    }

    #[test]
    fn target_single_token_buckets_fit_the_default_cache() {
        let bucket_count = cuda_graph_precapture_batches(CudaGraphComponent::Target, 1).count();
        assert_eq!(bucket_count, 30);
        assert_eq!(
            target_cuda_graph_cache_capacity(bucket_count, bucket_count),
            TARGET_CUDA_DECODE_GRAPH_CACHE_DEFAULT_CAPACITY
        );
    }

    #[test]
    fn startup_precapture_accepts_decode_and_verification_widths() {
        assert!(!cuda_graph_startup_capture_allowed(0));
        assert!(cuda_graph_startup_capture_allowed(1));
        assert!(cuda_graph_startup_capture_allowed(4));
        assert!(cuda_graph_startup_capture_allowed(8));
    }

    #[test]
    fn decode_graph_batch_kind_rejects_every_prefill_chunk() {
        assert!(!cuda_decode_graph_batch_kind_supported(
            RecurrentBatchKind::Prefill
        ));
        assert!(cuda_decode_graph_batch_kind_supported(
            RecurrentBatchKind::Decode
        ));
        assert!(cuda_decode_graph_batch_kind_supported(
            RecurrentBatchKind::SpeculativeDecode
        ));
    }

    #[test]
    fn decode_row_graph_key_is_independent_of_materialization() {
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
        let staged = rows.build_graph_staged().unwrap();
        let materialized = rows.build_materialized().unwrap();
        let input_ids = Tensor::zeros((1, 1), DType::U32, &Device::Cpu).unwrap();
        let staged_key =
            CudaDecodeGraphKey::new(&input_ids, &staged, RecurrentBatchKind::Decode).unwrap();
        let materialized_key =
            CudaDecodeGraphKey::new(&input_ids, &materialized, RecurrentBatchKind::Decode).unwrap();
        assert_eq!(staged_key, materialized_key);
        assert!(staged_key.tensors.is_empty());
        assert!(staged_key.decode_rows.is_some());
        let speculative_key = CudaDecodeGraphKey::new(
            &input_ids,
            &materialized,
            RecurrentBatchKind::SpeculativeDecode,
        )
        .unwrap();
        assert_ne!(staged_key, speculative_key);
        assert!(!staged_key.has_same_spec_state_shape(&speculative_key));

        let mut next_bucket_rows = (*rows).clone();
        next_bucket_rows.block_tables =
            BlockTableSnapshot::from_owned_sequence_tables(vec![(1..=65).collect()], 1);
        next_bucket_rows.context_lens = vec![2049];
        next_bucket_rows.full_context_lens = vec![2049];
        let next_bucket = Arc::new(next_bucket_rows).build_graph_staged().unwrap();
        let next_bucket_key =
            CudaDecodeGraphKey::new(&input_ids, &next_bucket, RecurrentBatchKind::Decode).unwrap();
        assert_ne!(staged_key, next_bucket_key);
        assert!(staged_key.has_same_spec_state_shape(&next_bucket_key));
    }

    const DECODE_CONTEXT_TEST_PAGE_SIZE: usize = 32;
    const DECODE_CONTEXT_TEST_MAX_BATCH: usize = 64;
    const DECODE_CONTEXT_TEST_FLOOR: usize = 2048;
    const DECODE_CONTEXT_TEST_MAX_CONTEXT: usize = 128 * 1024;
    const DECODE_CONTEXT_TEST_KV_HEADS: usize = 8;

    fn decode_context_rows(batch: usize, context: usize) -> DecodePagedRows {
        DecodePagedRows {
            slot_mappings: vec![vec![_PAD_SLOT_ID]; batch],
            block_tables: BlockTableSnapshot::from_owned_sequence_tables(
                vec![vec![0; context.div_ceil(DECODE_CONTEXT_TEST_PAGE_SIZE)]; batch],
                1,
            ),
            context_lens: vec![context; batch],
            full_context_lens: vec![context; batch],
            query_len: 1,
            block_size: DECODE_CONTEXT_TEST_PAGE_SIZE,
            use_standard_metadata: false,
            max_paged_context_len: DECODE_CONTEXT_TEST_MAX_CONTEXT,
            sliding_window: None,
            decode_window: 1,
            devices: vec![Device::Cpu],
            num_kv_heads: DECODE_CONTEXT_TEST_KV_HEADS,
        }
    }

    fn decode_context_key(rows: DecodePagedRows) -> CudaDecodeGraphKey {
        let inputs =
            Tensor::zeros((rows.slot_mappings.len(), 1), DType::U32, &Device::Cpu).unwrap();
        let metadata = Arc::new(rows).build_graph_staged().unwrap();
        CudaDecodeGraphKey::new(&inputs, &metadata, RecurrentBatchKind::Decode).unwrap()
    }

    #[test]
    fn startup_context_keys_cover_changed_admission_and_drain_batches() {
        let startup = cuda_graph_precapture_batches(CudaGraphComponent::Target, 1)
            .filter(|batch| *batch <= DECODE_CONTEXT_TEST_MAX_BATCH)
            .map(|batch| decode_context_key(decode_context_rows(1, 1).padded(batch)))
            .collect::<Vec<_>>();
        let even_warmup = [
            2, 6, 10, 14, 18, 22, 26, 30, 32, 30, 26, 22, 18, 14, 10, 6, 2,
        ];
        let odd_measured = [
            1, 5, 9, 13, 17, 21, 25, 29, 32, 31, 27, 23, 19, 15, 11, 7, 3,
        ];
        for batch in even_warmup.into_iter().chain(odd_measured) {
            let bucket = cuda_graph_batch_bucket(CudaGraphComponent::Target, 1, batch).unwrap();
            for context in [
                1,
                128,
                512,
                513,
                1024,
                1025,
                1151,
                1536,
                DECODE_CONTEXT_TEST_FLOOR,
            ] {
                let rows = decode_context_rows(batch, context).padded(bucket);
                assert!(startup.contains(&decode_context_key(rows)));
            }
        }
        assert_eq!(startup.len(), 22);
        assert!(!startup.contains(&decode_context_key(decode_context_rows(
            32,
            DECODE_CONTEXT_TEST_FLOOR + 1
        ))));
    }

    #[test]
    fn decode_context_floor_preserves_live_csr_splits_and_pad_slots() {
        let mut live_rows = decode_context_rows(3, 1025);
        live_rows.slot_mappings = vec![vec![1024]; 3];
        let rows = Arc::new(live_rows.padded(8));
        let metadata = rows.build_materialized().unwrap();
        let view = &metadata.flashinfer.as_ref().unwrap().views.logical;
        let location = Device::Cpu.location();
        let indptr = view.paged_kv.indptr[&location].to_vec1::<i32>().unwrap();
        assert_eq!(indptr, (0..=8).map(|row| row * 33).collect::<Vec<_>>());
        assert_eq!(
            view.paged_kv.indices[&location].dims(),
            &[8 * (DECODE_CONTEXT_TEST_FLOOR / DECODE_CONTEXT_TEST_PAGE_SIZE)]
        );
        assert_eq!(
            view.paged_kv.last_page_len[&location]
                .to_vec1::<i32>()
                .unwrap(),
            vec![1; 8]
        );
        let chunk = view.tile_plan.kv_chunk_size[&location]
            .to_vec1::<i32>()
            .unwrap()[0];
        let chunks_per_row = 1025usize.div_ceil(chunk as usize);
        let ends = view.tile_plan.o_indptr[&location].to_vec1::<i32>().unwrap();
        assert_eq!(
            ends,
            (0..=8)
                .map(|row| i32::try_from(row * chunks_per_row).unwrap())
                .collect::<Vec<_>>()
        );
        let mask = view.tile_plan.block_valid_mask[&location]
            .to_vec1::<u8>()
            .unwrap();
        assert_eq!(mask.len(), 8 * 8);
        assert!(mask[..8 * chunks_per_row].iter().all(|valid| *valid == 1));
        assert!(mask[8 * chunks_per_row..].iter().all(|valid| *valid == 0));
        assert_eq!(rows.slot_mappings[..3], vec![vec![1024]; 3]);
        assert!(
            rows.slot_mappings[3..]
                .iter()
                .all(|row| row == &[_PAD_SLOT_ID])
        );
        assert_eq!(
            decode_context_key((*rows).clone()),
            decode_context_key(decode_context_rows(1, 1).padded(8))
        );
    }

    #[test]
    fn decode_context_floor_preserves_sliding_and_full_capacities() {
        let mut startup = decode_context_rows(1, 1);
        startup.sliding_window = Some(128);
        let mut live = decode_context_rows(1, 1025);
        live.sliding_window = Some(128);
        live.context_lens = vec![129];
        assert_eq!(
            decode_context_key(startup),
            decode_context_key(live.clone())
        );
        let metadata = Arc::new(live).build_materialized().unwrap();
        let views = &metadata.flashinfer.as_ref().unwrap().views;
        let location = Device::Cpu.location();
        let sliding = views.sliding.as_ref().unwrap();
        assert_eq!(sliding.paged_kv.indices[&location].dims(), &[5]);
        assert_eq!(
            views.logical.paged_kv.indices[&location].dims(),
            &[DECODE_CONTEXT_TEST_FLOOR / DECODE_CONTEXT_TEST_PAGE_SIZE]
        );
        assert_eq!(
            sliding.paged_kv.indptr[&location].to_vec1::<i32>().unwrap(),
            vec![0, 5]
        );
        assert_eq!(
            views.logical.paged_kv.indptr[&location]
                .to_vec1::<i32>()
                .unwrap(),
            vec![0, 33]
        );
    }

    #[test]
    fn decode_context_floor_respects_small_caps_and_standard_metadata() {
        let mut short = decode_context_rows(1, 1);
        short.max_paged_context_len = 1024;
        let short_key = decode_context_key(short.clone());
        let mut later = decode_context_rows(1, 1024);
        later.max_paged_context_len = 1024;
        assert_eq!(short_key, decode_context_key(later));
        let mut standard = short.clone();
        standard.use_standard_metadata = true;
        let mut standard_later = decode_context_rows(1, 1025);
        standard_later.use_standard_metadata = true;
        standard_later.max_paged_context_len = 1024;
        assert_ne!(
            decode_context_key(standard),
            decode_context_key(standard_later)
        );
        let mut staged_decode = short;
        staged_decode.decode_window = 2;
        let mut staged_later = decode_context_rows(1, 1025);
        staged_later.decode_window = 2;
        staged_later.max_paged_context_len = 1024;
        assert_ne!(
            decode_context_key(staged_decode),
            decode_context_key(staged_later)
        );
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
        let input_ids = Tensor::zeros((1, 1), DType::U32, &Device::Cpu).unwrap();
        let key =
            CudaDecodeGraphKey::new(&input_ids, &metadata, RecurrentBatchKind::Decode).unwrap();
        assert!(
            key.tensors
                .iter()
                .all(|tensor| !tensor.name.starts_with("full_"))
        );

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
    fn graph_metadata_replay_updates_persistent_cuda_buffers() -> anyhow::Result<()> {
        skip_without_cuda!();
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
        let state_indices = Var::from_tensor(&Tensor::zeros((3,), DType::U32, &device)?)?;
        let mut state_indices_map = HashMap::from([(location, state_indices)]);
        let mut host_staging = CudaGraphHostStaging::new(device.as_cuda_device()?.cuda_stream())?;
        host_staging.update(|host_staging| {
            buffers.copy_from(&updated, &[256], 1, host_staging)?;
            copy_state_indices(&state_indices_map, &[3, 5, 7], host_staging)
        })?;
        host_staging.update(|host_staging| {
            buffers.copy_from(&updated, &[256], 1, host_staging)?;
            copy_state_indices(&state_indices_map, &[11, 13, 17], host_staging)
        })?;
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

        let state_indices = state_indices_map
            .remove(&location)
            .unwrap()
            .as_detached_tensor()
            .to_device(&Device::Cpu)?
            .to_vec1::<u32>()?;
        assert_eq!(state_indices, vec![11, 13, 17]);
        Ok(())
    }

    #[test]
    fn graph_replay_retains_captured_output_storage() -> anyhow::Result<()> {
        skip_without_cuda!();
        let device = Device::new_cuda(0)?;
        let cuda_device = device.as_cuda_device()?;
        let stream = cuda_device.cuda_stream();
        let _memory_pool_guard = prepare_cuda_graph_memory_pool(&stream)?;

        let input = Var::from_tensor(&Tensor::from_vec(vec![1f32, 2.0], 2, &device)?)?;
        let warmup = input.as_detached_tensor().affine(2.0, 1.0)?;
        device.synchronize()?;
        drop(warmup);

        let restore_event_tracking = disable_event_tracking_for_capture(&stream);
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
        let output = input.as_detached_tensor().affine(2.0, 1.0)?;
        let graph = CudaGraphHandle::end_capture(&stream)?
            .ok_or_else(|| anyhow::anyhow!("CUDA graph capture returned no graph"))?;
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        graph.upload()?;

        for (values, expected) in [
            (vec![3f32, 5.0], vec![7f32, 11.0]),
            (vec![8f32, 13.0], vec![17f32, 27.0]),
        ] {
            input.set(&Tensor::from_vec(values, 2, &device)?)?;
            graph.launch()?;
            stream.synchronize()?;
            assert_eq!(output.to_vec1::<f32>()?, expected);
        }

        drop(graph);
        drop(output);
        device.synchronize()?;
        Ok(())
    }

    #[test]
    fn replayed_graph_output_releases_without_driver_error() -> anyhow::Result<()> {
        skip_without_cuda!();
        let device = Device::new_cuda(0)?;
        let stream = device.as_cuda_device()?.cuda_stream();
        let _memory_pool_guard = prepare_cuda_graph_memory_pool(&stream)?;
        let input = Var::from_tensor(&Tensor::from_vec(vec![1f32, 2.0], 2, &device)?)?;

        let restore_event_tracking = disable_event_tracking_for_capture(&stream);
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
        let output = input.as_detached_tensor().affine(2.0, 1.0)?;
        let graph = CudaGraphHandle::end_capture(&stream)?
            .ok_or_else(|| anyhow::anyhow!("CUDA graph capture returned no graph"))?;
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        graph.upload()?;
        graph.launch()?;
        stream.synchronize()?;

        let mut release_result = Ok(());
        drop_cuda_graph_entry_output(output, &stream, 1, "test output", &mut release_result);
        release_result?;
        drop(graph);
        Ok(())
    }

    #[test]
    fn graph_copy_supports_dense_row_source() -> anyhow::Result<()> {
        skip_without_cuda!();
        let device = Device::new_cuda(0)?;
        let cuda_device = device.as_cuda_device()?;
        let stream = cuda_device.cuda_stream();
        let _memory_pool_guard = prepare_cuda_graph_memory_pool(&stream)?;

        let input = Var::from_tensor(&Tensor::from_vec(
            (1u16..=16).map(f32::from).collect::<Vec<_>>(),
            (2, 2, 4),
            &device,
        )?)?;
        let output = Var::from_tensor(&Tensor::zeros((2, 2, 2), DType::F32, &device)?)?;
        let source = input.as_detached_tensor().narrow(2, 1, 2)?;
        assert!(!source.is_contiguous());

        let restore_event_tracking = disable_event_tracking_for_capture(&stream);
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
        crate::cuda::graph::copy_tensor(&source, &output.as_detached_tensor())?;
        let graph = CudaGraphHandle::end_capture(&stream)?
            .ok_or_else(|| anyhow::anyhow!("CUDA graph capture returned no graph"))?;
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        graph.upload()?;

        graph.launch()?;
        stream.synchronize()?;
        assert_eq!(
            output.as_detached_tensor().to_vec3::<f32>()?,
            vec![
                vec![vec![2.0, 3.0], vec![6.0, 7.0]],
                vec![vec![10.0, 11.0], vec![14.0, 15.0]],
            ]
        );

        input.set(&Tensor::from_vec(
            (1u16..=16).rev().map(f32::from).collect::<Vec<_>>(),
            (2, 2, 4),
            &device,
        )?)?;
        graph.launch()?;
        stream.synchronize()?;
        assert_eq!(
            output.as_detached_tensor().to_vec3::<f32>()?,
            vec![
                vec![vec![15.0, 14.0], vec![11.0, 10.0]],
                vec![vec![7.0, 6.0], vec![3.0, 2.0]],
            ]
        );
        Ok(())
    }

    #[test]
    fn graph_suspension_does_not_clear_permanent_disable() {
        let mut state = CudaDecodeGraphState::default();
        state.suspend();
        assert!(state.disabled());
        state.resume();
        assert!(!state.disabled());
        state.disable();
        state.suspend();
        state.resume();
        assert!(state.disabled());
    }

    #[test]
    fn eager_retry_block_applies_to_one_failed_step() {
        let mut state = CudaDecodeGraphState::default();
        assert!(state.take_eager_retry_allowed());
        state.block_eager_retry();
        assert!(!state.take_eager_retry_allowed());
        assert!(state.take_eager_retry_allowed());
        state.block_eager_retry();
        state.clear();
        assert!(state.take_eager_retry_allowed());
    }

    #[test]
    fn graph_generations_are_not_reused_after_cleanup() {
        let mut state = CudaDecodeGraphState::default();
        let first = state.allocate_generation();
        state.clear();
        let second = state.allocate_generation();
        state.suspend();
        state.resume();
        let third = state.allocate_generation();
        assert!(first < second && second < third);
        assert!(!cuda_graph_replay_version_matches(second, 0, first, 1));
        assert!(!cuda_graph_replay_version_matches(third, 0, second, 1));
        assert!(cuda_graph_replay_version_matches(third, 7, third, 7));
    }

    #[test]
    fn memory_pressure_eviction_drains_lru_entries_in_one_batch() {
        let mut entries = vec![10, 20, 30, 40];
        assert!(drain_lru_entries(&mut entries, 0).is_empty());
        assert_eq!(drain_lru_entries(&mut entries, 2), vec![10, 20]);
        assert_eq!(entries, vec![30, 40]);
        assert_eq!(drain_lru_entries(&mut entries, usize::MAX), vec![30, 40]);
        assert!(entries.is_empty());
    }

    #[test]
    fn graph_reclaim_quota_is_shared_between_target_and_dflash() {
        let mut target_entries = 2usize;
        let mut dflash_entries = 4usize;
        let reclaimed = reclaim_cuda_graph_entries(
            3,
            |limit| {
                let reclaimed = target_entries.min(limit);
                target_entries -= reclaimed;
                reclaimed
            },
            |limit| {
                let reclaimed = dflash_entries.min(limit);
                dflash_entries -= reclaimed;
                reclaimed
            },
        );
        assert_eq!(reclaimed, 3);
        assert_eq!(target_entries, 0);
        assert_eq!(dflash_entries, 3);

        let mut dflash_called = false;
        assert_eq!(
            reclaim_cuda_graph_entries(
                2,
                |_| 2,
                |_| {
                    dflash_called = true;
                    0
                }
            ),
            2
        );
        assert!(!dflash_called);

        let mut target_called = false;
        assert_eq!(
            reclaim_cuda_graph_entries(
                0,
                |_| {
                    target_called = true;
                    0
                },
                |_| 0,
            ),
            0
        );
        assert!(!target_called);
    }

    #[test]
    fn speculative_state_budget_scales_with_the_largest_entry() {
        assert_eq!(default_spec_state_budget(10_000, 0), 400);
        assert_eq!(default_spec_state_budget(10_000, 50), 500);
        assert_eq!(default_spec_state_budget(10_000, 100), 800);
        assert_eq!(default_spec_state_budget(10_000, 200), 800);
        assert_eq!(default_spec_state_budget(10_000, 300), 800);
        assert_eq!(default_spec_state_budget(10_000, 900), 900);
    }

    #[test]
    fn only_unmaterialized_graph_output_frees_ignore_invalid_value() {
        assert!(unmaterialized_graph_output_drop_error(
            0,
            sys::CUresult::CUDA_ERROR_INVALID_VALUE
        ));
        assert!(!unmaterialized_graph_output_drop_error(
            1,
            sys::CUresult::CUDA_ERROR_INVALID_VALUE
        ));
        assert!(!unmaterialized_graph_output_drop_error(
            0,
            sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY
        ));
    }

    #[test]
    fn speculative_state_budget_evicts_lru_entries_per_device() {
        let gpu0 = DeviceLocation::Cuda { gpu_id: 0 };
        let gpu1 = DeviceLocation::Cuda { gpu_id: 1 };
        let budgets = HashMap::from([(gpu0, 100), (gpu1, 100)]);
        let existing = vec![
            spec_usage(&[]),
            spec_usage(&[(0, 40)]),
            spec_usage(&[(1, 60)]),
        ];

        assert_eq!(
            spec_state_eviction_plan(&existing, &spec_usage(&[(1, 50)]), &budgets),
            vec![2]
        );
        assert_eq!(
            spec_state_eviction_plan(&existing, &spec_usage(&[(0, 120)]), &budgets),
            vec![1]
        );
        let existing = vec![
            spec_usage(&[(0, 30)]),
            spec_usage(&[(0, 50)]),
            spec_usage(&[(0, 10)]),
        ];
        assert_eq!(
            spec_state_eviction_plan(&existing, &spec_usage(&[(0, 60)]), &budgets),
            vec![0, 1]
        );
    }

    #[test]
    fn one_token_continuation_advances_host_decode_state() -> anyhow::Result<()> {
        let rows = Arc::new(
            DecodePagedRows {
                slot_mappings: vec![vec![39]],
                block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![vec![9, 17]], 1),
                context_lens: vec![4],
                full_context_lens: vec![4],
                query_len: 1,
                block_size: 4,
                use_standard_metadata: false,
                max_paged_context_len: 64,
                sliding_window: Some(4),
                decode_window: 1,
                devices: vec![Device::Cpu],
                num_kv_heads: 1,
            }
            .padded(2),
        );
        let step = CudaGraphDecodeStep {
            input_ids: Tensor::zeros((2, 1), DType::U32, &Device::Cpu)?,
            seqlen_offsets: vec![100, 100],
            context_lens: vec![(0, 1), (0, 1)],
            position_ids: vec![43, 43],
            metadata: rows.build_graph_staged()?,
            state_indices: Some(vec![2, 5]),
            real_batch: 1,
        };
        let continuation = step
            .one_token_continuation(step.input_ids.clone())?
            .expect("next allocated block should permit one token");
        let rows = continuation.metadata.decode_rows.as_ref().unwrap();
        assert_eq!(rows.slot_mappings, vec![vec![68], vec![_PAD_SLOT_ID]]);
        assert_eq!(
            rows.materialized_block_tables(),
            vec![vec![9, 17], vec![9, 17]]
        );
        assert_eq!(rows.context_lens, vec![5, 5]);
        assert_eq!(rows.full_context_lens, vec![5, 5]);
        assert_eq!(continuation.seqlen_offsets, vec![101, 101]);
        assert_eq!(continuation.context_lens, vec![(0, 1), (0, 1)]);
        assert_eq!(continuation.position_ids, vec![44, 44]);
        assert_eq!(continuation.state_indices, Some(vec![2, 5]));
        Ok(())
    }

    #[test]
    fn one_token_continuation_requires_an_allocated_boundary_slot() -> anyhow::Result<()> {
        let rows = Arc::new(DecodePagedRows {
            slot_mappings: vec![vec![39]],
            block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![vec![9]], 1),
            context_lens: vec![4],
            full_context_lens: vec![4],
            query_len: 1,
            block_size: 4,
            use_standard_metadata: false,
            max_paged_context_len: 64,
            sliding_window: None,
            decode_window: 1,
            devices: vec![Device::Cpu],
            num_kv_heads: 1,
        });
        let step = CudaGraphDecodeStep {
            input_ids: Tensor::zeros((1, 1), DType::U32, &Device::Cpu)?,
            seqlen_offsets: vec![3],
            context_lens: vec![(0, 1)],
            position_ids: vec![4],
            metadata: rows.build_graph_staged()?,
            state_indices: None,
            real_batch: 1,
        };
        assert!(
            step.one_token_continuation(step.input_ids.clone())?
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn resident_replay_skips_the_host_input_update() -> anyhow::Result<()> {
        skip_without_cuda!();
        let device = Device::new_cuda(0)?;
        let metadata = Arc::new(DecodePagedRows {
            slot_mappings: vec![vec![0]],
            block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![vec![0]], 1),
            context_lens: vec![1],
            full_context_lens: vec![1],
            query_len: 1,
            block_size: 32,
            use_standard_metadata: false,
            max_paged_context_len: 32,
            sliding_window: None,
            decode_window: 1,
            devices: vec![device.clone()],
            num_kv_heads: 1,
        })
        .build_materialized()?;
        let initial_ids = Tensor::from_vec(vec![1u32], (1, 1), &device)?;
        let key = CudaDecodeGraphKey::new(&initial_ids, &metadata, RecurrentBatchKind::Decode)?;
        let warmup_logits = initial_ids.to_dtype(DType::F32)?;
        let entry = capture_cuda_decode_graph(
            CudaDecodeGraphCaptureCtx {
                key: key.clone(),
                input_ids: &initial_ids,
                seqlen_offsets: &[0],
                position_ids: &[1],
                kv_cache: &[],
                metadata: &metadata,
                model_metadata: None,
                activation_dtype: DType::F32,
                warmup_logits: &warmup_logits,
                state_indices: None,
                real_batch: 1,
            },
            |input_ids, _| input_ids.to_dtype(DType::F32),
        )?;
        let mut state = CudaDecodeGraphState::default();
        state.insert(entry);

        let step = |token: u32| -> candle_core::Result<CudaGraphDecodeStep> {
            Ok(CudaGraphDecodeStep {
                input_ids: Tensor::from_vec(vec![token], (1, 1), &device)?,
                seqlen_offsets: vec![0],
                context_lens: vec![(0, 1)],
                position_ids: vec![1],
                metadata: metadata.clone(),
                state_indices: None,
                real_batch: 1,
            })
        };
        let host_step = step(7)?;
        let host_replay = state
            .replay(&key, &host_step, CudaDecodeGraphReplayInput::Host)?
            .expect("captured graph missing");
        let launch = host_replay.launch.expect("qlen=1 launch missing");
        launch.graph_stream().synchronize()?;
        assert_eq!(host_replay.logits.to_vec2::<f32>()?, vec![vec![7.0]]);

        let resident_step = step(13)?;
        let resident_replay = state
            .replay(
                &key,
                &resident_step,
                CudaDecodeGraphReplayInput::Resident(&launch),
            )?
            .expect("resident graph entry changed");
        resident_replay
            .launch
            .as_ref()
            .unwrap()
            .graph_stream()
            .synchronize()?;
        assert_eq!(resident_replay.logits.to_vec2::<f32>()?, vec![vec![7.0]]);
        let resident_launch = resident_replay.launch.unwrap();
        assert_eq!(resident_launch.generation(), launch.generation());
        assert!(resident_launch.one_token_continuation()?.is_some());
        assert!(state.replay_one_token(launch)?.is_none());
        let lookahead = state
            .replay_one_token(resident_launch)?
            .expect("one-token continuation should retain the graph key");
        lookahead
            .launch
            .as_ref()
            .unwrap()
            .graph_stream()
            .synchronize()?;
        assert_eq!(lookahead.logits.to_vec2::<f32>()?, vec![vec![7.0]]);
        Ok(())
    }

    #[test]
    fn unlaunched_graph_cleanup_returns_allocator_to_baseline() -> anyhow::Result<()> {
        skip_without_cuda!();
        let device = Device::new_cuda(0)?;
        let stream = device.as_cuda_device()?.cuda_stream();
        let used_attribute = sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT;
        let reserved_attribute = sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT;
        trim_cuda_graph_memory(&stream)?;
        let used_before = cuda_graph_memory_attribute(&stream, used_attribute)?;
        let reserved_before = cuda_graph_memory_attribute(&stream, reserved_attribute)?;

        let guard = prepare_cuda_graph_memory_pool(&stream)?;
        let input = Var::from_tensor(&Tensor::from_vec(vec![1f32, 2.0], 2, &device)?)?;
        let restore_event_tracking = disable_event_tracking_for_capture(&stream);
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
        let logits = input.as_detached_tensor().affine(2.0, 1.0)?;
        let graph = CudaGraphHandle::end_capture(&stream)?
            .ok_or_else(|| anyhow::anyhow!("CUDA graph capture returned no graph"))?;
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        graph.upload()?;

        let mut release_result = Ok(());
        drop_cuda_graph_entry_output(logits, &stream, 0, "logits", &mut release_result);
        release_result?;
        drop(graph);
        drop(guard);
        trim_cuda_graph_memory(&stream)?;

        assert_eq!(
            cuda_graph_memory_attribute(&stream, used_attribute)?,
            used_before
        );
        assert_eq!(
            cuda_graph_memory_attribute(&stream, reserved_attribute)?,
            reserved_before
        );
        Ok(())
    }
}
