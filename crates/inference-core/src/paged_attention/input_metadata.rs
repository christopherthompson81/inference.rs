#![allow(clippy::cast_possible_truncation)]

use std::{collections::HashMap, fmt::Debug, sync::Arc};

use anyhow::Result;
use candle_core::{Device, DeviceLocation, Tensor, WithDType};

use crate::{
    flashinfer::{
        decode_split_capacity_pages as flashinfer_decode_split_capacity_pages,
        decode_split_pages as flashinfer_decode_split_pages, flashinfer_metadata,
        flashinfer_paged_kv, flashinfer_tile_plan, flashinfer_view, make_paged_kv_decode_tensors,
        make_paged_kv_decode_tensors_from_lens, make_paged_kv_tensors, FlashInferMetadata,
        FlashInferPagedAttentionView, FlashInferPagedAttentionViews,
    },
    paged_attention::{
        block_hash::MultimodalAttentionPolicy,
        block_table_rows::{BlockTableRanges, BlockTableRows, BlockTableSnapshot},
        _PAD_SLOT_ID,
    },
};

pub(crate) const CUDA_GRAPH_CONTEXT_BUCKET_MIN_TOKENS: usize = 512;
pub(crate) const CUDA_GRAPH_DECODE_CONTEXT_FLOOR_TOKENS: usize = 2048;

pub(crate) fn cuda_graph_context_bucket_tokens(
    required_tokens: usize,
    max_context_len: Option<usize>,
) -> usize {
    let required_tokens = required_tokens.max(1);
    let bucket = required_tokens
        .max(CUDA_GRAPH_CONTEXT_BUCKET_MIN_TOKENS)
        .checked_next_power_of_two()
        .unwrap_or(usize::MAX);
    max_context_len
        .map(|limit| bucket.min(limit.max(required_tokens)))
        .unwrap_or(bucket)
}

pub(crate) fn cuda_graph_block_table_len_with_cap(
    blocks: usize,
    block_size: usize,
    enable_cuda_graph_padding: bool,
    live_context_len: usize,
    max_context_len: Option<usize>,
) -> usize {
    if !enable_cuda_graph_padding || !crate::perf_flags::cuda_graphs_enabled() {
        return blocks;
    }
    let required_tokens = live_context_len.max(blocks.saturating_mul(block_size));
    cuda_graph_context_bucket_tokens(required_tokens, max_context_len)
        .div_ceil(block_size)
        .max(blocks)
        .max(1)
}

pub(crate) fn _make_tensor_with_pad<D: WithDType>(
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

pub(crate) fn make_block_table_tensor<T: BlockTableRows + ?Sized>(
    rows: &T,
    max_len: usize,
) -> Result<Tensor> {
    let mut values = Vec::with_capacity(rows.len() * max_len);
    for row in 0..rows.len() {
        let table = rows.row(row);
        assert!(table.len() <= max_len);
        values.extend(table.iter().map(|&block| block as u32));
        values.extend(std::iter::repeat_n(0, max_len - table.len()));
    }
    Ok(Tensor::from_vec(
        values,
        (rows.len(), max_len),
        &Device::Cpu,
    )?)
}

pub(crate) fn decode_metadata_tensor(
    tensor: &Tensor,
    device: &Device,
    stage_on_host: bool,
) -> candle_core::Result<Tensor> {
    if stage_on_host {
        Ok(tensor.clone())
    } else {
        tensor.to_device(device)
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct PagedAttentionInputMetadata {
    /// Block tables, windowed when a global sliding_window is set.
    pub block_tables: Option<HashMap<DeviceLocation, Tensor>>,
    /// Context lens, capped by sliding_window when set.
    pub context_lens: Option<HashMap<DeviceLocation, Tensor>>,
    pub block_size: Option<usize>,
    pub paged_context_lens_cpu: Option<Vec<usize>>,
    pub full_paged_context_lens_cpu: Option<Vec<usize>>,
    pub slot_mappings: HashMap<DeviceLocation, Tensor>,
    pub max_context_len: Option<usize>,
    /// Full (unwindowed) block tables, always covering the entire context.
    /// For models with per-layer sliding windows (GPT-OSS, Gemma2), layers
    /// without a sliding window should use these instead of `block_tables`.
    pub full_block_tables: Option<HashMap<DeviceLocation, Tensor>>,
    /// Full context lens (not capped by sliding_window).
    pub full_context_lens: Option<HashMap<DeviceLocation, Tensor>>,
    pub full_max_context_len: Option<usize>,
    pub is_first_prompt_chunk: bool,
    pub is_final_prompt_chunk: bool,
    pub needs_logits: bool,
    pub prompt_chunk_attention_policy: MultimodalAttentionPolicy,
    pub has_noncausal_mm_context: bool,
    pub prefix_gather_workspace_limit: Option<usize>,
    pub mm_prefix_ranges: Option<HashMap<DeviceLocation, Tensor>>,
    pub full_mm_prefix_ranges: Option<HashMap<DeviceLocation, Tensor>>,
    pub prefill_attention_heads: usize,
    pub prefill_key_value_heads: usize,
    pub prefill_head_dim: usize,
    pub flashinfer: Option<FlashInferMetadata>,
    pub rope_positions: Option<HashMap<DeviceLocation, Tensor>>,
    /// Number of cached tokens per sequence (from prefix cache hits).
    /// When present and > 0, gather_kv_cache + Sdpa is used during prefill
    /// instead of flash attention. The Q/K/V tensors should only contain
    /// the NEW (non-cached) tokens.
    pub num_cached_tokens: Option<Vec<usize>>,
    /// Number of new tokens per sequence (query lengths).
    pub query_lens: Option<Vec<usize>>,
    /// Cumulative query lengths [batch+1], u32, for Sdpa varlen flash path.
    /// Precomputed to avoid Tensor::new in the forward hot path.
    pub cu_seqlens_q: Option<HashMap<DeviceLocation, Tensor>>,
    /// Cumulative KV lengths [batch+1], u32, for gather_kv_cache and flash_attn_varlen.
    /// Each entry is sum of (cached + new) tokens.
    pub cu_seqlens_kv: Option<HashMap<DeviceLocation, Tensor>>,
    /// Host rows this decode metadata was built from (decode steps only).
    pub decode_rows: Option<Arc<DecodePagedRows>>,
}

impl PagedAttentionInputMetadata {
    pub(crate) fn is_decode_step(&self) -> bool {
        !self.is_first_prompt_chunk && self.query_lens.is_none()
    }

    pub(crate) fn has_host_staged_decode_tensors(&self) -> bool {
        self.decode_rows.is_some()
            && self
                .slot_mappings
                .iter()
                .any(|(location, tensor)| *location != tensor.device().location())
    }

    pub(crate) fn materialize_decode_tensors(&self) -> Result<Self> {
        if !self.has_host_staged_decode_tensors() {
            return Ok(self.clone());
        }
        self.decode_rows
            .as_ref()
            .expect("host-staged decode metadata requires source rows")
            .build_materialized()
    }

    /// Create a dummy input metadata, assuming that this will NOT be used for decoding.
    /// This is used for the case of imatrix generation.
    pub fn dummy(dev: &Device) -> candle_core::Result<Self> {
        Ok(PagedAttentionInputMetadata {
            block_tables: None,
            context_lens: None,
            block_size: None,
            paged_context_lens_cpu: None,
            full_paged_context_lens_cpu: None,
            max_context_len: None,
            full_block_tables: None,
            full_context_lens: None,
            full_max_context_len: None,
            slot_mappings: HashMap::from([(dev.location(), Tensor::new(&[0f32], dev)?)]),
            is_first_prompt_chunk: true,
            is_final_prompt_chunk: true,
            needs_logits: true,
            prompt_chunk_attention_policy: MultimodalAttentionPolicy::Causal,
            has_noncausal_mm_context: false,
            prefix_gather_workspace_limit: None,
            mm_prefix_ranges: None,
            full_mm_prefix_ranges: None,
            prefill_attention_heads: 1,
            prefill_key_value_heads: 1,
            prefill_head_dim: 1,
            flashinfer: None,
            rope_positions: None,
            num_cached_tokens: None,
            query_lens: None,
            cu_seqlens_q: None,
            cu_seqlens_kv: None,
            decode_rows: None,
        })
    }

    /// Build metadata for a prefill whose query tensor has been reduced to
    /// selected logits positions while K/V still live in the original paged
    /// cache. This is used by KV-sharing models that can skip hidden-state
    /// work for prompt tokens that will not produce logits.
    pub(crate) fn for_reduced_prefill_queries(
        &self,
        devices: &[Device],
        num_cached_tokens: &[usize],
        query_lens: &[usize],
    ) -> Result<Self> {
        if num_cached_tokens.len() != query_lens.len() {
            anyhow::bail!(
                "reduced prefill metadata length mismatch: cached={} query={}",
                num_cached_tokens.len(),
                query_lens.len()
            );
        }
        if query_lens.is_empty() || query_lens.contains(&0) {
            anyhow::bail!("reduced prefill metadata requires at least one query token");
        }

        let batch_size = query_lens.len();
        let max_query_len = query_lens.iter().copied().max().unwrap_or(0);
        let slot_mappings_cpu = _make_tensor_with_pad(
            query_lens.iter().map(|len| vec![0i64; *len]).collect(),
            max_query_len,
            _PAD_SLOT_ID,
            &Device::Cpu,
        )?
        .reshape((batch_size, max_query_len))?;

        let context_lens = num_cached_tokens
            .iter()
            .zip(query_lens.iter())
            .map(|(cached, query)| cached + query)
            .collect::<Vec<_>>();
        let context_lens_cpu = Tensor::from_vec(
            context_lens
                .iter()
                .map(|len| *len as u32)
                .collect::<Vec<_>>(),
            (batch_size,),
            &Device::Cpu,
        )?;
        let mut rope_positions = Vec::with_capacity(batch_size * max_query_len);
        for (&cached, &query_len) in num_cached_tokens.iter().zip(query_lens.iter()) {
            for seq_idx in 0..max_query_len {
                let seq_idx = seq_idx.min(query_len - 1);
                rope_positions.push((cached + seq_idx) as u32);
            }
        }
        let rope_positions_cpu =
            Tensor::from_vec(rope_positions, (batch_size * max_query_len,), &Device::Cpu)?;

        let mut cu_q = Vec::with_capacity(batch_size + 1);
        cu_q.push(0u32);
        for &query_len in query_lens {
            cu_q.push(cu_q.last().copied().unwrap_or(0) + query_len as u32);
        }
        let cu_q_cpu = Tensor::from_vec(cu_q, (batch_size + 1,), &Device::Cpu)?;

        let mut cu_kv = Vec::with_capacity(batch_size + 1);
        cu_kv.push(0u32);
        for (&cached, &query_len) in num_cached_tokens.iter().zip(query_lens.iter()) {
            cu_kv.push(cu_kv.last().copied().unwrap_or(0) + (cached + query_len) as u32);
        }
        let cu_kv_cpu = Tensor::from_vec(cu_kv, (batch_size + 1,), &Device::Cpu)?;
        let block_size = self
            .block_size
            .ok_or_else(|| anyhow::anyhow!("missing paged attention block size"))?;
        let paged_decode_context_lens = self
            .paged_context_lens_cpu
            .as_deref()
            .unwrap_or(&context_lens);
        let full_decode_context_lens = self
            .full_paged_context_lens_cpu
            .as_deref()
            .unwrap_or(paged_decode_context_lens);
        let (
            decode_request_indices_cpu,
            decode_kv_tile_indices_cpu,
            decode_o_indptr_cpu,
            decode_kv_chunk_size_cpu,
            decode_block_valid_mask_cpu,
        ) = make_paged_kv_decode_tensors_from_lens(
            paged_decode_context_lens,
            block_size,
            Some(flashinfer_decode_split_pages(
                block_size,
                batch_size,
                self.prefill_key_value_heads,
                paged_decode_context_lens.iter().copied().max().unwrap_or(0),
            )),
        )?;
        let (
            full_decode_request_indices_cpu,
            full_decode_kv_tile_indices_cpu,
            full_decode_o_indptr_cpu,
            full_decode_kv_chunk_size_cpu,
            full_decode_block_valid_mask_cpu,
        ) = make_paged_kv_decode_tensors_from_lens(
            full_decode_context_lens,
            block_size,
            Some(flashinfer_decode_split_pages(
                block_size,
                batch_size,
                self.prefill_key_value_heads,
                full_decode_context_lens.iter().copied().max().unwrap_or(0),
            )),
        )?;

        let mut slot_mappings = HashMap::new();
        let mut context_lens_map = HashMap::new();
        let mut rope_positions = HashMap::new();
        let mut cu_q_map = HashMap::new();
        let mut cu_kv_map = HashMap::new();
        let mut decode_request_indices_map = HashMap::new();
        let mut decode_kv_tile_indices_map = HashMap::new();
        let mut decode_o_indptr_map = HashMap::new();
        let mut decode_kv_chunk_size_map = HashMap::new();
        let mut decode_block_valid_mask_map = HashMap::new();
        let mut full_decode_request_indices_map = HashMap::new();
        let mut full_decode_kv_tile_indices_map = HashMap::new();
        let mut full_decode_o_indptr_map = HashMap::new();
        let mut full_decode_kv_chunk_size_map = HashMap::new();
        let mut full_decode_block_valid_mask_map = HashMap::new();
        for device in devices {
            slot_mappings.insert(device.location(), slot_mappings_cpu.to_device(device)?);
            context_lens_map.insert(device.location(), context_lens_cpu.to_device(device)?);
            rope_positions.insert(device.location(), rope_positions_cpu.to_device(device)?);
            cu_q_map.insert(device.location(), cu_q_cpu.to_device(device)?);
            cu_kv_map.insert(device.location(), cu_kv_cpu.to_device(device)?);
            decode_request_indices_map.insert(
                device.location(),
                decode_request_indices_cpu.to_device(device)?,
            );
            decode_kv_tile_indices_map.insert(
                device.location(),
                decode_kv_tile_indices_cpu.to_device(device)?,
            );
            decode_o_indptr_map.insert(device.location(), decode_o_indptr_cpu.to_device(device)?);
            decode_kv_chunk_size_map.insert(
                device.location(),
                decode_kv_chunk_size_cpu.to_device(device)?,
            );
            decode_block_valid_mask_map.insert(
                device.location(),
                decode_block_valid_mask_cpu.to_device(device)?,
            );
            full_decode_request_indices_map.insert(
                device.location(),
                full_decode_request_indices_cpu.to_device(device)?,
            );
            full_decode_kv_tile_indices_map.insert(
                device.location(),
                full_decode_kv_tile_indices_cpu.to_device(device)?,
            );
            full_decode_o_indptr_map.insert(
                device.location(),
                full_decode_o_indptr_cpu.to_device(device)?,
            );
            full_decode_kv_chunk_size_map.insert(
                device.location(),
                full_decode_kv_chunk_size_cpu.to_device(device)?,
            );
            full_decode_block_valid_mask_map.insert(
                device.location(),
                full_decode_block_valid_mask_cpu.to_device(device)?,
            );
        }
        let full_context_lens = self
            .full_block_tables
            .as_ref()
            .map(|_| context_lens_map.clone());
        let full_max_context_len = self
            .full_block_tables
            .as_ref()
            .and_then(|_| context_lens.iter().copied().max());
        let flashinfer = self.flashinfer.as_ref().map(|flashinfer| {
            let decode_tile_plan = flashinfer_tile_plan(
                decode_request_indices_map.clone(),
                decode_kv_tile_indices_map.clone(),
                decode_o_indptr_map.clone(),
                decode_kv_chunk_size_map.clone(),
                decode_block_valid_mask_map.clone(),
            );
            let full_decode_tile_plan = flashinfer_tile_plan(
                full_decode_request_indices_map.clone(),
                full_decode_kv_tile_indices_map.clone(),
                full_decode_o_indptr_map.clone(),
                full_decode_kv_chunk_size_map.clone(),
                full_decode_block_valid_mask_map.clone(),
            );
            let logical_tile_plan = if flashinfer.views.sliding.is_some() {
                full_decode_tile_plan
            } else {
                decode_tile_plan.clone()
            };
            let logical = FlashInferPagedAttentionView {
                tile_plan: logical_tile_plan,
                ..flashinfer.views.logical.clone()
            };
            let sliding =
                flashinfer
                    .views
                    .sliding
                    .as_ref()
                    .map(|view| FlashInferPagedAttentionView {
                        tile_plan: decode_tile_plan,
                        ..view.clone()
                    });
            FlashInferMetadata {
                views: FlashInferPagedAttentionViews { logical, sliding },
                decode_tmp_v: None,
                decode_tmp_s: None,
                fa3_decode: None,
                #[cfg(feature = "cuda")]
                decode_tile_plan_used: None,
            }
        });

        Ok(PagedAttentionInputMetadata {
            block_tables: self.block_tables.clone(),
            context_lens: Some(context_lens_map),
            block_size: self.block_size,
            paged_context_lens_cpu: Some(paged_decode_context_lens.to_vec()),
            full_paged_context_lens_cpu: Some(full_decode_context_lens.to_vec()),
            slot_mappings,
            max_context_len: context_lens.iter().copied().max(),
            full_block_tables: self.full_block_tables.clone(),
            full_context_lens,
            full_max_context_len,
            is_first_prompt_chunk: false,
            is_final_prompt_chunk: self.is_final_prompt_chunk,
            needs_logits: self.needs_logits,
            prompt_chunk_attention_policy: MultimodalAttentionPolicy::Causal,
            has_noncausal_mm_context: self.has_noncausal_mm_context,
            prefix_gather_workspace_limit: self.prefix_gather_workspace_limit,
            mm_prefix_ranges: self.mm_prefix_ranges.clone(),
            full_mm_prefix_ranges: self.full_mm_prefix_ranges.clone(),
            prefill_attention_heads: self.prefill_attention_heads,
            prefill_key_value_heads: self.prefill_key_value_heads,
            prefill_head_dim: self.prefill_head_dim,
            flashinfer,
            rope_positions: Some(rope_positions),
            num_cached_tokens: Some(num_cached_tokens.to_vec()),
            query_lens: Some(query_lens.to_vec()),
            cu_seqlens_q: Some(cu_q_map),
            cu_seqlens_kv: Some(cu_kv_map),
            decode_rows: None,
        })
    }
}

/// Host-side per-row decode inputs plus everything needed to materialize the paged-attention
/// metadata from them. Kept on the metadata so the CUDA graph layer can rebuild a batch-padded
/// twin through the same code path.
#[derive(Clone, Debug)]
pub struct DecodePagedRows {
    pub slot_mappings: Vec<Vec<i64>>,
    pub(crate) block_tables: BlockTableSnapshot,
    pub context_lens: Vec<usize>,
    pub full_context_lens: Vec<usize>,
    pub query_len: usize,
    pub block_size: usize,
    pub use_standard_metadata: bool,
    pub max_paged_context_len: usize,
    pub sliding_window: Option<usize>,
    pub decode_window: usize,
    pub devices: Vec<Device>,
    pub num_kv_heads: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PagedDecodeMetadataRequirements {
    pub block_tables: bool,
    pub context_lens: bool,
    pub flashinfer_paged_kv: bool,
    pub flashinfer_tile_plan: bool,
}

impl PagedDecodeMetadataRequirements {
    fn conservative(rows: &DecodePagedRows) -> Self {
        Self {
            block_tables: rows.use_standard_metadata || rows.decode_window > 1,
            context_lens: rows.use_standard_metadata,
            flashinfer_paged_kv: true,
            flashinfer_tile_plan: true,
        }
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn graph(
        block_tables: bool,
        context_lens: bool,
        flashinfer_paged_kv: bool,
        flashinfer_tile_plan: bool,
    ) -> Self {
        Self {
            block_tables: block_tables || context_lens,
            context_lens,
            flashinfer_paged_kv: flashinfer_paged_kv || flashinfer_tile_plan,
            flashinfer_tile_plan,
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DecodePagedRowsGraphKey {
    batch_size: usize,
    query_len: usize,
    block_size: usize,
    use_standard_metadata: bool,
    sliding_window: Option<usize>,
    decode_window: usize,
    devices: Vec<DeviceLocation>,
    num_kv_heads: usize,
    paged_block_table_len: usize,
    full_block_table_len: usize,
}

pub(crate) struct DecodeViewHostTensors {
    block_tables: Option<Tensor>,
    context_lens: Option<Tensor>,
    paged_kv: Option<(Tensor, Tensor, Tensor)>,
    tile_plan: Option<(Tensor, Tensor, Tensor, Tensor, Tensor)>,
}

#[derive(Default)]
pub(crate) struct DecodeViewDeviceMaps {
    block_tables: HashMap<DeviceLocation, Tensor>,
    context_lens: HashMap<DeviceLocation, Tensor>,
    paged_kv_indptr: HashMap<DeviceLocation, Tensor>,
    paged_kv_indices: HashMap<DeviceLocation, Tensor>,
    paged_kv_last_page_len: HashMap<DeviceLocation, Tensor>,
    request_indices: HashMap<DeviceLocation, Tensor>,
    kv_tile_indices: HashMap<DeviceLocation, Tensor>,
    o_indptr: HashMap<DeviceLocation, Tensor>,
    kv_chunk_size: HashMap<DeviceLocation, Tensor>,
    block_valid_mask: HashMap<DeviceLocation, Tensor>,
}

impl DecodeViewDeviceMaps {
    fn insert(
        &mut self,
        host: &DecodeViewHostTensors,
        device: &Device,
        stage_on_host: bool,
    ) -> candle_core::Result<()> {
        let location = device.location();
        if let Some(tensor) = host.block_tables.as_ref() {
            self.block_tables.insert(
                location,
                decode_metadata_tensor(tensor, device, stage_on_host)?,
            );
        }
        if let Some(tensor) = host.context_lens.as_ref() {
            self.context_lens.insert(
                location,
                decode_metadata_tensor(tensor, device, stage_on_host)?,
            );
        }
        if let Some((indptr, indices, last_page_len)) = host.paged_kv.as_ref() {
            self.paged_kv_indptr.insert(
                location,
                decode_metadata_tensor(indptr, device, stage_on_host)?,
            );
            self.paged_kv_indices.insert(
                location,
                decode_metadata_tensor(indices, device, stage_on_host)?,
            );
            self.paged_kv_last_page_len.insert(
                location,
                decode_metadata_tensor(last_page_len, device, stage_on_host)?,
            );
        }
        if let Some((request, tile, output, chunk, valid)) = host.tile_plan.as_ref() {
            self.request_indices.insert(
                location,
                decode_metadata_tensor(request, device, stage_on_host)?,
            );
            self.kv_tile_indices.insert(
                location,
                decode_metadata_tensor(tile, device, stage_on_host)?,
            );
            self.o_indptr.insert(
                location,
                decode_metadata_tensor(output, device, stage_on_host)?,
            );
            self.kv_chunk_size.insert(
                location,
                decode_metadata_tensor(chunk, device, stage_on_host)?,
            );
            self.block_valid_mask.insert(
                location,
                decode_metadata_tensor(valid, device, stage_on_host)?,
            );
        }
        Ok(())
    }
}

impl DecodePagedRows {
    pub fn batch_size(&self) -> usize {
        self.slot_mappings.len()
    }

    fn paged_block_tables(&self) -> BlockTableRanges<'_> {
        let ranges = self
            .context_lens
            .iter()
            .zip(&self.full_context_lens)
            .enumerate()
            .map(|(row, (&context_len, &full_context_len))| {
                let table = self.block_tables.row(row);
                if self.sliding_window.is_none() {
                    return 0..table.len();
                }
                let block_start = full_context_len.saturating_sub(context_len) / self.block_size;
                let block_end = block_start
                    .saturating_add(context_len.div_ceil(self.block_size))
                    .min(table.len());
                block_start.min(block_end)..block_end
            })
            .collect();
        BlockTableRanges::new(&self.block_tables, ranges)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn full_block_table(&self, row: usize) -> &[usize] {
        self.block_tables.row(row)
    }

    #[cfg(test)]
    pub(crate) fn materialized_block_tables(&self) -> Vec<Vec<usize>> {
        let tables = self.paged_block_tables();
        (0..tables.len())
            .map(|row| tables.row(row).to_vec())
            .collect()
    }

    /// Pad to `batch_size` rows. Pad rows alias row 0 for every read (same block table and context)
    /// and carry `_PAD_SLOT_ID` slot mappings, so the cache kernels skip their KV writes.
    pub fn padded(&self, batch_size: usize) -> Self {
        let mut rows = self.clone();
        let q = self.query_len;
        while rows.slot_mappings.len() < batch_size {
            rows.slot_mappings
                .push(vec![_PAD_SLOT_ID; self.slot_mappings[0].len()]);
            rows.block_tables
                .push_rows_for_table(self.block_tables.row_table_index(0), q);
            rows.context_lens.extend_from_slice(&self.context_lens[..q]);
            rows.full_context_lens
                .extend_from_slice(&self.full_context_lens[..q]);
        }
        rows
    }

    fn graph_block_table_len(
        &self,
        blocks: usize,
        live_context_len: usize,
        capacity: Option<usize>,
    ) -> usize {
        let minimum_blocks = if crate::perf_flags::cuda_graphs_enabled()
            && !self.use_standard_metadata
            && self.query_len == 1
            && self.decode_window == 1
        {
            capacity
                .unwrap_or(CUDA_GRAPH_DECODE_CONTEXT_FLOOR_TOKENS)
                .min(CUDA_GRAPH_DECODE_CONTEXT_FLOOR_TOKENS)
                .div_ceil(self.block_size)
        } else {
            0
        };
        cuda_graph_block_table_len_with_cap(
            blocks.max(minimum_blocks),
            self.block_size,
            true,
            live_context_len,
            capacity,
        )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn graph_key(&self) -> DecodePagedRowsGraphKey {
        let batch_size = self.batch_size();
        assert!(self
            .slot_mappings
            .iter()
            .all(|slots| slots.len() == self.query_len));
        assert_eq!(self.block_tables.len(), batch_size * self.query_len);
        assert_eq!(self.context_lens.len(), batch_size * self.query_len);
        assert_eq!(self.full_context_lens.len(), batch_size * self.query_len);
        let paged_block_tables = self.paged_block_tables();
        let max_context_len = self.context_lens.iter().copied().max().unwrap_or(0);
        let full_max_context_len = self.full_context_lens.iter().copied().max().unwrap_or(0);
        let graph_capacity = (!self.use_standard_metadata).then_some(self.max_paged_context_len);
        let paged_graph_capacity = self
            .sliding_window
            .map(|window| {
                window
                    .saturating_add(self.block_size.saturating_sub(1))
                    .min(self.max_paged_context_len)
            })
            .or(graph_capacity);
        let paged_block_table_len = self.graph_block_table_len(
            (0..paged_block_tables.len())
                .map(|row| paged_block_tables.row(row).len())
                .max()
                .unwrap_or(1),
            max_context_len,
            paged_graph_capacity,
        );
        let full_block_table_len = self.graph_block_table_len(
            (0..self.block_tables.len())
                .map(|row| self.block_tables.row(row).len())
                .max()
                .unwrap_or(1),
            full_max_context_len,
            graph_capacity,
        );
        DecodePagedRowsGraphKey {
            batch_size,
            query_len: self.query_len,
            block_size: self.block_size,
            use_standard_metadata: self.use_standard_metadata,
            sliding_window: self.sliding_window,
            decode_window: self.decode_window,
            devices: self.devices.iter().map(Device::location).collect(),
            num_kv_heads: self.num_kv_heads,
            paged_block_table_len,
            full_block_table_len,
        }
    }

    pub fn build(self: &Arc<Self>) -> Result<PagedAttentionInputMetadata> {
        let stage_on_host =
            crate::perf_flags::cuda_graphs_enabled() && self.devices.iter().all(Device::is_cuda);
        if stage_on_host {
            self.build_graph_staged()
        } else {
            self.build_inner(false, PagedDecodeMetadataRequirements::conservative(self))
        }
    }

    pub(crate) fn build_materialized(self: &Arc<Self>) -> Result<PagedAttentionInputMetadata> {
        self.build_inner(false, PagedDecodeMetadataRequirements::conservative(self))
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn build_graph_update(
        self: &Arc<Self>,
        requirements: PagedDecodeMetadataRequirements,
    ) -> Result<PagedAttentionInputMetadata> {
        self.build_inner(true, requirements)
    }

    pub(crate) fn build_graph_staged(self: &Arc<Self>) -> Result<PagedAttentionInputMetadata> {
        let max_slot_mapping_len = self.slot_mappings.iter().map(Vec::len).max().unwrap_or(1);
        let slot_mappings = _make_tensor_with_pad(
            self.slot_mappings.clone(),
            max_slot_mapping_len,
            _PAD_SLOT_ID,
            &Device::Cpu,
        )?;
        let slot_mappings = self
            .devices
            .iter()
            .map(|device| (device.location(), slot_mappings.clone()))
            .collect();
        let max_context_len = self.context_lens.iter().copied().max().unwrap_or(0);
        let full_max_context_len = self.full_context_lens.iter().copied().max().unwrap_or(0);
        Ok(PagedAttentionInputMetadata {
            block_tables: None,
            context_lens: None,
            block_size: Some(self.block_size),
            paged_context_lens_cpu: Some(self.context_lens.clone()),
            full_paged_context_lens_cpu: Some(self.full_context_lens.clone()),
            slot_mappings,
            max_context_len: self.use_standard_metadata.then_some(max_context_len),
            full_block_tables: None,
            full_context_lens: None,
            full_max_context_len: self.use_standard_metadata.then_some(full_max_context_len),
            is_first_prompt_chunk: false,
            is_final_prompt_chunk: true,
            needs_logits: true,
            prompt_chunk_attention_policy: MultimodalAttentionPolicy::Causal,
            has_noncausal_mm_context: false,
            prefix_gather_workspace_limit: None,
            mm_prefix_ranges: None,
            full_mm_prefix_ranges: None,
            prefill_attention_heads: 1,
            prefill_key_value_heads: 1,
            prefill_head_dim: 1,
            flashinfer: None,
            rope_positions: None,
            num_cached_tokens: None,
            query_lens: None,
            cu_seqlens_q: None,
            cu_seqlens_kv: None,
            decode_rows: Some(self.clone()),
        })
    }

    fn build_inner(
        self: &Arc<Self>,
        stage_on_host: bool,
        requirements: PagedDecodeMetadataRequirements,
    ) -> Result<PagedAttentionInputMetadata> {
        // Create paged attention tensors on CPU first (see make_prompt_chunk for explanation)
        let max_slot_mapping_len = self.slot_mappings.iter().map(Vec::len).max().unwrap_or(1);
        let slot_mappings = _make_tensor_with_pad(
            self.slot_mappings.clone(),
            max_slot_mapping_len,
            _PAD_SLOT_ID,
            &Device::Cpu,
        )?;

        let block_tables = self.paged_block_tables();
        let paged_attn_context_lens = &self.context_lens;
        let full_block_tables = &self.block_tables;
        let full_paged_attn_context_lens = &self.full_context_lens;
        let block_size = self.block_size;
        let use_standard_metadata = self.use_standard_metadata;
        let max_block_table_len = (0..block_tables.len())
            .map(|row| block_tables.row(row).len())
            .max()
            .expect("block_tables should not be empty when paged attention is enabled");
        let full_max_block_table_len = (0..full_block_tables.len())
            .map(|row| full_block_tables.row(row).len())
            .max()
            .unwrap_or(0)
            .max(1);
        let max_context_len = paged_attn_context_lens.iter().copied().max().unwrap_or(0);
        let full_max_context_len = full_paged_attn_context_lens
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let graph_capacity = (!use_standard_metadata).then_some(self.max_paged_context_len);
        let paged_graph_capacity = self
            .sliding_window
            .map(|window| {
                window
                    .saturating_add(block_size.saturating_sub(1))
                    .min(self.max_paged_context_len)
            })
            .or(graph_capacity);
        let max_block_table_len =
            self.graph_block_table_len(max_block_table_len, max_context_len, paged_graph_capacity);

        let batch_size = block_tables.len();
        let paged_kv = requirements
            .flashinfer_paged_kv
            .then(|| {
                make_paged_kv_tensors(
                    &block_tables,
                    paged_attn_context_lens,
                    block_size,
                    batch_size * max_block_table_len,
                )
            })
            .transpose()?;
        let tile_plan = requirements
            .flashinfer_tile_plan
            .then(|| {
                let decode_split_pages = flashinfer_decode_split_pages(
                    block_size,
                    batch_size,
                    self.num_kv_heads,
                    max_context_len,
                );
                let tiles_per_row = max_block_table_len
                    .max(1)
                    .div_ceil(flashinfer_decode_split_capacity_pages(block_size));
                make_paged_kv_decode_tensors(
                    &block_tables,
                    paged_attn_context_lens,
                    block_size,
                    Some(decode_split_pages),
                    batch_size * tiles_per_row,
                )
            })
            .transpose()?;
        let block_tables_tensor = if requirements.block_tables {
            Some(
                make_block_table_tensor(&block_tables, max_block_table_len)?
                    .reshape(((), max_block_table_len))?,
            )
        } else {
            None
        };
        let context_lens_tensor = requirements
            .context_lens
            .then(|| {
                Tensor::from_vec(
                    paged_attn_context_lens
                        .iter()
                        .map(|x| *x as u32)
                        .collect::<Vec<_>>(),
                    (paged_attn_context_lens.len(),),
                    &Device::Cpu,
                )
            })
            .transpose()?;
        let paged_tensors = DecodeViewHostTensors {
            block_tables: block_tables_tensor,
            context_lens: context_lens_tensor,
            paged_kv,
            tile_plan,
        };
        let full_matches_paged = self.sliding_window.is_none();
        let full_tensors = if full_matches_paged {
            None
        } else {
            let full_max_block_table_len = self.graph_block_table_len(
                full_max_block_table_len,
                full_max_context_len,
                graph_capacity,
            );
            let block_tables_tensor = if requirements.block_tables {
                Some(
                    make_block_table_tensor(full_block_tables, full_max_block_table_len)?
                        .reshape(((), full_max_block_table_len))?,
                )
            } else {
                None
            };
            let context_lens_tensor = requirements
                .context_lens
                .then(|| {
                    Tensor::from_vec(
                        full_paged_attn_context_lens
                            .iter()
                            .map(|x| *x as u32)
                            .collect::<Vec<_>>(),
                        (full_paged_attn_context_lens.len(),),
                        &Device::Cpu,
                    )
                })
                .transpose()?;
            let paged_kv = requirements
                .flashinfer_paged_kv
                .then(|| {
                    make_paged_kv_tensors(
                        full_block_tables,
                        full_paged_attn_context_lens,
                        block_size,
                        full_block_tables.len() * full_max_block_table_len,
                    )
                })
                .transpose()?;
            let tile_plan = requirements
                .flashinfer_tile_plan
                .then(|| {
                    let split_pages = flashinfer_decode_split_pages(
                        block_size,
                        batch_size,
                        self.num_kv_heads,
                        full_max_context_len,
                    );
                    let tiles_per_row = full_max_block_table_len
                        .max(1)
                        .div_ceil(flashinfer_decode_split_capacity_pages(block_size));
                    make_paged_kv_decode_tensors(
                        full_block_tables,
                        full_paged_attn_context_lens,
                        block_size,
                        Some(split_pages),
                        full_block_tables.len() * tiles_per_row,
                    )
                })
                .transpose()?;
            Some(DecodeViewHostTensors {
                block_tables: block_tables_tensor,
                context_lens: context_lens_tensor,
                paged_kv,
                tile_plan,
            })
        };

        let mut slot_mappings_map = HashMap::new();
        let mut paged_maps = DecodeViewDeviceMaps::default();
        let mut full_maps = DecodeViewDeviceMaps::default();
        for device in &self.devices {
            slot_mappings_map.insert(
                device.location(),
                decode_metadata_tensor(&slot_mappings, device, stage_on_host)?,
            );
            paged_maps.insert(&paged_tensors, device, stage_on_host)?;
            if let Some(full_tensors) = full_tensors.as_ref() {
                full_maps.insert(full_tensors, device, stage_on_host)?;
            }
        }
        if full_matches_paged {
            full_maps = DecodeViewDeviceMaps {
                block_tables: paged_maps.block_tables.clone(),
                context_lens: paged_maps.context_lens.clone(),
                paged_kv_indptr: paged_maps.paged_kv_indptr.clone(),
                paged_kv_indices: paged_maps.paged_kv_indices.clone(),
                paged_kv_last_page_len: paged_maps.paged_kv_last_page_len.clone(),
                request_indices: paged_maps.request_indices.clone(),
                kv_tile_indices: paged_maps.kv_tile_indices.clone(),
                o_indptr: paged_maps.o_indptr.clone(),
                kv_chunk_size: paged_maps.kv_chunk_size.clone(),
                block_valid_mask: paged_maps.block_valid_mask.clone(),
            };
        }

        let flashinfer = requirements.flashinfer_paged_kv.then(|| {
            let sliding = (!full_matches_paged).then(|| {
                flashinfer_view(
                    requirements
                        .context_lens
                        .then_some(paged_maps.block_tables.clone()),
                    requirements
                        .context_lens
                        .then_some(paged_maps.context_lens.clone()),
                    requirements.context_lens.then_some(max_context_len),
                    flashinfer_paged_kv(
                        paged_maps.paged_kv_indptr.clone(),
                        paged_maps.paged_kv_indices.clone(),
                        paged_maps.paged_kv_last_page_len.clone(),
                    ),
                    flashinfer_tile_plan(
                        paged_maps.request_indices.clone(),
                        paged_maps.kv_tile_indices.clone(),
                        paged_maps.o_indptr.clone(),
                        paged_maps.kv_chunk_size.clone(),
                        paged_maps.block_valid_mask.clone(),
                    ),
                )
            });
            let logical = flashinfer_view(
                requirements
                    .context_lens
                    .then_some(full_maps.block_tables.clone()),
                requirements
                    .context_lens
                    .then_some(full_maps.context_lens.clone()),
                requirements.context_lens.then_some(full_max_context_len),
                flashinfer_paged_kv(
                    full_maps.paged_kv_indptr.clone(),
                    full_maps.paged_kv_indices.clone(),
                    full_maps.paged_kv_last_page_len.clone(),
                ),
                flashinfer_tile_plan(
                    full_maps.request_indices.clone(),
                    full_maps.kv_tile_indices.clone(),
                    full_maps.o_indptr.clone(),
                    full_maps.kv_chunk_size.clone(),
                    full_maps.block_valid_mask.clone(),
                ),
            );
            flashinfer_metadata(logical, sliding)
        });

        Ok(PagedAttentionInputMetadata {
            slot_mappings: slot_mappings_map,
            block_tables: requirements.block_tables.then_some(paged_maps.block_tables),
            context_lens: requirements.context_lens.then_some(paged_maps.context_lens),
            block_size: Some(block_size),
            paged_context_lens_cpu: Some(paged_attn_context_lens.clone()),
            full_paged_context_lens_cpu: Some(full_paged_attn_context_lens.clone()),
            max_context_len: requirements.context_lens.then_some(max_context_len),
            full_block_tables: requirements.block_tables.then_some(full_maps.block_tables),
            full_context_lens: requirements.context_lens.then_some(full_maps.context_lens),
            full_max_context_len: requirements.context_lens.then_some(full_max_context_len),
            is_first_prompt_chunk: false,
            is_final_prompt_chunk: true,
            needs_logits: true,
            prompt_chunk_attention_policy: MultimodalAttentionPolicy::Causal,
            has_noncausal_mm_context: false,
            prefix_gather_workspace_limit: None,
            mm_prefix_ranges: None,
            full_mm_prefix_ranges: None,
            prefill_attention_heads: 1,
            prefill_key_value_heads: 1,
            prefill_head_dim: 1,
            flashinfer,
            rope_positions: None,
            num_cached_tokens: None,
            query_lens: None,
            cu_seqlens_q: None,
            cu_seqlens_kv: None,
            decode_rows: Some(self.clone()),
        })
    }
}
