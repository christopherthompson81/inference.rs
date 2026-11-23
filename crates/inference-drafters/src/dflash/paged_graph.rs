//! CUDA graphs of a paged-context drafter's pass (Inkling 2's): the rows' ids and positions, their slots, the block
//! tables and lengths live in the graph's buffers, refreshed from the host before each replay. The context append
//! before the pass stays eager, since its row count changes every step.

use std::collections::HashSet;
use std::sync::Arc;

use inference_nn::cuda::graph_capture::{
    CudaGraphComponent, CudaGraphEvent, CudaGraphEventGuard, CudaGraphHandle, CudaGraphHostStaging,
};
use inference_quant::QuantMethod;
use inference_tensor::cuda_backend::cudarc::driver::sys;
use inference_tensor::{DType, Device, Result, Tensor, Var};

use super::{
    DFlashDraftModel, DFlashPagedCache, DFlashProposalBatch, Markov, PagedDraftMeta,
    PagedDraftSlots, PoolDraftMeta, WindowedKvQuery, dflash_rope_from_positions,
};

// the fewest block-table columns a graph keeps; wider tables round up to a power of two, so a growing context
// recaptures a few times, never every block
const MIN_TABLE_WIDTH: usize = 16;
// graphs kept before the least recently used goes
const MAX_ENTRIES: usize = 16;
// a padded batch row's slot: the cache writer skips it
const PAD_SLOT: i64 = -1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PagedGraphKey {
    batch_bucket: usize,
    block: usize,
    width: usize,
}

/// One pass's inputs on the host, padded to the key's batch.
struct PagedGraphRows {
    token_ids: Vec<u32>,
    positions: Vec<u32>,
    anchors: Vec<u32>,
    block_tables: Vec<u32>,
    slots: Vec<i64>,
    seq_lens: Vec<u32>,
    // a mixed drafter's sliding layers: the windowed pool's graph batch for the same rows
    pool: Option<PoolGraphRows>,
}

struct PoolGraphRows {
    block_tables: Vec<u32>,
    width: usize,
    slots: Vec<i64>,
    cumulative_kv_lens: Vec<u32>,
    cumulative_query_lens: Vec<u32>,
    max_query_len: usize,
}

struct PoolGraphBuffers {
    block_tables: Var,
    slot_mapping: Var,
    cumulative_kv_lens: Var,
    cumulative_query_lens: Tensor,
    max_query_len: usize,
}

struct PagedGraphBuffers {
    token_ids: Var,
    positions: Var,
    anchors: Var,
    block_tables: Var,
    slot_mapping: Var,
    seq_lens: Var,
    cumulative_query_lens: Tensor,
    // `[batch, hidden]` each row's recurrent Markov head context
    contexts: Option<Var>,
    pool: Option<PoolGraphBuffers>,
    output_tokens: Var,
}

struct PagedGraphEntry {
    key: PagedGraphKey,
    staging: CudaGraphHostStaging,
    buffers: PagedGraphBuffers,
    token_embedding: Arc<dyn QuantMethod>,
    lm_head: Arc<dyn QuantMethod>,
    graph: CudaGraphHandle,
}

#[derive(Default)]
pub(super) struct PagedGraphState {
    entries: Vec<PagedGraphEntry>,
    warmed: HashSet<PagedGraphKey>,
    failed: HashSet<PagedGraphKey>,
}

/// A greedy pass for `seq_ids`, whose anchors sit at `start_positions`.
pub(crate) struct PagedGraphRun<'a> {
    pub(crate) seq_ids: &'a [usize],
    pub(crate) anchors: &'a [u32],
    pub(crate) start_positions: &'a [usize],
    pub(crate) n_predict: usize,
    // the target's token embedding as the anchor row takes it (with any norm it applies)
    pub(crate) embed_tokens: &'a dyn Fn(&Tensor) -> Result<Tensor>,
    pub(crate) token_embedding: &'a Arc<dyn QuantMethod>,
    pub(crate) lm_head: &'a Arc<dyn QuantMethod>,
    pub(crate) paged: DFlashPagedCache<'a>,
}

impl PagedGraphBuffers {
    fn new(
        key: PagedGraphKey,
        rows: &PagedGraphRows,
        model: &DFlashDraftModel,
        contexts: Option<&Tensor>,
    ) -> Result<Self> {
        let device = &model.device;
        let u32s = |values: &[u32], shape: (usize, usize)| -> Result<Var> {
            Var::from_tensor(&Tensor::from_vec(values.to_vec(), shape, device)?)
        };
        let b = key.batch_bucket;
        let block = u32::try_from(key.block).map_err(inference_tensor::Error::wrap)?;
        let cumulative_query_lens = Tensor::from_vec(
            (0..=b)
                .map(|row| u32::try_from(row).map(|row| row * block))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(inference_tensor::Error::wrap)?,
            (b + 1,),
            device,
        )?;
        Ok(Self {
            token_ids: u32s(&rows.token_ids, (b, key.block))?,
            positions: u32s(&rows.positions, (1, b * key.block))?,
            anchors: u32s(&rows.anchors, (1, b))?,
            block_tables: u32s(&rows.block_tables, (b, key.width))?,
            slot_mapping: Var::from_tensor(&Tensor::from_vec(
                rows.slots.clone(),
                (b * key.block,),
                device,
            )?)?,
            seq_lens: u32s(&rows.seq_lens, (1, b))?,
            cumulative_query_lens,
            contexts: contexts.map(Var::from_tensor).transpose()?,
            pool: rows
                .pool
                .as_ref()
                .map(|pool| -> Result<PoolGraphBuffers> {
                    Ok(PoolGraphBuffers {
                        block_tables: u32s(&pool.block_tables, (b, pool.width))?,
                        slot_mapping: Var::from_tensor(&Tensor::from_vec(
                            pool.slots.clone(),
                            (pool.slots.len(),),
                            device,
                        )?)?,
                        cumulative_kv_lens: u32s(&pool.cumulative_kv_lens, (1, b + 1))?,
                        cumulative_query_lens: Tensor::from_vec(
                            pool.cumulative_query_lens.clone(),
                            (b + 1,),
                            device,
                        )?,
                        max_query_len: pool.max_query_len,
                    })
                })
                .transpose()?,
            output_tokens: Var::zeros((b, key.block - model.draft_row_offset), DType::U32, device)?,
        })
    }

    fn update(
        &self,
        rows: &PagedGraphRows,
        contexts: Option<&Tensor>,
        staging: &mut CudaGraphHostStaging,
    ) -> Result<()> {
        let location = self.token_ids.device().location();
        staging.update(|staging| {
            staging.copy_from_u32_slice(
                "paged_draft_token_ids",
                location,
                &rows.token_ids,
                &self.token_ids,
            )?;
            staging.copy_from_u32_slice(
                "paged_draft_positions",
                location,
                &rows.positions,
                &self.positions,
            )?;
            staging.copy_from_u32_slice(
                "paged_draft_anchors",
                location,
                &rows.anchors,
                &self.anchors,
            )?;
            staging.copy_from_u32_slice(
                "paged_draft_block_tables",
                location,
                &rows.block_tables,
                &self.block_tables,
            )?;
            staging.copy_from_i64_slice(
                "paged_draft_slots",
                location,
                &rows.slots,
                &self.slot_mapping,
            )?;
            staging.copy_from_u32_slice(
                "paged_draft_seq_lens",
                location,
                &rows.seq_lens,
                &self.seq_lens,
            )?;
            if let (Some(src), Some(dst)) = (&rows.pool, &self.pool) {
                staging.copy_from_u32_slice(
                    "paged_draft_pool_tables",
                    location,
                    &src.block_tables,
                    &dst.block_tables,
                )?;
                staging.copy_from_i64_slice(
                    "paged_draft_pool_slots",
                    location,
                    &src.slots,
                    &dst.slot_mapping,
                )?;
                staging.copy_from_u32_slice(
                    "paged_draft_pool_kv_lens",
                    location,
                    &src.cumulative_kv_lens,
                    &dst.cumulative_kv_lens,
                )?;
            }
            Ok(())
        })?;
        if let (Some(src), Some(dst)) = (contexts, &self.contexts) {
            inference_nn::cuda::graph::copy_tensor(src, &dst.as_detached_tensor())?;
        }
        Ok(())
    }
}

impl DFlashDraftModel {
    /// The greedy pass of a paged-context drafter through a CUDA graph; `None` where it doesn't apply (another
    /// kind of drafter, graphs off, a batch past the buckets, a shape that failed to capture).
    pub(crate) fn paged_proposals_cuda_graph(
        &self,
        run: &PagedGraphRun<'_>,
    ) -> Result<Option<DFlashProposalBatch>> {
        if !inference_nn::cuda::graph_capture::cuda_decode_graphs_enabled()
            || !self.uses_paged_kv()
            || self.selector.is_some()
            || self.dspine.is_some()
            || run.seq_ids.is_empty()
            || run.n_predict == 0
        {
            return Ok(None);
        }
        let block = self.query_rows(run.n_predict);
        let real_batch = run.seq_ids.len();
        let Some(batch_bucket) = inference_nn::cuda::graph_capture::cuda_graph_batch_bucket(
            CudaGraphComponent::DFlash,
            block,
            real_batch,
        ) else {
            return Ok(None);
        };
        let page =
            inference_nn::get_mut_arcmutex!(run.paged.metadata.kv_cache_manager).block_size();
        let needed = run
            .start_positions
            .iter()
            .map(|start| (start + block).div_ceil(page))
            .max()
            .unwrap_or(1);
        let key = PagedGraphKey {
            batch_bucket,
            block,
            width: needed.next_power_of_two().max(MIN_TABLE_WIDTH),
        };
        let mut state = self
            .paged_graphs
            .lock()
            .expect("paged drafter graphs poisoned");
        if state.failed.contains(&key) {
            return Ok(None);
        }
        let rows = self.paged_graph_rows(key, run)?;
        let contexts = self.paged_graph_contexts(key, run.seq_ids)?;
        if let Some(position) = state.entries.iter().position(|entry| {
            entry.key == key
                && Arc::ptr_eq(&entry.token_embedding, run.token_embedding)
                && Arc::ptr_eq(&entry.lm_head, run.lm_head)
        }) {
            let mut entry = state.entries.remove(position);
            entry
                .buffers
                .update(&rows, contexts.as_ref(), &mut entry.staging)?;
            entry.staging.order_before_graph()?;
            let tokens = entry.launch(real_batch)?;
            state.entries.push(entry);
            return Ok(Some(tokens));
        }
        let buffers = PagedGraphBuffers::new(key, &rows, self, contexts.as_ref())?;
        // a shape's first pass runs eagerly, so whatever it builds once (the Markov fold) exists before a capture
        if state.warmed.insert(key) {
            let Device::Cuda(cuda_device) = &self.device else {
                return Ok(None);
            };
            // records the small host vectors the pass uploads, which the capture then reads from the cache
            let _htod_cache_guard = cuda_device.enable_cuda_graph_htod_cache();
            let tokens = self.paged_graph_pass(key, &buffers, run)?;
            return Ok(Some(DFlashProposalBatch::DeviceTokens(
                tokens.narrow(0, 0, real_batch)?.copy()?,
            )));
        }
        match self.capture_paged_graph(key, buffers, run) {
            Ok(mut entry) => {
                let tokens = entry.launch(real_batch)?;
                if state.entries.len() >= MAX_ENTRIES {
                    let evicted = state.entries.remove(0);
                    release(evicted);
                }
                state.entries.push(entry);
                Ok(Some(tokens))
            }
            Err(err) => {
                tracing::warn!(
                    ?key,
                    "paged drafter CUDA graph capture disabled for this shape: {err:?}"
                );
                state.failed.insert(key);
                Ok(None)
            }
        }
    }

    fn paged_graph_rows(
        &self,
        key: PagedGraphKey,
        run: &PagedGraphRun<'_>,
    ) -> Result<PagedGraphRows> {
        let PagedDraftSlots {
            tables,
            width,
            slots,
            kv_lens,
        } = self.paged_draft_slots(
            run.seq_ids,
            run.start_positions,
            key.block,
            key.width,
            run.paged,
        )?;
        debug_assert_eq!(width, key.width);
        let pad = key.batch_bucket - run.seq_ids.len();
        let mut token_ids = Vec::with_capacity(key.batch_bucket * key.block);
        let mut positions = Vec::with_capacity(key.batch_bucket * key.block);
        let mut anchors = run.anchors.to_vec();
        anchors.resize(key.batch_bucket, 0);
        for (row, anchor) in anchors.iter().enumerate() {
            token_ids.push(*anchor);
            token_ids.extend(std::iter::repeat_n(self.mask_token_id, key.block - 1));
            let start = run.start_positions.get(row).copied().unwrap_or(0);
            for offset in 0..key.block {
                positions
                    .push(u32::try_from(start + offset).map_err(inference_tensor::Error::wrap)?);
            }
        }
        let mut block_tables = tables;
        block_tables.resize(key.batch_bucket * key.width, 0);
        let mut slots = slots;
        slots.resize(key.batch_bucket * key.block, PAD_SLOT);
        let mut seq_lens = kv_lens;
        seq_lens.extend(std::iter::repeat_n(1, pad));
        let pool = match &self.windowed_pool {
            Some(pool) => {
                let queries = run
                    .seq_ids
                    .iter()
                    .map(|seq_id| WindowedKvQuery {
                        seq_id: *seq_id,
                        query_len: key.block,
                    })
                    .collect::<Vec<_>>();
                let batch = pool
                    .lock()
                    .expect("dflash windowed pool poisoned")
                    .scratch_graph_batch(&queries, key.batch_bucket)?;
                Some(PoolGraphRows {
                    block_tables: batch.block_tables_for_graph().to_vec(),
                    width: batch.block_table_width_for_graph(),
                    slots: batch.slot_mapping_for_graph().to_vec(),
                    cumulative_kv_lens: batch.cumulative_kv_lens_for_graph().to_vec(),
                    cumulative_query_lens: batch.cumulative_query_lens_for_graph().to_vec(),
                    max_query_len: batch.max_query_len(),
                })
            }
            None => None,
        };
        Ok(PagedGraphRows {
            token_ids,
            positions,
            anchors,
            block_tables,
            slots,
            seq_lens,
            pool,
        })
    }

    /// Each row's recurrent Markov head context `[bucket, hidden]`, padded rows zero.
    fn paged_graph_contexts(
        &self,
        key: PagedGraphKey,
        seq_ids: &[usize],
    ) -> Result<Option<Tensor>> {
        if !matches!(self.markov, Some(Markov::Recurrent(_))) {
            return Ok(None);
        }
        let contexts = self
            .markov_contexts
            .lock()
            .expect("dflash markov contexts poisoned");
        let rows = seq_ids
            .iter()
            .map(|seq_id| {
                contexts.get(seq_id).cloned().ok_or_else(|| {
                    inference_tensor::Error::msg(
                        "a recurrent Markov draft for a sequence with no committed context",
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let stacked = Tensor::stack(&rows, 0)?;
        let pad = key.batch_bucket - seq_ids.len();
        if pad == 0 {
            return Ok(Some(stacked));
        }
        let padding = Tensor::zeros((pad, stacked.dim(1)?), stacked.dtype(), stacked.device())?;
        Ok(Some(Tensor::cat(&[&stacked, &padding], 0)?))
    }

    /// The pass over a graph's buffers: the drafts `[bucket, drafts]`.
    fn paged_graph_pass(
        &self,
        key: PagedGraphKey,
        buffers: &PagedGraphBuffers,
        run: &PagedGraphRun<'_>,
    ) -> Result<Tensor> {
        let ids = buffers.token_ids.as_detached_tensor();
        let mut noise = self.with_own_mask_rows((run.embed_tokens)(&ids)?)?;
        if (self.input_embedding_scale - 1.0).abs() > f64::EPSILON {
            noise = (noise * self.input_embedding_scale)?;
        }
        let (q_cos, q_sin) = dflash_rope_from_positions(
            &buffers.positions.as_detached_tensor(),
            &self.inv_freq,
            self.dtype,
            self.rope_attention_factor,
        )?;
        let half = self.head_dim / 2;
        let q_cos = q_cos.reshape((key.batch_bucket, key.block, half))?;
        let q_sin = q_sin.reshape((key.batch_bucket, key.block, half))?;
        let anchors = buffers.anchors.as_detached_tensor().squeeze(0)?;
        let block_tables = buffers.block_tables.as_detached_tensor();
        let slot_mapping = buffers.slot_mapping.as_detached_tensor();
        let seq_lens = buffers.seq_lens.as_detached_tensor().squeeze(0)?;
        let pool = buffers
            .pool
            .as_ref()
            .map(|pool| -> Result<_> {
                let cumulative = pool.cumulative_kv_lens.as_detached_tensor().squeeze(0)?;
                let b = key.batch_bucket;
                let lens = (cumulative.narrow(0, 1, b)? - cumulative.narrow(0, 0, b)?)?;
                Ok((
                    pool.block_tables.as_detached_tensor(),
                    pool.slot_mapping.as_detached_tensor(),
                    lens,
                ))
            })
            .transpose()?;
        let mut drafted = self.draft_hidden_paged_tensors(
            &noise,
            Some(&anchors),
            (&q_cos, &q_sin),
            &PagedDraftMeta {
                block_tables: &block_tables,
                slot_mapping: &slot_mapping,
                seq_lens: &seq_lens,
                cumulative_query_lens: &buffers.cumulative_query_lens,
                pool: pool.as_ref().zip(buffers.pool.as_ref()).map(
                    |((tables, slots, lens), buffers)| PoolDraftMeta {
                        block_tables: tables,
                        slot_mapping: slots,
                        seq_lens: lens,
                        cumulative_query_lens: &buffers.cumulative_query_lens,
                        max_query_len: buffers.max_query_len,
                    },
                ),
            },
            run.paged,
        )?;
        drafted.markov_context = buffers.contexts.as_ref().map(Var::as_detached_tensor);
        let mut logits = run.lm_head.forward(&drafted.rows)?;
        if (self.output_multiplier - 1.0).abs() > f64::EPSILON {
            logits = (logits * self.output_multiplier)?;
        }
        self.pick_greedy(&drafted, &logits, &anchors, &**run.lm_head)?
            .contiguous()
    }

    fn capture_paged_graph(
        &self,
        key: PagedGraphKey,
        buffers: PagedGraphBuffers,
        run: &PagedGraphRun<'_>,
    ) -> Result<PagedGraphEntry> {
        let graph_event =
            CudaGraphEventGuard::new(CudaGraphComponent::DFlash, CudaGraphEvent::Capture);
        self.device.synchronize()?;
        let Device::Cuda(cuda_device) = &self.device else {
            inference_tensor::bail!("a paged drafter CUDA graph expected a CUDA device");
        };
        let stream = cuda_device.cuda_stream();
        let _memory_pool_guard =
            inference_nn::cuda::graph_capture::prepare_cuda_graph_memory_pool(&stream)?;
        let restore =
            inference_nn::cuda::graph_capture::disable_event_tracking_for_capture(&stream);
        let _htod_cache_guard = cuda_device.enable_cuda_graph_htod_cache();
        let _capture = inference_tensor::cuda_backend::capture_scope();
        let finish = |result: Result<Option<CudaGraphHandle>>| {
            inference_nn::cuda::graph_capture::restore_event_tracking_after_capture(
                &stream, restore,
            );
            result
        };
        if let Err(err) =
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        {
            finish(Ok(None))?;
            return Err(inference_tensor::Error::msg(err.to_string())
                .context("paged drafter graph begin capture"));
        }
        let pass = self
            .paged_graph_pass(key, &buffers, run)
            .and_then(|tokens| {
                inference_nn::cuda::graph::copy_tensor(
                    &tokens,
                    &buffers.output_tokens.as_detached_tensor(),
                )
            });
        if let Err(err) = pass {
            inference_nn::cuda::graph_capture::end_cuda_capture_discard(&stream);
            finish(Ok(None))?;
            return Err(err);
        }
        let graph = finish(CudaGraphHandle::end_capture(&stream))?.ok_or_else(|| {
            inference_tensor::Error::msg("paged drafter graph capture returned no graph")
        })?;
        graph.upload()?;
        let entry = PagedGraphEntry {
            key,
            staging: CudaGraphHostStaging::new(stream)?,
            buffers,
            token_embedding: run.token_embedding.clone(),
            lm_head: run.lm_head.clone(),
            graph,
        };
        graph_event.success();
        Ok(entry)
    }

    pub(crate) fn evict_paged_graphs(&self, max_entries: usize) -> usize {
        let mut state = self
            .paged_graphs
            .lock()
            .expect("paged drafter graphs poisoned");
        let evict = state.entries.len().min(max_entries);
        for entry in state.entries.drain(..evict) {
            release(entry);
        }
        evict
    }
}

impl PagedGraphEntry {
    fn launch(&mut self, real_batch: usize) -> Result<DFlashProposalBatch> {
        let graph_event =
            CudaGraphEventGuard::new(CudaGraphComponent::DFlash, CudaGraphEvent::Replay);
        self.graph.launch()?;
        self.staging.record_graph_complete()?;
        let tokens = self
            .buffers
            .output_tokens
            .as_detached_tensor()
            .narrow(0, 0, real_batch)?
            .copy()?;
        graph_event.success();
        Ok(DFlashProposalBatch::DeviceTokens(tokens))
    }
}

fn release(entry: PagedGraphEntry) {
    let PagedGraphEntry {
        key: _,
        staging,
        buffers,
        token_embedding,
        lm_head,
        graph,
    } = entry;
    let (_, result) = inference_nn::cuda::graph_capture::release_cuda_graph_resources(
        CudaGraphComponent::DFlash,
        graph,
        (staging, buffers),
    );
    if let Err(err) = result {
        tracing::warn!("Failed to release a paged drafter CUDA graph: {err:?}");
    }
    drop((token_embedding, lm_head));
}

impl Drop for PagedGraphState {
    fn drop(&mut self) {
        for entry in std::mem::take(&mut self.entries) {
            release(entry);
        }
    }
}
