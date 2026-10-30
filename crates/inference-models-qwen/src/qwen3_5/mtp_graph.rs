//! The MTP head's drafter forward replayed from a CUDA graph per row shape: eager, its ~100 launches cost more host
//! time than its kernels take (qwen4exp investigation Run 57).

use std::sync::{Arc, Mutex};

use inference_tensor::cuda_backend::cudarc::driver::sys;
use inference_tensor::{Device, Result, Tensor, TensorId, Var};

use crate::cuda::graph_capture::{
    CudaGraphComponent, CudaGraphDispatchMode, CudaGraphDispatchReason, CudaGraphEvent,
    CudaGraphEventGuard, CudaGraphEvictionReason, CudaGraphHandle, CudaGraphHostStaging,
    cuda_decode_graphs_enabled, disable_event_tracking_for_capture, end_cuda_capture_discard,
    prepare_cuda_graph_memory_pool, record_cuda_graph_dispatch, record_cuda_graph_evictions,
    record_cuda_graph_resident_entries, release_cuda_graph_resources,
    restore_event_tracking_after_capture, take_cuda_graph_capacity_eviction,
    trim_cuda_graph_memory,
};
use crate::cuda::graph_metadata::{CudaDecodeGraphMetadataBuffers, CudaDecodeGraphMetadataInput};
use crate::paged_attention::PagedAttentionInputMetadata;
use crate::paged_attention::input_metadata::{DecodePagedRows, DecodePagedRowsGraphKey};

use super::mtp::MtpDraft;
use super::speculative::MtpForwardInputs;

// a refresh covers the anchor, the accepted drafts and a pending prompt tail
const MTP_GRAPH_MAX_ROWS: usize = 8;
const MTP_GRAPH_CAPACITY: usize = 8;

/// What a graph forward stages: the rows' host values, the target hidden states and their decode rows.
pub(super) struct MtpGraphRun<'a> {
    pub(super) shape: (usize, usize),
    pub(super) tokens: &'a [u32],
    pub(super) mrope: &'a [u32],
    pub(super) positions: &'a [usize],
    pub(super) starts: &'a [usize],
    pub(super) target_hidden: &'a Tensor,
    pub(super) decode_rows: Arc<DecodePagedRows>,
    // the paged layers the forward writes, so a reallocated cache gets a new graph
    pub(super) kv: Vec<TensorId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MtpGraphKey {
    shape: (usize, usize),
    decode_rows: DecodePagedRowsGraphKey,
    kv: Vec<TensorId>,
}

struct MtpGraphEntry {
    key: MtpGraphKey,
    staging: CudaGraphHostStaging,
    tokens: Var,
    positions: Var,
    hidden: Var,
    metadata: CudaDecodeGraphMetadataBuffers,
    output: MtpDraft,
    // the captured forward's view of the buffers, which the graph reads
    _graph_metadata: PagedAttentionInputMetadata,
    graph: CudaGraphHandle,
}

/// A graph's inputs at fixed addresses, holding one run's values: what a capture records and its warm-up reads.
struct GraphInputs {
    tokens: Var,
    positions: Var,
    hidden: Var,
    metadata: CudaDecodeGraphMetadataBuffers,
    graph_metadata: PagedAttentionInputMetadata,
}

impl GraphInputs {
    fn new(run: &MtpGraphRun<'_>) -> Result<Self> {
        let device = run.target_hidden.device();
        let (rows, cols) = run.shape;
        let tokens = Var::from_tensor(&Tensor::from_vec(run.tokens.to_vec(), run.shape, device)?)?;
        let positions = Var::from_tensor(&Tensor::from_vec(
            run.mrope.to_vec(),
            (run.mrope.len() / (rows * cols), rows, cols),
            device,
        )?)?;
        let hidden = Var::from_tensor(&run.target_hidden.copy()?)?;
        let materialized = run
            .decode_rows
            .build_materialized()
            .map_err(inference_tensor::Error::msg)?;
        let offsets = run.positions.iter().map(|p| p - 1).collect::<Vec<_>>();
        let (metadata, mut graph_metadata) =
            CudaDecodeGraphMetadataBuffers::new(CudaDecodeGraphMetadataInput {
                metadata: &materialized,
                seqlen_offsets: &offsets,
                position_ids: run.positions,
                seq_len: 1,
                kv_cache: &[],
                model_metadata: None,
                activation_dtype: run.target_hidden.dtype(),
            })?;
        // the head ropes its own MRoPE positions, as the eager forward's metadata leaves these out
        graph_metadata.rope_positions = None;
        Ok(Self {
            tokens,
            positions,
            hidden,
            metadata,
            graph_metadata,
        })
    }

    fn forward(
        &self,
        run: &MtpGraphRun<'_>,
        forward: &dyn Fn(MtpForwardInputs<'_>) -> Result<MtpDraft>,
    ) -> Result<MtpDraft> {
        forward(MtpForwardInputs {
            tokens: &self.tokens.as_detached_tensor(),
            positions: &self.positions.as_detached_tensor(),
            target_hidden: &self.hidden.as_detached_tensor(),
            metadata: &self.graph_metadata,
            starts: run.starts,
        })
    }
}

// A shape's first run: eager over the graph inputs, so the parameter uploads a capture replays are cached
fn warm(
    run: &MtpGraphRun<'_>,
    forward: &dyn Fn(MtpForwardInputs<'_>) -> Result<MtpDraft>,
) -> Result<MtpDraft> {
    let Device::Cuda(cuda) = run.target_hidden.device() else {
        inference_tensor::bail!("an MTP graph needs a CUDA device");
    };
    let inputs = GraphInputs::new(run)?;
    let _htod_cache = cuda.enable_cuda_graph_htod_cache();
    inputs.forward(run, forward)
}

impl MtpGraphEntry {
    fn capture(
        key: MtpGraphKey,
        run: &MtpGraphRun<'_>,
        forward: &dyn Fn(MtpForwardInputs<'_>) -> Result<MtpDraft>,
    ) -> Result<Self> {
        let device = run.target_hidden.device();
        let Device::Cuda(cuda) = device else {
            inference_tensor::bail!("an MTP graph needs a CUDA device");
        };
        let GraphInputs {
            tokens,
            positions,
            hidden,
            mut metadata,
            graph_metadata,
        } = GraphInputs::new(run)?;
        device.synchronize()?;
        let stream = cuda.cuda_stream();
        let _memory_pool = prepare_cuda_graph_memory_pool(&stream)?;
        let restore = disable_event_tracking_for_capture(&stream);
        let _htod_cache = cuda.enable_cuda_graph_htod_cache();
        let _capture = inference_tensor::cuda_backend::capture_scope();
        if let Err(err) =
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        {
            restore_event_tracking_after_capture(&stream, restore);
            return Err(inference_tensor::Error::msg(err.to_string())
                .context("MTP graph begin capture failed"));
        }
        let output = forward(MtpForwardInputs {
            tokens: &tokens.as_detached_tensor(),
            positions: &positions.as_detached_tensor(),
            target_hidden: &hidden.as_detached_tensor(),
            metadata: &graph_metadata,
            starts: run.starts,
        });
        let output = match output {
            Ok(output) => output,
            Err(err) => {
                end_cuda_capture_discard(&stream);
                restore_event_tracking_after_capture(&stream, restore);
                return Err(inference_tensor::Error::msg(format!(
                    "MTP graph forward capture failed: {err}"
                )));
            }
        };
        metadata.finish_capture(&graph_metadata);
        let graph = CudaGraphHandle::end_capture(&stream);
        restore_event_tracking_after_capture(&stream, restore);
        let Some(graph) = graph? else {
            inference_tensor::bail!("MTP graph capture returned no graph");
        };
        graph.upload()?;
        Ok(Self {
            key,
            staging: CudaGraphHostStaging::new(stream)?,
            tokens,
            positions,
            hidden,
            metadata,
            output,
            _graph_metadata: graph_metadata,
            graph,
        })
    }

    // the captured outputs, valid until the next replay
    fn launch(&mut self) -> Result<MtpDraft> {
        self.staging.order_before_graph()?;
        self.graph.launch()?;
        self.staging.record_graph_complete()?;
        Ok(MtpDraft {
            logits_hidden: self.output.logits_hidden.clone(),
            chain_hidden: self.output.chain_hidden.clone(),
        })
    }

    fn replay(&mut self, run: &MtpGraphRun<'_>) -> Result<MtpDraft> {
        let event = CudaGraphEventGuard::new(CudaGraphComponent::Mtp, CudaGraphEvent::Replay);
        let metadata = run
            .decode_rows
            .build()
            .map_err(inference_tensor::Error::msg)?;
        let location = run.target_hidden.device().location();
        let Self {
            staging,
            tokens,
            positions,
            metadata: buffers,
            ..
        } = self;
        staging.update(|staging| {
            staging.copy_from_u32_slice("mtp_tokens", location, run.tokens, tokens)?;
            staging.copy_from_u32_slice("mtp_positions", location, run.mrope, positions)?;
            buffers.copy_from(&metadata, run.positions, 1, staging)
        })?;
        crate::cuda::graph::copy_tensor(run.target_hidden, &self.hidden.as_detached_tensor())?;
        let draft = self.launch()?;
        event.success();
        record_cuda_graph_dispatch(
            CudaGraphComponent::Mtp,
            CudaGraphDispatchMode::Replay,
            CudaGraphDispatchReason::CacheHit,
        );
        Ok(draft)
    }

    fn release(
        self,
    ) -> (
        Arc<inference_tensor::cuda_backend::cudarc::driver::CudaStream>,
        Result<()>,
    ) {
        let Self {
            graph,
            staging,
            tokens,
            positions,
            hidden,
            metadata,
            output,
            _graph_metadata,
            ..
        } = self;
        release_cuda_graph_resources(
            CudaGraphComponent::Mtp,
            graph,
            (
                staging,
                tokens,
                positions,
                hidden,
                metadata,
                output,
                _graph_metadata,
            ),
        )
    }
}

fn release(entries: Vec<MtpGraphEntry>) {
    let mut streams: Vec<Arc<inference_tensor::cuda_backend::cudarc::driver::CudaStream>> =
        Vec::new();
    for entry in entries {
        let (stream, result) = entry.release();
        if let Err(err) = result {
            tracing::warn!("Failed to release an MTP CUDA graph: {err:?}");
        }
        if !streams
            .iter()
            .any(|known| known.context().cu_device() == stream.context().cu_device())
        {
            streams.push(stream);
        }
    }
    for stream in streams {
        if let Err(err) = trim_cuda_graph_memory(&stream) {
            tracing::warn!("Failed to trim released MTP CUDA graph memory: {err:?}");
        }
    }
}

#[derive(Default)]
struct MtpGraphState {
    entries: Vec<MtpGraphEntry>,
    // shapes seen once, run eagerly first so lazily loaded kernels and workspaces exist before a capture
    warmed: Vec<MtpGraphKey>,
    failed: Vec<MtpGraphKey>,
}

/// The MTP head's captured drafter forwards, most recently used last.
#[derive(Default)]
pub(super) struct MtpGraphs(Mutex<MtpGraphState>);

impl MtpGraphs {
    /// The drafter forward over `run` from its shape's graph, capturing it on the shape's second use; `None` runs
    /// it eagerly.
    pub(super) fn forward(
        &self,
        run: &MtpGraphRun<'_>,
        forward: &dyn Fn(MtpForwardInputs<'_>) -> Result<MtpDraft>,
    ) -> Result<Option<MtpDraft>> {
        if !cuda_decode_graphs_enabled()
            || !run.target_hidden.device().is_cuda()
            || run.tokens.len() > MTP_GRAPH_MAX_ROWS
        {
            return Ok(None);
        }
        let key = MtpGraphKey {
            shape: run.shape,
            decode_rows: run.decode_rows.graph_key(),
            kv: run.kv.clone(),
        };
        let mut state = self.0.lock().expect("MTP graphs poisoned");
        if state.failed.contains(&key) {
            return Ok(None);
        }
        if let Some(position) = state.entries.iter().position(|entry| entry.key == key) {
            let mut entry = state.entries.remove(position);
            return match entry.replay(run) {
                Ok(draft) => {
                    state.entries.push(entry);
                    Ok(Some(draft))
                }
                Err(err) => {
                    tracing::warn!(
                        "MTP CUDA graph replay failed; this shape runs eagerly: {err:?}"
                    );
                    state.failed.push(key);
                    record_cuda_graph_resident_entries(
                        CudaGraphComponent::Mtp,
                        state.entries.len(),
                    );
                    release(vec![entry]);
                    Ok(None)
                }
            };
        }
        if !state.warmed.contains(&key) {
            state.warmed.push(key);
            return warm(run, forward).map(Some);
        }
        let event = CudaGraphEventGuard::new(CudaGraphComponent::Mtp, CudaGraphEvent::Capture);
        let mut entry = match MtpGraphEntry::capture(key.clone(), run, forward) {
            Ok(entry) => entry,
            Err(err) => {
                tracing::warn!("MTP CUDA graph capture failed; this shape runs eagerly: {err:?}");
                state.failed.push(key);
                return Ok(None);
            }
        };
        event.success();
        let draft = entry.launch()?;
        if let Some(evicted) =
            take_cuda_graph_capacity_eviction(&mut state.entries, MTP_GRAPH_CAPACITY)
        {
            record_cuda_graph_evictions(
                CudaGraphComponent::Mtp,
                CudaGraphEvictionReason::Capacity,
                1,
            );
            release(vec![evicted]);
        }
        state.warmed.retain(|warmed| *warmed != key);
        state.entries.push(entry);
        record_cuda_graph_resident_entries(CudaGraphComponent::Mtp, state.entries.len());
        Ok(Some(draft))
    }

    /// Drops up to `max_entries` least recently used graphs, returning how many went.
    pub(super) fn evict_lru(&self, max_entries: usize) -> usize {
        let mut state = self.0.lock().expect("MTP graphs poisoned");
        let count = max_entries.min(state.entries.len());
        let evicted = state.entries.drain(..count).collect::<Vec<_>>();
        record_cuda_graph_resident_entries(CudaGraphComponent::Mtp, state.entries.len());
        if count > 0 {
            record_cuda_graph_evictions(
                CudaGraphComponent::Mtp,
                CudaGraphEvictionReason::MemoryPressure,
                count,
            );
            release(evicted);
        }
        count
    }
}

impl Drop for MtpGraphs {
    fn drop(&mut self) {
        let entries = std::mem::take(&mut self.0.get_mut().expect("MTP graphs poisoned").entries);
        record_cuda_graph_resident_entries(CudaGraphComponent::Mtp, 0);
        release(entries);
    }
}

#[cfg(test)]
mod tests {
    use inference_tensor::DType;

    use super::*;
    use crate::paged_attention::block_table_rows::BlockTableSnapshot;

    const HIDDEN: usize = 4;
    const BLOCK_SIZE: usize = 32;

    fn decode_rows(device: &Device, context_lens: &[usize]) -> Arc<DecodePagedRows> {
        Arc::new(DecodePagedRows {
            slot_mappings: context_lens
                .iter()
                .map(|&len| vec![len as i64 - 1])
                .collect(),
            block_tables: BlockTableSnapshot::from_owned_sequence_tables(
                context_lens.iter().map(|_| vec![0, 1, 2, 3]).collect(),
                1,
            ),
            context_lens: context_lens.to_vec(),
            full_context_lens: context_lens.to_vec(),
            query_len: 1,
            block_size: BLOCK_SIZE,
            use_standard_metadata: false,
            max_paged_context_len: 4 * BLOCK_SIZE,
            sliding_window: None,
            decode_window: 1,
            devices: vec![device.clone()],
            num_kv_heads: 1,
        })
    }

    // reads every staged input: the hidden rows plus each row's token, its MRoPE sum and its device context length
    fn forward(inputs: MtpForwardInputs<'_>) -> Result<MtpDraft> {
        let (rows, cols) = inputs.tokens.dims2()?;
        let tokens = inputs.tokens.to_dtype(DType::F32)?;
        let positions = inputs.positions.to_dtype(DType::F32)?.sum(0)?;
        let lens = inputs
            .metadata
            .context_lens
            .as_ref()
            .and_then(|lens| lens.values().next())
            .expect("decode rows carry context lengths")
            .to_dtype(DType::F32)?
            .reshape((rows, cols))?;
        let rowwise = ((tokens + positions)? + lens)?.unsqueeze(2)?;
        Ok(MtpDraft {
            logits_hidden: inputs.target_hidden.broadcast_add(&rowwise)?,
            chain_hidden: rowwise,
        })
    }

    #[test]
    fn replays_take_each_runs_inputs() -> Result<()> {
        inference_nn::skip_without_cuda!();
        let device = Device::new_cuda(0)?;
        let graphs = MtpGraphs::default();
        let shape = (2, 1);
        for (step, start) in [40usize, 41, 70, 99].into_iter().enumerate() {
            let tokens = vec![start as u32 * 3, start as u32 * 5];
            let positions = vec![start + 1, start + 2];
            let starts = vec![start, start + 1];
            let mrope = (0..3)
                .flat_map(|_| starts.iter().map(|&p| p as u32))
                .collect::<Vec<_>>();
            let hidden =
                Tensor::arange(0f32, (2 * HIDDEN) as f32, &device)?.reshape((2, 1, HIDDEN))?;
            let hidden = (hidden + start as f64)?;
            let run = MtpGraphRun {
                shape,
                tokens: &tokens,
                mrope: &mrope,
                positions: &positions,
                starts: &starts,
                target_hidden: &hidden,
                decode_rows: decode_rows(&device, &positions),
                kv: Vec::new(),
            };
            let draft = graphs.forward(&run, &forward)?.expect("CUDA graphs are on");
            let eager = GraphInputs::new(&run)?.forward(&run, &forward)?;
            assert_eq!(
                draft.logits_hidden.to_vec3::<f32>()?,
                eager.logits_hidden.to_vec3::<f32>()?,
                "step {step}"
            );
            assert_eq!(
                draft.chain_hidden.to_vec3::<f32>()?,
                eager.chain_hidden.to_vec3::<f32>()?,
                "step {step}"
            );
        }
        assert_eq!(
            graphs.0.lock().unwrap().entries.len(),
            1,
            "one shape, one graph"
        );
        Ok(())
    }
}
