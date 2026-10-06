//! The paged-attention engine step: CUDA prompt admission, submission, completion and per-step bookkeeping.

use std::{collections::HashMap, sync::Arc, time::Instant};

#[cfg(feature = "cuda")]
use std::time::Duration;

use rand_isaac::Isaac64Rng;
use tokio::sync::Mutex;

#[cfg(feature = "cuda")]
use super::CudaDecodeBatchLease;
use super::Engine;
#[cfg(feature = "cuda")]
use super::{CudaPromptRejection, cuda_decode::CudaDecodeCompletionWorker, cuda_memory};
#[cfg(feature = "cuda")]
use crate::paged_attention::block_hash::MultimodalAttentionPolicy;
#[cfg(feature = "cuda")]
use crate::pipeline::execution::CudaStepSubmission;
use crate::{
    get_mut_arcmutex,
    paged_attention::{
        AttentionBackendKind, KVCacheManager, PagedAttentionMeta,
        block_hash::{adapter_generation_key, compute_block_hashes},
    },
    pipeline::{
        CacheBackendMetadata, Pipeline, StepLookahead, StepSubmission,
        execution::StepSubmissionKind, prompt_chunks::PromptChunkPlan, start_decoding_prompt_rows,
    },
    scheduler::{PagedAttentionSchedulerOutput, modality_signature},
    sequence::Sequence,
};

#[cfg(feature = "cuda")]
type Row = Arc<std::sync::Mutex<Sequence>>;

/// What one paged-attention step takes from the engine loop.
pub(super) struct PagedStepCtx<'a> {
    pub(super) block_size: usize,
    pub(super) kv_cache_manager: Arc<Mutex<KVCacheManager>>,
    pub(super) is_prompt: bool,
    pub(super) step_lookahead: StepLookahead,
    pub(super) rng: &'a Arc<std::sync::Mutex<Isaac64Rng>>,
    pub(super) run_start: Instant,
    #[cfg(feature = "cuda")]
    pub(super) cuda_completion_worker: &'a Option<CudaDecodeCompletionWorker>,
    #[cfg(feature = "cuda")]
    pub(super) cuda_decode_lease: &'a mut Option<CudaDecodeBatchLease>,
    #[cfg(feature = "cuda")]
    pub(super) cuda_memory_pool: &'a mut cuda_memory::CudaMemoryPoolMaintenance,
    #[cfg(feature = "cuda")]
    pub(super) cuda_prompt_preemption_workspace: &'a mut Option<usize>,
}

/// Tokens each scheduled row had computed before the step and the tokens the step schedules for it.
struct StepProgress {
    computed_before: Vec<usize>,
    scheduled: Vec<usize>,
}

impl StepProgress {
    /// Advances rows the step itself did not already move past what it had computed.
    fn advance(&self, seqs: &mut [&mut Sequence]) {
        for ((seq, before), scheduled) in seqs
            .iter_mut()
            .zip(self.computed_before.iter().copied())
            .zip(self.scheduled.iter().copied())
        {
            if seq.num_computed_tokens() == before {
                seq.advance_num_computed_tokens(scheduled);
            }
        }
    }
}

/// What finishing a CUDA paged step needs besides the submission.
#[cfg(feature = "cuda")]
struct CudaPagedFinish<'a> {
    worker: &'a CudaDecodeCompletionWorker,
    rows: &'a [Row],
    progress: &'a StepProgress,
    cuda_decode_lease: &'a mut Option<CudaDecodeBatchLease>,
    run_start: Instant,
}

impl Engine {
    /// One paged-attention step, run with the scheduler lock released; false when it failed and was reported.
    pub(super) async fn step_paged(
        &self,
        mut output: PagedAttentionSchedulerOutput,
        preempted_sequence_ids: Vec<usize>,
        ctx: PagedStepCtx<'_>,
    ) -> bool {
        let PagedStepCtx {
            block_size,
            kv_cache_manager,
            is_prompt,
            step_lookahead,
            rng,
            run_start,
            #[cfg(feature = "cuda")]
            cuda_completion_worker,
            #[cfg(feature = "cuda")]
            cuda_decode_lease,
            #[cfg(feature = "cuda")]
            cuda_memory_pool,
            #[cfg(feature = "cuda")]
            cuda_prompt_preemption_workspace,
        } = ctx;
        if !preempted_sequence_ids.is_empty()
            && let Err(err) = get_mut_arcmutex!(self.pipeline)
                .release_speculative_sequences(&preempted_sequence_ids)
        {
            tracing::error!("Failed to release preempted speculative state: {err}");
        }
        #[cfg(feature = "cuda")]
        let mut prefix_gather_workspace_limit = None;
        #[cfg(not(feature = "cuda"))]
        let prefix_gather_workspace_limit = None;
        #[cfg(feature = "cuda")]
        if is_prompt && !output.scheduled.is_empty() {
            debug_assert!(cuda_decode_lease.is_none());
            match self
                .admit_cuda_prompt(
                    &mut output,
                    &kv_cache_manager,
                    block_size,
                    cuda_memory_pool,
                    cuda_prompt_preemption_workspace,
                )
                .await
            {
                Some(limit) => prefix_gather_workspace_limit = Some(limit),
                None => return false,
            }
        }
        if !output.scheduled.is_empty() {
            for seq in output.scheduled.iter() {
                let mut seq_guard = get_mut_arcmutex!(seq);
                if is_prompt {
                    seq_guard.start_prompt_timing();
                } else {
                    seq_guard.start_completion_timing();
                }
            }

            let mut guards = output
                .scheduled
                .iter_mut()
                .map(|seq| seq.lock().unwrap())
                .collect::<Vec<_>>();

            let mut guards_mut = guards.iter_mut().map(|seq| &mut **seq).collect::<Vec<_>>();

            let staged_width = crate::speculative::staging::staged_batch_width(&guards_mut);
            let scheduler_visible_prompt_step = output.scheduled_prompt_chunks.is_some();
            let progress = StepProgress {
                computed_before: guards_mut
                    .iter()
                    .map(|seq| seq.num_computed_tokens())
                    .collect(),
                scheduled: guards_mut
                    .iter()
                    .enumerate()
                    .map(|(seq_idx, seq)| {
                        if is_prompt
                            && let Some(chunk) = output
                                .scheduled_prompt_chunks
                                .as_ref()
                                .and_then(|chunks| chunks.get(seq_idx))
                        {
                            return chunk.end - chunk.start;
                        }
                        let staged = staged_width
                            .map(|_| seq.active_staged_speculative_len())
                            .unwrap_or_default();
                        seq.num_uncomputed_tokens().saturating_add(staged)
                    })
                    .collect(),
            };

            let res = {
                let mut pipeline = get_mut_arcmutex!(self.pipeline);

                if guards_mut.is_empty() {
                    Ok(StepSubmission::ready(std::time::Duration::ZERO))
                } else {
                    let metadata = paged_attention_meta(
                        &*pipeline,
                        &guards_mut,
                        PagedMetaInputs {
                            kv_cache_manager: &kv_cache_manager,
                            block_size,
                            prompt_chunk_size: output.prompt_chunk_size,
                            scheduled_prompt_chunks: output.scheduled_prompt_chunks.take(),
                            prefix_gather_workspace_limit,
                        },
                    );

                    let return_raw_logits = guards_mut[0].return_raw_logits;
                    assert!(
                        guards_mut
                            .iter()
                            .all(|seq| seq.return_raw_logits == return_raw_logits),
                        "All sequences must either return raw logits, or not."
                    );

                    pipeline
                        .submit_step(
                            &mut guards_mut,
                            is_prompt,
                            return_raw_logits,
                            &mut *get_mut_arcmutex!(self.prefix_cacher),
                            self.disable_eos_stop,
                            rng.clone(),
                            CacheBackendMetadata::PagedAttention { metadata },
                            self.logger.as_ref(),
                            step_lookahead,
                        )
                        .await
                }
            };

            let submission = match res {
                Ok(v) => v,
                Err(e) => {
                    self.report_forward_error("step", e, &mut guards_mut).await;
                    return false;
                }
            };
            drop(guards_mut);
            drop(guards);

            #[cfg(feature = "cuda")]
            if submission.cuda_has_tail() {
                let mut scheduler = get_mut_arcmutex!(self.scheduler);
                scheduler.record_decode_continuation();
            }

            let step_exec_time = match submission.into_inner() {
                StepSubmissionKind::Ready(completion) => completion.duration(),
                #[cfg(feature = "cuda")]
                StepSubmissionKind::Cuda(submission) => {
                    let worker = cuda_completion_worker
                        .as_ref()
                        .expect("CUDA decode submission requires a completion worker");
                    let finish = CudaPagedFinish {
                        worker,
                        rows: &output.scheduled,
                        progress: &progress,
                        cuda_decode_lease,
                        run_start,
                    };
                    let Some(duration) = self.finish_cuda_paged_step(submission, finish).await
                    else {
                        return false;
                    };
                    duration
                }
            };

            let mut guards = output
                .scheduled
                .iter()
                .map(|seq| seq.lock().unwrap())
                .collect::<Vec<_>>();
            let mut guards_mut = guards.iter_mut().map(|seq| &mut **seq).collect::<Vec<_>>();
            progress.advance(&mut guards_mut);
            if is_prompt && !scheduler_visible_prompt_step {
                start_decoding_prompt_rows(&mut guards_mut);
            }
            for seq in guards_mut.iter_mut() {
                if is_prompt {
                    seq.finish_prompt_timing(step_exec_time);
                } else {
                    seq.finish_completion_timing(step_exec_time);
                }
            }

            let total_processed_tokens: usize = progress.scheduled.iter().sum();
            if is_prompt {
                self.logger
                    .add_prefill_tokens_processed(total_processed_tokens);
            } else {
                self.logger
                    .add_decode_tokens_processed(total_processed_tokens);
            }

            // Prompt steps only, as templates re-render finished turns; chunked prefill snapshots inline instead.
            if is_prompt && !scheduler_visible_prompt_step {
                self.snapshot_prompt_recurrent_prefixes(&guards_mut, block_size);
            }

            if self.is_debug {
                let ms_from_last_run = run_start.elapsed().as_secs_f64();
                let total_len = guards.len();
                if total_len > 0 {
                    let lengths = guards
                        .iter()
                        .map(|seq| seq.len().to_string())
                        .collect::<Vec<_>>()
                        .join(", ");

                    let (prompt_lengths, completion_lengths) = if is_prompt {
                        (lengths, "".to_string())
                    } else {
                        ("".to_string(), lengths)
                    };

                    tracing::info!(
                        "Prompt[{}] Completion[{}] - {}ms",
                        prompt_lengths,
                        completion_lengths,
                        ms_from_last_run * 1000.,
                    );
                }
            }
        }
        #[cfg(feature = "cuda")]
        if is_prompt && cuda_memory_pool.after_prompt_step() {
            debug_assert!(cuda_decode_lease.is_none());
            loop {
                let reclaimed = get_mut_arcmutex!(self.pipeline)
                    .reclaim_cuda_graph_memory(cuda_memory::GRAPH_RECLAIM_BATCH_SIZE);
                if reclaimed == 0 || !cuda_memory_pool.after_graph_reclaim(0).graph_pressure {
                    break;
                }
            }
        }
        true
    }

    #[cfg(feature = "cuda")]
    async fn report_rows_error(
        &self,
        stage: &'static str,
        e: impl std::fmt::Display + std::fmt::Debug,
        rows: &[Row],
    ) {
        let mut guards = rows
            .iter()
            .map(|seq| seq.lock().unwrap())
            .collect::<Vec<_>>();
        let mut seqs = guards.iter_mut().map(|seq| &mut **seq).collect::<Vec<_>>();
        self.report_forward_error(stage, e, &mut seqs).await;
    }

    /// Shrinks the prompt batch until its workspace fits; the prefix-gather workspace limit, or None once rejected.
    #[cfg(feature = "cuda")]
    async fn admit_cuda_prompt(
        &self,
        output: &mut PagedAttentionSchedulerOutput,
        kv_cache_manager: &Arc<Mutex<KVCacheManager>>,
        block_size: usize,
        cuda_memory_pool: &mut cuda_memory::CudaMemoryPoolMaintenance,
        cuda_prompt_preemption_workspace: &mut Option<usize>,
    ) -> Option<usize> {
        let (
            model_metadata,
            activation_dtype,
            cache_dtype,
            device_is_cuda,
            has_sliding_window,
            fa3_num_sm_by_layer,
        ) = {
            let pipeline = get_mut_arcmutex!(self.pipeline);
            let metadata = pipeline.get_metadata();
            let cache_dtype = metadata
                .cache_config
                .as_ref()
                .map(|config| config.cache_type.to_dtype(metadata.activation_dtype))
                .unwrap_or(metadata.activation_dtype);
            (
                metadata.model_metadata.clone(),
                metadata.activation_dtype,
                cache_dtype,
                pipeline
                    .execution_devices()
                    .iter()
                    .all(candle_core::Device::is_cuda),
                metadata.sliding_window.is_some(),
                metadata
                    .cache_engine
                    .as_ref()
                    .map(|engine| engine.fa3_prefill_num_sm_by_layer().to_vec())
                    .unwrap_or_default(),
            )
        };
        let rejection = loop {
            let query_lens = output
                .scheduled
                .iter()
                .enumerate()
                .map(|(seq_idx, seq)| {
                    let seq = get_mut_arcmutex!(seq);
                    output
                        .scheduled_prompt_chunks
                        .as_ref()
                        .and_then(|chunks| chunks.get(seq_idx))
                        .map(|chunk| chunk.end.saturating_sub(chunk.start))
                        .unwrap_or_else(|| seq.num_uncomputed_tokens())
                })
                .collect::<Vec<_>>();
            let full_context_lens = output
                .scheduled
                .iter()
                .zip(&query_lens)
                .map(|(seq, query_len)| {
                    get_mut_arcmutex!(seq)
                        .num_computed_tokens()
                        .saturating_add(*query_len)
                })
                .collect::<Vec<_>>();
            let max_pages_per_sequence = {
                let sequence_ids = output
                    .scheduled
                    .iter()
                    .map(|seq| *get_mut_arcmutex!(seq).id())
                    .collect::<Vec<_>>();
                let manager = get_mut_arcmutex!(kv_cache_manager);
                sequence_ids
                    .iter()
                    .map(|seq_id| manager.num_blocks_for_request(*seq_id))
                    .max()
                    .unwrap_or_default()
            };
            let has_noncausal_mm_context = output
                .scheduled_prompt_chunks
                .as_ref()
                .and_then(|chunks| chunks.first())
                .is_some_and(|chunk| {
                    chunk.attention_policy == MultimodalAttentionPolicy::NonCausal
                })
                || output.scheduled.iter().any(|seq| {
                    get_mut_arcmutex!(seq).mm_features().iter().any(|feature| {
                        feature.attention_policy == MultimodalAttentionPolicy::NonCausal
                    })
                });
            let has_donor_cache_layers = model_metadata
                .as_deref()
                .is_some_and(crate::paged_attention::plan::model_has_donor_paged_cache_layers);
            let requires_prefix_attention = has_noncausal_mm_context
                || has_donor_cache_layers
                || query_lens
                    .iter()
                    .zip(&full_context_lens)
                    .any(|(query, full)| full > query);
            let workspace = match crate::paged_attention::plan::prompt_prefill_workspace(
                model_metadata.as_deref(),
                crate::paged_attention::plan::PromptPrefillWorkspaceInput {
                    activation_dtype,
                    cache_dtype,
                    device_is_cuda,
                    block_size,
                    query_lens: &query_lens,
                    full_context_lens: &full_context_lens,
                    max_pages_per_sequence,
                    requires_prefix_attention,
                    is_causal: !has_noncausal_mm_context,
                    causality_known: true,
                    has_custom_mask: has_noncausal_mm_context,
                    has_noncausal_mm_context,
                    has_sliding_window,
                    fa3_num_sm_by_layer: &fa3_num_sm_by_layer,
                },
            ) {
                Ok(workspace) => workspace,
                Err(err) => {
                    break CudaPromptRejection::Internal(format!(
                        "CUDA prompt memory preflight could not establish a safe attention plan: {err}"
                    ));
                }
            };
            let workspace_bytes = workspace.bytes;
            let memory_status = self.maintain_cuda_prompt_memory(cuda_memory_pool, workspace_bytes);
            let previous = output.scheduled.len();
            match cuda_memory::prompt_batch_memory_action(
                previous,
                memory_status.transient_pressure,
            ) {
                cuda_memory::PromptBatchMemoryAction::Proceed => {
                    return Some(workspace.gather_workspace_bytes);
                }
                cuda_memory::PromptBatchMemoryAction::Retain(retained) => {
                    let first_omitted_id = output
                        .retain_prompt_prefix(retained)
                        .expect("reduced prompt batch must omit a tail");
                    get_mut_arcmutex!(self.scheduler).defer_prompt_tail(first_omitted_id);
                    cuda_memory::record_prompt_batch_reduction(previous, retained);
                }
                cuda_memory::PromptBatchMemoryAction::Reject => {
                    if !memory_status.insufficient_total_capacity {
                        *cuda_prompt_preemption_workspace = Some(
                            (*cuda_prompt_preemption_workspace)
                                .map_or(workspace_bytes, |current: usize| {
                                    current.max(workspace_bytes)
                                }),
                        );
                    }
                    break if memory_status.insufficient_total_capacity {
                        CudaPromptRejection::InvalidRequest(format!(
                            "CUDA prompt requires {workspace_bytes} bytes of transient workspace and cannot fit after device-memory reclamation"
                        ))
                    } else if memory_status.maintenance_failed {
                        CudaPromptRejection::Internal(
                            "CUDA prompt memory preflight could not verify allocator capacity"
                                .to_string(),
                        )
                    } else {
                        CudaPromptRejection::Unavailable(format!(
                            "CUDA memory pressure prevented prompt admission requiring {workspace_bytes} bytes of transient workspace"
                        ))
                    };
                }
            }
        };
        self.reject_prompt_for_cuda_memory(&output.scheduled, rejection)
            .await;
        None
    }

    /// Completes a CUDA step, commits its tokens and keeps or drains its tail; None once a failure is reported.
    #[cfg(feature = "cuda")]
    async fn finish_cuda_paged_step(
        &self,
        submission: CudaStepSubmission,
        finish: CudaPagedFinish<'_>,
    ) -> Option<Duration> {
        let CudaPagedFinish {
            worker,
            rows,
            progress,
            cuda_decode_lease,
            run_start,
        } = finish;
        let completion_result = self.complete_cuda_step(worker, submission).await;
        let mut completion_guards = rows
            .iter()
            .map(|seq| seq.lock().unwrap())
            .collect::<Vec<_>>();
        let mut completion_guards_mut = completion_guards
            .iter_mut()
            .map(|seq| &mut **seq)
            .collect::<Vec<_>>();
        let mut completion = match completion_result {
            Ok(v) => v,
            Err(e) => {
                self.report_forward_error("CUDA decode completion", e, &mut completion_guards_mut)
                    .await;
                return None;
            }
        };

        progress.advance(&mut completion_guards_mut);
        let commit_rows = completion_guards_mut
            .iter()
            .map(|seq| !seq.is_finished_paged_attn())
            .collect::<Vec<_>>();
        let finish_result: candle_core::Result<_> = async {
            let pipeline = get_mut_arcmutex!(self.pipeline);
            if crate::pipeline::sampling::cuda_token_batch_will_finish(
                &*pipeline,
                &completion_guards_mut,
                completion.token_ids(),
                &commit_rows,
                self.disable_eos_stop,
            )? {
                completion.synchronize_tail()?;
            }
            completion
                .finish(
                    &*pipeline,
                    &mut completion_guards_mut,
                    &commit_rows,
                    &mut *get_mut_arcmutex!(self.prefix_cacher),
                    self.disable_eos_stop,
                )
                .await
        }
        .await;
        let completion = match finish_result {
            Ok(v) => v,
            Err(e) => {
                self.report_forward_error("CUDA decode finalize", e, &mut completion_guards_mut)
                    .await;
                return None;
            }
        };
        let duration = run_start.elapsed();
        let tail = completion.into_cuda_tail();
        let any_live = completion_guards_mut
            .iter()
            .any(|seq| !seq.is_finished_paged_attn());
        drop(completion_guards_mut);
        drop(completion_guards);

        if let Some(tail) = tail {
            if any_live {
                match CudaDecodeBatchLease::new(rows.to_vec(), tail) {
                    Ok(lease) => *cuda_decode_lease = Some(lease),
                    Err(e) => {
                        self.report_rows_error("CUDA decode lease", e, rows).await;
                        return None;
                    }
                }
            } else {
                if let Err(e) = tail.drain() {
                    self.report_rows_error("CUDA decode tail drain", e, rows)
                        .await;
                    return None;
                }
                self.account_cuda_decode_rows(rows);
            }
        }
        Some(duration)
    }

    /// Snapshots recurrent state at full-block prompt boundaries so hybrid models can reuse it on a paged prefix hit.
    fn snapshot_prompt_recurrent_prefixes(&self, seqs: &[&mut Sequence], block_size: usize) {
        let mut pipeline = get_mut_arcmutex!(self.pipeline);
        let mut prefix_cacher = get_mut_arcmutex!(self.prefix_cacher);
        if !pipeline.cache().is_hybrid() || !prefix_cacher.accepts_paged_recurrent_prefix() {
            return;
        }
        let prefix_policy = pipeline.speculative_prefix_checkpoint_policy();
        for seq in seqs {
            if matches!(
                prefix_policy.replay_for(modality_signature(seq)),
                crate::speculative::SpeculativePrefixReplay::Full
            ) {
                continue;
            }
            let encoded_len = seq.num_computed_tokens();
            if encoded_len == 0 || encoded_len % block_size != 0 {
                continue;
            }

            let num_blocks = encoded_len / block_size;
            let adapter_key = adapter_generation_key(seq.adapter_generation());
            let block_hashes = compute_block_hashes(
                seq.get_toks(),
                block_size,
                seq.mm_features(),
                adapter_key.as_slice(),
            );
            if block_hashes.len() < num_blocks {
                continue;
            }
            let owner = block_hashes[num_blocks - 1];
            if prefix_cacher.has_paged_recurrent_owner(owner) {
                continue;
            }
            if let Err(e) = pipeline.snapshot_paged_recurrent_prefix(
                seq,
                &mut prefix_cacher,
                block_size,
                encoded_len,
            ) {
                tracing::warn!(
                    "Failed snapshotting recurrent prefix for sequence {}: {e}",
                    seq.id()
                );
            }
        }
    }
}

/// The step's scheduling facts that `PagedAttentionMeta` carries beside the pipeline's.
struct PagedMetaInputs<'a> {
    kv_cache_manager: &'a Arc<Mutex<KVCacheManager>>,
    block_size: usize,
    prompt_chunk_size: Option<usize>,
    scheduled_prompt_chunks: Option<Vec<PromptChunkPlan>>,
    prefix_gather_workspace_limit: Option<usize>,
}

fn paged_attention_meta(
    pipeline: &dyn Pipeline,
    seqs: &[&mut Sequence],
    inputs: PagedMetaInputs<'_>,
) -> PagedAttentionMeta {
    let PagedMetaInputs {
        kv_cache_manager,
        block_size,
        prompt_chunk_size,
        scheduled_prompt_chunks,
        prefix_gather_workspace_limit,
    } = inputs;
    let pipeline_metadata = pipeline.get_metadata();
    let model_metadata = pipeline_metadata.model_metadata.as_ref();
    let max_paged_context_len = {
        let kv_mgr = get_mut_arcmutex!(kv_cache_manager);
        kv_mgr.num_gpu_blocks().saturating_sub(1).max(1) * block_size
    };
    let prompt_chunk_attention_policy = scheduled_prompt_chunks
        .as_ref()
        .and_then(|chunks| chunks.first())
        .map(|chunk| chunk.attention_policy)
        .unwrap_or(crate::paged_attention::block_hash::MultimodalAttentionPolicy::Causal);
    let is_final_prompt_chunk = scheduled_prompt_chunks.as_ref().is_none_or(|chunks| {
        chunks
            .iter()
            .zip(seqs.iter())
            .all(|(chunk, seq)| chunk.end == seq.get_toks().len())
    });
    PagedAttentionMeta {
        block_size,
        max_paged_context_len,
        sliding_window: pipeline_metadata.sliding_window,
        attention_backend: model_metadata
            .map(|metadata| metadata.attention_backend_kind())
            .unwrap_or(AttentionBackendKind::Standard),
        prefill_attention_heads: model_metadata
            .map(|metadata| metadata.num_attn_heads())
            .unwrap_or(1)
            .max(1),
        prefill_key_value_heads: model_metadata
            .map(|metadata| metadata.num_kv_heads())
            .unwrap_or(1)
            .max(1),
        prefill_head_dim: model_metadata
            .map(|metadata| metadata.k_head_dim())
            .unwrap_or(1)
            .max(1),
        kv_cache_manager: kv_cache_manager.clone(),
        prompt_chunk_size,
        scheduled_prompt_chunks,
        prompt_chunk_attention_policy,
        has_noncausal_mm_context: false,
        prefix_gather_workspace_limit,
        mm_prefix_ranges_by_seq_id: HashMap::new(),
        full_mm_prefix_ranges_by_seq_id: HashMap::new(),
        enable_packed_prefill: pipeline.supports_packed_prefill(),
        is_final_prompt_chunk,
        needs_logits: is_final_prompt_chunk || seqs[0].return_raw_logits,
    }
}
