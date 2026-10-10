//! One pipeline step: forward the scheduled rows, then sample or respond with what the forward produced.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use futures::future::BoxFuture;
use inference_tensor::{DType, Device, IndexOp, Tensor};
use rand_isaac::Isaac64Rng;

use super::{
    CacheBackendMetadata, CacheInstruction, ForwardInputsResult, ForwardStepResult,
    InputProcessorOutput, Pipeline, SpeculativePromptChunk, SpeculativePromptRow, StepLookahead,
    StepSubmission, next_pipeline_prompt_chunk_group, prompt_chunk_is_final,
    prompt_chunks::{PromptChunkPlan, build_prompt_chunk_plan, recurrent_checkpoint_boundary},
    response, sampling, should_sample_step, should_try_speculative_sampling,
};
#[cfg(feature = "cuda")]
use super::{cuda_graph::CudaDecodeGraphLaunch, execution};
use crate::{
    IntervalLogger,
    paged_attention::PagedAttentionMeta,
    pipeline::text_models_inputs_processor::NoncausalMmContext,
    prefix_cacher::PrefixCacheManagerV2,
    scheduler::modality_signature,
    sequence::{SeqStepType, Sequence, SequenceState},
};

/// Moves prompt rows that go on to decode into the completion state once their prompt is done.
pub(crate) fn start_decoding_prompt_rows(seqs: &mut [&mut Sequence]) {
    for seq in seqs.iter_mut() {
        if !seq.is_finished_paged_attn()
            && matches!(seq.sequence_stepping_type(), SeqStepType::PromptAndDecode)
        {
            seq.set_state(SequenceState::RunningCompletion);
        }
    }
}

/// What every step shares: the batch kind and what sampling needs.
struct StepCtx<'a> {
    is_prompt: bool,
    return_raw_logits: bool,
    prefix_cacher: &'a mut PrefixCacheManagerV2,
    disable_eos_stop: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    logger: &'a IntervalLogger,
}

/// The speculative sampling a causal-generation step tries before plain sampling.
struct SpeculativeAttempt<'a> {
    batched_logits: Option<&'a Tensor>,
    metadata: Option<PagedAttentionMeta>,
}

/// Per-row forward results gathered across a step's forwards.
struct RowOutputs {
    generation: Vec<Option<ForwardInputsResult>>,
    raw_logits: Vec<Vec<Option<Tensor>>>,
    embeddings: Vec<Option<Tensor>>,
}

impl RowOutputs {
    fn new(rows: usize, forwards: usize) -> Self {
        Self {
            generation: vec![None; rows],
            raw_logits: vec![vec![None; forwards]; rows],
            embeddings: vec![None; rows],
        }
    }

    fn scatter(
        &mut self,
        forward_idx: usize,
        result: &ForwardInputsResult,
        seq_indices: Vec<usize>,
    ) -> inference_tensor::Result<()> {
        for (logit_idx, seq_idx) in seq_indices.into_iter().enumerate() {
            match result {
                ForwardInputsResult::RawLogits { logits } => {
                    self.raw_logits[seq_idx][forward_idx] =
                        Some(logits.i(logit_idx)?.to_device(&Device::Cpu)?);
                }
                ForwardInputsResult::Embeddings { embeddings } => {
                    self.embeddings[seq_idx] =
                        Some(embeddings.i(logit_idx)?.to_device(&Device::Cpu)?);
                }
                _ => self.generation[seq_idx] = Some(result.index_bs(logit_idx)?),
            }
        }
        Ok(())
    }

    /// Sends raw-logit or embedding responses when the step produced them; the time that took, if it did.
    async fn respond_without_generation(
        &mut self,
        seqs: &mut [&mut Sequence],
    ) -> inference_tensor::Result<Option<Duration>> {
        let start = Instant::now();
        if self.raw_logits[0][0].is_some() {
            response::send_raw_responses(
                seqs,
                std::mem::take(&mut self.raw_logits)
                    .into_iter()
                    .map(|raw| raw.into_iter().flatten().collect::<Vec<_>>())
                    .collect(),
            )
            .await?;
            return Ok(Some(start.elapsed()));
        }
        if self.embeddings[0].is_some() {
            response::send_embedding_responses(
                seqs,
                std::mem::take(&mut self.embeddings)
                    .into_iter()
                    .map(|raw| {
                        raw.unwrap()
                            .to_dtype(DType::F32)
                            .unwrap()
                            .to_vec1::<f32>()
                            .unwrap()
                    })
                    .collect(),
            )
            .await?;
            return Ok(Some(start.elapsed()));
        }
        Ok(None)
    }

    fn into_generation(self) -> Vec<ForwardInputsResult> {
        self.generation
            .into_iter()
            .map(|result| result.expect("missing forward result"))
            .collect()
    }
}

/// One forward of a paged step and the bookkeeping that follows it.
struct StepInput {
    processed: anyhow::Result<InputProcessorOutput>,
    recurrent_boundaries: Vec<(usize, usize)>,
    prompt_chunk: Option<SpeculativePromptChunk>,
    computed_updates: Vec<(usize, usize)>,
}

/// A paged step's forwards, before sampling.
struct PagedForwards {
    outputs: RowOutputs,
    batched_causal_logits: Option<Tensor>,
    #[cfg(feature = "cuda")]
    batched_cuda_decode: Option<CudaDecodeGraphLaunch>,
    exec_duration: Duration,
}

/// A prompt the pipeline splits into chunks, one forward per chunk group.
struct ChunkedPrompt<'a> {
    plans: Vec<Vec<PromptChunkPlan>>,
    metadata: &'a PagedAttentionMeta,
    block_size: usize,
    scheduler_visible: bool,
    scheduler_visible_is_final: bool,
}

impl dyn Pipeline {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn submit_step<'a>(
        &'a mut self,
        input_seqs: &'a mut [&mut Sequence],
        is_prompt: bool,
        return_raw_logits: bool,
        prefix_cacher: &'a mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
        backend_metadata: CacheBackendMetadata,
        logger: &'a IntervalLogger,
        lookahead: StepLookahead,
    ) -> BoxFuture<'a, Result<StepSubmission, inference_tensor::Error>> {
        Box::pin(async move {
            let mut ctx = StepCtx {
                is_prompt,
                return_raw_logits,
                prefix_cacher,
                disable_eos_stop,
                rng,
                logger,
            };
            match backend_metadata {
                CacheBackendMetadata::DefaultInstructions { pre_op, post_op } => {
                    self.default_step(input_seqs, &mut ctx, pre_op, post_op)
                        .await
                }
                CacheBackendMetadata::PagedAttention { metadata } => {
                    self.paged_step(input_seqs, &mut ctx, metadata, lookahead)
                        .await
                }
            }
        })
    }

    fn process_step_inputs(
        &self,
        seqs: &mut [&mut Sequence],
        ctx: &StepCtx<'_>,
        paged_metadata: Option<PagedAttentionMeta>,
    ) -> anyhow::Result<InputProcessorOutput> {
        self.get_processor().inputs_processor().process_inputs(
            self.tokenizer(),
            seqs,
            ctx.is_prompt,
            &self.device(),
            self.get_metadata().no_kv_cache,
            None,
            ctx.return_raw_logits,
            self.get_metadata().sliding_window,
            self.get_input_processor_config(),
            paged_metadata,
            self.device_mapper(),
        )
    }

    /// Whether a forward keeps its causal logits batched for device-side sampling or verification.
    fn preserves_causal_generation(
        &self,
        seqs: &[&mut Sequence],
        batched: bool,
        ctx: &StepCtx<'_>,
    ) -> bool {
        batched
            && !ctx.return_raw_logits
            && self.device().is_cuda()
            && ((self.supports_batched_cuda_sampling() && sampling::can_sample_batch_cuda(seqs))
                || crate::speculative::verifier::can_batch_device_verify(seqs))
    }

    async fn default_step(
        &mut self,
        input_seqs: &mut [&mut Sequence],
        ctx: &mut StepCtx<'_>,
        pre_op: CacheInstruction,
        post_op: CacheInstruction,
    ) -> inference_tensor::Result<StepSubmission> {
        if !ctx.is_prompt && !ctx.return_raw_logits {
            crate::speculative::driver::clear_staged_speculative_tokens(input_seqs);
        }

        let processed = self.process_step_inputs(input_seqs, ctx, None);
        let mut outputs = RowOutputs::new(input_seqs.len(), 1);
        let InputProcessorOutput {
            inputs,
            seq_indices,
        } = processed.map_err(inference_tensor::Error::msg)?;
        match pre_op {
            CacheInstruction::In => self.clone_in_cache(input_seqs)?,
            CacheInstruction::Nothing => (),
            CacheInstruction::Reset {
                load_preallocated_cache,
            } => self.set_none_cache(input_seqs, false, load_preallocated_cache)?,
            _ => unreachable!("Unreachable PRE cache op."),
        }

        let preserve_causal_generation =
            self.preserves_causal_generation(input_seqs, input_seqs.len() > 1, ctx);
        let start = Instant::now();
        let result = self
            .forward_inputs(inputs, ctx.return_raw_logits)?
            .into_cpu_for_batch(input_seqs.len(), preserve_causal_generation)?;
        let mut exec_duration = start.elapsed();
        outputs.scatter(0, &result, seq_indices)?;

        match post_op {
            CacheInstruction::Out => self.clone_out_cache(input_seqs),
            CacheInstruction::Nothing => (),
            CacheInstruction::Reset {
                load_preallocated_cache,
            } => self.set_none_cache(input_seqs, false, load_preallocated_cache)?,
            _ => unreachable!("Unreachable POST cache op."),
        }

        if let Some(spent) = outputs.respond_without_generation(input_seqs).await? {
            return Ok(StepSubmission::ready(exec_duration + spent));
        }

        let start = Instant::now();
        let speculative =
            (!ctx.is_prompt && !ctx.return_raw_logits).then_some(SpeculativeAttempt {
                batched_logits: None,
                metadata: None,
            });
        self.dispatch_generation(input_seqs, outputs.into_generation(), ctx, speculative)
            .await?;
        exec_duration += start.elapsed();

        Ok(StepSubmission::ready(exec_duration))
    }

    async fn paged_step(
        &mut self,
        input_seqs: &mut [&mut Sequence],
        ctx: &mut StepCtx<'_>,
        mut metadata: PagedAttentionMeta,
        lookahead: StepLookahead,
    ) -> inference_tensor::Result<StepSubmission> {
        let is_prompt = ctx.is_prompt;
        let block_size = metadata.block_size;
        let speculative_metadata = metadata.clone();
        let scheduled_prompt_chunks = metadata.scheduled_prompt_chunks.take();
        let scheduler_visible_prompt_step = scheduled_prompt_chunks.is_some();
        let scheduler_visible_prompt_is_final =
            scheduler_visible_prompt_step && metadata.is_final_prompt_chunk;
        let chunk_size = if !scheduler_visible_prompt_step
            && is_prompt
            && !ctx.return_raw_logits
            && self.device().is_cuda()
        {
            metadata.prompt_chunk_size
        } else {
            None
        };
        if is_prompt {
            self.get_processor()
                .inputs_processor()
                .prepare_for_paged_prompt_planning(
                    self.tokenizer(),
                    input_seqs,
                    &self.device(),
                    self.get_input_processor_config(),
                    Some(&mut metadata),
                )
                .map_err(|e| inference_tensor::Error::msg(e.to_string()))?;
            for seq in input_seqs.iter_mut() {
                seq.clip_prefix_cache_len_for_mm_features(metadata.block_size);
            }
        }
        let chunk_plans = scheduled_prompt_chunks
            .map(|chunks| chunks.into_iter().map(|chunk| vec![chunk]).collect())
            .or_else(|| self.plan_prompt_chunks(input_seqs, chunk_size, block_size));
        let should_chunk = scheduler_visible_prompt_step
            || chunk_plans
                .as_ref()
                .is_some_and(|plans| plans.iter().any(|plan| plan.len() > 1));
        let cuda_decode_lookahead = lookahead.is_enabled()
            && !is_prompt
            && !ctx.return_raw_logits
            && self.device().is_cuda()
            && self.supports_batched_cuda_sampling()
            && sampling::can_submit_cuda_token_batch_seqs(input_seqs)
            && sampling::can_launch_one_token_lookahead(
                input_seqs,
                self.get_metadata().max_seq_len,
            );

        let step_inputs = match chunk_plans.filter(|_| should_chunk) {
            Some(plans) => self.chunked_prompt_inputs(
                input_seqs,
                ctx,
                ChunkedPrompt {
                    plans,
                    metadata: &metadata,
                    block_size,
                    scheduler_visible: scheduler_visible_prompt_step,
                    scheduler_visible_is_final: scheduler_visible_prompt_is_final,
                },
            ),
            None => {
                metadata.set_noncausal_mm_context(input_seqs);
                let prompt_chunk = is_prompt.then(|| SpeculativePromptChunk {
                    rows: input_seqs
                        .iter()
                        .enumerate()
                        .map(|(seq_idx, seq)| SpeculativePromptRow {
                            seq_idx,
                            range: (seq.prefix_cache_len(), seq.get_toks().len()),
                            tokens: seq.get_toks().to_vec(),
                        })
                        .collect(),
                    is_final_prompt_chunk: true,
                });
                vec![StepInput {
                    processed: self.process_step_inputs(input_seqs, ctx, Some(metadata)),
                    recurrent_boundaries: Vec::new(),
                    prompt_chunk,
                    computed_updates: Vec::new(),
                }]
            }
        };
        let PagedForwards {
            mut outputs,
            batched_causal_logits,
            #[cfg(feature = "cuda")]
            mut batched_cuda_decode,
            mut exec_duration,
        } = self.run_paged_forwards(
            input_seqs,
            ctx,
            step_inputs,
            cuda_decode_lookahead,
            &speculative_metadata,
        )?;

        if let Some(spent) = outputs.respond_without_generation(input_seqs).await? {
            return Ok(StepSubmission::ready(exec_duration + spent));
        }
        if !should_sample_step(
            is_prompt,
            scheduler_visible_prompt_step,
            scheduler_visible_prompt_is_final,
        ) {
            return Ok(StepSubmission::ready(exec_duration));
        }

        let start = Instant::now();
        let mut speculative_batched_logits = None;
        if let Some(batched_causal_logits) = batched_causal_logits {
            #[cfg(feature = "cuda")]
            let mut batched_causal_logits = batched_causal_logits;
            #[cfg(feature = "cuda")]
            if cuda_decode_lookahead {
                let forward = ForwardStepResult::cuda_decode(
                    ForwardInputsResult::CausalGeneration {
                        logits: batched_causal_logits,
                    },
                    batched_cuda_decode.take(),
                );
                match execution::submit_forward_lookahead(
                    self,
                    input_seqs,
                    forward,
                    exec_duration,
                    &ctx.rng,
                )? {
                    Ok(submission) => return Ok(StepSubmission::cuda(submission)),
                    Err(forward) => {
                        let ForwardInputsResult::CausalGeneration { logits } = forward.output
                        else {
                            unreachable!("CUDA lookahead changed the forward result type")
                        };
                        batched_causal_logits = logits;
                    }
                }
            }
            if self
                .try_sample_causal_gen_batched(
                    input_seqs,
                    &batched_causal_logits,
                    ctx.prefix_cacher,
                    ctx.disable_eos_stop,
                    ctx.rng.clone(),
                )
                .await?
            {
                if scheduler_visible_prompt_step {
                    start_decoding_prompt_rows(input_seqs);
                }
                exec_duration += start.elapsed();
                return Ok(StepSubmission::ready(exec_duration));
            }
            for (seq_idx, logits) in outputs.generation.iter_mut().enumerate() {
                *logits = Some(ForwardInputsResult::CausalGeneration {
                    logits: batched_causal_logits.i(seq_idx)?,
                });
            }
            speculative_batched_logits = Some(batched_causal_logits);
        }
        let speculative = should_try_speculative_sampling(
            is_prompt,
            scheduler_visible_prompt_step,
            scheduler_visible_prompt_is_final,
            ctx.return_raw_logits,
            is_prompt && self.supports_speculative_prompt_bootstrap(),
        )
        .then(|| SpeculativeAttempt {
            batched_logits: speculative_batched_logits.as_ref(),
            metadata: Some(speculative_metadata),
        });
        self.dispatch_generation(input_seqs, outputs.into_generation(), ctx, speculative)
            .await?;
        if scheduler_visible_prompt_step {
            start_decoding_prompt_rows(input_seqs);
        }
        exec_duration += start.elapsed();

        Ok(StepSubmission::ready(exec_duration))
    }

    /// The pipeline's own prompt chunking, when a chunk size applies and nothing forces one forward per prompt.
    fn plan_prompt_chunks(
        &self,
        input_seqs: &[&mut Sequence],
        chunk_size: Option<usize>,
        block_size: usize,
    ) -> Option<Vec<Vec<PromptChunkPlan>>> {
        let chunk_size = chunk_size?;
        let has_deferred_multimodal_prompt = input_seqs.iter().any(|seq| {
            (seq.has_images() || seq.has_audios() || seq.has_videos())
                && seq.mm_features().is_empty()
        });
        let has_suffix_only_prefill = input_seqs
            .iter()
            .any(|seq| seq.has_suffix_only_prefill_toks());
        let keep_complete_packed_candidates = input_seqs.len() > 1
            && self.supports_packed_prefill()
            && input_seqs
                .iter()
                .all(|seq| seq.prefix_cache_len() == 0 && seq.len() <= chunk_size);
        if has_deferred_multimodal_prompt
            || has_suffix_only_prefill
            || keep_complete_packed_candidates
        {
            return None;
        }
        let block_align = self.cache().is_hybrid().then_some(block_size);
        let prefix_policy = self.speculative_prefix_checkpoint_policy();
        Some(
            input_seqs
                .iter()
                .map(|seq| {
                    build_prompt_chunk_plan(
                        seq.get_toks().len(),
                        seq.prefix_cache_len(),
                        chunk_size,
                        block_align,
                        prefix_policy.replay_for(modality_signature(seq)),
                        seq.mm_features(),
                    )
                })
                .collect(),
        )
    }

    /// One forward input per chunk group, with each row's prompt restored once all are built.
    fn chunked_prompt_inputs(
        &self,
        input_seqs: &mut [&mut Sequence],
        ctx: &StepCtx<'_>,
        prompt: ChunkedPrompt<'_>,
    ) -> Vec<StepInput> {
        let ChunkedPrompt {
            plans: chunk_plans,
            metadata,
            block_size,
            scheduler_visible,
            scheduler_visible_is_final,
        } = prompt;
        let hybrid_recurrent = self.cache().is_hybrid();
        let prefix_policy = self.speculative_prefix_checkpoint_policy();
        let originals = input_seqs
            .iter()
            .map(|seq| (seq.get_toks().to_vec(), seq.prefix_cache_len()))
            .collect::<Vec<_>>();
        let recurrent_checkpoint_boundaries = input_seqs
            .iter()
            .zip(&originals)
            .map(|(seq, (tokens, prefix_len))| {
                recurrent_checkpoint_boundary(
                    tokens.len(),
                    *prefix_len,
                    hybrid_recurrent.then_some(block_size),
                    prefix_policy.replay_for(modality_signature(seq)),
                    seq.mm_features(),
                )
            })
            .collect::<Vec<_>>();
        let mut plan_indices = vec![0usize; chunk_plans.len()];
        let requires_uniform_prompt_batch = self.requires_uniform_prompt_batch();
        let supports_packed_prefill = self.supports_packed_prefill();
        let mut inputs = Vec::new();
        while plan_indices
            .iter()
            .zip(chunk_plans.iter())
            .any(|(plan_idx, plan)| *plan_idx < plan.len())
        {
            let (active_indices, attention_policy, planned_final_prompt_chunk) =
                next_pipeline_prompt_chunk_group(
                    &plan_indices,
                    &chunk_plans,
                    requires_uniform_prompt_batch,
                    supports_packed_prefill,
                    hybrid_recurrent,
                )
                .expect("at least one chunk plan is active");
            let is_final_prompt_chunk = prompt_chunk_is_final(
                scheduler_visible,
                scheduler_visible_is_final,
                planned_final_prompt_chunk,
            );

            let mut recurrent_boundaries = Vec::new();
            let mut prompt_chunk = SpeculativePromptChunk {
                rows: Vec::with_capacity(active_indices.len()),
                is_final_prompt_chunk,
            };
            for &seq_idx in &active_indices {
                let chunk = chunk_plans[seq_idx][plan_indices[seq_idx]];
                let seq = &mut input_seqs[seq_idx];
                seq.set_prefix_cache_len(chunk.start);
                seq.set_prefill_toks(originals[seq_idx].0[..chunk.end].to_vec());
                if recurrent_checkpoint_boundaries[seq_idx] == Some(chunk.end) {
                    recurrent_boundaries.push((seq_idx, chunk.end));
                }
                prompt_chunk.rows.push(SpeculativePromptRow {
                    seq_idx,
                    range: (chunk.start, chunk.end),
                    tokens: originals[seq_idx].0.clone(),
                });
            }

            let mut chunk_metadata = metadata.clone();
            chunk_metadata.prompt_chunk_attention_policy = attention_policy;
            chunk_metadata.is_final_prompt_chunk = is_final_prompt_chunk;
            chunk_metadata.needs_logits = is_final_prompt_chunk || ctx.return_raw_logits;
            let mut active_input_seqs = input_seqs
                .iter_mut()
                .enumerate()
                .filter_map(|(idx, seq)| active_indices.contains(&idx).then_some(&mut **seq))
                .collect::<Vec<_>>();
            chunk_metadata.set_noncausal_mm_context(active_input_seqs.as_slice());
            let mut processed = self.process_step_inputs(
                active_input_seqs.as_mut_slice(),
                ctx,
                Some(chunk_metadata),
            );
            drop(active_input_seqs);
            if let Ok(processed) = &mut processed {
                for seq_idx in &mut processed.seq_indices {
                    *seq_idx = active_indices[*seq_idx];
                }
            }
            let computed_updates = if scheduler_visible {
                active_indices
                    .iter()
                    .map(|&seq_idx| (seq_idx, chunk_plans[seq_idx][plan_indices[seq_idx]].end))
                    .collect()
            } else {
                Vec::new()
            };
            inputs.push(StepInput {
                processed,
                recurrent_boundaries,
                prompt_chunk: Some(prompt_chunk),
                computed_updates,
            });
            for &seq_idx in &active_indices {
                plan_indices[seq_idx] += 1;
            }
        }
        for (seq, (tokens, prefix_len)) in input_seqs.iter_mut().zip(originals.iter()) {
            seq.set_prefix_cache_len(*prefix_len);
            seq.set_prefill_toks(tokens.clone());
        }
        inputs
    }

    fn install_recurrent_slots(
        &self,
        input_seqs: &[&mut Sequence],
        seq_indices: &[usize],
    ) -> inference_tensor::Result<()> {
        let mut hybrid_cache = self.cache().hybrid();
        let sequence_slots = seq_indices
            .iter()
            .map(|&seq_idx| {
                let seq = input_seqs.get(seq_idx).ok_or_else(|| {
                    inference_tensor::Error::msg(format!(
                        "processed sequence index {seq_idx} exceeds batch size {}",
                        input_seqs.len()
                    ))
                })?;
                let slot_idx = seq.recurrent_state_idx().ok_or_else(|| {
                    inference_tensor::Error::msg(format!(
                        "sequence {} has no recurrent state slot",
                        seq.id()
                    ))
                })?;
                Ok((*seq.id(), slot_idx))
            })
            .collect::<inference_tensor::Result<Vec<_>>>()?;
        hybrid_cache.install_sequence_state_indices(&sequence_slots)
    }

    fn run_paged_forwards(
        &mut self,
        input_seqs: &mut [&mut Sequence],
        ctx: &mut StepCtx<'_>,
        step_inputs: Vec<StepInput>,
        cuda_decode_lookahead: bool,
        speculative_metadata: &PagedAttentionMeta,
    ) -> inference_tensor::Result<PagedForwards> {
        let len_inputs = step_inputs.len();
        let mut outputs = RowOutputs::new(input_seqs.len(), len_inputs);
        let mut batched_causal_logits = None;
        #[cfg(feature = "cuda")]
        let mut batched_cuda_decode = None;
        let mut exec_duration = Duration::ZERO;
        for (i, step_input) in step_inputs.into_iter().enumerate() {
            let StepInput {
                processed,
                recurrent_boundaries,
                prompt_chunk,
                computed_updates,
            } = step_input;
            let InputProcessorOutput {
                inputs,
                seq_indices,
            } = processed.map_err(inference_tensor::Error::msg)?;

            let preserve_causal_generation = self.preserves_causal_generation(
                input_seqs,
                input_seqs.len() > 1 || cuda_decode_lookahead,
                ctx,
            );
            if self.cache().is_hybrid() {
                self.install_recurrent_slots(input_seqs, &seq_indices)?;
            }
            let start = Instant::now();
            #[cfg(feature = "cuda")]
            let forward = if cuda_decode_lookahead {
                self.forward_step(inputs, ctx.return_raw_logits)?
            } else {
                ForwardStepResult::eager(self.forward_inputs(inputs, ctx.return_raw_logits)?)
            };
            #[cfg(not(feature = "cuda"))]
            let forward =
                ForwardStepResult::eager(self.forward_inputs(inputs, ctx.return_raw_logits)?);
            #[cfg(feature = "cuda")]
            let mut cuda_decode = forward.cuda_decode;
            let result = forward
                .output
                .into_cpu_for_batch(input_seqs.len(), preserve_causal_generation)?;
            if let Some(prompt_chunk) = prompt_chunk.as_ref() {
                self.speculative_prompt_chunk(input_seqs, prompt_chunk, speculative_metadata)?;
            }
            for (seq_idx, end) in computed_updates {
                input_seqs[seq_idx].set_num_computed_tokens(end);
            }
            exec_duration += start.elapsed();

            for (seq_idx, end) in recurrent_boundaries {
                self.snapshot_paged_recurrent_prefix(
                    &*input_seqs[seq_idx],
                    ctx.prefix_cacher,
                    speculative_metadata.block_size,
                    end,
                )?;
            }

            let keep_batched_causal_logits = !ctx.is_prompt
                && preserve_causal_generation
                && len_inputs == 1
                && seq_indices.len() == input_seqs.len()
                && seq_indices.iter().copied().eq(0..input_seqs.len());
            let result = match result {
                ForwardInputsResult::CausalGeneration { logits } if keep_batched_causal_logits => {
                    #[cfg(feature = "cuda")]
                    {
                        batched_cuda_decode = cuda_decode.take();
                    }
                    batched_causal_logits = Some(logits);
                    continue;
                }
                result => result,
            };
            outputs.scatter(i, &result, seq_indices)?;
        }
        Ok(PagedForwards {
            outputs,
            batched_causal_logits,
            #[cfg(feature = "cuda")]
            batched_cuda_decode,
            exec_duration,
        })
    }

    /// Samples, or sends the images, audio or token blocks of, the step's generation results.
    async fn dispatch_generation(
        &mut self,
        input_seqs: &mut [&mut Sequence],
        results: Vec<ForwardInputsResult>,
        ctx: &mut StepCtx<'_>,
        speculative: Option<SpeculativeAttempt<'_>>,
    ) -> inference_tensor::Result<()> {
        match &results[0] {
            ForwardInputsResult::RawLogits { .. } | ForwardInputsResult::Embeddings { .. } => {
                unreachable!()
            }
            ForwardInputsResult::CausalGeneration { .. } => {
                let logits = results
                    .into_iter()
                    .map(|r| {
                        let ForwardInputsResult::CausalGeneration { logits } = r else {
                            unreachable!("All results must have same type, `CausalGeneration`")
                        };
                        logits
                    })
                    .collect::<Vec<_>>();
                let sampled_speculatively = match speculative {
                    Some(attempt) => {
                        self.try_sample_speculative_causal_gen(
                            input_seqs,
                            &logits,
                            attempt.batched_logits,
                            ctx.prefix_cacher,
                            ctx.disable_eos_stop,
                            ctx.rng.clone(),
                            attempt.metadata,
                            ctx.logger,
                        )
                        .await?
                    }
                    None => false,
                };
                if !sampled_speculatively {
                    self.sample_causal_gen(
                        input_seqs,
                        logits,
                        ctx.prefix_cacher,
                        ctx.disable_eos_stop,
                        ctx.rng.clone(),
                    )
                    .await?;
                }
            }
            ForwardInputsResult::Image { .. } => {
                let images = results
                    .into_iter()
                    .map(|r| {
                        let ForwardInputsResult::Image { images } = r else {
                            unreachable!("All results must have same type, `Image`")
                        };
                        images
                            .into_iter()
                            .next()
                            .expect("Must have at least 1 element.")
                    })
                    .collect::<Vec<_>>();
                response::send_image_responses(input_seqs, images).await?;
            }
            ForwardInputsResult::Speech { .. } => {
                let mut rates = Vec::with_capacity(results.len());
                let mut channels = Vec::with_capacity(results.len());
                let mut pcms = Vec::with_capacity(results.len());
                for r in results {
                    let ForwardInputsResult::Speech {
                        pcms: row_pcms,
                        rates: row_rates,
                        channels: row_channels,
                    } = r
                    else {
                        unreachable!("All results must have same type, `Speech`")
                    };
                    assert_eq!(row_rates.len(), 1, "Each sequence must have 1 PCM output.");
                    assert_eq!(
                        row_channels.len(),
                        1,
                        "Each sequence must have 1 PCM output."
                    );
                    assert_eq!(row_pcms.len(), 1, "Each sequence must have 1 PCM output.");
                    rates.push(row_rates[0]);
                    channels.push(row_channels[0]);
                    pcms.extend(row_pcms);
                }
                response::send_speech_responses(input_seqs, &pcms, &rates, &channels).await?;
            }
            ForwardInputsResult::Transcription { .. } => {
                let transcripts = results
                    .into_iter()
                    .flat_map(|r| {
                        let ForwardInputsResult::Transcription { transcripts } = r else {
                            unreachable!("All results must have same type, `Transcription`")
                        };
                        transcripts
                    })
                    .collect::<Vec<_>>();
                response::send_transcription_responses(input_seqs, transcripts).await?;
            }
            ForwardInputsResult::VoiceActivity { .. } => {
                let results = results
                    .into_iter()
                    .flat_map(|r| {
                        let ForwardInputsResult::VoiceActivity { results } = r else {
                            unreachable!("All results must have same type, `VoiceActivity`")
                        };
                        results
                    })
                    .collect::<Vec<_>>();
                response::send_voice_activity_responses(input_seqs, results).await?;
            }
            ForwardInputsResult::Diarization { .. } => {
                let results = results
                    .into_iter()
                    .flat_map(|r| {
                        let ForwardInputsResult::Diarization { results } = r else {
                            unreachable!("All results must have same type, `Diarization`")
                        };
                        results
                    })
                    .collect::<Vec<_>>();
                response::send_diarization_responses(input_seqs, results).await?;
            }
            ForwardInputsResult::BlockGeneration { .. } => {
                let mut denoise_times = Vec::with_capacity(results.len());
                let token_blocks = results
                    .into_iter()
                    .map(|r| {
                        let ForwardInputsResult::BlockGeneration {
                            token_blocks,
                            denoise_time,
                        } = r
                        else {
                            unreachable!("All results must have same type, `BlockGeneration`")
                        };
                        denoise_times.push(denoise_time);
                        token_blocks
                            .into_iter()
                            .next()
                            .expect("Must have at least 1 element.")
                    })
                    .collect::<Vec<_>>();
                self.sample_block_gen(
                    input_seqs,
                    token_blocks,
                    denoise_times,
                    ctx.prefix_cacher,
                    ctx.disable_eos_stop,
                )
                .await?;
            }
        }
        Ok(())
    }
}
