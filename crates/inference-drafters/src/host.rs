//! An external drafter attached to a target: the target-agnostic half of drafting with DFlash, DFlash 2 or DSpark.
//! The target implements [`DrafterTarget`] (its last forward's taps, its embeddings and the head it drafts with) and
//! keeps what is its own: tap capture, recurrent rollback and commit order.

use std::sync::Arc;

use inference_nn::speculative::{
    MtpConfig, MtpDraftSamplingMethod, MtpRuntimeConfig, SpeculativeAttachInfo,
    SpeculativeBatchPlan, SpeculativeCommitRow, SpeculativeGraphPlan, SpeculativeKvCache,
    SpeculativePrefillCtx, SpeculativeProposal, SpeculativeProposalBatch,
    SpeculativeProposeBatchCtx, SpeculativeProposePreparation, SpeculativeProposePrepareCtx,
    SpeculativeTapRouting, depths_up_to,
};
use inference_quant::QuantMethod;
use inference_tensor::{Result, Tensor};
use rand::Rng;

#[cfg(all(feature = "cuda", target_family = "unix"))]
use inference_nn::cuda::graph_capture::{CudaGraphComponent, CudaGraphEvent, CudaGraphEventGuard};

use crate::dflash::{
    AUTO_DEPTHS, CtxAppend, DEFAULT_MAX_DRAFTS, DFlashDraftModel, DFlashGraphProposalInputs,
    DFlashLoadTarget, DFlashPagedCache, DFlashPreparedContext, DFlashProposalBatch,
    DFlashSamplingInputs, dflash_adaptive_requested, windowed_kv_capacities,
};

/// What a target gives its external drafter.
pub trait DrafterTarget {
    /// The last forward's taps over the rows its logits came from, `[batch, rows, hidden]` each, in the drafter's
    /// `target_layer_ids` order.
    fn spec_taps(&self) -> Option<Vec<Tensor>>;
    /// The last prompt forward's taps over every row.
    fn prompt_taps(&self) -> Option<Vec<Tensor>>;
    /// Token ids `[batch, rows]` embedded as the target embeds them.
    fn embed_tokens(&self, ids: &Tensor) -> Result<Tensor>;
    fn token_embedding(&self) -> &Arc<dyn QuantMethod>;
    /// The lm_head drafts are scored with: the target's, or a cheaper copy of it.
    fn draft_lm_head(&self) -> Arc<dyn QuantMethod>;
}

/// The target's shape and cache facts an attach sizes the drafter by.
pub struct DrafterAttach<'a> {
    pub load: DFlashLoadTarget<'a>,
    /// Lanes its recurrent rollback reserved (one: the target rolls back without lanes).
    pub checkpoint_lanes: usize,
    pub recurrent_capacity: usize,
    pub serving_capacity: usize,
    /// The first paged-cache layer after the target's own (and any MTP head's).
    pub paged_first_layer: usize,
    pub runtime: MtpRuntimeConfig,
    pub recurrent_pad_slots: usize,
}

pub struct ExternalDrafter {
    drafter: Arc<DFlashDraftModel>,
    n_predict: usize,
}

impl std::ops::Deref for ExternalDrafter {
    type Target = DFlashDraftModel;

    fn deref(&self) -> &DFlashDraftModel {
        &self.drafter
    }
}

struct DFlashPreparedRow {
    seq_id: usize,
    batch_idx: usize,
    start_pos: usize,
    offset: usize,
    rows: usize,
}

struct DFlashProposePreparation {
    context: DFlashPreparedContext,
    rows: Vec<DFlashPreparedRow>,
}

fn dflash_speculative_batch(batch: DFlashProposalBatch) -> Result<SpeculativeProposalBatch> {
    let proposals = match batch {
        DFlashProposalBatch::Tokens(tokens) => {
            tokens.into_iter().map(SpeculativeProposal::new).collect()
        }
        #[cfg(all(feature = "cuda", target_family = "unix"))]
        DFlashProposalBatch::DeviceTokens(tokens) => {
            let batch = tokens.dim(0)?;
            (0..batch)
                .map(|row| SpeculativeProposal::from_device(tokens.get(row)?))
                .collect::<Result<Vec<_>>>()?
        }
        #[cfg(feature = "cuda")]
        DFlashProposalBatch::DeviceSparse {
            tokens,
            candidate_ids,
            candidate_probs,
        } => {
            let batch = tokens.dim(0)?;
            if candidate_ids.dim(0)? != batch || candidate_probs.dim(0)? != batch {
                inference_tensor::bail!("DFlash sparse proposal batch does not match token rows");
            }
            (0..batch)
                .map(|row| {
                    SpeculativeProposal::with_device_sparse_probs(
                        tokens.get(row)?,
                        candidate_ids.get(row)?,
                        candidate_probs.get(row)?,
                    )
                })
                .collect::<Result<Vec<_>>>()?
        }
    };
    Ok(SpeculativeProposalBatch::new(proposals))
}

/// The target's paged cache for a drafter that keeps its layers there.
fn dflash_paged_cache<'a>(
    drafter: &DFlashDraftModel,
    cache: &SpeculativeKvCache<'a>,
) -> Option<DFlashPagedCache<'a>> {
    let SpeculativeKvCache::Paged { metadata, kv_cache } = *cache;
    drafter
        .uses_paged_kv()
        .then_some(DFlashPagedCache { kv_cache, metadata })
}

pub(crate) fn resolve_dflash_n_predict(
    requested: Option<usize>,
    max_drafts: usize,
    checkpoint_lanes: usize,
) -> Result<usize> {
    if max_drafts == 0 {
        inference_tensor::bail!("DFlash drafter yields no drafts per block");
    }
    let configured = requested.unwrap_or(max_drafts.min(DEFAULT_MAX_DRAFTS));
    if configured == 0 || configured > max_drafts {
        inference_tensor::bail!(
            "requested {configured} draft tokens but this DFlash drafter yields at most {max_drafts} per block"
        );
    }
    if checkpoint_lanes == 1 {
        return Ok(configured);
    }
    let reserved = checkpoint_lanes.checked_sub(1).ok_or_else(|| {
        inference_tensor::Error::msg("recurrent checkpoint lane count must be nonzero")
    })?;
    if reserved > max_drafts {
        inference_tensor::bail!(
            "reserved {checkpoint_lanes} recurrent checkpoint lanes but this DFlash drafter yields at most {max_drafts} drafts per block (max {} lanes)",
            max_drafts + 1
        );
    }
    if let Some(requested) = requested
        && requested != reserved
    {
        inference_tensor::bail!(
            "requested {requested} draft tokens require {} recurrent checkpoint lanes, but {checkpoint_lanes} lanes were reserved",
            requested + 1
        );
    }
    Ok(reserved)
}

impl ExternalDrafter {
    /// Loads the drafter `config` names and sizes its context for the target; the target then records the drafter's
    /// `target_layer_ids` as taps and drafts at the returned depth.
    pub fn attach(
        config: &MtpConfig,
        target: DrafterAttach<'_>,
    ) -> Result<(Self, SpeculativeAttachInfo)> {
        let mut drafter = DFlashDraftModel::load(config, target.load, false)?;
        let n_predict = resolve_dflash_n_predict(
            config.n_predict,
            drafter.max_drafts(),
            target.checkpoint_lanes,
        )?;
        let (sequence_capacity, retained_prefixes) = windowed_kv_capacities(
            target.serving_capacity,
            target.recurrent_capacity,
            target.runtime.prefix_cache_capacity(),
        );
        let windowed_kv = drafter.enable_windowed_kv(sequence_capacity, retained_prefixes)?;
        let paged_kv = !windowed_kv && drafter.enable_paged_kv(target.paged_first_layer);
        let adaptive = config.n_predict.is_none() && windowed_kv && dflash_adaptive_requested();
        let max_live_sequences = sequence_capacity.saturating_sub(target.recurrent_pad_slots);
        let adaptive = adaptive && drafter.enable_adaptive(n_predict, max_live_sequences);
        let draft_sampling = match drafter.draft_sampling_method() {
            MtpDraftSamplingMethod::Auto => {
                unreachable!("DFlash draft sampling is resolved during loading")
            }
            MtpDraftSamplingMethod::Greedy => "greedy draft",
            MtpDraftSamplingMethod::Probabilistic => "probabilistic draft",
        };
        let context = match (windowed_kv, paged_kv) {
            (true, _) => "windowed",
            (false, true) => "paged",
            (false, false) => "eager",
        };
        let depth = if adaptive {
            format!("batch-adaptive depth <= {n_predict}")
        } else {
            format!("depth {n_predict}")
        };
        let name = format!(
            "{} `{}` (block of {}, {depth}, {draft_sampling}, {context} context, taps {:?})",
            drafter.kind(),
            config.model.as_deref().unwrap_or("dflash"),
            drafter.max_drafts(),
            drafter.target_layer_ids
        );
        let info = SpeculativeAttachInfo::mtp(name, n_predict);
        Ok((
            Self {
                drafter: Arc::new(drafter),
                n_predict,
            },
            info,
        ))
    }

    pub fn n_predict(&self) -> usize {
        self.n_predict
    }

    pub fn plan(&self, n: usize, batch_size: usize) -> SpeculativeBatchPlan {
        SpeculativeBatchPlan::new(self.drafter.plan_n(n, batch_size)).without_target_hiddens()
    }

    pub fn depth_candidates(n: usize) -> Vec<usize> {
        depths_up_to(&AUTO_DEPTHS, n)
    }

    pub fn graph_plans(&self, n: usize) -> Vec<SpeculativeGraphPlan> {
        self.drafter.graph_plans(n)
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    pub fn precapture(&self, target: &dyn DrafterTarget, n: usize) -> Result<()> {
        self.drafter
            .precapture_cuda_graphs(n, target.token_embedding(), &target.draft_lm_head())
    }

    /// The noise rows `[anchor, mask, ...]` that draft `n` tokens, `[batch, rows, hidden]`, embedded in one call.
    fn noise_embedding(
        &self,
        target: &dyn DrafterTarget,
        anchors: &[u32],
        n: usize,
        device: &inference_tensor::Device,
    ) -> Result<Tensor> {
        let block = self.drafter.query_rows(n);
        let mut ids = Vec::with_capacity(anchors.len() * block);
        for anchor in anchors {
            ids.push(*anchor);
            ids.extend(std::iter::repeat_n(self.drafter.mask_token_id(), block - 1));
        }
        let ids = Tensor::from_vec(ids, (anchors.len(), block), device)?;
        let mut emb = self
            .drafter
            .with_own_mask_rows(target.embed_tokens(&ids)?)?;
        let scale = self.drafter.input_embedding_scale();
        if (scale - 1.0).abs() > f64::EPSILON {
            emb = (emb * scale)?;
        }
        Ok(emb)
    }

    pub fn prepare_propose(
        &self,
        target: &dyn DrafterTarget,
        ctx: SpeculativeProposePrepareCtx<'_>,
    ) -> Result<Option<Box<dyn SpeculativeProposePreparation>>> {
        let drafter = &self.drafter;
        if ctx.seq_ids.is_empty() {
            return Ok(None);
        }
        if ctx.seq_ids.len() != ctx.base_lens.len()
            || ctx.seq_ids.len() != ctx.target_rows.len()
            || drafter.has_dormant_seq(ctx.seq_ids)
        {
            return Ok(None);
        }
        let Some(taps) = target.spec_taps() else {
            return Ok(None);
        };
        if taps.len() != drafter.target_layer_ids.len() {
            return Ok(None);
        }
        let (source_batch, source_rows, _) = taps[0].dims3()?;
        let mut appends = Vec::with_capacity(ctx.seq_ids.len());
        let mut rows = Vec::with_capacity(ctx.seq_ids.len());
        let mut flat_row_indices = Vec::new();
        for ((seq_id, base_len), &(batch_idx, count)) in
            ctx.seq_ids.iter().zip(ctx.base_lens).zip(ctx.target_rows)
        {
            if count == 0
                || batch_idx >= source_batch
                || count > source_rows
                || drafter.ctx_next_pos(*seq_id) != Some(*base_len)
            {
                return Ok(None);
            }
            let flat_start = batch_idx
                .checked_mul(source_rows)
                .ok_or_else(|| inference_tensor::Error::msg("DFlash tap row index overflow"))?;
            let offset = flat_row_indices.len();
            for row in flat_start..flat_start + count {
                flat_row_indices.push(u32::try_from(row).map_err(inference_tensor::Error::wrap)?);
            }
            appends.push(CtxAppend {
                seq_id: *seq_id,
                rows: count,
                start_pos: *base_len,
            });
            rows.push(DFlashPreparedRow {
                seq_id: *seq_id,
                batch_idx,
                start_pos: *base_len,
                offset,
                rows: count,
            });
        }
        let context = drafter.prepare_ctx_batch(&taps, flat_row_indices, &appends)?;
        Ok(Some(Box::new(DFlashProposePreparation { context, rows })))
    }

    /// Drafts up to `max_n` tokens a sequence, after appending the accepted rows' taps to the drafter's context.
    pub fn propose(
        &self,
        target: &dyn DrafterTarget,
        ctx: SpeculativeProposeBatchCtx<'_>,
        max_n: usize,
        device: &inference_tensor::Device,
    ) -> Result<Option<SpeculativeProposalBatch>> {
        let drafter = &self.drafter;
        let batch = ctx.seq_ids.len();
        if batch == 0 || max_n == 0 {
            return Ok(None);
        }
        let n_predict = ctx.proposal_len;
        if n_predict == 0 {
            return Ok(None);
        }
        if n_predict > max_n {
            inference_tensor::bail!(
                "DFlash proposal length {n_predict} exceeds configured maximum {max_n}"
            );
        }
        let Some(taps) = target.spec_taps() else {
            return Ok(None);
        };
        if taps.len() != drafter.target_layer_ids.len() {
            return Ok(None);
        }
        if drafter.has_dormant_seq(ctx.seq_ids) {
            return Ok(None);
        }

        let prepared = ctx.preparation.and_then(|preparation| {
            preparation
                .as_any()
                .downcast_ref::<DFlashProposePreparation>()
        });
        let prepared_commit = prepared.and_then(|prepared| {
            let mut appends = Vec::with_capacity(batch);
            let mut row_indices = Vec::new();
            for (i, seq_id) in ctx.seq_ids.iter().enumerate() {
                let (batch_idx, count) = ctx.target_rows[i];
                let base_len = ctx.base_lens[i];
                let row = prepared
                    .rows
                    .iter()
                    .find(|row| row.seq_id == *seq_id && row.batch_idx == batch_idx)?;
                if count == 0
                    || count > row.rows
                    || row.start_pos.checked_add(count) != Some(base_len)
                    || drafter.ctx_next_pos(*seq_id) != Some(row.start_pos)
                {
                    return None;
                }
                for index in row.offset..row.offset + count {
                    row_indices.push(u32::try_from(index).ok()?);
                }
                appends.push(CtxAppend {
                    seq_id: *seq_id,
                    rows: count,
                    start_pos: row.start_pos,
                });
            }
            Some((prepared, row_indices, appends))
        });
        if let Some((prepared, row_indices, appends)) = prepared_commit {
            drafter.commit_prepared_ctx_batch(
                &prepared.context,
                row_indices,
                &appends,
                dflash_paged_cache(drafter, &ctx.cache),
            )?;
        } else {
            let source_rows = taps[0].dim(1)?;
            let mut appends = Vec::with_capacity(batch);
            let mut flat_row_indices = Vec::new();
            for (i, seq_id) in ctx.seq_ids.iter().enumerate() {
                let (batch_idx, count) = ctx.target_rows[i];
                let base_len = ctx.base_lens[i];
                let ctx_next = drafter.ctx_next_pos(*seq_id);
                let needed = match ctx_next {
                    Some(next) if next <= base_len => base_len - next,
                    _ => count.min(base_len),
                };
                if needed > 0 {
                    if needed > count || source_rows < count {
                        return Ok(None);
                    }
                    let start_row = count - needed;
                    let flat_start = batch_idx
                        .checked_mul(source_rows)
                        .and_then(|row| row.checked_add(start_row))
                        .ok_or_else(|| {
                            inference_tensor::Error::msg("DFlash tap row index overflow")
                        })?;
                    let flat_end = flat_start.checked_add(needed).ok_or_else(|| {
                        inference_tensor::Error::msg("DFlash tap row index overflow")
                    })?;
                    for row in flat_start..flat_end {
                        flat_row_indices
                            .push(u32::try_from(row).map_err(inference_tensor::Error::wrap)?);
                    }
                    appends.push(CtxAppend {
                        seq_id: *seq_id,
                        rows: needed,
                        start_pos: base_len - needed,
                    });
                }
            }
            drafter.append_ctx_batch(
                &taps,
                flat_row_indices,
                &appends,
                dflash_paged_cache(drafter, &ctx.cache),
            )?;
        }
        if !drafter.contexts_ready_for_draft(ctx.seq_ids) {
            return Ok(None);
        }
        let sampling_values = if drafter.has_selector()
            && drafter.draft_sampling_method() == MtpDraftSamplingMethod::Probabilistic
        {
            let probabilistic_rows = ctx
                .sequences
                .iter()
                .map(|seq| seq.stochastic_verification_allowed())
                .collect::<Vec<_>>();
            if probabilistic_rows.iter().any(|eligible| *eligible) {
                let mut inverse_temperatures = Vec::with_capacity(batch);
                for (seq, eligible) in ctx.sequences.iter().zip(&probabilistic_rows) {
                    if !*eligible {
                        inverse_temperatures.push(0.0);
                        continue;
                    }
                    let Some(temperature) = seq.sampler().temperature() else {
                        return Ok(None);
                    };
                    #[allow(clippy::cast_possible_truncation)]
                    let inverse_temperature = (1.0 / temperature) as f32;
                    if !inverse_temperature.is_finite() || inverse_temperature <= 0.0 {
                        return Ok(None);
                    }
                    inverse_temperatures.push(inverse_temperature);
                }
                let mut uniforms = vec![0.0f32; batch * n_predict];
                for (row, inverse_temperature) in inverse_temperatures.iter().enumerate() {
                    if *inverse_temperature > 0.0 {
                        let rng = ctx.sequences[row].sampling_rng(&ctx.rng);
                        let mut rng = rng.lock().expect("could not lock rng mutex");
                        for uniform in &mut uniforms[row * n_predict..(row + 1) * n_predict] {
                            *uniform = rng.random();
                        }
                    }
                }
                Some((inverse_temperatures, uniforms))
            } else {
                None
            }
        } else {
            None
        };
        let sampling = sampling_values
            .as_ref()
            .map(|(inverse_temperatures, uniforms)| DFlashSamplingInputs {
                inverse_temperatures,
                uniforms,
            });
        let lm_head = target.draft_lm_head();
        let graph_proposals = drafter.proposals_cuda_graph(&DFlashGraphProposalInputs {
            seq_ids: ctx.seq_ids,
            anchors: ctx.sampled_tokens,
            start_positions: ctx.base_lens,
            n_predict,
            sampling,
            token_embedding: target.token_embedding(),
            lm_head: &lm_head,
        })?;
        if let Some(proposals) = graph_proposals {
            return dflash_speculative_batch(proposals).map(Some);
        }
        #[cfg(all(feature = "cuda", target_family = "unix"))]
        let graph_event =
            CudaGraphEventGuard::new(CudaGraphComponent::DFlash, CudaGraphEvent::EagerFallback);
        let fallback_result = (|| {
            let noise = self.noise_embedding(target, ctx.sampled_tokens, n_predict, device)?;
            let hidden = drafter.draft_hidden_batch(
                ctx.seq_ids,
                &noise,
                ctx.base_lens,
                dflash_paged_cache(drafter, &ctx.cache),
            )?;
            dflash_speculative_batch(drafter.finish_proposals(
                &hidden,
                ctx.sampled_tokens,
                sampling,
                &lm_head,
            )?)
            .map(Some)
        })();
        #[cfg(all(feature = "cuda", target_family = "unix"))]
        if fallback_result.is_ok() {
            graph_event.success();
        }
        fallback_result
    }

    /// Appends a prompt chunk's taps to the drafter's context.
    pub fn prefill(
        &self,
        target: &dyn DrafterTarget,
        ctx: SpeculativePrefillCtx<'_>,
    ) -> Result<()> {
        let drafter = &self.drafter;
        let Some(taps) = target.prompt_taps() else {
            return Ok(());
        };
        if taps.len() != drafter.target_layer_ids.len() {
            return Ok(());
        }
        let (capture_batch, capture_rows, _) = taps[0].dims3()?;
        let routing = SpeculativeTapRouting::new(
            ctx.capture_layout(),
            capture_batch,
            capture_rows,
            ctx.batch_indices,
            ctx.chunk_ranges,
        )?;
        drafter.activate_seqs(ctx.seq_ids);
        let mut appends = Vec::with_capacity(ctx.seq_ids.len());
        for ((seq_id, &(start, _)), span) in ctx
            .seq_ids
            .iter()
            .zip(ctx.chunk_ranges)
            .zip(routing.spans())
        {
            appends.push(CtxAppend {
                seq_id: *seq_id,
                rows: span.rows(),
                start_pos: start,
            });
        }
        let flat_row_indices = routing.flat_row_indices()?;
        drafter.append_ctx_batch(
            &taps,
            flat_row_indices,
            &appends,
            dflash_paged_cache(drafter, &ctx.cache),
        )
    }

    /// A finished sequence never drafts again, which is where its kept rows would reach the drafter's context; its
    /// final snapshot needs the drafter at the target's position.
    pub fn commit_terminal(
        &self,
        target: &dyn DrafterTarget,
        rows: &[SpeculativeCommitRow],
    ) -> Result<()> {
        let drafter = &self.drafter;
        if !rows.iter().any(|row| row.terminal) || !drafter.supports_paged_auxiliary_prefix_state()
        {
            return Ok(());
        }
        let Some(taps) = target.spec_taps() else {
            return Ok(());
        };
        if taps.len() != drafter.target_layer_ids.len() {
            return Ok(());
        }
        let source_rows = taps[0].dim(1)?;
        let mut appends = Vec::new();
        let mut flat_row_indices = Vec::new();
        for row in rows.iter().filter(|row| row.terminal) {
            let Some(start_pos) = drafter.ctx_next_pos(row.seq_id) else {
                continue;
            };
            if row.keep_rows > source_rows {
                continue;
            }
            let flat_start = row.batch_idx * source_rows;
            for flat_row in flat_start..flat_start + row.keep_rows {
                flat_row_indices
                    .push(u32::try_from(flat_row).map_err(inference_tensor::Error::wrap)?);
            }
            appends.push(CtxAppend {
                seq_id: row.seq_id,
                rows: row.keep_rows,
                start_pos,
            });
        }
        drafter.append_ctx_batch(&taps, flat_row_indices, &appends, None)
    }
}

/// The prefix-cache hooks of `SpeculativeTargetMixin` for a target with an external drafter, which keeps that state:
/// `$drafter` reads the attached drafter from `self` as an `Option<Arc<ExternalDrafter>>`.
#[macro_export]
macro_rules! external_drafter_prefix_hooks {
    (|$this:ident| $drafter:expr) => {
        fn speculative_prefix_replay(
            &self,
        ) -> ::inference_nn::speculative::SpeculativePrefixReplay {
            let $this = self;
            $drafter.map_or(
                ::inference_nn::speculative::SpeculativePrefixReplay::NotRequired,
                |drafter| drafter.prefix_replay(),
            )
        }

        fn supports_paged_auxiliary_prefix_state(&self) -> bool {
            let $this = self;
            $drafter.is_some_and(|drafter| drafter.supports_paged_auxiliary_prefix_state())
        }

        fn capture_paged_auxiliary_prefix_state(
            &mut self,
            sequence_id: usize,
            cached_tokens: usize,
        ) -> ::inference_tensor::Result<
            Option<::std::sync::Arc<dyn ::inference_nn::kv_cache::PagedAuxiliaryPrefixState>>,
        > {
            let $this = &*self;
            match $drafter {
                Some(drafter) => {
                    drafter.capture_paged_auxiliary_prefix_state(sequence_id, cached_tokens)
                }
                None => Ok(None),
            }
        }

        fn restore_paged_auxiliary_prefix_state(
            &mut self,
            sequence_id: usize,
            cached_tokens: usize,
            state: &dyn ::inference_nn::kv_cache::PagedAuxiliaryPrefixState,
        ) -> ::inference_tensor::Result<()> {
            let $this = &*self;
            let drafter = $drafter.ok_or_else(|| {
                ::inference_tensor::Error::msg("DFlash prefix restore without a drafter")
            })?;
            drafter.restore_paged_auxiliary_prefix_state(sequence_id, cached_tokens, state)
        }

        fn export_paged_auxiliary_prefix_state(
            &mut self,
            state: &dyn ::inference_nn::kv_cache::PagedAuxiliaryPrefixState,
        ) -> ::inference_tensor::Result<Option<::inference_nn::kv_cache::PagedAuxiliaryPrefixData>>
        {
            let $this = &*self;
            $drafter.map_or(Ok(None), |drafter| {
                drafter.export_paged_auxiliary_prefix_state(state)
            })
        }

        fn import_paged_auxiliary_prefix_state(
            &mut self,
            data: &::inference_nn::kv_cache::PagedAuxiliaryPrefixData,
        ) -> ::inference_tensor::Result<
            Option<::std::sync::Arc<dyn ::inference_nn::kv_cache::PagedAuxiliaryPrefixState>>,
        > {
            let $this = &*self;
            $drafter.map_or(Ok(None), |drafter| {
                drafter.import_paged_auxiliary_prefix_state(data)
            })
        }

        fn paged_auxiliary_prefix_signature(&self) -> Option<String> {
            let $this = self;
            $drafter.and_then(|drafter| drafter.paged_auxiliary_prefix_signature())
        }
    };
}

#[cfg(test)]
mod tests {
    use super::resolve_dflash_n_predict;

    #[test]
    fn auto_dflash_depth_follows_reserved_checkpoint_lanes() {
        assert_eq!(resolve_dflash_n_predict(None, 7, 8).unwrap(), 7);
        assert_eq!(resolve_dflash_n_predict(None, 7, 4).unwrap(), 3);
    }

    #[test]
    fn replay_sentinel_keeps_configured_or_default_dflash_depth() {
        assert_eq!(resolve_dflash_n_predict(None, 7, 1).unwrap(), 7);
        assert_eq!(resolve_dflash_n_predict(Some(3), 7, 1).unwrap(), 3);
    }

    #[test]
    fn explicit_dflash_depth_must_match_reserved_checkpoint_lanes() {
        assert_eq!(resolve_dflash_n_predict(Some(3), 7, 4).unwrap(), 3);
        let error = resolve_dflash_n_predict(Some(3), 7, 8)
            .expect_err("explicit depth must match the reserved lanes");
        assert!(error.to_string().contains(
            "requested 3 draft tokens require 4 recurrent checkpoint lanes, but 8 lanes were reserved"
        ));
    }

    #[test]
    fn reserved_dflash_depth_must_fit_the_drafter_block() {
        let error = resolve_dflash_n_predict(None, 7, 9)
            .expect_err("reserved lanes must fit the drafter block");
        assert!(
            error
                .to_string()
                .contains("reserved 9 recurrent checkpoint lanes")
        );
    }
}
