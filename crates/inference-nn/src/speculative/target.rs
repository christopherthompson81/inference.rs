use std::sync::Arc;

use candle_core::{Result, Tensor};

use crate::kv_cache::PagedAuxiliaryPrefixState;

use super::{
    SpeculativeAttachInfo, SpeculativeBatchObservation, SpeculativeBatchPlan, SpeculativeCommitRow,
    SpeculativeConfig, SpeculativeGraphPlan, SpeculativePrefillCtx, SpeculativeProposalBatch,
    SpeculativeProposeBatchCtx, SpeculativeProposePreparation, SpeculativeProposePrepareCtx,
    logging::log_attach,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpeculativePrefixReplay {
    #[default]
    NotRequired,
    Suffix(usize),
    Full,
}

impl SpeculativePrefixReplay {
    pub fn replay_tokens(self, cached_tokens: usize) -> usize {
        match self {
            Self::NotRequired => 0,
            Self::Suffix(tokens) => tokens.min(cached_tokens),
            Self::Full => cached_tokens,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpeculativePrefixCheckpointPolicy {
    fallback_replay: SpeculativePrefixReplay,
    text_auxiliary_state: bool,
}

impl SpeculativePrefixCheckpointPolicy {
    pub fn new(fallback_replay: SpeculativePrefixReplay, text_auxiliary_state: bool) -> Self {
        Self {
            fallback_replay,
            text_auxiliary_state,
        }
    }

    pub fn replay_for(self, modality_signature: u8) -> SpeculativePrefixReplay {
        if self.uses_auxiliary_state(modality_signature) {
            SpeculativePrefixReplay::NotRequired
        } else {
            self.fallback_replay
        }
    }

    pub fn fallback_replay(self) -> SpeculativePrefixReplay {
        self.fallback_replay
    }

    pub fn uses_auxiliary_state(self, modality_signature: u8) -> bool {
        self.text_auxiliary_state && modality_signature == 0
    }
}

pub fn clamp_speculative_prefix_cache_hit(
    cached_tokens: usize,
    block_size: usize,
    replay: SpeculativePrefixReplay,
) -> usize {
    let retained = match replay {
        SpeculativePrefixReplay::NotRequired => return cached_tokens,
        SpeculativePrefixReplay::Suffix(tokens) => cached_tokens.saturating_sub(tokens),
        SpeculativePrefixReplay::Full => 0,
    };
    retained - retained % block_size
}

/// Everything a target forward leaves behind for the proposer/commit (captured hidden states, rollback
/// stashes). A CUDA graph replay never runs the forward, so the pipeline copies these into persistent
/// buffers at capture time and re-installs them after every replay.
pub trait SpeculativeGraphState: Send + Sync {
    /// Device tensors in a fixed order; `with_tensors` rebuilds the same structure around replacements.
    fn tensors(&self) -> Vec<Tensor>;
    fn with_tensors(&self, tensors: Vec<Tensor>) -> Result<Box<dyn SpeculativeGraphState>>;
    /// Build views for the live rows before launching a padded CUDA graph.
    fn for_real_batch(&self, real_batch: usize) -> Result<Box<dyn SpeculativeGraphState>>;
    fn as_any(&self) -> &dyn std::any::Any;
}

pub trait SpeculativeTargetMixin {
    fn attach_speculative(
        &mut self,
        config: SpeculativeConfig,
    ) -> Result<Option<SpeculativeAttachInfo>> {
        match config {
            SpeculativeConfig::Off => Ok(None),
            _ => candle_core::bail!("This model does not support speculative decoding."),
        }
    }

    #[doc(hidden)]
    fn attach_speculative_with_runtime(
        &mut self,
        config: SpeculativeConfig,
        _runtime: super::MtpRuntimeConfig,
    ) -> Result<Option<SpeculativeAttachInfo>> {
        self.attach_speculative(config)
    }

    fn log_speculative_attach(&self, info: &SpeculativeAttachInfo) {
        log_attach(info);
    }

    fn has_speculative_proposer(&self) -> bool {
        false
    }

    fn supports_recurrent_speculative_checkpoints(&self) -> bool {
        false
    }

    fn supports_recurrent_speculative_transitions(&self) -> bool {
        false
    }

    fn reserve_recurrent_speculative_transition_storage(&self) -> Result<bool> {
        Ok(false)
    }

    fn reserve_recurrent_decode_deferred_storage(&self) -> Result<bool> {
        Ok(false)
    }

    fn disable_recurrent_decode_deferred_storage(&self) -> Result<bool> {
        Ok(false)
    }

    fn apply_recurrent_speculative_transitions_for_current_batch(&self) -> Result<bool> {
        Ok(false)
    }

    fn flush_recurrent_state_for_current_batch(&self) -> Result<()> {
        Ok(())
    }

    fn flush_recurrent_speculative_transitions(&self, _seq_ids: &[usize]) -> Result<()> {
        Ok(())
    }

    fn supports_speculative_prompt_bootstrap(&self) -> bool {
        false
    }

    fn supports_speculative_packed_prefill(&self) -> bool {
        false
    }

    fn speculative_prefix_replay(&self) -> SpeculativePrefixReplay {
        SpeculativePrefixReplay::NotRequired
    }

    fn supports_paged_auxiliary_prefix_state(&self) -> bool {
        false
    }

    fn capture_paged_auxiliary_prefix_state(
        &mut self,
        _sequence_id: usize,
        _cached_tokens: usize,
    ) -> Result<Option<Arc<dyn PagedAuxiliaryPrefixState>>> {
        Ok(None)
    }

    fn restore_paged_auxiliary_prefix_state(
        &mut self,
        _sequence_id: usize,
        _cached_tokens: usize,
        _state: &dyn PagedAuxiliaryPrefixState,
    ) -> Result<()> {
        candle_core::bail!("This model does not support auxiliary paged prefix state.")
    }

    fn speculative_plan(&self, _batch_size: usize) -> Option<SpeculativeBatchPlan> {
        None
    }

    fn speculative_graph_plans(&self) -> Vec<SpeculativeGraphPlan> {
        self.speculative_plan(1)
            .map(|plan| SpeculativeGraphPlan::new(plan.proposal_len, None))
            .into_iter()
            .collect()
    }

    fn precapture_speculative_cuda_graphs(&self) -> Result<()> {
        Ok(())
    }

    fn evict_speculative_cuda_graphs(&self, _max_entries: usize) -> usize {
        0
    }

    fn speculative_observe(&self, _observation: SpeculativeBatchObservation) {}

    fn speculative_bypass(&mut self, seq_ids: &[usize]) -> Result<()> {
        self.flush_recurrent_speculative_transitions(seq_ids)
    }

    fn release_speculative_sequences(&mut self, seq_ids: &[usize]) -> Result<()> {
        self.flush_recurrent_speculative_transitions(seq_ids)
    }

    /// Returns `Ok(None)` when speculation is unsupported for the current step.
    /// Return `Err` only for real failures that should stop generation.
    fn speculative_propose(
        &mut self,
        _ctx: SpeculativeProposeBatchCtx<'_>,
    ) -> Result<Option<SpeculativeProposalBatch>> {
        Ok(None)
    }

    fn speculative_prepare_propose(
        &mut self,
        _ctx: SpeculativeProposePrepareCtx<'_>,
    ) -> Result<Option<Box<dyn SpeculativeProposePreparation>>> {
        Ok(None)
    }

    /// Returns `Ok(None)` when the active proposer does not need target hidden state.
    /// Return `Err` only when hidden state was expected but unavailable or invalid.
    fn speculative_target_hiddens(&self, _rows: &[(usize, usize)]) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// Called after each prompt chunk so proposers with their own KV cache can process it.
    fn speculative_prefill(&mut self, _ctx: SpeculativePrefillCtx<'_>) -> Result<()> {
        Ok(())
    }

    /// Called once verification decided which rows of the last multi-token step survive, so models
    /// with state that is not a paged KV cache (recurrent layers) can roll rejected rows back.
    fn speculative_commit(&mut self, _rows: &[SpeculativeCommitRow]) -> Result<()> {
        Ok(())
    }

    /// Detach what the last forward left for the proposer. `None` means the model cannot be replayed
    /// through a CUDA graph while a proposer is attached.
    fn take_speculative_graph_state(&self) -> Option<Box<dyn SpeculativeGraphState>> {
        None
    }

    fn install_speculative_graph_state(&self, _state: &dyn SpeculativeGraphState) -> Result<()> {
        Ok(())
    }
}

// Forwards every method, defaults included, to the wrapped model; a method added to the trait has to be added here.
#[macro_export]
macro_rules! delegate_speculative_target {
    ($wrapper:ty, $field:ident: $inner:ty) => {
        impl $crate::speculative::SpeculativeTargetMixin for $wrapper {
            fn attach_speculative(
                &mut self,
                config: $crate::speculative::SpeculativeConfig,
            ) -> ::candle_core::Result<Option<$crate::speculative::SpeculativeAttachInfo>> {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::attach_speculative(
                    &mut self.$field,
                    config,
                )
            }

            fn attach_speculative_with_runtime(
                &mut self,
                config: $crate::speculative::SpeculativeConfig,
                runtime: $crate::speculative::MtpRuntimeConfig,
            ) -> ::candle_core::Result<Option<$crate::speculative::SpeculativeAttachInfo>> {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::attach_speculative_with_runtime(
                    &mut self.$field,
                    config,
                    runtime,
                )
            }

            fn log_speculative_attach(&self, info: &$crate::speculative::SpeculativeAttachInfo) {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::log_speculative_attach(
                    &self.$field,
                    info,
                )
            }

            fn has_speculative_proposer(&self) -> bool {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::has_speculative_proposer(
                    &self.$field,
                )
            }

            fn supports_recurrent_speculative_checkpoints(&self) -> bool {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::supports_recurrent_speculative_checkpoints(
                            &self.$field,
                        )
            }

            fn supports_recurrent_speculative_transitions(&self) -> bool {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::supports_recurrent_speculative_transitions(
                            &self.$field,
                        )
            }

            fn reserve_recurrent_speculative_transition_storage(&self) -> ::candle_core::Result<bool> {
                <$inner as $crate::speculative::SpeculativeTargetMixin>
                    ::reserve_recurrent_speculative_transition_storage(&self.$field)
            }

            fn reserve_recurrent_decode_deferred_storage(&self) -> ::candle_core::Result<bool> {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::reserve_recurrent_decode_deferred_storage(
                            &self.$field,
                        )
            }

            fn disable_recurrent_decode_deferred_storage(&self) -> ::candle_core::Result<bool> {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::disable_recurrent_decode_deferred_storage(
                            &self.$field,
                        )
            }

            fn apply_recurrent_speculative_transitions_for_current_batch(
                &self,
            ) -> ::candle_core::Result<bool> {
                <$inner as $crate::speculative::SpeculativeTargetMixin>
                    ::apply_recurrent_speculative_transitions_for_current_batch(&self.$field)
            }

            fn flush_recurrent_state_for_current_batch(&self) -> ::candle_core::Result<()> {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::flush_recurrent_state_for_current_batch(
                            &self.$field,
                        )
            }

            fn flush_recurrent_speculative_transitions(
                &self,
                seq_ids: &[usize],
            ) -> ::candle_core::Result<()> {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::flush_recurrent_speculative_transitions(
                            &self.$field, seq_ids,
                        )
            }

            fn supports_speculative_prompt_bootstrap(&self) -> bool {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::supports_speculative_prompt_bootstrap(
                            &self.$field,
                        )
            }

            fn supports_speculative_packed_prefill(&self) -> bool {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::supports_speculative_packed_prefill(
                    &self.$field,
                )
            }

            fn speculative_prefix_replay(&self) -> $crate::speculative::SpeculativePrefixReplay {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::speculative_prefix_replay(
                    &self.$field,
                )
            }

            fn supports_paged_auxiliary_prefix_state(&self) -> bool {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::supports_paged_auxiliary_prefix_state(
                            &self.$field,
                        )
            }

            fn capture_paged_auxiliary_prefix_state(
                &mut self,
                sequence_id: usize,
                cached_tokens: usize,
            ) -> ::candle_core::Result<
                Option<::std::sync::Arc<dyn $crate::kv_cache::PagedAuxiliaryPrefixState>>,
            > {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::capture_paged_auxiliary_prefix_state(
                            &mut self.$field,
                            sequence_id,
                            cached_tokens,
                        )
            }

            fn restore_paged_auxiliary_prefix_state(
                &mut self,
                sequence_id: usize,
                cached_tokens: usize,
                state: &dyn $crate::kv_cache::PagedAuxiliaryPrefixState,
            ) -> ::candle_core::Result<()> {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::restore_paged_auxiliary_prefix_state(
                            &mut self.$field,
                            sequence_id,
                            cached_tokens,
                            state,
                        )
            }

            fn speculative_plan(
                &self,
                batch_size: usize,
            ) -> Option<$crate::speculative::SpeculativeBatchPlan> {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::speculative_plan(
                    &self.$field,
                    batch_size,
                )
            }

            fn speculative_graph_plans(&self) -> Vec<$crate::speculative::SpeculativeGraphPlan> {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::speculative_graph_plans(
                    &self.$field,
                )
            }

            fn precapture_speculative_cuda_graphs(&self) -> ::candle_core::Result<()> {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::precapture_speculative_cuda_graphs(
                    &self.$field,
                )
            }

            fn evict_speculative_cuda_graphs(&self, max_entries: usize) -> usize {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::evict_speculative_cuda_graphs(
                    &self.$field,
                    max_entries,
                )
            }

            fn speculative_observe(&self, observation: $crate::speculative::SpeculativeBatchObservation) {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::speculative_observe(
                    &self.$field,
                    observation,
                )
            }

            fn speculative_bypass(&mut self, seq_ids: &[usize]) -> ::candle_core::Result<()> {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::speculative_bypass(
                    &mut self.$field,
                    seq_ids,
                )
            }

            fn release_speculative_sequences(&mut self, seq_ids: &[usize]) -> ::candle_core::Result<()> {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::release_speculative_sequences(
                    &mut self.$field,
                    seq_ids,
                )
            }

            fn speculative_propose(
                &mut self,
                ctx: $crate::speculative::SpeculativeProposeBatchCtx<'_>,
            ) -> ::candle_core::Result<Option<$crate::speculative::SpeculativeProposalBatch>> {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::speculative_propose(
                    &mut self.$field,
                    ctx,
                )
            }

            fn speculative_prepare_propose(
                &mut self,
                ctx: $crate::speculative::SpeculativeProposePrepareCtx<'_>,
            ) -> ::candle_core::Result<Option<Box<dyn $crate::speculative::SpeculativeProposePreparation>>>
            {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::speculative_prepare_propose(
                    &mut self.$field,
                    ctx,
                )
            }

            fn speculative_target_hiddens(
                &self,
                rows: &[(usize, usize)],
            ) -> ::candle_core::Result<Option<::candle_core::Tensor>> {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::speculative_target_hiddens(
                    &self.$field,
                    rows,
                )
            }

            fn speculative_prefill(
                &mut self,
                ctx: $crate::speculative::SpeculativePrefillCtx<'_>,
            ) -> ::candle_core::Result<()> {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::speculative_prefill(
                    &mut self.$field,
                    ctx,
                )
            }

            fn speculative_commit(
                &mut self,
                rows: &[$crate::speculative::SpeculativeCommitRow],
            ) -> ::candle_core::Result<()> {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::speculative_commit(
                    &mut self.$field,
                    rows,
                )
            }

            fn take_speculative_graph_state(
                &self,
            ) -> Option<Box<dyn $crate::speculative::SpeculativeGraphState>> {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::take_speculative_graph_state(
                    &self.$field,
                )
            }

            fn install_speculative_graph_state(
                &self,
                state: &dyn $crate::speculative::SpeculativeGraphState,
            ) -> ::candle_core::Result<()> {
                <$inner as $crate::speculative::SpeculativeTargetMixin>::install_speculative_graph_state(
                    &self.$field,
                    state,
                )
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use super::{
        SpeculativePrefixReplay, SpeculativeTargetMixin, clamp_speculative_prefix_cache_hit,
    };

    struct NoSpeculativeProposer;

    impl SpeculativeTargetMixin for NoSpeculativeProposer {}

    struct TransitionTarget {
        flushed: Rc<RefCell<Vec<Vec<usize>>>>,
    }

    impl SpeculativeTargetMixin for TransitionTarget {
        fn flush_recurrent_speculative_transitions(
            &self,
            seq_ids: &[usize],
        ) -> candle_core::Result<()> {
            self.flushed.borrow_mut().push(seq_ids.to_vec());
            Ok(())
        }
    }

    #[test]
    fn prefix_replay_clamp_preserves_block_alignment() {
        assert_eq!(
            clamp_speculative_prefix_cache_hit(4096, 32, SpeculativePrefixReplay::NotRequired),
            4096
        );
        assert_eq!(
            clamp_speculative_prefix_cache_hit(4096, 32, SpeculativePrefixReplay::Suffix(2048)),
            2048
        );
        assert_eq!(
            clamp_speculative_prefix_cache_hit(4096, 32, SpeculativePrefixReplay::Suffix(2049)),
            2016
        );
        assert_eq!(
            clamp_speculative_prefix_cache_hit(1024, 32, SpeculativePrefixReplay::Suffix(2048)),
            0
        );
        assert_eq!(
            clamp_speculative_prefix_cache_hit(4096, 32, SpeculativePrefixReplay::Full),
            0
        );
    }

    #[test]
    fn models_without_a_proposer_have_no_graphs_to_evict() {
        assert_eq!(
            NoSpeculativeProposer.evict_speculative_cuda_graphs(usize::MAX),
            0
        );
    }

    #[test]
    fn bypass_and_release_flush_recurrent_transitions() -> candle_core::Result<()> {
        let flushed = Rc::new(RefCell::new(Vec::new()));
        let mut target = TransitionTarget {
            flushed: Rc::clone(&flushed),
        };
        target.speculative_bypass(&[3, 8])?;
        target.release_speculative_sequences(&[8])?;
        assert_eq!(*flushed.borrow(), vec![vec![3, 8], vec![8]]);
        Ok(())
    }
}
