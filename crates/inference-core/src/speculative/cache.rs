use inference_tensor::{Result, Tensor};

use crate::paged_attention::CacheEngine;
use crate::paged_attention::PagedAttentionMeta;
use crate::sequence::Sequence;

use super::proposer::SpeculativeKvCache;

#[derive(Clone, Copy)]
pub struct SpeculativeCacheOutcome {
    pub keep_len: usize,
    pub accepted_all: bool,
}

pub trait SpeculativeCacheGuard {
    fn commit(&mut self) -> Result<()>;
    fn rollback_to(&mut self, keep_len: usize) -> Result<()>;
}

pub trait SpeculativeCacheAccess {
    type Guard: SpeculativeCacheGuard;

    fn guard_for_reserved(&self, seq_id: usize, base_len: usize, verify_len: usize) -> Self::Guard;

    fn proposer_cache(&self, sequences: &[&Sequence]) -> Result<SpeculativeKvCache<'_>>;

    fn finish_verification(
        &self,
        guard: &mut Self::Guard,
        _seq: &mut Sequence,
        keep_len: usize,
        accepted_all: bool,
    ) -> Result<()> {
        if accepted_all {
            guard.commit()
        } else {
            guard.rollback_to(keep_len)
        }
    }

    fn finish_verification_batch(
        &self,
        guards: &mut [Option<Self::Guard>],
        seqs: &mut [&mut Sequence],
        outcomes: &[Option<SpeculativeCacheOutcome>],
    ) -> Result<()> {
        if guards.len() != seqs.len() || outcomes.len() != seqs.len() {
            inference_tensor::bail!(
                "speculative cache batch shape mismatch: guards={}, seqs={}, outcomes={}",
                guards.len(),
                seqs.len(),
                outcomes.len()
            );
        }
        for ((guard, seq), outcome) in guards.iter_mut().zip(seqs.iter_mut()).zip(outcomes) {
            let (Some(guard), Some(outcome)) = (guard.as_mut(), outcome) else {
                continue;
            };
            self.finish_verification(guard, seq, outcome.keep_len, outcome.accepted_all)?;
        }
        Ok(())
    }

    fn can_stage_proposal(
        &self,
        _sequences: &[&Sequence],
        _base_lens: &[usize],
        _proposal_len: usize,
    ) -> bool {
        true
    }
}

pub struct PagedSpeculativeCacheAccess<'a> {
    metadata: &'a PagedAttentionMeta,
    kv_cache: Vec<(Tensor, Tensor)>,
}

impl<'a> PagedSpeculativeCacheAccess<'a> {
    pub fn new(metadata: &'a PagedAttentionMeta, cache_engine: &CacheEngine) -> Self {
        Self {
            metadata,
            kv_cache: cache_engine.get_kv_cache().clone(),
        }
    }
}

pub struct PagedSpeculativeCacheGuard<'a> {
    metadata: &'a PagedAttentionMeta,
    seq_id: usize,
    reserved_len: usize,
}

impl SpeculativeCacheGuard for PagedSpeculativeCacheGuard<'_> {
    fn commit(&mut self) -> Result<()> {
        Ok(())
    }

    fn rollback_to(&mut self, keep_len: usize) -> Result<()> {
        if keep_len < self.reserved_len {
            let mut kv_mgr = crate::get_mut_arcmutex!(self.metadata.kv_cache_manager);
            kv_mgr.trim_request_to_num_tokens(self.seq_id, keep_len);
        }
        Ok(())
    }
}

impl<'a> SpeculativeCacheAccess for PagedSpeculativeCacheAccess<'a> {
    type Guard = PagedSpeculativeCacheGuard<'a>;

    fn guard_for_reserved(
        &self,
        seq_id: usize,
        base_len: usize,
        verify_len: usize,
    ) -> PagedSpeculativeCacheGuard<'a> {
        PagedSpeculativeCacheGuard {
            metadata: self.metadata,
            seq_id,
            reserved_len: base_len + verify_len,
        }
    }

    fn proposer_cache(&self, _sequences: &[&Sequence]) -> Result<SpeculativeKvCache<'_>> {
        Ok(SpeculativeKvCache::Paged {
            metadata: self.metadata,
            kv_cache: &self.kv_cache,
        })
    }
}
