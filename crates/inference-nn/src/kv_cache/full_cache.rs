use std::sync::{Arc, Mutex, MutexGuard};

use inference_tensor::Tensor;

use super::{Cache, HybridCache, NormalCache};
use crate::gdn::RecurrentBatchKind;
use crate::model::RecurrentMetadata;

pub type LayerCaches = Vec<Option<(Tensor, Tensor)>>;

#[derive(Debug, Clone)]
pub enum EitherCache {
    Normal(Arc<Mutex<NormalCache>>),
    Full(Cache),
    Hybrid(Arc<Mutex<HybridCache>>),
}

impl EitherCache {
    /// Panics otherwise!
    pub fn full(&self) -> &Cache {
        match self {
            Self::Full(full) => full,
            Self::Normal(_) => panic!("Got normal cache, expected full cache."),
            Self::Hybrid(_) => panic!("Got hybrid cache, expected full cache."),
        }
    }

    /// Panics otherwise!
    pub fn normal(&self) -> MutexGuard<'_, NormalCache> {
        match self {
            Self::Normal(normal) => normal.lock().unwrap(),
            Self::Full(_) => panic!("Got full cache, expected normal cache."),
            Self::Hybrid(_) => panic!("Got hybrid cache, expected normal cache."),
        }
    }

    /// Panics otherwise!
    pub fn hybrid(&self) -> MutexGuard<'_, HybridCache> {
        match self {
            Self::Hybrid(hybrid) => hybrid.lock().unwrap(),
            Self::Normal(_) => panic!("Got normal cache, expected hybrid cache."),
            Self::Full(_) => panic!("Got full cache, expected hybrid cache."),
        }
    }

    pub fn is_hybrid(&self) -> bool {
        matches!(self, Self::Hybrid(_))
    }

    /// The hybrid cache's current state indices for a forward of `batch_kind`; None for an attention-only cache.
    pub fn recurrent_metadata(&self, batch_kind: RecurrentBatchKind) -> Option<RecurrentMetadata> {
        if !self.is_hybrid() {
            return None;
        }
        let hybrid_cache = self.hybrid();
        let state_indices_host = hybrid_cache.state_indices_host().map(ToOwned::to_owned);
        hybrid_cache.state_indices().cloned().map(|state_indices| {
            RecurrentMetadata::new(batch_kind, state_indices, state_indices_host)
        })
    }
}
