use inference_tensor::{DType, Device, Result, Tensor, TensorId};

use crate::kv_cache::{
    GdnDeferredStatePool, GdnPendingTransitionPool, RecurrentStateLayout, RecurrentStatePool,
};

use super::config::{GdnConfig, GdnDims};

/// Per-layer GDN state handed to the kernels. Gathered: `conv_state` `[B, conv_dim, k]` and
/// `recurrent_state` `[B, H, K, V]` hold this batch's rows. Pooled (`slots` set): they are the whole
/// recurrent state pool and `slots` is the `[B]` u32 row table, so CUDA kernels update it in place.
#[derive(Debug)]
pub struct GdnLayerCache {
    pub conv_state: Tensor,
    pub recurrent_state: Tensor,
    pub state_layout: RecurrentStateLayout,
    pub slots: Option<Tensor>,
    pub pending_transitions: Option<GdnPendingTransitionPool>,
    pub deferred_state: Option<GdnDeferredStatePool>,
    /// A CPU checkout of one pool row hands out views of it; their ids, so `commit` skips a state the kernels
    /// updated in place and copies back only one that was replaced.
    pub pool_views: Option<[TensorId; 2]>,
}

impl GdnLayerCache {
    pub fn new(cfg: &dyn GdnConfig, dtype: DType, device: &Device) -> Result<Self> {
        let dims = GdnDims::new(cfg);
        let conv_state = Tensor::zeros((1, dims.conv_dim, dims.conv_kernel_size), dtype, device)?;
        let recurrent_state = Tensor::zeros(
            (1, dims.num_v_heads, dims.head_k_dim, dims.head_v_dim),
            DType::F32,
            device,
        )?;
        Ok(Self::gathered(
            conv_state,
            recurrent_state,
            RecurrentStateLayout::GdnKeyMajor,
        ))
    }

    pub fn gathered(
        conv_state: Tensor,
        recurrent_state: Tensor,
        state_layout: RecurrentStateLayout,
    ) -> Self {
        Self {
            conv_state,
            recurrent_state,
            state_layout,
            slots: None,
            pending_transitions: None,
            deferred_state: None,
            pool_views: None,
        }
    }

    pub fn pooled(
        conv_state: Tensor,
        recurrent_state: Tensor,
        state_layout: RecurrentStateLayout,
        slots: Tensor,
    ) -> Self {
        Self {
            conv_state,
            recurrent_state,
            state_layout,
            slots: Some(slots),
            pending_transitions: None,
            deferred_state: None,
            pool_views: None,
        }
    }

    /// Check out the rows `indices` of `pool` for one layer forward. On CUDA the kernels update the
    /// pool in place through the slot table; elsewhere this is a gathered copy that `commit` scatters back.
    pub fn checkout(pool: &RecurrentStatePool, indices: &Tensor) -> Result<Self> {
        if pool.device().is_cuda() {
            let mut cache = Self::pooled(
                pool.conv_state.clone(),
                pool.recurrent_state.clone(),
                pool.state_layout(),
                indices.clone(),
            );
            cache.pending_transitions = pool.pending_transitions().cloned();
            cache.deferred_state = pool.deferred_state().cloned();
            return Ok(cache);
        }
        if indices.dims1()? == 1 {
            let row = indices.to_vec1::<u32>()?[0] as usize;
            let mut cache = Self::gathered(
                pool.conv_state.narrow(0, row, 1)?,
                pool.recurrent_state.narrow(0, row, 1)?,
                pool.state_layout(),
            );
            cache.pool_views = Some([cache.conv_state.id(), cache.recurrent_state.id()]);
            return Ok(cache);
        }
        Ok(Self::gathered(
            pool.gather_conv_state(indices)?,
            pool.gather_recurrent_state(indices)?,
            pool.state_layout(),
        ))
    }

    pub fn commit(
        self,
        pool: &mut RecurrentStatePool,
        indices: &Tensor,
        host_indices: Option<&[u32]>,
    ) -> Result<()> {
        if self.slots.is_some() {
            return Ok(());
        }
        let [conv_view, recurrent_view] = self.pool_views.map_or([None; 2], |ids| ids.map(Some));
        if conv_view != Some(self.conv_state.id()) {
            pool.scatter_conv_state_with_host_indices(indices, host_indices, &self.conv_state)?;
        }
        if recurrent_view != Some(self.recurrent_state.id()) {
            pool.scatter_recurrent_state_with_host_indices(
                indices,
                host_indices,
                &self.recurrent_state,
            )?;
        }
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        self.conv_state = self.conv_state.zeros_like()?;
        self.recurrent_state = self.recurrent_state.zeros_like()?;
        Ok(())
    }
}

impl Clone for GdnLayerCache {
    fn clone(&self) -> Self {
        Self {
            conv_state: self.conv_state.clone(),
            recurrent_state: self.recurrent_state.clone(),
            state_layout: self.state_layout,
            slots: self.slots.clone(),
            pending_transitions: self.pending_transitions.clone(),
            deferred_state: self.deferred_state.clone(),
            pool_views: self.pool_views,
        }
    }
}
