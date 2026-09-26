use crate::attention::AttentionMask;
use std::sync::{Arc, Mutex, MutexGuard};

use candle_core::{Result, Tensor, D};

use crate::get_mut_arcmutex;

mod full_cache;
mod hybrid_cache;
mod rotating_cache;
mod single_cache;

pub use full_cache::{EitherCache, LayerCaches};
#[cfg(feature = "cuda")]
pub use hybrid_cache::RecurrentCheckpointStateSnapshot;
#[cfg(feature = "cuda")]
pub use hybrid_cache::GDN_PENDING_KEY_BANK_COUNT;
pub use hybrid_cache::{
    GdnDeferredStatePool, GdnDeferredStateSpec, GdnPendingTransitionPool, GdnPendingTransitionSpec,
};
pub use hybrid_cache::{
    HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, RecurrentLayerConfig,
    RecurrentStateLayout, RecurrentStatePool, RecurrentStateSnapshot, RecurrentStateSpec,
};
pub use rotating_cache::{RotatingCache, RotatingCacheSnapshot};
pub use single_cache::{SingleCache, SingleCacheSnapshot};

pub trait PagedAuxiliaryPrefixState: std::any::Any + Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any;
    fn bytes(&self) -> usize;
}

#[derive(Debug, Clone)]
pub enum KvCache {
    Normal { k: SingleCache, v: SingleCache },
    Rotating { k: RotatingCache, v: RotatingCache },
    Shared { owner: usize },
}

#[derive(Debug, Clone)]
pub enum KvCacheSnapshot {
    Normal {
        k: SingleCacheSnapshot,
        v: SingleCacheSnapshot,
    },
    Rotating {
        k: RotatingCacheSnapshot,
        v: RotatingCacheSnapshot,
    },
    Shared {
        owner: usize,
    },
}

pub fn cpu_kv_f16() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        // only worth it where the native fp16 attention kernels exist; elsewhere f16 KV would
        // push attention onto the scalar convert fallback and lose to plain f32
        #[cfg(target_arch = "x86_64")]
        let fast = std::arch::is_x86_feature_detected!("avx512f")
            || (std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
                && std::arch::is_x86_feature_detected!("f16c"));
        #[cfg(target_arch = "aarch64")]
        let fast = std::arch::is_aarch64_feature_detected!("fp16")
            && std::arch::is_aarch64_feature_detected!("fhm");
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let fast = false;
        let on = fast && std::env::var("INFERENCE_RS_CPU_KV_F32").map_or(true, |v| v == "0");
        if on {
            tracing::info!("Using f16 KV cache on CPU (set INFERENCE_RS_CPU_KV_F32=1 for f32).");
        }
        on
    })
}

impl KvCache {
    pub fn new_normal(dim: usize, max_seq_len: usize, capacity_seq_len: usize) -> Self {
        let k = SingleCache::new(dim, max_seq_len, capacity_seq_len);
        let v = SingleCache::new(dim, max_seq_len, capacity_seq_len);
        Self::Normal { k, v }
    }

    pub fn new_rotating(dim: usize, sliding_window: usize, capacity_seq_len: usize) -> Self {
        let k = RotatingCache::new(dim, sliding_window, capacity_seq_len);
        let v = RotatingCache::new(dim, sliding_window, capacity_seq_len);
        Self::Rotating { k, v }
    }

    pub fn new_shared(owner: usize) -> Self {
        Self::Shared { owner }
    }

    pub fn k(&self) -> Result<Option<Tensor>> {
        match self {
            Self::Normal { k, .. } => k.current_data(),
            Self::Rotating { k, .. } => k.current_data(),
            Self::Shared { .. } => Ok(None),
        }
    }

    pub fn v(&self) -> Result<Option<Tensor>> {
        match self {
            Self::Normal { v, .. } => v.current_data(),
            Self::Rotating { v, .. } => v.current_data(),
            Self::Shared { .. } => Ok(None),
        }
    }

    /// Return the K tensor from the last `append()` call.
    ///
    /// For Normal caches this is identical to `k()`. For Rotating caches it
    /// returns the full (retained + new) tensor that `append()` produced,
    /// which during prefill may be larger than the internal sliding-window
    /// buffer returned by `k()`.  Shared KV layers must use this instead of
    /// `k()` so they see the same K/V the donor used for its own attention.
    pub fn appended_k(&self) -> Result<Option<Tensor>> {
        match self {
            Self::Normal { k, .. } => k.current_data(),
            Self::Rotating { k, .. } => Ok(k.last_append_result().cloned()),
            Self::Shared { .. } => Ok(None),
        }
    }

    /// Same as [`appended_k`](Self::appended_k) but for the V tensor.
    pub fn appended_v(&self) -> Result<Option<Tensor>> {
        match self {
            Self::Normal { v, .. } => v.current_data(),
            Self::Rotating { v, .. } => Ok(v.last_append_result().cloned()),
            Self::Shared { .. } => Ok(None),
        }
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        // Metal fast-path: fuse the K and V slice_set calls into one kernel.
        // Skip if inputs aren't already contiguous; the slow path will fix that.
        #[cfg(feature = "metal")]
        if k.device().is_metal() && k.is_contiguous() && v.is_contiguous() {
            #[allow(clippy::collapsible_match)]
            match self {
                Self::Normal { k: kc, v: vc } => {
                    if try_kv_append_dual_metal(kc, vc, k, v)? {
                        let out_k = kc.current_data()?;
                        let out_v = vc.current_data()?;
                        return Ok((out_k.unwrap(), out_v.unwrap()));
                    }
                }
                Self::Rotating { k: kc, v: vc } => {
                    if let Some((rk, rv)) = try_kv_append_rotating_metal(kc, vc, k, v)? {
                        return Ok((rk, rv));
                    }
                }
                _ => {}
            }
        }
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        // f16 KV on CPU halves attention memory traffic; kernels read it with native
        // fp16 NEON and accumulate in f32. INFERENCE_RS_CPU_KV_F32=1 restores f32 storage.
        let (k, v) = if k.device().is_cpu() && k.dtype() == candle_core::DType::F32 && cpu_kv_f16()
        {
            (
                k.to_dtype(candle_core::DType::F16)?,
                v.to_dtype(candle_core::DType::F16)?,
            )
        } else {
            (k, v)
        };
        let (out_k, out_v) = match self {
            Self::Normal { k: kc, v: vc } => {
                kc.append(&k)?;
                vc.append(&v)?;
                (kc.current_data()?, vc.current_data()?)
            }
            Self::Rotating { k: kc, v: vc } => {
                let out_k = kc.append(&k)?;
                let out_v = vc.append(&v)?;
                (Some(out_k), Some(out_v))
            }
            Self::Shared { owner } => {
                candle_core::bail!(
                    "attempted to append KV data to shared cache owned by layer {owner}"
                );
            }
        };
        let k = match out_k {
            None => {
                let mut shape = k.dims().to_vec();
                match self {
                    Self::Normal { k, .. } => shape[k.dim] = 0,
                    Self::Rotating { k, .. } => shape[k.dim] = 0,
                    Self::Shared { .. } => unreachable!(),
                }
                Tensor::zeros(shape, k.dtype(), k.device())?
            }
            Some(k) => k,
        };
        let v = match out_v {
            None => {
                let mut shape = v.dims().to_vec();
                match self {
                    Self::Normal { v, .. } => shape[v.dim] = 0,
                    Self::Rotating { v, .. } => shape[v.dim] = 0,
                    Self::Shared { .. } => unreachable!(),
                }
                Tensor::zeros(shape, v.dtype(), v.device())?
            }
            Some(v) => v,
        };
        Ok((k, v))
    }

    pub fn current_seq_len(&self) -> usize {
        match self {
            Self::Normal { k, .. } => k.current_seq_len(),
            Self::Rotating { k, .. } => k.current_seq_len(),
            Self::Shared { .. } => 0,
        }
    }

    pub fn snapshot(&self) -> Result<KvCacheSnapshot> {
        match self {
            Self::Normal { k, v } => Ok(KvCacheSnapshot::Normal {
                k: k.snapshot(),
                v: v.snapshot(),
            }),
            Self::Rotating { k, v } => Ok(KvCacheSnapshot::Rotating {
                k: k.snapshot()?,
                v: v.snapshot()?,
            }),
            Self::Shared { owner } => Ok(KvCacheSnapshot::Shared { owner: *owner }),
        }
    }

    pub fn can_append_from_snapshot(
        &self,
        snapshot: &KvCacheSnapshot,
        base_len: usize,
        append_len: usize,
        max_context_len: usize,
    ) -> bool {
        if base_len + append_len > max_context_len {
            return false;
        }
        match (self, snapshot) {
            (
                Self::Normal { k, v },
                KvCacheSnapshot::Normal {
                    k: k_snapshot,
                    v: v_snapshot,
                },
            ) => {
                k.current_seq_len() == base_len
                    && v.current_seq_len() == base_len
                    && k_snapshot.current_seq_len == base_len
                    && v_snapshot.current_seq_len == base_len
                    && k.can_append_from_snapshot(k_snapshot, append_len)
                    && v.can_append_from_snapshot(v_snapshot, append_len)
            }
            (
                Self::Rotating { k, v },
                KvCacheSnapshot::Rotating {
                    k: k_snapshot,
                    v: v_snapshot,
                },
            ) => {
                k.current_seq_len() == base_len
                    && v.current_seq_len() == base_len
                    && k_snapshot.current_seq_len == base_len
                    && v_snapshot.current_seq_len == base_len
                    && k.can_append_from_snapshot(k_snapshot, append_len)
                    && v.can_append_from_snapshot(v_snapshot, append_len)
            }
            (
                Self::Shared { owner },
                KvCacheSnapshot::Shared {
                    owner: snapshot_owner,
                },
            ) => owner == snapshot_owner,
            _ => false,
        }
    }

    pub fn restore_after_speculative_append(
        &mut self,
        snapshot: &KvCacheSnapshot,
        post_forward_layer: Option<&KvCache>,
        keep_len: usize,
        row_idx: usize,
        batch_len: usize,
    ) -> Result<()> {
        match (self, snapshot) {
            (Self::Normal { k, v }, KvCacheSnapshot::Normal { .. }) => {
                k.rollback_to(keep_len)?;
                v.rollback_to(keep_len)?;
            }
            (
                Self::Rotating { k, v },
                KvCacheSnapshot::Rotating {
                    k: k_snapshot,
                    v: v_snapshot,
                },
            ) => {
                let Some(KvCache::Rotating {
                    k: post_k,
                    v: post_v,
                }) = post_forward_layer
                else {
                    candle_core::bail!(
                        "rotating cache speculative rollback requires post-forward rotating layer"
                    );
                };
                let accepted_k = post_k.accepted_append_from_batched_append(
                    k_snapshot, keep_len, row_idx, batch_len,
                )?;
                let accepted_v = post_v.accepted_append_from_batched_append(
                    v_snapshot, keep_len, row_idx, batch_len,
                )?;
                *k = RotatingCache::restore_from_snapshot(k_snapshot, accepted_k, keep_len)?;
                *v = RotatingCache::restore_from_snapshot(v_snapshot, accepted_v, keep_len)?;
            }
            (
                Self::Shared { owner },
                KvCacheSnapshot::Shared {
                    owner: snapshot_owner,
                },
            ) => {
                *owner = *snapshot_owner;
            }
            (layer, KvCacheSnapshot::Shared { owner }) => {
                *layer = KvCache::Shared { owner: *owner };
            }
            _ => {
                candle_core::bail!("kv-cache speculative rollback snapshot kind mismatch");
            }
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        match self {
            Self::Normal { k, v } => {
                k.reset();
                v.reset();
            }
            Self::Rotating { k, v } => {
                k.reset();
                v.reset();
            }
            Self::Shared { .. } => {}
        }
    }

    /// Returns Ok if the length reassignment was successful, otherwise returns Err.
    pub fn set_len(&mut self, len: usize) -> candle_core::Result<()> {
        match self {
            Self::Normal { k, v } => {
                k.set_len(len)?;
                v.set_len(len)?;
                Ok(())
            }
            Self::Rotating { k, v } => {
                k.set_len(len)?;
                v.set_len(len)?;
                Ok(())
            }
            Self::Shared { .. } => Ok(()),
        }
    }

    pub fn try_set_len(&self, len: usize) -> candle_core::Result<()> {
        match self {
            Self::Normal { k, v } => {
                k.try_set_len(len)?;
                v.try_set_len(len)?;
                Ok(())
            }
            Self::Rotating { k, v } => {
                k.try_set_len(len)?;
                v.try_set_len(len)?;
                Ok(())
            }
            Self::Shared { .. } => Ok(()),
        }
    }

    pub fn is_rotating(&self) -> bool {
        matches!(self, Self::Rotating { .. })
    }

    pub fn is_shared(&self) -> bool {
        matches!(self, Self::Shared { .. })
    }
}

#[derive(Debug, Clone)]
pub struct NormalCache(pub Vec<KvCache>);

#[derive(Debug)]
pub enum NormalCacheType {
    Normal { max_seq_len: usize },
    SlidingWindow { window: usize },
    Shared { owner: usize },
}

impl NormalCache {
    /// The number of tokens to grow the cache by
    pub const CACHE_GROW_SIZE: usize = 512;

    pub fn new(len: usize, max_seq_len: usize) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self(vec![
            KvCache::new_normal(
                2,
                max_seq_len,
                Self::CACHE_GROW_SIZE
            );
            len
        ])))
    }

    pub fn new_sliding(
        len: usize,
        max_seq_len: usize,
        sliding_window: Option<usize>,
    ) -> Arc<Mutex<Self>> {
        match sliding_window {
            Some(sliding_window) => Arc::new(Mutex::new(Self(vec![
                KvCache::new_rotating(
                    2,
                    sliding_window,
                    Self::CACHE_GROW_SIZE
                );
                len
            ]))),
            None => Arc::new(Mutex::new(Self(vec![
                KvCache::new_normal(
                    2,
                    max_seq_len,
                    Self::CACHE_GROW_SIZE
                );
                len
            ]))),
        }
    }

    pub fn from_types(types: Vec<NormalCacheType>) -> Arc<Mutex<Self>> {
        let mut caches = Vec::new();
        for ty in types {
            match ty {
                NormalCacheType::Normal { max_seq_len } => {
                    caches.push(KvCache::new_normal(2, max_seq_len, Self::CACHE_GROW_SIZE));
                }
                NormalCacheType::SlidingWindow { window } => {
                    caches.push(KvCache::new_rotating(2, window, Self::CACHE_GROW_SIZE));
                }
                NormalCacheType::Shared { owner } => {
                    caches.push(KvCache::new_shared(owner));
                }
            }
        }
        Arc::new(Mutex::new(Self(caches)))
    }
}

#[derive(Debug, Clone)]
pub struct Cache {
    cache: Arc<Mutex<LayerCaches>>,
    xlora_cache: Option<Arc<Mutex<LayerCaches>>>,
    draft_cache: Arc<Mutex<LayerCaches>>,
    scalings_cache: Option<Arc<Mutex<Option<Tensor>>>>,
}

impl Cache {
    pub fn new(len: usize, is_xlora: bool) -> Self {
        Self {
            cache: Arc::new(Mutex::new(vec![None; len])),
            xlora_cache: if is_xlora {
                Some(Arc::new(Mutex::new(vec![None; len])))
            } else {
                None
            },
            draft_cache: Arc::new(Mutex::new(vec![None; len])),
            scalings_cache: if is_xlora {
                Some(Arc::new(Mutex::new(None)))
            } else {
                None
            },
        }
    }

    pub fn lock(&self) -> MutexGuard<'_, LayerCaches> {
        get_mut_arcmutex!(self.cache)
    }

    pub fn draft_lock(&self) -> MutexGuard<'_, LayerCaches> {
        get_mut_arcmutex!(self.draft_cache)
    }

    /// # Panics
    /// If there is no xlora cache
    pub fn xlora_lock(&self) -> MutexGuard<'_, LayerCaches> {
        get_mut_arcmutex!(self.xlora_cache.as_ref().expect("No X-LoRA cache."))
    }

    /// # Panics
    /// If there is no xlora cache
    pub fn get_scalings_cache(&self) -> MutexGuard<'_, Option<Tensor>> {
        get_mut_arcmutex!(self
            .scalings_cache
            .as_ref()
            .expect("No X-LoRA scalings cache."))
    }

    pub fn is_xlora(&self) -> bool {
        self.xlora_cache.is_some()
    }

    /// Update the KV cache and return (k,v)
    pub fn update_kv_cache(
        cache: &mut Option<(Tensor, Tensor)>,
        k: Tensor,
        v: Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (k, v) = match &*cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                let k = Tensor::cat(&[k_cache, &k], 2)?.contiguous()?;
                let v = Tensor::cat(&[v_cache, &v], 2)?.contiguous()?;
                (k, v)
            }
        };
        *cache = Some((k.clone(), v.clone()));
        Ok((k.contiguous()?, v.contiguous()?))
    }

    /// Update the KV cache and return (k,v,attn_mask)
    pub fn update_kv_cache_sliding_window(
        cache: &mut Option<(Tensor, Tensor)>,
        k: Tensor,
        v: Tensor,
        attention_mask: &AttentionMask,
        sliding_window: Option<usize>,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let mask_tensor = match attention_mask {
            AttentionMask::Custom(t) => Some(t.clone()),
            _ => None,
        };
        let (k, v, attention_mask) = match cache.clone() {
            None => (k, v, mask_tensor),
            Some((mut prev_k, mut prev_v)) => {
                let mut mask = mask_tensor;
                if let Some(sliding_window) = sliding_window {
                    let kv_seq_len = prev_k.dim(2)?;
                    if kv_seq_len > sliding_window {
                        prev_k = prev_k.narrow(
                            2,
                            kv_seq_len - (sliding_window - 1),
                            sliding_window - 1,
                        )?;
                        prev_v = prev_v.narrow(
                            2,
                            kv_seq_len - (sliding_window - 1),
                            sliding_window - 1,
                        )?;
                        if let Some(ref mut mask) = mask {
                            let mask_len = mask.dim(1)?;
                            *mask = mask.narrow(
                                1,
                                mask_len - (sliding_window - 1),
                                sliding_window - 1,
                            )?;
                            *mask = Tensor::cat(
                                &[&*mask, &mask.narrow(1, mask_len - 1, 1)?.ones_like()?],
                                D::Minus1,
                            )?;
                        }
                    }
                }
                let (k, v) = {
                    let k = Tensor::cat(&[prev_k, k], 2)?.contiguous()?;
                    let v = Tensor::cat(&[prev_v, v], 2)?.contiguous()?;
                    (k, v)
                };
                (k, v, mask)
            }
        };
        *cache = Some((k.clone(), v.clone()));
        Ok((k.contiguous()?, v.contiguous()?, attention_mask))
    }
}

#[cfg(feature = "metal")]
fn try_kv_append_dual_metal(
    kc: &mut single_cache::SingleCache,
    vc: &mut single_cache::SingleCache,
    k_src: &Tensor,
    v_src: &Tensor,
) -> Result<bool> {
    use candle_core::{backend::BackendStorage, Storage};

    // Layout requirements: dim=2, rank=4, source [b=1, n_kv, src_seq, head_dim],
    // dst (cache) [b=1, n_kv, max_seq, head_dim], both BF16/F16/F32.
    if kc.dim != 2 || vc.dim != 2 {
        return Ok(false);
    }
    if k_src.rank() != 4 || v_src.rank() != 4 {
        return Ok(false);
    }
    if !matches!(
        k_src.dtype(),
        candle_core::DType::BF16 | candle_core::DType::F16 | candle_core::DType::F32
    ) {
        return Ok(false);
    }
    if k_src.dtype() != v_src.dtype() {
        return Ok(false);
    }
    if k_src.shape() != v_src.shape() {
        return Ok(false);
    }
    let (b, n_kv, src_seq, head_dim) = k_src.dims4()?;
    if b != 1 {
        return Ok(false);
    }
    if kc.current_seq_len + src_seq > kc.capacity_seq_len {
        return Ok(false);
    }
    if vc.current_seq_len + src_seq > vc.capacity_seq_len {
        return Ok(false);
    }
    if kc.current_seq_len != vc.current_seq_len {
        return Ok(false);
    }
    if kc.all_data.is_none() || vc.all_data.is_none() {
        // First call: let the slow path allocate the cache buffers.
        return Ok(false);
    }
    let k_dst = kc.all_data.as_ref().unwrap();
    let v_dst = vc.all_data.as_ref().unwrap();
    if k_dst.shape() != v_dst.shape() {
        return Ok(false);
    }
    let max_seq = k_dst.dim(2)?;
    if k_dst.dims4()? != (b, n_kv, max_seq, head_dim) {
        return Ok(false);
    }
    if !k_dst.is_contiguous() || !v_dst.is_contiguous() {
        return Ok(false);
    }

    let (k_src_s, k_src_l) = k_src.storage_and_layout();
    let (v_src_s, v_src_l) = v_src.storage_and_layout();
    let (k_dst_s, _) = k_dst.storage_and_layout();
    let (v_dst_s, _) = v_dst.storage_and_layout();
    let (
        Storage::Metal(k_src_m),
        Storage::Metal(v_src_m),
        Storage::Metal(k_dst_m),
        Storage::Metal(v_dst_m),
    ) = (&*k_src_s, &*v_src_s, &*k_dst_s, &*v_dst_s)
    else {
        return Ok(false);
    };

    let device = k_src_m.device().clone();
    let encoder = device.command_encoder()?;
    encoder.set_label("kv-append-dual");

    inference_quant::metal_kernels::call_kv_append_dual(
        device.device(),
        &encoder,
        &inference_quant::metal_kernels::Kernels::new(),
        k_src.dtype(),
        k_src_m.buffer(),
        k_src_l.start_offset() * k_src.dtype().size_in_bytes(),
        v_src_m.buffer(),
        v_src_l.start_offset() * v_src.dtype().size_in_bytes(),
        k_dst_m.buffer(),
        v_dst_m.buffer(),
        head_dim,
        n_kv,
        src_seq,
        max_seq,
        kc.current_seq_len,
    )
    .map_err(candle_core::Error::wrap)?;

    kc.current_seq_len += src_seq;
    vc.current_seq_len += src_seq;
    Ok(true)
}

#[cfg(feature = "metal")]
fn try_kv_append_rotating_metal(
    kc: &mut rotating_cache::RotatingCache,
    vc: &mut rotating_cache::RotatingCache,
    k_src: &Tensor,
    v_src: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    use candle_core::{backend::BackendStorage, Storage};

    // Decode steady-state only: window is already full, one new token at a time.
    // Anything else falls back so the existing shift-based code handles it.
    if kc.dim != 2 || vc.dim != 2 {
        return Ok(None);
    }
    if k_src.rank() != 4 || v_src.rank() != 4 {
        return Ok(None);
    }
    if !matches!(
        k_src.dtype(),
        candle_core::DType::BF16 | candle_core::DType::F16 | candle_core::DType::F32
    ) || k_src.dtype() != v_src.dtype()
    {
        return Ok(None);
    }
    if k_src.shape() != v_src.shape() {
        return Ok(None);
    }
    let (b, n_kv, src_seq, head_dim) = k_src.dims4()?;
    if b != 1 || src_seq != 1 {
        return Ok(None);
    }
    // Window must be allocated and already full so we can use the buffer as a
    // circular window without breaking shared-KV / prefill paths.
    if kc.all_data.is_none() || vc.all_data.is_none() {
        return Ok(None);
    }
    if kc.current_seq_len < kc.max_seq_len || vc.current_seq_len < vc.max_seq_len {
        return Ok(None);
    }
    if kc.current_seq_len != vc.current_seq_len {
        return Ok(None);
    }
    let k_dst = kc.all_data.as_ref().unwrap().clone();
    let v_dst = vc.all_data.as_ref().unwrap().clone();
    if !k_dst.is_contiguous() || !v_dst.is_contiguous() {
        return Ok(None);
    }
    let max_seq = kc.max_seq_len;
    if k_dst.dims4()? != (b, n_kv, max_seq, head_dim) {
        return Ok(None);
    }

    // Write the new token to slot (current_seq_len) % max_seq, overwriting the
    // oldest entry. The attention math is order-invariant (RoPE is in K), and
    // the returned buffer is just the full window.
    let slot = kc.current_seq_len % max_seq;

    {
        let (k_src_s, k_src_l) = k_src.storage_and_layout();
        let (v_src_s, v_src_l) = v_src.storage_and_layout();
        let (k_dst_s, _) = k_dst.storage_and_layout();
        let (v_dst_s, _) = v_dst.storage_and_layout();
        let (
            Storage::Metal(k_src_m),
            Storage::Metal(v_src_m),
            Storage::Metal(k_dst_m),
            Storage::Metal(v_dst_m),
        ) = (&*k_src_s, &*v_src_s, &*k_dst_s, &*v_dst_s)
        else {
            return Ok(None);
        };

        let device = k_src_m.device().clone();
        let encoder = device.command_encoder()?;
        encoder.set_label("kv-append-rotating");

        inference_quant::metal_kernels::call_kv_append_dual(
            device.device(),
            &encoder,
            &inference_quant::metal_kernels::Kernels::new(),
            k_src.dtype(),
            k_src_m.buffer(),
            k_src_l.start_offset() * k_src.dtype().size_in_bytes(),
            v_src_m.buffer(),
            v_src_l.start_offset() * v_src.dtype().size_in_bytes(),
            k_dst_m.buffer(),
            v_dst_m.buffer(),
            head_dim,
            n_kv,
            src_seq,
            max_seq,
            slot,
        )
        .map_err(candle_core::Error::wrap)?;
    }

    kc.current_seq_len += src_seq;
    vc.current_seq_len += src_seq;
    kc.last_append_result = Some(k_dst.clone());
    vc.last_append_result = Some(v_dst.clone());
    Ok(Some((k_dst, v_dst)))
}
