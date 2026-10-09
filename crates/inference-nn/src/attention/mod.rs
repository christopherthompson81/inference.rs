#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::attention::backends::cpu;

use inference_tensor::{DType, Device, Result, Tensor};

/// Attention mask passed to [`Sdpa::run_attention`].
///
/// Encodes both the mask data and the *intent*, whether the attention layer
/// should use flash attention (causal handled by the kernel), eager attention
/// with an explicit mask tensor, or no masking at all.
#[derive(Clone, Debug)]
pub enum AttentionMask {
    /// No masking. Used for single-token decode or truly unmasked attention.
    None,
    /// Flash attention with `is_causal = true`. No mask tensor is needed;
    /// the flash kernel applies causal masking internally. Also signals
    /// "this is a prefill" to the paged attention layer.
    CausalFlash,
    /// An explicit mask tensor (causal, sliding window, bidirectional, etc).
    /// CPU fused attention can consume it directly; other backends route to eager as needed.
    Custom(Tensor),
}

impl AttentionMask {
    /// Extract the inner tensor as `Option<&Tensor>`.
    ///
    /// Returns `Some(&tensor)` for [`Custom`](Self::Custom), `None` otherwise.
    /// Useful for interfacing with paged-attention and MLA helpers that still
    /// accept `Option<&Tensor>`.
    pub fn as_option_tensor(&self) -> Option<&Tensor> {
        match self {
            Self::Custom(t) => Some(t),
            _ => None,
        }
    }

    /// Returns `true` when the mask carries an explicit tensor
    /// ([`Custom`](Self::Custom) variant), mirroring the old
    /// `Option<Tensor>::is_some()` semantics.
    pub fn is_custom(&self) -> bool {
        matches!(self, Self::Custom(_))
    }

    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

mod backends;
mod dispatch;
pub mod flash_params;

pub use flash_params::{FlashKMeta, FlashParams};

pub use dispatch::AttentionDispatch;

pub use backends::cpu::fast_exp;
#[cfg(feature = "cuda")]
pub use backends::fattn_sinks;
#[cfg(feature = "cuda")]
use backends::naive::maybe_synchronize;
pub use backends::{
    fattn_supports, flash_attn, flash_backend_supports, flash_backend_supports_sdpa, naive_sdpa,
    sinks_attn, sinks_backend_is_available, sinks_backend_supports,
};

// One additive mask for every head runs on fattn; per-head masks, F32 and head dims fattn lacks take the unfused path
#[cfg(feature = "cuda")]
fn masked_flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    sdpa_params: &SdpaParams,
) -> Result<Option<Tensor>> {
    if !q.device().is_cuda()
        || !crate::utils::using_flash_attn()
        || q.dtype() == DType::F32
        || !fattn_supports(q.dim(3)?, sdpa_params.softcap.is_some())
    {
        return Ok(None);
    }
    let (k, v) = if sdpa_params.n_kv_groups > FLASH_ATTN_NATIVE_MAX_GQA_GROUP {
        (
            repeat_kv(k.clone(), sdpa_params.n_kv_groups)?,
            repeat_kv(v.clone(), sdpa_params.n_kv_groups)?,
        )
    } else {
        (k.clone(), v.clone())
    };
    let (q, k, v) = (q.transpose(1, 2)?, k.transpose(1, 2)?, v.transpose(1, 2)?);
    backends::fattn_masked(&q, &k, &v, mask, sdpa_params)?
        .map(|out| out.transpose(1, 2))
        .transpose()
}

/// Chunk size for attention computation to avoid OOM on long sequences
pub const ATTENTION_CHUNK_SIZE: usize = 1024;
pub const FLASH_ATTN_NATIVE_MAX_GQA_GROUP: usize = 8;

#[cfg(any(
    feature = "flash-attn-v3",
    all(feature = "cuda", target_family = "unix")
))]
pub fn sliding_window_left(sliding_window: Option<usize>) -> Option<usize> {
    sliding_window.map(|window| window.saturating_sub(1))
}

fn eager_attention_mask(
    query_len: usize,
    key_len: usize,
    causal: bool,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Option<Tensor>> {
    if !causal && sliding_window.is_none() {
        return Ok(None);
    }
    let prefix_len = key_len.saturating_sub(query_len);
    let mut mask = Vec::with_capacity(query_len * key_len);
    for query_idx in 0..query_len {
        let query_pos = prefix_len + query_idx;
        for key_idx in 0..key_len {
            let future = causal && key_idx > query_pos;
            let too_old = sliding_window
                .is_some_and(|window| query_pos >= window && key_idx <= query_pos - window);
            mask.push(if future || too_old {
                f32::NEG_INFINITY
            } else {
                0.0
            });
        }
    }
    Tensor::from_vec(mask, (query_len, key_len), device)
        .and_then(|mask| mask.to_dtype(dtype))
        .map(Some)
}

/// Generic chunked attention computation that can be used by different backends
pub fn chunked_attention<F>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    attention_fn: F,
) -> Result<Tensor>
where
    F: Fn(&Tensor, &Tensor, &Tensor, Option<&Tensor>) -> Result<Tensor>,
{
    chunked_attention_with_offset(q, k, v, mask, |q, k, v, mask, _offset| {
        attention_fn(q, k, v, mask)
    })
}

pub fn chunked_attention_with_offset<F>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    attention_fn: F,
) -> Result<Tensor>
where
    F: Fn(&Tensor, &Tensor, &Tensor, Option<&Tensor>, usize) -> Result<Tensor>,
{
    let seq_len = q.dim(2)?;

    if seq_len <= ATTENTION_CHUNK_SIZE {
        return attention_fn(q, k, v, mask, 0);
    }

    let num_chunks = seq_len.div_ceil(ATTENTION_CHUNK_SIZE);
    let mut attn_chunks = Vec::with_capacity(num_chunks);

    for chunk_idx in 0..num_chunks {
        let offset = chunk_idx * ATTENTION_CHUNK_SIZE;
        let chunk_len = ATTENTION_CHUNK_SIZE.min(seq_len - offset);

        // Extract query chunk
        let q_chunk = q.narrow(2, offset, chunk_len)?;

        // Extract mask chunk if present
        let mask_chunk = mask
            .map(|m| {
                match m.rank() {
                    2 => {
                        // For 2D masks (seq_len, seq_len), narrow along dimension 0
                        m.narrow(0, offset, chunk_len)
                    }
                    3 => {
                        // For 3D masks (batch, seq_len, seq_len), narrow along dimension 1
                        m.narrow(1, offset, chunk_len)
                    }
                    4 => {
                        // For 4D masks (batch, heads, seq_len, seq_len), narrow along dimension 2
                        m.narrow(2, offset, chunk_len)
                    }
                    _ => m.narrow(2, offset, chunk_len), // Default to dimension 2
                }
            })
            .transpose()?;

        let att_chunk = attention_fn(&q_chunk, k, v, mask_chunk.as_ref(), offset)?;

        attn_chunks.push(att_chunk);
    }

    Tensor::cat(&attn_chunks, 2)
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        Tensor::cat(&vec![&x; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

fn run_flash_attn_cpu_for_dtype(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    // KV may be stored at lower precision than the activations (f16 CPU KV cache);
    // kernels accumulate in f32 either way, so convert q down and the output back up.
    let out_dtype = q.dtype();
    let q_conv;
    let q = if q.dtype() != k.dtype() {
        q_conv = q.to_dtype(k.dtype())?;
        &q_conv
    } else {
        q
    };
    let res = match k.dtype() {
        DType::F32 => cpu::run_flash_attn_cpu::<f32>(q, k, v, mask, sdpa_params),
        DType::F16 => cpu::run_flash_attn_cpu::<half::f16>(q, k, v, mask, sdpa_params),
        DType::BF16 => cpu::run_flash_attn_cpu::<half::bf16>(q, k, v, mask, sdpa_params),
        other => inference_tensor::bail!("Unsupported dtype for CPU flash attn: {other:?}"),
    }?;
    if res.dtype() != out_dtype {
        res.to_dtype(out_dtype)
    } else {
        Ok(res)
    }
}

fn packed_attention_backend_is_available(q: &Tensor, sdpa_params: &SdpaParams) -> Result<bool> {
    let head_dim = q.dim(3)?;
    let has_softcap = sdpa_params.softcap.is_some();
    if sdpa_params.sinks.is_some() && !q.device().is_cuda() {
        // Metal's varlen sinks kernel takes the packed query padded out per sequence
        return Ok(q.dim(0)? > 1 && sinks_backend_is_available(q, head_dim));
    }
    Ok(q.device().is_cuda()
        && crate::utils::using_flash_attn()
        && matches!(q.dtype(), DType::F16 | DType::BF16)
        && if sdpa_params.sinks.is_some() {
            fattn_supports(head_dim, has_softcap)
        } else {
            flash_backend_supports_sdpa(head_dim, has_softcap, sdpa_params.sliding_window.is_some())
        })
}

pub struct SdpaParams {
    pub n_kv_groups: usize,
    pub softcap: Option<f32>,
    pub softmax_scale: f32,
    pub sliding_window: Option<usize>,
    pub sinks: Option<Tensor>,
    /// Llama 4's chunked attention: each query sees only keys in its own chunk of this many positions.
    pub chunk: Option<usize>,
}

pub struct Sdpa;

impl Sdpa {
    /// Computes softmax(QK^T*sqrt(d_k))V
    ///
    /// Inputs:
    /// - q: (b_sz, n_attn_heads, q_len, head_dim)
    /// - k: (b_sz, n_kv_heads, q_len, head_dim)
    /// - v: (b_sz, n_kv_heads, q_len, head_dim)
    ///
    /// Dispatch attention based on the `AttentionMask` variant:
    ///
    /// - `AttentionMask::CausalFlash`: flash attention with `is_causal = true`
    /// - `AttentionMask::None`: flash if available (decode), else eager without mask
    /// - `AttentionMask::Custom`: CPU fused attention, fattn on CUDA for one mask over every head, else eager
    #[allow(clippy::too_many_arguments)]
    pub fn run_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &AttentionMask,
        flash_params: Option<&FlashParams>,
        sdpa_params: &SdpaParams,
    ) -> Result<Tensor> {
        if flash_params.is_some_and(|params| params.packed)
            && (!matches!(mask, AttentionMask::CausalFlash)
                || !flash_params.is_some_and(|params| params.causal)
                || !packed_attention_backend_is_available(q, sdpa_params)?)
        {
            inference_tensor::bail!("packed prefill requires causal varlen attention support");
        }

        // chunks run on fattn alone; a custom mask carries them itself
        if sdpa_params.chunk.is_some() && !mask.is_custom() {
            #[cfg(feature = "cuda")]
            if q.device().is_cuda() {
                let (qt, kt, vt) = (q.transpose(1, 2)?, k.transpose(1, 2)?, v.transpose(1, 2)?);
                if let Some(out) = flash_attn(&qt, &kt, &vt, flash_params, sdpa_params)? {
                    return out.transpose(1, 2);
                }
            }
            inference_tensor::bail!("chunked attention without a mask runs only on fattn");
        }

        if let Some(sinks) = &sdpa_params.sinks {
            #[cfg(feature = "cuda")]
            if q.device().is_cuda() && !mask.is_custom() {
                let (qt, kt, vt) = (q.transpose(1, 2)?, k.transpose(1, 2)?, v.transpose(1, 2)?);
                if let Some(out) = flash_attn(&qt, &kt, &vt, flash_params, sdpa_params)? {
                    return out.transpose(1, 2);
                }
                if flash_params.is_some_and(|params| params.packed) {
                    inference_tensor::bail!(
                        "no FlashAttention kernel takes this packed sinks prefill"
                    );
                }
            }
            // the unfused path needs CausalFlash and the window as an explicit mask; Metal's kernels apply them
            let fused = q.device().is_metal() && sinks_backend_is_available(q, q.dim(3)?);
            let causal_mask = match mask {
                AttentionMask::CausalFlash if q.dim(2)? > 1 && !fused => eager_attention_mask(
                    q.dim(2)?,
                    k.dim(2)?,
                    true,
                    sdpa_params.sliding_window,
                    q.dtype(),
                    q.device(),
                )?,
                _ => None,
            };
            let mask_tensor = match mask {
                AttentionMask::Custom(t) => Some(t),
                _ => causal_mask.as_ref(),
            };
            return sinks_attn(q, k, v, sinks, mask_tensor, flash_params, sdpa_params);
        }

        // The mask carries causality already; the kernel-level do_causal
        // early-exit is safe to enable only when the request is known causal.
        let do_causal = flash_params.is_some_and(|p| p.causal);

        if let AttentionMask::Custom(mask_tensor) = mask {
            if q.device().is_cpu() {
                let q = q.transpose(1, 2)?;
                let k = k.transpose(1, 2)?;
                let v = v.transpose(1, 2)?;
                return run_flash_attn_cpu_for_dtype(&q, &k, &v, Some(mask_tensor), sdpa_params);
            }
            #[cfg(feature = "cuda")]
            if let Some(out) = masked_flash_attn(q, k, v, mask_tensor, sdpa_params)? {
                return Ok(out);
            }

            return self.run_attention_noflash(q, k, v, Some(mask_tensor), sdpa_params, do_causal);
        }

        // CausalFlash or None: try flash attention, fall back to eager
        let needs_mask = matches!(mask, AttentionMask::CausalFlash) && q.dim(2)? > 1
            || sdpa_params.sliding_window.is_some();
        let can_use_flash = q.device().is_cpu()
            || q.device().is_cuda() && crate::utils::using_flash_attn() && q.dtype() != DType::F32;

        if can_use_flash {
            let expanded_kv = if q.device().is_cuda()
                && crate::utils::using_flash_attn()
                && q.dtype() != DType::F32
                && sdpa_params.n_kv_groups > FLASH_ATTN_NATIVE_MAX_GQA_GROUP
            {
                Some((
                    repeat_kv(k.clone(), sdpa_params.n_kv_groups)?,
                    repeat_kv(v.clone(), sdpa_params.n_kv_groups)?,
                    SdpaParams {
                        n_kv_groups: 1,
                        softcap: sdpa_params.softcap,
                        softmax_scale: sdpa_params.softmax_scale,
                        sliding_window: sdpa_params.sliding_window,
                        sinks: sdpa_params.sinks.clone(),
                        chunk: None,
                    },
                ))
            } else {
                None
            };
            let (k, v, sdpa_params) = match &expanded_kv {
                Some((k, v, sdpa_params)) => (k, v, sdpa_params),
                None => (k, v, sdpa_params),
            };

            let head_dim = q.dim(3)?;
            if q.device().is_cuda()
                && !flash_backend_supports_sdpa(
                    head_dim,
                    sdpa_params.softcap.is_some(),
                    sdpa_params.sliding_window.is_some(),
                )
            {
                if flash_params.is_some_and(|params| params.packed) {
                    inference_tensor::bail!(
                        "packed prefill requires FlashAttention support for head_dim={head_dim} \
                         with softcap={}, sliding_window={}",
                        sdpa_params.softcap.is_some(),
                        sdpa_params.sliding_window.is_some()
                    );
                }
                let causal = matches!(mask, AttentionMask::CausalFlash) || do_causal;
                let fallback_mask = eager_attention_mask(
                    q.dim(2)?,
                    k.dim(2)?,
                    causal,
                    sdpa_params.sliding_window,
                    q.dtype(),
                    q.device(),
                )?;
                return self.run_attention_noflash(
                    q,
                    k,
                    v,
                    fallback_mask.as_ref(),
                    sdpa_params,
                    causal,
                );
            }

            // the flash kernels take (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;

            // a CPU-mapped layer of a CUDA model gets the model's CausalFlash mask, which the CPU kernel cannot apply
            if q.device().is_cpu() && !needs_mask {
                return run_flash_attn_cpu_for_dtype(&q, &k, &v, None, sdpa_params);
            }
            if q.device().is_cpu() {
                let (q, k, v) = (q.transpose(1, 2)?, k.transpose(1, 2)?, v.transpose(1, 2)?);
                let causal = matches!(mask, AttentionMask::CausalFlash) || do_causal;
                let fallback_mask = eager_attention_mask(
                    q.dim(2)?,
                    k.dim(2)?,
                    causal,
                    sdpa_params.sliding_window,
                    q.dtype(),
                    q.device(),
                )?;
                return self.run_attention_noflash(
                    &q,
                    &k,
                    &v,
                    fallback_mask.as_ref(),
                    sdpa_params,
                    causal,
                );
            }
            if let Some(out) = flash_attn(&q, &k, &v, flash_params, sdpa_params)? {
                return out.transpose(1, 2);
            }
            if flash_params.is_some_and(|params| params.packed) {
                inference_tensor::bail!("no FlashAttention kernel takes this packed prefill");
            }
            let (q, k, v) = (q.transpose(1, 2)?, k.transpose(1, 2)?, v.transpose(1, 2)?);
            let causal = matches!(mask, AttentionMask::CausalFlash) || do_causal;
            let fallback_mask = eager_attention_mask(
                q.dim(2)?,
                k.dim(2)?,
                causal,
                sdpa_params.sliding_window,
                q.dtype(),
                q.device(),
            )?;
            return self.run_attention_noflash(
                &q,
                &k,
                &v,
                fallback_mask.as_ref(),
                sdpa_params,
                causal,
            );
        }

        // CausalFlash carries no tensor, and the eager kernels apply neither its causality nor a window themselves
        if needs_mask {
            let causal = matches!(mask, AttentionMask::CausalFlash) || do_causal;
            let fallback_mask = eager_attention_mask(
                q.dim(2)?,
                k.dim(2)?,
                causal,
                sdpa_params.sliding_window,
                q.dtype(),
                q.device(),
            )?;
            return self.run_attention_noflash(
                q,
                k,
                v,
                fallback_mask.as_ref(),
                sdpa_params,
                causal,
            );
        }
        self.run_attention_noflash(q, k, v, None, sdpa_params, do_causal)
    }

    /// Same as `run_attention`, but skips the flash-attention dispatch.
    ///
    /// `causal` tells the Metal SDPA-full kernel to enable its upper-triangle skip (`do_causal=true`).
    /// Pass `true` only when the caller's mask is causal-or-stricter.
    /// Pass false` for bidirectional masks (e.g. vision attention).
    #[allow(clippy::too_many_arguments)]
    pub fn run_attention_noflash(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        sdpa_params: &SdpaParams,
        causal: bool,
    ) -> Result<Tensor> {
        let (b_sz, n_attn_heads, seq_len, head_dim) = q.dims4()?;
        let (_, _, _, k_head_dim) = k.dims4()?;
        let (_, _, _, v_head_dim) = v.dims4()?;

        // We can use Metal SDPA (vector/full) if the mask is the correct size and head dims match.
        // If the mask is provided, then softcapping isn't allowed - default back to naive SDPA
        // Softcapping is implemented for vector SDPA.
        let all_head_dims_match = head_dim == k_head_dim && k_head_dim == v_head_dim;
        let tgt_mask_shape = vec![b_sz, n_attn_heads, seq_len, k.dim(2)?];
        let can_use_mask = mask.is_none_or(|mask| {
            mask.layout().broadcast_as(tgt_mask_shape.clone()).is_ok()
                && sdpa_params.softcap.is_none_or(|x| x == 1.0)
        });
        let valid_head_dims: &[usize] = &[32, 64, 72, 80, 96, 128, 256, 512];
        // Metal SDPA full kernel requires q_seq <= k_seq when a mask is present.
        let metal_supports_mask = mask.is_none() || seq_len <= k.dim(2)?;

        // Metal FA path for DK=512 BF16 with a mask. Two specializations:
        // prefill (seq_len > 8) goes through the BlockMMA kernel; decode
        // (seq_len == 1) uses a vector FA kernel ported from llama.cpp.
        if [q, k, v].into_iter().all(|x| x.device().is_metal())
            && head_dim == 512
            && k_head_dim == 512
            && v_head_dim == 512
            && q.dtype() == DType::BF16
            && k.dtype() == DType::BF16
            && v.dtype() == DType::BF16
            && seq_len == 1
            && mask.is_some()
            && sdpa_params.softcap.is_none_or(|x| x == 1.0)
            && let Some(out) =
                crate::attention::backends::metal_flash_attn::try_flash_attn_ext_vec_bf16_dk512(
                    q,
                    k,
                    v,
                    mask,
                    sdpa_params.softmax_scale,
                )?
        {
            return Ok(out);
        }
        if [q, k, v].into_iter().all(|x| x.device().is_metal())
            && head_dim == 512
            && k_head_dim == 512
            && v_head_dim == 512
            && q.dtype() == DType::BF16
            && k.dtype() == DType::BF16
            && v.dtype() == DType::BF16
            && seq_len > 8
            && sdpa_params.softcap.is_none_or(|x| x == 1.0)
            && let Some(mask) = mask
            && let Some(out) =
                crate::attention::backends::metal_flash_attn::try_flash_attn_ext_bf16_dk512(
                    q,
                    k,
                    v,
                    mask,
                    sdpa_params.softmax_scale,
                )?
        {
            return Ok(out);
        }

        if [q, k, v].into_iter().all(|x| x.device().is_metal())
            && all_head_dims_match
            && valid_head_dims.contains(&head_dim)
            && can_use_mask
            && metal_supports_mask
            && !(head_dim == 512 && seq_len > 8)
        {
            let mask = match mask {
                Some(mask) => Some(mask.broadcast_as(tgt_mask_shape)?),
                None => None,
            };
            // do_causal lets the steel_attention kernel bound its kb-loop to
            // the per-query position, skipping the upper triangle of Q*K^T
            // entirely (roughly halves matmul cost for prefill).
            let do_causal = seq_len > 1 && causal;
            return inference_tensor::nn::ops::sdpa(
                q,
                k,
                v,
                mask.as_ref(),
                do_causal,
                sdpa_params.softmax_scale,
                sdpa_params.softcap.unwrap_or(1.0),
            );
        }

        let k = repeat_kv(k.clone(), sdpa_params.n_kv_groups)?;
        let v = repeat_kv(v.clone(), sdpa_params.n_kv_groups)?;

        if mask.is_some_and(|x| x.rank() == 2) || inference_quant::distributed::use_nccl() {
            return naive_sdpa(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                mask,
                sdpa_params,
            );
        }

        #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
        if let (Device::Cuda(_), Some(cublaslt)) = (
            q.device(),
            inference_quant::cublaslt::CUBLASLT_CONTROLLER.get_for_device(q.device()),
        ) {
            #[cfg(feature = "cuda")]
            {
                maybe_synchronize(q.device())?;

                // Use chunked attention for cuBLASLt path
                let k_flat = k.flatten(0, 1)?;
                let v_flat = v.flatten(0, 1)?;

                let kv_len = k.dim(2)?;
                let prefix_len = kv_len.saturating_sub(seq_len);
                chunked_attention_with_offset(
                    q,
                    &k,
                    &v,
                    mask,
                    |q_chunk, _k, _v, mask_chunk, q_offset| {
                        // cuBLASLt batch matmul implementation requires inputs to be dims3
                        let (chunk_b_sz, chunk_n_heads, chunk_seq_len, _) = q_chunk.dims4()?;
                        let q_flat = q_chunk.flatten(0, 1)?;

                        let attention_bias = match mask_chunk {
                            Some(mask) if mask.rank() == 3 && mask.dims()[0] == 1 => {
                                Some(mask.repeat((chunk_n_heads, 1, 1))?)
                            }
                            Some(mask) if mask.rank() == 3 => Some(mask.clone()),
                            Some(mask) if mask.rank() == 4 => {
                                let tgt_shape =
                                    vec![chunk_b_sz, chunk_n_heads, chunk_seq_len, k.dim(2)?];
                                Some(mask.broadcast_as(tgt_shape)?.flatten(0, 1)?)
                            }
                            Some(_) => {
                                inference_tensor::bail!("cublaslt attn mask: rank must be 3 or 4")
                            }
                            None => None,
                        };

                        // If attention_bias is set, we fuse the add by giving it as the output matrix
                        // and setting beta to 1.0
                        let beta = match attention_bias.is_some() {
                            true => Some(1.0),
                            false => None,
                        };

                        // Batch matrix multiplication
                        // Fuse softmax scale and attention_bias add
                        let mut attention_scores = cublaslt.batch_matmul(
                            &k_flat,
                            &q_flat,
                            attention_bias.as_ref(),
                            Some(sdpa_params.softmax_scale / sdpa_params.softcap.unwrap_or(1.0)),
                            beta,
                            None,
                            None,
                        )?;
                        if let Some(softcap) = sdpa_params.softcap {
                            attention_scores = (attention_scores.tanh()? * softcap as f64)?;
                        }
                        // Compute softmax in F32 for precision. BF16's 7 mantissa
                        // bits cause exp() to lose information on long sequences.
                        // Flash attention already computes softmax in F32; this
                        // matches that behaviour for the eager path.
                        let scores_dtype = attention_scores.dtype();
                        if scores_dtype == DType::BF16 || scores_dtype == DType::F16 {
                            attention_scores = attention_scores.to_dtype(DType::F32)?;
                        }
                        if causal && mask_chunk.is_none() {
                            crate::ops::cuda_apply_causal_mask_f32(
                                &attention_scores,
                                q_offset,
                                prefix_len,
                            )?;
                        }
                        attention_scores =
                            inference_tensor::nn::ops::softmax_last_dim(&attention_scores)?;
                        if attention_scores.dtype() != scores_dtype {
                            attention_scores = attention_scores.to_dtype(scores_dtype)?;
                        }

                        let context_layer = cublaslt.batch_matmul(
                            &v_flat.t()?.contiguous()?,
                            &attention_scores,
                            // We save one allocation
                            Some(&q_flat),
                            None,
                            None,
                            None,
                            None,
                        )?;

                        // Reshape to dims4
                        context_layer.reshape((
                            chunk_b_sz,
                            chunk_n_heads,
                            chunk_seq_len,
                            v_head_dim,
                        ))
                    },
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                inference_tensor::bail!("`cuda` feature is not enabled")
            }
        } else {
            naive_sdpa(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                mask,
                sdpa_params,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_tensor::{D, Result as CandleResult};

    const EPS: f32 = 1e-4;
    const CAUSAL_FLASH_TOLERANCE: f32 = 1e-2;
    // bf16 rounding of an O(1) output, a few ulps
    #[cfg(feature = "cuda")]
    const SINKS_BF16_TOLERANCE: f32 = 3e-2;

    fn assert_close(lhs: &Tensor, rhs: &Tensor) -> CandleResult<()> {
        let lhs = lhs.flatten_all()?.to_vec1::<f32>()?;
        let rhs = rhs.flatten_all()?.to_vec1::<f32>()?;
        for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
            assert!((lhs - rhs).abs() < EPS, "{lhs} != {rhs}");
        }
        Ok(())
    }

    #[test]
    fn causal_flash_has_attention_intent_without_a_custom_tensor() {
        let mask = AttentionMask::CausalFlash;

        assert!(!mask.is_none());
        assert!(!mask.is_custom());
    }

    #[test]
    fn test_custom_cpu_mask_uses_attention_dispatch() -> CandleResult<()> {
        let (b, h, q_len, kv_len, d) = (1, 2, 3, 3, 4);
        let q = Tensor::from_vec(
            (0..b * h * q_len * d)
                .map(|x| x as f32 / 31.0)
                .collect::<Vec<_>>(),
            (b, h, q_len, d),
            &Device::Cpu,
        )?;
        let k = Tensor::from_vec(
            (0..b * h * kv_len * d)
                .map(|x| x as f32 / 37.0)
                .collect::<Vec<_>>(),
            (b, h, kv_len, d),
            &Device::Cpu,
        )?;
        let v = Tensor::from_vec(
            (0..b * h * kv_len * d)
                .map(|x| x as f32 / 41.0)
                .collect::<Vec<_>>(),
            (b, h, kv_len, d),
            &Device::Cpu,
        )?;
        let mask = Tensor::from_vec(
            vec![
                0.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                0.0,
            ],
            (q_len, kv_len),
            &Device::Cpu,
        )?;
        let sdpa_params = SdpaParams {
            n_kv_groups: 1,
            softcap: None,
            softmax_scale: 1.0,
            sliding_window: None,
            sinks: None,
            chunk: None,
        };

        let out = Sdpa.run_attention(
            &q,
            &k,
            &v,
            &AttentionMask::Custom(mask.clone()),
            Some(&FlashParams::empty(true)),
            &sdpa_params,
        )?;
        let logits = q.matmul(&k.transpose(2, 3)?)?.broadcast_add(&mask)?;
        let expected = inference_tensor::nn::ops::softmax(&logits, D::Minus1)?.matmul(&v)?;

        assert_eq!(out.shape().dims(), &[b, h, q_len, d]);
        assert_close(&out, &expected)
    }

    // softmax(q k^T + mask) v with an explicit causal (and optionally windowed) mask, the reference for CausalFlash
    fn causal_reference(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        window: Option<usize>,
    ) -> CandleResult<Tensor> {
        let mask =
            eager_attention_mask(q.dim(2)?, k.dim(2)?, true, window, DType::F32, q.device())?
                .unwrap();
        let logits = q.matmul(&k.transpose(2, 3)?)?.broadcast_add(&mask)?;
        inference_tensor::nn::ops::softmax(&logits, D::Minus1)?.matmul(v)
    }

    fn causal_flash_case(device: &Device, window: Option<usize>) -> CandleResult<()> {
        let (b, h, len, d) = (1, 2, 6, 64);
        let rand = || Tensor::randn(0f32, 1., (b, h, len, d), device);
        let (q, k, v) = (rand()?, rand()?, rand()?);
        let sdpa_params = SdpaParams {
            n_kv_groups: 1,
            softcap: None,
            softmax_scale: 1.0,
            sliding_window: window,
            sinks: None,
            chunk: None,
        };
        // unscaled d = 64 random logits are peaky enough that rounding drifted past the tolerance in ~1 run of 100
        let q = (q / (d as f64).sqrt())?;
        let out = Sdpa.run_attention(
            &q,
            &k,
            &v,
            &AttentionMask::CausalFlash,
            Some(&FlashParams::empty(true)),
            &sdpa_params,
        )?;
        let expected = causal_reference(&q, &k, &v, window)?;
        let diff = (out - expected)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        // a dropped causal mask moves outputs by O(1); the eager paths round within this
        assert!(diff < CAUSAL_FLASH_TOLERANCE, "max abs diff {diff}");
        Ok(())
    }

    #[test]
    fn causal_flash_masks_a_cpu_mapped_layer() -> CandleResult<()> {
        // a CUDA model's CausalFlash mask reaches the layers a device map puts on the CPU
        causal_flash_case(&Device::Cpu, None)?;
        causal_flash_case(&Device::Cpu, Some(3))
    }

    #[test]
    fn causal_flash_with_sinks_masks_the_unfused_path() -> CandleResult<()> {
        let (h, kv_h, len, d) = (4, 2, 12, 16);
        let rand = |heads| Tensor::randn(0f32, 1., (1, heads, len, d), &Device::Cpu);
        let (q, k, v) = (rand(h)?, rand(kv_h)?, rand(kv_h)?);
        for window in [None, Some(4)] {
            let sdpa_params = SdpaParams {
                n_kv_groups: h / kv_h,
                softcap: None,
                softmax_scale: 1. / (d as f32).sqrt(),
                sliding_window: window,
                sinks: Some(Tensor::new(&[0.5f32, -1., 2., 0.], &Device::Cpu)?),
                chunk: None,
            };
            let mask = eager_attention_mask(len, len, true, window, DType::F32, &Device::Cpu)?;
            let expected = Sdpa.run_attention(
                &q,
                &k,
                &v,
                &AttentionMask::Custom(mask.unwrap()),
                None,
                &sdpa_params,
            )?;
            let out = Sdpa.run_attention(
                &q,
                &k,
                &v,
                &AttentionMask::CausalFlash,
                Some(&FlashParams::empty(true)),
                &sdpa_params,
            )?;
            let diff = (out - expected)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(
                diff < CAUSAL_FLASH_TOLERANCE,
                "window {window:?}: max abs diff {diff}"
            );
        }
        Ok(())
    }

    // gpt-oss's prompt shape: GQA at head dim 64, a sink per head, every other layer windowed
    #[cfg(feature = "cuda")]
    #[test]
    fn sinks_prefill_on_cuda_matches_the_cpu() -> CandleResult<()> {
        crate::skip_without_cuda!();
        let dev = Device::new_cuda(0)?;
        // past one 64-query tile
        let (h, kv_h, len, d) = (8, 2, 81, 64);
        // (b, s, heads, d) rows viewed as (b, heads, s, d), as the projections leave them; rounded to bf16 first so
        // the reference sees the GPU's inputs
        let rand = |heads| {
            Tensor::randn(0f32, 1., (1, len, heads, d), &Device::Cpu)?
                .to_dtype(DType::BF16)?
                .to_dtype(DType::F32)?
                .transpose(1, 2)
        };
        let (q, k, v) = (rand(h)?, rand(kv_h)?, rand(kv_h)?);
        for (window, sink) in [(None, 0.5), (Some(8), 0.5), (Some(8), 8.)] {
            let sdpa_params = |device: &Device| -> CandleResult<SdpaParams> {
                Ok(SdpaParams {
                    n_kv_groups: h / kv_h,
                    softcap: None,
                    softmax_scale: 1. / (d as f32).sqrt(),
                    sliding_window: window,
                    sinks: Some(Tensor::arange(0f32, h as f32, device)?.affine(0.25, sink)?),
                    chunk: None,
                })
            };
            let cpu_mask =
                eager_attention_mask(len, len, true, window, DType::F32, &Device::Cpu)?.unwrap();
            let expected = Sdpa.run_attention(
                &q,
                &k,
                &v,
                &AttentionMask::Custom(cpu_mask),
                None,
                &sdpa_params(&Device::Cpu)?,
            )?;
            let on_gpu = |t: &Tensor| {
                t.transpose(1, 2)?
                    .contiguous()?
                    .to_device(&dev)?
                    .to_dtype(DType::BF16)?
                    .transpose(1, 2)
            };
            let out = Sdpa.run_attention(
                &on_gpu(&q)?,
                &on_gpu(&k)?,
                &on_gpu(&v)?,
                &AttentionMask::CausalFlash,
                Some(&FlashParams::empty(true)),
                &sdpa_params(&dev)?,
            )?;
            let diff = (out.to_dtype(DType::F32)?.to_device(&Device::Cpu)? - expected)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(
                diff < SINKS_BF16_TOLERANCE,
                "window {window:?} sink {sink}: max abs diff {diff}"
            );
        }
        Ok(())
    }

    // A custom mask on CUDA runs fattn when one mask serves every head; the CPU's fused path is the reference
    #[cfg(feature = "cuda")]
    #[test]
    fn custom_masks_on_cuda_run_fattn_and_match_the_cpu() -> CandleResult<()> {
        crate::skip_without_cuda!();
        let dev = Device::new_cuda(0)?;
        // past several 64-query tiles, GQA at head dim 128
        let (h, kv_h, len, d) = (8, 2, 300, 128);
        // a bidirectional span, as Gemma 3 gives an image's tokens
        let span = 40..120;
        // the second sequence is left-padded: its first keys are hidden from every query, as expand_mask does
        let pad = 50;
        // a prefix-cached prefill: the last queries over every key
        let suffix = 100;
        let visible = |b: usize, i: usize, j: usize| {
            let causal = j <= i || (span.contains(&i) && span.contains(&j));
            causal && (b == 0 || j >= pad)
        };
        // (batch, 1, rows, len) for query positions rows
        let mask =
            |batch: usize, fill: f32, rows: std::ops::Range<usize>| -> CandleResult<Tensor> {
                let n = rows.len();
                let values: Vec<f32> = (0..batch)
                    .flat_map(|b| {
                        rows.clone()
                            .flat_map(move |i| (0..len).map(move |j| (b, i, j)))
                    })
                    .map(|(b, i, j)| if visible(b, i, j) { 0. } else { fill })
                    .collect();
                Tensor::from_vec(values, (batch, 1, n, len), &Device::Cpu)
            };
        let key_padding = Tensor::from_vec(
            (0..2 * len)
                .map(|x| {
                    if x / len == 1 && x % len < pad {
                        f32::MIN
                    } else {
                        0.
                    }
                })
                .collect::<Vec<f32>>(),
            (2, 1, 1, len),
            &Device::Cpu,
        )?;
        let sdpa_params = SdpaParams {
            n_kv_groups: h / kv_h,
            softcap: None,
            softmax_scale: 1. / (d as f32).sqrt(),
            sliding_window: None,
            sinks: None,
            chunk: None,
        };
        let full = mask(1, f32::NEG_INFINITY, 0..len)?;
        // label, batch, query rows, mask, dtype, whether fattn takes it, the padded queries to leave out
        let cases = [
            (
                "span, 2d mask",
                1,
                len,
                full.squeeze(0)?.squeeze(0)?,
                DType::BF16,
                true,
                0,
            ),
            (
                "padded batch",
                2,
                len,
                mask(2, f32::MIN, 0..len)?,
                DType::BF16,
                true,
                pad,
            ),
            (
                "prefix-cached suffix",
                2,
                suffix,
                mask(2, f32::MIN, len - suffix..len)?,
                DType::BF16,
                true,
                0,
            ),
            ("key padding, f16", 2, len, key_padding, DType::F16, true, 0),
            (
                "heads broadcast",
                1,
                len,
                full.clone(),
                DType::BF16,
                true,
                0,
            ),
            (
                "per-head mask",
                1,
                len,
                full.repeat((1, h, 1, 1))?,
                DType::BF16,
                false,
                0,
            ),
        ];
        for (label, batch, q_len, mask, dtype, flash, skipped) in cases {
            let rand = |rows, heads| {
                Tensor::randn(0f32, 1., (batch, rows, heads, d), &Device::Cpu)?
                    .to_dtype(dtype)?
                    .to_dtype(DType::F32)?
                    .transpose(1, 2)
            };
            let on_gpu = |t: &Tensor| {
                t.transpose(1, 2)?
                    .contiguous()?
                    .to_device(&dev)?
                    .to_dtype(dtype)?
                    .transpose(1, 2)
            };
            let (q, k, v) = (rand(q_len, h)?, rand(len, kv_h)?, rand(len, kv_h)?);
            let expected = Sdpa.run_attention(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                &AttentionMask::Custom(mask.contiguous()?),
                None,
                &sdpa_params,
            )?;
            let (q, k, v) = (on_gpu(&q)?, on_gpu(&k)?, on_gpu(&v)?);
            let mut gpu_mask = mask.to_device(&dev)?.to_dtype(dtype)?;
            // a mask broadcast over heads keeps a zero stride there, as models build it
            if label == "heads broadcast" {
                gpu_mask = gpu_mask.broadcast_as((1, h, len, len))?;
            }
            assert_eq!(
                masked_flash_attn(&q, &k, &v, &gpu_mask, &sdpa_params)?.is_some(),
                flash,
                "{label}"
            );
            let out = Sdpa
                .run_attention(
                    &q,
                    &k,
                    &v,
                    &AttentionMask::Custom(gpu_mask),
                    None,
                    &sdpa_params,
                )?
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?;
            // padded queries see no key: fattn's rows stay finite, and only the rows a model keeps are compared
            assert!(
                out.flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|x| x.is_finite()),
                "{label}"
            );
            for b in 0..batch {
                let rows = if b == 0 { 0 } else { skipped };
                let diff = (out.get(b)?.narrow(1, rows, q_len - rows)?
                    - expected.get(b)?.narrow(1, rows, q_len - rows)?)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
                assert!(
                    diff < SINKS_BF16_TOLERANCE,
                    "{label} batch {b}: max abs diff {diff}"
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn causal_flash_masks_f32_on_cuda() -> CandleResult<()> {
        crate::skip_without_cuda!();
        let dev = Device::new_cuda(0)?;
        // f32 never takes the flash kernels, so its eager path has to build the causal and window mask
        causal_flash_case(&dev, None)?;
        causal_flash_case(&dev, Some(3))
    }

    #[cfg(any(
        feature = "flash-attn-v3",
        all(feature = "cuda", target_family = "unix")
    ))]
    #[test]
    fn sliding_window_capacity_converts_to_left_distance() {
        assert_eq!(sliding_window_left(None), None);
        assert_eq!(sliding_window_left(Some(1)), Some(0));
        assert_eq!(sliding_window_left(Some(4096)), Some(4095));
    }

    #[test]
    fn eager_mask_combines_suffix_causality_and_sliding_capacity() -> CandleResult<()> {
        let mask = eager_attention_mask(3, 8, true, Some(4), DType::F32, &Device::Cpu)?
            .expect("causal sliding mask");
        let mask = mask.to_vec2::<f32>()?;

        for (row, values) in mask.iter().enumerate() {
            let query_pos = row + 5;
            for (key_pos, &value) in values.iter().enumerate() {
                let visible = key_pos <= query_pos && key_pos + 4 > query_pos;
                assert_eq!(value == 0.0, visible);
            }
        }
        Ok(())
    }

    #[test]
    fn eager_mask_unit_window_keeps_only_current_token() -> CandleResult<()> {
        let mask = eager_attention_mask(1, 3, true, Some(1), DType::F32, &Device::Cpu)?
            .expect("unit sliding mask");

        assert_eq!(
            mask.flatten_all()?.to_vec1::<f32>()?,
            vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0]
        );
        Ok(())
    }
}
