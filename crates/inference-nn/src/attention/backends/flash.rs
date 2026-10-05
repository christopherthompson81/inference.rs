use candle_core::{Result, Tensor};

#[cfg(feature = "cuda")]
use crate::attention::FlashKMeta;
#[cfg(feature = "cuda")]
use crate::attention::sliding_window_left;
use crate::attention::{FlashParams, SdpaParams};

// Head dims fattn takes in every layout (dense, packed, any GQA ratio): 192/320/512/576 run only GQA-batched
const FATTN_HEAD_DIMS: [usize; 6] = [64, 80, 96, 112, 128, 256];
// fattn instantiates softcap only at these (and 512)
const FATTN_SOFTCAP_HEAD_DIMS: [usize; 2] = [128, 256];

fn fattn_supports(head_dim: usize, has_softcap: bool) -> bool {
    #[cfg(feature = "cuda")]
    let available = inference_fattn::mma_available();
    #[cfg(not(feature = "cuda"))]
    let available = false;
    available
        && FATTN_HEAD_DIMS.contains(&head_dim)
        && (!has_softcap || FATTN_SOFTCAP_HEAD_DIMS.contains(&head_dim))
}

// Dao-AILab's FA2 and FA3, behind fattn where built
fn dao_supports(head_dim: usize, has_softcap: bool) -> bool {
    let head_dim_supported = if cfg!(feature = "flash-attn") {
        head_dim.is_multiple_of(8) && head_dim <= 512
    } else if cfg!(feature = "flash-attn-v3") {
        matches!(head_dim, 64 | 128 | 256 | 512)
    } else {
        false
    };
    head_dim_supported && (!has_softcap || cfg!(feature = "flash-attn") && head_dim <= 256)
}

pub fn flash_backend_supports(head_dim: usize, has_softcap: bool) -> bool {
    fattn_supports(head_dim, has_softcap) || dao_supports(head_dim, has_softcap)
}

pub fn flash_backend_supports_sdpa(
    head_dim: usize,
    has_softcap: bool,
    has_sliding_window: bool,
) -> bool {
    // fattn's windows are causal; a bidirectional one falls back to FA2 or eager at the call
    if fattn_supports(head_dim, has_softcap) {
        return true;
    }
    if !dao_supports(head_dim, has_softcap) {
        return false;
    }
    if !has_sliding_window {
        return true;
    }
    // v3 never sets is_local, so without v2 to fall back to a sliding window has to go eager.
    cfg!(feature = "flash-attn") && head_dim <= 256
}

#[cfg(feature = "cuda")]
fn varlen_metadata<'a>(
    q: &Tensor,
    params: &'a FlashParams,
    sliding_window: Option<usize>,
) -> Result<Option<(&'a Tensor, &'a FlashKMeta, &'a Tensor)>> {
    let location = q.device().location();
    let Some(cumulative_seqlens_q) = params.cumulative_seqlens_q.get(&location) else {
        if params.packed {
            candle_core::bail!("packed prefill is missing query metadata for {location:?}");
        }
        return Ok(None);
    };
    let k_meta = params.k_meta(sliding_window);
    let Some(cumulative_seqlens_k) = k_meta.cumulative_seqlens.get(&location) else {
        if params.packed {
            candle_core::bail!("packed prefill is missing key metadata for {location:?}");
        }
        return Ok(None);
    };
    Ok(Some((cumulative_seqlens_q, k_meta, cumulative_seqlens_k)))
}

// fattn (llama.cpp's kernels) first; None for what it cannot take (head dims, softcap, f32 K/V, bidirectional windows)
#[cfg(feature = "cuda")]
fn try_fattn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Option<Tensor>> {
    use inference_fattn::{FattnOptions, Packed};

    let (b_sz, seq_len, _n_attn_heads, _head_dim) = q.dims4()?;
    // one query per sequence sees the same keys causal or not (decode over gathered K/V passes causal false)
    let causal = flash_params.map_or(seq_len > 1, |p| p.causal) || seq_len == 1;
    if !q.device().is_cuda() || sdpa_params.sliding_window.is_some() && !causal {
        return Ok(None);
    }
    let opts = FattnOptions {
        scale: sdpa_params.softmax_scale,
        softcap: sdpa_params.softcap.unwrap_or(0.),
        causal,
        window_left: sliding_window_left(sdpa_params.sliding_window),
        ..Default::default()
    };
    let use_varlen =
        b_sz > 1 || seq_len != k.dim(1)? || flash_params.is_some_and(|params| params.packed);
    if use_varlen
        && let Some(params) = flash_params
        && let Some((cumulative_seqlens_q, k_meta, cumulative_seqlens_k)) =
            varlen_metadata(q, params, sdpa_params.sliding_window)?
    {
        let q_seqs = Packed {
            cu_seqlens: cumulative_seqlens_q,
            max_len: params.max_q as usize,
        };
        let kv_seqs = Packed {
            cu_seqlens: cumulative_seqlens_k,
            max_len: k_meta.max as usize,
        };
        let (qf, kf, vf) = (q.flatten_to(1)?, k.flatten_to(1)?, v.flatten_to(1)?);
        if !inference_fattn::supported_varlen(&qf, &kf, &vf, &q_seqs, &kv_seqs, &opts)? {
            return Ok(None);
        }
        return inference_fattn::flash_attn_varlen(&qf, &kf, &vf, &q_seqs, &kv_seqs, &opts)?
            .reshape(q.shape())
            .map(Some);
    }
    if !inference_fattn::supported(q, k, v, &opts)? {
        return Ok(None);
    }
    inference_fattn::flash_attn(q, k, v, &opts).map(Some)
}

#[cfg(feature = "flash-attn")]
fn flash_attn_v2(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let (b_sz, seq_len, _n_attn_heads, _head_dim) = q.dims4()?;
    let window_size_left = sliding_window_left(sdpa_params.sliding_window);
    let default_causal = seq_len > 1;
    let use_varlen =
        b_sz > 1 || seq_len != k.dim(1)? || flash_params.is_some_and(|params| params.packed);

    if use_varlen {
        if let Some(params) = flash_params {
            if let Some((cumulative_seqlens_q, k_meta, cumulative_seqlens_k)) =
                varlen_metadata(q, params, sdpa_params.sliding_window)?
            {
                let window_size_right = if params.causal { Some(0) } else { None };
                let qshape = q.shape();
                let q = q.flatten_to(1)?;
                let k = k.flatten_to(1)?;
                let v = v.flatten_to(1)?;

                if let Some(softcap) = sdpa_params.softcap {
                    return inference_flash_attn::flash_attn_varlen_alibi_windowed_softcap(
                        &q,
                        &k,
                        &v,
                        None,
                        cumulative_seqlens_q,
                        cumulative_seqlens_k,
                        params.max_q as usize,
                        k_meta.max as usize,
                        sdpa_params.softmax_scale,
                        window_size_left,
                        window_size_right,
                        softcap,
                    )?
                    .reshape(qshape);
                } else {
                    return inference_flash_attn::flash_attn_varlen_windowed(
                        &q,
                        &k,
                        &v,
                        cumulative_seqlens_q,
                        cumulative_seqlens_k,
                        params.max_q as usize,
                        k_meta.max as usize,
                        sdpa_params.softmax_scale,
                        window_size_left,
                        window_size_right,
                    )?
                    .reshape(qshape);
                }
            }
        }
    }

    let causal = flash_params.map_or(default_causal, |p| p.causal);
    let window_size_right = if causal { Some(0) } else { None };
    if let Some(softcap) = sdpa_params.softcap {
        inference_flash_attn::flash_attn_alibi_windowed_softcap(
            q,
            k,
            v,
            None,
            sdpa_params.softmax_scale,
            window_size_left,
            window_size_right,
            softcap,
        )
    } else {
        inference_flash_attn::flash_attn_windowed(
            q,
            k,
            v,
            sdpa_params.softmax_scale,
            window_size_left,
            window_size_right,
        )
    }
}

#[cfg(feature = "flash-attn-v3")]
fn flash_attn_v3(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    if sdpa_params.softcap.is_some() {
        candle_core::bail!("FlashAttention v3 does not support attention softcap");
    }
    let head_dim = q.dim(3)?;
    if !matches!(head_dim, 64 | 128 | 256 | 512) {
        candle_core::bail!("FlashAttention v3 does not support head_dim={head_dim}");
    }
    let (b_sz, seq_len, _n_attn_heads, _head_dim) = q.dims4()?;
    let default_causal = seq_len > 1;
    let use_varlen =
        b_sz > 1 || seq_len != k.dim(1)? || flash_params.is_some_and(|params| params.packed);

    if use_varlen {
        if let Some(params) = flash_params {
            if let Some((cumulative_seqlens_q, k_meta, cumulative_seqlens_k)) =
                varlen_metadata(q, params, sdpa_params.sliding_window)?
            {
                let qshape = q.shape();
                let q = q.flatten_to(1)?;
                let k = k.flatten_to(1)?;
                let v = v.flatten_to(1)?;

                let window_size_left = sliding_window_left(sdpa_params.sliding_window);
                let window_size_right = if params.causal { Some(0) } else { None };

                return candle_flash_attn_v3::flash_attn_varlen_windowed(
                    &q,
                    &k,
                    &v,
                    cumulative_seqlens_q,
                    cumulative_seqlens_k,
                    params.max_q as usize,
                    k_meta.max as usize,
                    sdpa_params.softmax_scale,
                    window_size_left,
                    window_size_right,
                    true,
                )?
                .reshape(qshape);
            }
        }
    }

    let causal = flash_params.map_or(default_causal, |p| p.causal);
    candle_flash_attn_v3::flash_attn_windowed(
        q,
        k,
        v,
        sdpa_params.softmax_scale,
        sliding_window_left(sdpa_params.sliding_window),
        causal.then_some(0),
        true,
    )
}

/// Flash attention over `(b, seq, heads, head_dim)` operands; None when no built kernel takes the call.
pub fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Option<Tensor>> {
    let head_dim = q.dim(3)?;
    #[cfg(not(feature = "cuda"))]
    let _ = (k, v, flash_params);
    // v3 wins on single-sequence prefill and ties elsewhere, so it takes the head dims it supports.
    // v3 never sets is_local, so its sliding window is a no-op; leave those to v2.
    #[cfg(feature = "flash-attn-v3")]
    if matches!(head_dim, 64 | 128 | 256 | 512)
        && sdpa_params.softcap.is_none()
        && sdpa_params.sliding_window.is_none()
    {
        return flash_attn_v3(q, k, v, flash_params, sdpa_params).map(Some);
    }
    #[cfg(feature = "cuda")]
    if let Some(out) = try_fattn(q, k, v, flash_params, sdpa_params)? {
        return Ok(Some(out));
    }
    let supported = dao_supports(head_dim, sdpa_params.softcap.is_some())
        && (sdpa_params.sliding_window.is_none()
            || cfg!(feature = "flash-attn") && head_dim <= 256);
    if !supported {
        return Ok(None);
    }
    #[cfg(feature = "flash-attn")]
    {
        flash_attn_v2(q, k, v, flash_params, sdpa_params).map(Some)
    }
    #[cfg(all(feature = "flash-attn-v3", not(feature = "flash-attn")))]
    {
        flash_attn_v3(q, k, v, flash_params, sdpa_params).map(Some)
    }
    #[cfg(not(any(feature = "flash-attn", feature = "flash-attn-v3")))]
    {
        Ok(None)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn packed_varlen_metadata_fails_closed() {
        let q = Tensor::zeros((1, 1, 1, 1), DType::F32, &Device::Cpu).unwrap();
        let mut params = FlashParams::empty(true);
        params.packed = true;

        let missing_query = varlen_metadata(&q, &params, None).unwrap_err();

        assert!(
            missing_query
                .to_string()
                .contains("packed prefill is missing query metadata")
        );

        params.cumulative_seqlens_q.insert(
            Device::Cpu.location(),
            Tensor::new(&[0u32, 1], &Device::Cpu).unwrap(),
        );
        let missing_key = varlen_metadata(&q, &params, None).unwrap_err();

        assert!(
            missing_key
                .to_string()
                .contains("packed prefill is missing key metadata")
        );
    }

    #[test]
    fn varlen_metadata_selects_the_sliding_physical_layout() {
        let q = Tensor::zeros((1, 1, 1, 1), DType::F32, &Device::Cpu).unwrap();
        let location = Device::Cpu.location();
        let mut params = FlashParams::empty(true);
        params
            .cumulative_seqlens_q
            .insert(location, Tensor::new(&[0u32, 1], &Device::Cpu).unwrap());
        params.logical_k = FlashKMeta {
            max: 101,
            cumulative_seqlens: HashMap::from([(
                location,
                Tensor::new(&[0u32, 101], &Device::Cpu).unwrap(),
            )]),
        };
        params.sliding_k = Some(FlashKMeta {
            max: 4,
            cumulative_seqlens: HashMap::from([(
                location,
                Tensor::new(&[0u32, 4], &Device::Cpu).unwrap(),
            )]),
        });

        let (_, k_meta, cumulative_k) = varlen_metadata(&q, &params, Some(4)).unwrap().unwrap();

        assert_eq!(k_meta.max, 4);
        assert_eq!(cumulative_k.to_vec1::<u32>().unwrap(), vec![0u32, 4]);
    }

    #[test]
    fn backend_capabilities_reject_unsupported_softcap_and_head_dims() {
        assert!(!flash_backend_supports(640, false));
        assert!(flash_backend_supports(128, true));
        assert_eq!(
            flash_backend_supports(64, true),
            cfg!(feature = "flash-attn")
        );
        assert!(!flash_backend_supports(512, true));
        assert_eq!(
            flash_backend_supports(320, false),
            cfg!(feature = "flash-attn")
        );
        assert!(!flash_backend_supports(320, true));
        assert!(!flash_backend_supports_sdpa(320, false, true));
        assert_eq!(
            flash_backend_supports_sdpa(512, false, true),
            cfg!(all(feature = "flash-attn-v3", not(feature = "flash-attn")))
        );
    }
}
