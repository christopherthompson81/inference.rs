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

pub fn fattn_supports(head_dim: usize, has_softcap: bool) -> bool {
    #[cfg(feature = "cuda")]
    let available = inference_fattn::mma_available();
    #[cfg(not(feature = "cuda"))]
    let available = false;
    available
        && FATTN_HEAD_DIMS.contains(&head_dim)
        && (!has_softcap || FATTN_SOFTCAP_HEAD_DIMS.contains(&head_dim))
}

/// Sinks as fattn takes them: contiguous f32, one per query head.
#[cfg(feature = "cuda")]
pub fn fattn_sinks(sinks: Option<&Tensor>) -> Result<Option<Tensor>> {
    sinks
        .map(|sinks| sinks.to_dtype(candle_core::DType::F32)?.contiguous())
        .transpose()
}

// Dao-AILab's FA3 (Hopper), ahead of fattn for the head dims it takes when built
fn fa3_supports(head_dim: usize, has_softcap: bool) -> bool {
    cfg!(feature = "flash-attn-v3") && matches!(head_dim, 64 | 128 | 256 | 512) && !has_softcap
}

pub fn flash_backend_supports(head_dim: usize, has_softcap: bool) -> bool {
    fattn_supports(head_dim, has_softcap) || fa3_supports(head_dim, has_softcap)
}

pub fn flash_backend_supports_sdpa(
    head_dim: usize,
    has_softcap: bool,
    has_sliding_window: bool,
) -> bool {
    // FA3 never sets is_local, so it takes no window
    fattn_supports(head_dim, has_softcap)
        || fa3_supports(head_dim, has_softcap) && !has_sliding_window
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

// fattn (llama.cpp's kernels); None for what it cannot take (head dims, softcap, f32 K/V). A non-causal window bounds
// the left only, as FA2's did (window_size_right None)
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
    if !q.device().is_cuda() {
        return Ok(None);
    }
    let opts = FattnOptions {
        scale: sdpa_params.softmax_scale,
        softcap: sdpa_params.softcap.unwrap_or(0.),
        causal,
        window_left: sliding_window_left(sdpa_params.sliding_window),
        sinks: fattn_sinks(sdpa_params.sinks.as_ref())?,
        chunk: sdpa_params.chunk,
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
    // v3 wins on single-sequence prefill and ties elsewhere, so it takes the head dims it supports.
    #[cfg(feature = "flash-attn-v3")]
    if fa3_supports(q.dim(3)?, sdpa_params.softcap.is_some())
        && sdpa_params.sliding_window.is_none()
        && sdpa_params.sinks.is_none()
        && sdpa_params.chunk.is_none()
    {
        return flash_attn_v3(q, k, v, flash_params, sdpa_params).map(Some);
    }
    #[cfg(feature = "cuda")]
    {
        try_fattn(q, k, v, flash_params, sdpa_params)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q, k, v, flash_params, sdpa_params);
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
        assert_eq!(flash_backend_supports(128, true), fattn_supports(128, true));
        assert!(!flash_backend_supports(64, true));
        assert!(!flash_backend_supports(512, true));
        assert!(!flash_backend_supports(320, false));
        assert!(!flash_backend_supports_sdpa(320, false, true));
        assert!(!flash_backend_supports_sdpa(512, false, true));
    }
}
