use inference_tensor::{Result, Tensor};

#[cfg(feature = "cuda")]
use crate::attention::FlashKMeta;
#[cfg(feature = "cuda")]
use crate::attention::sliding_window_left;
use crate::attention::{FlashParams, SdpaParams};

// Head dims fattn takes in every layout (dense, packed, any GQA ratio): 192/320/512/576 run only GQA-batched
const FATTN_HEAD_DIMS: [usize; 6] = [64, 80, 96, 112, 128, 256];
// fattn instantiates softcap only at these (and 512)
const FATTN_SOFTCAP_HEAD_DIMS: [usize; 2] = [128, 256];
// Head dims only fattn's tile kernel takes (no mma, paged or packed instance): SigLIP-style vision towers (72)
const FATTN_TILE_HEAD_DIMS: [usize; 2] = [40, 72];
// f16 elements per 16-byte mask row chunk
#[cfg(feature = "cuda")]
const FATTN_MASK_ROW_ALIGN: usize = 8;

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
        .map(|sinks| sinks.to_dtype(inference_tensor::DType::F32)?.contiguous())
        .transpose()
}

// Dao-AILab's FA3 (Hopper), ahead of fattn for the head dims it takes when built
fn fa3_supports(head_dim: usize, has_softcap: bool) -> bool {
    cfg!(feature = "flash-attn-v3") && matches!(head_dim, 64 | 128 | 256 | 512) && !has_softcap
}

pub fn flash_backend_supports(head_dim: usize, has_softcap: bool) -> bool {
    fattn_supports(head_dim, has_softcap) || fa3_supports(head_dim, has_softcap)
}

/// Whether dense attention (unpaged, unpacked) at `head_dim` may try fattn: every layout's dims, and the tile-only
/// ones without softcap or a window. The call itself still asks fattn, so a shape it can't take falls back.
pub fn flash_backend_supports_dense_sdpa(
    head_dim: usize,
    has_softcap: bool,
    has_sliding_window: bool,
) -> bool {
    flash_backend_supports_sdpa(head_dim, has_softcap, has_sliding_window)
        || fattn_tile_available()
            && !has_softcap
            && !has_sliding_window
            && FATTN_TILE_HEAD_DIMS.contains(&head_dim)
}

fn fattn_tile_available() -> bool {
    #[cfg(feature = "cuda")]
    let available = inference_fattn::mma_available();
    #[cfg(not(feature = "cuda"))]
    let available = false;
    available
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
            inference_tensor::bail!("packed prefill is missing query metadata for {location:?}");
        }
        return Ok(None);
    };
    let k_meta = params.k_meta(sliding_window);
    let Some(cumulative_seqlens_k) = k_meta.cumulative_seqlens.get(&location) else {
        if params.packed {
            inference_tensor::bail!("packed prefill is missing key metadata for {location:?}");
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

// The mask as fattn takes it, (batch | 1, seq_q, seq_kv) in f16; None when it differs per head or does not fit
#[cfg(feature = "cuda")]
fn fattn_mask(mask: &Tensor, b_sz: usize, seq_q: usize, seq_kv: usize) -> Result<Option<Tensor>> {
    let mask = match mask.rank() {
        2 => mask.unsqueeze(0)?,
        3 if mask.dim(0)? == 1 => mask.clone(),
        4 if mask.dim(1)? == 1 || mask.stride()[1] == 0 => mask.narrow(1, 0, 1)?.squeeze(1)?,
        _ => return Ok(None),
    };
    // a batch broadcast by stride is one mask for every sequence
    let mask = if mask.dim(0)? > 1 && mask.stride()[0] == 0 {
        mask.narrow(0, 0, 1)?
    } else {
        mask
    };
    let (mb, mq, mkv) = mask.dims3()?;
    if mkv != seq_kv || !(mq == 1 || mq == seq_q) || !(mb == 1 || mb == b_sz) {
        return Ok(None);
    }
    // a key-padding mask (one row for every query) is expanded: fattn reads one row per query
    let mask = mask
        .to_dtype(inference_tensor::DType::F16)?
        .broadcast_as((mb, seq_q, seq_kv))?;
    // a query seeing no key comes out NaN and spreads via masked keys (0 * NaN); it sees all, as HF's unmask_unattended
    let unattended = mask
        .max_keepdim(inference_tensor::D::Minus1)?
        .eq(f64::NEG_INFINITY)?
        .broadcast_as(mask.shape())?;
    let zero = Tensor::zeros((), inference_tensor::DType::F16, mask.device())?
        .broadcast_as(mask.shape())?;
    let mask = unattended.where_cond(&zero, &mask)?;
    // fattn copies mask rows 16 bytes at a time, so each row starts on a whole number of them
    let padded = seq_kv.next_multiple_of(FATTN_MASK_ROW_ALIGN);
    Ok(Some(
        mask.pad_with_zeros(2, 0, padded - seq_kv)?
            .narrow(2, 0, seq_kv)?,
    ))
}

/// fattn over `(b, seq, heads, head_dim)` with an additive mask carrying all the masking; None for per-head masks.
#[cfg(feature = "cuda")]
pub(crate) fn fattn_masked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    sdpa_params: &SdpaParams,
) -> Result<Option<Tensor>> {
    let (b_sz, seq_q, _, _) = q.dims4()?;
    let Some(mask) = fattn_mask(mask, b_sz, seq_q, k.dim(1)?)? else {
        return Ok(None);
    };
    let opts = inference_fattn::FattnOptions {
        scale: sdpa_params.softmax_scale,
        softcap: sdpa_params.softcap.unwrap_or(0.),
        mask: Some(mask),
        ..Default::default()
    };
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
        inference_tensor::bail!("FlashAttention v3 does not support attention softcap");
    }
    let head_dim = q.dim(3)?;
    if !matches!(head_dim, 64 | 128 | 256 | 512) {
        inference_tensor::bail!("FlashAttention v3 does not support head_dim={head_dim}");
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

                return inference_flash_attn_v3::flash_attn_varlen_windowed(
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
    inference_flash_attn_v3::flash_attn_windowed(
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
        let head_dim = q.dim(3)?;
        if let Some(padded) = mma_padding(head_dim)
            && flash_params.is_none_or(|params| !params.packed)
            && sdpa_params.sinks.is_none()
        {
            // zero channels add nothing to q.k or the output, and the scale is explicit, so only the kernel changes
            let (qp, kp, vp) = (
                pad_head_dim(q, padded)?,
                pad_head_dim(k, padded)?,
                pad_head_dim(v, padded)?,
            );
            if let Some(out) = try_fattn(&qp, &kp, &vp, flash_params, sdpa_params)? {
                return Ok(Some(out.narrow(
                    inference_tensor::D::Minus1,
                    0,
                    head_dim,
                )?));
            }
        }
        try_fattn(q, k, v, flash_params, sdpa_params)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q, k, v, flash_params, sdpa_params);
        Ok(None)
    }
}

// a tile-only head dim padded to the next one fattn's tensor-core (mma) kernel takes, which runs it 2-3x faster
#[cfg(feature = "cuda")]
fn mma_padding(head_dim: usize) -> Option<usize> {
    match head_dim {
        40 => Some(64),
        72 => Some(80),
        _ => None,
    }
}

// padded in storage order: padding a `(b, seq, heads, d)` view of `(b, heads, seq, d)` scatters the head dim
#[cfg(feature = "cuda")]
fn pad_head_dim(t: &Tensor, padded: usize) -> Result<Tensor> {
    let extra = padded - t.dim(inference_tensor::D::Minus1)?;
    let heads_major = t.transpose(1, 2)?;
    if heads_major.is_contiguous() {
        heads_major
            .pad_with_zeros(inference_tensor::D::Minus1, 0, extra)?
            .transpose(1, 2)
    } else {
        t.contiguous()?
            .pad_with_zeros(inference_tensor::D::Minus1, 0, extra)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use inference_tensor::{DType, Device};

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

    // vision towers hand over `(b, seq, heads, d)` views of heads-major storage; padding must keep rows fattn can read
    #[test]
    fn padded_head_dims_stay_readable_by_the_mma_kernel() -> Result<()> {
        crate::skip_without_cuda!();
        let dev = Device::new_cuda(0)?;
        let (h, len) = (16, 300);
        for (d, padded) in [(40, 64), (72, 80)] {
            assert_eq!(mma_padding(d), Some(padded));
            let heads_major = Tensor::zeros((1, h, len, d), DType::BF16, &dev)?.transpose(1, 2)?;
            let seq_major = Tensor::zeros((1, len, h, d), DType::BF16, &dev)?;
            for t in [heads_major, seq_major] {
                let p = pad_head_dim(&t, padded)?;
                assert_eq!(p.dims(), [1, len, h, padded]);
                assert_eq!(p.stride()[3], 1, "head dim {d} scattered by the pad");
                let opts = inference_fattn::FattnOptions {
                    scale: 1.,
                    ..Default::default()
                };
                assert!(inference_fattn::supported(&p, &p, &p, &opts)?);
            }
        }
        Ok(())
    }
}
