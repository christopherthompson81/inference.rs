use crate::cuda::backend::{cache_input_layout, slice_ptr};
use crate::cuda::ffi;
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::{DType, Result, Storage, Tensor};
use candle_core as candle;
use candle_core::cuda::cudarc::driver::DeviceSlice;
use float8::F8E4M3;
use half::{bf16, f16};
use std::ffi::c_int;

fn validate_kv_cache_scales(
    cache_dtype: DType,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    op: &str,
) -> Result<()> {
    match (cache_dtype, k_scale, v_scale) {
        (DType::F8E4M3, Some(k_scale), Some(v_scale)) => {
            if k_scale.dtype() != DType::F32
                || v_scale.dtype() != DType::F32
                || k_scale.elem_count() != 1
                || v_scale.elem_count() != 1
            {
                candle::bail!("{op} requires scalar f32 K/V scales for an f8e4m3 cache");
            }
        }
        (DType::F8E4M3, _, _) => {
            candle::bail!("{op} requires explicit K/V scales for an f8e4m3 cache");
        }
        (_, None, None) => {}
        (_, _, _) => candle::bail!("{op} only accepts K/V scales for an f8e4m3 cache"),
    }
    Ok(())
}

fn update_cache<
    T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
>(
    key: &Tensor,
    value: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    let dtype = key.dtype();

    let internal_type = match dtype {
        DType::F16 => 0,
        DType::BF16 => 1,
        DType::F32 => 2,
        dtype => candle::bail!("dtype {dtype:?} is not supported"),
    };

    let cache_dtype = match key_cache.dtype() {
        DType::F16 => 0,
        DType::BF16 => 1,
        DType::F32 => 2,
        DType::F8E4M3 => 3,
        dtype => candle::bail!("cache dtype {dtype:?} is not supported"),
    };
    validate_kv_cache_scales(key_cache.dtype(), k_scale, v_scale, "reshape_and_cache")?;

    let (k, k_l) = key.storage_and_layout();
    let k = match &*k {
        Storage::Cuda(k) => k,
        _ => candle::bail!("key must be a cuda tensor"),
    };

    let (v, v_l) = value.storage_and_layout();
    let v = match &*v {
        Storage::Cuda(v) => v,
        _ => candle::bail!("value must be a cuda tensor"),
    };

    let (kc, kc_l) = key_cache.storage_and_layout();
    let kc = match &*kc {
        Storage::Cuda(kc) => kc,
        _ => candle::bail!("key_cache must be a cuda tensor"),
    };

    let (vc, vc_l) = value_cache.storage_and_layout();
    let vc = match &*vc {
        Storage::Cuda(vc) => vc,
        _ => candle::bail!("value_cache must be a cuda tensor"),
    };

    let (s, s_l) = slot_mapping.storage_and_layout();
    let s = match &*s {
        Storage::Cuda(s) => s,
        _ => candle::bail!("slot_mapping must be a cuda tensor"),
    };

    let kc_rank = kc_l.stride().len();
    let vc_rank = vc_l.stride().len();

    if kc_rank != 5 {
        candle::bail!(
            "paged-attention expects `key_cache` tensor to be of rank 5 \
                (key_cache: {kc_l:?})"
        )
    }

    if vc_rank != 4 {
        candle::bail!(
            "paged-attention expects `value_cache` tensor to be of rank 4 \
                (value_cache: {vc_l:?})"
        )
    }

    let dev = k.device();

    // Get cuda slices for all tensors
    let k = k.as_cuda_slice::<T>()?;
    let v = v.as_cuda_slice::<T>()?;
    let s = s.as_cuda_slice::<i64>()?;

    // For FP8 cache, we need to get as u8 slices instead
    let ((kc_ptr, _kc_guard), (vc_ptr, _vc_guard)) = if cache_dtype == 3 {
        if !crate::cuda::USE_FP8 {
            candle::bail!("FP8 is not supported on this system.");
        }

        let kc = kc.as_cuda_slice::<F8E4M3>()?;
        let vc = vc.as_cuda_slice::<F8E4M3>()?;
        (
            slice_ptr(kc, kc_l.start_offset()),
            slice_ptr(vc, vc_l.start_offset()),
        )
    } else {
        let kc = kc.as_cuda_slice::<T>()?;
        let vc = vc.as_cuda_slice::<T>()?;
        (
            slice_ptr(kc, kc_l.start_offset()),
            slice_ptr(vc, vc_l.start_offset()),
        )
    };

    // Get cuda views for all tensors
    let k = k.slice(k_l.start_offset()..);
    let v = v.slice(v_l.start_offset()..);
    let s = s.slice(s_l.start_offset()..);

    let (k_scale_ptr, v_scale_ptr) = if let (Some(k_scale), Some(v_scale)) = (k_scale, v_scale) {
        if !crate::cuda::USE_FP8 {
            candle::bail!("FP8 is not supported on this system.");
        }

        let (ks, ks_l) = k_scale.storage_and_layout();
        let ks = match &*ks {
            Storage::Cuda(ks) => ks,
            _ => candle::bail!("k_scale must be a cuda tensor"),
        };
        let ks = ks.as_cuda_slice::<f32>()?;
        let (ks, _ks_guard) = slice_ptr(ks, ks_l.start_offset());

        let (vs, vs_l) = v_scale.storage_and_layout();
        let vs = match &*vs {
            Storage::Cuda(vs) => vs,
            _ => candle::bail!("v_scale must be a cuda tensor"),
        };
        let vs = vs.as_cuda_slice::<f32>()?;
        let (vs, _vs_guard) = slice_ptr(vs, vs_l.start_offset());

        (ks as *const f32, vs as *const f32)
    } else {
        (std::ptr::null(), std::ptr::null())
    };

    let (num_tokens, num_heads, head_size, key_stride) =
        cache_input_layout(k_l, "key", "paged-attention")?;
    let (value_tokens, value_heads, value_head_size, value_stride) =
        cache_input_layout(v_l, "value", "paged-attention")?;
    if (num_tokens, num_heads, head_size) != (value_tokens, value_heads, value_head_size) {
        candle::bail!("shape mismatch k {:?} and v {:?}", k_l.shape(), v_l.shape())
    }

    let (num_blocks, num_heads_kc, head_size_kc, block_size, x) = kc_l.shape().dims5()?;
    if num_heads_kc != num_heads || head_size_kc != head_size / x {
        candle::bail!(
            "shape mismatch value_cache {:?}, expected {:?}",
            vc_l.shape(),
            (num_blocks, num_heads, head_size / x, block_size, x)
        )
    }

    if (num_blocks, num_heads, head_size, block_size) != vc_l.shape().dims4()? {
        candle::bail!(
            "shape mismatch key_cache {:?} and value_cache {:?}",
            kc_l.shape(),
            vc_l.shape()
        )
    }

    if (num_tokens) != s_l.shape().dims1()? {
        candle::bail!(
            "shape mismatch slot_mapping {:?}, expected {:?}",
            s_l.shape(),
            (num_tokens)
        )
    }

    let key_stride = c_int::try_from(key_stride).map_err(candle::Error::wrap)?;
    let value_stride = c_int::try_from(value_stride).map_err(candle::Error::wrap)?;

    let (k_ptr, _k_guard) = k.device_ptr(k.stream());
    let (v_ptr, _v_guard) = v.device_ptr(v.stream());
    let (s_ptr, _s_guard) = s.device_ptr(s.stream());

    unsafe {
        ffi::reshape_and_cache(
            k_ptr as *const core::ffi::c_void,
            v_ptr as *const core::ffi::c_void,
            kc_ptr as *const core::ffi::c_void,
            vc_ptr as *const core::ffi::c_void,
            s_ptr as *const core::ffi::c_long,
            num_tokens as c_int,
            num_heads as c_int,
            head_size as c_int,
            block_size as c_int,
            x as c_int,
            key_stride,
            value_stride,
            dev.cuda_stream().cu_stream(),
            internal_type,
            cache_dtype,
            k_scale_ptr,
            v_scale_ptr,
        )
    }
    Ok(())
}

/// Insert key and values at the provided slot mapping inside the key value paged cache
///
/// # Arguments
///
/// * `key` - Key tensor shaped `(num_tokens, num_heads, head_size)` or `(batch, seq_len, num_heads, head_size)`.
/// * `value` - Value tensor with the same logical shape as `key`.
/// * `key_cache` - Key cache paged tensor of shape `(num_blocks, num_heads, head_size / x, block_size, x)`
///   with `x` being the size of an element in bytes.
/// * `value_cache` - Value cache paged tensor of shape `(num_blocks, num_heads, head_size, block_size)`.
/// * `slot_mapping` - Mapping associating a slot to each token of shape `(num_tokens)`.
pub fn reshape_and_cache(
    key: &Tensor,
    value: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    match key.dtype() {
        DType::F16 => update_cache::<f16>(
            key,
            value,
            k_scale,
            v_scale,
            key_cache,
            value_cache,
            slot_mapping,
        ),
        DType::BF16 => update_cache::<bf16>(
            key,
            value,
            k_scale,
            v_scale,
            key_cache,
            value_cache,
            slot_mapping,
        ),
        DType::F32 => update_cache::<f32>(
            key,
            value,
            k_scale,
            v_scale,
            key_cache,
            value_cache,
            slot_mapping,
        ),
        dt => {
            candle::bail!("reshape_and_cache is only supported for f32, f16 and bf16 ({dt:?})")
        }
    }
}
