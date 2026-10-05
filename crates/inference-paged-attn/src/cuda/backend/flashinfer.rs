use crate::cuda::backend::{cache_input_layout, slice_ptr};
use crate::cuda::ffi::{
    gather_kv_cache_flashinfer as ffi_gather_kv_cache_flashinfer,
    reshape_and_cache_flashinfer as ffi_reshape_and_cache_flashinfer,
};
use candle_core::backend::BackendStorage;
use candle_core::{DType, Result, Storage, Tensor};
use float8::F8E4M3;

use crate::{DEFAULT_FP8_KV_CACHE_SCALES, KvCacheScales};

fn dtype_code(dtype: DType, op: &str) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F32 => Ok(2),
        other => candle_core::bail!("{op} only supports f16, bf16, f32 (got {other:?})"),
    }
}

fn cache_dtype_code(dtype: DType, op: &str) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F32 => Ok(2),
        DType::F8E4M3 if crate::cuda::USE_FP8 => Ok(3),
        DType::F8E4M3 => candle_core::bail!("{op} requires FP8 CUDA support"),
        other => {
            candle_core::bail!("{op} only supports f16, bf16, f32, f8e4m3 cache (got {other:?})")
        }
    }
}

fn validate_cache_dtype(activation_dtype: DType, cache_dtype: DType, op: &str) -> Result<()> {
    dtype_code(activation_dtype, op)?;
    cache_dtype_code(cache_dtype, op)?;
    if cache_dtype != activation_dtype && cache_dtype != DType::F8E4M3 {
        candle_core::bail!(
            "{op} requires matching activation/cache dtypes or an f8e4m3 cache, got activation={activation_dtype:?}, cache={cache_dtype:?}"
        );
    }
    Ok(())
}

fn validate_cache_scales(cache_dtype: DType, scales: KvCacheScales, op: &str) -> Result<()> {
    if !scales.k.is_finite() || scales.k <= 0.0 || !scales.v.is_finite() || scales.v <= 0.0 {
        candle_core::bail!(
            "{op} requires finite positive K/V cache scales, got k={} v={}",
            scales.k,
            scales.v
        );
    }
    if cache_dtype != DType::F8E4M3 && scales != DEFAULT_FP8_KV_CACHE_SCALES {
        candle_core::bail!(
            "{op} only accepts non-unit cache scales for f8e4m3 caches, got cache={cache_dtype:?} k={} v={}",
            scales.k,
            scales.v
        );
    }
    Ok(())
}

pub fn is_flashinfer_cache(key_cache: &Tensor, value_cache: &Tensor) -> bool {
    key_cache.dims().len() == 4
        && value_cache.dims().len() == 4
        && key_cache.dims() == value_cache.dims()
}

#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_flashinfer(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
    scales: KvCacheScales,
) -> Result<()> {
    let dtype = key.dtype();
    let cache_dtype = key_cache.dtype();
    if value.dtype() != dtype || value_cache.dtype() != cache_dtype {
        candle_core::bail!(
            "reshape_and_cache_flashinfer expects matching K/V dtypes and matching cache dtypes, got key={:?}, value={:?}, key_cache={:?}, value_cache={:?}",
            key.dtype(),
            value.dtype(),
            key_cache.dtype(),
            value_cache.dtype()
        );
    }
    validate_cache_dtype(dtype, cache_dtype, "reshape_and_cache_flashinfer")?;
    validate_cache_scales(cache_dtype, scales, "reshape_and_cache_flashinfer")?;
    if slot_mapping.dtype() != DType::I64 {
        candle_core::bail!("reshape_and_cache_flashinfer expects i64 slot_mapping");
    }

    let (key_s, key_l) = key.storage_and_layout();
    let (value_s, value_l) = value.storage_and_layout();
    let (num_tokens, num_heads, head_size, key_stride) =
        cache_input_layout(key_l, "key", "reshape_and_cache_flashinfer")?;
    let (value_tokens, value_heads, value_head_size, value_stride) =
        cache_input_layout(value_l, "value", "reshape_and_cache_flashinfer")?;
    if (value_tokens, value_heads, value_head_size) != (num_tokens, num_heads, head_size) {
        candle_core::bail!(
            "reshape_and_cache_flashinfer key/value shape mismatch: {:?} vs {:?}",
            key.shape(),
            value.shape()
        );
    }
    let (_, cache_heads, block_size, cache_head_size) = key_cache.dims4()?;
    if value_cache.dims4()? != key_cache.dims4()? {
        candle_core::bail!("reshape_and_cache_flashinfer cache shape mismatch");
    }
    if cache_heads != num_heads || cache_head_size != head_size {
        candle_core::bail!(
            "reshape_and_cache_flashinfer cache shape {:?} incompatible with key {:?}",
            key_cache.shape(),
            key.shape()
        );
    }
    if slot_mapping.dims1()? != num_tokens {
        candle_core::bail!(
            "reshape_and_cache_flashinfer slot_mapping length mismatch: expected {num_tokens}, got {}",
            slot_mapping.dims1()?
        );
    }

    let (key_cache_s, key_cache_l) = key_cache.storage_and_layout();
    let (value_cache_s, value_cache_l) = value_cache.storage_and_layout();
    let (slot_s, slot_l) = slot_mapping.storage_and_layout();

    let key_s = match &*key_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("key must be a cuda tensor"),
    };
    let value_s = match &*value_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("value must be a cuda tensor"),
    };
    let key_cache_s = match &*key_cache_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("key_cache must be a cuda tensor"),
    };
    let value_cache_s = match &*value_cache_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("value_cache must be a cuda tensor"),
    };
    let slot_s = match &*slot_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("slot_mapping must be a cuda tensor"),
    };

    let (key_ptr, _key_guard) = match dtype {
        DType::F16 => slice_ptr(key_s.as_cuda_slice::<half::f16>()?, key_l.start_offset()),
        DType::BF16 => slice_ptr(key_s.as_cuda_slice::<half::bf16>()?, key_l.start_offset()),
        DType::F32 => slice_ptr(key_s.as_cuda_slice::<f32>()?, key_l.start_offset()),
        _ => unreachable!(),
    };
    let (value_ptr, _value_guard) = match dtype {
        DType::F16 => slice_ptr(
            value_s.as_cuda_slice::<half::f16>()?,
            value_l.start_offset(),
        ),
        DType::BF16 => slice_ptr(
            value_s.as_cuda_slice::<half::bf16>()?,
            value_l.start_offset(),
        ),
        DType::F32 => slice_ptr(value_s.as_cuda_slice::<f32>()?, value_l.start_offset()),
        _ => unreachable!(),
    };
    let (key_cache_ptr, _key_cache_guard) = match cache_dtype {
        DType::F16 => slice_ptr(
            key_cache_s.as_cuda_slice::<half::f16>()?,
            key_cache_l.start_offset(),
        ),
        DType::BF16 => slice_ptr(
            key_cache_s.as_cuda_slice::<half::bf16>()?,
            key_cache_l.start_offset(),
        ),
        DType::F32 => slice_ptr(
            key_cache_s.as_cuda_slice::<f32>()?,
            key_cache_l.start_offset(),
        ),
        DType::F8E4M3 => slice_ptr(
            key_cache_s.as_cuda_slice::<F8E4M3>()?,
            key_cache_l.start_offset(),
        ),
        _ => unreachable!(),
    };
    let (value_cache_ptr, _value_cache_guard) = match cache_dtype {
        DType::F16 => slice_ptr(
            value_cache_s.as_cuda_slice::<half::f16>()?,
            value_cache_l.start_offset(),
        ),
        DType::BF16 => slice_ptr(
            value_cache_s.as_cuda_slice::<half::bf16>()?,
            value_cache_l.start_offset(),
        ),
        DType::F32 => slice_ptr(
            value_cache_s.as_cuda_slice::<f32>()?,
            value_cache_l.start_offset(),
        ),
        DType::F8E4M3 => slice_ptr(
            value_cache_s.as_cuda_slice::<F8E4M3>()?,
            value_cache_l.start_offset(),
        ),
        _ => unreachable!(),
    };
    let (slot_ptr, _slot_guard) = slice_ptr(slot_s.as_cuda_slice::<i64>()?, slot_l.start_offset());

    unsafe {
        ffi_reshape_and_cache_flashinfer(
            key_ptr as *const core::ffi::c_void,
            value_ptr as *const core::ffi::c_void,
            key_cache_ptr as *const core::ffi::c_void,
            value_cache_ptr as *const core::ffi::c_void,
            slot_ptr as *const core::ffi::c_long,
            num_tokens as i32,
            num_heads as i32,
            head_size as i32,
            block_size as i32,
            i32::try_from(key_stride).map_err(candle_core::Error::wrap)?,
            i32::try_from(value_stride).map_err(candle_core::Error::wrap)?,
            scales.k,
            scales.v,
            dtype_code(dtype, "reshape_and_cache_flashinfer")?,
            cache_dtype_code(cache_dtype, "reshape_and_cache_flashinfer")?,
            key_s.device().cuda_stream().cu_stream(),
        );
    }

    Ok(())
}

pub fn gather_kv_cache_flashinfer(
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_table: &Tensor,
    cu_seq_lens: &Tensor,
    num_tokens: usize, // Must equal cu_seq_lens[-1].
    out_dtype: DType,
    scales: KvCacheScales,
) -> Result<(Tensor, Tensor)> {
    let cache_dtype = key_cache.dtype();
    if value_cache.dtype() != cache_dtype {
        candle_core::bail!("gather_kv_cache_flashinfer expects matching cache dtypes");
    }
    validate_cache_dtype(out_dtype, cache_dtype, "gather_kv_cache_flashinfer")?;
    validate_cache_scales(cache_dtype, scales, "gather_kv_cache_flashinfer")?;

    let (_, num_kv_heads, block_size, head_size) = key_cache.dims4()?;
    if value_cache.dims4()? != key_cache.dims4()? {
        candle_core::bail!("gather_kv_cache_flashinfer cache shape mismatch");
    }

    let block_table = block_table.contiguous()?;
    let cu_seq_lens = cu_seq_lens.contiguous()?;
    if !matches!(block_table.dtype(), DType::I32 | DType::U32)
        || !matches!(cu_seq_lens.dtype(), DType::I32 | DType::U32)
    {
        candle_core::bail!("gather_kv_cache_flashinfer expects i32/u32 metadata");
    }

    let cu_len = cu_seq_lens.dims1()?;
    let num_seqs = cu_len
        .checked_sub(1)
        .ok_or_else(|| candle_core::Error::msg("cu_seq_lens must contain an initial offset"))?;
    let num_tokens_i32 = i32::try_from(num_tokens)
        .map_err(|_| candle_core::Error::msg("num_tokens exceeds the kernel i32 limit"))?;
    let num_seqs_i32 = i32::try_from(num_seqs)
        .map_err(|_| candle_core::Error::msg("num_seqs exceeds the kernel i32 limit"))?;

    let k_out = unsafe {
        Tensor::empty(
            (num_tokens, num_kv_heads, head_size),
            out_dtype,
            key_cache.device(),
        )?
    };
    let v_out = unsafe {
        Tensor::empty(
            (num_tokens, num_kv_heads, head_size),
            out_dtype,
            value_cache.device(),
        )?
    };
    if num_tokens == 0 {
        return Ok((k_out, v_out));
    }

    let (kc_s, kc_l) = key_cache.storage_and_layout();
    let (vc_s, vc_l) = value_cache.storage_and_layout();
    let (ko_s, ko_l) = k_out.storage_and_layout();
    let (vo_s, vo_l) = v_out.storage_and_layout();
    let (bt_s, bt_l) = block_table.storage_and_layout();
    let (cu_s, cu_l) = cu_seq_lens.storage_and_layout();

    let kc_s = match &*kc_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("key_cache must be a cuda tensor"),
    };
    let vc_s = match &*vc_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("value_cache must be a cuda tensor"),
    };
    let ko_s = match &*ko_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("k_out must be a cuda tensor"),
    };
    let vo_s = match &*vo_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("v_out must be a cuda tensor"),
    };
    let bt_s = match &*bt_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("block_table must be a cuda tensor"),
    };
    let cu_s = match &*cu_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("cu_seq_lens must be a cuda tensor"),
    };

    let (kc_ptr, _kc_guard) = match cache_dtype {
        DType::F16 => slice_ptr(kc_s.as_cuda_slice::<half::f16>()?, kc_l.start_offset()),
        DType::BF16 => slice_ptr(kc_s.as_cuda_slice::<half::bf16>()?, kc_l.start_offset()),
        DType::F32 => slice_ptr(kc_s.as_cuda_slice::<f32>()?, kc_l.start_offset()),
        DType::F8E4M3 => slice_ptr(kc_s.as_cuda_slice::<F8E4M3>()?, kc_l.start_offset()),
        _ => unreachable!(),
    };
    let (vc_ptr, _vc_guard) = match cache_dtype {
        DType::F16 => slice_ptr(vc_s.as_cuda_slice::<half::f16>()?, vc_l.start_offset()),
        DType::BF16 => slice_ptr(vc_s.as_cuda_slice::<half::bf16>()?, vc_l.start_offset()),
        DType::F32 => slice_ptr(vc_s.as_cuda_slice::<f32>()?, vc_l.start_offset()),
        DType::F8E4M3 => slice_ptr(vc_s.as_cuda_slice::<F8E4M3>()?, vc_l.start_offset()),
        _ => unreachable!(),
    };
    let (ko_ptr, _ko_guard) = match out_dtype {
        DType::F16 => slice_ptr(ko_s.as_cuda_slice::<half::f16>()?, ko_l.start_offset()),
        DType::BF16 => slice_ptr(ko_s.as_cuda_slice::<half::bf16>()?, ko_l.start_offset()),
        DType::F32 => slice_ptr(ko_s.as_cuda_slice::<f32>()?, ko_l.start_offset()),
        _ => unreachable!(),
    };
    let (vo_ptr, _vo_guard) = match out_dtype {
        DType::F16 => slice_ptr(vo_s.as_cuda_slice::<half::f16>()?, vo_l.start_offset()),
        DType::BF16 => slice_ptr(vo_s.as_cuda_slice::<half::bf16>()?, vo_l.start_offset()),
        DType::F32 => slice_ptr(vo_s.as_cuda_slice::<f32>()?, vo_l.start_offset()),
        _ => unreachable!(),
    };
    let (bt_ptr, _bt_guard) = if block_table.dtype() == DType::I32 {
        slice_ptr(bt_s.as_cuda_slice::<i32>()?, bt_l.start_offset())
    } else {
        slice_ptr(bt_s.as_cuda_slice::<u32>()?, bt_l.start_offset())
    };
    let (cu_ptr, _cu_guard) = if cu_seq_lens.dtype() == DType::I32 {
        slice_ptr(cu_s.as_cuda_slice::<i32>()?, cu_l.start_offset())
    } else {
        slice_ptr(cu_s.as_cuda_slice::<u32>()?, cu_l.start_offset())
    };
    let (_, block_table_stride) = bt_l.shape().dims2()?;

    unsafe {
        ffi_gather_kv_cache_flashinfer(
            kc_ptr as *const core::ffi::c_void,
            vc_ptr as *const core::ffi::c_void,
            ko_ptr as *const core::ffi::c_void,
            vo_ptr as *const core::ffi::c_void,
            bt_ptr as *const i32,
            cu_ptr as *const i32,
            num_tokens_i32,
            num_seqs_i32,
            block_size as i32,
            block_table_stride as i32,
            num_kv_heads as i32,
            head_size as i32,
            dtype_code(out_dtype, "gather_kv_cache_flashinfer")?,
            cache_dtype_code(cache_dtype, "gather_kv_cache_flashinfer")?,
            scales.k,
            scales.v,
            kc_s.device().cuda_stream().cu_stream(),
        );
    }

    Ok((k_out.clone(), v_out.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    const BLOCK_SIZE: usize = 8;
    const HEAD_SIZE: usize = 64;
    const TEST_FP8_SCALES: KvCacheScales = KvCacheScales { k: 0.25, v: 0.5 };

    fn cuda_tensor(
        values: Vec<f32>,
        shape: (usize, usize, usize),
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        Tensor::from_vec(values, shape, &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(device)
    }

    fn cache(dtype: DType, device: &Device) -> Result<Tensor> {
        unsafe { Tensor::empty((1, 1, BLOCK_SIZE, HEAD_SIZE), dtype, device) }
    }

    fn max_diff(lhs: &Tensor, rhs: &Tensor) -> Result<f32> {
        let lhs = lhs
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let rhs = rhs
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(lhs
            .into_iter()
            .zip(rhs)
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max))
    }

    #[test]
    fn mixed_f16_bf16_fp8_hnd_write_and_gather() -> Result<()> {
        if !crate::cuda::USE_FP8 {
            return Ok(());
        }
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let values = (0..2 * HEAD_SIZE)
            .map(|idx| (idx as f32 % 17.0 - 8.0) / 8.0)
            .collect::<Vec<_>>();
        let block_table = Tensor::new(&[[0i32]], &device)?;
        let cu_seq_lens = Tensor::new(&[0i32, 2], &device)?;
        let slots = Tensor::new(&[0i64, 1], &device)?;
        for dtype in [DType::F16, DType::BF16] {
            let key = cuda_tensor(values.clone(), (2, 1, HEAD_SIZE), dtype, &device)?;
            let value = cuda_tensor(values.clone(), (2, 1, HEAD_SIZE), dtype, &device)?;
            let key_cache = cache(DType::F8E4M3, &device)?;
            let value_cache = cache(DType::F8E4M3, &device)?;
            reshape_and_cache_flashinfer(
                &key,
                &value,
                &key_cache,
                &value_cache,
                &slots,
                TEST_FP8_SCALES,
            )?;

            let (gathered_key, gathered_value) = gather_kv_cache_flashinfer(
                &key_cache,
                &value_cache,
                &block_table,
                &cu_seq_lens,
                2,
                dtype,
                TEST_FP8_SCALES,
            )?;
            assert!(max_diff(&key, &gathered_key)? <= 0.063);
            assert!(max_diff(&value, &gathered_value)? <= 0.063);
        }
        Ok(())
    }

    #[test]
    fn cache_scale_validation_rejects_invalid_values() {
        assert!(
            validate_cache_scales(DType::F8E4M3, KvCacheScales { k: 0.0, v: 1.0 }, "test").is_err()
        );
        assert!(validate_cache_scales(DType::BF16, TEST_FP8_SCALES, "test").is_err());
    }
}
