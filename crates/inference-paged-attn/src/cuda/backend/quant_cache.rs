use inference_tensor::backend::BackendStorage;
use inference_tensor::cuda_backend::CudaStorage;
use inference_tensor::{DType, Result, Storage, Tensor};

use crate::cuda::backend::{cache_input_layout, slice_ptr};
use crate::cuda::ffi;
use crate::kv_quant;

fn dtype_code(dtype: DType, op: &str) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F32 => Ok(2),
        other => inference_tensor::bail!("{op} only supports f16, bf16, f32 (got {other:?})"),
    }
}

fn cuda<'a>(storage: &'a Storage, name: &str) -> Result<&'a CudaStorage> {
    match storage {
        Storage::Cuda(s) => Ok(s),
        _ => inference_tensor::bail!("{name} must be a cuda tensor"),
    }
}

type Guard<'a> = inference_tensor::cuda::cudarc::driver::SyncOnDrop<'a>;

// The raw device pointer of an f16, bf16, f32, i32, u32 or u8 storage at `offset` elements.
fn raw<'a>(
    s: &'a CudaStorage,
    dtype: DType,
    offset: usize,
    name: &str,
) -> Result<(u64, Guard<'a>)> {
    Ok(match dtype {
        DType::F16 => slice_ptr(s.as_cuda_slice::<half::f16>()?, offset),
        DType::BF16 => slice_ptr(s.as_cuda_slice::<half::bf16>()?, offset),
        DType::F32 => slice_ptr(s.as_cuda_slice::<f32>()?, offset),
        DType::I32 => slice_ptr(s.as_cuda_slice::<i32>()?, offset),
        DType::U32 => slice_ptr(s.as_cuda_slice::<u32>()?, offset),
        DType::U8 => slice_ptr(s.as_cuda_slice::<u8>()?, offset),
        other => inference_tensor::bail!("{name} has unsupported dtype {other:?}"),
    })
}

/// The bit widths of a quantized HND cache pair (`[blocks, kv_heads, block_size, row_bytes]` u8) of `head_dim` values.
pub fn quant_cache_bits(
    key_cache: &Tensor,
    value_cache: &Tensor,
    head_dim: usize,
) -> Option<(u32, u32)> {
    if key_cache.dtype() != DType::U8 || value_cache.dtype() != DType::U8 {
        return None;
    }
    let (k_blocks, k_heads, k_block_size, k_row) = key_cache.dims4().ok()?;
    let (v_blocks, v_heads, v_block_size, v_row) = value_cache.dims4().ok()?;
    if (k_blocks, k_heads, k_block_size) != (v_blocks, v_heads, v_block_size) {
        return None;
    }
    Some((
        kv_quant::bits_of(k_row, head_dim)?,
        kv_quant::bits_of(v_row, head_dim)?,
    ))
}

/// Quantizes `key`/`value` (`[tokens, heads, head_dim]`) into the quantized HND caches at `slot_mapping`.
pub fn reshape_and_cache_quant(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    const OP: &str = "reshape_and_cache_quant";
    if value.dtype() != key.dtype() || slot_mapping.dtype() != DType::I64 {
        inference_tensor::bail!("{OP} expects matching K/V dtypes and an i64 slot_mapping");
    }
    let (num_tokens, num_heads, head_size, _) = cache_input_layout(key.layout(), "key", OP)?;
    let (v_tokens, v_heads, v_head_size, _) = cache_input_layout(value.layout(), "value", OP)?;
    if (v_tokens, v_heads, v_head_size) != (num_tokens, num_heads, head_size) {
        inference_tensor::bail!(
            "{OP} key/value shape mismatch: {:?} vs {:?}",
            key.shape(),
            value.shape()
        );
    }
    let Some((k_bits, v_bits)) = quant_cache_bits(key_cache, value_cache, head_size) else {
        inference_tensor::bail!(
            "{OP} needs quantized caches for head dim {head_size}, got {:?} and {:?}",
            key_cache.shape(),
            value_cache.shape()
        );
    };
    if key_cache.dim(1)? != num_heads || slot_mapping.dims1()? != num_tokens {
        inference_tensor::bail!(
            "{OP} cache {:?} incompatible with key {:?}",
            key_cache.shape(),
            key.shape()
        );
    }
    if num_tokens == 0 {
        return Ok(());
    }
    write_rows(
        key,
        key_cache,
        slot_mapping,
        k_bits,
        kv_quant::k_rotation(head_size),
    )?;
    write_rows(
        value,
        value_cache,
        slot_mapping,
        v_bits,
        kv_quant::v_rotation(head_size),
    )
}

// One cache's write: `src`'s rows rotated `width` wide and quantized to `bits`.
fn write_rows(
    src: &Tensor,
    cache: &Tensor,
    slot_mapping: &Tensor,
    bits: u32,
    width: usize,
) -> Result<()> {
    const OP: &str = "reshape_and_cache_quant";
    let (num_tokens, num_heads, head_size, stride) = cache_input_layout(src.layout(), "src", OP)?;
    let (_, _, block_size, _) = cache.dims4()?;
    let (src_s, src_l) = src.storage_and_layout();
    let (cache_s, cache_l) = cache.storage_and_layout();
    let (slot_s, slot_l) = slot_mapping.storage_and_layout();
    let src_s = cuda(&src_s, "src")?;
    let (src_ptr, _src) = raw(src_s, src.dtype(), src_l.start_offset(), "src")?;
    let (cache_ptr, _cache) = raw(
        cuda(&cache_s, "cache")?,
        DType::U8,
        cache_l.start_offset(),
        "cache",
    )?;
    let (slot_ptr, _slot) = slice_ptr(
        cuda(&slot_s, "slot_mapping")?.as_cuda_slice::<i64>()?,
        slot_l.start_offset(),
    );
    let int = |v: usize| i32::try_from(v).map_err(inference_tensor::Error::wrap);
    // SAFETY: every pointer is a live cuda allocation of the shape checked by the caller, on the source's stream
    unsafe {
        ffi::reshape_and_cache_quant(
            src_ptr as *const _,
            cache_ptr as *mut _,
            slot_ptr as *const _,
            int(num_tokens)?,
            int(num_heads)?,
            int(head_size)?,
            int(block_size)?,
            int(stride)?,
            bits as i32,
            int(width)?,
            dtype_code(src.dtype(), OP)?,
            src_s.device().cuda_stream().cu_stream(),
        );
    }
    Ok(())
}

/// Dequantizes `num_tokens` rows of each sequence's quantized cache blocks into `[num_tokens, kv_heads, head_dim]`.
pub fn gather_kv_cache_quant(
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_table: &Tensor,
    cu_seq_lens: &Tensor,
    num_tokens: usize,
    head_dim: usize,
    out_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    const OP: &str = "gather_kv_cache_quant";
    dtype_code(out_dtype, OP)?;
    let Some((k_bits, v_bits)) = quant_cache_bits(key_cache, value_cache, head_dim) else {
        inference_tensor::bail!("{OP} needs quantized caches for head dim {head_dim}");
    };
    let block_table = block_table.contiguous()?;
    let cu_seq_lens = cu_seq_lens.contiguous()?;
    if !matches!(block_table.dtype(), DType::I32 | DType::U32)
        || !matches!(cu_seq_lens.dtype(), DType::I32 | DType::U32)
    {
        inference_tensor::bail!("{OP} expects i32/u32 metadata");
    }
    let gather = Gather {
        block_table: &block_table,
        cu_seq_lens: &cu_seq_lens,
        num_tokens,
        head_dim,
        out_dtype,
    };
    Ok((
        gather.rows(key_cache, k_bits, kv_quant::k_rotation(head_dim))?,
        gather.rows(value_cache, v_bits, kv_quant::v_rotation(head_dim))?,
    ))
}

struct Gather<'a> {
    block_table: &'a Tensor,
    cu_seq_lens: &'a Tensor,
    num_tokens: usize,
    head_dim: usize,
    out_dtype: DType,
}

impl Gather<'_> {
    // One cache's rows, dequantized and rotated back.
    fn rows(&self, cache: &Tensor, bits: u32, width: usize) -> Result<Tensor> {
        let (_, num_kv_heads, block_size, _) = cache.dims4()?;
        let num_seqs = self.cu_seq_lens.dims1()?.checked_sub(1).ok_or_else(|| {
            inference_tensor::Error::msg("cu_seq_lens must contain an initial offset")
        })?;
        // SAFETY: the kernel writes every element, zeros past the last sequence included
        let out = unsafe {
            Tensor::empty(
                (self.num_tokens, num_kv_heads, self.head_dim),
                self.out_dtype,
                cache.device(),
            )?
        };
        if self.num_tokens == 0 {
            return Ok(out);
        }
        {
            let (cache_s, cache_l) = cache.storage_and_layout();
            let (out_s, _) = out.storage_and_layout();
            let (bt_s, bt_l) = self.block_table.storage_and_layout();
            let (cu_s, cu_l) = self.cu_seq_lens.storage_and_layout();
            let cache_s = cuda(&cache_s, "cache")?;
            let (cache_ptr, _c) = raw(cache_s, DType::U8, cache_l.start_offset(), "cache")?;
            let (out_ptr, _o) = raw(cuda(&out_s, "out")?, self.out_dtype, 0, "out")?;
            let (bt_ptr, _bt) = raw(
                cuda(&bt_s, "block_table")?,
                self.block_table.dtype(),
                bt_l.start_offset(),
                "block_table",
            )?;
            let (cu_ptr, _cu) = raw(
                cuda(&cu_s, "cu_seq_lens")?,
                self.cu_seq_lens.dtype(),
                cu_l.start_offset(),
                "cu_seq_lens",
            )?;
            let int = |v: usize| i32::try_from(v).map_err(inference_tensor::Error::wrap);
            // SAFETY: every pointer is a live cuda allocation of the shape checked above, on the cache's stream
            unsafe {
                ffi::gather_kv_cache_quant(
                    cache_ptr as *const _,
                    out_ptr as *mut _,
                    bt_ptr as *const _,
                    cu_ptr as *const _,
                    int(self.num_tokens)?,
                    int(num_seqs)?,
                    int(block_size)?,
                    int(self.block_table.dim(1)?)?,
                    int(num_kv_heads)?,
                    int(self.head_dim)?,
                    bits as i32,
                    int(width)?,
                    dtype_code(self.out_dtype, "gather_kv_cache_quant")?,
                    cache_s.device().cuda_stream().cu_stream(),
                );
            }
        }
        Ok(out)
    }
}

/// `x` with its last dim transformed by the normalized Hadamard in blocks `width` wide (a power of two, 32..512), which
/// is its own inverse. Attention over a quantized cache rotates Q as the cache's K rows were (`kv_quant::k_rotation`)
/// and its output as V's were (`kv_quant::v_rotation`).
pub fn hadamard(x: &Tensor, width: usize) -> Result<Tensor> {
    const OP: &str = "hadamard";
    let code = dtype_code(x.dtype(), OP)?;
    if !x.dim(inference_tensor::D::Minus1)?.is_multiple_of(width) {
        inference_tensor::bail!(
            "{OP} needs a last dim that is a multiple of {width}, got {:?}",
            x.shape()
        );
    }
    let x = x.contiguous()?;
    // SAFETY: the kernel writes every element
    let out = unsafe { Tensor::empty(x.shape(), x.dtype(), x.device())? };
    {
        let (x_s, x_l) = x.storage_and_layout();
        let (out_s, _) = out.storage_and_layout();
        let x_s = cuda(&x_s, "x")?;
        let (src, _s) = raw(x_s, x.dtype(), x_l.start_offset(), "x")?;
        let (dst, _d) = raw(cuda(&out_s, "out")?, x.dtype(), 0, "out")?;
        // SAFETY: both pointers are live contiguous cuda allocations of x's shape, on its stream
        unsafe {
            ffi::hadamard_rows(
                src as *const _,
                dst as *mut _,
                x.elem_count() as i64,
                i32::try_from(width).map_err(inference_tensor::Error::wrap)?,
                code,
                x_s.device().cuda_stream().cu_stream(),
            );
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_tensor::Device;

    const HEAD_DIM: usize = 128;
    const HEADS: usize = 2;
    const BLOCK_SIZE: usize = 8;
    const BLOCKS: usize = 3;
    // f16 scales and the f32 dequantization add a little past the grid's half step
    const ROUNDING: f32 = 1.02;

    // The normalized Hadamard transform `width` wide over each block of `x`, as the kernels apply it.
    fn hadamard_ref(x: &[f32], width: usize) -> Vec<f32> {
        x.chunks(width)
            .flat_map(|block| {
                let mut v = block.to_vec();
                let mut s = 1;
                while s < width {
                    for i in 0..width {
                        if i & s == 0 {
                            let (a, b) = (v[i], v[i | s]);
                            v[i] = a + b;
                            v[i | s] = a - b;
                        }
                    }
                    s <<= 1;
                }
                let norm = (width as f32).sqrt();
                v.into_iter().map(move |x| x / norm)
            })
            .collect()
    }

    // a smooth signal with one outlier per row, the case the rotation is for
    fn values(tokens: usize) -> Vec<f32> {
        (0..tokens * HEADS * HEAD_DIM)
            .map(|i| {
                if i % HEAD_DIM == 5 {
                    12.0
                } else {
                    ((i * 37 % 101) as f32 / 50.0 - 1.0) * 0.7
                }
            })
            .collect()
    }

    fn cache(bits: u32, device: &Device) -> Result<Tensor> {
        Tensor::zeros(
            (
                BLOCKS,
                HEADS,
                BLOCK_SIZE,
                kv_quant::row_bytes(HEAD_DIM, bits),
            ),
            DType::U8,
            device,
        )
    }

    // Over each rotation block, the RMS error the grid allows: half a step of each group's rotated values.
    fn assert_within_half_steps(want: &[f32], got: &[f32], bits: u32, width: usize) {
        let mid = (1u32 << (bits - 1)) as f32;
        for (block, (want, got)) in want.chunks(width).zip(got.chunks(width)).enumerate() {
            let rotated = hadamard_ref(want, width);
            let allowed = (rotated
                .chunks(kv_quant::GROUP)
                .map(|g| (g.iter().fold(0f32, |m, x| m.max(x.abs())) / mid / 2.0).powi(2))
                .sum::<f32>()
                / (width / kv_quant::GROUP) as f32)
                .sqrt();
            let rms = (want
                .iter()
                .zip(got)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                / width as f32)
                .sqrt();
            assert!(
                rms <= allowed * ROUNDING,
                "{bits} bits, block {block}: rms {rms} > {allowed}"
            );
        }
    }

    #[test]
    fn written_rows_gather_back_within_half_a_grid_step() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        // 10 positions over table blocks 2 then 0, plus a token with slot -1, which is never written
        let tokens = 11;
        let mut slots: Vec<i64> = (16..24).chain(0..2).collect();
        slots.push(-1);
        let slots = Tensor::new(slots.as_slice(), &device)?;
        let block_table = Tensor::new(&[[2i32, 0]], &device)?;
        let cu_seq_lens = Tensor::new(&[0i32, 10], &device)?;
        let data = values(tokens);
        let input = Tensor::from_vec(data.clone(), (tokens, HEADS, HEAD_DIM), &device)?;
        for (k_bits, v_bits) in [(8, 8), (6, 4), (5, 3), (4, 2), (7, 5)] {
            let (key_cache, value_cache) = (cache(k_bits, &device)?, cache(v_bits, &device)?);
            assert_eq!(
                quant_cache_bits(&key_cache, &value_cache, HEAD_DIM),
                Some((k_bits, v_bits))
            );
            reshape_and_cache_quant(&input, &input, &key_cache, &value_cache, &slots)?;
            let (k, v) = gather_kv_cache_quant(
                &key_cache,
                &value_cache,
                &block_table,
                &cu_seq_lens,
                12,
                HEAD_DIM,
                DType::F32,
            )?;
            for (out, bits, width) in [
                (k, k_bits, kv_quant::k_rotation(HEAD_DIM)),
                (v, v_bits, kv_quant::v_rotation(HEAD_DIM)),
            ] {
                let out = out.flatten_all()?.to_vec1::<f32>()?;
                let rows = 10 * HEADS * HEAD_DIM;
                assert_within_half_steps(&data[..rows], &out[..rows], bits, width);
                // rows past the sequences come back as zeros
                assert!(out[rows..].iter().all(|x| *x == 0.0));
            }
        }
        Ok(())
    }

    #[test]
    fn the_rotation_is_its_own_inverse_and_matches_the_reference() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let data = values(3);
        let x = Tensor::from_vec(data.clone(), (3, HEADS, HEAD_DIM), &device)?;
        for width in [32, 64, 128] {
            let want = hadamard_ref(&data, width);
            let once = hadamard(&x, width)?.flatten_all()?.to_vec1::<f32>()?;
            assert!(
                once.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-4),
                "width {width}"
            );
            let twice = hadamard(&hadamard(&x, width)?, width)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert!(
                twice.iter().zip(&data).all(|(a, b)| (a - b).abs() < 1e-4),
                "width {width}"
            );
            let bf16 = hadamard(&x.to_dtype(DType::BF16)?, width)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert!(
                bf16.iter()
                    .zip(&want)
                    .all(|(a, b)| (a - b).abs() < 0.05 * b.abs().max(1.0)),
                "width {width}"
            );
        }
        Ok(())
    }
}
