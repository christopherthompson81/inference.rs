#[cfg(feature = "cuda")]
mod ffi;

use inference_tensor::{CpuStorage, Layout, Result, Storage, Tensor, WithDType};
#[cfg(feature = "metal")]
use inference_tensor::{Shape, backend::BackendStorage};
use rayon::prelude::*;

fn cache_dims(l_src: &Layout, l_cos: &Layout, l_sin: &Layout) -> Result<(usize, usize)> {
    let (batch, _, seq_len, head_dim) = l_src.shape().dims4()?;
    let (cos_rows, rot_dim) = match l_cos.shape().dims() {
        [rows, dim] => (*rows, *dim),
        [cos_batch, cos_seq, dim] if *cos_batch == batch && *cos_seq == seq_len => {
            (batch * seq_len, *dim)
        }
        _ => inference_tensor::bail!("invalid RoPE cos shape {:?}", l_cos.shape()),
    };
    let (sin_rows, sin_dim) = match l_sin.shape().dims() {
        [rows, dim] => (*rows, *dim),
        [sin_batch, sin_seq, dim] if *sin_batch == batch && *sin_seq == seq_len => {
            (batch * seq_len, *dim)
        }
        _ => inference_tensor::bail!("invalid RoPE sin shape {:?}", l_sin.shape()),
    };
    if (cos_rows, rot_dim) != (sin_rows, sin_dim) {
        inference_tensor::bail!(
            "RoPE cos/sin shape mismatch {:?} {:?}",
            l_cos.shape(),
            l_sin.shape()
        );
    }
    if cos_rows != seq_len && cos_rows != batch * seq_len {
        inference_tensor::bail!(
            "RoPE cache rows {cos_rows} are incompatible with batch {batch} and seq {seq_len}"
        );
    }
    if rot_dim == 0 || rot_dim * 2 > head_dim {
        inference_tensor::bail!(
            "RoPE rot dim {} is incompatible with head dim {head_dim}",
            rot_dim * 2
        );
    }
    Ok((cos_rows, rot_dim))
}

#[cfg(feature = "metal")]
#[derive(Clone, Copy)]
struct RotaryDims {
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
    rot_dim: usize,
    cache_rows: usize,
}

#[cfg(feature = "metal")]
fn rotary_dims(x: &Tensor, cos: &Tensor, sin: &Tensor, positioned: bool) -> Result<RotaryDims> {
    let (batch, heads, seq_len, head_dim) = x.dims4()?;
    let (cache_rows, rot_dim) = if positioned {
        cos.shape().dims2()?
    } else {
        match cos.dims() {
            [rows, dim] => (*rows, *dim),
            [cos_batch, cos_seq, dim] if *cos_batch == batch && *cos_seq == seq_len => {
                (batch * seq_len, *dim)
            }
            _ => inference_tensor::bail!("invalid RoPE cos shape {:?}", cos.shape()),
        }
    };
    let (sin_rows, sin_dim) = if positioned {
        sin.shape().dims2()?
    } else {
        match sin.dims() {
            [rows, dim] => (*rows, *dim),
            [sin_batch, sin_seq, dim] if *sin_batch == batch && *sin_seq == seq_len => {
                (batch * seq_len, *dim)
            }
            _ => inference_tensor::bail!("invalid RoPE sin shape {:?}", sin.shape()),
        }
    };
    if (cache_rows, rot_dim) != (sin_rows, sin_dim) {
        inference_tensor::bail!(
            "RoPE cos/sin shape mismatch {:?} {:?}",
            cos.shape(),
            sin.shape()
        );
    }
    if !positioned && cache_rows != seq_len && cache_rows != batch * seq_len {
        inference_tensor::bail!(
            "RoPE cache rows {cache_rows} are incompatible with batch {batch} and seq {seq_len}"
        );
    }
    if rot_dim == 0 || rot_dim * 2 > head_dim {
        inference_tensor::bail!(
            "RoPE rot dim {} is incompatible with head dim {head_dim}",
            rot_dim * 2
        );
    }
    Ok(RotaryDims {
        batch,
        heads,
        seq_len,
        head_dim,
        rot_dim,
        cache_rows,
    })
}

fn check_qk_shape(q: &Tensor, k: &Tensor) -> Result<usize> {
    let (batch, _, seq_len, head_dim) = q.dims4()?;
    let (k_batch, k_heads, k_seq_len, k_head_dim) = k.dims4()?;
    if (k_batch, k_seq_len, k_head_dim) != (batch, seq_len, head_dim) {
        inference_tensor::bail!("q/k RoPE shape mismatch {:?} {:?}", q.shape(), k.shape());
    }
    Ok(k_heads)
}

fn typed_slice<'a, T>(xs: &'a [T], layout: &Layout, name: &'static str) -> Result<&'a [T]> {
    match layout.contiguous_offsets() {
        Some((start, end)) => Ok(&xs[start..end]),
        None => inference_tensor::bail!("{name} must be contiguous for RoPE"),
    }
}

fn cpu_positions<'a, G: std::ops::Deref<Target = Storage>>(
    storage_and_layout: &'a Option<(G, &'a Layout)>,
) -> Result<Option<&'a [u32]>> {
    let Some((storage, layout)) = storage_and_layout else {
        return Ok(None);
    };
    let Storage::Cpu(CpuStorage::U32(positions)) = &**storage else {
        inference_tensor::bail!("RoPE positions must be CPU u32");
    };
    Ok(Some(typed_slice(positions, layout, "positions")?))
}

struct CpuRotaryInput<'a, T> {
    src: &'a [T],
    src_l: &'a Layout,
    cos: &'a [T],
    cos_l: &'a Layout,
    sin: &'a [T],
    sin_l: &'a Layout,
    positions: Option<&'a [u32]>,
    is_neox: bool,
}

fn apply_rotary_cpu_inner<T>(input: CpuRotaryInput<'_, T>) -> Result<Tensor>
where
    T: WithDType
        + Copy
        + Send
        + Sync
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>,
{
    let CpuRotaryInput {
        src,
        src_l,
        cos,
        cos_l,
        sin,
        sin_l,
        positions,
        is_neox,
    } = input;
    let src = typed_slice(src, src_l, "RoPE input")?;
    let cos = typed_slice(cos, cos_l, "RoPE cos")?;
    let sin = typed_slice(sin, sin_l, "RoPE sin")?;
    let (batch, heads, seq_len, head_dim) = src_l.shape().dims4()?;
    let positioned = positions.is_some();
    let (cache_rows, rot_dim) = {
        let (cache_rows, rot_dim) = if positioned {
            cos_l.shape().dims2()?
        } else {
            cache_dims(src_l, cos_l, sin_l)?
        };
        if positioned && sin_l.shape().dims2()? != (cache_rows, rot_dim) {
            inference_tensor::bail!(
                "RoPE cos/sin shape mismatch {:?} {:?}",
                cos_l.shape(),
                sin_l.shape()
            );
        }
        if rot_dim == 0 || rot_dim * 2 > head_dim {
            inference_tensor::bail!(
                "RoPE rot dim {} is incompatible with head dim {head_dim}",
                rot_dim * 2
            );
        }
        (cache_rows, rot_dim)
    };
    if let Some(positions) = positions {
        let expected = batch * seq_len;
        if positions.len() != expected {
            inference_tensor::bail!(
                "RoPE positions length {} does not match token count {expected}",
                positions.len()
            );
        }
        for position in positions {
            if *position as usize >= cache_rows {
                inference_tensor::bail!(
                    "RoPE position {} exceeds cache rows {}",
                    position,
                    cache_rows
                );
            }
        }
    }
    let mut dst = src.to_vec();
    dst.par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(row, dst)| {
            let batch_idx = row / (heads * seq_len);
            let seq_idx = row % seq_len;
            let token_idx = batch_idx * seq_len + seq_idx;
            let cache_row = if let Some(positions) = positions {
                positions[token_idx] as usize
            } else if cache_rows == batch * seq_len {
                token_idx
            } else {
                seq_idx
            };
            let cache_offset = cache_row * rot_dim;
            for pair_idx in 0..rot_dim {
                let (x_idx, y_idx) = if is_neox {
                    (pair_idx, pair_idx + rot_dim)
                } else {
                    (pair_idx * 2, pair_idx * 2 + 1)
                };
                let x = dst[x_idx];
                let y = dst[y_idx];
                let cos = cos[cache_offset + pair_idx];
                let sin = sin[cache_offset + pair_idx];
                dst[x_idx] = x * cos - y * sin;
                dst[y_idx] = y * cos + x * sin;
            }
        });
    Tensor::from_vec(dst, src_l.shape().clone(), &inference_tensor::Device::Cpu)
}

fn cpu_apply_rotary_q(
    q: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<Tensor> {
    let q = q.contiguous()?;
    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    let positions = positions.map(Tensor::contiguous).transpose()?;
    let position_storage_and_layout = positions.as_ref().map(Tensor::storage_and_layout);
    let positions = cpu_positions(&position_storage_and_layout)?;

    let (q_s, q_l) = q.storage_and_layout();
    let (cos_s, cos_l) = cos.storage_and_layout();
    let (sin_s, sin_l) = sin.storage_and_layout();
    match (&*q_s, &*cos_s, &*sin_s) {
        (
            Storage::Cpu(CpuStorage::BF16(q)),
            Storage::Cpu(CpuStorage::BF16(cos)),
            Storage::Cpu(CpuStorage::BF16(sin)),
        ) => apply_rotary_cpu_inner(CpuRotaryInput {
            src: q,
            src_l: q_l,
            cos,
            cos_l,
            sin,
            sin_l,
            positions,
            is_neox,
        }),
        (
            Storage::Cpu(CpuStorage::F16(q)),
            Storage::Cpu(CpuStorage::F16(cos)),
            Storage::Cpu(CpuStorage::F16(sin)),
        ) => apply_rotary_cpu_inner(CpuRotaryInput {
            src: q,
            src_l: q_l,
            cos,
            cos_l,
            sin,
            sin_l,
            positions,
            is_neox,
        }),
        (
            Storage::Cpu(CpuStorage::F32(q)),
            Storage::Cpu(CpuStorage::F32(cos)),
            Storage::Cpu(CpuStorage::F32(sin)),
        ) => apply_rotary_cpu_inner(CpuRotaryInput {
            src: q,
            src_l: q_l,
            cos,
            cos_l,
            sin,
            sin_l,
            positions,
            is_neox,
        }),
        (
            Storage::Cpu(CpuStorage::F64(q)),
            Storage::Cpu(CpuStorage::F64(cos)),
            Storage::Cpu(CpuStorage::F64(sin)),
        ) => apply_rotary_cpu_inner(CpuRotaryInput {
            src: q,
            src_l: q_l,
            cos,
            cos_l,
            sin,
            sin_l,
            positions,
            is_neox,
        }),
        _ => inference_tensor::bail!(
            "unsupported CPU RoPE dtype {:?} {:?} {:?}",
            q.dtype(),
            cos.dtype(),
            sin.dtype()
        ),
    }
}

fn cpu_apply_rotary_qk(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<(Tensor, Tensor)> {
    if q.dtype() != k.dtype() {
        inference_tensor::bail!("q/k dtype mismatch {:?} {:?}", q.dtype(), k.dtype());
    }
    check_qk_shape(q, k)?;
    let (q_out, k_out) = rayon::join(
        || cpu_apply_rotary_q(q, cos, sin, positions, is_neox),
        || cpu_apply_rotary_q(k, cos, sin, positions, is_neox),
    );
    Ok((q_out?, k_out?))
}

#[cfg(feature = "cuda")]
fn restore_cuda_rope_layout(
    x: Tensor,
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<Tensor> {
    x.reshape((batch, seq_len, heads, head_dim))?
        .transpose(1, 2)
}

// The CUDA kernels read one cache row per token, so per-batch and batch-shared caches are flattened to that.
#[cfg(feature = "cuda")]
fn cuda_token_cache(cache: &Tensor, batch: usize, seq_len: usize) -> Result<Tensor> {
    match *cache.dims() {
        [b, s, dim] if (b, s) == (batch, seq_len) => cache.reshape((batch * seq_len, dim)),
        [rows, dim] if rows == seq_len && batch > 1 => cache
            .unsqueeze(0)?
            .broadcast_as((batch, seq_len, dim))?
            .reshape((batch * seq_len, dim)),
        _ => Ok(cache.clone()),
    }
}

#[cfg(feature = "cuda")]
fn cuda_apply_rotary_q(
    q: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<Tensor> {
    let (batch, heads, seq_len, head_dim) = q.dims4()?;
    let q_embed = q.transpose(1, 2)?.flatten(0, 1)?;
    if let Some(positions) = positions {
        apply_rotary_inplace_q_positions(&q_embed, cos, sin, positions, is_neox)?;
    } else {
        let cos = cuda_token_cache(cos, batch, seq_len)?;
        let sin = cuda_token_cache(sin, batch, seq_len)?;
        apply_rotary_inplace_q(&q_embed, &cos, &sin, is_neox)?;
    }
    restore_cuda_rope_layout(q_embed, batch, heads, seq_len, head_dim)
}

#[cfg(feature = "cuda")]
fn cuda_apply_rotary_qk(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<(Tensor, Tensor)> {
    let (batch, q_heads, seq_len, head_dim) = q.dims4()?;
    let (k_batch, k_heads, k_seq_len, k_head_dim) = k.dims4()?;
    if (k_batch, k_seq_len, k_head_dim) != (batch, seq_len, head_dim) {
        inference_tensor::bail!("q/k RoPE shape mismatch {:?} {:?}", q.shape(), k.shape());
    }
    let q_embed = q.transpose(1, 2)?.flatten(0, 1)?;
    let k_embed = k.transpose(1, 2)?.flatten(0, 1)?;
    if let Some(positions) = positions {
        apply_rotary_inplace_positions(&q_embed, &k_embed, cos, sin, positions, is_neox)?;
    } else {
        let cos = cuda_token_cache(cos, batch, seq_len)?;
        let sin = cuda_token_cache(sin, batch, seq_len)?;
        apply_rotary_inplace(&q_embed, &k_embed, &cos, &sin, is_neox)?;
    }
    Ok((
        restore_cuda_rope_layout(q_embed, batch, q_heads, seq_len, head_dim)?,
        restore_cuda_rope_layout(k_embed, batch, k_heads, seq_len, head_dim)?,
    ))
}

#[cfg(feature = "metal")]
fn metal_tensor(storage: Storage, shape: Shape) -> Tensor {
    Tensor::from((storage, shape))
}

#[cfg(feature = "metal")]
fn metal_apply_rotary_q(
    q: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<Tensor> {
    use inference_tensor::MetalStorage;

    let q = q.contiguous()?;
    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    let positions = positions.map(Tensor::contiguous).transpose()?;
    let dims = rotary_dims(&q, &cos, &sin, positions.is_some())?;
    if let Some(positions) = positions.as_ref() {
        let expected = dims.batch * dims.seq_len;
        if positions.dtype() != inference_tensor::DType::U32 || positions.dims1()? != expected {
            inference_tensor::bail!("RoPE positions must be u32 with length {expected}");
        }
    }

    let (q_s, q_l) = q.storage_and_layout();
    let (cos_s, cos_l) = cos.storage_and_layout();
    let (sin_s, sin_l) = sin.storage_and_layout();
    let q_s = match &*q_s {
        Storage::Metal(storage) => storage,
        _ => inference_tensor::bail!("q must be a Metal tensor"),
    };
    let cos_s = match &*cos_s {
        Storage::Metal(storage) => storage,
        _ => inference_tensor::bail!("cos must be a Metal tensor"),
    };
    let sin_s = match &*sin_s {
        Storage::Metal(storage) => storage,
        _ => inference_tensor::bail!("sin must be a Metal tensor"),
    };
    let device = q_s.device();
    let output = device.new_buffer(q_l.shape().elem_count(), q_s.dtype(), "rotary-q")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("rotary-q");

    if let Some(positions) = positions.as_ref() {
        let (positions_s, positions_l) = positions.storage_and_layout();
        let positions_s = match &*positions_s {
            Storage::Metal(storage) => storage,
            _ => inference_tensor::bail!("positions must be a Metal tensor"),
        };
        crate::metal_kernels::call_rotary_q_positions(
            device.device(),
            &encoder,
            crate::metal_kernels::Kernels::global(),
            q_s.dtype(),
            q_s.buffer(),
            cos_s.buffer(),
            sin_s.buffer(),
            positions_s.buffer(),
            q_l.start_offset() * q_s.dtype().size_in_bytes(),
            cos_l.start_offset() * cos_s.dtype().size_in_bytes(),
            sin_l.start_offset() * sin_s.dtype().size_in_bytes(),
            positions_l.start_offset() * positions_s.dtype().size_in_bytes(),
            dims.batch,
            dims.heads,
            dims.seq_len,
            dims.head_dim,
            dims.rot_dim,
            is_neox,
            &output,
        )
    } else {
        crate::metal_kernels::call_rotary_q(
            device.device(),
            &encoder,
            crate::metal_kernels::Kernels::global(),
            q_s.dtype(),
            q_s.buffer(),
            cos_s.buffer(),
            sin_s.buffer(),
            q_l.start_offset() * q_s.dtype().size_in_bytes(),
            cos_l.start_offset() * cos_s.dtype().size_in_bytes(),
            sin_l.start_offset() * sin_s.dtype().size_in_bytes(),
            dims.batch,
            dims.heads,
            dims.seq_len,
            dims.head_dim,
            dims.rot_dim,
            dims.cache_rows,
            is_neox,
            &output,
        )
    }
    .map_err(inference_tensor::Error::wrap)?;

    Ok(metal_tensor(
        Storage::Metal(MetalStorage::new(
            output,
            device.clone(),
            q_l.shape().elem_count(),
            q_s.dtype(),
        )),
        q_l.shape().clone(),
    ))
}

#[cfg(feature = "metal")]
fn metal_apply_rotary_qk(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<(Tensor, Tensor)> {
    use inference_tensor::MetalStorage;

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    let positions = positions.map(Tensor::contiguous).transpose()?;
    let dims = rotary_dims(&q, &cos, &sin, positions.is_some())?;
    let k_heads = check_qk_shape(&q, &k)?;
    if let Some(positions) = positions.as_ref() {
        let expected = dims.batch * dims.seq_len;
        if positions.dtype() != inference_tensor::DType::U32 || positions.dims1()? != expected {
            inference_tensor::bail!("RoPE positions must be u32 with length {expected}");
        }
    }

    let (q_s, q_l) = q.storage_and_layout();
    let (k_s, k_l) = k.storage_and_layout();
    let (cos_s, cos_l) = cos.storage_and_layout();
    let (sin_s, sin_l) = sin.storage_and_layout();
    let q_s = match &*q_s {
        Storage::Metal(storage) => storage,
        _ => inference_tensor::bail!("q must be a Metal tensor"),
    };
    let k_s = match &*k_s {
        Storage::Metal(storage) => storage,
        _ => inference_tensor::bail!("k must be a Metal tensor"),
    };
    let cos_s = match &*cos_s {
        Storage::Metal(storage) => storage,
        _ => inference_tensor::bail!("cos must be a Metal tensor"),
    };
    let sin_s = match &*sin_s {
        Storage::Metal(storage) => storage,
        _ => inference_tensor::bail!("sin must be a Metal tensor"),
    };
    let device = q_s.device();
    let q_out = device.new_buffer(q_l.shape().elem_count(), q_s.dtype(), "rotary-q")?;
    let k_out = device.new_buffer(k_l.shape().elem_count(), k_s.dtype(), "rotary-k")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("rotary-qk");

    if let Some(positions) = positions.as_ref() {
        let (positions_s, positions_l) = positions.storage_and_layout();
        let positions_s = match &*positions_s {
            Storage::Metal(storage) => storage,
            _ => inference_tensor::bail!("positions must be a Metal tensor"),
        };
        crate::metal_kernels::call_rotary_qk_positions(
            device.device(),
            &encoder,
            crate::metal_kernels::Kernels::global(),
            q_s.dtype(),
            q_s.buffer(),
            k_s.buffer(),
            cos_s.buffer(),
            sin_s.buffer(),
            positions_s.buffer(),
            q_l.start_offset() * q_s.dtype().size_in_bytes(),
            k_l.start_offset() * k_s.dtype().size_in_bytes(),
            cos_l.start_offset() * cos_s.dtype().size_in_bytes(),
            sin_l.start_offset() * sin_s.dtype().size_in_bytes(),
            positions_l.start_offset() * positions_s.dtype().size_in_bytes(),
            dims.batch,
            dims.heads,
            k_heads,
            dims.seq_len,
            dims.head_dim,
            dims.rot_dim,
            is_neox,
            &q_out,
            &k_out,
        )
    } else {
        crate::metal_kernels::call_rotary_qk(
            device.device(),
            &encoder,
            crate::metal_kernels::Kernels::global(),
            q_s.dtype(),
            q_s.buffer(),
            k_s.buffer(),
            cos_s.buffer(),
            sin_s.buffer(),
            q_l.start_offset() * q_s.dtype().size_in_bytes(),
            k_l.start_offset() * k_s.dtype().size_in_bytes(),
            cos_l.start_offset() * cos_s.dtype().size_in_bytes(),
            sin_l.start_offset() * sin_s.dtype().size_in_bytes(),
            dims.batch,
            dims.heads,
            k_heads,
            dims.seq_len,
            dims.head_dim,
            dims.rot_dim,
            dims.cache_rows,
            is_neox,
            &q_out,
            &k_out,
        )
    }
    .map_err(inference_tensor::Error::wrap)?;

    Ok((
        metal_tensor(
            Storage::Metal(MetalStorage::new(
                q_out,
                device.clone(),
                q_l.shape().elem_count(),
                q_s.dtype(),
            )),
            q_l.shape().clone(),
        ),
        metal_tensor(
            Storage::Metal(MetalStorage::new(
                k_out,
                device.clone(),
                k_l.shape().elem_count(),
                k_s.dtype(),
            )),
            k_l.shape().clone(),
        ),
    ))
}

pub fn apply_rotary_q(
    q: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
    is_neox: bool,
) -> Result<Tensor> {
    apply_rotary_q_inner(q, cos, sin, Some(positions), is_neox)
}

/// On CUDA this can rotate `q` in place (one head or one token), so callers must not reuse their input.
pub fn apply_rotary_q_preselected(
    q: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    is_neox: bool,
) -> Result<Tensor> {
    apply_rotary_q_inner(q, cos, sin, None, is_neox)
}

fn apply_rotary_q_inner(
    q: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if q.device().is_cuda() {
        return cuda_apply_rotary_q(q, cos, sin, positions, is_neox);
    }
    #[cfg(feature = "metal")]
    if q.device().is_metal() {
        return metal_apply_rotary_q(q, cos, sin, positions, is_neox);
    }
    cpu_apply_rotary_q(q, cos, sin, positions, is_neox)
}

pub fn apply_rotary_qk_preselected(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    is_neox: bool,
) -> Result<(Tensor, Tensor)> {
    apply_rotary_qk_inner(q, k, cos, sin, None, is_neox)
}

pub fn apply_rotary_qk(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
    is_neox: bool,
) -> Result<(Tensor, Tensor)> {
    apply_rotary_qk_inner(q, k, cos, sin, Some(positions), is_neox)
}

fn apply_rotary_qk_inner(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<(Tensor, Tensor)> {
    if q.dtype() != k.dtype() {
        inference_tensor::bail!("q/k dtype mismatch {:?} {:?}", q.dtype(), k.dtype());
    }
    #[cfg(feature = "cuda")]
    if q.device().is_cuda() {
        return cuda_apply_rotary_qk(q, k, cos, sin, positions, is_neox);
    }
    #[cfg(feature = "metal")]
    if q.device().is_metal() {
        return metal_apply_rotary_qk(q, k, cos, sin, positions, is_neox);
    }
    cpu_apply_rotary_qk(q, k, cos, sin, positions, is_neox)
}

#[cfg(feature = "cuda")]
mod cuda {
    use half::{bf16, f16};
    use inference_tensor::{
        CpuStorage, DType, InplaceOp3, Layout, MetalStorage, Result, Storage, Tensor,
        backend::{BackendDevice, BackendStorage},
        cuda_backend::{CudaDType, CudaStorage, CudaStorageSlice},
    };
    use std::ffi::{c_int, c_long};

    use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

    fn rotary_dtype(dtype: DType) -> Result<u32> {
        Ok(match dtype {
            DType::F16 => 0,
            DType::BF16 => 1,
            DType::F32 => 2,
            dtype => inference_tensor::bail!("dtype {dtype:?} is not supported"),
        })
    }

    struct RotaryLaunch<'a> {
        query: &'a mut CudaStorage,
        query_l: &'a Layout,
        cos_cache: &'a CudaStorage,
        cos_l: &'a Layout,
        sin_cache: &'a CudaStorage,
        sin_l: &'a Layout,
        positions: Option<(&'a CudaStorage, &'a Layout)>,
        is_neox: bool,
    }

    fn launch_rotary<T>(args: RotaryLaunch<'_>) -> Result<()>
    where
        T: CudaDType + inference_tensor::cuda_backend::cudarc::driver::DeviceRepr,
    {
        let RotaryLaunch {
            query,
            query_l,
            cos_cache,
            cos_l,
            sin_cache,
            sin_l,
            positions,
            is_neox,
        } = args;

        if cos_cache.dtype() != query.dtype() || sin_cache.dtype() != query.dtype() {
            inference_tensor::bail!("apply-rotary expects all tensors to have the same dtype");
        }

        let dev = query.device().clone();
        if !cos_cache.device().same_device(&dev) || !sin_cache.device().same_device(&dev) {
            inference_tensor::bail!("apply-rotary tensors must be on the same cuda device");
        }

        if query_l.stride().len() != 3 {
            inference_tensor::bail!("apply-rotary expects query rank 3 ({query_l:?})")
        }
        if cos_l.stride().len() != 2 || sin_l.stride().len() != 2 {
            inference_tensor::bail!("apply-rotary expects rank 2 caches")
        }

        let (num_tokens, num_heads, head_size) = query_l.shape().dims3()?;
        let rot_dim = cos_l.dims()[1];
        if sin_l.shape().dims2()? != (cos_l.dims()[0], rot_dim) {
            inference_tensor::bail!(
                "shape mismatch cos_cache {:?} and sin_cache {:?}",
                cos_l.shape(),
                sin_l.shape()
            )
        }
        if positions.is_none() && (num_tokens, rot_dim) != cos_l.shape().dims2()? {
            inference_tensor::bail!(
                "shape mismatch cos_cache {:?}, expected {:?}",
                cos_l.shape(),
                (num_tokens, rot_dim)
            )
        }
        if rot_dim == 0 || rot_dim * 2 > head_size {
            inference_tensor::bail!(
                "rotary dimension {rot_dim} is incompatible with head size {head_size}"
            )
        }

        let query_dtype = query.dtype();
        let stream = dev.cuda_stream();
        let query = query.as_cuda_slice_mut::<T>()?;
        let cos_cache = cos_cache.as_cuda_slice::<T>()?;
        let sin_cache = sin_cache.as_cuda_slice::<T>()?;
        let (query, _query_guard) = slice_ptr_mut_on_stream(query, query_l.start_offset(), &stream);
        let (cos_cache, _cos_guard) = slice_ptr_on_stream(cos_cache, cos_l.start_offset(), &stream);
        let (sin_cache, _sin_guard) = slice_ptr_on_stream(sin_cache, sin_l.start_offset(), &stream);

        let positions = if let Some((positions, positions_l)) = positions {
            if positions.dtype() != DType::U32 {
                inference_tensor::bail!("apply-rotary-positions expects positions to be u32");
            }
            if !positions.device().same_device(&dev) {
                inference_tensor::bail!("positions must be on the same cuda device as query");
            }
            if positions_l.stride().len() != 1 {
                inference_tensor::bail!("apply-rotary-positions expects rank 1 positions")
            }
            let positions_len = positions_l.shape().dims1()?;
            if positions_len != num_tokens {
                inference_tensor::bail!(
                    "positions length {positions_len} does not match token count {num_tokens}"
                );
            }
            let positions = match &positions.slice {
                CudaStorageSlice::U32(positions) => positions,
                _ => inference_tensor::bail!("positions dtype mismatch"),
            };
            let (positions, guard) =
                slice_ptr_on_stream(positions, positions_l.start_offset(), &stream);
            Some((positions, 1, guard))
        } else {
            None
        };

        let neox = if is_neox { 1 } else { 0 };
        let stream = stream.cu_stream() as c_long;
        let internal_type = rotary_dtype(query_dtype)?;
        match positions {
            None => unsafe {
                super::ffi::rotary_embedding(
                    query as *const core::ffi::c_void,
                    std::ptr::null(),
                    cos_cache as *const core::ffi::c_void,
                    sin_cache as *const core::ffi::c_void,
                    neox,
                    head_size as c_int,
                    num_tokens as c_long,
                    rot_dim as c_int,
                    num_heads as c_int,
                    0,
                    query_l.stride()[0] as c_long,
                    0,
                    internal_type,
                    stream,
                )
            },
            Some((positions, seq_len, _positions_guard)) => unsafe {
                super::ffi::rotary_embedding_positions(
                    query as *const core::ffi::c_void,
                    std::ptr::null(),
                    cos_cache as *const core::ffi::c_void,
                    sin_cache as *const core::ffi::c_void,
                    positions as *const core::ffi::c_void,
                    neox,
                    head_size as c_int,
                    num_tokens as c_long,
                    rot_dim as c_int,
                    seq_len as c_int,
                    num_heads as c_int,
                    0,
                    query_l.stride()[0] as c_long,
                    0,
                    internal_type,
                    stream,
                )
            },
        }
        Ok(())
    }

    struct RotaryInplace {
        is_neox: bool,
    }

    impl InplaceOp3 for RotaryInplace {
        fn name(&self) -> &'static str {
            "inference-rotary-inplace"
        }

        fn cpu_fwd(
            &self,
            _: &mut CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> Result<()> {
            inference_tensor::bail!("apply-rotary-inplace is only supported for cuda")
        }

        fn cuda_fwd(
            &self,
            query: &mut CudaStorage,
            query_l: &Layout,
            cos_cache: &CudaStorage,
            cos_l: &Layout,
            sin_cache: &CudaStorage,
            sin_l: &Layout,
        ) -> Result<()> {
            match query.dtype() {
                DType::F16 => launch_rotary::<f16>(RotaryLaunch {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: None,
                    is_neox: self.is_neox,
                }),
                DType::BF16 => launch_rotary::<bf16>(RotaryLaunch {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: None,
                    is_neox: self.is_neox,
                }),
                DType::F32 => launch_rotary::<f32>(RotaryLaunch {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: None,
                    is_neox: self.is_neox,
                }),
                dt => {
                    inference_tensor::bail!(
                        "apply_rotary is only supported for f32, f16 and bf16 ({dt:?})"
                    )
                }
            }
        }

        fn metal_fwd(
            &self,
            _: &mut MetalStorage,
            _: &Layout,
            _: &MetalStorage,
            _: &Layout,
            _: &MetalStorage,
            _: &Layout,
        ) -> Result<()> {
            inference_tensor::bail!("apply-rotary-inplace is only supported for cuda")
        }
    }

    struct RotaryPositionsInplace<'a> {
        is_neox: bool,
        positions: &'a Tensor,
    }

    impl InplaceOp3 for RotaryPositionsInplace<'_> {
        fn name(&self) -> &'static str {
            "inference-rotary-positions-inplace"
        }

        fn cpu_fwd(
            &self,
            _: &mut CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> Result<()> {
            inference_tensor::bail!("apply-rotary-positions-inplace is only supported for cuda")
        }

        fn cuda_fwd(
            &self,
            query: &mut CudaStorage,
            query_l: &Layout,
            cos_cache: &CudaStorage,
            cos_l: &Layout,
            sin_cache: &CudaStorage,
            sin_l: &Layout,
        ) -> Result<()> {
            let (positions_storage, positions_l) = self.positions.storage_and_layout();
            let positions = match &*positions_storage {
                Storage::Cuda(positions) => positions,
                _ => inference_tensor::bail!("positions must be a cuda tensor"),
            };
            match query.dtype() {
                DType::F16 => launch_rotary::<f16>(RotaryLaunch {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: Some((positions, positions_l)),
                    is_neox: self.is_neox,
                }),
                DType::BF16 => launch_rotary::<bf16>(RotaryLaunch {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: Some((positions, positions_l)),
                    is_neox: self.is_neox,
                }),
                DType::F32 => launch_rotary::<f32>(RotaryLaunch {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: Some((positions, positions_l)),
                    is_neox: self.is_neox,
                }),
                dt => {
                    inference_tensor::bail!(
                        "apply_rotary is only supported for f32, f16 and bf16 ({dt:?})"
                    )
                }
            }
        }

        fn metal_fwd(
            &self,
            _: &mut MetalStorage,
            _: &Layout,
            _: &MetalStorage,
            _: &Layout,
            _: &MetalStorage,
            _: &Layout,
        ) -> Result<()> {
            inference_tensor::bail!("apply-rotary-positions-inplace is only supported for cuda")
        }
    }

    fn apply_rotary_(
        query: &Tensor,
        key: Option<&Tensor>,
        cos_cache: &Tensor,
        sin_cache: &Tensor,
        is_neox: bool,
    ) -> Result<()> {
        let dtype = query.dtype();
        if key.is_some_and(|key| key.dtype() != dtype)
            || cos_cache.dtype() != dtype
            || sin_cache.dtype() != dtype
        {
            inference_tensor::bail!("apply-rotary expects all tensors to have the same dtype");
        }
        let op = RotaryInplace { is_neox };
        query.inplace_op3(cos_cache, sin_cache, &op)?;
        if let Some(key) = key {
            key.inplace_op3(cos_cache, sin_cache, &op)?;
        }
        Ok(())
    }

    fn apply_rotary_positions_(
        query: &Tensor,
        key: Option<&Tensor>,
        cos_cache: &Tensor,
        sin_cache: &Tensor,
        positions: &Tensor,
        is_neox: bool,
    ) -> Result<()> {
        let dtype = query.dtype();
        if key.is_some_and(|key| key.dtype() != dtype)
            || cos_cache.dtype() != dtype
            || sin_cache.dtype() != dtype
            || positions.dtype() != DType::U32
        {
            inference_tensor::bail!(
                "apply-rotary-positions expects q/k/caches to share dtype and positions to be u32"
            );
        }

        let cos_cache = cos_cache.contiguous()?;
        let sin_cache = sin_cache.contiguous()?;
        let positions = positions.contiguous()?;
        let op = RotaryPositionsInplace {
            is_neox,
            positions: &positions,
        };
        query.inplace_op3(&cos_cache, &sin_cache, &op)?;
        if let Some(key) = key {
            key.inplace_op3(&cos_cache, &sin_cache, &op)?;
        }
        Ok(())
    }

    /// Apply Rotary position encoding inplace
    ///
    /// # Arguments
    ///
    /// * `query` - Query tensor of shape `(num_tokens, num_heads, head_size)`.
    /// * `key` - Key tensor of shape `(num_tokens, num_kv_heads, head_size)`.
    /// * `cos_cache` - Aligned cache of shape `(num_tokens, rot_dim)`
    /// * `sin_cache` - Aligned cache of shape `(num_tokens, rot_dim)`
    /// * `is_neox` - Use neox encoding instead of gpt-j style rotary
    pub fn apply_rotary_inplace(
        query: &Tensor,
        key: &Tensor,
        cos_cache: &Tensor,
        sin_cache: &Tensor,
        is_neox: bool,
    ) -> Result<()> {
        match key.dtype() {
            DType::F16 | DType::BF16 | DType::F32 => {
                apply_rotary_(query, Some(key), cos_cache, sin_cache, is_neox)
            }
            dt => {
                inference_tensor::bail!(
                    "apply_rotary is only supported for f32, f16 and bf16 ({dt:?})"
                )
            }
        }
    }

    pub fn apply_rotary_inplace_q(
        query: &Tensor,
        cos_cache: &Tensor,
        sin_cache: &Tensor,
        is_neox: bool,
    ) -> Result<()> {
        match query.dtype() {
            DType::F16 | DType::BF16 | DType::F32 => {
                apply_rotary_(query, None, cos_cache, sin_cache, is_neox)
            }
            dt => {
                inference_tensor::bail!(
                    "apply_rotary is only supported for f32, f16 and bf16 ({dt:?})"
                )
            }
        }
    }

    pub fn apply_rotary_inplace_positions(
        query: &Tensor,
        key: &Tensor,
        cos_cache: &Tensor,
        sin_cache: &Tensor,
        positions: &Tensor,
        is_neox: bool,
    ) -> Result<()> {
        match key.dtype() {
            DType::F16 | DType::BF16 | DType::F32 => {
                apply_rotary_positions_(query, Some(key), cos_cache, sin_cache, positions, is_neox)
            }
            dt => {
                inference_tensor::bail!(
                    "apply_rotary is only supported for f32, f16 and bf16 ({dt:?})"
                )
            }
        }
    }

    pub fn apply_rotary_inplace_q_positions(
        query: &Tensor,
        cos_cache: &Tensor,
        sin_cache: &Tensor,
        positions: &Tensor,
        is_neox: bool,
    ) -> Result<()> {
        match query.dtype() {
            DType::F16 | DType::BF16 | DType::F32 => {
                apply_rotary_positions_(query, None, cos_cache, sin_cache, positions, is_neox)
            }
            dt => {
                inference_tensor::bail!(
                    "apply_rotary is only supported for f32, f16 and bf16 ({dt:?})"
                )
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::*;

/// Apply Rotary position encoding inplace
///
/// # Arguments
///
/// * `query` - Query tensor of shape `(num_tokens, num_heads, head_size)`.
/// * `key` - Key tensor of shape `(num_tokens, num_kv_heads, head_size)`.
/// * `cos_cache` - Aligned cache of shape `(num_tokens, rot_dim)`
/// * `sin_cache` - Aligned cache of shape `(num_tokens, rot_dim)`
/// * `is_neox` - Use neox encoding instead of gpt-j style rotary
#[cfg(not(feature = "cuda"))]
pub fn apply_rotary_inplace(
    _query: &inference_tensor::Tensor,
    _key: &inference_tensor::Tensor,
    _cos_cache: &inference_tensor::Tensor,
    _sin_cache: &inference_tensor::Tensor,
    _is_neox: bool,
) -> inference_tensor::Result<()> {
    inference_tensor::bail!("apply_rotary is only supported for cuda");
}

#[cfg(not(feature = "cuda"))]
pub fn apply_rotary_inplace_q(
    _query: &inference_tensor::Tensor,
    _cos_cache: &inference_tensor::Tensor,
    _sin_cache: &inference_tensor::Tensor,
    _is_neox: bool,
) -> inference_tensor::Result<()> {
    inference_tensor::bail!("apply_rotary is only supported for cuda");
}

#[cfg(not(feature = "cuda"))]
pub fn apply_rotary_inplace_positions(
    _query: &inference_tensor::Tensor,
    _key: &inference_tensor::Tensor,
    _cos_cache: &inference_tensor::Tensor,
    _sin_cache: &inference_tensor::Tensor,
    _positions: &inference_tensor::Tensor,
    _is_neox: bool,
) -> inference_tensor::Result<()> {
    inference_tensor::bail!("apply_rotary is only supported for cuda");
}

#[cfg(not(feature = "cuda"))]
pub fn apply_rotary_inplace_q_positions(
    _query: &inference_tensor::Tensor,
    _cos_cache: &inference_tensor::Tensor,
    _sin_cache: &inference_tensor::Tensor,
    _positions: &inference_tensor::Tensor,
    _is_neox: bool,
) -> inference_tensor::Result<()> {
    inference_tensor::bail!("apply_rotary is only supported for cuda");
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use inference_tensor::{Device, Result, Tensor};

    const BATCH: usize = 2;
    const Q_HEADS: usize = 4;
    const K_HEADS: usize = 2;
    const SEQ_LEN: usize = 3;
    const HEAD_DIM: usize = 16;

    fn ramp(shape: &[usize], scale: f64, offset: f64) -> Result<Tensor> {
        let n: usize = shape.iter().product();
        Tensor::arange(0f32, n as f32, &Device::Cpu)?
            .affine(scale, offset)?
            .sin()?
            .reshape(shape)
    }

    fn assert_close(cuda: &Tensor, cpu: &Tensor) -> Result<()> {
        let cuda = cuda
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let cpu = cpu.flatten_all()?.to_vec1::<f32>()?;
        for (a, b) in cuda.iter().zip(&cpu) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        Ok(())
    }

    #[test]
    fn preselected_rope_on_cuda_matches_the_cpu_for_every_cache_shape() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let q = ramp(&[BATCH, Q_HEADS, SEQ_LEN, HEAD_DIM], 0.07, 0.1)?;
        let k = ramp(&[BATCH, K_HEADS, SEQ_LEN, HEAD_DIM], 0.05, 0.4)?;
        let caches = [
            vec![BATCH, SEQ_LEN, HEAD_DIM / 2],
            vec![BATCH * SEQ_LEN, HEAD_DIM / 2],
            vec![SEQ_LEN, HEAD_DIM / 2],
        ];
        for shape in caches {
            let angles = ramp(&shape, 0.3, 0.0)?.affine(3.0, 0.0)?;
            let (cos, sin) = (angles.cos()?, angles.sin()?);
            let (cuda_cos, cuda_sin) = (cos.to_device(&device)?, sin.to_device(&device)?);
            for neox in [true, false] {
                let (cpu_q, cpu_k) = super::apply_rotary_qk_preselected(&q, &k, &cos, &sin, neox)?;
                let (cuda_q, cuda_k) = super::apply_rotary_qk_preselected(
                    &q.to_device(&device)?,
                    &k.to_device(&device)?,
                    &cuda_cos,
                    &cuda_sin,
                    neox,
                )?;
                assert_close(&cuda_q, &cpu_q)?;
                assert_close(&cuda_k, &cpu_k)?;
                let cuda_q_only = super::apply_rotary_q_preselected(
                    &q.to_device(&device)?,
                    &cuda_cos,
                    &cuda_sin,
                    neox,
                )?;
                assert_close(&cuda_q_only, &cpu_q)?;
            }
        }
        Ok(())
    }
}
