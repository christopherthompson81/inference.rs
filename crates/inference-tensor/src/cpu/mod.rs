//! Traits and methods for CPU-backed Tensors
#![allow(clippy::needless_range_loop)]
pub mod erf;
pub(crate) mod features;
pub mod kernels;
mod row_weights;

pub use row_weights::Bf16RowWeights;

/// Transformer-XL relative-position attention fused on the CPU's AVX-512 BF16, inside [`with_bf16_matmuls`]:
/// score(i, j) = `q_content`_i . `k`_j + `q_position`_i . `positions`[pos_base - i + j], both query sets scaled
/// beforehand, a softmax over the keys, times `v`. The queries are `(1, heads, t, d)`, `k` and `v` `(1, heads, keys,
/// d)`, `positions` `(1, heads, p_rows, d)`; the output is laid out as the queries. `None` outside the scope, off a CPU
/// with bf16 pairs, or for a shape the kernel does not take.
pub fn rel_pos_attention(
    q_content: &crate::Tensor,
    q_position: &crate::Tensor,
    k: &crate::Tensor,
    v: &crate::Tensor,
    positions: &crate::Tensor,
    pos_base: usize,
) -> crate::Result<Option<crate::Tensor>> {
    let (batch, heads, t, d) = q_content.dims4()?;
    let (_, _, keys, _) = k.dims4()?;
    let (_, _, p_rows, _) = positions.dims4()?;
    let operands = [q_content, q_position, k, v, positions];
    let shapes = q_position.dims() == q_content.dims()
        && k.dims() == [batch, heads, keys, d]
        && v.dims() == k.dims()
        && positions.dims() == [batch, heads, p_rows, d];
    if batch != 1
        || !shapes
        || !operands
            .iter()
            .all(|x| x.device().is_cpu() && x.dtype() == crate::DType::F32)
        || !crate::cpu_backend::gemm_bf16::enabled()
        || !crate::cpu_backend::relpos_attn::supports(d)
        || pos_base + 1 < t
        || pos_base + keys > p_rows
    {
        return Ok(None);
    }
    // the operands' storage, borrowed (a layer's had taken ~50 ms to copy out)
    let contiguous = operands
        .iter()
        .map(|x| x.contiguous())
        .collect::<crate::Result<Vec<_>>>()?;
    let held: Vec<_> = contiguous.iter().map(|x| x.storage_and_layout()).collect();
    let slices = held
        .iter()
        .map(|(storage, layout)| match &**storage {
            crate::Storage::Cpu(crate::CpuStorage::F32(values)) => {
                Ok(&values[layout.start_offset()..][..layout.shape().elem_count()])
            }
            _ => crate::bail!("relative-position attention takes CPU f32 operands"),
        })
        .collect::<crate::Result<Vec<&[f32]>>>()?;
    let out =
        crate::cpu_backend::relpos_attn::attention(&crate::cpu_backend::relpos_attn::RelPosAttn {
            qc: slices[0],
            qp: slices[1],
            k: slices[2],
            v: slices[3],
            p: slices[4],
            heads,
            t,
            keys,
            d,
            p_rows,
            pos_base,
        });
    Ok(Some(crate::Tensor::from_vec(
        out,
        (batch, heads, t, d),
        &crate::Device::Cpu,
    )?))
}

pub use crate::cpu_backend::gemm_bf16::{
    BF16_ENV, NO_BF16_ENV, enabled as bf16_enabled, forced as bf16_forced, with_bf16_matmuls,
};

#[allow(unused)]
trait Cpu {
    type Unit;
    type Array;
    const STEP: usize;
    const EPR: usize;
    const ARR: usize = Self::STEP / Self::EPR;

    unsafe fn zero() -> Self::Unit;
    unsafe fn zero_array() -> Self::Array;
    unsafe fn load(mem_addr: *const f32) -> Self::Unit;
    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit;
    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit;
    unsafe fn vec_reduce(x: Self::Array, y: *mut f32);
    unsafe fn from_f32(v: f32) -> Self::Unit;
    unsafe fn vec_store(mem_addr: *mut f32, a: Self::Unit);
}

#[allow(unused)]
trait CpuF16 {
    type Unit;
    type Array;
    const STEP: usize;
    const EPR: usize;
    const ARR: usize = Self::STEP / Self::EPR;
    // How many outer loop steps to accumulate in Self::Unit before flushing to f32.
    // Set to a finite value to bound rounding error (16 for example).
    const FLUSH_INTERVAL: usize = usize::MAX;

    unsafe fn zero() -> Self::Unit;
    unsafe fn zero_array() -> Self::Array;
    unsafe fn load(mem_addr: *const f16) -> Self::Unit;
    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit;
    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit;
    unsafe fn vec_reduce(x: Self::Array, y: *mut f32);
    unsafe fn from_f32(v: f32) -> Self::Unit;
    unsafe fn vec_store(mem_addr: *mut f16, a: Self::Unit);
}

#[allow(unused)]
trait CpuBF16 {
    type Unit;
    type Array;
    const STEP: usize;
    const EPR: usize;
    const ARR: usize = Self::STEP / Self::EPR;

    unsafe fn zero() -> Self::Unit;
    unsafe fn zero_array() -> Self::Array;
    unsafe fn load(mem_addr: *const bf16) -> Self::Unit;
    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit;
    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit;
    unsafe fn vec_reduce(x: Self::Array, y: *mut f32);
    unsafe fn from_f32(v: f32) -> Self::Unit;
    unsafe fn vec_store(mem_addr: *mut bf16, a: Self::Unit);
}

use half::{bf16, f16};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx;

#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "arm", target_feature = "neon")
))]
pub mod neon;

#[inline(always)]
unsafe fn scalar_vec_dot_f32(a_row: *const f32, b_row: *const f32, c: *mut f32, k: usize) {
    unsafe {
        for i in 0..k {
            *c += *a_row.add(i) * (*b_row.add(i));
        }
    }
}

#[inline(always)]
unsafe fn scalar_vec_sum(row: *const f32, b: *mut f32, k: usize) {
    unsafe {
        *b = 0f32;
        for i in 0..k {
            *b += *row.add(i)
        }
    }
}

#[inline(always)]
unsafe fn scalar_vec_dot_f16(a_row: *const f16, b_row: *const f16, c: *mut f32, k: usize) {
    unsafe {
        let mut sum = 0.0;
        for i in 0..k {
            sum += (*a_row.add(i)).to_f32() * (*b_row.add(i)).to_f32();
        }
        *c = sum;
    }
}

#[inline(always)]
unsafe fn scalar_vec_dot_bf16(a_row: *const bf16, b_row: *const bf16, c: *mut f32, k: usize) {
    unsafe {
        let mut sum = 0.0;
        for i in 0..k {
            sum += (*a_row.add(i)).to_f32() * (*b_row.add(i)).to_f32();
        }
        *c = sum;
    }
}

#[inline(always)]
unsafe fn scalar_vec_add_f16(a_row: *const f16, b_row: *const f16, c: *mut f16, k: usize) {
    unsafe {
        for i in 0..k {
            *c.add(i) = *a_row.add(i) + *b_row.add(i);
        }
    }
}

#[inline(always)]
unsafe fn scalar_vec_add_bf16(a_row: *const bf16, b_row: *const bf16, c: *mut bf16, k: usize) {
    unsafe {
        for i in 0..k {
            *c.add(i) = *a_row.add(i) + *b_row.add(i);
        }
    }
}

#[inline(always)]
unsafe fn scalar_vec_scalar_add_f16(scalar: f16, xs: *const f16, ys: *mut f16, k: usize) {
    unsafe {
        for i in 0..k {
            *ys.add(i) = *xs.add(i) + scalar;
        }
    }
}

#[inline(always)]
unsafe fn scalar_vec_scalar_add_bf16(scalar: bf16, xs: *const bf16, ys: *mut bf16, k: usize) {
    unsafe {
        for i in 0..k {
            *ys.add(i) = *xs.add(i) + scalar;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn use_avx2() -> bool {
    features::get().avx2
}

#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "arm", target_feature = "neon")
))]
#[inline(always)]
fn use_neon() -> bool {
    features::get().neon
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn use_avx2_f16() -> bool {
    let features = features::get();
    features.avx2 && (!cfg!(target_feature = "f16c") || features.f16c)
}

#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "arm", target_feature = "neon")
))]
#[inline(always)]
fn use_neon_f16() -> bool {
    let features = features::get();
    features.neon && (!cfg!(target_feature = "fp16") || features.fp16)
}

#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "arm", target_feature = "neon")
))]
#[inline(always)]
fn use_neon_bf16() -> bool {
    let features = features::get();
    features.neon && (!cfg!(target_feature = "bf16") || features.bf16)
}

#[inline(always)]
pub(crate) unsafe fn vec_dot_f32(a_row: *const f32, b_row: *const f32, c: *mut f32, k: usize) {
    unsafe {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if use_avx2() {
            return avx::vec_dot_f32(a_row, b_row, c, k);
        }
        #[cfg(any(
            target_arch = "aarch64",
            all(target_arch = "arm", target_feature = "neon")
        ))]
        if use_neon() {
            return neon::vec_dot_f32(a_row, b_row, c, k);
        }
        scalar_vec_dot_f32(a_row, b_row, c, k)
    }
}

#[inline(always)]
pub(crate) unsafe fn vec_sum(row: *const f32, b: *mut f32, k: usize) {
    unsafe {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if use_avx2() {
            return avx::vec_sum(row, b, k);
        }
        #[cfg(any(
            target_arch = "aarch64",
            all(target_arch = "arm", target_feature = "neon")
        ))]
        if use_neon() {
            return neon::vec_sum(row, b, k);
        }
        scalar_vec_sum(row, b, k)
    }
}

#[inline(always)]
pub(crate) unsafe fn vec_dot_f16(a_row: *const f16, b_row: *const f16, c: *mut f32, k: usize) {
    unsafe {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if use_avx2_f16() {
            return avx::vec_dot_f16(a_row, b_row, c, k);
        }
        #[cfg(any(
            target_arch = "aarch64",
            all(target_arch = "arm", target_feature = "neon")
        ))]
        if use_neon_f16() {
            return neon::vec_dot_f16(a_row, b_row, c, k);
        }
        scalar_vec_dot_f16(a_row, b_row, c, k)
    }
}

#[inline(always)]
pub(crate) unsafe fn vec_dot_bf16(a_row: *const bf16, b_row: *const bf16, c: *mut f32, k: usize) {
    unsafe {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if use_avx2() {
            return avx::vec_dot_bf16(a_row, b_row, c, k);
        }
        #[cfg(any(
            target_arch = "aarch64",
            all(target_arch = "arm", target_feature = "neon")
        ))]
        if use_neon_bf16() {
            return neon::vec_dot_bf16(a_row, b_row, c, k);
        }
        scalar_vec_dot_bf16(a_row, b_row, c, k)
    }
}

#[inline(always)]
pub(crate) unsafe fn vec_add_f16(a_row: *const f16, b_row: *const f16, c: *mut f16, k: usize) {
    unsafe {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if use_avx2_f16() {
            return avx::vec_add_f16(a_row, b_row, c, k);
        }
        #[cfg(any(
            target_arch = "aarch64",
            all(target_arch = "arm", target_feature = "neon")
        ))]
        if use_neon_f16() {
            return neon::vec_add_f16(a_row, b_row, c, k);
        }
        scalar_vec_add_f16(a_row, b_row, c, k)
    }
}

#[inline(always)]
pub(crate) unsafe fn vec_add_bf16(a_row: *const bf16, b_row: *const bf16, c: *mut bf16, k: usize) {
    unsafe {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if use_avx2() {
            return avx::vec_add_bf16(a_row, b_row, c, k);
        }
        #[cfg(any(
            target_arch = "aarch64",
            all(target_arch = "arm", target_feature = "neon")
        ))]
        if use_neon_bf16() {
            return neon::vec_add_bf16(a_row, b_row, c, k);
        }
        scalar_vec_add_bf16(a_row, b_row, c, k)
    }
}

#[inline(always)]
pub(crate) unsafe fn vec_scalar_add_f16(scalar: f16, xs: *const f16, ys: *mut f16, k: usize) {
    unsafe {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if use_avx2_f16() {
            return avx::vec_scalar_add_f16(scalar, xs, ys, k);
        }
        #[cfg(any(
            target_arch = "aarch64",
            all(target_arch = "arm", target_feature = "neon")
        ))]
        if use_neon_f16() {
            return neon::vec_scalar_add_f16(scalar, xs, ys, k);
        }
        scalar_vec_scalar_add_f16(scalar, xs, ys, k)
    }
}

#[inline(always)]
pub(crate) unsafe fn vec_mul_bf16(a_row: *const bf16, b_row: *const bf16, c: *mut bf16, k: usize) {
    unsafe {
        #[cfg(any(
            target_arch = "aarch64",
            all(target_arch = "arm", target_feature = "neon")
        ))]
        if use_neon_bf16() {
            return neon::vec_mul_bf16(a_row, b_row, c, k);
        }
        for j in 0..k {
            *c.add(j) = *a_row.add(j) * *b_row.add(j);
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn vec_scalar_add_bf16(scalar: bf16, xs: *const bf16, ys: *mut bf16, k: usize) {
    unsafe {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if use_avx2() {
            return avx::vec_scalar_add_bf16(scalar, xs, ys, k);
        }
        #[cfg(any(
            target_arch = "aarch64",
            all(target_arch = "arm", target_feature = "neon")
        ))]
        if use_neon_bf16() {
            return neon::vec_scalar_add_bf16(scalar, xs, ys, k);
        }
        scalar_vec_scalar_add_bf16(scalar, xs, ys, k)
    }
}
