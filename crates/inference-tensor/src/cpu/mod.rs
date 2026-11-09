//! Traits and methods for CPU-backed Tensors
#![allow(clippy::needless_range_loop)]
pub mod erf;
pub(crate) mod features;
pub mod kernels;
mod row_weights;

pub use row_weights::HalfRowWeights;

/// A relative-position attention's operands as a layer's projections give them: `q` `(1, t, heads * d)`, `k` and `v`
/// `(1, keys, heads * d)`, `positions` `(1, p_rows, heads * d)`, and the content and position biases `bias_u`,
/// `bias_v` (`heads * d` values, any shape); pair `(i, j)` reads position row `pos_base - i + j`.
pub struct RelPosOperands<'a> {
    pub q: &'a crate::Tensor,
    pub bias_u: &'a crate::Tensor,
    pub bias_v: &'a crate::Tensor,
    pub scale: f32,
    pub k: &'a crate::Tensor,
    pub v: &'a crate::Tensor,
    pub positions: &'a crate::Tensor,
    pub heads: usize,
    pub pos_base: usize,
}

/// Transformer-XL relative-position attention fused on the CPU: on AVX-512 BF16 inside [`with_bf16_matmuls`], else in
/// f32 on AVX2 and FMA. score(i, j) = qc_i . k_j + qp_i . positions[pos_base - i + j] per head, with qc = (q + bias_u)
/// scale and qp = (q + bias_v) scale, a softmax over the keys, times `v`. Each head is read where the projections put
/// it, so nothing is split per head first; the output is `(1, t, heads * d)`, as `q`. `None` on a CPU with neither, or
/// for a shape the kernels do not take.
pub fn rel_pos_attention(op: &RelPosOperands) -> crate::Result<Option<crate::Tensor>> {
    let RelPosOperands {
        q,
        k,
        v,
        positions,
        heads,
        pos_base,
        ..
    } = *op;
    let (batch, t, width) = q.dims3()?;
    let (_, keys, _) = k.dims3()?;
    let (_, p_rows, _) = positions.dims3()?;
    let d = width / heads.max(1);
    let operands = [q, k, v, positions];
    let shapes = heads * d == width
        && k.dims() == [batch, keys, width]
        && v.dims() == k.dims()
        && positions.dims() == [batch, p_rows, width]
        && op.bias_u.elem_count() == width
        && op.bias_v.elem_count() == width;
    if batch != 1
        || !shapes
        || !operands
            .iter()
            .all(|x| x.device().is_cpu() && x.dtype() == crate::DType::F32)
        || pos_base + 1 < t
        || pos_base + keys > p_rows
    {
        return Ok(None);
    }
    let bf16 =
        crate::cpu_backend::gemm_bf16::enabled() && crate::cpu_backend::relpos_attn::supports(d);
    if !bf16 && !crate::cpu_backend::relpos_attn_f32::supports(d) {
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
    let bias = |b: &crate::Tensor| {
        b.flatten_all()?
            .to_dtype(crate::DType::F32)?
            .to_vec1::<f32>()
    };
    let (bias_u, bias_v) = (bias(op.bias_u)?, bias(op.bias_v)?);
    let op = crate::cpu_backend::relpos_attn::RelPosAttn {
        q: slices[0],
        bias_u: &bias_u,
        bias_v: &bias_v,
        scale: op.scale,
        k: slices[1],
        v: slices[2],
        p: slices[3],
        heads,
        t,
        keys,
        d,
        p_rows,
        pos_base,
    };
    let out = if bf16 {
        crate::cpu_backend::relpos_attn::attention(&op)
    } else {
        crate::cpu_backend::relpos_attn_f32::attention(&op)
    };
    Ok(Some(crate::Tensor::from_vec(
        out,
        (batch, t, width),
        &crate::Device::Cpu,
    )?))
}

/// The 16-bit type an `(n, _)` f32 weight of a model measured under bf16 is held as on this CPU, if any: bf16 where it
/// multiplies bf16 pairs (its products read it as it is), f16 where AVX2 widens whole 64-column panels as it packs
/// them (three more mantissa bits than bf16 at the same size: bf16 weights changed a word on two of four 10-minute
/// clips on an i7-10700K); [`NO_BF16_ENV`] unset either way.
pub fn half_weight_dtype(n: usize) -> Option<crate::DType> {
    if crate::cpu_backend::gemm_bf16::allowed() {
        Some(crate::DType::BF16)
    } else if crate::cpu_backend::gemm_bf16::weights_allowed()
        && crate::cpu_backend::gemm_halfw::supports(n)
    {
        Some(crate::DType::F16)
    } else {
        None
    }
}

/// `x` (f32, `(.., k)`) times a 16-bit `(n, k)` weight transposed (`Linear::to_half`'s on a CPU), plus `bias`, `(.., n)`
/// f32, the bias added in place: a bf16 weight's large products inside [`with_bf16_matmuls`] as the scope takes them
/// on bf16 pairs; others, on AVX2, f32 products over the weight widened as it packs. `None` where neither applies (the
/// caller widens the weight).
pub fn linear_bf16_weight(
    x: &crate::Tensor,
    weight: &crate::Tensor,
    bias: Option<&crate::Tensor>,
) -> crate::Result<Option<crate::Tensor>> {
    use crate::cpu_backend::{BF16_MIN_COLS, REST_POOL_MIN_WORK, gemm_bf16};
    let (n, k) = weight.dims2()?;
    let rows = x.elem_count() / k.max(1);
    let half = match weight.dtype() {
        crate::DType::BF16 => crate::cpu_backend::gemm_halfw::Half::Bf16,
        crate::DType::F16 => crate::cpu_backend::gemm_halfw::Half::F16,
        _ => return Ok(None),
    };
    if x.dtype() != crate::DType::F32
        || !x.device().is_cpu()
        || !weight.device().is_cpu()
        || x.dims().last() != Some(&k)
    {
        return Ok(None);
    }
    // the bf16 pairs' kernel for a large product inside the scope; else the weight widened as it packs, any size
    let pairs = half == crate::cpu_backend::gemm_halfw::Half::Bf16
        && rows * n * k >= REST_POOL_MIN_WORK
        && n >= BF16_MIN_COLS
        && gemm_bf16::enabled();
    if !pairs && !crate::cpu_backend::gemm_halfw::supports(n) {
        return Ok(None);
    }
    let (x, weight) = (x.contiguous()?, weight.contiguous()?);
    let (xs, xl) = x.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let crate::Storage::Cpu(crate::CpuStorage::F32(xv)) = &*xs else {
        crate::bail!("a half-weight linear takes CPU f32 rows");
    };
    let xv = &xv[xl.start_offset()..][..rows * k];
    let at = wl.start_offset();
    // SAFETY: bf16 and f16 are a u16's bits
    let wv = match &*ws {
        crate::Storage::Cpu(crate::CpuStorage::BF16(w)) => unsafe {
            std::slice::from_raw_parts(w[at..].as_ptr().cast::<u16>(), n * k)
        },
        crate::Storage::Cpu(crate::CpuStorage::F16(w)) => unsafe {
            std::slice::from_raw_parts(w[at..].as_ptr().cast::<u16>(), n * k)
        },
        _ => crate::bail!("a half-weight linear takes a CPU bf16 or f16 weight"),
    };
    let mut out = vec![0f32; rows * n];
    if !pairs {
        // the elementwise pool's idle workers park for a long product rather than spin beside its threads, as matmul's
        if rows * n * k >= REST_POOL_MIN_WORK {
            crate::utils::barrier_pool().rest();
        }
        crate::cpu_backend::gemm_halfw::gemm(xv, wv, half, rows, n, k, &mut out);
    } else {
        gemm_bf16::gemm(
            1,
            rows,
            n,
            k,
            &gemm_bf16::Strided {
                data: xv,
                rs: k,
                cs: 1,
            },
            0,
            &gemm_bf16::BOperand::Bf16 {
                data: wv,
                rs: 1,
                cs: k,
            },
            0,
            &mut out,
        );
    }
    if let Some(bias) = bias {
        let bias = bias
            .flatten_all()?
            .to_dtype(crate::DType::F32)?
            .to_vec1::<f32>()?;
        if bias.len() != n {
            crate::bail!("a bias of {} for {n} outputs", bias.len());
        }
        rows_in_place(&mut out, n, |row| {
            row.iter_mut().zip(&bias).for_each(|(y, b)| *y += b)
        });
    }
    let mut dims = x.dims().to_vec();
    *dims.last_mut().expect("rank >= 1") = n;
    Ok(Some(crate::Tensor::from_vec(
        out,
        dims,
        &crate::Device::Cpu,
    )?))
}

// `f` over each `width`-long row of `values` (the last may be shorter), the rows split over the elementwise pool
fn rows_in_place(values: &mut [f32], width: usize, f: impl Fn(&mut [f32]) + Sync) {
    let (at, total) = (values.as_mut_ptr() as usize, values.len());
    crate::utils::barrier_pool().execute_chunked(total.div_ceil(width), |rows| {
        for row in rows {
            let len = width.min(total - row * width);
            // SAFETY: the chunks are disjoint rows of `values`, which outlives the call
            f(unsafe { std::slice::from_raw_parts_mut((at as *mut f32).add(row * width), len) });
        }
    });
}

// SiLU over a CPU f32 storage's values, in place
struct SiluInPlace;

impl crate::InplaceOp1 for SiluInPlace {
    fn name(&self) -> &'static str {
        "silu-in-place"
    }

    fn cpu_fwd(
        &self,
        storage: &mut crate::CpuStorage,
        layout: &crate::Layout,
    ) -> crate::Result<()> {
        use crate::op::UnaryOpT;
        let crate::CpuStorage::F32(values) = storage else {
            crate::bail!("an in-place SiLU takes f32");
        };
        let values = &mut values[layout.start_offset()..][..layout.shape().elem_count()];
        rows_in_place(values, SILU_ROW, |row| {
            row.iter_mut().for_each(|v| *v = crate::op::Silu::f32(*v))
        });
        Ok(())
    }
}

// values a task of the in-place SiLU takes
const SILU_ROW: usize = 1 << 14;

/// SiLU over `x`'s values in place, on the CPU, where `x` is contiguous f32: for an activation nothing else holds (a
/// projection's fresh output), which then needs no second buffer of its size. Whether it ran.
pub fn silu_in_place(x: &crate::Tensor) -> crate::Result<bool> {
    if !x.device().is_cpu()
        || x.dtype() != crate::DType::F32
        || !x.is_contiguous()
        || x.elem_count() == 0
    {
        return Ok(false);
    }
    x.inplace_op1(&SiluInPlace)?;
    Ok(true)
}

pub use crate::cpu_backend::gemm_bf16::{
    BF16_ENV, NO_BF16_ENV, allowed as bf16_allowed, enabled as bf16_enabled, forced as bf16_forced,
    in_scope as bf16_scope, with_bf16_matmuls,
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

#[cfg(test)]
mod tests {
    use crate::{Device, Tensor};

    // the in-place SiLU equals the op's over a length past one task with a short tail, and leaves a strided view alone
    #[test]
    fn silu_in_place_matches_the_op() -> crate::Result<()> {
        let x = Tensor::randn(0f32, 3., (3, super::SILU_ROW + 17), &Device::Cpu)?;
        let want = crate::nn::ops::silu(&x)?.to_vec2::<f32>()?;
        assert!(super::silu_in_place(&x)?);
        assert_eq!(x.to_vec2::<f32>()?, want);
        assert!(
            !super::silu_in_place(&x.t()?)?,
            "a strided tensor is left to the op"
        );
        Ok(())
    }
}
