use crate::{DType, Device, Result, Tensor};

// a lane per accumulator, as `dot_f32`'s, so the loop vectorizes into 16-wide FMAs
const LANES: usize = 16;

/// An `(n, k)` f32 weight held in 16 bits for a CPU's few-row products, which stream the whole weight a row: half the
/// bytes, and a weight near the last-level cache's size fits it. f16 where the CPU widens it in hardware (F16C), bf16
/// elsewhere: f16's three more mantissa bits kept Parakeet's decoder at its f32 words on four 10-minute clips, where
/// bf16 changed a word on two (an i7-10700K). The products and their sums stay f32.
pub struct HalfRowWeights {
    rows: Vec<u16>,
    n: usize,
    k: usize,
    f16: bool,
}

impl HalfRowWeights {
    pub fn new(weight: &Tensor) -> Result<Self> {
        Self::held_as(weight, f16_dot_available())
    }

    // `weight` as f16 bits, or bf16's
    fn held_as(weight: &Tensor, f16: bool) -> Result<Self> {
        let (n, k) = weight.dims2()?;
        let flat = weight.flatten_all()?;
        let rows = if f16 {
            let values = flat.to_dtype(DType::F16)?.to_vec1::<half::f16>()?;
            values.into_iter().map(half::f16::to_bits).collect()
        } else {
            let values = flat.to_dtype(DType::BF16)?.to_vec1::<half::bf16>()?;
            values.into_iter().map(half::bf16::to_bits).collect()
        };
        Ok(Self { rows, n, k, f16 })
    }

    /// `x` `(m, k)` times the weight's transpose, `(m, n)`.
    pub fn matmul_t(&self, x: &Tensor) -> Result<Tensor> {
        let (m, k) = x.dims2()?;
        if k != self.k {
            crate::bail!("a ({m}, {k}) row times a ({}, {}) weight", self.n, self.k);
        }
        let x = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let n = self.n;
        let mut out = vec![0f32; m * n];
        let out_at = out.as_mut_ptr() as usize;
        crate::utils::barrier_pool().execute_chunked(n, |cols| {
            for row in 0..m {
                let x = &x[row * k..][..k];
                for col in cols.clone() {
                    let w = &self.rows[col * k..][..k];
                    let dot = if self.f16 {
                        dot_f16(x, w)
                    } else {
                        dot_bf16(x, w)
                    };
                    // SAFETY: the chunks are disjoint columns of `out`, which outlives the call
                    unsafe { *(out_at as *mut f32).add(row * n + col) = dot };
                }
            }
        });
        Tensor::from_vec(out, (m, n), &Device::Cpu)
    }
}

fn f16_dot_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("f16c")
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

fn dot_bf16(x: &[f32], w: &[u16]) -> f32 {
    let widen = |bits: u16| f32::from_bits(u32::from(bits) << 16);
    let mut acc = [0f32; LANES];
    let (x_lanes, x_tail) = x.as_chunks::<LANES>();
    let (w_lanes, w_tail) = w.as_chunks::<LANES>();
    for (x, w) in x_lanes.iter().zip(w_lanes) {
        for lane in 0..LANES {
            acc[lane] = x[lane].mul_add(widen(w[lane]), acc[lane]);
        }
    }
    acc.iter().sum::<f32>()
        + x_tail
            .iter()
            .zip(w_tail)
            .map(|(x, &w)| x * widen(w))
            .sum::<f32>()
}

// `HalfRowWeights` holds f16 only where `f16_dot_available` holds
fn dot_f16(x: &[f32], w: &[u16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: f16 rows are made only where F16C, AVX2 and FMA are
    return unsafe { x86::dot_f16(x, w) };
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (x, w);
        unreachable!("f16 rows are held on x86_64 alone")
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    // two 8-lane accumulators, each 8 halves widened by F16C
    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn dot_f16(x: &[f32], w: &[u16]) -> f32 {
        const STEP: usize = 16;
        let whole = x.len() / STEP * STEP;
        unsafe {
            let (mut a, mut b) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            for i in (0..whole).step_by(STEP) {
                let w0 = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(i).cast()));
                let w1 = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(i + 8).cast()));
                a = _mm256_fmadd_ps(_mm256_loadu_ps(x.as_ptr().add(i)), w0, a);
                b = _mm256_fmadd_ps(_mm256_loadu_ps(x.as_ptr().add(i + 8)), w1, b);
            }
            let mut lanes = [0f32; 8];
            _mm256_storeu_ps(lanes.as_mut_ptr(), _mm256_add_ps(a, b));
            let tail: f32 = x[whole..]
                .iter()
                .zip(&w[whole..])
                .map(|(x, &w)| x * half::f16::from_bits(w).to_f32())
                .sum();
            lanes.iter().sum::<f32>() + tail
        }
    }
}

#[cfg(test)]
mod tests {
    use super::HalfRowWeights;
    use crate::{DType, Device, Tensor};

    // against the f32 product over the weight rounded as it is held: bf16, and f16 where the CPU widens it (k = 1281: a
    // tail past whole 16-lane steps)
    #[test]
    fn matches_the_rounded_weights_product() -> crate::Result<()> {
        let w = Tensor::randn(0f32, 1., (300, 1281), &Device::Cpu)?;
        let x = Tensor::randn(0f32, 1., (2, 1281), &Device::Cpu)?;
        let formats: &[(bool, DType)] = if super::f16_dot_available() {
            &[(false, DType::BF16), (true, DType::F16)]
        } else {
            &[(false, DType::BF16)]
        };
        for &(f16, dtype) in formats {
            let got = HalfRowWeights::held_as(&w, f16)?.matmul_t(&x)?;
            let rounded = w.to_dtype(dtype)?.to_dtype(DType::F32)?;
            let want = x.matmul(&rounded.t()?)?;
            let worst = (got - want)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(worst < 1e-3, "{dtype:?}: worst {worst}");
        }
        Ok(())
    }
}
