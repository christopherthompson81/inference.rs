use crate::{DType, Device, Result, Tensor};

// a lane per accumulator, as `dot_f32`'s, so the loop vectorizes into 16-wide FMAs
const LANES: usize = 16;

/// An `(n, k)` f32 weight held as bf16 for a CPU's few-row products, which stream the whole weight a row: half the
/// bytes, and a weight near the last-level cache's size fits it. The products and their sums stay f32.
pub struct Bf16RowWeights {
    rows: Vec<u16>,
    n: usize,
    k: usize,
}

impl Bf16RowWeights {
    pub fn new(weight: &Tensor) -> Result<Self> {
        let (n, k) = weight.dims2()?;
        let rows = weight
            .to_dtype(DType::BF16)?
            .flatten_all()?
            .to_vec1::<half::bf16>()?
            .into_iter()
            .map(half::bf16::to_bits)
            .collect();
        Ok(Self { rows, n, k })
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
                    let dot = dot_bf16(x, &self.rows[col * k..][..k]);
                    // SAFETY: the chunks are disjoint columns of `out`, which outlives the call
                    unsafe { *(out_at as *mut f32).add(row * n + col) = dot };
                }
            }
        });
        Tensor::from_vec(out, (m, n), &Device::Cpu)
    }
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

#[cfg(test)]
mod tests {
    use super::Bf16RowWeights;
    use crate::{DType, Device, Tensor};

    #[test]
    fn matches_the_rounded_weights_product() -> crate::Result<()> {
        let w = Tensor::randn(0f32, 1., (300, 1281), &Device::Cpu)?;
        let x = Tensor::randn(0f32, 1., (2, 1281), &Device::Cpu)?;
        let got = Bf16RowWeights::new(&w)?.matmul_t(&x)?;
        let rounded = w.to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
        let want = x.matmul(&rounded.t()?)?;
        let worst = (got - want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(worst < 1e-3, "worst {worst}");
        Ok(())
    }
}
