//! A dense GGUF projection in a 32-value block type, dotted on the CPU straight from its blocks.

use super::cpu_dot::{self, QK32};
use super::cpu_experts::{Disjoint, for_each_parallel, quantize_rows_32};
use super::kernel::GgufType;

/// Most activation rows a call dots in place; a longer batch dequantizes once for the dense matmul instead.
pub(crate) const DENSE_DOT_MAX_TOKENS: usize = 32;
// Rows one register tile covers (a lane each) and the tokens it dots per pass
const TILE_ROWS: usize = 8;
const DOT_TOKENS: usize = 4;
// Output rows one parallel item covers
const ITEM_ROWS: usize = 64;

/// Whether [`matmul`] runs a `ty` weight `cols` wide.
pub(crate) fn supports(ty: GgufType, cols: usize) -> bool {
    cpu_dot::supports_32(ty) && cols.is_multiple_of(QK32)
}

/// `xs` (`tokens x cols`) times the transposed `rows x cols` weight in `bytes`, as `tokens x rows`.
pub(crate) fn matmul(ty: GgufType, bytes: &[u8], rows: usize, cols: usize, xs: &[f32]) -> Vec<f32> {
    let tokens = xs.len() / cols;
    let blocks = cols / QK32;
    let row_bytes = ty
        .row_bytes(cols)
        .expect("a 32-wide type has fixed-size rows");
    let q = quantize_rows_32(xs);
    let x = |t: usize| &q[t * blocks..][..blocks];
    let row = |r: usize| &bytes[r * row_bytes..][..row_bytes];
    let mut out = vec![0f32; tokens * rows];
    let dst = Disjoint(out.as_mut_ptr());
    for_each_parallel(rows.div_ceil(ITEM_ROWS), |item| {
        let end = rows.min((item + 1) * ITEM_ROWS);
        let mut r = item * ITEM_ROWS;
        // SAFETY: each item writes only its own rows' outputs
        let set = |t: usize, r: usize, v: f32| unsafe { dst.set(t * rows + r, v) };
        while r + TILE_ROWS <= end {
            let ws: [&[u8]; TILE_ROWS] = std::array::from_fn(|i| row(r + i));
            let mut t = 0;
            while t + DOT_TOKENS <= tokens {
                let dots = cpu_dot::vec_dot_32_8x4(ty, ws, std::array::from_fn(|k| x(t + k)));
                for (k, dots) in dots.iter().enumerate() {
                    for (i, &v) in dots.iter().enumerate() {
                        set(t + k, r + i, v);
                    }
                }
                t += DOT_TOKENS;
            }
            for t in t..tokens {
                for (i, v) in cpu_dot::vec_dot_32_8x1(ty, ws, x(t))
                    .into_iter()
                    .enumerate()
                {
                    set(t, r + i, v);
                }
            }
            r += TILE_ROWS;
        }
        for r in r..end {
            for t in 0..tokens {
                set(t, r, cpu_dot::vec_dot_32(ty, row(r), x(t)));
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // one item and a partial one, each ending in rows no full tile covers
    const ROWS: usize = ITEM_ROWS + 13;
    const COLS: usize = 3 * QK32;
    const TOLERANCE: f32 = 0.02;

    fn weight(ty: GgufType, seed: u32) -> Vec<u8> {
        let size = ty.type_size();
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(7);
        let mut bytes: Vec<u8> = (0..ROWS * ty.row_bytes(COLS).unwrap())
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            })
            .collect();
        for (b, block) in bytes.chunks_exact_mut(size).enumerate() {
            let d = 0.002 + 0.0001 * (b % 7) as f32;
            block[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
        }
        bytes
    }

    fn dequantized(ty: GgufType, bytes: &[u8]) -> Vec<f32> {
        match ty.candle() {
            Some(dtype) => inference_tensor::quantized::ggml_file::qtensor_from_ggml(
                dtype,
                bytes,
                vec![ROWS, COLS],
                &inference_tensor::Device::Cpu,
            )
            .and_then(|q| q.dequantize(&inference_tensor::Device::Cpu))
            .and_then(|t| t.flatten_all()?.to_vec1::<f32>())
            .unwrap(),
            None => super::super::raw::dequantize_rows(ty, COLS, bytes).unwrap(),
        }
    }

    #[test]
    fn dense_dots_match_the_dequantized_matmul() {
        let types = [
            GgufType::Iq4Nl,
            GgufType::Q4_0,
            GgufType::Q5_0,
            GgufType::Q8_0,
        ];
        for (seed, ty) in types.into_iter().enumerate() {
            let bytes = weight(ty, seed as u32);
            let w = dequantized(ty, &bytes);
            for tokens in [1, 3, 4, 6] {
                let xs: Vec<f32> = (0..tokens * COLS)
                    .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
                    .collect();
                let out = matmul(ty, &bytes, ROWS, COLS, &xs);
                for t in 0..tokens {
                    let x = &xs[t * COLS..][..COLS];
                    for r in 0..ROWS {
                        let row = &w[r * COLS..][..COLS];
                        let want: f32 = row.iter().zip(x).map(|(a, b)| a * b).sum();
                        let norm: f32 = row.iter().zip(x).map(|(a, b)| (a * b).abs()).sum();
                        let got = out[t * ROWS + r];
                        assert!(
                            (got - want).abs() <= TOLERANCE * norm,
                            "{ty:?} t{t} r{r}: {got} vs {want}"
                        );
                    }
                }
            }
        }
    }
}
