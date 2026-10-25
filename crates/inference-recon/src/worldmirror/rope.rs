//! WorldMirror's normalized 2D RoPE: each token rotates by the normalized coordinates of its cell in a grid one larger
//! than the patch grid (the special tokens sit at cell (0, 0)), half the pairs by the row and half by the column.

use half::bf16;
use inference_tensor::{DType, Device, Result, Tensor};

/// Cosines and sines over every cell of a grid, `[cells, head_dim]`, in the dtype the attention runs.
pub(crate) struct RopeTable {
    cos: Tensor,
    sin: Tensor,
    half: usize,
}

// One value through bf16, as each op of the reference rounds its bf16 result
fn r(v: f32) -> f32 {
    bf16::from_f32(v).to_f32()
}

impl RopeTable {
    /// The table for a `rows x cols` grid of cells. In bf16 every step rounds as the reference's bf16 ops do; in f32
    /// it is the exact table.
    pub(crate) fn new(
        rows: usize,
        cols: usize,
        head_dim: usize,
        base: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let quarter = head_dim / 4;
        let round = |v: f32| if dtype == DType::BF16 { r(v) } else { v };
        // the checkpoint's f32 periods, 100^(2k / half), then cast to the attention dtype
        let periods = (0..quarter)
            .map(|k| round(base.powf(2.0 * k as f64 / (head_dim / 2) as f64) as f32))
            .collect::<Vec<_>>();
        let coord = |i: usize, n: usize| round(2.0 * round((i as f32 + 0.5) / n as f32) - 1.0);
        let two_pi = (2.0 * std::f64::consts::PI) as f32;
        let mut cos = Vec::with_capacity(rows * cols * head_dim);
        let mut sin = Vec::with_capacity(rows * cols * head_dim);
        for y in 0..rows {
            for x in 0..cols {
                let angles = [coord(y, rows), coord(x, cols)]
                    .iter()
                    .flat_map(|&c| periods.iter().map(move |&p| (c, p)))
                    .map(|(c, p)| round(round(two_pi * c) / p))
                    .collect::<Vec<_>>();
                for _ in 0..2 {
                    cos.extend(angles.iter().map(|&a| round(a.cos())));
                    sin.extend(angles.iter().map(|&a| round(a.sin())));
                }
            }
        }
        let shape = (rows * cols, head_dim);
        Ok(Self {
            cos: Tensor::from_vec(cos, shape, device)?.to_dtype(dtype)?,
            sin: Tensor::from_vec(sin, shape, device)?.to_dtype(dtype)?,
            half: head_dim / 2,
        })
    }

    /// `t * cos + rotate_half(t) * sin` for `t` `[b, heads, n, head_dim]`, its `n` tokens at `cells` (`[n]` u32).
    pub(crate) fn apply(&self, t: &Tensor, cells: &Tensor) -> Result<Tensor> {
        let n = cells.dim(0)?;
        let cos = self
            .cos
            .index_select(cells, 0)?
            .reshape((1, 1, n, 2 * self.half))?;
        let sin = self
            .sin
            .index_select(cells, 0)?
            .reshape((1, 1, n, 2 * self.half))?;
        let first = t.narrow(3, 0, self.half)?;
        let second = t.narrow(3, self.half, self.half)?;
        let rotated = Tensor::cat(&[&second.neg()?, &first], 3)?;
        t.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?
    }
}

/// The cell of every token of one frame: the `specials` tokens at (0, 0), then patch (y, x) at (y + 1, x + 1), in a
/// grid `wp + 1` wide.
pub(crate) fn frame_cells(specials: usize, hp: usize, wp: usize) -> Vec<u32> {
    let width = wp + 1;
    let mut cells = vec![0u32; specials];
    cells.extend((0..hp).flat_map(|y| (0..wp).map(move |x| ((y + 1) * width + x + 1) as u32)));
    cells
}
