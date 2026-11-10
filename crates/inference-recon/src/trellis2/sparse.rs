//! Sparse voxel features and the submanifold 3^3 convolution over them (FlexGEMM's `sparse_submanifold_conv3d`): each
//! active cell gathers its 27 neighbours' features (zero where a neighbour is inactive) and one matmul mixes them.

use std::collections::HashMap;

use inference_tensor::{DType, Device, Result, Tensor};
use rayon::prelude::*;

pub const TAPS: usize = 27;
// bits a coordinate takes in a cell's hash key: grids up to 2048 a side
const KEY_BITS: u32 = 11;
// the most im2col bytes one chunk of a convolution may hold
const CHUNK_BYTES: usize = 512 << 20;

fn key([x, y, z]: [i32; 3]) -> u64 {
    ((x as u64) << (2 * KEY_BITS)) | ((y as u64) << KEY_BITS) | z as u64
}

/// Active cells: their integer coordinates (host) and their features `[cells, channels]` (device).
#[derive(Clone)]
pub struct Sparse {
    pub coords: Vec<[i32; 3]>,
    pub feats: Tensor,
}

impl Sparse {
    pub fn replace(&self, feats: Tensor) -> Self {
        Self {
            coords: self.coords.clone(),
            feats,
        }
    }
}

/// Each cell's 27 neighbours as rows of the feature table, row `cells` (one past the end) for an inactive one; tap
/// `v` is the offset `(v / 9 - 1, v / 3 % 3 - 1, v % 3 - 1)`, as the weights' `[out, 3, 3, 3, in]` layout orders it.
pub struct Neighbors {
    // `[cells * 27]` (u32, device)
    index: Tensor,
    cells: usize,
}

impl Neighbors {
    pub fn new(coords: &[[i32; 3]], device: &Device) -> Result<Self> {
        let cells = coords.len();
        let table: HashMap<u64, u32> = coords
            .iter()
            .enumerate()
            .map(|(i, &c)| (key(c), i as u32))
            .collect();
        let index: Vec<u32> = coords
            .par_iter()
            .flat_map_iter(|&[x, y, z]| {
                let table = &table;
                (0..TAPS as i32).map(move |v| {
                    let n = [x + v / 9 - 1, y + v / 3 % 3 - 1, z + v % 3 - 1];
                    if n.iter().any(|&c| c < 0) {
                        return cells as u32;
                    }
                    table.get(&key(n)).copied().unwrap_or(cells as u32)
                })
            })
            .collect();
        Ok(Self {
            index: Tensor::from_vec(index, cells * TAPS, device)?,
            cells,
        })
    }
}

/// A submanifold sparse convolution, kernel 3, weights `[taps * in, out]` in its dtype.
pub struct SparseConv {
    weight: Tensor,
    bias: Tensor,
    out: usize,
}

impl SparseConv {
    pub fn load(
        cin: usize,
        cout: usize,
        dtype: DType,
        vb: inference_tensor::nn::VarBuilder,
    ) -> Result<Self> {
        let weight = vb
            .get((cout, 3, 3, 3, cin), "weight")?
            .reshape((cout, TAPS * cin))?
            .t()?
            .contiguous()?
            .to_dtype(dtype)?;
        Ok(Self {
            weight,
            bias: vb.get(cout, "bias")?.to_dtype(dtype)?,
            out: cout,
        })
    }

    /// A dense `nn.Conv3d` (kernel 3, padding 1) as a submanifold convolution over a full grid: its weights
    /// `[out, in, 3, 3, 3]` reordered to the taps' `[out, 3, 3, 3, in]`.
    pub fn load_dense(
        cin: usize,
        cout: usize,
        dtype: DType,
        vb: inference_tensor::nn::VarBuilder,
    ) -> Result<Self> {
        let weight = vb
            .get((cout, cin, 3, 3, 3), "weight")?
            .permute((0, 2, 3, 4, 1))?
            .reshape((cout, TAPS * cin))?
            .t()?
            .contiguous()?
            .to_dtype(dtype)?;
        Ok(Self {
            weight,
            bias: vb.get(cout, "bias")?.to_dtype(dtype)?,
            out: cout,
        })
    }

    /// `feats` `[cells, in]` convolved over `neighbors`: `[cells, out]`, in the weights' dtype.
    pub fn forward(&self, feats: &Tensor, neighbors: &Neighbors) -> Result<Tensor> {
        let feats = feats.to_dtype(self.weight.dtype())?;
        let (cells, cin) = feats.dims2()?;
        let padded = Tensor::cat(&[&feats, &feats.zeros_like()?.narrow(0, 0, 1)?], 0)?;
        let row_bytes = TAPS * cin * feats.dtype().size_in_bytes();
        let chunk = (CHUNK_BYTES / row_bytes).max(1);
        let mut parts = Vec::with_capacity(cells.div_ceil(chunk));
        let mut start = 0;
        while start < neighbors.cells {
            let n = chunk.min(neighbors.cells - start);
            let index = neighbors.index.narrow(0, start * TAPS, n * TAPS)?;
            let cols = padded.index_select(&index, 0)?.reshape((n, TAPS * cin))?;
            parts.push(cols.matmul(&self.weight)?.broadcast_add(&self.bias)?);
            start += n;
        }
        let out = Tensor::cat(&parts, 0)?;
        debug_assert_eq!(out.dim(1)?, self.out);
        Ok(out)
    }
}

/// The children each cell keeps (`keep[cell][child]`, child bit 0 for x, 1 for y, 2 for z) as their own cells at
/// twice the resolution: their coordinates, and for each the row of the parent's `[cells * 8, channels / 8]` features
/// it takes (`SparseChannel2Spatial`).
pub fn subdivide(coords: &[[i32; 3]], keep: &[[bool; 8]]) -> (Vec<[i32; 3]>, Vec<u32>) {
    let mut children = Vec::new();
    let mut rows = Vec::new();
    for (i, (c, k)) in coords.iter().zip(keep).enumerate() {
        for (j, _) in k.iter().enumerate().filter(|&(_, &kept)| kept) {
            let j = j as i32;
            children.push([
                2 * c[0] + (j & 1),
                2 * c[1] + ((j >> 1) & 1),
                2 * c[2] + ((j >> 2) & 1),
            ]);
            rows.push((i * 8) as u32 + j as u32);
        }
    }
    (children, rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dense_block_convolves_as_a_dense_conv3d() -> Result<()> {
        // a full 3 x 4 x 5 block: the sparse conv is conv3d with zero padding 1
        let (cin, cout) = (2, 3);
        let dev = Device::Cpu;
        let coords: Vec<[i32; 3]> = (0..3)
            .flat_map(|x| (0..4).flat_map(move |y| (0..5).map(move |z| [x, y, z])))
            .collect();
        let n = coords.len();
        let feats = Tensor::randn(0f32, 1.0, (n, cin), &dev)?;
        let w = Tensor::randn(0f32, 1.0, (cout, 3, 3, 3, cin), &dev)?;
        let b = Tensor::randn(0f32, 1.0, cout, &dev)?;
        let conv = SparseConv {
            weight: w.reshape((cout, TAPS * cin))?.t()?.contiguous()?,
            bias: b.clone(),
            out: cout,
        };
        let got = conv
            .forward(&feats, &Neighbors::new(&coords, &dev)?)?
            .to_vec2::<f32>()?;
        let f = feats.to_vec2::<f32>()?;
        let w = w.flatten_all()?.to_vec1::<f32>()?;
        let b = b.to_vec1::<f32>()?;
        let at = |p: [i32; 3]| coords.iter().position(|&c| c == p);
        for (i, &[x, y, z]) in coords.iter().enumerate() {
            for o in 0..cout {
                let mut want = b[o];
                for v in 0..TAPS as i32 {
                    if let Some(j) = at([x + v / 9 - 1, y + v / 3 % 3 - 1, z + v % 3 - 1]) {
                        for c in 0..cin {
                            want += w[(o * TAPS + v as usize) * cin + c] * f[j][c];
                        }
                    }
                }
                assert!((got[i][o] - want).abs() < 1e-4, "cell {i} out {o}");
            }
        }
        Ok(())
    }

    #[test]
    fn subdivision_keeps_children_in_bit_order() {
        let keep = [[true, false, false, true, false, false, false, true]];
        let (children, rows) = subdivide(&[[1, 2, 3]], &keep);
        assert_eq!(children, [[2, 4, 6], [3, 5, 6], [3, 5, 7]]);
        assert_eq!(rows, [0, 3, 7]);
    }
}
