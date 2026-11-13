//! ExLlamaV3's EXL3 weights (exllamav3's `LinearEXL3`): a bitshift trellis per 16x16 tile, each weight the codebook
//! value of the 16-bit window that ends at its own K bits, over weights rotated by 128-wide Hadamard transforms on
//! both sides and scaled by the `suh` (input) and `svh` (output) vectors.

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub(crate) mod ffi;

#[cfg(feature = "cuda")]
pub use cuda::Exl3Layer;

use std::sync::Arc;

use inference_tensor::{DType, Device, Result, Tensor};

use crate::{QuantMethod, QuantMethodConfig, ShardedVarBuilder, UnquantLinear};

use crate::gguf::trellis::{Codebook, MCG, MUL1, THREE_INST};

const TILE: usize = 16;
const TILE_VALUES: usize = TILE * TILE;
const WINDOW_BITS: usize = 16;
// the Hadamard blocks on both sides
const HAD: usize = 128;

/// Bits per weight of a tile: `16 K` uint16 words, or `16 K + 8` for a half-integer K (alternating steps of K_A + 1
/// and K_A bits).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Exl3Bits {
    Whole(usize),
    Half(usize),
}

impl Exl3Bits {
    /// The bitrate of tiles `words` uint16 wide.
    pub fn from_words(words: usize) -> Result<Self> {
        match (words % TILE, words / TILE) {
            (0, k @ 1..=8) => Ok(Self::Whole(k)),
            (8, ka @ 1..=7) => Ok(Self::Half(ka)),
            _ => inference_tensor::bail!("an EXL3 tile of {words} uint16 words has no bitrate"),
        }
    }

    // the stream bit (exclusive) where position `p`'s window ends
    fn window_end(self, p: usize) -> usize {
        match self {
            Self::Whole(k) => (p + 1) * k,
            Self::Half(ka) => {
                let pair_end = (p / 2 + 1) * (2 * ka + 1);
                if p % 2 == 1 {
                    pair_end
                } else {
                    pair_end - (ka + 1)
                }
            }
        }
    }
}

/// The codebook a tensor decodes through.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Exl3Codebook {
    // the kernels' `cb` numbers
    ThreeInst = 0,
    Mcg = 1,
    Mul1 = 2,
}

impl Exl3Codebook {
    /// The codebook a tensor names by its marker (`mcg` or `mul1`, each holding its multiplier), else 3INST.
    pub fn from_markers(mcg: bool, mul1: bool) -> Self {
        match (mcg, mul1) {
            (true, _) => Self::Mcg,
            (_, true) => Self::Mul1,
            _ => Self::ThreeInst,
        }
    }

    fn codebook(self) -> Codebook {
        match self {
            Self::ThreeInst => THREE_INST,
            Self::Mcg => MCG,
            Self::Mul1 => MUL1,
        }
    }
}

// the stream position each row-major tile element is encoded at, the tensor-core order exllamav3's kernels read
fn tile_order() -> [usize; TILE_VALUES] {
    let mut perm = [0usize; TILE_VALUES];
    for t in 0..32 {
        let (r0, c0) = ((t % 4) * 2, t / 4);
        let cells = [(r0, c0), (r0 + 1, c0), (r0 + 8, c0), (r0 + 9, c0)];
        for (i, (r, c)) in cells
            .iter()
            .chain(&cells.map(|(r, c)| (r, c + 8)))
            .enumerate()
        {
            perm[t * 8 + i] = r * TILE + c;
        }
    }
    perm
}

/// One 16x16 tile's values, row major, from its `words` uint16 words.
pub fn decode_tile(words: &[u16], bits: Exl3Bits, codebook: Exl3Codebook) -> [f32; TILE_VALUES] {
    let codebook = codebook.codebook();
    // the stream reads each pair of words as a little-endian u32, most significant bit first
    let stream = words
        .as_chunks::<2>()
        .0
        .iter()
        .map(|w| u32::from(w[0]) | (u32::from(w[1]) << 16))
        .collect::<Vec<_>>();
    let total = stream.len() * 32;
    let perm = tile_order();
    let mut tile = [0f32; TILE_VALUES];
    for (p, &cell) in perm.iter().enumerate() {
        let start = (bits.window_end(p) + total - WINDOW_BITS) % total;
        let (i, s) = (start / 32, start % 32);
        let pair = (u64::from(stream[i]) << 32) | u64::from(stream[(i + 1) % stream.len()]);
        let window = ((pair >> (64 - WINDOW_BITS - s)) & 0xffff) as u32;
        tile[cell] = codebook.value(codebook.step(window));
    }
    tile
}

// an in-place Walsh-Hadamard transform of `x` (Sylvester order), scaled to keep norms
fn hadamard(x: &mut [f32]) {
    let mut h = 1;
    while h < x.len() {
        for block in x.chunks_exact_mut(2 * h) {
            let (a, b) = block.split_at_mut(h);
            for (a, b) in a.iter_mut().zip(b) {
                (*a, *b) = (*a + *b, *a - *b);
            }
        }
        h *= 2;
    }
    let scale = 1.0 / (x.len() as f32).sqrt();
    x.iter_mut().for_each(|v| *v *= scale);
}

/// One EXL3 linear's stored tensors.
pub struct Exl3Weight<'a> {
    /// `(in / 16, out / 16, words)` uint16 words, tile by tile.
    pub trellis: &'a [u16],
    pub in_features: usize,
    pub out_features: usize,
    pub words: usize,
    pub suh: &'a [f32],
    pub svh: &'a [f32],
    pub codebook: Exl3Codebook,
}

impl Exl3Weight<'_> {
    /// The dense weight, `out_features x in_features` row major (the transpose of exllamav3's `[in, out]`).
    pub fn dequantize(&self) -> Result<Vec<f32>> {
        let (k, n) = (self.in_features, self.out_features);
        if !k.is_multiple_of(HAD)
            || !n.is_multiple_of(HAD)
            || self.trellis.len() != k * n / TILE_VALUES * self.words
        {
            inference_tensor::bail!(
                "an EXL3 trellis of {} words for {k}x{n} weights",
                self.trellis.len()
            );
        }
        let bits = Exl3Bits::from_words(self.words)?;
        // exllamav3's [in, out] matrix
        let mut w = vec![0f32; k * n];
        for (t, words) in self.trellis.chunks_exact(self.words).enumerate() {
            let (tk, tn) = (t / (n / TILE), t % (n / TILE));
            let tile = decode_tile(words, bits, self.codebook);
            for r in 0..TILE {
                w[(tk * TILE + r) * n + tn * TILE..][..TILE]
                    .copy_from_slice(&tile[r * TILE..][..TILE]);
            }
        }
        // the input side's rotation runs down each column, 128 rows at a time
        let mut column = vec![0f32; HAD];
        for c in 0..n {
            for b in (0..k).step_by(HAD) {
                for (i, v) in column.iter_mut().enumerate() {
                    *v = w[(b + i) * n + c];
                }
                hadamard(&mut column);
                for (i, v) in column.iter().enumerate() {
                    w[(b + i) * n + c] = v * self.suh[b + i];
                }
            }
        }
        let mut out = vec![0f32; n * k];
        for (r, row) in w.chunks_exact_mut(n).enumerate() {
            for block in row.as_chunks_mut::<HAD>().0 {
                hadamard(block);
            }
            for (c, v) in row.iter().enumerate() {
                out[c * k + r] = v * self.svh[c];
            }
        }
        Ok(out)
    }
}

/// An EXL3 linear's dense `out x in` weight as a tensor of `dtype` on `device`, from its stored tensors.
pub fn dense_weight(
    trellis: &Tensor,
    suh: &Tensor,
    svh: &Tensor,
    codebook: Exl3Codebook,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let [tk, tn, words] = <[usize; 3]>::try_from(trellis.dims()).map_err(|_| {
        inference_tensor::Error::msg(format!("an EXL3 trellis of shape {:?}", trellis.dims()))
    })?;
    let trellis = trellis
        .to_device(&Device::Cpu)?
        .to_dtype(DType::I16)?
        .flatten_all()?
        .to_vec1::<i16>()?
        .into_iter()
        .map(|w| w as u16)
        .collect::<Vec<_>>();
    let vec = |t: &Tensor| {
        t.to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()
    };
    let (suh, svh) = (vec(suh)?, vec(svh)?);
    let weight = Exl3Weight {
        trellis: &trellis,
        in_features: tk * TILE,
        out_features: tn * TILE,
        words,
        suh: &suh,
        svh: &svh,
        codebook,
    }
    .dequantize()?;
    Tensor::from_vec(weight, (tn * TILE, tk * TILE), &Device::Cpu)?
        .to_dtype(dtype)?
        .to_device(device)
}

/// An EXL3 checkpoint's linear: its trellis reconstructed into a dense layer, or the plain weight EXL3 left
/// unquantized.
pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: &ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let bias = bias.then(|| vb.get((out_dim,), "bias")).transpose()?;
    let weight = if vb.contains_tensor("trellis") {
        let trellis = vb.get_unchecked_dtype("trellis", DType::I16)?;
        if trellis.dims().len() != 3 || trellis.dims()[..2] != [in_dim / TILE, out_dim / TILE] {
            inference_tensor::bail!(
                "{}: an EXL3 trellis of {:?} for a {in_dim} -> {out_dim} linear",
                vb.prefix(),
                trellis.dims()
            );
        }
        let suh = vb.get_unchecked_dtype("suh", DType::F32)?;
        let svh = vb.get_unchecked_dtype("svh", DType::F32)?;
        let codebook =
            Exl3Codebook::from_markers(vb.contains_tensor("mcg"), vb.contains_tensor("mul1"));
        // on a GPU the trellis stays as it is, multiplied by exllamav3's kernels
        #[cfg(feature = "cuda")]
        if vb.device().is_cuda() {
            let layer = Exl3Layer::new_on_device(trellis, suh, svh, codebook, bias, vb.dtype())?;
            return Ok(Arc::new(layer));
        }
        dense_weight(&trellis, &suh, &svh, codebook, vb.dtype(), vb.device())?
    } else {
        vb.get_with_hints((out_dim, in_dim), "weight", Default::default())?
    };
    Ok(Arc::new(UnquantLinear::new(
        QuantMethodConfig::Unquantized(inference_tensor::nn::Linear::new(weight, bias)),
    )?))
}

#[cfg(test)]
mod tests {
    use super::{Exl3Bits, Exl3Codebook, decode_tile, tile_order};

    fn words(n: usize, seed: u32) -> Vec<u16> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 16) as u16
            })
            .collect()
    }

    // random tiles as exllamav3's `reconstruct` kernel decodes them: the tile's sum, its sum of squares, and its
    // first three values, for each codebook, whole bitrates 1 to 8 and two half-integer ones
    #[test]
    fn tiles_decode_as_exllamav3_does() {
        use Exl3Codebook::{Mcg, Mul1, ThreeInst};
        let cases: [(usize, Exl3Codebook, u32, f64, f64, [f32; 3]); 7] = [
            (
                48,
                ThreeInst,
                1,
                6.380371,
                387.513184,
                [1.145508, 0.332764, -1.255859],
            ),
            (
                64,
                Mcg,
                2,
                34.632568,
                414.628387,
                [-0.002563, 0.436279, -1.045898],
            ),
            (
                96,
                Mul1,
                3,
                6.956253,
                244.281143,
                [0.756348, -0.84082, -0.482178],
            ),
            (
                16,
                ThreeInst,
                4,
                33.559448,
                424.375732,
                [0.044189, 0.723633, -1.753906],
            ),
            (
                128,
                Mcg,
                5,
                -1.617065,
                413.472168,
                [-1.167969, -1.242188, -1.500977],
            ),
            (
                24,
                Mul1,
                6,
                -1.386124,
                251.795593,
                [-0.008583, -0.252197, -0.637695],
            ),
            (
                40,
                Mul1,
                7,
                16.217773,
                257.338562,
                [1.196289, -1.672852, -2.349609],
            ),
        ];
        for (n, codebook, seed, sum, sum2, first) in cases {
            let bits = Exl3Bits::from_words(n).unwrap();
            let tile = decode_tile(&words(n, seed), bits, codebook);
            let s = tile.iter().map(|&v| f64::from(v)).sum::<f64>();
            let s2 = tile
                .iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>();
            assert!(
                (s - sum).abs() < 1e-3 && (s2 - sum2).abs() < 1e-3,
                "{bits:?} {codebook:?}: {s} {s2}"
            );
            for (v, e) in tile.iter().zip(first) {
                assert!((v - e).abs() < 1e-6, "{bits:?} {codebook:?}: {v} vs {e}");
            }
        }
    }

    #[test]
    fn the_tile_order_visits_every_cell_once() {
        let mut seen = tile_order();
        seen.sort_unstable();
        assert!(seen.iter().enumerate().all(|(i, &c)| i == c));
    }

    #[test]
    fn half_steps_alternate_and_fill_the_tile() {
        let bits = Exl3Bits::Half(1);
        assert_eq!(
            (0..4).map(|p| bits.window_end(p)).collect::<Vec<_>>(),
            [1, 3, 4, 6]
        );
        assert_eq!(bits.window_end(255), 24 * 16);
        assert_eq!(Exl3Bits::from_words(24).unwrap(), bits);
        assert_eq!(Exl3Bits::from_words(48).unwrap(), Exl3Bits::Whole(3));
    }
}
