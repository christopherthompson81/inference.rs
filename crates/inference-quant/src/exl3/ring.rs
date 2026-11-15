//! exllamav3's trellis rings (`modules/quant/exl3_lib/ngram_codec.py`): a tail-biting mul1 trellis over `dim` values
//! with an f16 scale, packed as `1 + dim * K / 16` little-endian u16 words. Word 0 holds the scale; stream bits
//! `[i * K, (i + 1) * K)` are the low K bits of position i's 16-bit state, whose higher bits are the symbols of the
//! positions before it (mod dim). Hashed n-gram table rows are one 160-wide ring plus their hash head's bias; token
//! embedding rows are 256-wide groups, each a ring rotated by a Hadamard transform and a sign vector.

use std::sync::OnceLock;

use crate::gguf::trellis::MUL1;

/// An n-gram table row's width.
pub const NGRAM_ROW: usize = 160;
/// A token embedding group's width.
pub const EMBED_GROUP: usize = 256;
// the states a mul1 window takes
const STATES: usize = 1 << 16;

/// The u16 words of a `dim`-wide ring at K bits a value.
pub fn ring_words(dim: usize, k: usize) -> usize {
    1 + dim * k / 16
}

// every 16-bit state's mul1 value, as the CUDA decode rounds it (f16)
fn codebook() -> &'static [f32] {
    static TABLE: OnceLock<Vec<f32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        (0..STATES as u32)
            .map(|s| MUL1.value(MUL1.step(s)))
            .collect()
    })
}

// position i's 16-bit state: its own K-bit symbol and those of the positions before it, from bit 0 up
fn states(stream: &[u16], k: usize, dim: usize) -> Vec<u32> {
    let n = stream.len();
    let mask = (1u32 << k) - 1;
    let symbols = (0..dim)
        .map(|i| {
            let bit = i * k;
            let window =
                u32::from(stream[bit >> 4]) | (u32::from(stream[((bit >> 4) + 1) % n]) << 16);
            (window >> (bit & 15)) & mask
        })
        .collect::<Vec<_>>();
    let depth = (15 + k) / k;
    (0..dim)
        .map(|i| {
            (0..depth).fold(0u32, |state, j| {
                state | (symbols[(i + dim - j % dim) % dim] << (j * k))
            }) & 0xffff
        })
        .collect()
}

/// A ring (`words`, K bits a value) decoded into `out`, its `dim` values times the ring's scale.
pub fn decode_ring(words: &[u16], k: usize, out: &mut [f32]) {
    let scale = half::f16::from_bits(words[0]).to_f32();
    let book = codebook();
    let states = states(&words[1..], k, out.len());
    for (o, s) in out.iter_mut().zip(states) {
        *o = book[s as usize] * scale;
    }
}

/// An embedding row's groups (`words`, back to back) decoded into `out`, each rotated and multiplied by its signs
/// (`signs`: groups x EMBED_GROUP, f16 bits), the row cut to `out`'s width.
pub fn decode_embedding_row(words: &[u16], k: usize, signs: &[u16], out: &mut [f32]) {
    let per = ring_words(EMBED_GROUP, k);
    let mut group = [0f32; EMBED_GROUP];
    for (g, ring) in words.chunks_exact(per).enumerate() {
        decode_ring(ring, k, &mut group);
        super::hadamard(&mut group);
        let start = g * EMBED_GROUP;
        if start >= out.len() {
            break;
        }
        let sign = &signs[start..start + EMBED_GROUP];
        for (i, (o, v)) in out[start..].iter_mut().zip(group).enumerate() {
            *o = v * half::f16::from_bits(sign[i]).to_f32();
        }
    }
}

/// An embedding table stored as rings (`trellis`: `rows x groups * ring_words(EMBED_GROUP, K)` i16, `signs`: `groups x
/// EMBED_GROUP` f16) decoded to its dense `rows x dim` values, row by row in parallel.
pub fn dense_embedding(trellis: &[i16], signs: &[half::f16], rows: usize, dim: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let groups = dim.div_ceil(EMBED_GROUP);
    let row_words = trellis.len() / rows.max(1);
    let k = (row_words / groups - 1) * 16 / EMBED_GROUP;
    let signs = signs.iter().map(|s| s.to_bits()).collect::<Vec<_>>();
    let mut out = vec![0f32; rows * dim];
    out.par_chunks_mut(dim)
        .zip(trellis.par_chunks(row_words))
        .for_each(|(row, words)| {
            // SAFETY: i16 and u16 share size and alignment, and every bit pattern is valid for both
            let words =
                unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u16>(), words.len()) };
            decode_embedding_row(words, k, &signs, row);
        });
    out
}

/// A hashed n-gram table stored as rings: shards of `rows x ring_words(NGRAM_ROW, K)` words, and each hash head's
/// bias vector, added to its rows.
pub struct NgramRings {
    // (first row, the shard's words)
    shards: Vec<(usize, Vec<u16>)>,
    words: usize,
    k: usize,
    head_bias: Vec<f32>,
    head_offsets: Vec<i64>,
}

impl NgramRings {
    /// `shards` in row order (each `rows x words` i16), `head_bias` `heads x NGRAM_ROW`, `head_offsets` each head's
    /// first row.
    pub fn new(
        shards: Vec<(usize, Vec<i16>)>,
        words: usize,
        head_bias: Vec<f32>,
        head_offsets: Vec<i64>,
    ) -> inference_tensor::Result<Self> {
        let k = (words - 1) * 16 / NGRAM_ROW;
        if ring_words(NGRAM_ROW, k) != words || head_bias.len() != head_offsets.len() * NGRAM_ROW {
            inference_tensor::bail!(
                "an n-gram ring table of {words} words a row and {} head biases for {} heads",
                head_bias.len(),
                head_offsets.len()
            );
        }
        let shards = shards
            .into_iter()
            .map(|(first, words)| (first, words.into_iter().map(|w| w as u16).collect()))
            .collect();
        Ok(Self {
            shards,
            words,
            k,
            head_bias,
            head_offsets,
        })
    }

    /// Each hash head's first row, as the table was quantized.
    pub fn head_offsets(&self) -> &[i64] {
        &self.head_offsets
    }

    /// The rows in all shards.
    pub fn rows(&self) -> usize {
        self.shards
            .last()
            .map_or(0, |(first, words)| first + words.len() / self.words)
    }

    /// The rows `ids` decoded into `out` (`ids.len() x NGRAM_ROW`), each with its head's bias.
    pub fn gather(&self, ids: &[u32], out: &mut [f32]) -> inference_tensor::Result<()> {
        for (row, &id) in out.as_chunks_mut::<NGRAM_ROW>().0.iter_mut().zip(ids) {
            let id = id as usize;
            let shard = self.shards.partition_point(|(first, _)| *first <= id);
            let Some((first, words)) = shard.checked_sub(1).map(|s| &self.shards[s]) else {
                inference_tensor::bail!("n-gram row {id} precedes every shard");
            };
            let at = (id - first) * self.words;
            let Some(ring) = words.get(at..at + self.words) else {
                inference_tensor::bail!("n-gram row {id} falls past its shard");
            };
            decode_ring(ring, self.k, row);
            let head = self
                .head_offsets
                .partition_point(|&o| o <= id as i64)
                .saturating_sub(1);
            for (v, b) in row.iter_mut().zip(&self.head_bias[head * NGRAM_ROW..]) {
                *v += b;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // pack_rows' layout: position i's K-bit symbol at stream bits [i * K, (i + 1) * K), little-endian within words
    fn pack(symbols: &[u32], k: usize, scale: f32) -> Vec<u16> {
        let mut stream = vec![0u16; symbols.len() * k / 16];
        for (i, &s) in symbols.iter().enumerate() {
            for b in 0..k {
                let bit = i * k + b;
                stream[bit >> 4] |= (((s >> b) & 1) as u16) << (bit & 15);
            }
        }
        std::iter::once(half::f16::from_f32(scale).to_bits())
            .chain(stream)
            .collect()
    }

    #[test]
    fn rings_decode_each_position_from_its_own_and_earlier_symbols() {
        let mut seed = 0x9e3779b9u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for k in [1, 2, 3, 4, 5, 8] {
            let dim = NGRAM_ROW;
            let symbols = (0..dim)
                .map(|_| next() & ((1 << k) - 1))
                .collect::<Vec<_>>();
            let words = pack(&symbols, k, 0.75);
            assert_eq!(words.len(), ring_words(dim, k));
            let mut out = vec![0f32; dim];
            decode_ring(&words, k, &mut out);
            for i in 0..dim {
                let mut state = 0u32;
                for j in 0..16usize.div_ceil(k) {
                    state |= symbols[(i + dim * 4 - j) % dim] << (j * k);
                }
                let expected = MUL1.value(MUL1.step(state & 0xffff)) * 0.75;
                assert_eq!(out[i], expected, "K {k}, position {i}");
            }
        }
    }
}
