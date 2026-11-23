//! Q6_K regrouped for the tensor-core kernels: every 16 rows' 256-value superblocks as one 3360-byte tile, the bytes
//! the same 16 GGUF blocks take, read in two 128-value halves. A tile holds 8 entries of 4 bytes (row e's f16 `d`,
//! then row e + 8's), then 8 entries of 32 sub-block scales (per half: row e's 8, then row e + 8's), then per half 32
//! lanes' 48 bytes: lane 4 g + t's word i holds the low four bits of half sub-block i's values 4 t .. 4 t + 3, row g
//! in the low nibbles and row g + 8 in the high ones, and its 16 bytes after them the top two bits (byte 2 i row g's
//! four values, 2 i + 1 row g + 8's, value j at bits 2 j). So a lane reads a half's mma.m16n8k16 A fragments with
//! three 16-byte loads and its rows' scales with one.

use rayon::prelude::*;

pub(crate) const TILE_ROWS: usize = 16;
const HALF_ROWS: usize = TILE_ROWS / 2;
pub(crate) const BLOCK_VALUES: usize = 256;
const BLOCK_BYTES: usize = 210;
const QL: usize = 0;
const QH: usize = 128;
const SC: usize = 192;
const D: usize = 208;
const D_BYTES: usize = HALF_ROWS * 4;
const SC_BYTES: usize = HALF_ROWS * 32;
const HALVES: usize = 2;
const HALF_VALUES: usize = BLOCK_VALUES / HALVES;
const SUB_VALUES: usize = 16;
const HALF_SUBS: usize = HALF_VALUES / SUB_VALUES;
pub(crate) const TILE_BYTES: usize = TILE_ROWS * BLOCK_BYTES;
const LANES: usize = 32;
const LANE_BYTES: usize = 48;
const QUAD: usize = 4;
const WORD_VALUES: usize = 4;
const HALF_BYTES: usize = LANES * LANE_BYTES;

/// Whether `rows` x `cols` Q6_K matrices regroup into whole tiles.
pub(crate) fn tiles(rows: usize, cols: usize) -> bool {
    rows.is_multiple_of(TILE_ROWS) && cols.is_multiple_of(BLOCK_VALUES)
}

// value `v`'s six bits in a GGUF Q6_K block: four in `ql` (byte 64 n + l + 32 (q % 2), the high nibble from q = 2),
// two in `qh` (byte 32 n + l, at bit 2 q), for n = v / 128, q = v % 128 / 32, l = v % 32
fn gguf_q(block: &[u8], v: usize) -> u8 {
    let (n, q, l) = (v / 128, v % 128 / 32, v % 32);
    let ql = block[QL + 64 * n + l + 32 * (q % 2)];
    let low = if q < 2 { ql & 0x0F } else { ql >> 4 };
    let high = (block[QH + 32 * n + l] >> (2 * q)) & 3;
    low | (high << 4)
}

fn set_gguf_q(block: &mut [u8], v: usize, value: u8) {
    let (n, q, l) = (v / 128, v % 128 / 32, v % 32);
    let ql = &mut block[QL + 64 * n + l + 32 * (q % 2)];
    if q < 2 {
        *ql = (*ql & 0xF0) | (value & 0x0F);
    } else {
        *ql = (*ql & 0x0F) | ((value & 0x0F) << 4);
    }
    let qh = &mut block[QH + 32 * n + l];
    *qh = (*qh & !(3 << (2 * q))) | (((value >> 4) & 3) << (2 * q));
}

fn half_base(h: usize) -> usize {
    D_BYTES + SC_BYTES + h * HALF_BYTES
}

// the ql byte of lane (g, t) for half `h`'s sub-block `i`, value 4 t + j (both rows), and its qh byte for `upper`
fn ql_byte(g: usize, t: usize, h: usize, i: usize, j: usize) -> usize {
    half_base(h) + (QUAD * g + t) * LANE_BYTES + WORD_VALUES * i + j
}

fn qh_byte(g: usize, t: usize, h: usize, i: usize, upper: usize) -> usize {
    half_base(h) + (QUAD * g + t) * LANE_BYTES + WORD_VALUES * HALF_SUBS + 2 * i + upper
}

fn sc_byte(e: usize, h: usize, upper: usize, i: usize) -> usize {
    D_BYTES + e * 32 + h * 16 + upper * 8 + i
}

fn tile_from_blocks(blocks: [&[u8]; TILE_ROWS], tile: &mut [u8]) {
    for (r, block) in blocks.iter().enumerate() {
        let (e, upper) = (r % HALF_ROWS, r / HALF_ROWS);
        tile[e * 4 + upper * 2..][..2].copy_from_slice(&block[D..D + 2]);
        for h in 0..HALVES {
            for i in 0..HALF_SUBS {
                tile[sc_byte(e, h, upper, i)] = block[SC + h * HALF_SUBS + i];
                for t in 0..QUAD {
                    let mut high = 0u8;
                    for j in 0..WORD_VALUES {
                        let q = gguf_q(
                            block,
                            HALF_VALUES * h + SUB_VALUES * i + WORD_VALUES * t + j,
                        );
                        let byte = &mut tile[ql_byte(e, t, h, i, j)];
                        *byte |= (q & 0x0F) << (4 * upper);
                        high |= (q >> 4) << (2 * j);
                    }
                    tile[qh_byte(e, t, h, i, upper)] = high;
                }
            }
        }
    }
}

fn block_from_tile(tile: &[u8], r: usize, block: &mut [u8]) {
    let (e, upper) = (r % HALF_ROWS, r / HALF_ROWS);
    block[D..D + 2].copy_from_slice(&tile[e * 4 + upper * 2..][..2]);
    for h in 0..HALVES {
        for i in 0..HALF_SUBS {
            block[SC + h * HALF_SUBS + i] = tile[sc_byte(e, h, upper, i)];
            for t in 0..QUAD {
                let high = tile[qh_byte(e, t, h, i, upper)];
                for j in 0..WORD_VALUES {
                    let low = (tile[ql_byte(e, t, h, i, j)] >> (4 * upper)) & 0x0F;
                    let q = low | (((high >> (2 * j)) & 3) << 4);
                    set_gguf_q(
                        block,
                        HALF_VALUES * h + SUB_VALUES * i + WORD_VALUES * t + j,
                        q,
                    );
                }
            }
        }
    }
}

/// Row-major Q6_K matrices (`matrices` of `rows` x `cols`, back to back) in the tile order; `tiles` must hold.
pub(crate) fn to_tiled(bytes: &[u8], matrices: usize, rows: usize, cols: usize) -> Vec<u8> {
    let nb = cols / BLOCK_VALUES;
    let row_bytes = nb * BLOCK_BYTES;
    debug_assert!(tiles(rows, cols) && bytes.len() == matrices * rows * row_bytes);
    let mut out = vec![0u8; bytes.len()];
    out.par_chunks_exact_mut(nb * TILE_BYTES)
        .zip(bytes.par_chunks_exact(TILE_ROWS * row_bytes))
        .for_each(|(group, rows)| {
            for (kb, tile) in group.as_chunks_mut::<TILE_BYTES>().0.iter_mut().enumerate() {
                let blocks = std::array::from_fn(|r| {
                    &rows[r * row_bytes + kb * BLOCK_BYTES..][..BLOCK_BYTES]
                });
                tile_from_blocks(blocks, tile);
            }
        });
    out
}

/// `to_tiled`'s inverse: the matrices back in GGUF's row-major blocks.
pub(crate) fn from_tiled(tiled: &[u8], matrices: usize, rows: usize, cols: usize) -> Vec<u8> {
    let nb = cols / BLOCK_VALUES;
    let row_bytes = nb * BLOCK_BYTES;
    debug_assert!(tiles(rows, cols) && tiled.len() == matrices * rows * row_bytes);
    let mut out = vec![0u8; tiled.len()];
    out.par_chunks_exact_mut(TILE_ROWS * row_bytes)
        .zip(tiled.par_chunks_exact(nb * TILE_BYTES))
        .for_each(|(rows, group)| {
            for (r, row) in rows.chunks_exact_mut(row_bytes).enumerate() {
                for (kb, block) in row.as_chunks_mut::<BLOCK_BYTES>().0.iter_mut().enumerate() {
                    block_from_tile(&group[kb * TILE_BYTES..][..TILE_BYTES], r, block);
                }
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 131 + i / 7) as u8).collect()
    }

    #[test]
    fn q6k_tiles_round_trip_to_the_gguf_blocks() {
        let (matrices, rows, cols) = (2, 32, 512);
        let bytes = pattern(matrices * rows * cols / BLOCK_VALUES * BLOCK_BYTES);
        let tiled = to_tiled(&bytes, matrices, rows, cols);
        assert_ne!(tiled, bytes);
        assert_eq!(from_tiled(&tiled, matrices, rows, cols), bytes);
    }

    #[test]
    fn a_q6k_lane_word_holds_two_rows_low_bits_and_its_bytes_the_high() {
        let bytes = pattern(TILE_ROWS * BLOCK_BYTES);
        let tiled = to_tiled(&bytes, 1, TILE_ROWS, BLOCK_VALUES);
        let block = |r: usize| &bytes[r * BLOCK_BYTES..(r + 1) * BLOCK_BYTES];
        for (g, t, h, i, j) in [(0, 0, 0, 0, 0), (3, 2, 1, 5, 3), (7, 3, 1, 7, 1)] {
            let v = HALF_VALUES * h + SUB_VALUES * i + WORD_VALUES * t + j;
            let (a, b) = (gguf_q(block(g), v), gguf_q(block(g + HALF_ROWS), v));
            assert_eq!(
                tiled[ql_byte(g, t, h, i, j)],
                (a & 0x0F) | ((b & 0x0F) << 4)
            );
            assert_eq!((tiled[qh_byte(g, t, h, i, 0)] >> (2 * j)) & 3, a >> 4);
            assert_eq!((tiled[qh_byte(g, t, h, i, 1)] >> (2 * j)) & 3, b >> 4);
        }
        assert_eq!(D_BYTES + SC_BYTES + HALVES * HALF_BYTES, TILE_BYTES);
        assert_eq!(LANE_BYTES, WORD_VALUES * HALF_SUBS + 2 * HALF_SUBS);
    }

    #[test]
    fn every_q6k_value_reads_back_as_it_was_set() {
        let mut block = vec![0u8; BLOCK_BYTES];
        for v in 0..BLOCK_VALUES {
            set_gguf_q(&mut block, v, (v * 7 % 64) as u8);
        }
        for v in 0..BLOCK_VALUES {
            assert_eq!(gguf_q(&block, v), (v * 7 % 64) as u8);
        }
    }
}
