//! Q8_0 regrouped for the tensor-core kernels: every 16 rows' 32-value blocks as one 544-byte tile, the bytes the same
//! 16 GGUF blocks take. A tile holds 8 entries of 4 scale bytes (row e's f16 scale, then row e + 8's), then 32 lanes'
//! 16 bytes: lane 4 g + t holds, for sub-blocks 0 and 1 (values 0-15, 16-31), values 4 t .. 4 t + 3 of row g and then
//! of row g + 8, so a lane reads both mma.m16n8k16 A fragments of the tile with one 16-byte load. Tiles run along K
//! within a 16-row group, groups along the rows, matrices after one another.

use rayon::prelude::*;

pub(crate) const TILE_ROWS: usize = 16;
const HALF_ROWS: usize = TILE_ROWS / 2;
pub(crate) const BLOCK_VALUES: usize = 32;
const BLOCK_BYTES: usize = 34;
const SCALE: usize = 2;
const SCALE_BYTES: usize = HALF_ROWS * 2 * SCALE;
pub(crate) const TILE_BYTES: usize = TILE_ROWS * BLOCK_BYTES;
const LANES: usize = 32;
const LANE_BYTES: usize = (TILE_BYTES - SCALE_BYTES) / LANES;
const QUAD: usize = 4;
const WORD_VALUES: usize = 4;
const SUB_VALUES: usize = 16;
const SUBS: usize = BLOCK_VALUES / SUB_VALUES;

/// Whether `rows` x `cols` Q8_0 matrices regroup into whole tiles.
pub(crate) fn tiles(rows: usize, cols: usize) -> bool {
    rows.is_multiple_of(TILE_ROWS) && cols.is_multiple_of(BLOCK_VALUES)
}

// the byte of a tile holding lane (g, t)'s sub-block `s` value 4 t + j of row g (`upper` 0) or g + 8 (`upper` 1)
fn lane_byte(g: usize, t: usize, s: usize, upper: usize, j: usize) -> usize {
    SCALE_BYTES + (QUAD * g + t) * LANE_BYTES + (2 * s + upper) * WORD_VALUES + j
}

fn tile_from_blocks(blocks: [&[u8]; TILE_ROWS], tile: &mut [u8]) {
    for (r, block) in blocks.iter().enumerate() {
        let (e, upper) = (r % HALF_ROWS, r / HALF_ROWS);
        let at = e * 2 * SCALE + upper * SCALE;
        tile[at..at + SCALE].copy_from_slice(&block[..SCALE]);
        for t in 0..QUAD {
            for s in 0..SUBS {
                for j in 0..WORD_VALUES {
                    tile[lane_byte(e, t, s, upper, j)] =
                        block[SCALE + SUB_VALUES * s + WORD_VALUES * t + j];
                }
            }
        }
    }
}

fn block_from_tile(tile: &[u8], r: usize, block: &mut [u8]) {
    let (e, upper) = (r % HALF_ROWS, r / HALF_ROWS);
    let at = e * 2 * SCALE + upper * SCALE;
    block[..SCALE].copy_from_slice(&tile[at..at + SCALE]);
    for t in 0..QUAD {
        for s in 0..SUBS {
            for j in 0..WORD_VALUES {
                block[SCALE + SUB_VALUES * s + WORD_VALUES * t + j] =
                    tile[lane_byte(e, t, s, upper, j)];
            }
        }
    }
}

/// Row-major Q8_0 matrices (`matrices` of `rows` x `cols`, back to back) in the tile order; `tiles` must hold.
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
    fn q8_0_tiles_round_trip_to_the_gguf_blocks() {
        let (matrices, rows, cols) = (2, 32, 96);
        let bytes = pattern(matrices * rows * cols / BLOCK_VALUES * BLOCK_BYTES);
        let tiled = to_tiled(&bytes, matrices, rows, cols);
        assert_ne!(tiled, bytes);
        assert_eq!(from_tiled(&tiled, matrices, rows, cols), bytes);
    }

    #[test]
    fn a_q8_0_lane_holds_both_rows_values_of_each_sub_block() {
        let bytes = pattern(TILE_ROWS * BLOCK_BYTES);
        let tiled = to_tiled(&bytes, 1, TILE_ROWS, BLOCK_VALUES);
        let block = |r: usize| &bytes[r * BLOCK_BYTES..(r + 1) * BLOCK_BYTES];
        for (g, t, s, j) in [(0, 0, 0, 0), (3, 2, 1, 3), (7, 3, 1, 1)] {
            let v = SCALE + SUB_VALUES * s + WORD_VALUES * t + j;
            assert_eq!(tiled[lane_byte(g, t, s, 0, j)], block(g)[v]);
            assert_eq!(tiled[lane_byte(g, t, s, 1, j)], block(g + HALF_ROWS)[v]);
        }
        assert_eq!(SCALE_BYTES + LANES * LANE_BYTES, TILE_BYTES);
        assert_eq!(LANE_BYTES, SUBS * 2 * WORD_VALUES);
    }
}
