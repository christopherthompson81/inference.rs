//! NVFP4 regrouped for the tensor-core kernels: every 16 rows' 64-value blocks as one 576-byte tile, the bytes the
//! same 16 GGUF blocks take. A tile holds 8 entries of 8 scale bytes (row e's four UE4M3 scales, then row e + 8's),
//! then 32 lanes' 16 bytes: lane 4 g + t's word `is` holds sub-block `is` values 4 t .. 4 t + 3, row g in the low
//! nibbles and row g + 8 in the high ones, so a lane reads an mma.m16n8k16 A fragment (both row halves) and its rows'
//! scales with two loads and decodes it with one table lookup. Tiles run along K within a 16-row group, groups along
//! the rows, experts after one another (`[experts, rows / 16, cols / 64]` tiles).

use rayon::prelude::*;

pub(crate) const TILE_ROWS: usize = 16;
const HALF_ROWS: usize = TILE_ROWS / 2;
pub(crate) const BLOCK_VALUES: usize = 64;
const BLOCK_BYTES: usize = 36;
const SCALES: usize = 4;
const SCALE_BYTES: usize = TILE_ROWS * SCALES;
pub(crate) const TILE_BYTES: usize = TILE_ROWS * BLOCK_BYTES;
const LANES: usize = 32;
const LANE_BYTES: usize = (TILE_BYTES - SCALE_BYTES) / LANES;
const QUAD: usize = 4;
const WORD_VALUES: usize = 4;

/// Whether `rows` x `cols` NVFP4 matrices regroup into whole tiles.
pub(crate) fn tiles(rows: usize, cols: usize) -> bool {
    rows.is_multiple_of(TILE_ROWS) && cols.is_multiple_of(BLOCK_VALUES)
}

// value `v` of sub-block `s` in a GGUF block: byte 8 s + v % 8 of its nibbles, the high nibble from v = 8 on
fn gguf_nibble(block: &[u8], s: usize, v: usize) -> u8 {
    let byte = block[SCALES + 8 * s + v % 8];
    if v < 8 { byte & 0x0F } else { byte >> 4 }
}

fn set_gguf_nibble(block: &mut [u8], s: usize, v: usize, nibble: u8) {
    let byte = &mut block[SCALES + 8 * s + v % 8];
    if v < 8 {
        *byte = (*byte & 0xF0) | nibble;
    } else {
        *byte = (*byte & 0x0F) | (nibble << 4);
    }
}

// the byte of a tile holding lane (g, t)'s sub-block `s` values 4 t + j of rows g and g + 8
fn lane_byte(g: usize, t: usize, s: usize, j: usize) -> usize {
    SCALE_BYTES + (QUAD * g + t) * LANE_BYTES + s * WORD_VALUES + j
}

// one tile from 16 rows' GGUF blocks (each `BLOCK_BYTES`), and back
fn tile_from_blocks(blocks: [&[u8]; TILE_ROWS], tile: &mut [u8]) {
    for (r, block) in blocks.iter().enumerate() {
        let (e, upper) = (r % HALF_ROWS, r / HALF_ROWS);
        let at = e * 2 * SCALES + upper * SCALES;
        tile[at..at + SCALES].copy_from_slice(&block[..SCALES]);
    }
    for g in 0..HALF_ROWS {
        for t in 0..QUAD {
            for s in 0..SCALES {
                for j in 0..WORD_VALUES {
                    let v = WORD_VALUES * t + j;
                    let low = gguf_nibble(blocks[g], s, v);
                    let high = gguf_nibble(blocks[g + HALF_ROWS], s, v);
                    tile[lane_byte(g, t, s, j)] = low | (high << 4);
                }
            }
        }
    }
}

fn blocks_from_tile(tile: &[u8], r: usize, block: &mut [u8]) {
    let (g, upper) = (r % HALF_ROWS, r / HALF_ROWS);
    let at = g * 2 * SCALES + upper * SCALES;
    block[..SCALES].copy_from_slice(&tile[at..at + SCALES]);
    for t in 0..QUAD {
        for s in 0..SCALES {
            for j in 0..WORD_VALUES {
                let byte = tile[lane_byte(g, t, s, j)];
                let nibble = if upper == 0 { byte & 0x0F } else { byte >> 4 };
                set_gguf_nibble(block, s, WORD_VALUES * t + j, nibble);
            }
        }
    }
}

/// Row-major NVFP4 matrices (`matrices` of `rows` x `cols`, back to back) in the tile order; `tiles` must hold.
pub(crate) fn to_tiled(bytes: &[u8], matrices: usize, rows: usize, cols: usize) -> Vec<u8> {
    let nb = cols / BLOCK_VALUES;
    let row_bytes = nb * BLOCK_BYTES;
    let group_bytes = nb * TILE_BYTES;
    debug_assert!(tiles(rows, cols) && bytes.len() == matrices * rows * row_bytes);
    let mut out = vec![0u8; bytes.len()];
    out.par_chunks_exact_mut(group_bytes)
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
    let group_bytes = nb * TILE_BYTES;
    debug_assert!(tiles(rows, cols) && tiled.len() == matrices * rows * row_bytes);
    let mut out = vec![0u8; tiled.len()];
    out.par_chunks_exact_mut(TILE_ROWS * row_bytes)
        .zip(tiled.par_chunks_exact(group_bytes))
        .for_each(|(rows, group)| {
            for (r, row) in rows.chunks_exact_mut(row_bytes).enumerate() {
                for (kb, block) in row.as_chunks_mut::<BLOCK_BYTES>().0.iter_mut().enumerate() {
                    blocks_from_tile(&group[kb * TILE_BYTES..][..TILE_BYTES], r, block);
                }
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // a fixed byte pattern that touches every nibble value in every position
    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 131 + i / 7) as u8).collect()
    }

    #[test]
    fn tiles_round_trip_to_the_gguf_blocks() {
        let (matrices, rows, cols) = (3, 32, 192);
        let bytes = pattern(matrices * rows * cols / BLOCK_VALUES * BLOCK_BYTES);
        let tiled = to_tiled(&bytes, matrices, rows, cols);
        assert_ne!(tiled, bytes);
        assert_eq!(from_tiled(&tiled, matrices, rows, cols), bytes);
    }

    #[test]
    fn a_lane_word_holds_two_rows_values_of_one_sub_block() {
        let (rows, cols) = (TILE_ROWS, BLOCK_VALUES);
        let bytes = pattern(rows * BLOCK_BYTES);
        let tiled = to_tiled(&bytes, 1, rows, cols);
        let block = |r: usize| &bytes[r * BLOCK_BYTES..(r + 1) * BLOCK_BYTES];
        for (g, t, s, j) in [(0, 0, 0, 0), (3, 2, 1, 3), (7, 3, 3, 1)] {
            let byte = tiled[lane_byte(g, t, s, j)];
            let v = WORD_VALUES * t + j;
            assert_eq!(byte & 0x0F, gguf_nibble(block(g), s, v));
            assert_eq!(byte >> 4, gguf_nibble(block(g + HALF_ROWS), s, v));
        }
        assert_eq!(
            &tiled[5 * 2 * SCALES..5 * 2 * SCALES + SCALES],
            &block(5)[..SCALES]
        );
        assert_eq!(
            &tiled[5 * 2 * SCALES + SCALES..6 * 2 * SCALES],
            &block(13)[..SCALES]
        );
    }

    #[test]
    fn lanes_and_scales_fill_the_tile() {
        assert_eq!(SCALE_BYTES + LANES * LANE_BYTES, TILE_BYTES);
        assert_eq!(LANE_BYTES, SCALES * WORD_VALUES);
    }
}
