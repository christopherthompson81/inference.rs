//! The quantized paged KV cache's row format: per 32-value group, `bits` bits a value and one f16 scale.

/// Values sharing one scale and one Hadamard rotation.
pub const GROUP: usize = 32;
pub const MIN_BITS: u32 = 2;
pub const MAX_BITS: u32 = 8;
// a row's head dim must make its byte length a multiple of 4, which the attention loader addresses rows in
pub const HEAD_DIM_MULTIPLE: usize = 64;
const SCALE_BYTES: usize = 2;

/// Bytes one head's row of `head_dim` values takes at `bits` bits.
pub fn row_bytes(head_dim: usize, bits: u32) -> usize {
    head_dim * bits as usize / 8 + head_dim / GROUP * SCALE_BYTES
}

/// The bit width of a quantized row of `row_bytes` bytes holding `head_dim` values, if it is one.
pub fn bits_of(row_bytes: usize, head_dim: usize) -> Option<u32> {
    if head_dim == 0 || !head_dim.is_multiple_of(HEAD_DIM_MULTIPLE) {
        return None;
    }
    let data = row_bytes.checked_sub(head_dim / GROUP * SCALE_BYTES)?;
    let bits = u32::try_from(data * 8 / head_dim).ok()?;
    ((MIN_BITS..=MAX_BITS).contains(&bits) && row_bytes == self::row_bytes(head_dim, bits))
        .then_some(bits)
}

// One group per rotation: with a scale per group of 32, a wider rotation spreads an outlier channel into every group's
// scale, which cost a dense model 15% perplexity at 4-bit K and 5x at 3-bit (kv_cache_quant_investigation.md, Run 6)
const ROTATION: usize = GROUP;

/// The Hadamard rotation width of K rows, which Q is rotated by too.
pub fn k_rotation(_head_dim: usize) -> usize {
    ROTATION
}

/// The Hadamard rotation width of V rows, which the attention output is rotated back by.
pub fn v_rotation(_head_dim: usize) -> usize {
    ROTATION
}

/// Average bytes a value takes at `bits` bits, scales included.
pub fn bytes_per_value(bits: u32) -> f64 {
    (2 * bits + 1) as f64 / 16.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_lengths_round_trip_to_their_bit_width() {
        for head_dim in [64, 128, 256, 512] {
            for bits in MIN_BITS..=MAX_BITS {
                let bytes = row_bytes(head_dim, bits);
                assert_eq!(bytes % 4, 0, "{head_dim} at {bits} bits");
                assert_eq!(bits_of(bytes, head_dim), Some(bits));
                assert_eq!(bytes as f64, bytes_per_value(bits) * head_dim as f64);
            }
        }
        assert_eq!(bits_of(row_bytes(96, 4), 96), None);
        assert_eq!(bits_of(128 * 2, 128), None);
    }
}
