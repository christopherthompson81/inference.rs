pub const MODULE: &str = "inference_layout";
pub const IM2COL: &str = "im2col_cols_last_f32";
pub const MASK_TO_BOX: &str = "mask_to_box_f32";
/// Threads per `mask_to_box_f32` block; also sizes its shared reduction arrays.
pub const MASK_TO_BOX_BLOCK: u32 = 256;
pub const MS_DEFORM_ATTN: &str = "ms_deform_attn_f32";

/// All layout kernels as one compressed SASS fatbin, built ahead of time by `build.rs`.
pub static IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/layout.fatbin"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_to_box_block_matches_the_kernel_source() {
        let source = include_str!("../kernels/cuda/layout.cu");
        assert!(source.contains(&format!("#define MASK_TO_BOX_BLOCK {MASK_TO_BOX_BLOCK}\n")));
    }
}
