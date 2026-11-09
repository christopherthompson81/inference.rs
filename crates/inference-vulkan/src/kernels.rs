use std::sync::Arc;

use crate::map::{MAP_BINDINGS, MAP_PUSH_BYTES};
use crate::memory::{Bound, Buffer};
use crate::ops;
use crate::pipeline::{Pipeline, PipelineSpec};
use crate::{Error, Recorder, Result, shaders};

// values a block of the 32-value types holds
pub(crate) const BLOCK_VALUES: usize = 32;
// rows one matvec workgroup takes (the shader's `ROWS`), and the most workgroups one grid dimension holds
const MATVEC_ROWS: usize = 4;
// the matvecs' buffers and push constants, and an int8 block's bytes (8 packed words, then its (d, d * sum q) in the
// same buffer)
const MATVECS_BINDINGS: usize = 4;
const MATVECS_PUSH_BYTES: u32 = 7 * 4;
// the matvec's, plus the up weight and its offset and the GLU's op
const GLU_MATVECS_BINDINGS: usize = 5;
const GLU_MATVECS_PUSH_BYTES: u32 = 9 * 4;
const EMBED_BINDINGS: usize = 3;
const EMBED_PUSH_BYTES: u32 = 5 * 4;
const EMBED_THREADS: usize = 64;
pub(crate) const Q8_WORD_BYTES: usize = 32;
pub(crate) const Q8_SCALE_BYTES: usize = 8;
const QUANTIZE_BINDINGS: usize = 3;
const QUANTIZE_PUSH_BYTES: u32 = 3 * 4;
const QUANTIZE_THREADS: usize = 64;
const MAX_GROUPS_X: usize = 65_535;
// the most activation rows the matvec kernels take (their MAX_M); more take the tiled matmul, its tile's weight rows
// and tokens
pub const MATVEC_MAX_ROWS: usize = 8;
const MATMUL_Q_TILE_N: usize = 64;
const MATMUL_Q_TILE_M: usize = 64;
const MATMUL_Q_PUSH_BYTES: u32 = 6 * 4;

/// Every kernel's pipeline on one device: its field, its embedded SPIR-V, its storage buffers and push-constant bytes.
macro_rules! kernels {
    ($($name:ident: $spirv:ident, $bindings:expr, $push:expr;)*) => {
        pub(crate) struct Kernels {
            $(pub(crate) $name: Pipeline,)*
        }

        impl Kernels {
            pub(crate) fn new(device: &ash::Device, cache: ash::vk::PipelineCache) -> Result<Self> {
                let specs = [$(PipelineSpec::new(shaders::$spirv, $bindings, $push, stringify!($name)),)*];
                let mut built = Pipeline::build_all(device, cache, &specs)?.into_iter();
                Ok(Self {
                    $($name: built.next().expect("a pipeline a spec"),)*
                })
            }

            /// # Safety
            /// No submit may still use the pipelines.
            pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
                // SAFETY: the caller's contract
                unsafe {
                    $(self.$name.destroy(device);)*
                }
            }
        }
    };
}

kernels! {
    mat_vecs_q8_0: MUL_MAT_VECS_Q8_0, MATVECS_BINDINGS, MATVECS_PUSH_BYTES;
    embed_q8_0: EMBED_Q8_0, EMBED_BINDINGS, EMBED_PUSH_BYTES;
    embed_q5_0: EMBED_Q5_0, EMBED_BINDINGS, EMBED_PUSH_BYTES;
    embed_q4_0: EMBED_Q4_0, EMBED_BINDINGS, EMBED_PUSH_BYTES;
    embed_q4_1: EMBED_Q4_1, EMBED_BINDINGS, EMBED_PUSH_BYTES;
    embed_q5_1: EMBED_Q5_1, EMBED_BINDINGS, EMBED_PUSH_BYTES;
    embed_q4_k: EMBED_Q4_K, EMBED_BINDINGS, EMBED_PUSH_BYTES;
    embed_q5_k: EMBED_Q5_K, EMBED_BINDINGS, EMBED_PUSH_BYTES;
    embed_q6_k: EMBED_Q6_K, EMBED_BINDINGS, EMBED_PUSH_BYTES;
    embed_iq4_xs: EMBED_IQ4_XS, EMBED_BINDINGS, EMBED_PUSH_BYTES;
    embed_iq3_s: EMBED_IQ3_S, EMBED_BINDINGS, EMBED_PUSH_BYTES;
    embed_q3_k: EMBED_Q3_K, EMBED_BINDINGS, EMBED_PUSH_BYTES;
    mat_vecs_q5_0: MUL_MAT_VECS_Q5_0, MATVECS_BINDINGS, MATVECS_PUSH_BYTES;
    mat_vecs_q4_0: MUL_MAT_VECS_Q4_0, MATVECS_BINDINGS, MATVECS_PUSH_BYTES;
    mat_vecs_q4_1: MUL_MAT_VECS_Q4_1, MATVECS_BINDINGS, MATVECS_PUSH_BYTES;
    mat_vecs_q5_1: MUL_MAT_VECS_Q5_1, MATVECS_BINDINGS, MATVECS_PUSH_BYTES;
    mat_vecs_q4_k: MUL_MAT_VECS_Q4_K, MATVECS_BINDINGS, MATVECS_PUSH_BYTES;
    mat_vecs_q5_k: MUL_MAT_VECS_Q5_K, MATVECS_BINDINGS, MATVECS_PUSH_BYTES;
    mat_vecs_q6_k: MUL_MAT_VECS_Q6_K, MATVECS_BINDINGS, MATVECS_PUSH_BYTES;
    mat_vecs_iq4_xs: MUL_MAT_VECS_IQ4_XS, MATVECS_BINDINGS, MATVECS_PUSH_BYTES;
    mat_vecs_iq3_s: MUL_MAT_VECS_IQ3_S, MATVECS_BINDINGS, MATVECS_PUSH_BYTES;
    mat_vecs_q3_k: MUL_MAT_VECS_Q3_K, MATVECS_BINDINGS, MATVECS_PUSH_BYTES;
    glu_mat_vecs_q8_0: MUL_MAT_VECS_GLU_Q8_0, GLU_MATVECS_BINDINGS, GLU_MATVECS_PUSH_BYTES;
    glu_mat_vecs_q5_0: MUL_MAT_VECS_GLU_Q5_0, GLU_MATVECS_BINDINGS, GLU_MATVECS_PUSH_BYTES;
    glu_mat_vecs_q4_0: MUL_MAT_VECS_GLU_Q4_0, GLU_MATVECS_BINDINGS, GLU_MATVECS_PUSH_BYTES;
    glu_mat_vecs_q4_1: MUL_MAT_VECS_GLU_Q4_1, GLU_MATVECS_BINDINGS, GLU_MATVECS_PUSH_BYTES;
    glu_mat_vecs_q5_1: MUL_MAT_VECS_GLU_Q5_1, GLU_MATVECS_BINDINGS, GLU_MATVECS_PUSH_BYTES;
    glu_mat_vecs_q4_k: MUL_MAT_VECS_GLU_Q4_K, GLU_MATVECS_BINDINGS, GLU_MATVECS_PUSH_BYTES;
    glu_mat_vecs_q5_k: MUL_MAT_VECS_GLU_Q5_K, GLU_MATVECS_BINDINGS, GLU_MATVECS_PUSH_BYTES;
    glu_mat_vecs_q6_k: MUL_MAT_VECS_GLU_Q6_K, GLU_MATVECS_BINDINGS, GLU_MATVECS_PUSH_BYTES;
    glu_mat_vecs_iq4_xs: MUL_MAT_VECS_GLU_IQ4_XS, GLU_MATVECS_BINDINGS, GLU_MATVECS_PUSH_BYTES;
    glu_mat_vecs_iq3_s: MUL_MAT_VECS_GLU_IQ3_S, GLU_MATVECS_BINDINGS, GLU_MATVECS_PUSH_BYTES;
    glu_mat_vecs_q3_k: MUL_MAT_VECS_GLU_Q3_K, GLU_MATVECS_BINDINGS, GLU_MATVECS_PUSH_BYTES;
    quantize_q8_1: QUANTIZE_Q8_1, QUANTIZE_BINDINGS, QUANTIZE_PUSH_BYTES;
    widen_f16: WIDEN_F16, CONVERT_BINDINGS, CONVERT_PUSH_BYTES;
    convert_bf16: CONVERT_BF16, CONVERT_BINDINGS, CONVERT_BF16_PUSH_BYTES;
    mul_mm_q8_0: MUL_MM_Q8_0, MATVECS_BINDINGS, MATMUL_Q_PUSH_BYTES;
    mul_mm_q5_0: MUL_MM_Q5_0, MATVECS_BINDINGS, MATMUL_Q_PUSH_BYTES;
    mul_mm_q4_0: MUL_MM_Q4_0, MATVECS_BINDINGS, MATMUL_Q_PUSH_BYTES;
    mul_mm_q4_1: MUL_MM_Q4_1, MATVECS_BINDINGS, MATMUL_Q_PUSH_BYTES;
    mul_mm_q5_1: MUL_MM_Q5_1, MATVECS_BINDINGS, MATMUL_Q_PUSH_BYTES;
    mul_mm_q4_k: MUL_MM_Q4_K, MATVECS_BINDINGS, MATMUL_Q_PUSH_BYTES;
    mul_mm_q5_k: MUL_MM_Q5_K, MATVECS_BINDINGS, MATMUL_Q_PUSH_BYTES;
    mul_mm_q6_k: MUL_MM_Q6_K, MATVECS_BINDINGS, MATMUL_Q_PUSH_BYTES;
    mul_mm_iq4_xs: MUL_MM_IQ4_XS, MATVECS_BINDINGS, MATMUL_Q_PUSH_BYTES;
    mul_mm_iq3_s: MUL_MM_IQ3_S, MATVECS_BINDINGS, MATMUL_Q_PUSH_BYTES;
    mul_mm_q3_k: MUL_MM_Q3_K, MATVECS_BINDINGS, MATMUL_Q_PUSH_BYTES;
    map_f32: MAP_F32, MAP_BINDINGS, MAP_PUSH_BYTES;
    map_bf16: MAP_BF16, MAP_BINDINGS, MAP_PUSH_BYTES;
    copy_b32: COPY_B32, MAP_BINDINGS, MAP_PUSH_BYTES;
    copy_b16: COPY_B16, MAP_BINDINGS, MAP_PUSH_BYTES;
    rows_f32: ROWS_F32, ops::ROWS_BINDINGS, ops::ROWS_PUSH_BYTES;
    rows_bf16: ROWS_BF16, ops::ROWS_BINDINGS, ops::ROWS_PUSH_BYTES;
    rms_norm_q8: RMS_NORM_Q8, ops::RMS_Q8_BINDINGS, ops::RMS_Q8_PUSH_BYTES;
    relu2_q8: RELU2_Q8, crate::map::RELU2_Q8_BINDINGS, crate::map::RELU2_Q8_PUSH_BYTES;
    index_b32: INDEX_B32, ops::BINDINGS, ops::INDEX_PUSH_BYTES;
    matmul_f32: MATMUL_F32, ops::BINDINGS, ops::MATMUL_PUSH_BYTES;
    conv2d_f32: CONV2D_F32, ops::BINDINGS, ops::CONV2D_PUSH_BYTES;
    conv2d_bf16: CONV2D_BF16, ops::BINDINGS, ops::CONV2D_PUSH_BYTES;
    im2col_bf16: IM2COL_BF16, ops::IM2COL_BINDINGS, ops::CONV2D_PUSH_BYTES;
    matmul_f32_dot: MATMUL_F32_DOT, ops::BINDINGS, ops::MATMUL_PUSH_BYTES;
    matmul_bf16: MATMUL_BF16, ops::BINDINGS, ops::MATMUL_PUSH_BYTES;
    matmul_bf16_dot: MATMUL_BF16_DOT, ops::BINDINGS, ops::MATMUL_PUSH_BYTES;
    matmul_f32_tiled: MATMUL_F32_TILED, ops::BINDINGS, ops::MATMUL_PUSH_BYTES;
    matmul_f32_tiled32: MATMUL_F32_TILED32, ops::BINDINGS, ops::MATMUL_PUSH_BYTES;
    ssm_conv_step: SSM_CONV_STEP, ops::CONV_STEP_BINDINGS, ops::CONV_STEP_PUSH_BYTES;
    ssm_scan_step: SSM_SCAN_STEP, ops::SCAN_STEP_BINDINGS, ops::SCAN_STEP_PUSH_BYTES;
    gdn_recurrence: GDN_RECURRENCE, ops::GDN_BINDINGS, ops::GDN_PUSH_BYTES;
    rope: ROPE, ops::ROPE_BINDINGS, ops::ROPE_PUSH_BYTES;
    ssd_chunk_state: SSD_CHUNK_STATE, ops::SSD_BINDINGS, ops::SCAN_STEP_PUSH_BYTES;
    ssd_state_pass: SSD_STATE_PASS, ops::SSD_BINDINGS, ops::SCAN_STEP_PUSH_BYTES;
    ssd_chunk_out: SSD_CHUNK_OUT, ops::SSD_BINDINGS, ops::SCAN_STEP_PUSH_BYTES;
    moe_mat_vecs_q8_0: MOE_MAT_VECS_Q8_0, ops::MOE_MATVEC_BINDINGS, ops::MOE_MATVEC_PUSH_BYTES;
    moe_mat_vecs_q5_0: MOE_MAT_VECS_Q5_0, ops::MOE_MATVEC_BINDINGS, ops::MOE_MATVEC_PUSH_BYTES;
    moe_mat_vecs_q4_0: MOE_MAT_VECS_Q4_0, ops::MOE_MATVEC_BINDINGS, ops::MOE_MATVEC_PUSH_BYTES;
    moe_mat_vecs_q4_1: MOE_MAT_VECS_Q4_1, ops::MOE_MATVEC_BINDINGS, ops::MOE_MATVEC_PUSH_BYTES;
    moe_mat_vecs_q5_1: MOE_MAT_VECS_Q5_1, ops::MOE_MATVEC_BINDINGS, ops::MOE_MATVEC_PUSH_BYTES;
    moe_mat_vecs_q4_k: MOE_MAT_VECS_Q4_K, ops::MOE_MATVEC_BINDINGS, ops::MOE_MATVEC_PUSH_BYTES;
    moe_mat_vecs_q5_k: MOE_MAT_VECS_Q5_K, ops::MOE_MATVEC_BINDINGS, ops::MOE_MATVEC_PUSH_BYTES;
    moe_mat_vecs_q6_k: MOE_MAT_VECS_Q6_K, ops::MOE_MATVEC_BINDINGS, ops::MOE_MATVEC_PUSH_BYTES;
    moe_mat_vecs_iq4_xs: MOE_MAT_VECS_IQ4_XS, ops::MOE_MATVEC_BINDINGS, ops::MOE_MATVEC_PUSH_BYTES;
    moe_mat_vecs_iq3_s: MOE_MAT_VECS_IQ3_S, ops::MOE_MATVEC_BINDINGS, ops::MOE_MATVEC_PUSH_BYTES;
    moe_mat_vecs_q3_k: MOE_MAT_VECS_Q3_K, ops::MOE_MATVEC_BINDINGS, ops::MOE_MATVEC_PUSH_BYTES;
    moe_sort: MOE_SORT, ops::MOE_SORT_BINDINGS, ops::MOE_SORT_PUSH_BYTES;
    moe_mm_q8_0: MOE_MM_Q8_0, ops::MOE_MM_BINDINGS, ops::MOE_MM_PUSH_BYTES;
    moe_mm_q5_0: MOE_MM_Q5_0, ops::MOE_MM_BINDINGS, ops::MOE_MM_PUSH_BYTES;
    moe_mm_q4_0: MOE_MM_Q4_0, ops::MOE_MM_BINDINGS, ops::MOE_MM_PUSH_BYTES;
    moe_mm_q4_1: MOE_MM_Q4_1, ops::MOE_MM_BINDINGS, ops::MOE_MM_PUSH_BYTES;
    moe_mm_q5_1: MOE_MM_Q5_1, ops::MOE_MM_BINDINGS, ops::MOE_MM_PUSH_BYTES;
    moe_mm_q4_k: MOE_MM_Q4_K, ops::MOE_MM_BINDINGS, ops::MOE_MM_PUSH_BYTES;
    moe_mm_q5_k: MOE_MM_Q5_K, ops::MOE_MM_BINDINGS, ops::MOE_MM_PUSH_BYTES;
    moe_mm_q6_k: MOE_MM_Q6_K, ops::MOE_MM_BINDINGS, ops::MOE_MM_PUSH_BYTES;
    moe_mm_iq4_xs: MOE_MM_IQ4_XS, ops::MOE_MM_BINDINGS, ops::MOE_MM_PUSH_BYTES;
    moe_mm_iq3_s: MOE_MM_IQ3_S, ops::MOE_MM_BINDINGS, ops::MOE_MM_PUSH_BYTES;
    moe_mm_q3_k: MOE_MM_Q3_K, ops::MOE_MM_BINDINGS, ops::MOE_MM_PUSH_BYTES;
    moe_router: MOE_ROUTER, ops::ROUTER_BINDINGS, ops::ROUTER_PUSH_BYTES;
    paged_cache_write: PAGED_CACHE_WRITE, ops::PAGED_WRITE_BINDINGS, ops::PAGED_WRITE_PUSH_BYTES;
    paged_cache_write_f16: PAGED_CACHE_WRITE_F16, ops::PAGED_WRITE_BINDINGS, ops::PAGED_WRITE_PUSH_BYTES;
    paged_decode: PAGED_DECODE, ops::PAGED_DECODE_BINDINGS, ops::PAGED_DECODE_PUSH_BYTES;
    paged_decode_f16: PAGED_DECODE_F16, ops::PAGED_DECODE_BINDINGS, ops::PAGED_DECODE_PUSH_BYTES;
    paged_decode_reduce: PAGED_DECODE_REDUCE, ops::PAGED_REDUCE_BINDINGS, ops::PAGED_REDUCE_PUSH_BYTES;
    flash_attn: FLASH_ATTN, ops::FLASH_BINDINGS, ops::FLASH_PUSH_BYTES;
    flash_decode: FLASH_DECODE, ops::FLASH_BINDINGS, ops::FLASH_PUSH_BYTES;
    gated_rmsnorm: GATED_RMSNORM, ops::GATED_RMSNORM_BINDINGS, ops::GATED_RMSNORM_PUSH_BYTES;
    gated_rmsnorm_q8: GATED_RMSNORM_Q8, ops::GATED_Q8_BINDINGS, ops::GATED_Q8_PUSH_BYTES;
    paged_prefill: PAGED_PREFILL, ops::PAGED_PREFILL_BINDINGS, ops::PAGED_PREFILL_PUSH_BYTES;
    paged_prefill_f16: PAGED_PREFILL_F16, ops::PAGED_PREFILL_BINDINGS, ops::PAGED_PREFILL_PUSH_BYTES;
    ranked_topk: RANKED_TOPK, ops::RANKED_BINDINGS, ops::RANKED_PUSH_BYTES;
    ranked_topk_part: RANKED_TOPK_PART, ops::RANKED_BINDINGS, ops::RANKED_PART_PUSH_BYTES;
    ranked_draw: RANKED_DRAW, ops::RANKED_DRAW_BINDINGS, ops::RANKED_DRAW_PUSH_BYTES;
    penalize: PENALIZE, ops::PENALIZE_BINDINGS, ops::PENALIZE_PUSH_BYTES;
    paged_prefill_quant: PAGED_PREFILL_QUANT, ops::PAGED_PREFILL_BINDINGS, ops::PAGED_PREFILL_PUSH_BYTES;
    paged_prefill_h256: PAGED_PREFILL_H256, ops::PAGED_PREFILL_BINDINGS, ops::PAGED_PREFILL_PUSH_BYTES;
    paged_prefill_h256_f16: PAGED_PREFILL_H256_F16, ops::PAGED_PREFILL_BINDINGS, ops::PAGED_PREFILL_PUSH_BYTES;
    paged_prefill_h256_quant: PAGED_PREFILL_H256_QUANT, ops::PAGED_PREFILL_BINDINGS, ops::PAGED_PREFILL_PUSH_BYTES;
    paged_decode_quant: PAGED_DECODE_QUANT, ops::PAGED_DECODE_BINDINGS, ops::PAGED_DECODE_PUSH_BYTES;
    paged_decode_h256: PAGED_DECODE_H256, ops::PAGED_DECODE_BINDINGS, ops::PAGED_DECODE_PUSH_BYTES;
    paged_decode_h256_f16: PAGED_DECODE_H256_F16, ops::PAGED_DECODE_BINDINGS, ops::PAGED_DECODE_PUSH_BYTES;
    paged_decode_h256_quant: PAGED_DECODE_H256_QUANT, ops::PAGED_DECODE_BINDINGS, ops::PAGED_DECODE_PUSH_BYTES;
    paged_cache_write_quant: PAGED_CACHE_WRITE_QUANT, ops::PAGED_WRITE_BINDINGS, ops::PAGED_WRITE_QUANT_PUSH_BYTES;
    hadamard_32: HADAMARD_32, ops::HADAMARD_BINDINGS, ops::HADAMARD_PUSH_BYTES;
}

impl Kernels {
    /// The paged-cache kernels for a cache of `shape`'s element type.
    pub(crate) fn paged_cache_write(&self, shape: ops::CacheShape) -> &Pipeline {
        if shape.f16 {
            &self.paged_cache_write_f16
        } else {
            &self.paged_cache_write
        }
    }

    pub(crate) fn paged_decode(&self, shape: ops::CacheShape) -> &Pipeline {
        let wide = shape.head_size > ops::MAX_PAGED_NARROW_HEAD;
        match (shape.quant.is_some(), shape.f16, wide) {
            (true, _, false) => &self.paged_decode_quant,
            (true, _, true) => &self.paged_decode_h256_quant,
            (false, true, false) => &self.paged_decode_f16,
            (false, true, true) => &self.paged_decode_h256_f16,
            (false, false, false) => &self.paged_decode,
            (false, false, true) => &self.paged_decode_h256,
        }
    }

    pub(crate) fn paged_prefill(&self, shape: ops::CacheShape) -> &Pipeline {
        let wide = shape.head_size > ops::MAX_PAGED_NARROW_HEAD;
        match (shape.quant.is_some(), shape.f16, wide) {
            (true, _, false) => &self.paged_prefill_quant,
            (true, _, true) => &self.paged_prefill_h256_quant,
            (false, true, false) => &self.paged_prefill_f16,
            (false, true, true) => &self.paged_prefill_h256_f16,
            (false, false, false) => &self.paged_prefill,
            (false, false, true) => &self.paged_prefill_h256,
        }
    }

    /// The decode matvec over a gate and an up weight of `ty`'s blocks, ending in their GLU.
    pub(crate) fn glu_mat_vecs(&self, ty: BlockType) -> &Pipeline {
        match ty {
            BlockType::Q8_0 => &self.glu_mat_vecs_q8_0,
            BlockType::Q5_0 => &self.glu_mat_vecs_q5_0,
            BlockType::Q4_0 => &self.glu_mat_vecs_q4_0,
            BlockType::Q4_1 => &self.glu_mat_vecs_q4_1,
            BlockType::Q5_1 => &self.glu_mat_vecs_q5_1,
            BlockType::Q4K => &self.glu_mat_vecs_q4_k,
            BlockType::Q5K => &self.glu_mat_vecs_q5_k,
            BlockType::Q6K => &self.glu_mat_vecs_q6_k,
            BlockType::IQ4XS => &self.glu_mat_vecs_iq4_xs,
            BlockType::IQ3S => &self.glu_mat_vecs_iq3_s,
            BlockType::Q3K => &self.glu_mat_vecs_q3_k,
        }
    }

    /// The expert matvec over `ty`'s blocks.
    pub(crate) fn moe_mat_vecs(&self, ty: BlockType) -> &Pipeline {
        match ty {
            BlockType::Q8_0 => &self.moe_mat_vecs_q8_0,
            BlockType::Q5_0 => &self.moe_mat_vecs_q5_0,
            BlockType::Q4_0 => &self.moe_mat_vecs_q4_0,
            BlockType::Q4_1 => &self.moe_mat_vecs_q4_1,
            BlockType::Q5_1 => &self.moe_mat_vecs_q5_1,
            BlockType::Q4K => &self.moe_mat_vecs_q4_k,
            BlockType::Q5K => &self.moe_mat_vecs_q5_k,
            BlockType::Q6K => &self.moe_mat_vecs_q6_k,
            BlockType::IQ4XS => &self.moe_mat_vecs_iq4_xs,
            BlockType::IQ3S => &self.moe_mat_vecs_iq3_s,
            BlockType::Q3K => &self.moe_mat_vecs_q3_k,
        }
    }

    /// The prompt's expert matmul (routes sorted by expert, rows streamed) over `ty`'s blocks.
    pub(crate) fn moe_mm(&self, ty: BlockType) -> &Pipeline {
        match ty {
            BlockType::Q8_0 => &self.moe_mm_q8_0,
            BlockType::Q5_0 => &self.moe_mm_q5_0,
            BlockType::Q4_0 => &self.moe_mm_q4_0,
            BlockType::Q4_1 => &self.moe_mm_q4_1,
            BlockType::Q5_1 => &self.moe_mm_q5_1,
            BlockType::Q4K => &self.moe_mm_q4_k,
            BlockType::Q5K => &self.moe_mm_q5_k,
            BlockType::Q6K => &self.moe_mm_q6_k,
            BlockType::IQ4XS => &self.moe_mm_iq4_xs,
            BlockType::IQ3S => &self.moe_mm_iq3_s,
            BlockType::Q3K => &self.moe_mm_q3_k,
        }
    }
}

/// The matrix-core prompt matmuls (`mul_mmc_*`), on a device with VK_KHR_cooperative_matrix's f16 16 x 16 x 16
/// shape: weights, activations (f32) and outputs bound, and their push constants.
const MMC_BINDINGS: usize = 3;
const MMC_PUSH_BYTES: u32 = 6 * 4;
// a matrix-core workgroup's tile: weight rows by tokens
const MMC_TILE_N: usize = 64;
pub(crate) const MMC_TILE_M: usize = 64;
// mul_mmcl's tile, for matrices of MMCL_MIN_ROWS rows or more (on a 3090 at 512 tokens, 4096 and 10304 rows ran 26
// and 11 % faster than in 64 x 64 tiles, 2688 and 3712 rows 13-25 % slower), and the tokens the scratch pads to
// (whole tiles of either)
const MMCL_TILE: usize = 128;
const MMCL_MIN_ROWS: usize = 4096;
const MMC_PAD_M: usize = 128;
// the halves mul_mmcl loads a token's row by, so a row starts whole
const MMCL_ALIGN: usize = 8;

pub(crate) struct CoopKernels {
    // by BlockType
    mul_mmc: Vec<Pipeline>,
    // by BlockType: 128 x 128 tiles
    mul_mmcl: Vec<Pipeline>,
    pub(crate) convert_f16: Pipeline,
    // by BlockType
    moe_mmc: Vec<Pipeline>,
    pub(crate) moe_gather_f16: Pipeline,
    // prompt attention over f32, f16 and quantized paged caches
    paged_prefill: [Pipeline; 3],
    // prompt attention over contiguous f32 keys and values
    pub(crate) flash_attn: Pipeline,
    // matmul_f32_tiled's product, its tiles in f16
    pub(crate) matmul_f32: Pipeline,
    // the same over an f16 b
    pub(crate) matmul_f32_bh: Pipeline,
    pub(crate) matmul_bf16_cm: Pipeline,
    // prompt attention with a relative-position term
    pub(crate) flash_attn_relpos: Pipeline,
    // Q4_0 experts on NV_cooperative_matrix2, where the device has it: a build a tile width, 16 to 128 routes
    // by BlockType, where the type has decode functions
    moe_mm_cm2: Vec<Option<[Pipeline; 4]>>,
    // by BlockType, the same: the prompt matmul on NV_cooperative_matrix2
    mul_mm_cm2: Vec<Option<Pipeline>>,
}
// BlockType's variants
const BLOCK_TYPES: usize = 11;
// the cm2 kernels' tile (weight rows, and tokens of the prompt matmul), and their K step: a row a whole number of them
const CM2_TILE: usize = 128;
pub(crate) const CM2_K_STEP: usize = 64;
pub(crate) const MOE_CM2_BINDINGS: usize = 4;
pub(crate) const MOE_CM2_PUSH_BYTES: u32 = 8 * 4;
pub(crate) const MOE_MMC_BINDINGS: usize = 4;
pub(crate) const MOE_MMC_PUSH_BYTES: u32 = 7 * 4;
pub(crate) const MOE_GATHER_BINDINGS: usize = 3;
pub(crate) const MOE_GATHER_PUSH_BYTES: u32 = 4 * 4;
const CONVERT_BINDINGS: usize = 2;
const CONVERT_PUSH_BYTES: u32 = 3 * 4;
// count, the two bases, and the direction
const CONVERT_BF16_PUSH_BYTES: u32 = 4 * 4;
pub(crate) const CONVERT_THREADS: usize = 256;

impl CoopKernels {
    pub(crate) fn new(
        device: &ash::Device,
        cache: ash::vk::PipelineCache,
        subgroup: Option<u32>,
        cm2: bool,
    ) -> Result<Self> {
        let mul_mmc = [
            (shaders::MUL_MMC_Q8_0, "mul_mmc_q8_0"),
            (shaders::MUL_MMC_Q5_0, "mul_mmc_q5_0"),
            (shaders::MUL_MMC_Q4_0, "mul_mmc_q4_0"),
            (shaders::MUL_MMC_Q4_1, "mul_mmc_q4_1"),
            (shaders::MUL_MMC_Q5_1, "mul_mmc_q5_1"),
            (shaders::MUL_MMC_Q4_K, "mul_mmc_q4_k"),
            (shaders::MUL_MMC_Q5_K, "mul_mmc_q5_k"),
            (shaders::MUL_MMC_Q6_K, "mul_mmc_q6_k"),
            (shaders::MUL_MMC_IQ4_XS, "mul_mmc_iq4_xs"),
            (shaders::MUL_MMC_IQ3_S, "mul_mmc_iq3_s"),
            (shaders::MUL_MMC_Q3_K, "mul_mmc_q3_k"),
        ]
        .map(|(spirv, name)| {
            PipelineSpec::new(spirv, MMC_BINDINGS, MMC_PUSH_BYTES, name).subgroup(subgroup)
        });
        let mul_mmcl = [
            (shaders::MUL_MMCL_Q8_0, "mul_mmcl_q8_0"),
            (shaders::MUL_MMCL_Q5_0, "mul_mmcl_q5_0"),
            (shaders::MUL_MMCL_Q4_0, "mul_mmcl_q4_0"),
            (shaders::MUL_MMCL_Q4_1, "mul_mmcl_q4_1"),
            (shaders::MUL_MMCL_Q5_1, "mul_mmcl_q5_1"),
            (shaders::MUL_MMCL_Q4_K, "mul_mmcl_q4_k"),
            (shaders::MUL_MMCL_Q5_K, "mul_mmcl_q5_k"),
            (shaders::MUL_MMCL_Q6_K, "mul_mmcl_q6_k"),
            (shaders::MUL_MMCL_IQ4_XS, "mul_mmcl_iq4_xs"),
            (shaders::MUL_MMCL_IQ3_S, "mul_mmcl_iq3_s"),
            (shaders::MUL_MMCL_Q3_K, "mul_mmcl_q3_k"),
        ]
        .map(|(spirv, name)| {
            PipelineSpec::new(spirv, MMC_BINDINGS, MMC_PUSH_BYTES, name).subgroup(subgroup)
        });
        let moe_mmc = [
            (shaders::MOE_MMC_Q8_0, "moe_mmc_q8_0"),
            (shaders::MOE_MMC_Q5_0, "moe_mmc_q5_0"),
            (shaders::MOE_MMC_Q4_0, "moe_mmc_q4_0"),
            (shaders::MOE_MMC_Q4_1, "moe_mmc_q4_1"),
            (shaders::MOE_MMC_Q5_1, "moe_mmc_q5_1"),
            (shaders::MOE_MMC_Q4_K, "moe_mmc_q4_k"),
            (shaders::MOE_MMC_Q5_K, "moe_mmc_q5_k"),
            (shaders::MOE_MMC_Q6_K, "moe_mmc_q6_k"),
            (shaders::MOE_MMC_IQ4_XS, "moe_mmc_iq4_xs"),
            (shaders::MOE_MMC_IQ3_S, "moe_mmc_iq3_s"),
            (shaders::MOE_MMC_Q3_K, "moe_mmc_q3_k"),
        ]
        .map(|(spirv, name)| {
            PipelineSpec::new(spirv, MOE_MMC_BINDINGS, MOE_MMC_PUSH_BYTES, name).subgroup(subgroup)
        });
        let prefill = [
            (shaders::PAGED_PREFILL_CM, "paged_prefill_cm"),
            (shaders::PAGED_PREFILL_CM_F16, "paged_prefill_cm_f16"),
            (shaders::PAGED_PREFILL_CM_QUANT, "paged_prefill_cm_quant"),
        ]
        .map(|(spirv, name)| {
            PipelineSpec::new(
                spirv,
                ops::PAGED_PREFILL_BINDINGS,
                ops::PAGED_PREFILL_PUSH_BYTES,
                name,
            )
            .subgroup(subgroup)
        });
        let cm2_spec =
            |(spirv, name)| PipelineSpec::new(spirv, MOE_CM2_BINDINGS, MOE_CM2_PUSH_BYTES, name);
        let mul_mm_cm2_types = [
            (BlockType::Q8_0, shaders::MUL_MM_CM2_Q8_0, "mul_mm_cm2_q8_0"),
            (BlockType::Q5_0, shaders::MUL_MM_CM2_Q5_0, "mul_mm_cm2_q5_0"),
            (BlockType::Q4_0, shaders::MUL_MM_CM2_Q4_0, "mul_mm_cm2_q4_0"),
            (BlockType::Q4_1, shaders::MUL_MM_CM2_Q4_1, "mul_mm_cm2_q4_1"),
            (BlockType::Q5_1, shaders::MUL_MM_CM2_Q5_1, "mul_mm_cm2_q5_1"),
        ];
        let moe_mm_cm2_types = [
            (
                BlockType::Q8_0,
                [
                    (shaders::MOE_MM_CM2_Q8_0_N16, "moe_mm_cm2_q8_0_n16"),
                    (shaders::MOE_MM_CM2_Q8_0_N32, "moe_mm_cm2_q8_0_n32"),
                    (shaders::MOE_MM_CM2_Q8_0_N64, "moe_mm_cm2_q8_0_n64"),
                    (shaders::MOE_MM_CM2_Q8_0_N128, "moe_mm_cm2_q8_0_n128"),
                ],
            ),
            (
                BlockType::Q5_0,
                [
                    (shaders::MOE_MM_CM2_Q5_0_N16, "moe_mm_cm2_q5_0_n16"),
                    (shaders::MOE_MM_CM2_Q5_0_N32, "moe_mm_cm2_q5_0_n32"),
                    (shaders::MOE_MM_CM2_Q5_0_N64, "moe_mm_cm2_q5_0_n64"),
                    (shaders::MOE_MM_CM2_Q5_0_N128, "moe_mm_cm2_q5_0_n128"),
                ],
            ),
            (
                BlockType::Q4_0,
                [
                    (shaders::MOE_MM_CM2_Q4_0_N16, "moe_mm_cm2_q4_0_n16"),
                    (shaders::MOE_MM_CM2_Q4_0_N32, "moe_mm_cm2_q4_0_n32"),
                    (shaders::MOE_MM_CM2_Q4_0_N64, "moe_mm_cm2_q4_0_n64"),
                    (shaders::MOE_MM_CM2_Q4_0_N128, "moe_mm_cm2_q4_0_n128"),
                ],
            ),
            (
                BlockType::Q4_1,
                [
                    (shaders::MOE_MM_CM2_Q4_1_N16, "moe_mm_cm2_q4_1_n16"),
                    (shaders::MOE_MM_CM2_Q4_1_N32, "moe_mm_cm2_q4_1_n32"),
                    (shaders::MOE_MM_CM2_Q4_1_N64, "moe_mm_cm2_q4_1_n64"),
                    (shaders::MOE_MM_CM2_Q4_1_N128, "moe_mm_cm2_q4_1_n128"),
                ],
            ),
            (
                BlockType::Q5_1,
                [
                    (shaders::MOE_MM_CM2_Q5_1_N16, "moe_mm_cm2_q5_1_n16"),
                    (shaders::MOE_MM_CM2_Q5_1_N32, "moe_mm_cm2_q5_1_n32"),
                    (shaders::MOE_MM_CM2_Q5_1_N64, "moe_mm_cm2_q5_1_n64"),
                    (shaders::MOE_MM_CM2_Q5_1_N128, "moe_mm_cm2_q5_1_n128"),
                ],
            ),
        ];
        let mut specs = Vec::new();
        specs.extend(mul_mmc);
        specs.extend(mul_mmcl);
        specs.extend(moe_mmc);
        specs.extend(prefill);
        specs.push(
            PipelineSpec::new(
                shaders::FLASH_ATTN_CM,
                ops::FLASH_BINDINGS,
                ops::FLASH_PUSH_BYTES,
                "flash_attn_cm",
            )
            .subgroup(subgroup),
        );
        specs.push(
            PipelineSpec::new(
                shaders::MATMUL_F32_CM,
                ops::BINDINGS,
                ops::MATMUL_PUSH_BYTES,
                "matmul_f32_cm",
            )
            .subgroup(subgroup),
        );
        specs.push(
            PipelineSpec::new(
                shaders::MATMUL_F32_CM_BH,
                ops::BINDINGS,
                ops::MATMUL_PUSH_BYTES,
                "matmul_f32_cm_bh",
            )
            .subgroup(subgroup),
        );
        specs.push(
            PipelineSpec::new(
                shaders::MATMUL_BF16_CM,
                ops::BINDINGS,
                ops::MATMUL_PUSH_BYTES,
                "matmul_bf16_cm",
            )
            .subgroup(subgroup),
        );
        specs.push(
            PipelineSpec::new(
                shaders::FLASH_ATTN_RELPOS_CM,
                ops::RELPOS_BINDINGS,
                ops::RELPOS_PUSH_BYTES,
                "flash_attn_relpos_cm",
            )
            .subgroup(subgroup),
        );
        specs.push(PipelineSpec::new(
            shaders::CONVERT_F16,
            CONVERT_BINDINGS,
            CONVERT_PUSH_BYTES,
            "convert_f16",
        ));
        specs.push(PipelineSpec::new(
            shaders::MOE_GATHER_F16,
            MOE_GATHER_BINDINGS,
            MOE_GATHER_PUSH_BYTES,
            "moe_gather_f16",
        ));
        if cm2 {
            specs.extend(mul_mm_cm2_types.map(|(_, spirv, name)| cm2_spec((spirv, name))));
            specs.extend(
                moe_mm_cm2_types
                    .iter()
                    .flat_map(|(_, builds)| builds.map(cm2_spec)),
            );
        }
        let mut built = Pipeline::build_all(device, cache, &specs)?.into_iter();
        let mut take = |n: usize| built.by_ref().take(n).collect::<Vec<_>>();
        let mul_mmc = take(BLOCK_TYPES);
        let mul_mmcl = take(BLOCK_TYPES);
        let moe_mmc = take(BLOCK_TYPES);
        let paged_prefill: [Pipeline; 3] =
            take(3).try_into().ok().expect("three prefill pipelines");
        let [flash_attn]: [Pipeline; 1] = take(1).try_into().ok().expect("one attention pipeline");
        let [matmul_f32, matmul_f32_bh, matmul_bf16_cm, flash_attn_relpos]: [Pipeline; 4] =
            take(4).try_into().ok().expect("four pipelines");
        let [convert_f16, moe_gather_f16]: [Pipeline; 2] =
            take(2).try_into().ok().expect("two pipelines");
        let mut moe_mm_cm2: Vec<Option<[Pipeline; 4]>> = (0..BLOCK_TYPES).map(|_| None).collect();
        let mut mul_mm_cm2: Vec<Option<Pipeline>> = (0..BLOCK_TYPES).map(|_| None).collect();
        if cm2 {
            for ((ty, _, _), pipeline) in mul_mm_cm2_types
                .into_iter()
                .zip(take(mul_mm_cm2_types.len()))
            {
                mul_mm_cm2[ty as usize] = Some(pipeline);
            }
            for (ty, _) in moe_mm_cm2_types {
                moe_mm_cm2[ty as usize] = Some(take(4).try_into().ok().expect("four tile widths"));
            }
        }
        Ok(Self {
            mul_mmc,
            mul_mmcl,
            convert_f16,
            moe_mmc,
            moe_gather_f16,
            paged_prefill,
            flash_attn,
            matmul_f32,
            matmul_f32_bh,
            matmul_bf16_cm,
            flash_attn_relpos,
            moe_mm_cm2,
            mul_mm_cm2,
        })
    }

    /// Prompt attention over `shape`'s cache on matrix cores, where this device runs it.
    pub(crate) fn paged_prefill(&self, shape: ops::CacheShape) -> &Pipeline {
        let [f32, f16, quant] = &self.paged_prefill;
        if shape.quant.is_some() {
            quant
        } else if shape.f16 {
            f16
        } else {
            f32
        }
    }

    fn mul_mmc(&self, ty: BlockType) -> &Pipeline {
        &self.mul_mmc[ty as usize]
    }

    fn mul_mmcl(&self, ty: BlockType) -> &Pipeline {
        &self.mul_mmcl[ty as usize]
    }

    pub(crate) fn moe_mmc(&self, ty: BlockType) -> &Pipeline {
        &self.moe_mmc[ty as usize]
    }

    /// The expert matmul on NV_cooperative_matrix2 over `ty`'s blocks, where the device and type have one: a build
    /// each for the tiles of up to 16, 32, 64 and 128 routes.
    pub(crate) fn moe_mm_cm2(&self, ty: BlockType) -> Option<&[Pipeline; 4]> {
        self.moe_mm_cm2[ty as usize].as_ref()
    }

    /// # Safety
    /// No submit may still use the pipelines.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        for pipeline in self
            .mul_mmc
            .iter()
            .chain(&self.mul_mmcl)
            .chain(&self.moe_mmc)
            .chain([&self.convert_f16, &self.moe_gather_f16, &self.flash_attn])
            .chain(&self.paged_prefill)
            .chain(self.moe_mm_cm2.iter().flatten().flatten())
            .chain(self.mul_mm_cm2.iter().flatten())
        {
            // SAFETY: the caller's contract
            unsafe { pipeline.destroy(device) };
        }
    }
}

/// A GGUF weight's block format the matvec kernels read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlockType {
    /// an f16 scale and 32 int8 values
    Q8_0,
    /// an f16 scale, 32 fifth bits and 32 low nibbles, values offset by 16
    Q5_0,
    /// an f16 scale and 32 nibbles, values offset by 8
    Q4_0,
    /// f16 scale and min, 32 nibbles
    Q4_1,
    /// f16 scale and min, 32 fifth bits and 32 low nibbles
    Q5_1,
    /// 256 values in eight 32-value sub-blocks of nibbles, each with a 6-bit scale and min
    Q4K,
    /// Q4K with a fifth bit a value
    Q5K,
    /// 256 six-bit values, an int8 scale per 16
    Q6K,
    /// 256 values as indices into a 16-entry nonlinear table, a 6-bit scale per 32
    IQ4XS,
    /// 256 values from a 512-entry grid of four, with signs, a 4-bit scale per 32
    IQ3S,
    /// 256 three-bit values (a high bit apart), a 6-bit scale per 16
    Q3K,
}

impl BlockType {
    /// The bytes of one stored block.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Q8_0 => 34,
            Self::Q5_0 => 22,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_1 => 24,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::IQ4XS => 136,
            Self::IQ3S => 110,
            Self::Q3K => 110,
        }
    }

    /// The values one stored block holds; a row's length is a multiple of it.
    pub fn block_values(self) -> usize {
        match self {
            Self::Q4K | Self::Q5K | Self::Q6K | Self::IQ4XS | Self::IQ3S | Self::Q3K => 256,
            _ => BLOCK_VALUES,
        }
    }
}

/// `y[y_offset + m * rows..][..rows] = W x[x_offset + m * cols..][..cols]` for each of `x_rows` activation rows, for a
/// `rows x cols` weight of `ty` blocks held in `weights`: up to [`MATVEC_MAX_ROWS`] rows a matvec reading the weight
/// once, more a tiled matmul. The rows are quantized to int8 blocks (as the CPU's quantized matmuls do) into
/// `scratch`, of [`MatVec::scratch_bytes`].
pub struct MatVec<'a> {
    pub ty: BlockType,
    pub weights: &'a Arc<dyn Bound>,
    pub rows: usize,
    pub cols: usize,
    pub x: &'a Arc<Buffer>,
    pub x_offset: usize,
    pub y: &'a Arc<Buffer>,
    pub y_offset: usize,
    pub x_rows: usize,
    pub scratch: &'a Arc<Buffer>,
}

impl MatVec<'_> {
    /// The scratch bytes `x_rows` rows of `cols` take.
    pub fn scratch_bytes(x_rows: usize, cols: usize) -> usize {
        let int8 = x_rows * cols / BLOCK_VALUES * (Q8_WORD_BYTES + Q8_SCALE_BYTES);
        // a prompt on the matrix cores takes its tokens as f16, padded to whole tiles
        let f16 = if x_rows > MATVEC_MAX_ROWS {
            x_rows.next_multiple_of(MMC_PAD_M) * cols * 2
        } else {
            0
        };
        int8.max(f16).next_multiple_of(4)
    }
}

/// A decode GLU: `y[m] = act(G x[m]) * (U x[m])` for `x_rows` (at most [`MATVEC_MAX_ROWS`]) activation rows, `G`
/// and `U` both `rows x cols` weights of `ty` blocks, in one matvec; `act` is a GLU's [`MapOp`](crate::MapOp).
pub struct GluMatVec<'a> {
    pub ty: BlockType,
    pub gate: &'a Arc<dyn Bound>,
    pub up: &'a Arc<dyn Bound>,
    pub act: crate::MapOp,
    pub rows: usize,
    pub cols: usize,
    pub x: &'a Arc<Buffer>,
    pub x_offset: usize,
    pub y: &'a Arc<Buffer>,
    pub x_rows: usize,
    pub scratch: &'a Arc<Buffer>,
}

/// The rows of a `vocab x cols` matrix of `ty` blocks in `weights` at the u32 `ids` (`count` of them), dequantized to
/// `count x cols` floats in `y`.
pub struct Embed<'a> {
    pub ty: BlockType,
    pub weights: &'a Arc<dyn Bound>,
    pub vocab: usize,
    pub cols: usize,
    pub ids: (&'a Arc<Buffer>, usize),
    pub count: usize,
    pub y: &'a Arc<Buffer>,
}

fn push_words(values: &[usize]) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &v in values {
        let v = u32::try_from(v)
            .map_err(|_| Error::Invalid(format!("matvec constant {v} past u32")))?;
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Ok(bytes)
}

impl Recorder<'_> {
    // `rows` f32 rows of `cols` from `x` as int8 blocks in `scratch`, giving where their scales start (in vec2s)
    pub(crate) fn quantize_rows(
        &self,
        x: crate::At<'_>,
        rows: usize,
        cols: usize,
        scratch: &Arc<Buffer>,
    ) -> Result<usize> {
        let x_blocks = rows * cols / BLOCK_VALUES;
        if scratch.len() < MatVec::scratch_bytes(rows, cols) || x.0.len() / 4 < x.1 + rows * cols {
            return Err(Error::Invalid(format!(
                "{rows} rows of {cols} quantized into {} bytes",
                scratch.len()
            )));
        }
        let s_off = x_blocks * Q8_WORD_BYTES / Q8_SCALE_BYTES;
        self.context.kernels.quantize_q8_1.dispatch(
            self,
            &[x.0.buffer, scratch.buffer, scratch.buffer],
            &push_words(&[x_blocks, x.1, s_off])?,
            [x_blocks.div_ceil(QUANTIZE_THREADS) as u32, 1, 1],
        );
        self.retain(x.0.clone());
        self.retain(scratch.clone());
        Ok(s_off)
    }

    /// A decode activation's int8 blocks: the cached ones when nothing has written `x` or them since they were made,
    /// else quantized into `scratch` and cached. The buffer to bind for them, and where their scales start.
    pub(crate) fn quantized(
        &self,
        x: crate::At<'_>,
        rows: usize,
        cols: usize,
        scratch: &Arc<Buffer>,
    ) -> Result<(Arc<Buffer>, usize)> {
        if let Some(hit) = self.context.find_quantized(x.0, x.1, rows * cols) {
            self.retain(hit.0.clone());
            return Ok(hit);
        }
        let s_off = self.quantize_rows(x, rows, cols, scratch)?;
        self.context
            .remember_quantized(x.0, x.1, rows * cols, scratch, s_off);
        Ok((scratch.clone(), s_off))
    }

    pub fn embed(&self, op: &Embed<'_>) -> Result<()> {
        let blocks = op.cols / BLOCK_VALUES;
        let fits = op.vocab > 0
            && op.cols.is_multiple_of(op.ty.block_values())
            && op.weights.offset().is_multiple_of(2)
            && op.weights.bytes()
                >= op.vocab * op.cols / op.ty.block_values() * op.ty.block_bytes()
            && op.y.len() / 4 >= op.count * op.cols;
        if !fits {
            return Err(Error::Invalid(format!(
                "an embedding of {} ids from a {} x {} {:?} matrix",
                op.count, op.vocab, op.cols, op.ty
            )));
        }
        if op.count == 0 {
            return Ok(());
        }
        let kernels = &self.context.kernels;
        let pipeline = match op.ty {
            BlockType::Q8_0 => &kernels.embed_q8_0,
            BlockType::Q5_0 => &kernels.embed_q5_0,
            BlockType::Q4_0 => &kernels.embed_q4_0,
            BlockType::Q4_1 => &kernels.embed_q4_1,
            BlockType::Q5_1 => &kernels.embed_q5_1,
            BlockType::Q4K => &kernels.embed_q4_k,
            BlockType::Q5K => &kernels.embed_q5_k,
            BlockType::Q6K => &kernels.embed_q6_k,
            BlockType::IQ4XS => &kernels.embed_iq4_xs,
            BlockType::IQ3S => &kernels.embed_iq3_s,
            BlockType::Q3K => &kernels.embed_q3_k,
        };
        let bytes = push_words(&[
            op.vocab,
            blocks,
            op.weights.offset() as usize / 2,
            op.ids.1,
            op.count,
        ])?;
        pipeline.dispatch(
            self,
            &[op.weights.vk_buffer(), op.ids.0.buffer, op.y.buffer],
            &bytes,
            [(op.count * blocks).div_ceil(EMBED_THREADS) as u32, 1, 1],
        );
        self.retain(op.weights.clone() as crate::Retained);
        self.retain(op.ids.0.clone());
        self.retain(op.y.clone());
        Ok(())
    }

    /// `count` f16s at `src` as f32s into `dst` from its start.
    pub fn widen_f16(&self, src: crate::At<'_>, count: usize, dst: &Arc<Buffer>) -> Result<()> {
        if src.0.len() / 2 < src.1 + count || dst.len() / 4 < count {
            return Err(Error::Invalid(format!("a widening of {count} halves")));
        }
        if count == 0 {
            return Ok(());
        }
        let groups = count.div_ceil(CONVERT_THREADS);
        let groups_x = groups.min(MAX_GROUPS_X);
        self.context.kernels.widen_f16.dispatch(
            self,
            &[src.0.buffer, dst.buffer],
            &push_words(&[count, src.1, 0])?,
            [groups_x as u32, groups.div_ceil(groups_x) as u32, 1],
        );
        self.retain(src.0.clone());
        self.retain(dst.clone());
        Ok(())
    }

    /// `count` values at `src` into `dst` from its start: f32s as bf16s, or with `widen` bf16s as f32s.
    pub fn convert_bf16(
        &self,
        src: crate::At<'_>,
        count: usize,
        dst: &Arc<Buffer>,
        widen: bool,
    ) -> Result<()> {
        let (src_size, dst_size) = if widen { (2, 4) } else { (4, 2) };
        if src.0.len() / src_size < src.1 + count || dst.len() / dst_size < count {
            return Err(Error::Invalid(format!(
                "a bf16 conversion of {count} values"
            )));
        }
        if count == 0 {
            return Ok(());
        }
        let groups = count.div_ceil(CONVERT_THREADS);
        let groups_x = groups.min(MAX_GROUPS_X);
        self.context.kernels.convert_bf16.dispatch(
            self,
            &[src.0.buffer, dst.buffer],
            &push_words(&[count, src.1, 0, usize::from(widen)])?,
            [groups_x as u32, groups.div_ceil(groups_x) as u32, 1],
        );
        self.retain(src.0.clone());
        self.retain(dst.clone());
        Ok(())
    }

    /// `count` f32s at `src` as f16s into `dst` from its start, on a device whose matrix cores have the conversion
    /// kernel; whether it did.
    pub fn narrow_f16(&self, src: crate::At<'_>, count: usize, dst: &Arc<Buffer>) -> Result<bool> {
        let Some(coop) = self.context.coop.as_ref() else {
            return Ok(false);
        };
        // the kernel writes whole pairs
        if src.0.len() / 4 < src.1 + count || dst.len() < count.next_multiple_of(2) * 2 {
            return Err(Error::Invalid(format!("a narrowing of {count} floats")));
        }
        let pairs = count.div_ceil(2);
        let groups = pairs.div_ceil(CONVERT_THREADS);
        let groups_x = groups.min(MAX_GROUPS_X);
        coop.convert_f16.dispatch(
            self,
            &[src.0.buffer, dst.buffer],
            &push_words(&[count, src.1, 0])?,
            [groups_x as u32, groups.div_ceil(groups_x) as u32, 1],
        );
        self.retain(src.0.clone());
        self.retain(dst.clone());
        Ok(true)
    }

    pub fn mat_vec(&self, op: &MatVec<'_>) -> Result<()> {
        let fits = op.cols.is_multiple_of(op.ty.block_values())
            && op.weights.offset().is_multiple_of(2)
            && op.weights.bytes() >= op.rows * op.cols / op.ty.block_values() * op.ty.block_bytes()
            && op.x.len() / 4 >= op.x_offset + op.x_rows * op.cols
            && op.y.len() / 4 >= op.y_offset + op.x_rows * op.rows
            && op.x_rows > 0;
        if !fits {
            return Err(Error::Invalid(format!(
                "a {}x{} {:?} matvec over {} weight bytes at offset {}",
                op.rows,
                op.cols,
                op.ty,
                op.weights.bytes(),
                op.weights.offset()
            )));
        }
        let scratch = op.scratch;
        if scratch.len() < MatVec::scratch_bytes(op.x_rows, op.cols) {
            return Err(Error::Invalid(format!(
                "a {}-row matvec's scratch of {} bytes",
                op.x_rows,
                scratch.len()
            )));
        }
        let kernels = &self.context.kernels;
        if op.x_rows > MATVEC_MAX_ROWS
            && let Some(coop) = self.context.matmul_coop()
        {
            // the tokens as f16 into the scratch, padded to whole tiles with zeros: a tile's last matrices read past
            // its live tokens (outputs dropped), and past a row's end into the next row (against zero weights)
            let count = op.x_rows * op.cols;
            let padded = (op.x_rows.next_multiple_of(MMC_PAD_M) * op.cols * 2) as u64;
            self.fill(scratch.buffer, padded.next_multiple_of(4), 0);
            let pairs = count.div_ceil(2).div_ceil(CONVERT_THREADS);
            let pairs_x = pairs.min(MAX_GROUPS_X);
            coop.convert_f16.dispatch(
                self,
                &[op.x.buffer, scratch.buffer],
                &push_words(&[count, op.x_offset, 0])?,
                [pairs_x as u32, pairs.div_ceil(pairs_x) as u32, 1],
            );
            let bytes = push_words(&[
                op.rows,
                op.cols,
                op.weights.offset() as usize / 2,
                0,
                op.y_offset,
                op.x_rows,
            ])?;
            self.profile_shape(
                || format!("m={} n={} k={}", op.rows, op.x_rows, op.cols),
                op.rows * op.cols * op.x_rows,
            );
            if let Some(cm2) = coop.mul_mm_cm2[op.ty as usize].as_ref()
                && op.cols.is_multiple_of(CM2_K_STEP)
            {
                cm2.dispatch(
                    self,
                    &[
                        op.weights.vk_buffer(),
                        scratch.buffer,
                        op.y.buffer,
                        scratch.buffer,
                    ],
                    &push_words(&[
                        op.rows,
                        op.cols,
                        op.weights.offset() as usize / 2,
                        0,
                        op.y_offset,
                        0,
                        op.x_rows,
                        op.x_rows.next_multiple_of(MMC_PAD_M),
                    ])?,
                    [
                        op.rows.div_ceil(CM2_TILE) as u32,
                        op.x_rows.div_ceil(CM2_TILE) as u32,
                        1,
                    ],
                );
                self.work.set(
                    self.work
                        .get()
                        .saturating_add((op.rows * op.cols * op.x_rows) as u64),
                );
                self.retain(op.weights.clone() as crate::Retained);
                self.retain(op.x.clone());
                self.retain(op.y.clone());
                self.retain(scratch.clone());
                return Ok(());
            }
            let large = op.x_rows >= MMCL_TILE
                && op.rows >= MMCL_MIN_ROWS
                && op.cols.is_multiple_of(MMCL_ALIGN);
            let (pipeline, tile_n, tile_m) = if large {
                (coop.mul_mmcl(op.ty), MMCL_TILE, MMCL_TILE)
            } else {
                (coop.mul_mmc(op.ty), MMC_TILE_N, MMC_TILE_M)
            };
            pipeline.dispatch(
                self,
                &[op.weights.vk_buffer(), scratch.buffer, op.y.buffer],
                &bytes,
                [
                    op.rows.div_ceil(tile_n) as u32,
                    op.x_rows.div_ceil(tile_m) as u32,
                    1,
                ],
            );
            self.work.set(
                self.work
                    .get()
                    .saturating_add((op.rows * op.cols * op.x_rows) as u64),
            );
            self.retain(op.weights.clone() as crate::Retained);
            self.retain(op.x.clone());
            self.retain(op.y.clone());
            self.retain(scratch.clone());
            return Ok(());
        }
        let (blocks, s_off) = if op.x_rows <= MATVEC_MAX_ROWS {
            self.quantized((op.x, op.x_offset), op.x_rows, op.cols, scratch)?
        } else {
            let s_off = self.quantize_rows((op.x, op.x_offset), op.x_rows, op.cols, scratch)?;
            (scratch.clone(), s_off)
        };
        let scratch = &blocks;
        if op.x_rows > MATVEC_MAX_ROWS {
            let pipeline = match op.ty {
                BlockType::Q8_0 => &kernels.mul_mm_q8_0,
                BlockType::Q5_0 => &kernels.mul_mm_q5_0,
                BlockType::Q4_0 => &kernels.mul_mm_q4_0,
                BlockType::Q4_1 => &kernels.mul_mm_q4_1,
                BlockType::Q5_1 => &kernels.mul_mm_q5_1,
                BlockType::Q4K => &kernels.mul_mm_q4_k,
                BlockType::Q5K => &kernels.mul_mm_q5_k,
                BlockType::Q6K => &kernels.mul_mm_q6_k,
                BlockType::IQ4XS => &kernels.mul_mm_iq4_xs,
                BlockType::IQ3S => &kernels.mul_mm_iq3_s,
                BlockType::Q3K => &kernels.mul_mm_q3_k,
            };
            let bytes = push_words(&[
                op.rows,
                op.cols,
                op.weights.offset() as usize / 2,
                op.y_offset,
                op.x_rows,
                s_off,
            ])?;
            self.profile_shape(
                || format!("m={} n={} k={}", op.rows, op.x_rows, op.cols),
                op.rows * op.cols * op.x_rows,
            );
            pipeline.dispatch(
                self,
                &[
                    op.weights.vk_buffer(),
                    scratch.buffer,
                    op.y.buffer,
                    scratch.buffer,
                ],
                &bytes,
                [
                    op.rows.div_ceil(MATMUL_Q_TILE_N) as u32,
                    op.x_rows.div_ceil(MATMUL_Q_TILE_M) as u32,
                    1,
                ],
            );
            self.work.set(
                self.work
                    .get()
                    .saturating_add((op.rows * op.cols * op.x_rows) as u64),
            );
            self.retain(scratch.clone());
            self.retain(op.weights.clone() as crate::Retained);
            self.retain(op.y.clone());
            return Ok(());
        }
        let groups = op.rows.div_ceil(MATVEC_ROWS);
        let groups_x = groups.min(MAX_GROUPS_X);
        let bytes = push_words(&[
            op.rows,
            op.cols,
            op.weights.offset() as usize / 2,
            op.y_offset,
            groups_x,
            op.x_rows,
            s_off,
        ])?;
        let pipeline = match op.ty {
            BlockType::Q8_0 => &kernels.mat_vecs_q8_0,
            BlockType::Q5_0 => &kernels.mat_vecs_q5_0,
            BlockType::Q4_0 => &kernels.mat_vecs_q4_0,
            BlockType::Q4_1 => &kernels.mat_vecs_q4_1,
            BlockType::Q5_1 => &kernels.mat_vecs_q5_1,
            BlockType::Q4K => &kernels.mat_vecs_q4_k,
            BlockType::Q5K => &kernels.mat_vecs_q5_k,
            BlockType::Q6K => &kernels.mat_vecs_q6_k,
            BlockType::IQ4XS => &kernels.mat_vecs_iq4_xs,
            BlockType::IQ3S => &kernels.mat_vecs_iq3_s,
            BlockType::Q3K => &kernels.mat_vecs_q3_k,
        };
        self.profile_shape(
            || format!("m={} n={} k={}", op.rows, op.x_rows, op.cols),
            op.rows * op.cols * op.x_rows,
        );
        pipeline.dispatch(
            self,
            &[
                op.weights.vk_buffer(),
                scratch.buffer,
                op.y.buffer,
                scratch.buffer,
            ],
            &bytes,
            [groups_x as u32, groups.div_ceil(groups_x) as u32, 1],
        );
        self.retain(scratch.clone());
        self.retain(op.weights.clone() as crate::Retained);
        self.retain(op.x.clone());
        self.retain(op.y.clone());
        Ok(())
    }
    pub fn glu_mat_vec(&self, op: &GluMatVec<'_>) -> Result<()> {
        let weight_fits = |w: &Arc<dyn Bound>| {
            w.offset().is_multiple_of(2)
                && w.bytes() >= op.rows * op.cols / op.ty.block_values() * op.ty.block_bytes()
        };
        let fits = op.cols.is_multiple_of(op.ty.block_values())
            && weight_fits(op.gate)
            && weight_fits(op.up)
            && op.act.is_glu()
            && op.x.len() / 4 >= op.x_offset + op.x_rows * op.cols
            && op.y.len() / 4 >= op.x_rows * op.rows
            && (1..=MATVEC_MAX_ROWS).contains(&op.x_rows)
            && op.scratch.len() >= MatVec::scratch_bytes(op.x_rows, op.cols);
        if !fits {
            return Err(Error::Invalid(format!(
                "a {}x{} {:?} GLU matvec over {} rows",
                op.rows, op.cols, op.ty, op.x_rows
            )));
        }
        let (blocks, s_off) =
            self.quantized((op.x, op.x_offset), op.x_rows, op.cols, op.scratch)?;
        let groups = op.rows.div_ceil(MATVEC_ROWS);
        let groups_x = groups.min(MAX_GROUPS_X);
        let bytes = push_words(&[
            op.rows,
            op.cols,
            op.gate.offset() as usize / 2,
            0,
            groups_x,
            op.x_rows,
            s_off,
            op.up.offset() as usize / 2,
            op.act as usize,
        ])?;
        self.profile_shape(
            || format!("glu m={} n={} k={}", op.rows, op.x_rows, op.cols),
            2 * op.rows * op.cols * op.x_rows,
        );
        self.context.kernels.glu_mat_vecs(op.ty).dispatch(
            self,
            &[
                op.gate.vk_buffer(),
                blocks.buffer,
                op.y.buffer,
                blocks.buffer,
                op.up.vk_buffer(),
            ],
            &bytes,
            [groups_x as u32, groups.div_ceil(groups_x) as u32, 1],
        );
        self.retain(blocks);
        self.retain(op.gate.clone() as crate::Retained);
        self.retain(op.up.clone() as crate::Retained);
        self.retain(op.x.clone());
        self.retain(op.y.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Context, ImportedMemory, f32_bytes};
    use std::alloc::{Layout, alloc_zeroed, dealloc};

    const PAGE: usize = 4096;
    const TOLERANCE: f64 = 1e-4;

    // page-aligned host memory, so an import of it covers no one else's pages
    struct Pages {
        ptr: *mut u8,
        layout: Layout,
    }

    impl Pages {
        fn new(len: usize) -> Self {
            let layout = Layout::from_size_align(len.next_multiple_of(PAGE), PAGE).unwrap();
            // SAFETY: the layout is nonzero and page aligned
            let ptr = unsafe { alloc_zeroed(layout) };
            assert!(!ptr.is_null());
            Self { ptr, layout }
        }

        fn bytes(&mut self) -> &mut [u8] {
            // SAFETY: the allocation is `layout.size()` bytes and owned by `self`
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.layout.size()) }
        }
    }

    impl Drop for Pages {
        fn drop(&mut self) {
            // SAFETY: allocated in `new` with this layout
            unsafe { dealloc(self.ptr, self.layout) };
        }
    }

    fn context() -> Option<Arc<Context>> {
        match Context::new() {
            Ok(context) => Some(context),
            Err(err) => {
                eprintln!("skipping: {err}");
                None
            }
        }
    }

    fn len(ty: BlockType, rows: usize, cols: usize) -> usize {
        rows * cols / BLOCK_VALUES * ty.block_bytes()
    }

    // random blocks of `ty` with small finite scales
    fn weights(ty: BlockType, bytes: &mut [u8], rows: usize, cols: usize) {
        let mut state = 0x2545_f491u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let size = ty.block_bytes();
        for block in bytes[..len(ty, rows, cols)].chunks_exact_mut(size) {
            let d = half::f16::from_f32(0.001 + (next() % 100) as f32 * 1e-5);
            block[..2].copy_from_slice(&d.to_le_bytes());
            block[2..].iter_mut().for_each(|q| *q = next() as u8);
        }
    }

    // value `j` of a block, as ggml's dequantize_row reads it
    fn value(ty: BlockType, block: &[u8], j: usize) -> f64 {
        match ty {
            BlockType::Q8_0 => f64::from(block[2 + j] as i8),
            BlockType::Q5_0 => {
                let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
                let byte = block[6 + j % 16];
                let nibble = if j < 16 { byte & 0xf } else { byte >> 4 };
                f64::from(i32::from(nibble) | (((qh >> j) & 1) as i32) << 4) - 16.0
            }
            other => unreachable!(
                "{other:?} is checked against the CPU's dequantization in inference-tensor"
            ),
        }
    }

    fn reference(ty: BlockType, bytes: &[u8], rows: usize, cols: usize, x: &[f32]) -> Vec<f64> {
        let (blocks, size) = (cols / BLOCK_VALUES, ty.block_bytes());
        (0..rows)
            .map(|r| {
                (0..blocks)
                    .map(|b| {
                        let block = &bytes[(r * blocks + b) * size..][..size];
                        let d = f64::from(half::f16::from_le_bytes([block[0], block[1]]).to_f32());
                        let dot: f64 = (0..BLOCK_VALUES)
                            .map(|j| value(ty, block, j) * f64::from(x[b * BLOCK_VALUES + j]))
                            .sum();
                        d * dot
                    })
                    .sum()
            })
            .collect()
    }

    // matvecs (and past 8 rows the tiled matmul) take int8 activations: each output within the rounding bound of its
    // the SPIR-V scan's writable bindings: quantize writes its blocks and scales, a matvec its output alone
    #[test]
    fn pipelines_know_which_bindings_they_write() {
        let Some(context) = context() else {
            return;
        };
        assert_eq!(context.kernels.quantize_q8_1.writes, 0b110);
        assert_eq!(context.kernels.mat_vecs_q8_0.writes, 0b100);
    }

    // a decode row's RMS norm writes its int8 blocks for the next matvec: that matvec matches one over the same values
    // quantized apart (copied to a buffer of their own)
    #[test]
    fn a_norms_blocks_match_quantizing_its_output() -> Result<()> {
        let Some(context) = context() else {
            return Ok(());
        };
        const ROWS: usize = 96;
        // a decode row in one pass, and rows of two passes with a partial second
        for (x_rows, cols) in [(1usize, 2688usize), (3, 4128)] {
            let ty = BlockType::Q8_0;
            let len = len(ty, ROWS, cols);
            let mut pages = Pages::new(len);
            weights(ty, pages.bytes(), ROWS, cols);
            let weights: Arc<dyn Bound> =
                Arc::new(Buffer::with_bytes(&context, &pages.bytes()[..len])?);
            let x: Vec<f32> = (0..x_rows * cols)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) / (9.0 + (i / cols) as f32))
                .collect();
            let alpha: Vec<f32> = (0..cols).map(|i| 0.5 + (i % 11) as f32 / 10.0).collect();
            let (xs, alphas) = (
                Arc::new(Buffer::with_f32(&context, &x)?),
                Arc::new(Buffer::with_f32(&context, &alpha)?),
            );
            let normed = Arc::new(Buffer::new(&context, x_rows * cols * 4)?);
            let (cached_y, apart_y) = (
                Arc::new(Buffer::new(&context, x_rows * ROWS * 4)?),
                Arc::new(Buffer::new(&context, x_rows * ROWS * 4)?),
            );
            let scratch = Arc::new(Buffer::new(&context, MatVec::scratch_bytes(x_rows, cols))?);
            let op = |x, y| MatVec {
                ty,
                weights: &weights,
                rows: ROWS,
                cols,
                x,
                x_offset: 0,
                y,
                y_offset: 0,
                x_rows,
                scratch: &scratch,
            };
            context.submit(|r| {
                r.rows_f32(&crate::Rows {
                    op: crate::RowOp::RmsNorm,
                    rows: x_rows,
                    cols,
                    eps: 1e-5,
                    x: &xs,
                    x_offset: 0,
                    alpha: Some((&alphas, 0)),
                    beta: None,
                    out: &normed,
                    out_offset: 0,
                })?;
                r.mat_vec(&op(&normed, &cached_y))
            })?;
            let values = normed.read_f32()?;
            for (r, row) in x.chunks(cols).enumerate() {
                let rms = (row.iter().map(|v| v * v).sum::<f32>() / cols as f32 + 1e-5).sqrt();
                for (c, (&v, &a)) in row.iter().zip(&alpha).enumerate() {
                    let want = v / rms * a;
                    let got = values[r * cols + c];
                    assert!(
                        (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                        "{r},{c}: {got} vs {want}"
                    );
                }
            }
            let apart = Arc::new(Buffer::with_f32(&context, &values)?);
            context.submit(|r| r.mat_vec(&op(&apart, &apart_y)))?;
            assert_eq!(
                cached_y.read_f32()?,
                apart_y.read_f32()?,
                "{x_rows} x {cols}"
            );
        }
        Ok(())
    }

    // the same for Mamba-2's gated RMS norm (a row of 8 groups of 512, Lightning's)
    #[test]
    fn a_gated_norms_blocks_match_quantizing_its_output() -> Result<()> {
        let Some(context) = context() else {
            return Ok(());
        };
        const ROWS: usize = 96;
        const GROUPS: usize = 8;
        const GROUP: usize = 512;
        const COLS: usize = GROUPS * GROUP;
        let ty = BlockType::Q8_0;
        let len = len(ty, ROWS, COLS);
        let mut pages = Pages::new(len);
        weights(ty, pages.bytes(), ROWS, COLS);
        let weights: Arc<dyn Bound> =
            Arc::new(Buffer::with_bytes(&context, &pages.bytes()[..len])?);
        let wave = |k: usize| -> Vec<f32> {
            (0..COLS)
                .map(|i| ((i * k % 101) as f32 - 50.0) / 13.0)
                .collect()
        };
        let (xs, gate, scale) = (
            Arc::new(Buffer::with_f32(&context, &wave(37))?),
            Arc::new(Buffer::with_f32(&context, &wave(53))?),
            Arc::new(Buffer::with_f32(&context, &wave(7))?),
        );
        let normed = Arc::new(Buffer::new(&context, COLS * 4)?);
        let (cached_y, apart_y) = (
            Arc::new(Buffer::new(&context, ROWS * 4)?),
            Arc::new(Buffer::new(&context, ROWS * 4)?),
        );
        let scratch = Arc::new(Buffer::new(&context, MatVec::scratch_bytes(1, COLS))?);
        let op = |x, y| MatVec {
            ty,
            weights: &weights,
            rows: ROWS,
            cols: COLS,
            x,
            x_offset: 0,
            y,
            y_offset: 0,
            x_rows: 1,
            scratch: &scratch,
        };
        context.submit(|r| {
            r.gated_rmsnorm(&crate::GatedRmsNorm {
                rows: 1,
                groups: GROUPS,
                group_size: GROUP,
                eps: 1e-5,
                norm_first: false,
                sigmoid: false,
                x: (&xs, 0),
                gate: (&gate, 0),
                weight: (&scale, 0),
                out: &normed,
            })?;
            r.mat_vec(&op(&normed, &cached_y))
        })?;
        let apart = Arc::new(Buffer::with_f32(&context, &normed.read_f32()?)?);
        context.submit(|r| r.mat_vec(&op(&apart, &apart_y)))?;
        assert_eq!(cached_y.read_f32()?, apart_y.read_f32()?);
        Ok(())
    }

    // the same for relu^2 into a few rows (a decode step's routed experts' down-projection input)
    #[test]
    fn a_relu_squares_blocks_match_quantizing_its_output() -> Result<()> {
        let Some(context) = context() else {
            return Ok(());
        };
        const ROWS: usize = 96;
        const X_ROWS: usize = 6;
        const COLS: usize = 1856;
        let ty = BlockType::Q8_0;
        let len = len(ty, ROWS, COLS);
        let mut pages = Pages::new(len);
        weights(ty, pages.bytes(), ROWS, COLS);
        let weights: Arc<dyn Bound> =
            Arc::new(Buffer::with_bytes(&context, &pages.bytes()[..len])?);
        let a: Vec<f32> = (0..X_ROWS * COLS)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 17.0)
            .collect();
        let xs = Arc::new(Buffer::with_f32(&context, &a)?);
        let squared = Arc::new(Buffer::new(&context, X_ROWS * COLS * 4)?);
        let (cached_y, apart_y) = (
            Arc::new(Buffer::new(&context, X_ROWS * ROWS * 4)?),
            Arc::new(Buffer::new(&context, X_ROWS * ROWS * 4)?),
        );
        let scratch = Arc::new(Buffer::new(&context, MatVec::scratch_bytes(X_ROWS, COLS))?);
        let op = |x, y| MatVec {
            ty,
            weights: &weights,
            rows: ROWS,
            cols: COLS,
            x,
            x_offset: 0,
            y,
            y_offset: 0,
            x_rows: X_ROWS,
            scratch: &scratch,
        };
        let strides = [COLS, 1];
        context.submit(|r| {
            r.map_f32(&crate::map::Map {
                op: crate::map::MapOp::Relu2,
                dims: &[X_ROWS, COLS],
                a: crate::map::Operand {
                    buffer: &xs,
                    offset: 0,
                    strides: &strides,
                },
                b: None,
                out: crate::map::Operand {
                    buffer: &squared,
                    offset: 0,
                    strides: &strides,
                },
                params: [0.0; 2],
            })?;
            r.mat_vec(&op(&squared, &cached_y))
        })?;
        let values = squared.read_f32()?;
        let want: Vec<f32> = a.iter().map(|v| v.max(0.0) * v.max(0.0)).collect();
        assert_eq!(values, want);
        let apart = Arc::new(Buffer::with_f32(&context, &values)?);
        context.submit(|r| r.mat_vec(&op(&apart, &apart_y)))?;
        assert_eq!(cached_y.read_f32()?, apart_y.read_f32()?);
        Ok(())
    }

    // a decode activation's cached blocks serve a second matvec of it, and a write to it drops them: a fill of the
    // activations between two matvecs in one submit shows in the second's output
    #[test]
    fn quantized_activations_are_dropped_when_written() -> Result<()> {
        let Some(context) = context() else {
            return Ok(());
        };
        const ROWS: usize = 64;
        const COLS: usize = 256;
        const ONE: u32 = 0x3f80_0000;
        let ty = BlockType::Q8_0;
        let len = len(ty, ROWS, COLS);
        let mut pages = Pages::new(len);
        weights(ty, pages.bytes(), ROWS, COLS);
        let weights: Arc<dyn Bound> =
            Arc::new(Buffer::with_bytes(&context, &pages.bytes()[..len])?);
        let x: Vec<f32> = (0..COLS).map(|i| (i % 7) as f32 - 3.0).collect();
        let xs = Arc::new(Buffer::with_f32(&context, &x)?);
        let ys: Vec<_> = (0..3)
            .map(|_| Buffer::new(&context, ROWS * 4).map(Arc::new))
            .collect::<Result<_>>()?;
        let scratches: Vec<_> = (0..3)
            .map(|_| Buffer::new(&context, MatVec::scratch_bytes(1, COLS)).map(Arc::new))
            .collect::<Result<_>>()?;
        let op = |y, scratch| MatVec {
            ty,
            weights: &weights,
            rows: ROWS,
            cols: COLS,
            x: &xs,
            x_offset: 0,
            y,
            y_offset: 0,
            x_rows: 1,
            scratch,
        };
        context.submit(|r| {
            r.mat_vec(&op(&ys[0], &scratches[0]))?;
            r.mat_vec(&op(&ys[1], &scratches[1]))?;
            r.fill(xs.buffer, (COLS * 4) as u64, ONE);
            r.mat_vec(&op(&ys[2], &scratches[2]))
        })?;
        let (first, again, after) = (ys[0].read_f32()?, ys[1].read_f32()?, ys[2].read_f32()?);
        assert_eq!(first, again);
        // against an all-ones row: each output the row's weight sum
        let ones = Arc::new(Buffer::with_f32(&context, &vec![1.0; COLS])?);
        let want = Arc::new(Buffer::new(&context, ROWS * 4)?);
        let fresh = Arc::new(Buffer::new(&context, MatVec::scratch_bytes(1, COLS))?);
        context.submit(|r| {
            r.mat_vec(&MatVec {
                x: &ones,
                ..op(&want, &fresh)
            })
        })?;
        assert_eq!(after, want.read_f32()?);
        assert_ne!(after, first);
        Ok(())
    }

    // blocks (|w| times half a block's step, summed) of the f32 product, over weight rows and tokens past a
    // workgroup's and activation rows unaligned in their buffer
    #[test]
    fn matvecs_round_only_their_activations() -> Result<()> {
        let Some(context) = context() else {
            return Ok(());
        };
        const ROWS: usize = 333;
        const COLS: usize = 2688;
        const X_AT: usize = 3;
        for (ty, x_rows) in [
            (BlockType::Q8_0, 1),
            (BlockType::Q8_0, 3),
            (BlockType::Q8_0, 70),
            (BlockType::Q5_0, 1),
            (BlockType::Q5_0, 3),
            (BlockType::Q5_0, 70),
        ] {
            let len = len(ty, ROWS, COLS);
            let mut pages = Pages::new(len);
            weights(ty, pages.bytes(), ROWS, COLS);
            let x: Vec<f32> = (0..x_rows * COLS)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) / (25.0 + (i / COLS) as f32))
                .collect();
            let size = ty.block_bytes();
            let blocks = COLS / BLOCK_VALUES;
            let weights: Arc<dyn Bound> =
                Arc::new(Buffer::with_bytes(&context, &pages.bytes()[..len])?);
            let mut xs = Buffer::new(&context, (X_AT + x_rows * COLS) * 4)?;
            xs.write(X_AT * 4, f32_bytes(&x))?;
            let (xs, ys) = (
                Arc::new(xs),
                Arc::new(Buffer::new(&context, x_rows * ROWS * 4)?),
            );
            let scratch = Arc::new(Buffer::new(&context, MatVec::scratch_bytes(x_rows, COLS))?);
            context.submit(|r| {
                r.mat_vec(&MatVec {
                    ty,
                    weights: &weights,
                    rows: ROWS,
                    cols: COLS,
                    x: &xs,
                    x_offset: X_AT,
                    y: &ys,
                    y_offset: 0,
                    x_rows,
                    scratch: &scratch,
                })
            })?;
            let got = ys.read_f32()?;
            for m in 0..x_rows {
                let xm = &x[m * COLS..][..COLS];
                let want = reference(ty, pages.bytes(), ROWS, COLS, xm);
                for (row, &want) in want.iter().enumerate() {
                    let bound: f64 = (0..blocks)
                        .map(|b| {
                            let block = &pages.bytes()[(row * blocks + b) * size..][..size];
                            let d =
                                f64::from(half::f16::from_le_bytes([block[0], block[1]]).to_f32());
                            let xb = &xm[b * BLOCK_VALUES..][..BLOCK_VALUES];
                            let step =
                                f64::from(xb.iter().fold(0f32, |a, v| a.max(v.abs()))) / 127.0;
                            let w: f64 = (0..BLOCK_VALUES)
                                .map(|j| (d * value(ty, block, j)).abs())
                                .sum();
                            w * step / 2.0
                        })
                        .sum();
                    let got = f64::from(got[m * ROWS + row]);
                    assert!(
                        (got - want).abs() <= bound * 1.01 + 1e-5,
                        "{ty:?} x row {m} row {row}: {got} vs {want} (bound {bound})"
                    );
                }
            }
        }
        Ok(())
    }

    // the matrix-core prompt matmuls (64 x 64 and 128 x 128 tiles, both with partial tiles of rows and tokens) against
    // an f64 product on sampled rows, within the f16 rounding of the weights and the activations
    #[test]
    fn matrix_core_prompt_matmuls_match_their_reference() -> Result<()> {
        // an iGPU's matrix cores too, which the default context leaves off
        let Ok(context) = Context::with_matrix_cores(crate::MatrixCores::Any) else {
            return Ok(());
        };
        if context.coop.is_none() {
            return Ok(());
        }
        const COLS: usize = 2688;
        const ROW_STEP: usize = 41;
        // two halves of f16 rounding (the weight's and the activation's) on each product
        const F16_STEP: f64 = 1.0 / 1024.0;
        for (ty, rows, x_rows) in [
            (BlockType::Q8_0, 333usize, 70usize),
            (BlockType::Q8_0, 4100, 200),
            (BlockType::Q5_0, 4096, 130),
        ] {
            let len = len(ty, rows, COLS);
            let mut pages = Pages::new(len);
            weights(ty, pages.bytes(), rows, COLS);
            let x: Vec<f32> = (0..x_rows * COLS)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) / (25.0 + (i / COLS) as f32))
                .collect();
            let size = ty.block_bytes();
            let blocks = COLS / BLOCK_VALUES;
            let weights: Arc<dyn Bound> =
                Arc::new(Buffer::with_bytes(&context, &pages.bytes()[..len])?);
            let xs = Arc::new(Buffer::with_f32(&context, &x)?);
            let ys = Arc::new(Buffer::new(&context, x_rows * rows * 4)?);
            let scratch = Arc::new(Buffer::new(&context, MatVec::scratch_bytes(x_rows, COLS))?);
            context.submit(|r| {
                r.mat_vec(&MatVec {
                    ty,
                    weights: &weights,
                    rows,
                    cols: COLS,
                    x: &xs,
                    x_offset: 0,
                    y: &ys,
                    y_offset: 0,
                    x_rows,
                    scratch: &scratch,
                })
            })?;
            let got = ys.read_f32()?;
            let sampled = (0..rows).step_by(ROW_STEP).chain([rows - 1]);
            for row in sampled {
                for m in [0, x_rows / 2, x_rows - 1] {
                    let xm = &x[m * COLS..][..COLS];
                    let (mut want, mut mass) = (0f64, 0f64);
                    for b in 0..blocks {
                        let block = &pages.bytes()[(row * blocks + b) * size..][..size];
                        let d = f64::from(half::f16::from_le_bytes([block[0], block[1]]).to_f32());
                        for j in 0..BLOCK_VALUES {
                            let p = d * value(ty, block, j) * f64::from(xm[b * BLOCK_VALUES + j]);
                            want += p;
                            mass += p.abs();
                        }
                    }
                    let got = f64::from(got[m * rows + row]);
                    assert!(
                        (got - want).abs() <= mass * F16_STEP + 1e-4,
                        "{ty:?} {rows} x {x_rows}: token {m} row {row}: {got} vs {want}"
                    );
                }
            }
        }
        Ok(())
    }

    // more dispatches than one batch takes, with the caller's handles to the inputs dropped before the GPU finishes
    #[test]
    fn recorded_work_keeps_its_buffers_alive_across_batches() -> Result<()> {
        let Some(context) = context() else {
            return Ok(());
        };
        const ROWS: usize = 64;
        const COLS: usize = 256;
        const DISPATCHES: usize = 3 * crate::stream::DISPATCHES_PER_SUBMIT / 2;
        let ty = BlockType::Q8_0;
        let len = len(ty, ROWS, COLS);
        let mut pages = Pages::new(len);
        weights(ty, pages.bytes(), ROWS, COLS);
        // -1, 0 and 1, which the int8 activations hold exactly
        let x: Vec<f32> = (0..COLS).map(|i| (i % 3) as f32 - 1.0).collect();
        let want = reference(ty, pages.bytes(), ROWS, COLS, &x);
        let ys = Arc::new(Buffer::new(&context, DISPATCHES * ROWS * 4)?);
        let scratch = Arc::new(Buffer::new(&context, MatVec::scratch_bytes(1, COLS))?);
        {
            // SAFETY: `pages` outlives every submit, which `synchronize` below waits for
            let weights: Arc<dyn Bound> =
                Arc::new(unsafe { ImportedMemory::new(&context, &pages.bytes()[..len])? });
            let xs = Arc::new(Buffer::with_f32(&context, &x)?);
            for d in 0..DISPATCHES {
                context.record(|r| {
                    r.mat_vec(&MatVec {
                        ty,
                        weights: &weights,
                        rows: ROWS,
                        cols: COLS,
                        x: &xs,
                        x_offset: 0,
                        y: &ys,
                        y_offset: d * ROWS,
                        x_rows: 1,
                        scratch: &scratch,
                    })
                })?;
            }
        }
        context.synchronize()?;
        for (i, &got) in ys.read_f32()?.iter().enumerate() {
            let want = want[i % ROWS];
            assert!(
                (f64::from(got) - want).abs() <= TOLERANCE * want.abs().max(1.0),
                "{i}: {got} vs {want}"
            );
        }
        Ok(())
    }

    // an lm_head-sized weight (131072 x 2688) read twenty times a submit, per block type and activation rows
    #[test]
    #[ignore = "benchmark"]
    fn matvec_bandwidth() -> Result<()> {
        let Some(context) = context() else {
            return Ok(());
        };
        const ROWS: usize = 131_072;
        const COLS: usize = 2688;
        const REPEATS: usize = 20;
        const SUBMITS: usize = 5;
        for ty in [BlockType::Q8_0, BlockType::Q5_0] {
            let len = len(ty, ROWS, COLS);
            let mut pages = Pages::new(len);
            weights(ty, pages.bytes(), ROWS, COLS);
            let weights: Arc<dyn Bound> =
                Arc::new(Buffer::with_bytes(&context, &pages.bytes()[..len])?);
            for x_rows in [1usize, 2, 3, 4, 8, 64, 512] {
                let mut xs = Buffer::new(&context, x_rows * COLS * 4)?;
                xs.fill(0.5f32.to_bits())?;
                let (xs, ys) = (
                    Arc::new(xs),
                    Arc::new(Buffer::new(&context, x_rows * ROWS * 4)?),
                );
                let scratch = Arc::new(Buffer::new(&context, MatVec::scratch_bytes(x_rows, COLS))?);
                let op = MatVec {
                    ty,
                    weights: &weights,
                    rows: ROWS,
                    cols: COLS,
                    x: &xs,
                    x_offset: 0,
                    y: &ys,
                    y_offset: 0,
                    x_rows,
                    scratch: &scratch,
                };
                // a record a call, as the backend's ops do, so heavy calls split across submits
                let run = || -> Result<()> {
                    for _ in 0..REPEATS {
                        context.record(|r| r.mat_vec(&op))?;
                    }
                    context.synchronize()
                };
                run()?;
                let start = std::time::Instant::now();
                for _ in 0..SUBMITS {
                    run()?;
                }
                let secs = start.elapsed().as_secs_f64();
                let calls = (SUBMITS * REPEATS) as f64;
                println!(
                    "{ty:?} x {x_rows} rows: {:.1} GB/s, {:.2} TOPS, {:.0} us each",
                    calls * len as f64 / secs / 1e9,
                    calls * (2 * x_rows * ROWS * COLS) as f64 / secs / 1e12,
                    secs / calls * 1e6
                );
            }
        }
        Ok(())
    }

    // one token through a hybrid MoE's dense weights (Nemotron Lightning's: Mamba in and out, attention q and k, the
    // shared expert's up), a record each as decode makes them
    #[test]
    #[ignore = "benchmark"]
    fn matvec_bandwidth_at_decode_shapes() -> Result<()> {
        let Some(context) = context() else {
            return Ok(());
        };
        const REPEATS: usize = 200;
        let shapes = [
            (BlockType::Q8_0, 10_304, 2688),
            (BlockType::Q8_0, 2688, 4096),
            (BlockType::Q8_0, 4096, 2688),
            (BlockType::Q8_0, 256, 2688),
            (BlockType::Q5_0, 3712, 2688),
            (BlockType::Q5_0, 2688, 3712),
        ];
        for (ty, rows, cols) in shapes {
            let len = len(ty, rows, cols);
            let mut pages = Pages::new(len);
            weights(ty, pages.bytes(), rows, cols);
            let weights: Arc<dyn Bound> =
                Arc::new(Buffer::with_bytes(&context, &pages.bytes()[..len])?);
            let mut xs = Buffer::new(&context, cols * 4)?;
            xs.fill(0.5f32.to_bits())?;
            let (xs, ys) = (Arc::new(xs), Arc::new(Buffer::new(&context, rows * 4)?));
            let scratch = Arc::new(Buffer::new(&context, MatVec::scratch_bytes(1, cols))?);
            let op = MatVec {
                ty,
                weights: &weights,
                rows,
                cols,
                x: &xs,
                x_offset: 0,
                y: &ys,
                y_offset: 0,
                x_rows: 1,
                scratch: &scratch,
            };
            let run = || -> Result<()> {
                for _ in 0..REPEATS {
                    context.record(|r| r.mat_vec(&op))?;
                }
                context.synchronize()
            };
            run()?;
            let start = std::time::Instant::now();
            run()?;
            let secs = start.elapsed().as_secs_f64() / REPEATS as f64;
            println!(
                "{ty:?} {rows} x {cols}: {:.1} us, {:.0} GB/s",
                secs * 1e6,
                len as f64 / secs / 1e9
            );
        }
        Ok(())
    }

    // one token's routes through a stack of 128 Q4_0 experts, as decode runs them: up (one activation row for the six
    // routes) and down (a row a route)
    #[test]
    #[ignore = "benchmark"]
    fn expert_matvec_bandwidth_at_decode() -> Result<()> {
        let Some(context) = context() else {
            return Ok(());
        };
        const EXPERTS: usize = 128;
        const K: usize = 6;
        const REPEATS: usize = 200;
        let ty = BlockType::Q4_0;
        for (rows, cols, x_div) in [(1856usize, 2688usize, K), (2688, 1856, 1)] {
            let len = EXPERTS * len(ty, rows, cols);
            let mut pages = Pages::new(len);
            weights(ty, pages.bytes(), EXPERTS * rows, cols);
            let weights: Arc<dyn Bound> =
                Arc::new(Buffer::with_bytes(&context, &pages.bytes()[..len])?);
            let id_bytes: Vec<u8> = (0..K as u32)
                .flat_map(|j| (j * 21 + 5).to_le_bytes())
                .collect();
            let ids = Arc::new(Buffer::with_bytes(&context, &id_bytes)?);
            let x: Vec<f32> = (0..K.div_ceil(x_div) * cols)
                .map(|i| (i % 13) as f32 - 6.0)
                .collect();
            let xs = Arc::new(Buffer::with_f32(&context, &x)?);
            let ys = Arc::new(Buffer::new(&context, K * rows * 4)?);
            let scratch = Arc::new(Buffer::new(
                &context,
                crate::MoeMatVec::scratch_bytes(&context, K, x_div, cols),
            )?);
            let routing = Arc::new(Buffer::new(
                &context,
                crate::MoeMatVec::routing_bytes(K, EXPERTS),
            )?);
            let op = crate::MoeMatVec {
                ty,
                weights: &weights,
                experts: EXPERTS,
                rows,
                cols,
                slots: K,
                x_div,
                x: (&xs, 0),
                ids: (&ids, 0),
                y: &ys,
                scratch: &scratch,
                routing: &routing,
            };
            let run = || -> Result<()> {
                for _ in 0..REPEATS {
                    context.record(|r| r.moe_mat_vec(&op))?;
                }
                context.synchronize()
            };
            run()?;
            let start = std::time::Instant::now();
            run()?;
            let secs = start.elapsed().as_secs_f64() / REPEATS as f64;
            let bytes = K * len / EXPERTS;
            println!(
                "Q4_0 experts {rows} x {cols}, {K} routes: {:.1} us, {:.0} GB/s",
                secs * 1e6,
                bytes as f64 / secs / 1e9
            );
        }
        Ok(())
    }

    // a prompt step's experts: 512 tokens routed to 6 of 128 experts (1856 x 2688, Q5_0), each pass timed on its own
    #[test]
    #[ignore = "benchmark"]
    fn expert_matmul_throughput() -> Result<()> {
        // one shape of the list alone (its index), and routes drawn uniformly (llama.cpp's test-backend-ops perf)
        // where the default is one real prompt layer's
        const BENCH_SHAPE_ENV: &str = "INFERENCE_VULKAN_BENCH_SHAPE";
        const BENCH_EVEN_ENV: &str = "INFERENCE_VULKAN_BENCH_EVEN";
        let Some(context) = context() else {
            return Ok(());
        };
        const EXPERTS: usize = 128;
        const TOKENS: usize = 512;
        const K: usize = 6;
        const REPEATS: usize = 20;
        const BATCH: usize = 40;
        // one 512-token prompt layer's routes an expert, busiest first (18 experts unrouted)
        const PROMPT_ROUTES: [usize; 110] = [
            264, 213, 198, 181, 171, 98, 96, 95, 91, 85, 75, 59, 56, 51, 47, 47, 47, 47, 46, 41,
            40, 39, 36, 35, 35, 32, 31, 30, 30, 29, 28, 27, 27, 26, 25, 23, 22, 22, 22, 21, 21, 20,
            19, 18, 18, 18, 17, 17, 16, 16, 16, 16, 14, 14, 13, 13, 11, 10, 10, 9, 9, 9, 8, 8, 7,
            7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2,
            2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ];
        // a hybrid MoE's up (one activation row a token's routes) and down (a row a route) projections
        for (ty, rows, cols, x_div) in [
            (BlockType::Q5_0, 1856usize, 2688usize, K),
            (BlockType::Q4_0, 1856, 2688, K),
            (BlockType::Q4_0, 2688, 1856, 1),
        ]
        .into_iter()
        .enumerate()
        .filter(|(i, _)| std::env::var(BENCH_SHAPE_ENV).map_or(true, |v| v == i.to_string()))
        .map(|(_, shape)| shape)
        {
            let len = EXPERTS * len(ty, rows, cols);
            let mut pages = Pages::new(len);
            weights(ty, pages.bytes(), EXPERTS * rows, cols);
            let weights: Arc<dyn Bound> =
                Arc::new(Buffer::with_bytes(&context, &pages.bytes()[..len])?);
            let slots = TOKENS * K;
            let mut id_bytes = vec![0u8; slots * 4];
            if std::env::var_os(BENCH_EVEN_ENV).is_some() {
                let mut state = 0x9e37_79b9u32;
                for t in 0..TOKENS {
                    let mut picked = Vec::new();
                    while picked.len() < K {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        let e = (state >> 8) as usize % EXPERTS;
                        if !picked.contains(&e) {
                            picked.push(e);
                        }
                    }
                    for (j, e) in picked.into_iter().enumerate() {
                        let slot = t * K + j;
                        id_bytes[slot * 4..slot * 4 + 4].copy_from_slice(&(e as u32).to_le_bytes());
                    }
                }
            } else {
                // route i to token i % TOKENS: an expert's (at most TOKENS) routes land on distinct tokens
                let mut i = 0;
                for (e, &count) in PROMPT_ROUTES.iter().enumerate() {
                    for _ in 0..count {
                        let slot = (i % TOKENS) * K + i / TOKENS;
                        id_bytes[slot * 4..slot * 4 + 4].copy_from_slice(&(e as u32).to_le_bytes());
                        i += 1;
                    }
                }
            }
            let ids = Arc::new(Buffer::with_bytes(&context, &id_bytes)?);
            let x: Vec<f32> = (0..slots / x_div * cols)
                .map(|i| (i % 13) as f32 - 6.0)
                .collect();
            let xs = Arc::new(Buffer::with_f32(&context, &x)?);
            let ys = Arc::new(Buffer::new(&context, slots * rows * 4)?);
            let scratch = Arc::new(Buffer::new(
                &context,
                crate::MoeMatVec::scratch_bytes(&context, slots, x_div, cols),
            )?);
            let routing = Arc::new(Buffer::new(
                &context,
                crate::MoeMatVec::routing_bytes(slots, EXPERTS),
            )?);
            let op = crate::MoeMatVec {
                ty,
                weights: &weights,
                experts: EXPERTS,
                rows,
                cols,
                slots,
                x_div,
                x: (&xs, 0),
                ids: (&ids, 0),
                y: &ys,
                scratch: &scratch,
                routing: &routing,
            };
            // a submit of BATCH back to back, as a prompt records its layers, so its wall time is the GPU's
            let batch = |r: &crate::Recorder<'_>| {
                for _ in 0..BATCH {
                    r.moe_mat_vec(&op)?;
                }
                Ok(())
            };
            context.submit(batch)?;
            let start = std::time::Instant::now();
            for _ in 0..REPEATS {
                context.submit(batch)?;
            }
            let secs = start.elapsed().as_secs_f64() / (REPEATS * BATCH) as f64;
            println!(
                "experts {ty:?} {rows} x {cols}: {:.2} ms a pass, {:.2} int8 TOPS",
                secs * 1e3,
                (2 * slots * rows * cols) as f64 / secs / 1e12
            );
        }
        Ok(())
    }
}
