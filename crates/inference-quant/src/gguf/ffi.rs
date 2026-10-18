//! FFI bindings for indexed MoE CUDA kernels.
//!
//! These bindings allow Rust code to call the optimized CUDA kernels
//! for indexed MoE forward pass with GGUF quantized weights.

#![allow(dead_code)]
#![allow(improper_ctypes)]

use std::ffi::c_void;

macro_rules! moe_lora_decode_kernels {
    ($( ($gate_up:ident, $down:ident) ),* $(,)?) => {
        inference_tensor::kernel_library! {
            library = "inferencequant";
            path = option_env!("INFERENCE_KERNEL_LIB_INFERENCEQUANT").unwrap_or("");
            id = option_env!("INFERENCE_KERNEL_ID_INFERENCEQUANT").unwrap_or("");
            unsafe extern "C" {
                $(
                pub fn $gate_up(
                gate_weights: *const c_void,
                up_weights: *const c_void,
                all_inputs: *const c_void,
                indices: *const u32,
                all_outputs: *mut c_void,
                n: i32,
                k: i32,
                batch: i32,
                topk: i32,
                k_padded: i32,
                num_experts: i32,
                output_type: i32,
                stream: *mut c_void,
                ) -> i32;
                pub fn $down(
                all_weights: *const c_void,
                all_inputs: *const c_void,
                indices: *const u32,
                all_outputs: *mut c_void,
                n: i32,
                k: i32,
                batch: i32,
                topk: i32,
                k_padded: i32,
                num_experts: i32,
                output_type: i32,
                stream: *mut c_void,
                ) -> i32;
                )*
            }
        }
    };
}

inference_tensor::kernel_library! {
    library = "inferencequant";
    path = option_env!("INFERENCE_KERNEL_LIB_INFERENCEQUANT").unwrap_or("");
    id = option_env!("INFERENCE_KERNEL_ID_INFERENCEQUANT").unwrap_or("");
    unsafe extern "C" {
        /// Launch indexed MoE forward kernel for Q2_K weights with Q8_1 input
        pub fn launch_indexed_moe_forward_q2k_q8_1(
            all_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            input_dim1: i32,
            stream: *mut c_void,
        );

        /// Launch indexed MoE forward kernel for Q3_K weights with Q8_1 input
        pub fn launch_indexed_moe_forward_q3k_q8_1(
            all_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            input_dim1: i32,
            stream: *mut c_void,
        );

        /// Launch indexed MoE forward kernel for Q4_K weights with Q8_1 input
        pub fn launch_indexed_moe_forward_q4k_q8_1(
            all_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            input_dim1: i32,
            stream: *mut c_void,
        );

        /// Launch indexed MoE forward kernel for Q5_K weights with Q8_1 input
        pub fn launch_indexed_moe_forward_q5k_q8_1(
            all_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            input_dim1: i32,
            stream: *mut c_void,
        );

        /// Launch indexed MoE forward kernel for Q6_K weights with Q8_1 input
        pub fn launch_indexed_moe_forward_q6k_q8_1(
            all_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            input_dim1: i32,
            stream: *mut c_void,
        );

        /// Launch indexed MoE forward kernel for Q8_0 weights with Q8_1 input
        pub fn launch_indexed_moe_forward_q8_0_q8_1(
            all_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            input_dim1: i32,
            stream: *mut c_void,
        );

        /// Launch indexed MoE forward kernel for Q4_0 weights with Q8_1 input
        pub fn launch_indexed_moe_forward_q4_0_q8_1(
            all_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            input_dim1: i32,
            stream: *mut c_void,
        );

        /// Launch indexed MoE forward kernel for Q4_1 weights with Q8_1 input
        pub fn launch_indexed_moe_forward_q4_1_q8_1(
            all_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            input_dim1: i32,
            stream: *mut c_void,
        );

        /// Launch indexed MoE forward kernel for Q5_0 weights with Q8_1 input
        pub fn launch_indexed_moe_forward_q5_0_q8_1(
            all_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            input_dim1: i32,
            stream: *mut c_void,
        );

        /// Launch indexed MoE forward kernel for Q5_1 weights with Q8_1 input
        pub fn launch_indexed_moe_forward_q5_1_q8_1(
            all_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            input_dim1: i32,
            stream: *mut c_void,
        );

        /// Launch indexed MoE forward kernel for Q8_1 weights with Q8_1 input
        pub fn launch_indexed_moe_forward_q8_1_q8_1(
            all_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            input_dim1: i32,
            stream: *mut c_void,
        );

        // Grouped MoE dispatch and GEMM

        /// Build expert dispatch tables on GPU: expert_bounds + sorted_token_ids
        pub fn launch_moe_dispatch(
            topk_ids: *const i32,
            expert_bounds: *mut i32,
            sorted_token_ids: *mut i32,
            sorted_source_ids: *mut i32,
            total_assignments: i32,
            num_experts: i32,
            topk: i32,
            expert_counts: *mut i32,
            expert_cursors: *mut i32,
            stream: *mut c_void,
        );

        pub fn launch_moe_weighted_reduce_flat(
            inputs: *const c_void,
            topk_weights: *const f32,
            outputs: *mut c_void,
            num_tokens: i32,
            hidden: i32,
            topk: i32,
            stream: *mut c_void,
        ) -> i32;

        pub fn launch_moe_weighted_reduce_flat_bf16(
            inputs: *const c_void,
            topk_weights: *const f32,
            outputs: *mut c_void,
            num_tokens: i32,
            hidden: i32,
            topk: i32,
            stream: *mut c_void,
        ) -> i32;

        pub fn launch_moe_weighted_reduce_flat_f16_input(
            inputs: *const c_void,
            topk_weights: *const f32,
            outputs: *mut c_void,
            num_tokens: i32,
            hidden: i32,
            topk: i32,
            stream: *mut c_void,
        ) -> i32;

        pub fn launch_moe_weighted_reduce_flat_bf16_input(
            inputs: *const c_void,
            topk_weights: *const f32,
            outputs: *mut c_void,
            num_tokens: i32,
            hidden: i32,
            topk: i32,
            stream: *mut c_void,
        ) -> i32;

        // Fused MoE decode kernels

        // Fused gate+up+activation+multiply launchers
        // All share the same signature: (gate_weights, up_weights, inputs_q8_1,
        //   indices, outputs, n, k, batch, topk, k_padded, act_type, stream)

        pub fn launch_moe_gemv_fused_gate_up_q8_0_q8_1(
            gate_weights: *const c_void,
            up_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            act_type: i32,
            stream: *mut c_void,
        );
        pub fn launch_moe_gemv_fused_gate_up_q4_0_q8_1(
            gate_weights: *const c_void,
            up_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            act_type: i32,
            stream: *mut c_void,
        );
        pub fn launch_moe_gemv_fused_gate_up_q4_1_q8_1(
            gate_weights: *const c_void,
            up_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            act_type: i32,
            stream: *mut c_void,
        );
        pub fn launch_moe_gemv_fused_gate_up_q5_0_q8_1(
            gate_weights: *const c_void,
            up_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            act_type: i32,
            stream: *mut c_void,
        );
        pub fn launch_moe_gemv_fused_gate_up_q5_1_q8_1(
            gate_weights: *const c_void,
            up_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            act_type: i32,
            stream: *mut c_void,
        );
        pub fn launch_moe_gemv_fused_gate_up_q8_1_q8_1(
            gate_weights: *const c_void,
            up_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            act_type: i32,
            stream: *mut c_void,
        );
        pub fn launch_moe_gemv_fused_gate_up_q2k_q8_1(
            gate_weights: *const c_void,
            up_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            act_type: i32,
            stream: *mut c_void,
        );
        pub fn launch_moe_gemv_fused_gate_up_q3k_q8_1(
            gate_weights: *const c_void,
            up_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            act_type: i32,
            stream: *mut c_void,
        );
        pub fn launch_moe_gemv_fused_gate_up_q4k_q8_1(
            gate_weights: *const c_void,
            up_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            act_type: i32,
            stream: *mut c_void,
        );
        pub fn launch_moe_gemv_fused_gate_up_q5k_q8_1(
            gate_weights: *const c_void,
            up_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            act_type: i32,
            stream: *mut c_void,
        );
        pub fn launch_moe_gemv_fused_gate_up_q6k_q8_1(
            gate_weights: *const c_void,
            up_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            act_type: i32,
            stream: *mut c_void,
        );


        // Fused down+aggregate launchers
        // All share the same signature: (weights, inputs_q8_1, indices,
        //   topk_weights, outputs, n, k, batch, topk, k_padded, stream)

        pub fn launch_moe_gemv_down_aggregate_q8_0_q8_1(
            all_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            topk_weights: *const f32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            stream: *mut c_void,
        );
        pub fn launch_moe_gemv_down_aggregate_q4_0_q8_1(
            all_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            topk_weights: *const f32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            stream: *mut c_void,
        );
        pub fn launch_moe_gemv_down_aggregate_q4_1_q8_1(
            all_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            topk_weights: *const f32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            stream: *mut c_void,
        );
        pub fn launch_moe_gemv_down_aggregate_q5_0_q8_1(
            all_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            topk_weights: *const f32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            stream: *mut c_void,
        );
        pub fn launch_moe_gemv_down_aggregate_q5_1_q8_1(
            all_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            topk_weights: *const f32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            stream: *mut c_void,
        );
        pub fn launch_moe_gemv_down_aggregate_q8_1_q8_1(
            all_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            topk_weights: *const f32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            stream: *mut c_void,
        );
        pub fn launch_moe_gemv_down_aggregate_q2k_q8_1(
            all_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            topk_weights: *const f32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            stream: *mut c_void,
        );
        pub fn launch_moe_gemv_down_aggregate_q3k_q8_1(
            all_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            topk_weights: *const f32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            stream: *mut c_void,
        );
        pub fn launch_moe_gemv_down_aggregate_q4k_q8_1(
            all_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            topk_weights: *const f32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            stream: *mut c_void,
        );
        pub fn launch_moe_gemv_down_aggregate_q5k_q8_1(
            all_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            topk_weights: *const f32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            stream: *mut c_void,
        );
        pub fn launch_moe_gemv_down_aggregate_q6k_q8_1(
            all_weights: *const c_void,
            all_inputs: *const c_void,
            indices: *const u32,
            topk_weights: *const f32,
            all_outputs: *mut f32,
            n: i32,
            k: i32,
            batch: i32,
            topk: i32,
            k_padded: i32,
            stream: *mut c_void,
        );


        pub fn launch_gather_rows_u8(
            src: *const c_void,
            ids: *const c_void,
            dst: *mut c_void,
            row_bytes: i64,
            rows: i64,
            n: i64,
            stream: *mut c_void,
        );

        /// BF16 -> Q8_1 quantize
        pub fn launch_mmvq_gguf_quantize_q8_1_bf16(
            x: *const c_void,
            vy: *mut c_void,
            kx: i32,
            kx_padded: i32,
            num_rows: i32,
            stream: *mut c_void,
        );

        /// F16 -> Q8_1 quantize
        pub fn launch_mmvq_gguf_quantize_q8_1_f16(
            x: *const c_void,
            vy: *mut c_void,
            kx: i32,
            kx_padded: i32,
            num_rows: i32,
            stream: *mut c_void,
        );

        /// F32 -> Q8_1 quantize
        pub fn launch_mmvq_gguf_quantize_q8_1_f32(
            x: *const c_void,
            vy: *mut c_void,
            kx: i32,
            kx_padded: i32,
            num_rows: i32,
            stream: *mut c_void,
        );

        /// BF16 / F16 / F32 -> the tensor-core matvec's activation layout (`mmvq_tc_quantize`)
        pub fn launch_mmvq_tc_quantize_bf16(
            x: *const c_void,
            vy: *mut c_void,
            kx: i32,
            num_rows: i32,
            stream: *mut c_void,
        );
        pub fn launch_mmvq_tc_quantize_f16(
            x: *const c_void,
            vy: *mut c_void,
            kx: i32,
            num_rows: i32,
            stream: *mut c_void,
        );
        pub fn launch_mmvq_tc_quantize_f32(
            x: *const c_void,
            vy: *mut c_void,
            kx: i32,
            num_rows: i32,
            stream: *mut c_void,
        );

        // MMQ (prompt) kernels

        // MMQ quantize launchers (f32 -> block_q8_1_mmq)
        pub fn launch_mmq_quantize_q8_1_D4(
            x: *const c_void,
            ids: *const i32,
            vy: *mut c_void,
            type_x: i32,
            ne00: i64,
            s01: i64,
            s02: i64,
            s03: i64,
            ne0: i64,
            ne1: i64,
            ne2: i64,
            ne3: i64,
            stream: *mut c_void,
        );
        pub fn launch_mmq_quantize_q8_1_DS4(
            x: *const c_void,
            ids: *const i32,
            vy: *mut c_void,
            type_x: i32,
            ne00: i64,
            s01: i64,
            s02: i64,
            s03: i64,
            ne0: i64,
            ne1: i64,
            ne2: i64,
            ne3: i64,
            stream: *mut c_void,
        );
        pub fn launch_mmq_quantize_q8_1_D2S6(
            x: *const c_void,
            ids: *const i32,
            vy: *mut c_void,
            type_x: i32,
            ne00: i64,
            s01: i64,
            s02: i64,
            s03: i64,
            ne0: i64,
            ne1: i64,
            ne2: i64,
            ne3: i64,
            stream: *mut c_void,
        );
        pub fn launch_mmq_quantize_glu_q8_1_D4_f32(
            gate: *const f32,
            up: *const f32,
            ids: *const i32,
            vy: *mut c_void,
            ne00: i64,
            s01: i64,
            ne0: i64,
            ne1: i64,
            activation: i32,
            stream: *mut c_void,
        );
        pub fn launch_mmq_quantize_glu_q8_1_DS4_f32(
            gate: *const f32,
            up: *const f32,
            ids: *const i32,
            vy: *mut c_void,
            ne00: i64,
            s01: i64,
            ne0: i64,
            ne1: i64,
            activation: i32,
            stream: *mut c_void,
        );
        pub fn launch_mmq_quantize_glu_q8_1_D2S6_f32(
            gate: *const f32,
            up: *const f32,
            ids: *const i32,
            vy: *mut c_void,
            ne00: i64,
            s01: i64,
            ne0: i64,
            ne1: i64,
            activation: i32,
            stream: *mut c_void,
        );
        pub fn launch_mmq_quantize_glu_q8_1_D4(
            gate: *const c_void,
            up: *const c_void,
            ids: *const i32,
            vy: *mut c_void,
            type_x: i32,
            ne00: i64,
            s01: i64,
            ne0: i64,
            ne1: i64,
            activation: i32,
            stream: *mut c_void,
        );
        pub fn launch_mmq_quantize_glu_q8_1_DS4(
            gate: *const c_void,
            up: *const c_void,
            ids: *const i32,
            vy: *mut c_void,
            type_x: i32,
            ne00: i64,
            s01: i64,
            ne0: i64,
            ne1: i64,
            activation: i32,
            stream: *mut c_void,
        );
        pub fn launch_mmq_quantize_glu_q8_1_D2S6(
            gate: *const c_void,
            up: *const c_void,
            ids: *const i32,
            vy: *mut c_void,
            type_x: i32,
            ne00: i64,
            s01: i64,
            ne0: i64,
            ne1: i64,
            activation: i32,
            stream: *mut c_void,
        );
    }
}

// launchers sharing one signature
mod moe_lora_decode {
    use super::*;

    moe_lora_decode_kernels!(
        (
            launch_moe_gemv_gate_up_pair_q8_0_q8_1,
            launch_moe_gemv_lora_down_q8_0_q8_1
        ),
        (
            launch_moe_gemv_gate_up_pair_q4_0_q8_1,
            launch_moe_gemv_lora_down_q4_0_q8_1
        ),
        (
            launch_moe_gemv_gate_up_pair_q4_1_q8_1,
            launch_moe_gemv_lora_down_q4_1_q8_1
        ),
        (
            launch_moe_gemv_gate_up_pair_q5_0_q8_1,
            launch_moe_gemv_lora_down_q5_0_q8_1
        ),
        (
            launch_moe_gemv_gate_up_pair_q5_1_q8_1,
            launch_moe_gemv_lora_down_q5_1_q8_1
        ),
        (
            launch_moe_gemv_gate_up_pair_q8_1_q8_1,
            launch_moe_gemv_lora_down_q8_1_q8_1
        ),
        (
            launch_moe_gemv_gate_up_pair_q2k_q8_1,
            launch_moe_gemv_lora_down_q2k_q8_1
        ),
        (
            launch_moe_gemv_gate_up_pair_q3k_q8_1,
            launch_moe_gemv_lora_down_q3k_q8_1
        ),
        (
            launch_moe_gemv_gate_up_pair_q4k_q8_1,
            launch_moe_gemv_lora_down_q4k_q8_1
        ),
        (
            launch_moe_gemv_gate_up_pair_q5k_q8_1,
            launch_moe_gemv_lora_down_q5k_q8_1
        ),
        (
            launch_moe_gemv_gate_up_pair_q6k_q8_1,
            launch_moe_gemv_lora_down_q6k_q8_1
        ),
    );
}
#[allow(unused_imports)]
pub use moe_lora_decode::*;

// One GGUF type's kernels, in `libinferencequant_<type>.so`: the matvecs every type has, plus `$extra`.
macro_rules! gguf_type_library {
    ($t:ident, $upper:literal; $($extra:tt)*) => {
        paste::paste! {
            pub mod $t {
                use super::*;

                inference_tensor::kernel_library! {
                    library = concat!("inferencequant_", stringify!($t));
                    path = option_env!(concat!("INFERENCE_KERNEL_LIB_INFERENCEQUANT_", $upper)).unwrap_or("");
                    id = option_env!(concat!("INFERENCE_KERNEL_ID_INFERENCEQUANT_", $upper)).unwrap_or("");
                    unsafe extern "C" {
                        pub fn [<launch_mmvq_gguf_ $t _bf16_plain>](
                            vx: *const c_void,
                            vy: *const c_void,
                            dst: *mut c_void,
                            ncols_x: i32,
                            nrows_x: i32,
                            stride_col_y: i32,
                            stride_col_dst: i32,
                            b_size: i32,
                            stream: *mut c_void,
                        );
                        pub fn [<launch_mmvq_gguf_ $t _f16_plain>](
                            vx: *const c_void,
                            vy: *const c_void,
                            dst: *mut c_void,
                            ncols_x: i32,
                            nrows_x: i32,
                            stride_col_y: i32,
                            stride_col_dst: i32,
                            b_size: i32,
                            stream: *mut c_void,
                        );
                        pub fn [<launch_mmvq_gguf_ $t _f32_plain>](
                            vx: *const c_void,
                            vy: *const c_void,
                            dst: *mut c_void,
                            ncols_x: i32,
                            nrows_x: i32,
                            stride_col_y: i32,
                            stride_col_dst: i32,
                            b_size: i32,
                            stream: *mut c_void,
                        );
                        pub fn [<launch_mmvq_gguf_ $t _moe>](
                            vx_gate: *const c_void,
                            vx_up: *const c_void,
                            expert_stride: i64,
                            vy: *const c_void,
                            ids: *const c_void,
                            dst: *mut f32,
                            ncols_x: i32,
                            nrows_x: i32,
                            stride_col_y: i32,
                            y_div: i32,
                            n_pairs: i32,
                            activation: i32,
                            stream: *mut c_void,
                        );
                        $($extra)*
                    }
                }
            }
            #[allow(unused_imports)]
            pub use $t::*;
        }
    };
}

// the prompt kernels, which every type but IQ1_M has
macro_rules! gguf_mmq_type_library {
    ($t:ident, $upper:literal; $($extra:tt)*) => {
        paste::paste! {
            gguf_type_library! {
                $t, $upper;
                pub fn [<launch_mmq_gguf_ $t>](
                    tmp_fixup: *mut c_void,
                    x: *const c_void,
                    y: *const c_void,
                    dst: *mut c_void,
                    ncols_x: i64,
                    nrows_x: i64,
                    ncols_y: i64,
                    stride_row_x: i64,
                    stride_col_dst: i64,
                    cc: i32,
                    nsm: i32,
                    smpbo: i64,
                    warp_size: i32,
                    type_dst: i32,
                    stream: *mut c_void,
                );
                pub fn [<launch_mmq_gguf_ $t _moe>](
                    tmp_fixup: *mut c_void,
                    x: *const c_void,
                    y: *const c_void,
                    ids_dst: *const i32,
                    expert_bounds: *const i32,
                    dst: *mut c_void,
                    ncols_x: i64,
                    nrows_x: i64,
                    ncols_dst: i64,
                    stride_row_x: i64,
                    stride_col_dst: i64,
                    num_experts: i64,
                    ncols_max: i64,
                    cc: i32,
                    nsm: i32,
                    smpbo: i64,
                    warp_size: i32,
                    stream: *mut c_void,
                );
                $($extra)*
            }
        }
    };
}

// the standard types, with fused gate/up and Q/K/V matvecs
macro_rules! gguf_standard_types {
    ($( ($t:ident, $upper:literal) ),* $(,)?) => {
        $(
        paste::paste! {
            gguf_mmq_type_library! {
                $t, $upper;
                pub fn [<launch_mmvq_gguf_ $t _bf16_fused_glu>](
                    vx_gate: *const c_void,
                    vx_up: *const c_void,
                    vy: *const c_void,
                    dst: *mut c_void,
                    ncols_x: i32,
                    nrows_x: i32,
                    stride_col_y: i32,
                    stride_col_dst: i32,
                    b_size: i32,
                    activation: i32,
                    stream: *mut c_void,
                );
                pub fn [<launch_mmvq_gguf_ $t _f16_fused_glu>](
                    vx_gate: *const c_void,
                    vx_up: *const c_void,
                    vy: *const c_void,
                    dst: *mut c_void,
                    ncols_x: i32,
                    nrows_x: i32,
                    stride_col_y: i32,
                    stride_col_dst: i32,
                    b_size: i32,
                    activation: i32,
                    stream: *mut c_void,
                );
                pub fn [<launch_mmvq_gguf_ $t _f32_fused_glu>](
                    vx_gate: *const c_void,
                    vx_up: *const c_void,
                    vy: *const c_void,
                    dst: *mut c_void,
                    ncols_x: i32,
                    nrows_x: i32,
                    stride_col_y: i32,
                    stride_col_dst: i32,
                    b_size: i32,
                    activation: i32,
                    stream: *mut c_void,
                );
                pub fn [<launch_mmvq_gguf_ $t _bf16_fused_qkv>](
                    vx_q: *const c_void,
                    vx_k: *const c_void,
                    vx_v: *const c_void,
                    vy: *const c_void,
                    q_dst: *mut c_void,
                    k_dst: *mut c_void,
                    v_dst: *mut c_void,
                    ncols_x: i32,
                    nrows_q: i32,
                    nrows_k: i32,
                    nrows_v: i32,
                    stride_col_y: i32,
                    b_size: i32,
                    stream: *mut c_void,
                );
                pub fn [<launch_mmvq_gguf_ $t _f16_fused_qkv>](
                    vx_q: *const c_void,
                    vx_k: *const c_void,
                    vx_v: *const c_void,
                    vy: *const c_void,
                    q_dst: *mut c_void,
                    k_dst: *mut c_void,
                    v_dst: *mut c_void,
                    ncols_x: i32,
                    nrows_q: i32,
                    nrows_k: i32,
                    nrows_v: i32,
                    stride_col_y: i32,
                    b_size: i32,
                    stream: *mut c_void,
                );
                pub fn [<launch_mmvq_gguf_ $t _f32_fused_qkv>](
                    vx_q: *const c_void,
                    vx_k: *const c_void,
                    vx_v: *const c_void,
                    vy: *const c_void,
                    q_dst: *mut c_void,
                    k_dst: *mut c_void,
                    v_dst: *mut c_void,
                    ncols_x: i32,
                    nrows_q: i32,
                    nrows_k: i32,
                    nrows_v: i32,
                    stride_col_y: i32,
                    b_size: i32,
                    stream: *mut c_void,
                );
            }
        }
        )*
    };
}

// the IQ types, which dequantize with their own kernels
macro_rules! gguf_iq_types {
    ($( ($t:ident, $upper:literal) ),* $(,)?) => {
        $(
        paste::paste! {
            gguf_mmq_type_library! {
                $t, $upper;
                pub fn [<launch_dequantize_ $t _bf16>](
                    vx: *const c_void,
                    dst: *mut c_void,
                    nrows: i64,
                    ncols: i64,
                    stream: *mut c_void,
                );
                pub fn [<launch_dequantize_ $t _f16>](
                    vx: *const c_void,
                    dst: *mut c_void,
                    nrows: i64,
                    ncols: i64,
                    stream: *mut c_void,
                );
                pub fn [<launch_dequantize_ $t _f32>](
                    vx: *const c_void,
                    dst: *mut c_void,
                    nrows: i64,
                    ncols: i64,
                    stream: *mut c_void,
                );
            }
        }
        )*
    };
}

gguf_standard_types!(
    (q4_0, "Q4_0"),
    (q4_1, "Q4_1"),
    (q5_0, "Q5_0"),
    (q5_1, "Q5_1"),
    (q8_0, "Q8_0"),
    (q2_k, "Q2_K"),
    (q3_k, "Q3_K"),
    (q4_k, "Q4_K"),
    (q5_k, "Q5_K"),
    (q6_k, "Q6_K"),
);

// The types with a tile-ordered layout for 1-8 row matvecs (mmvq_tc.cuh) and prefill (mmq) besides their own kernels
macro_rules! gguf_iq_tc_types {
    ($( ($t:ident, $upper:literal) ),* $(,)?) => {
        $(
        paste::paste! {
            gguf_mmq_type_library! {
                $t, $upper;
                pub fn [<launch_dequantize_ $t _bf16>](
                    vx: *const c_void,
                    dst: *mut c_void,
                    nrows: i64,
                    ncols: i64,
                    stream: *mut c_void,
                );
                pub fn [<launch_dequantize_ $t _f16>](
                    vx: *const c_void,
                    dst: *mut c_void,
                    nrows: i64,
                    ncols: i64,
                    stream: *mut c_void,
                );
                pub fn [<launch_dequantize_ $t _f32>](
                    vx: *const c_void,
                    dst: *mut c_void,
                    nrows: i64,
                    ncols: i64,
                    stream: *mut c_void,
                );
                pub fn [<launch_ $t _untile_rows>](
                    src: *const c_void,
                    ids: *const c_void,
                    dst: *mut c_void,
                    n: i64,
                    ncols: i64,
                    stream: *mut c_void,
                );
                pub fn [<launch_mmq_gguf_ $t _tiled>](
                    tmp_fixup_ptr: *mut c_void,
                    x: *const c_void,
                    y_q8_1_mmq: *const c_void,
                    dst: *mut c_void,
                    ncols_x: i64,
                    nrows_x: i64,
                    ncols_y: i64,
                    stride_row_x: i64,
                    stride_col_dst: i64,
                    cc: i32,
                    nsm: i32,
                    smpbo: i64,
                    warp_size_host: i32,
                    type_dst: i32,
                    stream: *mut c_void,
                );
                pub fn [<launch_mmvq_tc_gguf_ $t _bf16_plain>](
                    vx: *const c_void,
                    vy: *const c_void,
                    dst: *mut c_void,
                    ncols_x: i32,
                    nrows_x: i32,
                    stride_col_dst: i32,
                    b_size: i32,
                    cfg: i32,
                    stream: *mut c_void,
                );
                pub fn [<launch_mmvq_tc_gguf_ $t _f16_plain>](
                    vx: *const c_void,
                    vy: *const c_void,
                    dst: *mut c_void,
                    ncols_x: i32,
                    nrows_x: i32,
                    stride_col_dst: i32,
                    b_size: i32,
                    cfg: i32,
                    stream: *mut c_void,
                );
                pub fn [<launch_mmvq_tc_gguf_ $t _f32_plain>](
                    vx: *const c_void,
                    vy: *const c_void,
                    dst: *mut c_void,
                    ncols_x: i32,
                    nrows_x: i32,
                    stride_col_dst: i32,
                    b_size: i32,
                    cfg: i32,
                    stream: *mut c_void,
                );
            }
        }
        )*
    };
}

gguf_iq_tc_types!((iq4_ks, "IQ4_KS"),);

gguf_iq_types!(
    (iq4_nl, "IQ4_NL"),
    (iq4_xs, "IQ4_XS"),
    (iq2_xxs, "IQ2_XXS"),
    (iq2_xs, "IQ2_XS"),
    (iq2_s, "IQ2_S"),
    (iq3_xxs, "IQ3_XXS"),
    (iq3_s, "IQ3_S"),
    (iq1_s, "IQ1_S"),
    (iq1_kt, "IQ1_KT"),
    (iq2_kt, "IQ2_KT"),
    (iq3_kt, "IQ3_KT"),
    (iq4_kt, "IQ4_KT"),
    (iq2_k, "IQ2_K"),
    (iq3_k, "IQ3_K"),
    (iq4_k, "IQ4_K"),
    (iq5_k, "IQ5_K"),
    (iq6_k, "IQ6_K"),
    (iq2_ks, "IQ2_KS"),
    (iq4_kss, "IQ4_KSS"),
    (iq5_ks, "IQ5_KS"),
    (iq3_ks, "IQ3_KS"),
    (iq2_kl, "IQ2_KL"),
);

// IQ1_M has no prompt kernel
gguf_type_library! {
    iq1_m, "IQ1_M";
    pub fn launch_dequantize_iq1_m_bf16(
        vx: *const c_void,
        dst: *mut c_void,
        nrows: i64,
        ncols: i64,
        stream: *mut c_void,
    );
    pub fn launch_dequantize_iq1_m_f16(
        vx: *const c_void,
        dst: *mut c_void,
        nrows: i64,
        ncols: i64,
        stream: *mut c_void,
    );
    pub fn launch_dequantize_iq1_m_f32(
        vx: *const c_void,
        dst: *mut c_void,
        nrows: i64,
        ncols: i64,
        stream: *mut c_void,
    );
}

/// Each GGUF type with a kernel library of its own: its library suffix (`q4_k`), its type, and the opener of its library.
#[allow(clippy::type_complexity)]
pub(crate) const TYPE_LIBRARIES: &[(&str, super::kernel::GgufType, fn() -> Result<(), String>)] = &[
    ("q4_0", super::kernel::GgufType::Q4_0, q4_0::ensure_loaded),
    ("q4_1", super::kernel::GgufType::Q4_1, q4_1::ensure_loaded),
    ("q5_0", super::kernel::GgufType::Q5_0, q5_0::ensure_loaded),
    ("q5_1", super::kernel::GgufType::Q5_1, q5_1::ensure_loaded),
    ("q8_0", super::kernel::GgufType::Q8_0, q8_0::ensure_loaded),
    ("q2_k", super::kernel::GgufType::Q2K, q2_k::ensure_loaded),
    ("q3_k", super::kernel::GgufType::Q3K, q3_k::ensure_loaded),
    ("q4_k", super::kernel::GgufType::Q4K, q4_k::ensure_loaded),
    ("q5_k", super::kernel::GgufType::Q5K, q5_k::ensure_loaded),
    ("q6_k", super::kernel::GgufType::Q6K, q6_k::ensure_loaded),
    (
        "iq4_nl",
        super::kernel::GgufType::Iq4Nl,
        iq4_nl::ensure_loaded,
    ),
    (
        "iq4_xs",
        super::kernel::GgufType::Iq4Xs,
        iq4_xs::ensure_loaded,
    ),
    (
        "iq2_xxs",
        super::kernel::GgufType::Iq2Xxs,
        iq2_xxs::ensure_loaded,
    ),
    (
        "iq2_xs",
        super::kernel::GgufType::Iq2Xs,
        iq2_xs::ensure_loaded,
    ),
    ("iq2_s", super::kernel::GgufType::Iq2S, iq2_s::ensure_loaded),
    (
        "iq3_xxs",
        super::kernel::GgufType::Iq3Xxs,
        iq3_xxs::ensure_loaded,
    ),
    ("iq3_s", super::kernel::GgufType::Iq3S, iq3_s::ensure_loaded),
    ("iq1_s", super::kernel::GgufType::Iq1S, iq1_s::ensure_loaded),
    ("iq1_m", super::kernel::GgufType::Iq1M, iq1_m::ensure_loaded),
    (
        "iq1_kt",
        super::kernel::GgufType::Iq1Kt,
        iq1_kt::ensure_loaded,
    ),
    (
        "iq2_kt",
        super::kernel::GgufType::Iq2Kt,
        iq2_kt::ensure_loaded,
    ),
    (
        "iq3_kt",
        super::kernel::GgufType::Iq3Kt,
        iq3_kt::ensure_loaded,
    ),
    (
        "iq4_kt",
        super::kernel::GgufType::Iq4Kt,
        iq4_kt::ensure_loaded,
    ),
    ("iq2_k", super::kernel::GgufType::Iq2K, iq2_k::ensure_loaded),
    ("iq3_k", super::kernel::GgufType::Iq3K, iq3_k::ensure_loaded),
    ("iq4_k", super::kernel::GgufType::Iq4K, iq4_k::ensure_loaded),
    ("iq5_k", super::kernel::GgufType::Iq5K, iq5_k::ensure_loaded),
    ("iq6_k", super::kernel::GgufType::Iq6K, iq6_k::ensure_loaded),
    (
        "iq4_ks",
        super::kernel::GgufType::Iq4Ks,
        iq4_ks::ensure_loaded,
    ),
    (
        "iq2_ks",
        super::kernel::GgufType::Iq2Ks,
        iq2_ks::ensure_loaded,
    ),
    (
        "iq4_kss",
        super::kernel::GgufType::Iq4Kss,
        iq4_kss::ensure_loaded,
    ),
    (
        "iq5_ks",
        super::kernel::GgufType::Iq5Ks,
        iq5_ks::ensure_loaded,
    ),
    (
        "iq3_ks",
        super::kernel::GgufType::Iq3Ks,
        iq3_ks::ensure_loaded,
    ),
    (
        "iq2_kl",
        super::kernel::GgufType::Iq2Kl,
        iq2_kl::ensure_loaded,
    ),
];

/// Opens GGUF type `ty`'s kernel library now, so a missing one fails at model load; float and Q8 types have none.
pub(crate) fn ensure_type_loaded(ty: super::kernel::GgufType) -> Result<(), String> {
    match TYPE_LIBRARIES.iter().find(|(_, of, _)| *of == ty) {
        Some((_, _, open)) => open(),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    // the build gives every `kernels/cuda/*/*_instance_<type>.cu` type a library; the table must name each one
    #[test]
    fn every_type_with_instance_files_is_in_the_table() {
        let mut built = std::collections::BTreeSet::new();
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/cuda");
        for dir in std::fs::read_dir(root).unwrap() {
            let dir = dir.unwrap().path();
            if !dir.is_dir() {
                continue;
            }
            for file in std::fs::read_dir(dir).unwrap() {
                let path = file.unwrap().path();
                let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
                if path.extension().is_some_and(|e| e == "cu")
                    && let Some((_, ty)) = stem.split_once("_instance_")
                {
                    built.insert(ty.to_string());
                }
            }
        }
        let listed: std::collections::BTreeSet<String> = super::TYPE_LIBRARIES
            .iter()
            .map(|(name, _, _)| name.to_string())
            .collect();
        assert_eq!(built, listed);
    }
}
