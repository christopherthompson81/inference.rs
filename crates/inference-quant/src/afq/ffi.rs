//! FFI bindings for AFQ CUDA kernels.
//!
//! These bindings allow Rust code to call the optimized CUDA kernels
//! for AFQ quantization, dequantization, and matrix operations.

#![allow(dead_code)]
#![allow(improper_ctypes)]

use half::{bf16, f16};

// Dequantize kernel bindings

macro_rules! dequant_kernel_power_of_2_kernels {
    ($( ($bits:tt, $gs:tt, $scalar:ty, $postfix:tt) ),* $(,)?) => {
        paste::paste! {
        inference_tensor::kernel_library! {
            library = "inferencequant";
            path = option_env!("INFERENCE_KERNEL_LIB_INFERENCEQUANT").unwrap_or("");
            id = option_env!("INFERENCE_KERNEL_ID_INFERENCEQUANT").unwrap_or("");
            unsafe extern "C" {
                $(
                pub fn [< afq_dequantize_ $bits bit_gs $gs _ $postfix >](
                w_q: *const u32,
                scales: *const $scalar,
                biases: *const $scalar,
                output: *mut $scalar,
                rows: i32,
                cols: i32,
                );
                )*
            }
        }
        }
    };
}

macro_rules! dequant_kernel_3bit_kernels {
    ($( ($gs:tt, $scalar:ty, $postfix:tt) ),* $(,)?) => {
        paste::paste! {
        inference_tensor::kernel_library! {
            library = "inferencequant";
            path = option_env!("INFERENCE_KERNEL_LIB_INFERENCEQUANT").unwrap_or("");
            id = option_env!("INFERENCE_KERNEL_ID_INFERENCEQUANT").unwrap_or("");
            unsafe extern "C" {
                $(
                pub fn [< afq_dequantize_3bit_gs $gs _ $postfix >](
                w_q: *const u8,
                scales: *const $scalar,
                biases: *const $scalar,
                output: *mut $scalar,
                rows: i32,
                cols: i32,
                );
                )*
            }
        }
        }
    };
}

macro_rules! dequant_kernel_6bit_kernels {
    ($( ($gs:tt, $scalar:ty, $postfix:tt) ),* $(,)?) => {
        paste::paste! {
        inference_tensor::kernel_library! {
            library = "inferencequant";
            path = option_env!("INFERENCE_KERNEL_LIB_INFERENCEQUANT").unwrap_or("");
            id = option_env!("INFERENCE_KERNEL_ID_INFERENCEQUANT").unwrap_or("");
            unsafe extern "C" {
                $(
                pub fn [< afq_dequantize_6bit_gs $gs _ $postfix >](
                w_q: *const u8,
                scales: *const $scalar,
                biases: *const $scalar,
                output: *mut $scalar,
                rows: i32,
                cols: i32,
                );
                )*
            }
        }
        }
    };
}

// Embedding kernel bindings

macro_rules! embedding_kernel_kernels {
    ($( ($bits:tt, $gs:tt, $scalar:ty, $postfix:tt) ),* $(,)?) => {
        paste::paste! {
        inference_tensor::kernel_library! {
            library = "inferencequant";
            path = option_env!("INFERENCE_KERNEL_LIB_INFERENCEQUANT").unwrap_or("");
            id = option_env!("INFERENCE_KERNEL_ID_INFERENCEQUANT").unwrap_or("");
            unsafe extern "C" {
                $(
                pub fn [< afq_embedding_ $bits bit_gs $gs _ $postfix >](
                w_q: *const u8,
                scales: *const $scalar,
                biases: *const $scalar,
                ids: *const u32,
                output: *mut $scalar,
                num_ids: i32,
                hidden: i32,
                );
                )*
            }
        }
        }
    };
}

// Quantize kernel bindings

macro_rules! quant_kernel_kernels {
    ($( ($bits:tt, $gs:tt, $scalar:ty, $postfix:tt) ),* $(,)?) => {
        paste::paste! {
        inference_tensor::kernel_library! {
            library = "inferencequant";
            path = option_env!("INFERENCE_KERNEL_LIB_INFERENCEQUANT").unwrap_or("");
            id = option_env!("INFERENCE_KERNEL_ID_INFERENCEQUANT").unwrap_or("");
            unsafe extern "C" {
                $(
                pub fn [< afq_quantize_ $bits bit_gs $gs _ $postfix >](
                w: *const $scalar,
                w_q: *mut u32,
                scales: *mut $scalar,
                biases: *mut $scalar,
                rows: i32,
                cols: i32,
                );
                )*
            }
        }
        }
    };
}

// QMV (quantized matrix-vector) kernel bindings

macro_rules! qmv_kernel_power_of_2_kernels {
    ($( ($bits:tt, $gs:tt, $scalar:ty, $postfix:tt) ),* $(,)?) => {
        paste::paste! {
        inference_tensor::kernel_library! {
            library = "inferencequant";
            path = option_env!("INFERENCE_KERNEL_LIB_INFERENCEQUANT").unwrap_or("");
            id = option_env!("INFERENCE_KERNEL_ID_INFERENCEQUANT").unwrap_or("");
            unsafe extern "C" {
                $(
                pub fn [< afq_qmv_ $bits bit_gs $gs _ $postfix >](
                x: *const $scalar,
                w_q: *const u32,
                scales: *const $scalar,
                biases: *const $scalar,
                y: *mut $scalar,
                m: i32,
                n: i32,
                k: i32,
                );
                )*
            }
        }
        }
    };
}

macro_rules! qmv_kernel_3bit_kernels {
    ($( ($gs:tt, $scalar:ty, $postfix:tt) ),* $(,)?) => {
        paste::paste! {
        inference_tensor::kernel_library! {
            library = "inferencequant";
            path = option_env!("INFERENCE_KERNEL_LIB_INFERENCEQUANT").unwrap_or("");
            id = option_env!("INFERENCE_KERNEL_ID_INFERENCEQUANT").unwrap_or("");
            unsafe extern "C" {
                $(
                pub fn [< afq_qmv_3bit_gs $gs _ $postfix >](
                x: *const $scalar,
                w_q: *const u8,
                scales: *const $scalar,
                biases: *const $scalar,
                y: *mut $scalar,
                m: i32,
                n: i32,
                k: i32,
                );
                )*
            }
        }
        }
    };
}

macro_rules! qmv_kernel_6bit_kernels {
    ($( ($gs:tt, $scalar:ty, $postfix:tt) ),* $(,)?) => {
        paste::paste! {
        inference_tensor::kernel_library! {
            library = "inferencequant";
            path = option_env!("INFERENCE_KERNEL_LIB_INFERENCEQUANT").unwrap_or("");
            id = option_env!("INFERENCE_KERNEL_ID_INFERENCEQUANT").unwrap_or("");
            unsafe extern "C" {
                $(
                pub fn [< afq_qmv_6bit_gs $gs _ $postfix >](
                x: *const $scalar,
                w_q: *const u8,
                scales: *const $scalar,
                biases: *const $scalar,
                y: *mut $scalar,
                m: i32,
                n: i32,
                k: i32,
                );
                )*
            }
        }
        }
    };
}

// QMM (quantized matrix-matrix) kernel bindings

macro_rules! qmm_kernel_kernels {
    ($( ($bits:tt, $gs:tt, $scalar:ty, $postfix:tt) ),* $(,)?) => {
        paste::paste! {
        inference_tensor::kernel_library! {
            library = "inferencequant";
            path = option_env!("INFERENCE_KERNEL_LIB_INFERENCEQUANT").unwrap_or("");
            id = option_env!("INFERENCE_KERNEL_ID_INFERENCEQUANT").unwrap_or("");
            unsafe extern "C" {
                $(
                pub fn [< afq_qmm_ $bits bit_gs $gs _ $postfix >](
                x: *const $scalar,
                w_q: *const u32,
                scales: *const $scalar,
                biases: *const $scalar,
                y: *mut $scalar,
                m: i32,
                n: i32,
                k: i32,
                );
                )*
            }
        }
        }
    };
}

// launchers sharing one signature
mod dequant_kernel_power_of_2 {
    use super::*;

    dequant_kernel_power_of_2_kernels!(
        (2, 32, f32, f32),
        (2, 64, f32, f32),
        (2, 128, f32, f32),
        (2, 32, f16, f16),
        (2, 64, f16, f16),
        (2, 128, f16, f16),
        (2, 32, bf16, bf16),
        (2, 64, bf16, bf16),
        (2, 128, bf16, bf16),
        (4, 32, f32, f32),
        (4, 64, f32, f32),
        (4, 128, f32, f32),
        (4, 32, f16, f16),
        (4, 64, f16, f16),
        (4, 128, f16, f16),
        (4, 32, bf16, bf16),
        (4, 64, bf16, bf16),
        (4, 128, bf16, bf16),
        (8, 32, f32, f32),
        (8, 64, f32, f32),
        (8, 128, f32, f32),
        (8, 32, f16, f16),
        (8, 64, f16, f16),
        (8, 128, f16, f16),
        (8, 32, bf16, bf16),
        (8, 64, bf16, bf16),
        (8, 128, bf16, bf16),
    );
}
#[allow(unused_imports)]
pub use dequant_kernel_power_of_2::*;

// launchers sharing one signature
mod dequant_kernel_3bit {
    use super::*;

    dequant_kernel_3bit_kernels!(
        (32, f32, f32),
        (64, f32, f32),
        (128, f32, f32),
        (32, f16, f16),
        (64, f16, f16),
        (128, f16, f16),
        (32, bf16, bf16),
        (64, bf16, bf16),
        (128, bf16, bf16),
    );
}
#[allow(unused_imports)]
pub use dequant_kernel_3bit::*;

// launchers sharing one signature
mod dequant_kernel_6bit {
    use super::*;

    dequant_kernel_6bit_kernels!(
        (32, f32, f32),
        (64, f32, f32),
        (128, f32, f32),
        (32, f16, f16),
        (64, f16, f16),
        (128, f16, f16),
        (32, bf16, bf16),
        (64, bf16, bf16),
        (128, bf16, bf16),
    );
}
#[allow(unused_imports)]
pub use dequant_kernel_6bit::*;

// launchers sharing one signature
mod embedding_kernel {
    use super::*;

    embedding_kernel_kernels!(
        (2, 32, f32, f32),
        (2, 64, f32, f32),
        (2, 128, f32, f32),
        (2, 32, f16, f16),
        (2, 64, f16, f16),
        (2, 128, f16, f16),
        (2, 32, bf16, bf16),
        (2, 64, bf16, bf16),
        (2, 128, bf16, bf16),
        (3, 32, f32, f32),
        (3, 64, f32, f32),
        (3, 128, f32, f32),
        (3, 32, f16, f16),
        (3, 64, f16, f16),
        (3, 128, f16, f16),
        (3, 32, bf16, bf16),
        (3, 64, bf16, bf16),
        (3, 128, bf16, bf16),
        (4, 32, f32, f32),
        (4, 64, f32, f32),
        (4, 128, f32, f32),
        (4, 32, f16, f16),
        (4, 64, f16, f16),
        (4, 128, f16, f16),
        (4, 32, bf16, bf16),
        (4, 64, bf16, bf16),
        (4, 128, bf16, bf16),
        (6, 32, f32, f32),
        (6, 64, f32, f32),
        (6, 128, f32, f32),
        (6, 32, f16, f16),
        (6, 64, f16, f16),
        (6, 128, f16, f16),
        (6, 32, bf16, bf16),
        (6, 64, bf16, bf16),
        (6, 128, bf16, bf16),
        (8, 32, f32, f32),
        (8, 64, f32, f32),
        (8, 128, f32, f32),
        (8, 32, f16, f16),
        (8, 64, f16, f16),
        (8, 128, f16, f16),
        (8, 32, bf16, bf16),
        (8, 64, bf16, bf16),
        (8, 128, bf16, bf16),
    );
}
#[allow(unused_imports)]
pub use embedding_kernel::*;

// launchers sharing one signature
mod quant_kernel {
    use super::*;

    quant_kernel_kernels!(
        (2, 32, f32, f32),
        (2, 64, f32, f32),
        (2, 128, f32, f32),
        (2, 32, f16, f16),
        (2, 64, f16, f16),
        (2, 128, f16, f16),
        (2, 32, bf16, bf16),
        (2, 64, bf16, bf16),
        (2, 128, bf16, bf16),
        (4, 32, f32, f32),
        (4, 64, f32, f32),
        (4, 128, f32, f32),
        (4, 32, f16, f16),
        (4, 64, f16, f16),
        (4, 128, f16, f16),
        (4, 32, bf16, bf16),
        (4, 64, bf16, bf16),
        (4, 128, bf16, bf16),
        (8, 32, f32, f32),
        (8, 64, f32, f32),
        (8, 128, f32, f32),
        (8, 32, f16, f16),
        (8, 64, f16, f16),
        (8, 128, f16, f16),
        (8, 32, bf16, bf16),
        (8, 64, bf16, bf16),
        (8, 128, bf16, bf16),
    );
}
#[allow(unused_imports)]
pub use quant_kernel::*;

// launchers sharing one signature
mod qmv_kernel_power_of_2 {
    use super::*;

    qmv_kernel_power_of_2_kernels!(
        (2, 32, f32, f32),
        (2, 64, f32, f32),
        (2, 128, f32, f32),
        (2, 32, f16, f16),
        (2, 64, f16, f16),
        (2, 128, f16, f16),
        (2, 32, bf16, bf16),
        (2, 64, bf16, bf16),
        (2, 128, bf16, bf16),
        (4, 32, f32, f32),
        (4, 64, f32, f32),
        (4, 128, f32, f32),
        (4, 32, f16, f16),
        (4, 64, f16, f16),
        (4, 128, f16, f16),
        (4, 32, bf16, bf16),
        (4, 64, bf16, bf16),
        (4, 128, bf16, bf16),
        (8, 32, f32, f32),
        (8, 64, f32, f32),
        (8, 128, f32, f32),
        (8, 32, f16, f16),
        (8, 64, f16, f16),
        (8, 128, f16, f16),
        (8, 32, bf16, bf16),
        (8, 64, bf16, bf16),
        (8, 128, bf16, bf16),
    );
}
#[allow(unused_imports)]
pub use qmv_kernel_power_of_2::*;

// launchers sharing one signature
mod qmv_kernel_3bit {
    use super::*;

    qmv_kernel_3bit_kernels!(
        (32, f32, f32),
        (64, f32, f32),
        (128, f32, f32),
        (32, f16, f16),
        (64, f16, f16),
        (128, f16, f16),
        (32, bf16, bf16),
        (64, bf16, bf16),
        (128, bf16, bf16),
    );
}
#[allow(unused_imports)]
pub use qmv_kernel_3bit::*;

// launchers sharing one signature
mod qmv_kernel_6bit {
    use super::*;

    qmv_kernel_6bit_kernels!(
        (32, f32, f32),
        (64, f32, f32),
        (128, f32, f32),
        (32, f16, f16),
        (64, f16, f16),
        (128, f16, f16),
        (32, bf16, bf16),
        (64, bf16, bf16),
        (128, bf16, bf16),
    );
}
#[allow(unused_imports)]
pub use qmv_kernel_6bit::*;

// launchers sharing one signature
mod qmm_kernel {
    use super::*;

    qmm_kernel_kernels!(
        (2, 32, f32, f32),
        (2, 64, f32, f32),
        (2, 128, f32, f32),
        (2, 32, f16, f16),
        (2, 64, f16, f16),
        (2, 128, f16, f16),
        (2, 32, bf16, bf16),
        (2, 64, bf16, bf16),
        (2, 128, bf16, bf16),
        (4, 32, f32, f32),
        (4, 64, f32, f32),
        (4, 128, f32, f32),
        (4, 32, f16, f16),
        (4, 64, f16, f16),
        (4, 128, f16, f16),
        (4, 32, bf16, bf16),
        (4, 64, bf16, bf16),
        (4, 128, bf16, bf16),
        (8, 32, f32, f32),
        (8, 64, f32, f32),
        (8, 128, f32, f32),
        (8, 32, f16, f16),
        (8, 64, f16, f16),
        (8, 128, f16, f16),
        (8, 32, bf16, bf16),
        (8, 64, bf16, bf16),
        (8, 128, bf16, bf16),
    );
}
#[allow(unused_imports)]
pub use qmm_kernel::*;
