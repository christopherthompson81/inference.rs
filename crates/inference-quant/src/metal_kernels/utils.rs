// Portions of this file are adapted from Apple's MLX framework
// (https://github.com/ml-explore/mlx)
// Licensed under the Apache License 2.0
// Copyright © 2023 Apple Inc.

use candle_metal_kernels::metal::{
    Buffer, CommandBuffer, CommandsGuard, ComputeCommandEncoder, ComputePipeline,
};
use objc2_metal::MTLSize;
use std::ffi::c_void;

pub(crate) fn get_2d_grid_dims(shape: &[usize], strides: &[usize]) -> MTLSize {
    let mut grid_x: usize = 1;
    let mut grid_y: usize = 1;

    for i in 0..shape.len() {
        if strides[i] == 0 {
            continue;
        }
        if grid_x.saturating_mul(shape[i]) < u32::MAX as usize {
            grid_x *= shape[i];
        } else {
            grid_y *= shape[i];
        }
    }

    if grid_y > u32::MAX as usize || grid_x > u32::MAX as usize {
        panic!("Unable to safely factor shape.");
    }

    if grid_y > grid_x {
        std::mem::swap(&mut grid_x, &mut grid_y);
    }

    MTLSize {
        width: grid_x,
        height: grid_y,
        depth: 1,
    }
}

/// Most kernels apply similarly across the tensors
/// This creates a strategy that uses the maximum amount of threads per threadgroup (capped at the
/// actual total buffer length).
/// Then kernels can just do their op on their single point in the buffer.
pub(crate) fn linear_split(pipeline: &ComputePipeline, length: usize) -> (MTLSize, MTLSize) {
    let size = length;
    let width = std::cmp::min(pipeline.max_total_threads_per_threadgroup(), size);
    let count = size.div_ceil(width);
    let thread_group_count = MTLSize {
        width: count,
        height: 1,
        depth: 1,
    };

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    (thread_group_count, thread_group_size)
}

/// Extension to mimic the old `set_bytes` signature used by the `metal` crate.
pub trait RawBytesEncoder {
    fn set_bytes_raw(&self, index: usize, length: usize, bytes: *const c_void);
}

impl RawBytesEncoder for ComputeCommandEncoder {
    fn set_bytes_raw(&self, index: usize, length: usize, bytes: *const c_void) {
        self.set_bytes_directly(index, length, bytes);
    }
}

pub fn set_param<P: EncoderParam>(encoder: &ComputeCommandEncoder, position: usize, data: P) {
    <P as EncoderParam>::set_param(encoder, position, data)
}

/// Helper functions to create the various objects on the compute command encoder
/// on a single line.
/// Prevents getting wrong some arguments number and mixing length and size in bytes.
pub trait EncoderParam {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self);
}
macro_rules! primitive {
    ($type:ty) => {
        impl EncoderParam for $type {
            fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
                encoder.set_bytes(position, &data);
            }
        }
    };
}
primitive!(bool);
primitive!(usize);
primitive!(i32);
primitive!(i64);
primitive!(u32);
primitive!(u64);
primitive!(f32);

pub struct BufferOffset<'a> {
    pub buffer: &'a Buffer,
    pub offset_in_bytes: usize,
}

impl<T> EncoderParam for &[T] {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_bytes_directly(position, core::mem::size_of_val(data), data.as_ptr().cast());
    }
}

impl EncoderParam for &Buffer {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_input_buffer(position, Some(data), 0);
    }
}

impl EncoderParam for (&Buffer, usize) {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_input_buffer(position, Some(data.0), data.1);
    }
}

impl EncoderParam for &BufferOffset<'_> {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_input_buffer(position, Some(data.buffer), data.offset_in_bytes);
    }
}

impl EncoderParam for &mut Buffer {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_output_buffer(position, Some(data), 0);
    }
}

impl EncoderParam for (&mut Buffer, usize) {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_output_buffer(position, Some(data.0), data.1);
    }
}

/// Wrapper for `set_params!` callers to mark a buffer slot as a kernel output
/// (writes), so hazard tracking and inter-encoder fence ordering see the write.
/// Use this whenever a kernel's `device T*` (non-const) argument is passed
/// through `set_params!`.
#[derive(Copy, Clone)]
pub struct Output<'a> {
    buffer: &'a Buffer,
    offset: usize,
}

impl<'a> Output<'a> {
    #[inline]
    pub fn new(buffer: &'a Buffer) -> Self {
        Self { buffer, offset: 0 }
    }
}

impl<'a> EncoderParam for Output<'a> {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_output_buffer(position, Some(data.buffer), data.offset);
    }
}

#[macro_export]
macro_rules! set_params {
    ($encoder:ident, ($($param:expr),+)) => (
        let mut _index = 0;
        $(
            $crate::metal_kernels::utils::set_param($encoder, _index, $param);
            _index += 1;
        )*
    );
}

pub trait EncoderProvider {
    type Encoder<'a>: AsRef<ComputeCommandEncoder>
    where
        Self: 'a;
    fn encoder(&self) -> Self::Encoder<'_>;
}

pub struct WrappedEncoder<'a> {
    inner: &'a ComputeCommandEncoder,
    end_encoding_on_drop: bool,
}

impl Drop for WrappedEncoder<'_> {
    fn drop(&mut self) {
        if self.end_encoding_on_drop {
            self.inner.end_encoding()
        }
    }
}

impl AsRef<ComputeCommandEncoder> for WrappedEncoder<'_> {
    fn as_ref(&self) -> &ComputeCommandEncoder {
        self.inner
    }
}

impl EncoderProvider for &CommandBuffer {
    type Encoder<'a>
        = ComputeCommandEncoder
    where
        Self: 'a;
    fn encoder(&self) -> Self::Encoder<'_> {
        self.compute_command_encoder_no_fence()
    }
}

impl EncoderProvider for &ComputeCommandEncoder {
    type Encoder<'a>
        = WrappedEncoder<'a>
    where
        Self: 'a;
    fn encoder(&self) -> Self::Encoder<'_> {
        WrappedEncoder {
            inner: self,
            end_encoding_on_drop: false,
        }
    }
}

impl EncoderProvider for &CommandsGuard<'_> {
    type Encoder<'a>
        = &'a CommandsGuard<'a>
    where
        Self: 'a;
    fn encoder(&self) -> Self::Encoder<'_> {
        self
    }
}
