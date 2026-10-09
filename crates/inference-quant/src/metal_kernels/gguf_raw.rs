use super::*;

// Values one threadgroup (dequantize) or one simdgroup step (matvec) covers: 32 lanes of 8
const STEP: usize = 256;
const SIMD_WIDTH: usize = 32;
// Weight rows per matvec threadgroup, one per simdgroup (MV_SIMDGROUPS in gguf_raw.metal)
const MV_ROWS_PER_GROUP: usize = 4;
/// Activation rows one matvec launch takes (MV_MAX_BATCH in gguf_raw.metal).
pub const GGUF_RAW_MV_MAX_BATCH: usize = 8;

/// A raw GGUF weight's blocks on Metal: `rows` rows of `cols` values, `row_bytes` apart; `ty` names its decoder.
pub struct GgufRawBlocks<'a> {
    pub ty: &'static str,
    pub buffer: &'a Buffer,
    pub offset: usize,
    pub rows: usize,
    pub cols: usize,
    pub row_bytes: usize,
}

fn dtype_tag(dtype: DType) -> Result<&'static str, MetalKernelError> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        other => Err(MetalKernelError::DTypeMismatch {
            expected: vec![DType::F32, DType::F16, DType::BF16],
            got: other,
        }),
    }
}

fn grid(width: usize, height: usize) -> MTLSize {
    MTLSize {
        width,
        height,
        depth: 1,
    }
}

/// `w` dequantized to `dtype` as `[rows, cols]`, or only the rows `ids` (u32, `n` of them) names as `[n, cols]`.
#[allow(clippy::too_many_arguments)]
pub fn call_gguf_raw_dequant(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    w: &GgufRawBlocks,
    dtype: DType,
    ids: Option<(&Buffer, usize, usize)>,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let tag = dtype_tag(dtype)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    let rows = match ids {
        Some((ids, ids_offset, n)) => {
            let pipeline =
                kernels.load_pipeline(device, format!("gguf_raw_get_rows_{}_{tag}", w.ty))?;
            encoder.set_compute_pipeline_state(&pipeline);
            set_params!(
                encoder,
                (
                    (w.buffer, w.offset),
                    (ids, ids_offset),
                    Output::new(output),
                    w.cols as u32,
                    w.row_bytes as u64,
                    w.rows as u32
                )
            );
            n
        }
        None => {
            let pipeline =
                kernels.load_pipeline(device, format!("gguf_raw_dequant_{}_{tag}", w.ty))?;
            encoder.set_compute_pipeline_state(&pipeline);
            set_params!(
                encoder,
                (
                    (w.buffer, w.offset),
                    Output::new(output),
                    w.cols as u32,
                    w.row_bytes as u64
                )
            );
            w.rows
        }
    };
    encoder.dispatch_thread_groups(grid(w.cols.div_ceil(STEP), rows), grid(SIMD_WIDTH, 1));
    Ok(())
}

/// `output[b, r] = x[b] . w[r]` for `batch` (at most [`GGUF_RAW_MV_MAX_BATCH`]) contiguous rows of `x` in `dtype`.
#[allow(clippy::too_many_arguments)]
pub fn call_gguf_raw_mv(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    w: &GgufRawBlocks,
    dtype: DType,
    x: &Buffer,
    x_offset: usize,
    batch: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let tag = dtype_tag(dtype)?;
    let pipeline = kernels.load_pipeline(device, format!("gguf_raw_mv_{}_{tag}", w.ty))?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            (w.buffer, w.offset),
            (x, x_offset),
            Output::new(output),
            w.cols as u32,
            w.row_bytes as u64,
            w.rows as u32,
            batch as u32
        )
    );
    encoder.dispatch_thread_groups(
        grid(w.rows.div_ceil(MV_ROWS_PER_GROUP), 1),
        grid(SIMD_WIDTH * MV_ROWS_PER_GROUP, 1),
    );
    Ok(())
}
