//! ModelOpt NVFP4 (two E2M1 values a byte, an E4M3 scale per 16 values, an F32 scale per tensor) in llama.cpp's NVFP4
//! blocks, which the GGUF matvec and MMQ kernels read on any CUDA arch: the same codes and scale bytes, reordered.

use std::sync::Arc;

use inference_tensor::{DType, Device, Result, Tensor};

use crate::gguf::kernel::GgufType;
use crate::gguf::raw::{GgufRawMatMul, RawGgufTensor};
use crate::{NVFP4_BLOCK_SIZE, QuantMethod};

// a llama.cpp NVFP4 block: four 16-value sub-blocks, their scale bytes ahead of their nibbles
const GGUF_BLOCK_VALUES: usize = 64;
const GGUF_SUB_BLOCKS: usize = GGUF_BLOCK_VALUES / NVFP4_BLOCK_SIZE;
const SUB_BLOCK_BYTES: usize = NVFP4_BLOCK_SIZE / 2;

/// Reorders a `[rows, cols]` ModelOpt NVFP4 weight (`packed` `[rows, cols / 2]`, `scales` `[rows, cols / 16]` E4M3
/// bytes) into llama.cpp NVFP4 blocks. ModelOpt packs adjacent values; a llama.cpp sub-block's byte `j` holds values
/// `j` and `j + 8`.
pub fn modelopt_to_gguf_blocks(
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Vec<u8>> {
    if !cols.is_multiple_of(GGUF_BLOCK_VALUES) {
        inference_tensor::bail!(
            "NVFP4 rows of {cols} values are not whole {GGUF_BLOCK_VALUES}-value blocks"
        );
    }
    if packed.len() != rows * cols / 2 || scales.len() != rows * cols / NVFP4_BLOCK_SIZE {
        inference_tensor::bail!(
            "NVFP4 [{rows}, {cols}] needs {} packed and {} scale bytes, got {} and {}",
            rows * cols / 2,
            rows * cols / NVFP4_BLOCK_SIZE,
            packed.len(),
            scales.len()
        );
    }
    let nibble = |value: usize| {
        let byte = packed[value / 2];
        if value.is_multiple_of(2) {
            byte & 0x0F
        } else {
            byte >> 4
        }
    };
    let sub_blocks = rows * cols / NVFP4_BLOCK_SIZE;
    let mut out = Vec::with_capacity(sub_blocks * (1 + SUB_BLOCK_BYTES));
    for block in 0..rows * cols / GGUF_BLOCK_VALUES {
        let first_sub = block * GGUF_SUB_BLOCKS;
        out.extend_from_slice(&scales[first_sub..first_sub + GGUF_SUB_BLOCKS]);
        for sub in first_sub..first_sub + GGUF_SUB_BLOCKS {
            let base = sub * NVFP4_BLOCK_SIZE;
            for j in 0..SUB_BLOCK_BYTES {
                out.push(nibble(base + j) | (nibble(base + j + SUB_BLOCK_BYTES) << 4));
            }
        }
    }
    Ok(out)
}

/// A ModelOpt NVFP4 projection's tensors: `weight` U8 `[rows, cols / 2]`, `scales` F8E4M3 `[rows, cols / 16]`.
pub struct ModeloptNvfp4Weight {
    pub weight: Tensor,
    pub scales: Tensor,
    pub global_scale: f32,
}

impl ModeloptNvfp4Weight {
    fn rows_cols(&self) -> Result<(usize, usize)> {
        let (rows, half) = self.weight.dims2()?;
        Ok((rows, half * 2))
    }

    fn block_bytes(&self) -> Result<Vec<u8>> {
        let (rows, cols) = self.rows_cols()?;
        if self.weight.dtype() != DType::U8 || self.scales.dtype() != DType::F8E4M3 {
            inference_tensor::bail!(
                "ModelOpt NVFP4 expects U8 weights and F8E4M3 scales, got {:?} and {:?}",
                self.weight.dtype(),
                self.scales.dtype()
            );
        }
        let packed = self.weight.flatten_all()?.to_vec1::<u8>()?;
        let scales = self
            .scales
            .flatten_all()?
            .to_vec1::<float8::F8E4M3>()?
            .into_iter()
            .map(|scale| scale.to_bits())
            .collect::<Vec<_>>();
        modelopt_to_gguf_blocks(&packed, &scales, rows, cols)
    }
}

/// A GGUF NVFP4 matmul over `parts` stacked by rows (one projection, or several fused). The layer applies the first
/// part's tensor scale to every output, so the later parts' ratios to it are returned for the caller to fold in.
pub fn gguf_nvfp4_linear(
    parts: &[ModeloptNvfp4Weight],
    device: &Device,
) -> Result<(Arc<dyn QuantMethod>, Vec<f32>)> {
    let Some(first) = parts.first() else {
        inference_tensor::bail!("an NVFP4 projection needs at least one weight");
    };
    let cols = first.rows_cols()?.1;
    let mut bytes = Vec::new();
    let mut rows = 0;
    for part in parts {
        let (part_rows, part_cols) = part.rows_cols()?;
        if part_cols != cols {
            inference_tensor::bail!("fused NVFP4 parts have {part_cols} and {cols} columns");
        }
        bytes.extend(part.block_bytes()?);
        rows += part_rows;
    }
    let raw = RawGgufTensor::new(GgufType::Nvfp4, &[rows, cols], bytes, device)?;
    let linear = GgufRawMatMul::new(raw, None).with_output_scale(Some(first.global_scale));
    let ratios = parts[1..]
        .iter()
        .map(|part| part.global_scale / first.global_scale)
        .collect();
    Ok((Arc::new(linear), ratios))
}

#[cfg(test)]
mod tests {
    use inference_tensor::{DType, Device, Result, Tensor};

    use super::{GGUF_BLOCK_VALUES, ModeloptNvfp4Weight, gguf_nvfp4_linear};
    use crate::gguf::kernel::GgufType;
    use crate::gguf::raw::RawGgufTensor;

    const ROWS: usize = 3;
    const COLS: usize = 2 * GGUF_BLOCK_VALUES;
    const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

    // positive, finite E4M3 bytes (ModelOpt's block scales never set the sign bit)
    fn scale_byte(i: usize) -> u8 {
        u8::try_from(1 + (i * 37) % 0x7E).unwrap()
    }

    // ModelOpt's decode: value i is byte i / 2's low nibble when even, times its 16-value group's scale
    fn modelopt_values(packed: &[u8], scales: &[u8], global: f32) -> Vec<f32> {
        (0..packed.len() * 2)
            .map(|i| {
                let byte = packed[i / 2];
                let code = if i.is_multiple_of(2) {
                    byte & 0x0F
                } else {
                    byte >> 4
                };
                let magnitude = E2M1[usize::from(code & 7)];
                let value = if code & 8 != 0 { -magnitude } else { magnitude };
                value * float8::F8E4M3::from_bits(scales[i / 16]).to_f32() * global
            })
            .collect()
    }

    #[test]
    fn llama_cpp_blocks_decode_to_modelopt_values() -> Result<()> {
        let packed = (0..ROWS * COLS / 2)
            .map(|i| u8::try_from((i * 73 + 11) % 256).unwrap())
            .collect::<Vec<_>>();
        let scales = (0..ROWS * COLS / 16).map(scale_byte).collect::<Vec<_>>();
        let blocks = super::modelopt_to_gguf_blocks(&packed, &scales, ROWS, COLS)?;
        let gguf = RawGgufTensor::new(GgufType::Nvfp4, &[ROWS, COLS], blocks, &Device::Cpu)?
            .dequantize(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(gguf, modelopt_values(&packed, &scales, 1.0));
        Ok(())
    }

    // two parts fused by rows, the second's tensor scale returned relative to the first's
    #[test]
    fn fused_parts_keep_their_rows_and_scale_ratios() -> Result<()> {
        let part = |seed: usize, global: f32| -> Result<ModeloptNvfp4Weight> {
            let packed = (0..ROWS * COLS / 2)
                .map(|i| u8::try_from((i * 29 + seed) % 256).unwrap())
                .collect::<Vec<_>>();
            let scales = (0..ROWS * COLS / 16)
                .map(|i| scale_byte(i + seed))
                .map(float8::F8E4M3::from_bits)
                .collect::<Vec<_>>();
            Ok(ModeloptNvfp4Weight {
                weight: Tensor::from_vec(packed, (ROWS, COLS / 2), &Device::Cpu)?,
                scales: Tensor::from_vec(scales, (ROWS, COLS / 16), &Device::Cpu)?,
                global_scale: global,
            })
        };
        let parts = [part(3, 0.5)?, part(9, 0.25)?];
        let (linear, ratios) = gguf_nvfp4_linear(&parts, &Device::Cpu)?;
        assert_eq!(ratios, vec![0.5]);
        let weight = linear.dequantize_w()?.to_dtype(DType::F32)?;
        assert_eq!(weight.dims(), &[2 * ROWS, COLS]);
        let expected = parts
            .iter()
            .map(|part| -> Result<Vec<f32>> {
                let packed = part.weight.flatten_all()?.to_vec1::<u8>()?;
                let scales = part
                    .scales
                    .flatten_all()?
                    .to_vec1::<float8::F8E4M3>()?
                    .into_iter()
                    .map(|scale| scale.to_bits())
                    .collect::<Vec<_>>();
                // the layer applies the first part's tensor scale to every row
                Ok(modelopt_values(&packed, &scales, parts[0].global_scale))
            })
            .collect::<Result<Vec<_>>>()?
            .concat();
        assert_eq!(weight.flatten_all()?.to_vec1::<f32>()?, expected);
        Ok(())
    }
}
