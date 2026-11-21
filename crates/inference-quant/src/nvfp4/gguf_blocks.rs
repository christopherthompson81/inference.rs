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

    /// The weight as llama.cpp NVFP4 blocks.
    pub fn block_bytes(&self) -> Result<Vec<u8>> {
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

// MXFP8's values sharing a power-of-two scale; E8M0's 0 is 2^-127, an f32 denormal
const MXFP8_BLOCK: usize = 32;
const E8M0_ZERO_BITS: u32 = 1 << 22;

/// Where a GGUF tensor's experts are in a ModelOpt checkpoint: the stack `[experts, rows, cols / 2]` and whether its
/// rows interleave two projections (a fused gate/up's, even rows the gate's), which GGUF stacks one after the other.
pub struct ModeloptStack {
    pub weight: String,
    pub interleaved: bool,
}

/// A GGUF tensor's stack in a checkpoint, where it has one.
pub type StackOf = Box<dyn Fn(&str) -> Option<ModeloptStack> + Send + Sync>;

/// A ModelOpt checkpoint's NVFP4 expert stacks (`<w>`, `<w>_scale`, and the per-expert second level
/// `<w>_amax / <w>_outer_scale_denominator`) as GGUF tensors, for a quantizer to write as stored.
pub struct ModeloptExperts {
    tensors: crate::safetensors::MmapedSafetensors,
    stack: StackOf,
}

impl ModeloptExperts {
    /// The checkpoint's safetensors `paths`, with `stack` naming a GGUF tensor's stack where it has one.
    ///
    /// # Safety
    ///
    /// The files are memory-mapped (see [`crate::safetensors::MmapedSafetensors::multi`]).
    pub unsafe fn open(paths: &[std::path::PathBuf], stack: StackOf) -> Result<Self> {
        // SAFETY: the caller's
        let tensors = unsafe { crate::safetensors::MmapedSafetensors::multi(paths)? };
        Ok(Self { tensors, stack })
    }

    fn scalar_f32(&self, name: &str) -> Result<Vec<f32>> {
        let view = self.tensors.get(name)?;
        Ok(view
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }
}

// the GGUF's row order of a stack's `rows`: the gate's (even) then the up's (odd) where the checkpoint interleaves them
fn gguf_row_order(rows: usize, interleaved: bool) -> Vec<usize> {
    (0..rows)
        .map(|r| match interleaved {
            true if r < rows / 2 => 2 * r,
            true => 2 * (r - rows / 2) + 1,
            false => r,
        })
        .collect()
}

impl ModeloptExperts {
    // the stack's tensor `suffix` names, where it has it
    fn has(&self, tensor: &str, suffix: &str) -> bool {
        (self.stack)(tensor).is_some_and(|stack| {
            self.tensors
                .get(&format!("{}{suffix}", stack.weight))
                .is_ok()
        })
    }
}

impl crate::requantize::CheckpointSource for ModeloptExperts {
    fn holds_nvfp4(&self, tensor: &str) -> bool {
        self.has(tensor, "_amax")
    }

    // MXFP8: E4M3 values, a power-of-two (E8M0) scale each 32, and no second level
    fn holds_values(&self, tensor: &str) -> bool {
        self.has(tensor, "_scale") && !self.has(tensor, "_amax")
    }

    fn values(&self, tensor: &str) -> Result<Vec<f32>> {
        use rayon::prelude::*;
        let stack = (self.stack)(tensor)
            .ok_or_else(|| inference_tensor::Error::msg("not an expert stack"))?;
        let weight = self.tensors.get(&stack.weight)?;
        let scales = self.tensors.get(&format!("{}_scale", stack.weight))?;
        let &[experts, rows, cols] = weight.shape() else {
            inference_tensor::bail!("an expert stack of shape {:?}", weight.shape());
        };
        let scale_cols = cols / MXFP8_BLOCK;
        if weight.dtype() != safetensors::Dtype::F8_E4M3
            || scales.dtype() != safetensors::Dtype::U8
            || scales.shape() != [experts, rows, scale_cols]
        {
            inference_tensor::bail!(
                "an MXFP8 stack of {:?} {:?} with {:?} scales {:?}",
                weight.dtype(),
                weight.shape(),
                scales.dtype(),
                scales.shape()
            );
        }
        let (codes, exponents) = (weight.data(), scales.data());
        let e4m3 = (0..=255u8)
            .map(|b| float8::F8E4M3::from_bits(b).to_f32())
            .collect::<Vec<_>>();
        let order = gguf_row_order(rows, stack.interleaved);
        let mut values = vec![0f32; experts * rows * cols];
        values
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(out_row, out)| {
                let (e, r) = (out_row / rows, order[out_row % rows]);
                let row = e * rows + r;
                let codes = &codes[row * cols..(row + 1) * cols];
                let exponents = &exponents[row * scale_cols..(row + 1) * scale_cols];
                for (block, (out, codes)) in out
                    .chunks_mut(MXFP8_BLOCK)
                    .zip(codes.chunks(MXFP8_BLOCK))
                    .enumerate()
                {
                    let scale = match exponents[block] {
                        0 => f32::from_bits(E8M0_ZERO_BITS),
                        e => f32::from_bits(u32::from(e) << 23),
                    };
                    for (o, &c) in out.iter_mut().zip(codes) {
                        *o = e4m3[usize::from(c)] * scale;
                    }
                }
            });
        Ok(values)
    }

    fn read_nvfp4(&self, tensor: &str) -> Result<(Vec<u8>, Vec<f32>)> {
        let stack = (self.stack)(tensor)
            .ok_or_else(|| inference_tensor::Error::msg("not an NVFP4 expert stack"))?;
        let weight = self.tensors.get(&stack.weight)?;
        let scales = self.tensors.get(&format!("{}_scale", stack.weight))?;
        let &[experts, rows, half] = weight.shape() else {
            inference_tensor::bail!("an expert stack of shape {:?}", weight.shape());
        };
        let cols = half * 2;
        let scale_cols = cols / NVFP4_BLOCK_SIZE;
        if scales.shape() != [experts, rows, scale_cols] {
            inference_tensor::bail!("block scales of shape {:?}", scales.shape());
        }
        let (packed, block_scales) = (weight.data(), scales.data());
        let order = gguf_row_order(rows, stack.interleaved);
        let mut blocks = Vec::new();
        let (mut expert_packed, mut expert_scales) = (Vec::new(), Vec::new());
        for e in 0..experts {
            expert_packed.clear();
            expert_scales.clear();
            for &r in &order {
                let row = e * rows + r;
                expert_packed.extend_from_slice(&packed[row * half..(row + 1) * half]);
                expert_scales
                    .extend_from_slice(&block_scales[row * scale_cols..(row + 1) * scale_cols]);
            }
            blocks.extend(modelopt_to_gguf_blocks(
                &expert_packed,
                &expert_scales,
                rows,
                cols,
            )?);
        }
        let amax = self.scalar_f32(&format!("{}_amax", stack.weight))?;
        let denominator = self.scalar_f32(&format!("{}_outer_scale_denominator", stack.weight))?;
        let &[denominator] = denominator.as_slice() else {
            inference_tensor::bail!("an outer scale denominator of {} values", denominator.len());
        };
        if amax.len() != experts {
            inference_tensor::bail!("{} amax values for {experts} experts", amax.len());
        }
        Ok((blocks, amax.iter().map(|a| a / denominator).collect()))
    }
}

#[cfg(test)]
mod tests {
    use inference_tensor::{DType, Device, Result, Tensor};

    use super::{GGUF_BLOCK_VALUES, ModeloptNvfp4Weight, gguf_nvfp4_linear};
    use crate::NVFP4_BLOCK_SIZE;
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

    // a fused gate/up stack's interleaved rows come out the gate's then the up's, each expert's blocks after the last's,
    // with its scale the stack's amax over the denominator
    #[test]
    fn modelopt_experts_repack_their_stacks_with_their_scales() -> Result<()> {
        use crate::requantize::CheckpointSource;
        let (experts, rows) = (2, 4);
        let packed = (0..experts * rows * COLS / 2)
            .map(|i| (i * 29) as u8)
            .collect::<Vec<_>>();
        let scales = (0..experts * rows * COLS / NVFP4_BLOCK_SIZE)
            .map(scale_byte)
            .collect::<Vec<_>>();
        let dev = &Device::Cpu;
        let tensors = std::collections::HashMap::from([
            (
                "w13".to_string(),
                Tensor::from_raw_buffer(&packed, DType::U8, &[experts, rows, COLS / 2], dev)?,
            ),
            (
                "w13_scale".to_string(),
                Tensor::from_raw_buffer(
                    &scales,
                    DType::F8E4M3,
                    &[experts, rows, COLS / NVFP4_BLOCK_SIZE],
                    dev,
                )?,
            ),
            ("w13_amax".to_string(), Tensor::new(&[3.0f32, 6.0], dev)?),
            (
                "w13_outer_scale_denominator".to_string(),
                Tensor::new(&[1.5f32], dev)?,
            ),
        ]);
        let dir = tempfile::tempdir().map_err(inference_tensor::Error::wrap)?;
        let path = dir.path().join("experts.safetensors");
        inference_tensor::safetensors::save(&tensors, &path)?;
        // SAFETY: the file is only read
        let source = unsafe {
            super::ModeloptExperts::open(
                &[path],
                Box::new(|tensor| {
                    (tensor == "stack").then(|| super::ModeloptStack {
                        weight: "w13".to_string(),
                        interleaved: true,
                    })
                }),
            )?
        };
        assert!(source.holds_nvfp4("stack") && !source.holds_nvfp4("other"));
        let (blocks, expert_scales) = source.read_nvfp4("stack")?;
        assert_eq!(expert_scales, [2.0, 4.0]);
        let (half, scale_cols) = (COLS / 2, COLS / NVFP4_BLOCK_SIZE);
        let mut expected = Vec::new();
        for e in 0..experts {
            let (mut p, mut s) = (Vec::new(), Vec::new());
            for r in [0, 2, 1, 3] {
                let row = e * rows + r;
                p.extend_from_slice(&packed[row * half..(row + 1) * half]);
                s.extend_from_slice(&scales[row * scale_cols..(row + 1) * scale_cols]);
            }
            expected.extend(super::modelopt_to_gguf_blocks(&p, &s, rows, COLS)?);
        }
        assert_eq!(blocks, expected);
        Ok(())
    }

    // an MXFP8 stack's values are its E4M3 codes times their block's power of two (E8M0's 0 the denormal 2^-127), its
    // interleaved rows the gate's then the up's
    #[test]
    fn modelopt_mxfp8_stacks_read_as_their_values() -> Result<()> {
        use crate::requantize::CheckpointSource;
        const MX_COLS: usize = 64;
        let (experts, rows) = (2, 4);
        let codes = (0..experts * rows * MX_COLS)
            .map(|i| (i * 13 % 0x7e) as u8 | if i % 3 == 0 { 0x80 } else { 0 })
            .collect::<Vec<_>>();
        let exponents = (0..experts * rows * MX_COLS / 32)
            .map(|i| if i == 1 { 0 } else { 120 + (i % 9) as u8 })
            .collect::<Vec<_>>();
        let dev = &Device::Cpu;
        let tensors = std::collections::HashMap::from([
            (
                "w13".to_string(),
                Tensor::from_raw_buffer(&codes, DType::F8E4M3, &[experts, rows, MX_COLS], dev)?,
            ),
            (
                "w13_scale".to_string(),
                Tensor::from_raw_buffer(
                    &exponents,
                    DType::U8,
                    &[experts, rows, MX_COLS / 32],
                    dev,
                )?,
            ),
        ]);
        let dir = tempfile::tempdir().map_err(inference_tensor::Error::wrap)?;
        let path = dir.path().join("experts.safetensors");
        inference_tensor::safetensors::save(&tensors, &path)?;
        // SAFETY: the file is only read
        let source = unsafe {
            super::ModeloptExperts::open(
                &[path],
                Box::new(|tensor| {
                    (tensor == "stack").then(|| super::ModeloptStack {
                        weight: "w13".to_string(),
                        interleaved: true,
                    })
                }),
            )?
        };
        assert!(source.holds_values("stack") && !source.holds_nvfp4("stack"));
        let values = source.values("stack")?;
        let mut expected = Vec::new();
        for e in 0..experts {
            for r in [0, 2, 1, 3] {
                let row = e * rows + r;
                for c in 0..MX_COLS {
                    let scale = 2f32.powi(i32::from(exponents[row * MX_COLS / 32 + c / 32]) - 127);
                    expected
                        .push(float8::F8E4M3::from_bits(codes[row * MX_COLS + c]).to_f32() * scale);
                }
            }
        }
        assert_eq!(values, expected);
        Ok(())
    }
}
