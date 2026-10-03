//! GGUF weights in formats Candle has no `QTensor` for, kept as their ggml blocks and read by our own kernels.

use std::sync::{Arc, atomic::AtomicUsize};

use candle_core::{DType, Device, Result, Shape, Tensor};
use candle_nn::Linear;

use super::kernel::GgufType;
use crate::{IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde};

// ggml's IQ4_NL / IQ4_XS codebook (`kvalues_iq4nl` in ggml-common.h)
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];
const IQ4_NL_BLOCK: usize = 32;
const IQ4_XS_BLOCK: usize = 256;
const IQ4_XS_SUB_BLOCK: usize = 32;
// Byte ranges inside block_iq4_nl / block_iq4_xs, after the leading f16 scale
const IQ4_NL_QS: std::ops::Range<usize> = 2..18;
const IQ4_XS_SCALES_H: usize = 2;
const IQ4_XS_SCALES_L: std::ops::Range<usize> = 4..8;
const IQ4_XS_QS: std::ops::Range<usize> = 8..136;
// Zeroed blocks past the last row, as Candle pads its CUDA QTensors: mmq reads whole tiles of K
#[cfg(feature = "cuda")]
const MATRIX_ROW_PADDING: usize = 512;

#[derive(Debug, Clone)]
enum RawStorage {
    Cpu(Arc<Vec<u8>>),
    #[cfg(feature = "cuda")]
    Cuda {
        blocks: Arc<candle_core::cuda_backend::cudarc::driver::CudaSlice<u8>>,
        len: usize,
        device: candle_core::CudaDevice,
    },
}

/// A 2-D weight of ggml blocks, row-major: `rows` rows of `cols / block_size` blocks each.
#[derive(Debug, Clone)]
pub struct RawGgufTensor {
    ty: GgufType,
    shape: Shape,
    storage: RawStorage,
}

impl RawGgufTensor {
    pub fn new(ty: GgufType, dims: &[usize], bytes: Vec<u8>, device: &Device) -> Result<Self> {
        let &[rows, cols] = dims else {
            candle_core::bail!("{ty:?} weights must be rank 2, got {dims:?}");
        };
        if !cols.is_multiple_of(ty.block_size()) {
            candle_core::bail!(
                "{ty:?} rows of {cols} elements are not whole {}-element blocks",
                ty.block_size()
            );
        }
        let expected = rows * cols / ty.block_size() * ty.type_size();
        if bytes.len() != expected {
            candle_core::bail!(
                "{ty:?} [{rows}, {cols}] takes {expected} bytes, got {}",
                bytes.len()
            );
        }
        let cpu = Self {
            ty,
            shape: Shape::from((rows, cols)),
            storage: RawStorage::Cpu(Arc::new(bytes)),
        };
        cpu.to_device(device)
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn device(&self) -> Device {
        match &self.storage {
            RawStorage::Cpu(_) => Device::Cpu,
            #[cfg(feature = "cuda")]
            RawStorage::Cuda { device, .. } => Device::Cuda(device.clone()),
        }
    }

    pub fn bytes(&self) -> Result<Vec<u8>> {
        match &self.storage {
            RawStorage::Cpu(bytes) => Ok(bytes.as_ref().clone()),
            #[cfg(feature = "cuda")]
            RawStorage::Cuda {
                blocks,
                len,
                device,
            } => {
                let mut bytes = device.clone_dtoh(&**blocks)?;
                bytes.truncate(*len);
                Ok(bytes)
            }
        }
    }

    pub fn to_device(&self, device: &Device) -> Result<Self> {
        if self.device().same_device(device) {
            return Ok(self.clone());
        }
        let storage = match device {
            Device::Cpu => RawStorage::Cpu(Arc::new(self.bytes()?)),
            #[cfg(feature = "cuda")]
            Device::Cuda(device) => {
                let bytes = self.bytes()?;
                let padding = MATRIX_ROW_PADDING * self.ty.type_size() / self.ty.block_size();
                let mut blocks = device.alloc_zeros::<u8>(bytes.len() + padding)?;
                device.memcpy_htod(&bytes, &mut blocks.slice_mut(..bytes.len()))?;
                RawStorage::Cuda {
                    blocks: Arc::new(blocks),
                    len: bytes.len(),
                    device: device.clone(),
                }
            }
            other => candle_core::bail!("{:?} weights are not supported on {other:?}", self.ty),
        };
        Ok(Self {
            ty: self.ty,
            shape: self.shape.clone(),
            storage,
        })
    }

    /// The weight as an F32 tensor on `device`.
    pub fn dequantize(&self, device: &Device) -> Result<Tensor> {
        let values = dequantize_blocks(self.ty, &self.bytes()?)?;
        Tensor::from_vec(values, self.shape.clone(), &Device::Cpu)?.to_device(device)
    }
}

#[cfg(feature = "cuda")]
impl super::kernel::KernelWeight for RawGgufTensor {
    fn gguf_type(&self) -> GgufType {
        self.ty
    }

    fn kernel_shape(&self) -> &Shape {
        &self.shape
    }

    fn kernel_device(&self) -> Device {
        self.device()
    }

    fn kernel_ptr<'a>(
        &'a self,
        stream: &'a candle_core::cuda_backend::cudarc::driver::CudaStream,
    ) -> Result<(
        *const u8,
        candle_core::cuda_backend::cudarc::driver::SyncOnDrop<'a>,
    )> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        let RawStorage::Cuda { blocks, .. } = &self.storage else {
            candle_core::bail!("{:?} kernels need the weight on CUDA", self.ty);
        };
        let (ptr, guard) = blocks.device_ptr(stream);
        Ok((ptr as *const u8, guard))
    }
}

/// ggml's reference dequantization (`dequantize_row_*` in ggml-quants.c) of whole blocks.
pub fn dequantize_blocks(ty: GgufType, bytes: &[u8]) -> Result<Vec<f32>> {
    let (block, size) = (ty.block_size(), ty.type_size());
    if !bytes.len().is_multiple_of(size) {
        candle_core::bail!("{ty:?} data of {} bytes is not whole blocks", bytes.len());
    }
    let mut out = vec![0f32; bytes.len() / size * block];
    for (block_bytes, values) in bytes.chunks_exact(size).zip(out.chunks_exact_mut(block)) {
        match ty {
            GgufType::Iq4Nl => dequantize_iq4_nl(block_bytes, values),
            GgufType::Iq4Xs => dequantize_iq4_xs(block_bytes, values),
            GgufType::Iq2Xxs => super::iq_dequant::iq2_xxs(block_bytes, values),
            GgufType::Iq2Xs => super::iq_dequant::iq2_xs(block_bytes, values),
            GgufType::Iq2S => super::iq_dequant::iq2_s(block_bytes, values),
            GgufType::Iq3Xxs => super::iq_dequant::iq3_xxs(block_bytes, values),
            GgufType::Iq3S => super::iq_dequant::iq3_s(block_bytes, values),
            GgufType::Iq1S => super::iq_dequant::iq1_s(block_bytes, values),
            GgufType::Iq1M => super::iq_dequant::iq1_m(block_bytes, values),
            other => candle_core::bail!("{other:?} is held by Candle, not as raw GGUF blocks"),
        }
    }
    Ok(out)
}

fn f16_at(bytes: &[u8], at: usize) -> f32 {
    half::f16::from_le_bytes([bytes[at], bytes[at + 1]]).to_f32()
}

// Each 16-byte group packs 32 values: low nibbles are the first 16, high nibbles the next 16.
fn unpack_iq4(scale: f32, qs: &[u8], values: &mut [f32]) {
    let (low, high) = values.split_at_mut(qs.len());
    for ((q, lo), hi) in qs.iter().zip(low).zip(high) {
        *lo = scale * f32::from(KVALUES_IQ4NL[usize::from(q & 0x0F)]);
        *hi = scale * f32::from(KVALUES_IQ4NL[usize::from(q >> 4)]);
    }
}

// block_iq4_nl: f16 d, 16 bytes of 4-bit indices
fn dequantize_iq4_nl(block: &[u8], values: &mut [f32]) {
    debug_assert_eq!(values.len(), IQ4_NL_BLOCK);
    unpack_iq4(f16_at(block, 0), &block[IQ4_NL_QS], values);
}

// block_iq4_xs: f16 d, u16 scales_h, 4 bytes scales_l, 128 index bytes; eight 32-value sub-blocks, 6-bit scales
fn dequantize_iq4_xs(block: &[u8], values: &mut [f32]) {
    debug_assert_eq!(values.len(), IQ4_XS_BLOCK);
    let d = f16_at(block, 0);
    let scales_h = u16::from_le_bytes([block[IQ4_XS_SCALES_H], block[IQ4_XS_SCALES_H + 1]]);
    let scales_l = &block[IQ4_XS_SCALES_L];
    let qs = &block[IQ4_XS_QS];
    let (qs, _) = qs.as_chunks::<{ IQ4_XS_SUB_BLOCK / 2 }>();
    let (values, _) = values.as_chunks_mut::<IQ4_XS_SUB_BLOCK>();
    for (ib, (qs, values)) in qs.iter().zip(values).enumerate() {
        let low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F;
        let high = ((scales_h >> (2 * ib)) & 0x03) as u8;
        let ls = i32::from(low | (high << 4));
        unpack_iq4(d * (ls - 32) as f32, qs, values);
    }
}

/// A linear layer over a raw GGUF weight.
#[derive(Debug)]
pub struct GgufRawMatMul {
    w: RawGgufTensor,
    b: Option<Tensor>,
}

impl GgufRawMatMul {
    pub fn new(w: RawGgufTensor, b: Option<Tensor>) -> Self {
        Self { w, b }
    }

    fn add_bias(&self, x: Tensor) -> Result<Tensor> {
        match &self.b {
            Some(b) => x.broadcast_add(b),
            None => Ok(x),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_forward(&self, a: &Tensor) -> Result<Option<Tensor>> {
        if !self.w.device().is_cuda() || !matches!(a.dtype(), DType::BF16 | DType::F16 | DType::F32)
        {
            return Ok(None);
        }
        let flat_batch = a.dims()[..a.rank().saturating_sub(1)]
            .iter()
            .product::<usize>();
        let out = match flat_batch {
            0 => return Ok(None),
            batch if batch <= super::fast_mmvq::MMVQ_MAX_BATCH => {
                super::fast_mmvq::plain(&self.w, a)?
            }
            _ if super::fast_mmq::supports(super::kernel::KernelWeight::gguf_type(&self.w)) => {
                super::fast_mmq::plain(&self.w, a)?
            }
            // ggml has no IQ1_M mmq tile; like ggml, prefill dequantizes to F16 (not BF16) for a dense matmul
            _ => {
                let compute = if a.dtype() == DType::F32 {
                    DType::F32
                } else {
                    DType::F16
                };
                let w = super::fast_mmvq::dequantize(&self.w, compute)?;
                candle_nn::Module::forward(&Linear::new(w, None), &a.to_dtype(compute)?)?
                    .to_dtype(a.dtype())?
            }
        };
        Ok(Some(out))
    }
}

impl QuantMethod for GgufRawMatMul {
    fn new(_method: QuantMethodConfig) -> Result<Self> {
        candle_core::bail!("raw GGUF layers are built by the GGUF weight source")
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.w.dequantize(&self.w.device())
    }

    fn forward_raw(&self, a: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(out) = self.cuda_forward(a)? {
            return self.add_bias(out);
        }
        let w = self.w.dequantize(a.device())?.to_dtype(a.dtype())?;
        let out = candle_nn::Module::forward(&Linear::new(w, None), a)?;
        self.add_bias(out)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (DType::F32, self.w.device())
    }

    fn plan_isq(&self, request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        Ok(crate::plan_weight_isq(
            DType::F32,
            self.w.device(),
            self.w.shape().dims().to_vec(),
            request,
            true,
        ))
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        let w = (self.dequantize_w()? + delta)?;
        Ok(Arc::new(crate::UnquantLinear::new(
            QuantMethodConfig::Unquantized(Linear::new(w, self.b.clone())),
        )?))
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        let Some(dtype) = dtype else {
            let b = self.b.as_ref().map(|b| b.to_device(&device)).transpose()?;
            return Ok(Arc::new(Self::new(self.w.to_device(&device)?, b)));
        };
        let unquant = Arc::new(crate::UnquantLinear::new(QuantMethodConfig::Unquantized(
            Linear::new(self.dequantize_w()?, self.b.clone()),
        ))?) as Arc<dyn QuantMethod>;
        unquant.apply_isq(Some(dtype), device, n_quantized, imatrix_weight, guard)
    }

    fn has_bias(&self) -> bool {
        self.b.is_some()
    }
}

impl QuantizedSerde for GgufRawMatMul {
    fn name(&self) -> &'static str {
        "gguf-raw"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Blocks and expected values from gguf-py's IQ dequantizers (tests/fixtures/gguf_iq/make_goldens.py).
    const GOLDENS: &str = include_str!("../../tests/fixtures/gguf_iq/goldens.json");

    #[derive(serde::Deserialize)]
    struct Golden {
        ty: String,
        bytes: Vec<u8>,
        values: Vec<f32>,
    }

    #[cfg(feature = "cuda")]
    const IQ1_M_SCALE_WORDS: usize = 48;

    // Deterministic blocks with a small finite f16 scale; every other byte is arbitrary scale bits and indices.
    #[cfg(feature = "cuda")]
    fn random_blocks(ty: GgufType, blocks: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u8
        };
        (0..blocks)
            .flat_map(|i| {
                let mut block = (0..ty.type_size()).map(|_| next()).collect::<Vec<_>>();
                let scale = half::f16::from_f32(0.002 + 0.001 * (i % 7) as f32).to_bits();
                if ty == GgufType::Iq1M {
                    // IQ1_M's f16 is the top nibble of each of the four u16 scale words
                    for k in 0..4 {
                        let at = IQ1_M_SCALE_WORDS + 2 * k;
                        let word = u16::from_le_bytes([block[at], block[at + 1]]) & 0x0FFF
                            | ((scale >> (4 * k)) & 0xF) << 12;
                        block[at..at + 2].copy_from_slice(&word.to_le_bytes());
                    }
                } else {
                    block[..2].copy_from_slice(&scale.to_le_bytes());
                }
                block
            })
            .collect()
    }

    #[cfg(feature = "cuda")]
    fn cosine(a: &Tensor, b: &Tensor) -> Result<f32> {
        let (a, b) = (
            a.flatten_all()?.to_dtype(DType::F32)?,
            b.flatten_all()?.to_dtype(DType::F32)?,
        );
        let dot = (&a * &b)?.sum_all()?.to_scalar::<f32>()?;
        let norms =
            (a.sqr()?.sum_all()?.sqrt()? * b.sqr()?.sum_all()?.sqrt()?)?.to_scalar::<f32>()?;
        Ok(dot / norms)
    }

    // Batches up to 8 take mmvq, larger ones mmq or IQ1_M's dense matmul; Q8_1 activations make it a cosine bound.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_kernels_match_the_dequantized_weight() -> Result<()> {
        const ROWS: usize = 64;
        // 288 columns leave IQ4_NL rows short of a whole mmq K tile, so the tail reads the padding
        const NL_COLS: usize = 288;
        const COLS: usize = 512;
        let Ok(cuda) = Device::new_cuda(0) else {
            eprintln!("SKIP: no CUDA device");
            return Ok(());
        };
        for (ty, cols) in GgufType::RAW_BLOCKS
            .into_iter()
            .map(|ty| (ty, COLS))
            .chain([(GgufType::Iq4Nl, NL_COLS)])
        {
            let bytes = random_blocks(ty, ROWS * cols / ty.block_size(), 7);
            let cpu = RawGgufTensor::new(ty, &[ROWS, cols], bytes.clone(), &Device::Cpu)?;
            let gpu =
                GgufRawMatMul::new(RawGgufTensor::new(ty, &[ROWS, cols], bytes, &cuda)?, None);
            let weight = cpu.dequantize(&Device::Cpu)?;
            for batch in [1, 8, 33] {
                let xs = Tensor::randn(0f32, 1f32, (batch, cols), &Device::Cpu)?;
                let expected = xs.matmul(&weight.t()?)?;
                for dtype in [DType::F32, DType::BF16, DType::F16] {
                    let actual = gpu.forward(&xs.to_dtype(dtype)?.to_device(&cuda)?)?;
                    let similarity = cosine(&actual.to_device(&Device::Cpu)?, &expected)?;
                    assert!(
                        similarity > 0.999,
                        "{ty:?} batch {batch} {dtype:?}: cosine {similarity}"
                    );
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_dequantization_matches_the_cpu() -> Result<()> {
        const ROWS: usize = 16;
        const COLS: usize = 512;
        let Ok(cuda) = Device::new_cuda(0) else {
            eprintln!("SKIP: no CUDA device");
            return Ok(());
        };
        let ty = GgufType::Iq1M;
        let bytes = random_blocks(ty, ROWS * COLS / ty.block_size(), 11);
        let expected = RawGgufTensor::new(ty, &[ROWS, COLS], bytes.clone(), &Device::Cpu)?
            .dequantize(&Device::Cpu)?;
        let gpu = RawGgufTensor::new(ty, &[ROWS, COLS], bytes, &cuda)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let actual = super::super::fast_mmvq::dequantize(&gpu, dtype)?
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?;
            let rounded = expected.to_dtype(dtype)?.to_dtype(DType::F32)?;
            let diff = (actual - rounded)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(
                diff < 1e-5,
                "IQ1_M CUDA dequantization to {dtype:?} differs by {diff}"
            );
        }
        Ok(())
    }

    #[test]
    fn dequantization_matches_gguf_py() -> Result<()> {
        let goldens: Vec<Golden> =
            serde_json::from_str(GOLDENS).map_err(candle_core::Error::wrap)?;
        assert!(!goldens.is_empty());
        for golden in goldens {
            let ty = GgufType::RAW_BLOCKS
                .into_iter()
                .find(|ty| {
                    format!("{ty:?}").to_uppercase().replace('_', "") == golden.ty.replace('_', "")
                })
                .unwrap_or_else(|| panic!("unexpected golden type {}", golden.ty));
            let values = dequantize_blocks(ty, &golden.bytes)?;
            assert_eq!(values, golden.values, "{ty:?}");
        }
        Ok(())
    }
}
