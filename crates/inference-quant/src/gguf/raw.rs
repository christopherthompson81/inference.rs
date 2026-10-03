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
        let Some(row_bytes) = ty.row_bytes(cols) else {
            candle_core::bail!(
                "{ty:?} rows of {cols} elements are not whole {}-element blocks",
                ty.block_size()
            );
        };
        let expected = rows * row_bytes;
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

    /// Rows `ids` of the weight, dequantized to F32, shaped `ids.dims() + [cols]`, on the weight's device.
    pub fn embedding(&self, ids: &Tensor) -> Result<Tensor> {
        let (rows, cols) = self.shape.dims2()?;
        let row_bytes = self.ty.row_bytes(cols).expect("validated at construction");
        let mut dims = ids.dims().to_vec();
        dims.push(cols);
        let flat = ids.flatten_all()?.to_dtype(DType::U32)?;
        if flat.elem_count() == 0 {
            return Tensor::zeros(dims, DType::F32, &self.device());
        }
        let gathered = match &self.storage {
            RawStorage::Cpu(bytes) => {
                let ids = flat.to_vec1::<u32>()?;
                if let Some(id) = ids.iter().find(|&&id| id as usize >= rows) {
                    candle_core::bail!("embedding id {id} is out of range for {rows} rows");
                }
                let rows = ids
                    .into_iter()
                    .flat_map(|id| {
                        bytes[id as usize * row_bytes..][..row_bytes]
                            .iter()
                            .copied()
                    })
                    .collect();
                Self::new(self.ty, &[flat.elem_count(), cols], rows, &Device::Cpu)?
            }
            #[cfg(feature = "cuda")]
            RawStorage::Cuda { .. } => self.gather_cuda(&flat, row_bytes)?,
        };
        #[cfg(feature = "cuda")]
        if gathered.device().is_cuda() && super::fast_mmvq::can_dequantize(self.ty, DType::F32) {
            return super::fast_mmvq::dequantize(&gathered, DType::F32)?.reshape(dims);
        }
        gathered.dequantize(&self.device())?.reshape(dims)
    }

    // The rows selected by `ids`, copied on the GPU into a weight of their own
    #[cfg(feature = "cuda")]
    fn gather_cuda(&self, ids: &Tensor, row_bytes: usize) -> Result<Self> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        let RawStorage::Cuda { blocks, device, .. } = &self.storage else {
            candle_core::bail!("{:?} gather needs the weight on CUDA", self.ty);
        };
        let ids = ids.to_device(&Device::Cuda(device.clone()))?.contiguous()?;
        let (ids_storage, ids_layout) = ids.storage_and_layout();
        let candle_core::Storage::Cuda(ids_cuda) = &*ids_storage else {
            candle_core::bail!("embedding ids must live on CUDA");
        };
        let n = ids.elem_count();
        let len = n * row_bytes;
        let padding = MATRIX_ROW_PADDING * self.ty.type_size() / self.ty.block_size();
        let mut rows = device.alloc_zeros::<u8>(len + padding)?;
        let stream = device.cuda_stream();
        {
            let ids_slice = ids_cuda.as_cuda_slice::<u32>()?;
            let (ids_ptr, _ids_guard) =
                crate::utils::slice_ptr_on_stream(ids_slice, ids_layout.start_offset(), &stream);
            let (src_ptr, _src_guard) = blocks.device_ptr(&stream);
            let (dst_ptr, _dst_guard) =
                crate::utils::slice_ptr_mut_on_stream(&mut rows, 0, &stream);
            unsafe {
                super::ffi::launch_gather_rows_u8(
                    src_ptr as *const std::ffi::c_void,
                    ids_ptr as *const std::ffi::c_void,
                    dst_ptr as *mut std::ffi::c_void,
                    row_bytes as i64,
                    self.shape.dims2()?.0 as i64,
                    n as i64,
                    stream.cu_stream() as *mut std::ffi::c_void,
                )
            };
        }
        Ok(Self {
            ty: self.ty,
            shape: Shape::from((n, self.shape.dims2()?.1)),
            storage: RawStorage::Cuda {
                blocks: Arc::new(rows),
                len,
                device: device.clone(),
            },
        })
    }

    /// The weight as an F32 tensor on `device`.
    pub fn dequantize(&self, device: &Device) -> Result<Tensor> {
        let values = dequantize_rows(self.ty, self.shape.dims2()?.1, &self.bytes()?)?;
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

/// ggml's reference dequantization (`dequantize_row_*` in ggml-quants.c, or ik_llama.cpp's) of rows of `cols`.
pub fn dequantize_rows(ty: GgufType, cols: usize, bytes: &[u8]) -> Result<Vec<f32>> {
    if ty.is_trellis() || ty.is_iqk() {
        let row_bytes = ty.row_bytes(cols).ok_or_else(|| {
            candle_core::Error::Msg(format!("{ty:?} rows cannot hold {cols} elements"))
        })?;
        if !bytes.len().is_multiple_of(row_bytes) {
            candle_core::bail!("{ty:?} data of {} bytes is not whole rows", bytes.len());
        }
        let mut out = vec![0f32; bytes.len() / row_bytes * cols];
        for (row, values) in bytes
            .chunks_exact(row_bytes)
            .zip(out.chunks_exact_mut(cols))
        {
            if ty.is_trellis() {
                super::kt_dequant::dequantize_row(ty, row, values);
            } else {
                super::iqk_dequant::dequantize_row(ty, row, values);
            }
        }
        return Ok(out);
    }
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
            // IQ1_M and the ik types have no mmq tile here; like ggml, prefill dequantizes to F16 for a dense matmul
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

    fn embedding_forward_raw(&self, ids: &Tensor) -> Result<Tensor> {
        self.w.embedding(ids)
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
    // Rows and values from ik_llama.cpp's reference dequantizers (tests/fixtures/gguf_ik/make_goldens.py).
    const IK_GOLDENS: &str = include_str!("../../tests/fixtures/gguf_ik/goldens.json");

    #[derive(serde::Deserialize)]
    struct Golden {
        ty: String,
        // ik goldens are rows of this many elements
        #[serde(default)]
        cols: Option<usize>,
        // Set where the reference computes values our kernels' tables round
        #[serde(default)]
        tolerance: Option<f32>,
        bytes: Vec<u8>,
        values: Vec<f32>,
    }

    const IQ1_M_SCALE_WORDS: usize = 48;
    // 480 columns end IQ3_KT / IQ4_KT rows in seven 32-element tail sub-blocks, reaching both scale nibbles
    #[cfg(feature = "cuda")]
    const KT_TAIL_TYPES: [GgufType; 2] = [GgufType::Iq3Kt, GgufType::Iq4Kt];
    #[cfg(feature = "cuda")]
    const KT_TAIL_COLS: usize = 480;

    // Row-scaled rows: a small finite f32 or f16 row scale, then arbitrary block bytes.
    fn random_rows(ty: GgufType, rows: usize, cols: usize, seed: u64) -> Vec<u8> {
        if !ty.has_row_scale() {
            return random_blocks(ty, rows * cols / ty.block_size(), seed);
        }
        let row_bytes = ty.row_bytes(cols).expect("valid row-scaled row");
        let mut bytes = random_blocks(GgufType::Q8_0, (rows * row_bytes).div_ceil(34), seed);
        bytes.truncate(rows * row_bytes);
        for (i, row) in bytes.chunks_exact_mut(row_bytes).enumerate() {
            let scale = 0.0005 + 0.0002 * (i % 5) as f32;
            match ty.row_scale_bytes() {
                2 => row[..2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes()),
                _ => row[..4].copy_from_slice(&scale.to_le_bytes()),
            }
        }
        bytes
    }

    // Deterministic blocks with a small finite f16 scale; every other byte is arbitrary scale bits and indices.
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
        // An odd row count leaves the last mmvq row pair half empty
        const ROWS: usize = 63;
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
            .chain(KT_TAIL_TYPES.map(|ty| (ty, KT_TAIL_COLS)))
        {
            let bytes = random_rows(ty, ROWS, cols, 7);
            let cpu = RawGgufTensor::new(ty, &[ROWS, cols], bytes.clone(), &Device::Cpu)?;
            let gpu =
                GgufRawMatMul::new(RawGgufTensor::new(ty, &[ROWS, cols], bytes, &cuda)?, None);
            let weight = cpu.dequantize(&Device::Cpu)?;
            for batch in [1, 3, 8, 33] {
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

    // Tied embeddings in ik's mixes are IQ*_K; one IQ4_NL row is less than the 256 elements its GPU dequantizer fills
    #[test]
    fn embedding_rows_match_the_dequantized_weight() -> Result<()> {
        const ROWS: usize = 40;
        const COLS: usize = 512;
        const NL_COLS: usize = 288;
        let ids = Tensor::new(&[[3u32, 0, 39], [17, 17, 8]], &Device::Cpu)?;
        let devices: Vec<Device> = std::iter::once(Device::Cpu)
            .chain(Device::new_cuda(0).ok().filter(|_| cfg!(feature = "cuda")))
            .collect();
        // 6 x 288 IQ4_NL elements leave the last of its 256-element super-blocks partly filled
        let cases = GgufType::RAW_BLOCKS
            .map(|ty| (ty, COLS))
            .into_iter()
            .chain([(GgufType::Iq4Nl, NL_COLS)]);
        for (ty, cols) in cases {
            let bytes = random_rows(ty, ROWS, cols, 5);
            let weight = RawGgufTensor::new(ty, &[ROWS, cols], bytes.clone(), &Device::Cpu)?;
            let expected = weight
                .dequantize(&Device::Cpu)?
                .embedding(&ids.flatten_all()?)?
                .reshape((2, 3, cols))?;
            for device in &devices {
                let weight = RawGgufTensor::new(ty, &[ROWS, cols], bytes.clone(), device)?;
                let actual = weight
                    .embedding(&ids.to_device(device)?)?
                    .to_device(&Device::Cpu)?;
                let diff = (actual - &expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
                assert!(
                    diff < 1e-6,
                    "{ty:?} on {device:?}: embedding differs by {diff}"
                );
                let empty = weight.embedding(&Tensor::zeros((0,), DType::U32, device)?)?;
                assert_eq!(empty.dims(), [0, cols]);
            }
            assert!(
                weight
                    .embedding(&Tensor::new(&[ROWS as u32], &Device::Cpu)?)
                    .is_err()
            );
        }
        Ok(())
    }

    // A host dequant in an embedding lookup breaks the decode CUDA graph, so every raw type needs a GPU dequantizer
    #[cfg(feature = "cuda")]
    #[test]
    fn every_raw_type_dequantizes_on_the_gpu() {
        for ty in GgufType::RAW_BLOCKS {
            assert!(
                super::super::fast_mmvq::can_dequantize(ty, DType::F32),
                "{ty:?}"
            );
        }
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
        let cases = GgufType::RAW_BLOCKS
            .into_iter()
            .map(|ty| (ty, COLS))
            .chain(KT_TAIL_TYPES.map(|ty| (ty, KT_TAIL_COLS)));
        for (ty, cols) in cases {
            let bytes = random_rows(ty, ROWS, cols, 11);
            let expected = RawGgufTensor::new(ty, &[ROWS, cols], bytes.clone(), &Device::Cpu)?
                .dequantize(&Device::Cpu)?;
            let peak = expected.abs()?.max_all()?.to_scalar::<f32>()?;
            let gpu = RawGgufTensor::new(ty, &[ROWS, cols], bytes, &cuda)?;
            // One rounding step of the target dtype, relative to the largest value
            for (dtype, tolerance) in [(DType::F32, 1e-6), (DType::F16, 1e-3), (DType::BF16, 8e-3)]
            {
                let actual = super::super::fast_mmvq::dequantize(&gpu, dtype)?
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?;
                let rounded = expected.to_dtype(dtype)?.to_dtype(DType::F32)?;
                let diff = (actual - rounded)?.abs()?.max_all()?.to_scalar::<f32>()?;
                assert!(
                    diff <= tolerance * peak,
                    "{ty:?} [{ROWS}, {cols}] CUDA dequantization to {dtype:?} differs by {diff} (peak {peak})"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn dequantization_matches_the_references() -> Result<()> {
        let mut goldens: Vec<Golden> =
            serde_json::from_str(GOLDENS).map_err(candle_core::Error::wrap)?;
        goldens.extend(
            serde_json::from_str::<Vec<Golden>>(IK_GOLDENS).map_err(candle_core::Error::wrap)?,
        );
        assert!(!goldens.is_empty());
        for golden in goldens {
            let ty = GgufType::RAW_BLOCKS
                .into_iter()
                .find(|ty| {
                    format!("{ty:?}").to_uppercase().replace('_', "") == golden.ty.replace('_', "")
                })
                .unwrap_or_else(|| panic!("unexpected golden type {}", golden.ty));
            let cols = golden.cols.unwrap_or(golden.values.len());
            let values = dequantize_rows(ty, cols, &golden.bytes)?;
            match golden.tolerance {
                Some(tolerance) => {
                    let peak = golden.values.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let diff = values
                        .iter()
                        .zip(&golden.values)
                        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                    assert!(
                        diff <= tolerance * peak,
                        "{ty:?}: differs by {diff} (peak {peak})"
                    );
                }
                None => assert_eq!(values, golden.values, "{ty:?}"),
            }
        }
        Ok(())
    }
}
