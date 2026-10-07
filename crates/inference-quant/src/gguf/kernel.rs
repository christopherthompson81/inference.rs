//! The GGUF weight formats our matmul kernels read, including ones Candle has no `GgmlDType` for.

use inference_tensor::quantized::GgmlDType;

use super::archive::GgufDType;

// ik_llama.cpp numbers its own types from IQ2_K
const IQK_FIRST_ID: u32 = 137;

/// A ggml tensor type; Candle's variants keep their names so the kernel tables read the same.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GgufType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Iq4Nl,
    Iq4Xs,
    Iq2Xxs,
    Iq2Xs,
    Iq2S,
    Iq3Xxs,
    Iq3S,
    Iq1S,
    Iq1M,
    Iq1Kt,
    Iq2Kt,
    Iq3Kt,
    Iq4Kt,
    Iq2K,
    Iq3K,
    Iq4K,
    Iq5K,
    Iq6K,
    Iq4Ks,
    Iq2Ks,
    Iq4Kss,
    Iq5Ks,
    Iq3Ks,
    Iq2Kl,
}

impl From<GgmlDType> for GgufType {
    fn from(dtype: GgmlDType) -> Self {
        match dtype {
            GgmlDType::F32 => Self::F32,
            GgmlDType::F16 => Self::F16,
            GgmlDType::BF16 => Self::BF16,
            GgmlDType::Q4_0 => Self::Q4_0,
            GgmlDType::Q4_1 => Self::Q4_1,
            GgmlDType::Q5_0 => Self::Q5_0,
            GgmlDType::Q5_1 => Self::Q5_1,
            GgmlDType::Q8_0 => Self::Q8_0,
            GgmlDType::Q8_1 => Self::Q8_1,
            GgmlDType::Q2K => Self::Q2K,
            GgmlDType::Q3K => Self::Q3K,
            GgmlDType::Q4K => Self::Q4K,
            GgmlDType::Q5K => Self::Q5K,
            GgmlDType::Q6K => Self::Q6K,
            GgmlDType::Q8K => Self::Q8K,
        }
    }
}

impl GgufType {
    /// The types only our own kernels read; Candle cannot hold them in a `QTensor`.
    pub const RAW_BLOCKS: [Self; 24] = [
        Self::Iq4Nl,
        Self::Iq4Xs,
        Self::Iq2Xxs,
        Self::Iq2Xs,
        Self::Iq2S,
        Self::Iq3Xxs,
        Self::Iq3S,
        Self::Iq1S,
        Self::Iq1M,
        Self::Iq1Kt,
        Self::Iq2Kt,
        Self::Iq3Kt,
        Self::Iq4Kt,
        Self::Iq2K,
        Self::Iq3K,
        Self::Iq4K,
        Self::Iq5K,
        Self::Iq6K,
        Self::Iq4Ks,
        Self::Iq2Ks,
        Self::Iq4Kss,
        Self::Iq5Ks,
        Self::Iq3Ks,
        Self::Iq2Kl,
    ];

    /// The ggml type id, as stored in a GGUF tensor header.
    pub fn id(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            Self::Iq2Xxs => 16,
            Self::Iq2Xs => 17,
            Self::Iq3Xxs => 18,
            Self::Iq1S => 19,
            Self::Iq4Nl => 20,
            Self::Iq3S => 21,
            Self::Iq2S => 22,
            Self::Iq4Xs => 23,
            Self::Iq1M => 29,
            Self::BF16 => 30,
            Self::Iq2Kt => 153,
            Self::Iq3Kt => 154,
            Self::Iq4Kt => 155,
            Self::Iq1Kt => 158,
            Self::Iq2K => 137,
            Self::Iq3K => 138,
            Self::Iq4K => 139,
            Self::Iq5K => 140,
            Self::Iq6K => 141,
            Self::Iq4Ks => 144,
            Self::Iq2Ks => 145,
            Self::Iq4Kss => 146,
            Self::Iq5Ks => 152,
            Self::Iq3Ks => 156,
            Self::Iq2Kl => 157,
        }
    }

    pub fn candle(self) -> Option<GgmlDType> {
        GgufDType::new(self.id()).candle_dtype().ok()
    }

    /// The raw-block type for a ggml id Candle cannot hold, if our kernels read it.
    pub fn raw_from_id(id: u32) -> Option<Self> {
        Self::RAW_BLOCKS.into_iter().find(|ty| ty.id() == id)
    }

    pub fn block_size(self) -> usize {
        GgufDType::new(self.id())
            .block_size()
            .expect("every GgufType has a ggml block size")
    }

    /// Bytes per row of `cols` elements, if that is a valid row length for this type.
    pub fn row_bytes(self, cols: usize) -> Option<usize> {
        GgufDType::new(self.id()).row_size(cols)
    }

    /// ik_llama.cpp's row-scaled types prefix each row with its scale, so their rows are the unit bytes split on.
    pub fn has_row_scale(self) -> bool {
        self.row_scale_bytes() > 0
    }

    pub fn row_scale_bytes(self) -> usize {
        GgufDType::new(self.id()).row_meta_size()
    }

    pub fn is_trellis(self) -> bool {
        matches!(self, Self::Iq1Kt | Self::Iq2Kt | Self::Iq3Kt | Self::Iq4Kt)
    }

    /// ik_llama.cpp's non-trellis types, dequantized a row at a time by `iqk_dequant`.
    pub fn is_iqk(self) -> bool {
        self.id() >= IQK_FIRST_ID && !self.is_trellis()
    }

    /// The (elements, bytes) unit rows of `cols` elements are cut into: a block, or a whole row-scaled row.
    pub fn row_unit(self, cols: usize) -> inference_tensor::Result<(usize, usize)> {
        if !self.has_row_scale() {
            return Ok((self.block_size(), self.type_size()));
        }
        let bytes = self.row_bytes(cols).ok_or_else(|| {
            inference_tensor::Error::Msg(format!("{self:?} rows cannot hold {cols} elements"))
        })?;
        Ok((cols, bytes))
    }

    pub fn type_size(self) -> usize {
        GgufDType::new(self.id())
            .type_size()
            .expect("every GgufType has a ggml type size")
    }
}

/// A GGUF weight our CUDA matmul kernels can read in place: its type, shape and device buffer.
#[cfg(feature = "cuda")]
pub trait KernelWeight {
    fn gguf_type(&self) -> GgufType;
    fn kernel_shape(&self) -> &inference_tensor::Shape;
    fn kernel_device(&self) -> inference_tensor::Device;
    fn kernel_ptr<'a>(
        &'a self,
        stream: &'a inference_tensor::cuda_backend::cudarc::driver::CudaStream,
    ) -> inference_tensor::Result<(
        *const u8,
        inference_tensor::cuda_backend::cudarc::driver::SyncOnDrop<'a>,
    )>;
}

#[cfg(feature = "cuda")]
impl KernelWeight for inference_tensor::quantized::QTensor {
    fn gguf_type(&self) -> GgufType {
        self.dtype().into()
    }

    fn kernel_shape(&self) -> &inference_tensor::Shape {
        self.shape()
    }

    fn kernel_device(&self) -> inference_tensor::Device {
        self.device()
    }

    fn kernel_ptr<'a>(
        &'a self,
        stream: &'a inference_tensor::cuda_backend::cudarc::driver::CudaStream,
    ) -> inference_tensor::Result<(
        *const u8,
        inference_tensor::cuda_backend::cudarc::driver::SyncOnDrop<'a>,
    )> {
        self.device_ptr_with_guard(stream)
    }
}
