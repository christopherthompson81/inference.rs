//! The GGUF weight formats our matmul kernels read, including ones Candle has no `GgmlDType` for.

use candle_core::quantized::GgmlDType;

use super::archive::GgufDType;

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
    pub const RAW_BLOCKS: [Self; 2] = [Self::Iq4Nl, Self::Iq4Xs];

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
            Self::Iq4Nl => 20,
            Self::Iq4Xs => 23,
            Self::BF16 => 30,
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
    fn kernel_shape(&self) -> &candle_core::Shape;
    fn kernel_device(&self) -> candle_core::Device;
    fn kernel_ptr<'a>(
        &'a self,
        stream: &'a candle_core::cuda_backend::cudarc::driver::CudaStream,
    ) -> candle_core::Result<(
        *const u8,
        candle_core::cuda_backend::cudarc::driver::SyncOnDrop<'a>,
    )>;
}

#[cfg(feature = "cuda")]
impl KernelWeight for candle_core::quantized::QTensor {
    fn gguf_type(&self) -> GgufType {
        self.dtype().into()
    }

    fn kernel_shape(&self) -> &candle_core::Shape {
        self.shape()
    }

    fn kernel_device(&self) -> candle_core::Device {
        self.device()
    }

    fn kernel_ptr<'a>(
        &'a self,
        stream: &'a candle_core::cuda_backend::cudarc::driver::CudaStream,
    ) -> candle_core::Result<(
        *const u8,
        candle_core::cuda_backend::cudarc::driver::SyncOnDrop<'a>,
    )> {
        self.device_ptr_with_guard(stream)
    }
}
