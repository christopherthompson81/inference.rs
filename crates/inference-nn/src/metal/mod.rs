pub mod gdn;
pub mod ssm;

use inference_tensor::{Device, Result};

#[cfg(feature = "metal")]
use candle_metal_kernels::source::Source;

pub fn warmup_metal_kernels(device: &Device) -> Result<()> {
    #[cfg(not(feature = "metal"))]
    let _ = device;

    #[cfg(feature = "metal")]
    if let Device::Metal(device) = device {
        let sources = [
            Source::Affine,
            Source::Binary,
            Source::Cast,
            Source::Conv,
            Source::Fill,
            Source::Gemm,
            Source::Gemv,
            Source::Indexing,
            Source::MlxSort,
            Source::Quantized,
            Source::Random,
            Source::Reduce,
            Source::Sort,
            Source::Ternary,
            Source::Unary,
            Source::Sdpa,
        ];
        for source in sources {
            device
                .kernels()
                .load_library(device.device(), source)
                .map_err(inference_tensor::Error::wrap)?;
        }
        inference_quant::metal_kernels::Kernels::global()
            .load_library(device.device())
            .map_err(inference_tensor::Error::wrap)?;
    }
    Ok(())
}
