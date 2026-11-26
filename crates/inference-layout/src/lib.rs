//! Document layout detection models built on candle.

mod cpu_conv;
mod cpu_direct;
#[cfg(feature = "cuda")]
mod cuda_kernels;
mod depthwise;
pub mod doclayout_yolo;
mod im2col;
mod layers;
mod mask_box;
mod msda;
pub mod pp_doclayout_v3;

pub use doclayout_yolo::DocLayoutYoloDetector;
pub use pp_doclayout_v3::{LayoutDetection, PPDocLayoutV3Detector};

use std::path::Path;

use std::sync::OnceLock;

const RAYON_THREADS_ENV: &str = "RAYON_NUM_THREADS";

static CPU_POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();

/// Process-wide pool for the AVX2 kernels: hyperthreads slow them (16 -> 8 threads: 0.78 -> 0.63 s), so use physical
/// cores, capped by the CPUs this process may run on (affinity, cgroup quota). `None` defers to rayon's global pool.
pub(crate) fn cpu_pool() -> Option<&'static rayon::ThreadPool> {
    CPU_POOL
        .get_or_init(|| {
            if std::env::var_os(RAYON_THREADS_ENV).is_some() || !crate::cpu_direct::available() {
                return None;
            }
            let allowed = std::thread::available_parallelism().map_or(1, |n| n.get());
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_cpus::get_physical().min(allowed))
                .build()
                .ok()
        })
        .as_ref()
}

/// Either layout model, by what `path` holds: a `.pt` checkpoint (or a directory of one) is DocLayout-YOLO, an HF
/// directory or GGUF is PP-DocLayoutV3.
pub enum LayoutDetector {
    PPDocLayoutV3(Box<PPDocLayoutV3Detector>),
    DocLayoutYolo(Box<DocLayoutYoloDetector>),
}

impl LayoutDetector {
    pub fn load(
        path: impl AsRef<Path>,
        device: &inference_tensor::Device,
    ) -> inference_tensor::Result<Self> {
        let path = path.as_ref();
        if is_doclayout_yolo(path) {
            DocLayoutYoloDetector::load(path, device).map(|d| Self::DocLayoutYolo(Box::new(d)))
        } else {
            PPDocLayoutV3Detector::load(path, device).map(|d| Self::PPDocLayoutV3(Box::new(d)))
        }
    }

    /// Gives the detector its own `threads`-thread CPU pool; no-op off CPU.
    pub fn with_cpu_threads(self, threads: usize) -> inference_tensor::Result<Self> {
        Ok(match self {
            Self::PPDocLayoutV3(d) => Self::PPDocLayoutV3(Box::new(d.with_cpu_threads(threads)?)),
            Self::DocLayoutYolo(d) => Self::DocLayoutYolo(Box::new(d.with_cpu_threads(threads)?)),
        })
    }

    /// The score each model is meant to be read at.
    pub fn default_threshold(&self) -> f32 {
        match self {
            Self::PPDocLayoutV3(_) => pp_doclayout_v3::DEFAULT_THRESHOLD,
            Self::DocLayoutYolo(_) => doclayout_yolo::DEFAULT_THRESHOLD,
        }
    }

    /// Class names by id, as used in `LayoutDetection::label`.
    pub fn labels(&self) -> &[String] {
        match self {
            Self::PPDocLayoutV3(d) => d.labels(),
            Self::DocLayoutYolo(d) => d.labels(),
        }
    }

    /// Detections in reading order.
    pub fn detect(
        &self,
        image: &image::RgbImage,
        threshold: f32,
    ) -> inference_tensor::Result<Vec<LayoutDetection>> {
        match self {
            Self::PPDocLayoutV3(d) => d.detect(image, threshold),
            Self::DocLayoutYolo(d) => d.detect(image, threshold),
        }
    }

    pub fn detect_batch(
        &self,
        images: &[image::RgbImage],
        threshold: f32,
    ) -> inference_tensor::Result<Vec<Vec<LayoutDetection>>> {
        match self {
            Self::PPDocLayoutV3(d) => d.detect_batch(images, threshold),
            Self::DocLayoutYolo(d) => d.detect_batch(images, threshold),
        }
    }
}

fn is_doclayout_yolo(path: &Path) -> bool {
    if path.is_file() {
        return path.extension().is_some_and(|e| e == "pt");
    }
    !path.join("config.json").exists() && doclayout_yolo::checkpoint_file(path).is_ok()
}

/// Whether this crate's custom ops can run on `dev`; everything else takes the generic tensor-op fallbacks.
/// CUDA needs this crate's own `cuda` feature (candle's alone can be enabled by workspace feature unification).
fn has_kernels(dev: &inference_tensor::Device) -> bool {
    dev.is_cpu() || (dev.is_cuda() && cfg!(feature = "cuda"))
}

#[cfg(test)]
mod test_util {
    use inference_tensor::{D, Device, Result, Tensor};

    pub fn max_abs(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a.to_device(&Device::Cpu)? - b.to_device(&Device::Cpu)?)?
            .abs()?
            .flatten_all()?
            .max(D::Minus1)?
            .to_scalar::<f32>()
    }

    /// `max_abs` relative to the reference's largest magnitude, for many-term fp32 sums.
    pub fn rel_err(a: &Tensor, reference: &Tensor) -> Result<f32> {
        let scale = reference
            .abs()?
            .flatten_all()?
            .max(D::Minus1)?
            .to_scalar::<f32>()?
            .max(1e-6);
        Ok(max_abs(a, reference)? / scale)
    }

    /// CPU, plus CUDA when this crate is built with it.
    pub fn devices() -> Result<Vec<Device>> {
        std::iter::once(Ok(Device::Cpu))
            .chain(cfg!(feature = "cuda").then(|| Device::new_cuda(0)))
            .collect()
    }
}
