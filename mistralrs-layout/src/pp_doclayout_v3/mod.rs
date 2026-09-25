mod backbone;
pub mod config;
mod decoder;
mod encoder;
mod model;
pub mod postprocess;
pub mod preprocess;

use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use image::RgbImage;
use rayon::prelude::*;

pub use config::{PPDocLayoutV3Config, PPDocLayoutV3PreprocessorConfig, LABELS};
pub use model::{Intermediates, PPDocLayoutV3, RawOutputs};
pub use postprocess::{LayoutDetection, PostprocessArgs};
pub use preprocess::Preprocessor;

pub const DEFAULT_THRESHOLD: f32 = 0.5;
const RAYON_THREADS_ENV: &str = "RAYON_NUM_THREADS";

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let s = std::fs::read_to_string(path).map_err(candle_core::Error::wrap)?;
    serde_json::from_str(&s).map_err(candle_core::Error::wrap)
}

/// Loads an HF-format `PP-DocLayoutV3_safetensors` directory and runs detection end to end.
pub struct PPDocLayoutV3Detector {
    model: PPDocLayoutV3,
    preprocessor: Preprocessor,
    device: Device,
    /// CPU only: physical-core pool; hyperthreads slow the FMA-bound conv kernels (16 -> 8 threads: 0.78 -> 0.63 s).
    pool: Option<rayon::ThreadPool>,
}

impl PPDocLayoutV3Detector {
    pub fn load(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let dir = dir.as_ref();
        let cfg: PPDocLayoutV3Config = read_json(&dir.join("config.json"))?;
        let pp_cfg: PPDocLayoutV3PreprocessorConfig =
            read_json(&dir.join("preprocessor_config.json"))?;
        let preprocessor = Preprocessor::new(&pp_cfg);
        // SAFETY: the weight file is mmapped read-only and not modified while the model is alive.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dir.join("model.safetensors")],
                DType::F32,
                device,
            )?
        };
        let model = PPDocLayoutV3::new(cfg, (preprocessor.height, preprocessor.width), vb)?;
        let pool = if device.is_cpu() && std::env::var_os(RAYON_THREADS_ENV).is_none() {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(num_cpus::get_physical())
                    .build()
                    .map_err(candle_core::Error::wrap)?,
            )
        } else {
            None
        };
        Ok(Self {
            model,
            preprocessor,
            device: device.clone(),
            pool,
        })
    }

    /// Runs `f` on the detector's CPU thread pool (or inline on other devices / when `RAYON_NUM_THREADS` is set).
    pub fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        match &self.pool {
            Some(pool) => pool.install(f),
            None => f(),
        }
    }

    pub fn model(&self) -> &PPDocLayoutV3 {
        &self.model
    }

    pub fn preprocessor(&self) -> &Preprocessor {
        &self.preprocessor
    }

    pub fn detect(&self, image: &RgbImage, threshold: f32) -> Result<Vec<LayoutDetection>> {
        Ok(self
            .detect_batch(std::slice::from_ref(image), threshold)?
            .remove(0))
    }

    pub fn detect_batch(
        &self,
        images: &[RgbImage],
        threshold: f32,
    ) -> Result<Vec<Vec<LayoutDetection>>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        self.install(|| self.detect_batch_inner(images, threshold))
    }

    fn detect_batch_inner(
        &self,
        images: &[RgbImage],
        threshold: f32,
    ) -> Result<Vec<Vec<LayoutDetection>>> {
        let pixels = images
            .par_iter()
            .map(|im| self.preprocessor.preprocess(im, &self.device))
            .collect::<Result<Vec<_>>>()?;
        let out = self.model.forward(&Tensor::stack(&pixels, 0)?, false)?;
        // one host transfer per output instead of one per image
        let logits = out.logits.to_device(&Device::Cpu)?;
        let boxes = out.pred_boxes.to_device(&Device::Cpu)?;
        let order = out.order_logits.to_device(&Device::Cpu)?;
        images
            .iter()
            .enumerate()
            .map(|(i, im)| {
                let args = PostprocessArgs {
                    threshold,
                    orig_size: im.dimensions(),
                };
                postprocess::postprocess(&logits.get(i)?, &boxes.get(i)?, &order.get(i)?, &args)
            })
            .collect()
    }
}
