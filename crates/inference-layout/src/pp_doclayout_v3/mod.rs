mod backbone;
pub mod config;
mod decoder;
mod encoder;
pub mod gguf;
mod model;
pub mod outline;
pub mod postprocess;
pub mod preprocess;

use std::path::Path;
use std::sync::OnceLock;

use image::RgbImage;
use inference_tensor::nn::VarBuilder;
use inference_tensor::{DType, Device, Result, Tensor};
use rayon::prelude::*;

pub use config::{LABELS, PPDocLayoutV3Config, PPDocLayoutV3PreprocessorConfig};
pub use model::{Intermediates, PPDocLayoutV3, RawOutputs};
pub use postprocess::{LayoutDetection, PostprocessArgs};
pub use preprocess::Preprocessor;

pub const DEFAULT_THRESHOLD: f32 = 0.5;
const RAYON_THREADS_ENV: &str = "RAYON_NUM_THREADS";

fn parse_json<T: serde::de::DeserializeOwned>(s: &str) -> Result<T> {
    serde_json::from_str(s).map_err(inference_tensor::Error::wrap)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    parse_json(&std::fs::read_to_string(path).map_err(inference_tensor::Error::wrap)?)
}

static CPU_POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();

/// Process-wide pool for the AVX2 kernels: hyperthreads slow them (16 -> 8 threads: 0.78 -> 0.63 s), so use physical
/// cores, capped by the CPUs this process may run on (affinity, cgroup quota). `None` defers to rayon's global pool.
fn cpu_pool() -> Option<&'static rayon::ThreadPool> {
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

/// Loads an HF-format `PP-DocLayoutV3_safetensors` directory or a GGUF of it and runs detection end to end.
pub struct PPDocLayoutV3Detector {
    model: PPDocLayoutV3,
    preprocessor: Preprocessor,
    device: Device,
    pool: Option<Pool>,
    labels: Vec<String>,
}

enum Pool {
    /// The process-wide physical-core pool from `cpu_pool`.
    Shared(&'static rayon::ThreadPool),
    /// Set by `with_cpu_threads`.
    Owned(rayon::ThreadPool),
}

impl PPDocLayoutV3Detector {
    /// `path` is the HF directory or a GGUF file `gguf::write_gguf` made.
    pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            inference_tensor::bail!("{} does not exist", path.display())
        }
        if path.is_file() {
            let file = std::fs::File::open(path).map_err(inference_tensor::Error::wrap)?;
            let ckpt = gguf::read_gguf(&mut std::io::BufReader::new(file), device)?;
            return Self::from_parts(
                parse_json(&ckpt.config)?,
                &parse_json(&ckpt.preprocessor_config)?,
                ckpt.weights,
                device,
            );
        }
        let cfg = read_json(&path.join("config.json"))?;
        let pp_cfg = read_json(&path.join("preprocessor_config.json"))?;
        // SAFETY: the weight file is mmapped read-only and not modified while the model is alive.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[path.join("model.safetensors")],
                DType::F32,
                device,
            )?
        };
        Self::from_parts(cfg, &pp_cfg, vb, device)
    }

    fn from_parts(
        cfg: PPDocLayoutV3Config,
        pp_cfg: &PPDocLayoutV3PreprocessorConfig,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let preprocessor = Preprocessor::new(pp_cfg);
        let labels = cfg.labels();
        let model = PPDocLayoutV3::new(cfg, (preprocessor.height, preprocessor.width), vb)?;
        let pool = if device.is_cpu() {
            cpu_pool().map(Pool::Shared)
        } else {
            None
        };
        Ok(Self {
            model,
            preprocessor,
            device: device.clone(),
            pool,
            labels,
        })
    }

    /// Class names by id, as used in `LayoutDetection::label`.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Gives this detector its own `threads`-thread CPU pool instead of the shared physical-core one; no-op off CPU.
    pub fn with_cpu_threads(mut self, threads: usize) -> Result<Self> {
        if self.device.is_cpu() {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads.max(1))
                .build()
                .map_err(inference_tensor::Error::wrap)?;
            self.pool = Some(Pool::Owned(pool));
        }
        Ok(self)
    }

    /// Runs `f` on the detector's CPU pool; wrap direct `model()` calls in this to get the same threading as `detect`.
    pub fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        match &self.pool {
            Some(Pool::Shared(pool)) => pool.install(f),
            Some(Pool::Owned(pool)) => pool.install(f),
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
        let out = self.model.forward(&Tensor::stack(&pixels, 0)?, true)?;
        // one host transfer per output instead of one per image
        let logits = out.logits.to_device(&Device::Cpu)?;
        let boxes = out.pred_boxes.to_device(&Device::Cpu)?;
        let order = out.order_logits.to_device(&Device::Cpu)?;
        let masks = out.masks.expect("forward was asked for masks");
        images
            .iter()
            .enumerate()
            .map(|(i, im)| {
                let args = PostprocessArgs {
                    threshold,
                    labels: &self.labels,
                    orig_size: im.dimensions(),
                };
                let mut dets = postprocess::postprocess(
                    &logits.get(i)?,
                    &boxes.get(i)?,
                    &order.get(i)?,
                    &args,
                )?;
                self.add_outlines(&mut dets, &masks.get(i)?, im.dimensions(), threshold)?;
                Ok(dets)
            })
            .collect()
    }

    // Only the kept queries' masks leave the device; each outline is traced at its box's size in original pixels
    fn add_outlines(
        &self,
        dets: &mut [LayoutDetection],
        masks: &Tensor,
        (width, height): (u32, u32),
        threshold: f32,
    ) -> Result<()> {
        if dets.is_empty() {
            return Ok(());
        }
        let (_, mask_h, mask_w) = masks.dims3()?;
        let queries: Vec<u32> = dets.iter().map(|d| d.query as u32).collect();
        let kept = masks
            .index_select(&Tensor::new(queries.as_slice(), masks.device())?, 0)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        // as transformers computes it from a target_sizes tensor: size * (1 / target) in f32, not size / target
        let scale = (
            (1. / width as f32) * self.preprocessor.width as f32,
            (1. / height as f32) * self.preprocessor.height as f32,
        );
        dets.par_iter_mut()
            .zip(kept.par_chunks(mask_h * mask_w))
            .for_each(|(det, logits)| {
                let mask = outline::MaskView {
                    logits,
                    height: mask_h,
                    width: mask_w,
                };
                det.polygon = outline::region_outline(&mask, det.bbox, scale, threshold);
            });
        Ok(())
    }
}
