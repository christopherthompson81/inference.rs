//! DocLayout-YOLO (YOLOv10 with G2L_CRM stages), from its `torch.save` checkpoint.

mod model;
pub mod postprocess;
pub mod preprocess;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use image::RgbImage;
use inference_tensor::nn::VarBuilder;
use inference_tensor::pickle::{Object, PthTensors, read_pth_object};
use inference_tensor::{DType, Device, Result, Tensor};

pub use model::{DocLayoutYolo, RawOutputs};
pub use preprocess::{INPUT_SIZE, Letterbox};

use crate::LayoutDetection;

/// The score DocLayout-YOLO's own examples keep detections at.
pub const DEFAULT_THRESHOLD: f32 = 0.2;
/// The checkpoint dict entry holding the pickled model.
const MODEL_KEY: &str = "model";
/// The model attribute mapping class ids to names.
const NAMES_KEY: &str = "names";
/// The training settings entry, whose `imgsz` is the letterbox size the model expects.
const TRAIN_ARGS_KEY: &str = "train_args";
const IMAGE_SIZE_KEY: &str = "imgsz";
const CHECKPOINT_EXT: &str = "pt";

/// The `.pt` checkpoint `path` names: the file itself, or the only one in a directory (a Hugging Face snapshot).
pub fn checkpoint_file(path: &Path) -> Result<PathBuf> {
    if path.is_file() {
        return Ok(path.to_owned());
    }
    let mut found = std::fs::read_dir(path)
        .map_err(inference_tensor::Error::wrap)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == CHECKPOINT_EXT));
    match (found.next(), found.next()) {
        (Some(file), None) => Ok(file),
        (None, _) => inference_tensor::bail!("no .pt checkpoint in {}", path.display()),
        _ => inference_tensor::bail!("more than one .pt checkpoint in {}", path.display()),
    }
}

fn field<'a>(fields: &'a [(Object, Object)], name: &str) -> Option<&'a Object> {
    fields
        .iter()
        .find(|(k, _)| matches!(k, Object::Unicode(k) if k == name))
        .map(|(_, v)| v)
}

/// Class names by id from the pickled model's `names` attribute, and the training image size.
fn read_meta(file: &Path) -> Result<(Vec<String>, usize)> {
    let Object::Dict(top) = read_pth_object(file)? else {
        inference_tensor::bail!("{} is not a checkpoint dict", file.display())
    };
    let size = match field(&top, TRAIN_ARGS_KEY) {
        Some(Object::Dict(args)) => match field(args, IMAGE_SIZE_KEY) {
            Some(Object::Int(size)) if *size > 0 => *size as usize,
            _ => INPUT_SIZE,
        },
        _ => INPUT_SIZE,
    };
    let Some(Object::Build { args, .. }) = field(&top, MODEL_KEY) else {
        inference_tensor::bail!("{} holds no pickled model", file.display())
    };
    let Object::Dict(attrs) = args.as_ref() else {
        inference_tensor::bail!("the pickled model has no attributes")
    };
    let Some(Object::Dict(names)) = field(attrs, NAMES_KEY) else {
        inference_tensor::bail!("the pickled model has no class names")
    };
    let mut labels: Vec<(i64, String)> = names
        .iter()
        .filter_map(|(k, v)| match (k, v) {
            (Object::Int(i), Object::Unicode(n)) => Some((*i as i64, n.clone())),
            (Object::Long(i), Object::Unicode(n)) => Some((*i, n.clone())),
            _ => None,
        })
        .collect();
    labels.sort();
    if labels
        .iter()
        .enumerate()
        .any(|(i, (id, _))| *id != i as i64)
    {
        inference_tensor::bail!("class ids are not 0..{}", labels.len());
    }
    Ok((labels.into_iter().map(|(_, n)| n).collect(), size))
}

/// Loads a DocLayout-YOLO checkpoint and runs detection end to end.
pub struct DocLayoutYoloDetector {
    model: DocLayoutYolo,
    device: Device,
    labels: Vec<String>,
    /// The letterbox's long side.
    input_size: usize,
    /// Set by `with_cpu_threads`; otherwise the crate's shared physical-core pool.
    pool: Option<rayon::ThreadPool>,
}

impl DocLayoutYoloDetector {
    /// `path` is the `.pt` file or a directory holding one.
    pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let file = checkpoint_file(path.as_ref())?;
        let pth = PthTensors::new(&file, Some(MODEL_KEY))?;
        let tensors = pth
            .tensor_infos()
            .keys()
            .map(|name| {
                let t = pth.get(name)?.expect("listed by the archive");
                Ok((name.clone(), t))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        let model = DocLayoutYolo::new(vb)?;
        let (labels, input_size) = read_meta(&file)?;
        if labels.len() != model.classes() {
            inference_tensor::bail!(
                "{} class names for a {}-class head",
                labels.len(),
                model.classes()
            );
        }
        Ok(Self {
            model,
            device: device.clone(),
            labels,
            input_size,
            pool: None,
        })
    }

    /// Gives this detector its own `threads`-thread CPU pool instead of the shared one; no-op off CPU.
    pub fn with_cpu_threads(mut self, threads: usize) -> Result<Self> {
        if self.device.is_cpu() {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads.max(1))
                .build()
                .map_err(inference_tensor::Error::wrap)?;
            self.pool = Some(pool);
        }
        Ok(self)
    }

    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    pub fn model(&self) -> &DocLayoutYolo {
        &self.model
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The long side pages are letterboxed to.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Runs `f` on the detector's CPU pool when on CPU.
    pub fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        if !self.device.is_cpu() {
            return f();
        }
        match (&self.pool, crate::cpu_pool()) {
            (Some(pool), _) | (None, Some(pool)) => pool.install(f),
            (None, None) => f(),
        }
    }

    /// Head outputs for one letterboxed `(3, h, w)` input, on the host: boxes per anchor and `(classes, anchors)` scores.
    pub fn raw(&self, input: &Tensor) -> Result<(Vec<[f32; 4]>, Vec<f32>)> {
        let out = self.model.forward(&input.unsqueeze(0)?)?;
        let boxes = out
            .boxes
            .squeeze(0)?
            .t()?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?
            .into_iter()
            .map(|b| [b[0], b[1], b[2], b[3]])
            .collect();
        let scores = out
            .scores
            .squeeze(0)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok((boxes, scores))
    }

    pub fn detect(&self, image: &RgbImage, threshold: f32) -> Result<Vec<LayoutDetection>> {
        self.install(|| {
            let (input, letterbox) = preprocess::preprocess(image, self.input_size, &self.device)?;
            let (boxes, scores) = self.raw(&input)?;
            Ok(postprocess::detections(
                &boxes,
                &scores,
                &self.labels,
                threshold,
                &letterbox,
            ))
        })
    }

    /// Pages letterbox to their own sizes, so each runs on its own.
    pub fn detect_batch(
        &self,
        images: &[RgbImage],
        threshold: f32,
    ) -> Result<Vec<Vec<LayoutDetection>>> {
        images.iter().map(|im| self.detect(im, threshold)).collect()
    }
}
