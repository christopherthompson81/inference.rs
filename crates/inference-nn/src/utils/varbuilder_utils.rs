//! Utilities for creating a VarBuilder from a VarMap loaded from tensor storage formats.

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::Arc,
    thread::{self, JoinHandle},
};

use inference_quant::{
    ShardedSafeTensors, ShardedVarBuilder,
    safetensors::{AdapterFold, MmapedSafetensors},
};
use inference_tensor::{DType, Device, Result, Tensor, pickle::PthTensors};
use regex::Regex;

use crate::utils::progress::{NiceProgressBar, multi_progress_unless_silent};
use indicatif::MultiProgress;

const INFERENCE_RS_NO_MMAP: &str = "INFERENCE_RS_NO_MMAP";
const SAFETENSORS_EXTENSION: &str = "safetensors";
const PICKLE_EXTENSIONS: &[&str] = &["pth", "pt", "bin"];
// a PEFT adapter shipped beside a checkpoint's weights
pub const CHECKPOINT_ADAPTER_CONFIG: &str = "adapter_config.json";
pub const CHECKPOINT_ADAPTER_WEIGHTS: &str = "adapter_model.safetensors";

trait TensorLoaderBackend {
    fn get_names(&self) -> Vec<String>;
    fn load_name(&self, name: &str, device: &Device, dtype: Option<DType>) -> Result<Tensor>;
}

struct SafetensorBackend(MmapedSafetensors);

impl TensorLoaderBackend for SafetensorBackend {
    fn get_names(&self) -> Vec<String> {
        self.0
            .tensors()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>()
    }
    fn load_name(&self, name: &str, device: &Device, dtype: Option<DType>) -> Result<Tensor> {
        self.0.load(name, device, dtype)
    }
}

struct PickleBackend(PthTensors);

impl TensorLoaderBackend for PickleBackend {
    fn get_names(&self) -> Vec<String> {
        self.0.tensor_infos().keys().cloned().collect::<Vec<_>>()
    }
    fn load_name(&self, name: &str, device: &Device, _dtype: Option<DType>) -> Result<Tensor> {
        self.0
            .get(name)?
            .ok_or(inference_tensor::Error::Msg(format!(
                "Could not load tensor {name}"
            )))?
            .to_device(device)
    }
}

pub enum DeviceForLoadTensor {
    Base,
    Idx(usize),
}

/// Load tensors into a VarBuilder backed by a VarMap using MmapedSafetensors.
/// Set `silent` to not show a progress bar.
///
/// # Predicate semantics:
/// - If `regexes` is specified, this will be used in `make_dummy_predicate` based on `.any`
/// - Otherwise, only include keys for which predicate evaluates to true.
#[allow(clippy::too_many_arguments)]
pub fn from_mmaped_safetensors(
    paths: Vec<PathBuf>,
    dtype: Option<DType>,
    base_device: &Device,
    layer_devices: Vec<Option<Device>>,
    silent: bool,
    make_dummy_regexes: Option<Arc<Vec<Regex>>>,
    predicate: impl Fn(String) -> bool + Send + Sync + Clone + 'static,
    get_device_for_tensor: Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>,
) -> Result<ShardedVarBuilder> {
    // Erased here so the loading body compiles once rather than once per caller's closure type.
    load_safetensors(
        paths,
        dtype,
        base_device,
        layer_devices,
        silent,
        make_dummy_regexes,
        Arc::new(predicate),
        get_device_for_tensor,
    )
}

#[allow(clippy::too_many_arguments)]
fn load_safetensors(
    paths: Vec<PathBuf>,
    dtype: Option<DType>,
    base_device: &Device,
    layer_devices: Vec<Option<Device>>,
    silent: bool,
    make_dummy_regexes: Option<Arc<Vec<Regex>>>,
    predicate: Arc<dyn Fn(String) -> bool + Send + Sync + 'static>,
    get_device_for_tensor: Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>,
) -> Result<ShardedVarBuilder> {
    // only safetensors map in place; pickle checkpoints read through the per-file loader
    let has_pickle = paths.iter().any(|path| {
        path.extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| PICKLE_EXTENSIONS.contains(&ext))
    });
    let use_no_mmap = std::env::var(INFERENCE_RS_NO_MMAP).is_ok_and(|x| x == "1") || has_pickle;
    let fold = checkpoint_adapter(&paths)?;
    if !use_no_mmap {
        if !silent {
            tracing::debug!("Loading model using mmap strategy.");
        }
        return Ok(unsafe {
            ShardedSafeTensors::sharded_with_fold(
                &paths,
                dtype.unwrap_or(DType::F16),
                base_device,
                make_dummy_regexes,
                predicate,
                fold,
            )?
        });
    }

    // One MultiProgress so the per-file bars from the loader threads stack instead of overwriting each other.
    let progress = multi_progress_unless_silent(silent);
    let make_dummy: Arc<dyn Fn(&str) -> bool + Send + Sync> = match make_dummy_regexes.clone() {
        Some(regexes) => Arc::new(move |key| regexes.iter().any(|r| r.is_match(key))),
        None => Arc::new(|_| false),
    };
    #[allow(clippy::type_complexity)]
    let handles: Vec<JoinHandle<Result<HashMap<String, Tensor>>>> = paths
        .into_iter()
        .map(|path| {
            let base_device = base_device.clone();
            let layer_devices = layer_devices.clone();
            let get_device_for_tensor = get_device_for_tensor.clone();
            let predicate = predicate.clone();
            let make_dummy = make_dummy.clone();
            let progress = progress.clone();
            thread::spawn(move || {
                load_tensors_from_path(TensorLoad {
                    path: &path,
                    base_device: &base_device,
                    layer_devices,
                    get_device_for_tensor,
                    dtype,
                    progress,
                    predicate: &*predicate,
                    make_dummy_predicate: &*make_dummy,
                })
            })
        })
        .collect();

    let mut ws = HashMap::new();
    // Wait until all spawned threads have finished loading tensors:
    while !handles.iter().all(|h| h.is_finished()) {}
    for h in handles {
        ws.extend(h.join().unwrap()?);
    }
    if let Some(fold) = fold {
        for target in fold.targets() {
            if let Some(weight) = ws.get_mut(target) {
                let delta = fold.delta(target, weight.device())?.expect("a target");
                *weight = (weight.to_dtype(DType::F32)? + delta)?.to_dtype(weight.dtype())?;
            }
        }
    }

    // TODO(EricLBuehler): separation of concerns.
    // This is to have WNA16 for GPTQ which is required. No bf16 for GPTQ
    Ok(ShardedSafeTensors::wrap_with_dummy_regexes(
        ws,
        dtype.unwrap_or(DType::F16),
        base_device.clone(),
        make_dummy_regexes,
    ))
}

/// The PEFT adapter a checkpoint ships beside its weight files, which transformers folds in on load.
fn checkpoint_adapter(paths: &[PathBuf]) -> Result<Option<Arc<AdapterFold>>> {
    let Some(dir) = paths.first().and_then(|path| path.parent()) else {
        return Ok(None);
    };
    let (config, weights) = (
        dir.join(CHECKPOINT_ADAPTER_CONFIG),
        dir.join(CHECKPOINT_ADAPTER_WEIGHTS),
    );
    if !config.is_file() || !weights.is_file() || paths.contains(&weights) {
        return Ok(None);
    }
    let fold = unsafe { AdapterFold::open(&config, &weights)? };
    tracing::info!(
        "Folding the checkpoint's adapter `{}` into {} weights.",
        weights.display(),
        fold.len()
    );
    Ok(Some(Arc::new(fold)))
}

/// One checkpoint file to load and how to place and filter its tensors.
struct TensorLoad<'a> {
    path: &'a PathBuf,
    base_device: &'a Device,
    layer_devices: Vec<Option<Device>>,
    get_device_for_tensor: Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>,
    dtype: Option<DType>,
    progress: Option<MultiProgress>,
    predicate: &'a dyn Fn(String) -> bool,
    make_dummy_predicate: &'a dyn Fn(&str) -> bool,
}

fn load_tensors_from_path(load: TensorLoad<'_>) -> Result<HashMap<String, Tensor>> {
    let TensorLoad {
        path,
        base_device,
        layer_devices,
        get_device_for_tensor,
        dtype,
        progress,
        predicate,
        make_dummy_predicate,
    } = load;
    let tensors: Box<dyn TensorLoaderBackend> = match path
        .extension()
        .expect("Expected extension")
        .to_str()
        .expect("Expected to convert")
    {
        SAFETENSORS_EXTENSION => {
            Box::new(SafetensorBackend(unsafe { MmapedSafetensors::new(path)? }))
        }
        "pth" | "pt" | "bin" => Box::new(PickleBackend(inference_tensor::pickle::PthTensors::new(
            path, None,
        )?)),
        other => inference_tensor::bail!(
            "Unexpected extension `{other}`, this should have been handled by `get_model_paths`."
        ),
    };

    // Extracts the tensor name and processes it, filtering tensors and deriving the key name:
    let names_only = tensors
        .get_names()
        .into_iter()
        .filter(|x| predicate(x.to_string()));
    let iter = names_only
        .map(|name| {
            let key = name.replace("base_model.model.model", "model");
            (name, key)
        })
        .collect::<Vec<_>>();

    // Take the filtered list of tensors to load, store with derived lookup key:
    let mut loaded_tensors = HashMap::new();
    if !iter.is_empty() {
        let pairs: Box<dyn Iterator<Item = (String, String)>> = match &progress {
            Some(multi) => {
                // each shard loads on its own thread, so each is a stage of its own
                let stage = format!(
                    "Loading {}",
                    path.file_name().unwrap_or_default().to_string_lossy()
                );
                Box::new(NiceProgressBar::<_, 'b', _>(iter.into_iter(), stage, multi).into_iter())
            }
            None => Box::new(iter.into_iter()),
        };
        for (load_name, key_name) in pairs {
            if !make_dummy_predicate(&load_name) {
                let dev = match get_device_for_tensor(load_name.clone()) {
                    DeviceForLoadTensor::Base => base_device,
                    DeviceForLoadTensor::Idx(i) => layer_devices
                        .get(i)
                        .and_then(|d| d.as_ref())
                        .unwrap_or(base_device),
                };
                // If making a dummy, don't add the tensor. `inference_quant` handles this!
                let tensor = tensors.load_name(&load_name, dev, dtype)?;

                loaded_tensors.insert(key_name, tensor);
            }
        }
    }

    Ok(loaded_tensors)
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use inference_quant::Shard;
    use inference_tensor::{DType, Device, Tensor};

    use super::{
        CHECKPOINT_ADAPTER_CONFIG, CHECKPOINT_ADAPTER_WEIGHTS, DeviceForLoadTensor,
        from_mmaped_safetensors,
    };

    const OUT: usize = 4;
    const IN: usize = 3;
    const RANK: usize = 2;
    const ALPHA: f64 = 4.0;

    fn write_checkpoint(dir: &std::path::Path, config: serde_json::Value) -> anyhow::Result<()> {
        let w = Tensor::arange(0f32, (OUT * IN) as f32, &Device::Cpu)?.reshape((OUT, IN))?;
        let other = Tensor::ones(IN, DType::F32, &Device::Cpu)?;
        inference_tensor::safetensors::save(
            &HashMap::from([("layer.proj.weight", w), ("layer.norm.weight", other)]),
            dir.join("model.safetensors"),
        )?;
        let a = Tensor::arange(0f32, (RANK * IN) as f32, &Device::Cpu)?.reshape((RANK, IN))?;
        let b =
            (Tensor::arange(0f32, (OUT * RANK) as f32, &Device::Cpu)?.reshape((OUT, RANK))? * 0.1)?;
        inference_tensor::safetensors::save(
            &HashMap::from([
                ("base_model.model.layer.proj.lora_A.weight", a),
                ("base_model.model.layer.proj.lora_B.weight", b),
            ]),
            dir.join(CHECKPOINT_ADAPTER_WEIGHTS),
        )?;
        std::fs::write(dir.join(CHECKPOINT_ADAPTER_CONFIG), config.to_string())?;
        Ok(())
    }

    fn load(dir: &std::path::Path) -> inference_tensor::Result<inference_quant::ShardedVarBuilder> {
        from_mmaped_safetensors(
            vec![dir.join("model.safetensors")],
            Some(DType::F32),
            &Device::Cpu,
            vec![],
            true,
            None,
            |_| true,
            Arc::new(|_| DeviceForLoadTensor::Base),
        )
    }

    #[test]
    fn a_checkpoint_adapter_folds_into_the_weights_it_targets() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        write_checkpoint(
            dir.path(),
            serde_json::json!({"r": RANK, "lora_alpha": ALPHA}),
        )?;
        let vb = load(dir.path())?;
        let w = Tensor::arange(0f32, (OUT * IN) as f32, &Device::Cpu)?.reshape((OUT, IN))?;
        let a = Tensor::arange(0f32, (RANK * IN) as f32, &Device::Cpu)?.reshape((RANK, IN))?;
        let b =
            (Tensor::arange(0f32, (OUT * RANK) as f32, &Device::Cpu)?.reshape((OUT, RANK))? * 0.1)?;
        let expected = (w + (b.matmul(&a)? * (ALPHA / RANK as f64))?)?;
        let folded = vb.pp("layer").pp("proj").get((OUT, IN), "weight")?;
        assert_eq!(folded.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
        // a tensor-parallel rank reads its rows of the folded weight
        let half = vb.pp("layer").pp("proj").get_with_hints(
            (OUT / 2, IN),
            "weight",
            Shard::Simple {
                dim: 0,
                rank: 1,
                world_size: 2,
            },
        )?;
        assert_eq!(
            half.to_vec2::<f32>()?,
            expected.narrow(0, OUT / 2, OUT / 2)?.to_vec2::<f32>()?
        );
        // an untargeted weight reads as stored
        let norm = vb.pp("layer").pp("norm").get(IN, "weight")?;
        assert_eq!(norm.to_vec1::<f32>()?, vec![1f32; IN]);
        Ok(())
    }

    #[test]
    fn adapters_it_cannot_fold_are_refused() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        write_checkpoint(
            dir.path(),
            serde_json::json!({"r": RANK, "lora_alpha": ALPHA, "use_dora": true}),
        )?;
        let error = load(dir.path())
            .err()
            .expect("a DoRA adapter is refused")
            .to_string();
        assert!(error.contains("DoRA"), "{error}");
        Ok(())
    }
}
