//! Utilities for creating a VarBuilder from a VarMap loaded from tensor storage formats.

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::Arc,
    thread::{self, JoinHandle},
};

use inference_quant::{ShardedSafeTensors, ShardedVarBuilder, safetensors::MmapedSafetensors};
use inference_tensor::{DType, Device, Result, Tensor, pickle::PthTensors};
use regex::Regex;

use crate::lora::LoraConfig;
use crate::utils::progress::{NiceProgressBar, new_multi_progress};
use indicatif::MultiProgress;

const INFERENCE_RS_NO_MMAP: &str = "INFERENCE_RS_NO_MMAP";

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
    xlora_paths: Vec<PathBuf>,
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
        xlora_paths,
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
    xlora_paths: Vec<PathBuf>,
    dtype: Option<DType>,
    base_device: &Device,
    layer_devices: Vec<Option<Device>>,
    silent: bool,
    make_dummy_regexes: Option<Arc<Vec<Regex>>>,
    predicate: Arc<dyn Fn(String) -> bool + Send + Sync + 'static>,
    get_device_for_tensor: Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>,
) -> Result<ShardedVarBuilder> {
    let use_no_mmap = std::env::var(INFERENCE_RS_NO_MMAP).is_ok_and(|x| x == "1");
    if xlora_paths.is_empty() && !use_no_mmap {
        if !silent {
            tracing::debug!("Loading model using mmap strategy.");
        }
        return Ok(unsafe {
            ShardedSafeTensors::sharded(
                &paths,
                dtype.unwrap_or(DType::F16),
                base_device,
                make_dummy_regexes,
                predicate,
            )?
        });
    }

    // One MultiProgress so the per-file bars from the loader threads stack instead of overwriting each other.
    let progress = (!silent).then(new_multi_progress);
    let make_dummy: Arc<dyn Fn(&str) -> bool + Send + Sync> = match make_dummy_regexes.clone() {
        Some(regexes) => Arc::new(move |key| regexes.iter().any(|r| r.is_match(key))),
        None => Arc::new(|_| false),
    };
    let loaders = paths.into_iter().map(|path| (path, None)).chain(
        xlora_paths
            .into_iter()
            .enumerate()
            .map(|(i, path)| (path, Some(i + 1))),
    );
    #[allow(clippy::type_complexity)]
    let handles: Vec<JoinHandle<Result<HashMap<String, Tensor>>>> = loaders
        .map(|(path, xlora_index)| {
            let base_device = base_device.clone();
            let layer_devices = layer_devices.clone();
            let get_device_for_tensor = get_device_for_tensor.clone();
            let predicate = predicate.clone();
            let make_dummy = make_dummy.clone();
            let progress = progress.clone();
            thread::spawn(move || {
                let load = TensorLoad {
                    path: &path,
                    base_device: &base_device,
                    layer_devices,
                    get_device_for_tensor,
                    dtype,
                    progress,
                    predicate: &*predicate,
                    make_dummy_predicate: &*make_dummy,
                };
                match xlora_index {
                    None => Common.load_tensors_from_path(load),
                    Some(adapter_index) => XLora { adapter_index }.load_tensors_from_path(load),
                }
            })
        })
        .collect();

    let mut ws = HashMap::new();
    // Wait until all spawned threads have finished loading tensors:
    while !handles.iter().all(|h| h.is_finished()) {}
    for h in handles {
        ws.extend(h.join().unwrap()?);
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

pub fn load_preload_adapters(
    paths: &Option<HashMap<String, (PathBuf, LoraConfig)>>,
    dtype: DType,
    device: &Device,
    silent: bool,
) -> Result<Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>> {
    if let Some(paths) = paths {
        let mut map = HashMap::new();
        for (name, (path, config)) in paths {
            let loader = Common;
            let loaded_tensors = loader.load_tensors_from_path(TensorLoad {
                path,
                base_device: device,
                layer_devices: vec![None],
                get_device_for_tensor: Arc::new(|_| DeviceForLoadTensor::Base),
                dtype: Some(dtype),
                progress: (!silent).then(new_multi_progress),
                predicate: &|_| true,
                make_dummy_predicate: &|_| false,
            })?;

            // TODO(EricLBuehler): separation of concerns.
            // This is to have WNA16 for GPTQ which is required. No bf16 for GPTQ
            let vb = ShardedSafeTensors::wrap(loaded_tensors, dtype, device.clone());

            map.insert(name.clone(), (vb, config.clone()));
        }
        Ok(Some(map))
    } else {
        Ok(None)
    }
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

// Presently this logic only needs to diverge for X-LoRA support via `get_name_key_pairs()`
trait LoadTensors {
    fn load_tensors_from_path(&self, load: TensorLoad<'_>) -> Result<HashMap<String, Tensor>> {
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
            "safetensors" => Box::new(SafetensorBackend(unsafe { MmapedSafetensors::new(path)? })),
            "pth" | "pt" | "bin" => Box::new(PickleBackend(
                inference_tensor::pickle::PthTensors::new(path, None)?,
            )),
            other => inference_tensor::bail!(
                "Unexpected extension `{other}`, this should have been handled by `get_model_paths`."
            ),
        };

        // Extracts the tensor name and processes it, filtering tensors and deriving the key name:
        let names_only = tensors
            .get_names()
            .into_iter()
            .filter(|x| predicate(x.to_string()));
        let iter = self.get_name_key_pairs(names_only).collect::<Vec<_>>();

        // Take the filtered list of tensors to load, store with derived lookup key:
        let mut loaded_tensors = HashMap::new();
        if !iter.is_empty() {
            let pairs: Box<dyn Iterator<Item = (String, String)>> = match &progress {
                Some(multi) => Box::new(
                    NiceProgressBar::<_, 'b'>(iter.into_iter(), "Loading", multi).into_iter(),
                ),
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

    fn get_name_key_pairs(
        &self,
        tensors: impl Iterator<Item = String>,
    ) -> impl Iterator<Item = (String, String)> {
        tensors.map(|name| {
            let new_name = name.replace("base_model.model.model", "model");

            (name, new_name)
        })
    }
}

struct Common;
impl LoadTensors for Common {}

struct XLora {
    // Matches the associated path instance for reference in `get_name_key_pairs()`
    adapter_index: usize,
}

impl LoadTensors for XLora {
    fn get_name_key_pairs(
        &self,
        tensors: impl Iterator<Item = String>,
    ) -> impl Iterator<Item = (String, String)> {
        let expectation = "tensor name `{new_name}` should have substring `.lora`";

        tensors
            .filter(|name| !name.contains("internal_xlora_classifier"))
            .map(|name| {
                let mut new_name = name.replace("base_model.model.model", "model");
                // TODO: Add better context to describe intent / requirement:
                let pos = new_name.find(".lora").expect(expectation);
                new_name.insert_str(pos + 7, &format!(".{}", self.adapter_index));

                (name, new_name)
            })
    }
}
