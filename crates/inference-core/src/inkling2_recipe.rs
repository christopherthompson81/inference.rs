//! Inkling 2's own checkpoint written as the GGUF its fast paths read (recipe C), with no source GGUF: the model's
//! tensors read from the checkpoint by its names and laid out as llama.cpp's `inkling2_loop` GGUFs store them, the
//! routed experts' NVFP4 blocks as stored and the rest quantized by the recipe's rules.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use inference_quant::requantize::{
    Quantizer, RequantizePlan, RequantizeSource, RequantizeSummary, requantize_source,
    writable_type,
};
use inference_tensor::quantized::gguf_file::{self, Header, RawTensorInfo, Value, VersionedMagic};
use inference_tensor::{DType, Device};
use regex::Regex;

use crate::utils::varbuilder_utils::{DeviceForLoadTensor, from_mmaped_safetensors};

// llama.cpp's F32 and BF16 type ids
const GGML_F32: u32 = 0;
const GGML_BF16: u32 = 30;
// the drafter keeps its own file, which the text model does not read
const DRAFTER_FILE: &str = "mtp.safetensors";

/// Recipe C (inkling2_speed investigation): every projection Q8_0, the head and embedding Q6_K, and everything else
/// F32; the NVFP4 expert stacks as stored, and the ones stored otherwise (MXFP8) as `INKLING2_RECIPE_C_EXPERTS`.
pub const INKLING2_RECIPE_C: &[(&str, &str)] = &[
    (r"^(output|token_embd)\.weight$", "q6_k"),
    (
        r"^blk\.\d+\.attn_(q|k|v|output|gate|rel|sink_proj|k_conv_proj|v_conv_proj)\.weight$",
        "q8_0",
    ),
];
/// What a checkpoint is written as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Inkling2Recipe {
    /// Recipe C, the GGUF its fast paths were measured on.
    C,
    /// The dense tensors at F32 and the NVFP4 experts as stored (the others as recipe C's): the checkpoint's own
    /// numbers, a base for measuring a recipe against.
    Unquantized,
}

/// Recipe C's types for the expert stacks a checkpoint stores other than as NVFP4: down's rows of 1152 take no K-quant.
pub const INKLING2_RECIPE_C_EXPERTS: &[(&str, &str)] =
    &[("ffn_gate_up_exps", "q6_k"), ("ffn_down_exps", "q8_0")];

/// Whether `dir` holds one of Inkling 2's own checkpoints (thinkingmachines' releases).
pub fn is_inkling2_checkpoint(dir: &Path) -> bool {
    cfg!(feature = "models-other")
        && inference_gguf::official_checkpoint_experts(dir).is_ok_and(|experts| experts.is_some())
}

#[cfg(feature = "models-other")]
// a dense tensor: the model's name it reads from the checkpoint by, and whether the GGUF stores its transpose
struct Dense {
    native: String,
    transposed: bool,
}

#[cfg(feature = "models-other")]
/// One of Inkling 2's own checkpoints laid out as llama.cpp's `inkling2_loop` GGUF: its dense tensors at F32 (the
/// checkpoint's BF16 exactly), its expert stacks listed for the plan's checkpoint to read.
pub struct Inkling2CheckpointSource {
    header: Header,
    metadata: Vec<(String, Value)>,
    official: inference_models_other::Inkling2OfficialTensors,
    dense: HashMap<String, Dense>,
    // the last tensor read, as the requantizer reads one at a time
    last: Mutex<Option<(String, Arc<Vec<u8>>)>>,
}

fn safetensors(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .filter(|p| p.file_name().is_none_or(|name| name != DRAFTER_FILE))
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

#[cfg(feature = "models-other")]
impl Inkling2CheckpointSource {
    pub fn open(dir: &Path) -> Result<Self> {
        let config: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("config.json")).context("reading config.json")?,
        )?;
        let text = config
            .get("text_config")
            .context("Inkling 2's config has no `text_config`")?;
        let cfg = inference_models_other::inkling2::Config::from_json(&text.to_string())?;
        let vb = from_mmaped_safetensors(
            safetensors(dir)?,
            Some(DType::F32),
            &Device::Cpu,
            Vec::new(),
            true,
            None,
            |_| true,
            Arc::new(|_| DeviceForLoadTensor::Base),
        )?;
        let official = inference_models_other::Inkling2OfficialTensors::open(&cfg, vb)?
            .with_context(|| format!("`{}` is no Inkling 2 checkpoint", dir.display()))?;
        let mut tensor_order = Vec::new();
        let mut tensor_infos = HashMap::new();
        let mut dense = HashMap::new();
        for (native, shape) in official.tensors() {
            let target = inference_gguf::inkling2_gguf_tensor(native, shape)
                .with_context(|| format!("Inkling 2: no GGUF tensor for `{native}`"))?;
            tensor_infos.insert(
                target.name.clone(),
                RawTensorInfo {
                    ggml_dtype: GGML_F32,
                    shape: target.dims.clone().into(),
                    offset: 0,
                },
            );
            tensor_order.push(target.name.clone());
            dense.insert(
                target.name,
                Dense {
                    native: native.to_string(),
                    transposed: target.transposed,
                },
            );
        }
        let (experts, inter, hidden) =
            (cfg.n_routed_experts, cfg.intermediate_size, cfg.hidden_size);
        for layer in 0..cfg.num_hidden_layers {
            for (role, dims) in [
                ("ffn_gate_up_exps", vec![experts, 2 * inter, hidden]),
                ("ffn_down_exps", vec![experts, hidden, inter]),
            ] {
                let name = format!("blk.{layer}.{role}.weight");
                tensor_infos.insert(
                    name.clone(),
                    RawTensorInfo {
                        ggml_dtype: GGML_BF16,
                        shape: dims.into(),
                        offset: 0,
                    },
                );
                tensor_order.push(name);
            }
        }
        tensor_order.sort_by_key(|name| gguf_order(name));
        let metadata = inference_gguf::inkling2_checkpoint_metadata(dir)?;
        let header = Header {
            magic: VersionedMagic::GgufV3,
            metadata: metadata.iter().cloned().collect(),
            large_arrays: HashMap::new(),
            tensor_infos,
            tensor_data_offset: 0,
            metadata_span: 0..0,
            tensor_order,
        };
        Ok(Self {
            header,
            metadata,
            official,
            dense,
            last: Mutex::new(None),
        })
    }

    fn bytes(&self, name: &str) -> Result<Arc<Vec<u8>>> {
        let mut last = self.last.lock().expect("checkpoint source poisoned");
        if let Some((held, bytes)) = last.as_ref()
            && held == name
        {
            return Ok(bytes.clone());
        }
        let dense = self.dense.get(name).with_context(|| {
            format!(
                "Inkling 2: `{name}` is read from the checkpoint's expert stacks, not as values"
            )
        })?;
        let mut values = self.official.values(&dense.native)?;
        if dense.transposed {
            values = values.t()?.contiguous()?;
        }
        let bytes = Arc::new(
            values
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        );
        *last = Some((name.to_string(), bytes.clone()));
        Ok(bytes)
    }
}

#[cfg(feature = "models-other")]
// llama.cpp's order: the embedding first, then each block's tensors, then the output's
fn gguf_order(name: &str) -> (usize, usize, String) {
    match name
        .strip_prefix("blk.")
        .and_then(|rest| rest.split_once('.'))
    {
        Some((layer, role)) => (1, layer.parse().unwrap_or(usize::MAX), role.to_string()),
        None if name.starts_with("token_embd") => (0, 0, name.to_string()),
        None => (2, 0, name.to_string()),
    }
}

#[cfg(feature = "models-other")]
impl RequantizeSource for Inkling2CheckpointSource {
    fn header(&self) -> &Header {
        &self.header
    }

    fn metadata(&self) -> inference_tensor::Result<(Vec<u8>, usize)> {
        let entries = self
            .metadata
            .iter()
            .map(|(name, value)| (name.as_str(), value))
            .collect::<Vec<_>>();
        Ok((gguf_file::metadata_bytes(&entries)?, entries.len()))
    }

    fn read(&self, name: &str, offset: u64, buf: &mut [u8]) -> inference_tensor::Result<()> {
        let bytes = self.bytes(name).map_err(inference_tensor::Error::msg)?;
        let start = usize::try_from(offset).map_err(inference_tensor::Error::wrap)?;
        let held = bytes
            .get(start..start + buf.len())
            .ok_or_else(|| inference_tensor::Error::msg(format!("{name}: a read past its end")))?;
        buf.copy_from_slice(held);
        Ok(())
    }
}

#[cfg(feature = "models-other")]
/// The plan writing `recipe_kind` from the checkpoint in `dir`, quantizing on `gpu` where its quantizers cover a type.
pub fn inkling2_recipe_plan(
    dir: &Path,
    source: &Inkling2CheckpointSource,
    recipe_kind: Inkling2Recipe,
    gpu: Option<Device>,
) -> Result<RequantizePlan> {
    use inference_quant::requantize::CheckpointSource;
    let experts = inference_gguf::official_checkpoint_experts(dir)?
        .with_context(|| format!("`{}` is no Inkling 2 checkpoint", dir.display()))?;
    // each tensor's recipe type, or the nearest its rows take: Q6_K's blocks need 256-wide rows, Q8_0's 32
    let recipe = INKLING2_RECIPE_C
        .iter()
        .map(|(pattern, ty)| Ok((Regex::new(pattern)?, writable_type(ty)?)))
        .collect::<Result<Vec<_>>>()?;
    let q8_0 = writable_type("q8_0")?;
    let mut rules = Vec::new();
    for name in &source.header.tensor_order {
        let stored_otherwise = experts.holds_values(name) && !experts.holds_nvfp4(name);
        let ty = match recipe.iter().find(|(re, _)| re.is_match(name)) {
            Some((_, ty)) => Some(*ty),
            None if stored_otherwise => INKLING2_RECIPE_C_EXPERTS
                .iter()
                .find(|(role, _)| name.ends_with(&format!(".{role}.weight")))
                .map(|(_, ty)| writable_type(ty))
                .transpose()?,
            None => None,
        };
        let Some(ty) = ty else { continue };
        if recipe_kind == Inkling2Recipe::Unquantized && !stored_otherwise {
            continue;
        }
        let cols = source.header.tensor_infos[name]
            .shape
            .dims()
            .last()
            .copied()
            .unwrap_or(1);
        let Some(ty) = [ty, q8_0]
            .into_iter()
            .find(|ty| ty.row_bytes(cols).is_some())
        else {
            continue;
        };
        rules.push((Regex::new(&format!("^{}$", regex::escape(name)))?, ty));
    }
    Ok(RequantizePlan {
        rules,
        imatrix: None,
        quantizer: Quantizer::IkLlama,
        reuse: Vec::new(),
        only: None,
        gpu,
        hessians: None,
        hessian_weights_only: false,
        gptq_damp: inference_quant::requantize::DEFAULT_GPTQ_DAMP,
        report_hessians: None,
        checkpoint: Some(Box::new(experts)),
    })
}

#[cfg(feature = "models-other")]
/// `recipe` written from the Inkling 2 checkpoint in `dir` to `dst`; `progress` hears each tensor as it starts.
pub fn write_inkling2_recipe(
    dir: &Path,
    dst: &Path,
    recipe: Inkling2Recipe,
    gpu: Option<Device>,
    progress: impl FnMut(usize, usize, &str),
) -> Result<RequantizeSummary> {
    let source = Inkling2CheckpointSource::open(dir)?;
    let plan = inkling2_recipe_plan(dir, &source, recipe, gpu)?;
    let _pool = plan.gpu.as_ref().map(crate::retain_memory_pool);
    Ok(requantize_source(&source, dst, &plan, progress)?)
}

// what the recipe reads beside the weights; the drafter's weights come along for a later MTP load
const CHECKPOINT_FILES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "hf_quant_config.json",
];
// a file name the cache keys on changes with what it holds: bumping this rebuilds every cached recipe
const RECIPE_VERSION: u32 = 1;
const RECIPE_STEM: &str = "recipe-c";
// 64-bit FNV-1a, stable across builds (std's hasher is not)
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv(hash: u64, bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(hash, |h, b| (h ^ u64::from(*b)).wrapping_mul(FNV_PRIME))
}

fn names_inkling2(config: &serde_json::Value) -> bool {
    config["text_config"]["model_type"]
        .as_str()
        .is_some_and(|ty| ty.starts_with("inkling2"))
}

/// The Inkling 2 checkpoint `model_id` names, as a local directory: the directory itself, or a hub repo's snapshot
/// with its files fetched; `None` when it names another model.
pub fn inkling2_checkpoint_dir(
    model_id: &str,
    token_source: &crate::TokenSource,
    revision: Option<&str>,
) -> Result<Option<PathBuf>> {
    let local = Path::new(model_id);
    if local.is_dir() {
        return Ok(is_inkling2_checkpoint(local).then(|| local.to_path_buf()));
    }
    let revision = revision.unwrap_or("main");
    let api = crate::pipeline::hf::build_api(token_source, true)
        .map_err(anyhow::Error::msg)?
        .repo(hf_hub::Repo::with_revision(
            model_id.to_string(),
            hf_hub::RepoType::Model,
            revision.to_string(),
        ));
    let id = Path::new(model_id);
    let Ok(config_path) = crate::pipeline::hf::get_file(&api, id, "config.json", revision) else {
        return Ok(None);
    };
    let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
    if !names_inkling2(&config) {
        return Ok(None);
    }
    for file in CHECKPOINT_FILES {
        crate::pipeline::hf::try_get_file(&api, id, file, revision).map_err(anyhow::Error::msg)?;
    }
    let files = crate::pipeline::hf::list_repo_files(&api, id, true, revision)
        .map_err(anyhow::Error::msg)?;
    for file in files.iter().filter(|f| f.ends_with(".safetensors")) {
        crate::pipeline::hf::get_file(&api, id, file, revision).map_err(anyhow::Error::msg)?;
    }
    let dir = config_path
        .parent()
        .context("a snapshot file has a directory")?
        .to_path_buf();
    Ok(is_inkling2_checkpoint(&dir).then_some(dir))
}

/// Where recipe C of the checkpoint in `dir` is cached under `root`: keyed on its config and its weight files' names
/// and sizes, so another release or recipe version builds its own.
pub fn inkling2_recipe_path(dir: &Path, root: &Path) -> Result<PathBuf> {
    let mut hash = fnv(FNV_OFFSET, &RECIPE_VERSION.to_le_bytes());
    hash = fnv(hash, &std::fs::read(dir.join("config.json"))?);
    for path in safetensors(dir)? {
        hash = fnv(
            hash,
            path.file_name().unwrap_or_default().as_encoded_bytes(),
        );
        hash = fnv(hash, &std::fs::metadata(&path)?.len().to_le_bytes());
    }
    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "inkling2".to_string());
    // a directory of its own, as a GGUF load is named after its directory
    Ok(root
        .join(format!("{name}-{RECIPE_STEM}-{hash:016x}"))
        .join(format!("{name}-{RECIPE_STEM}.gguf")))
}

/// The bytes recipe C of the checkpoint in `dir` takes: its weights less what quantizing the dense ones saves, as an
/// upper bound for the space a build needs.
pub fn inkling2_recipe_bytes(dir: &Path) -> Result<u64> {
    safetensors(dir)?
        .iter()
        .map(|path| Ok(std::fs::metadata(path)?.len()))
        .sum()
}

/// Recipe C of the checkpoint in `dir` at `path`, built there first if it is not: written beside it and renamed into
/// place, so an interrupted build leaves nothing a later load would take.
// the log's sizes are GiB to two places
#[allow(clippy::cast_precision_loss)]
pub fn ensure_inkling2_recipe(dir: &Path, path: &Path, gpu: Option<Device>) -> Result<()> {
    if path.is_file() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let partial = path.with_extension("gguf.partial");
    tracing::info!(
        "Writing recipe C of `{}` to `{}` (once; later loads read it)",
        dir.display(),
        path.display()
    );
    let start = std::time::Instant::now();
    #[cfg(not(feature = "models-other"))]
    let summary: Result<RequantizeSummary> = {
        let _ = gpu;
        Err(anyhow::anyhow!("this build has no Inkling 2 model"))
    };
    #[cfg(feature = "models-other")]
    let summary = write_inkling2_recipe(dir, &partial, Inkling2Recipe::C, gpu, |i, n, name| {
        tracing::debug!("[{}/{n}] {name}", i + 1);
    });
    let summary = match summary {
        Ok(summary) => summary,
        Err(error) => {
            let _ = std::fs::remove_file(&partial);
            return Err(error);
        }
    };
    std::fs::rename(&partial, path)?;
    tracing::info!(
        "Wrote recipe C ({:.2} GiB, {} tensors) in {:.0} s",
        summary.bytes_out as f64 / (1u64 << 30) as f64,
        summary.tensors,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
