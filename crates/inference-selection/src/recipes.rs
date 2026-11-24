//! Load-time recipes: a checkpoint whose fast paths read another format loads as that format, built from it once into
//! a cache. Inkling 2's own checkpoints load as recipe C (its NVFP4 experts as stored, the rest quantized), the GGUF
//! its kernels were measured on.

use std::path::{Path, PathBuf};

use inference_core::TokenSource;
use inference_tensor::Device;
use sysinfo::Disks;

use crate::ModelSelected;

/// Where built recipes are kept, in place of `~/.cache/inference-rs/recipes`.
pub const RECIPE_DIR_ENV: &str = "INFERENCE_RS_RECIPE_DIR";
/// Set (to anything), checkpoints load as they are stored, without a recipe.
pub const NO_RECIPE_ENV: &str = "INFERENCE_RS_NO_RECIPE";
const RECIPE_SUBDIR: &str = "recipes";
const CACHE_SUBDIR: &str = "inference-rs";
const MMPROJ_PREFIX: &str = "mmproj-";
const PARTIAL_SUFFIX: &str = ".partial";

/// Whether loads take recipes, and where built ones are kept.
#[derive(Clone, Debug)]
pub struct RecipeOptions {
    pub enabled: bool,
    /// The checkpoint's vision tower loads too, from a projector written beside the recipe.
    pub vision: bool,
    /// Unset: `INFERENCE_RS_RECIPE_DIR`, else `~/.cache/inference-rs/recipes`.
    pub dir: Option<PathBuf>,
}

impl Default for RecipeOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            vision: false,
            dir: None,
        }
    }
}

fn recipe_root(options: &RecipeOptions) -> Option<PathBuf> {
    options
        .dir
        .clone()
        .or_else(|| std::env::var_os(RECIPE_DIR_ENV).map(PathBuf::from))
        .or_else(|| {
            Some(
                inference_tensor::kernel_library::user_cache()?
                    .join(CACHE_SUBDIR)
                    .join(RECIPE_SUBDIR),
            )
        })
}

/// Whether `model` loads a checkpoint as it is stored and asks for nothing a recipe would undo: no in-situ
/// quantization, UQFF, calibration, Hessians, EXL3 writes or MatFormer slice.
pub fn may_take_recipe(model: &ModelSelected, isq: Option<&str>, options: &RecipeOptions) -> bool {
    if !options.enabled || isq.is_some() || std::env::var_os(NO_RECIPE_ENV).is_some() {
        return false;
    }
    matches!(
        model,
        ModelSelected::Run {
            quant: None,
            write_uqff: None,
            from_uqff: None,
            imatrix: None,
            calibration_file: None,
            hessians_out: None,
            exl3_out: None,
            matformer_config_path: None,
            matformer_slice_name: None,
            ..
        } | ModelSelected::Plain {
            quant: None,
            write_uqff: None,
            from_uqff: None,
            imatrix: None,
            calibration_file: None,
            hessians_out: None,
            exl3_out: None,
            matformer_config_path: None,
            matformer_slice_name: None,
            ..
        }
    )
}

/// The checkpoint a recipe may apply to: a `Run` or `Plain` selection's model id.
pub fn checkpoint_id(model: &ModelSelected) -> Option<&str> {
    match model {
        ModelSelected::Run { model_id, .. } | ModelSelected::Plain { model_id, .. } => {
            Some(model_id)
        }
        _ => None,
    }
}

fn available_bytes(path: &Path) -> Option<u64> {
    let disks = Disks::new_with_refreshed_list();
    disks
        .list()
        .iter()
        .filter(|disk| path.starts_with(disk.mount_point()))
        .max_by_key(|disk| disk.mount_point().as_os_str().len())
        .map(|disk| disk.available_space())
}

/// `model` as its recipe's GGUF when it names a checkpoint with one (building it first if the cache lacks it);
/// otherwise `model` as it is. `isq` is the load's in-situ quantization, which keeps the checkpoint as stored.
pub fn with_load_recipe(
    model: ModelSelected,
    isq: Option<&str>,
    token_source: &TokenSource,
    revision: Option<&str>,
    options: &RecipeOptions,
) -> anyhow::Result<ModelSelected> {
    if !may_take_recipe(&model, isq, options) {
        return Ok(model);
    }
    let Some(id) = checkpoint_id(&model) else {
        return Ok(model);
    };
    let Some(dir) = inference_core::inkling2_checkpoint_dir(id, token_source, revision)? else {
        return Ok(model);
    };
    let root = recipe_root(options).ok_or_else(|| {
        anyhow::anyhow!(
            "no cache directory for the recipe: set `{RECIPE_DIR_ENV}` or `{NO_RECIPE_ENV}`"
        )
    })?;
    let path = inference_core::inkling2_recipe_path(&dir, &root)?;
    if !path.is_file() {
        std::fs::create_dir_all(&root)?;
        let needed = inference_core::inkling2_recipe_bytes(&dir)?;
        if let Some(free) = available_bytes(&root)
            && free < needed
        {
            anyhow::bail!(
                "building `{}`'s recipe needs up to {:.1} GiB in `{}`, which has {:.1} GiB free: set `{RECIPE_DIR_ENV}` \
                 to a roomier directory, or `{NO_RECIPE_ENV}` to load the checkpoint as it is stored",
                dir.display(),
                needed as f64 / (1u64 << 30) as f64,
                root.display(),
                free as f64 / (1u64 << 30) as f64,
            );
        }
        inference_core::ensure_inkling2_recipe(&dir, &path, Device::new_cuda(0).ok())?;
    }
    let mmproj = if options.vision {
        ensure_projector(&dir, &path)?
    } else {
        None
    };
    let dir_of = |p: &Path| {
        p.parent()
            .map(|d| d.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    let file = path
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_default();
    Ok(match model {
        ModelSelected::Run {
            tokenizer_json,
            dtype,
            topology,
            organization,
            max_seq_len,
            max_batch_size,
            hf_cache_path,
            ..
        }
        | ModelSelected::Plain {
            tokenizer_json,
            dtype,
            topology,
            organization,
            max_seq_len,
            max_batch_size,
            hf_cache_path,
            ..
        } => ModelSelected::GGUF {
            tok_model_id: None,
            quantized_model_id: dir_of(&path),
            quantized_filename: file,
            quant: None,
            tokenizer_json,
            mmproj_filename: mmproj,
            mmproj_selection: Default::default(),
            lora_adapters: Vec::new(),
            lora_runtime_config: None,
            dtype,
            topology,
            organization,
            write_uqff: None,
            imatrix: None,
            calibration_file: None,
            hessians_out: None,
            max_edge: None,
            max_seq_len,
            max_batch_size,
            max_num_images: None,
            max_image_length: None,
            hf_cache_path,
            matformer_config_path: None,
            matformer_slice_name: None,
        },
        other => other,
    })
}

// the checkpoint's projector beside its recipe, written once; `None` for a checkpoint without a tower
fn ensure_projector(dir: &Path, recipe: &Path) -> anyhow::Result<Option<String>> {
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    if config.get("vision_config").is_none() {
        return Ok(None);
    }
    let stem = recipe
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let name = format!("{MMPROJ_PREFIX}{stem}.gguf");
    let path = recipe.with_file_name(&name);
    if !path.is_file() {
        let partial = recipe.with_file_name(format!("{name}{PARTIAL_SUFFIX}"));
        inference_core::write_mmproj(dir, &partial, None)?;
        std::fs::rename(&partial, &path)?;
    }
    Ok(Some(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(model_id: &str) -> ModelSelected {
        serde_json::from_value(serde_json::json!({"Run": {"model_id": model_id}})).unwrap()
    }

    // a recipe replaces only a checkpoint loaded as stored; anything asking for its own weights keeps them
    #[test]
    fn only_a_checkpoint_loaded_as_stored_takes_its_recipe() {
        let on = RecipeOptions::default();
        assert!(may_take_recipe(&run("some/checkpoint"), None, &on));
        assert!(!may_take_recipe(&run("some/checkpoint"), Some("q4k"), &on));
        let off = RecipeOptions {
            enabled: false,
            ..Default::default()
        };
        assert!(!may_take_recipe(&run("some/checkpoint"), None, &off));
        let uqff: ModelSelected = serde_json::from_value(
            serde_json::json!({"Run": {"model_id": "some/checkpoint", "from_uqff": "model.uqff"}}),
        )
        .unwrap();
        assert!(!may_take_recipe(&uqff, None, &on));
        let gguf: ModelSelected = serde_json::from_value(serde_json::json!({
            "GGUF": {"quantized_model_id": "some/gguf", "quantized_filename": "model.gguf"}
        }))
        .unwrap();
        assert!(!may_take_recipe(&gguf, None, &on));
    }
}
