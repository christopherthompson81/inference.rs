pub use crate::model::DiffusionModel;
use std::{
    collections::HashMap,
    fmt::Debug,
    path::{Path, PathBuf},
    str::FromStr,
};

use anyhow::{Context, Result};

use hf_hub::api::sync::ApiRepo;
use inference_quant::ShardedVarBuilder;

use regex::Regex;
use serde::Deserialize;

use super::{ModelPaths, NormalLoadingMetadata};
use inference_models_diffusion::flux::{
    self,
    stepper::{
        FluxStepper, FluxStepperConfig, FluxStepperLoad, LocalTextEncoders, RepoFileFetcher,
    },
};

use crate::{
    api_dir_list, api_get_file,
    pipeline::{EmbeddingModulePaths, hf, paths::AdapterPaths},
};

const AE_FILE: &str = "ae.safetensors";
const CLIP_L_FILE: &str = "clip_l.safetensors";
const FLUX_SAFETENSORS_PATTERN: &str = r"^flux\d+-(schnell|dev)\.safetensors$";
const FLUX_GGUF_PATTERN: &str = r"^flux\d+-(dev|schnell).*\.gguf$";
const T5_GGUF_PATTERN: &str = r"^t5.*\.gguf$";

fn hub_file_fetcher() -> Result<RepoFileFetcher> {
    let api = hf_hub::api::sync::ApiBuilder::from_env().build()?;
    Ok(Box::new(move |repo_id, revision, file| {
        let model_id = Path::new(repo_id);
        if hf::is_hf_hub_offline() {
            return hf::offline_cache_repo(model_id, revision)
                .get(file)
                .ok_or_else(|| {
                    candle_core::Error::msg(hf::offline_missing_file_error(
                        model_id, file, revision,
                    ))
                });
        }
        api.repo(hf_hub::Repo::with_revision(
            repo_id.to_string(),
            hf_hub::RepoType::Model,
            revision.to_string(),
        ))
        .get(file)
        .map_err(candle_core::Error::msg)
    }))
}

pub trait DiffusionModelLoader: Send + Sync {
    /// If the model is being loaded with `load_model_from_hf` (so manual paths not provided), this will be called.
    fn get_model_paths(
        &self,
        api: &ApiRepo,
        model_id: &Path,
        revision: &str,
    ) -> Result<Vec<PathBuf>>;
    /// If the model is being loaded with `load_model_from_hf` (so manual paths not provided), this will be called.
    fn get_config_filenames(
        &self,
        api: &ApiRepo,
        model_id: &Path,
        revision: &str,
    ) -> Result<Vec<PathBuf>>;
    /// The paths of a local layout that needs no hub listing, when `model_id` is one.
    fn local_paths(&self, _model_id: &Path) -> Result<Option<DiffusionModelPathsInner>> {
        Ok(None)
    }
    fn force_cpu_vb(&self) -> Vec<bool>;
    fn load(&self, inputs: DiffusionLoad) -> Result<Box<dyn DiffusionModel + Send + Sync>>;
}

/// Per weight file: `vbs`, `shapes` (GGUF files only) and `configs` (empty when the layout ships none).
pub struct DiffusionLoad {
    pub configs: Vec<String>,
    pub vbs: Vec<ShardedVarBuilder>,
    pub shapes: Vec<Option<HashMap<String, Vec<usize>>>>,
    pub text_encoders: LocalTextEncoders,
    pub metadata: NormalLoadingMetadata,
    pub silent: bool,
}

/// The single-file layout: a FLUX GGUF with `ae.safetensors`, and optionally a T5 GGUF and `clip_l.safetensors`.
struct FluxLocalFiles {
    transformer: PathBuf,
    ae: PathBuf,
    text_encoders: LocalTextEncoders,
}

fn only_match(names: &[String], regex: &Regex, what: &str, dir: &Path) -> Result<Option<PathBuf>> {
    let found = names
        .iter()
        .filter(|name| regex.is_match(name))
        .collect::<Vec<_>>();
    match found.as_slice() {
        [] => Ok(None),
        [name] => Ok(Some(dir.join(name))),
        many => anyhow::bail!(
            "`{}` holds several {what} files ({many:?}); keep one",
            dir.display()
        ),
    }
}

fn flux_local_files(dir: &Path) -> Result<Option<FluxLocalFiles>> {
    if !dir.is_dir() {
        return Ok(None);
    }
    let names = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
        .collect::<Vec<_>>();
    let Some(transformer) = only_match(&names, &Regex::new(FLUX_GGUF_PATTERN)?, "FLUX GGUF", dir)?
    else {
        return Ok(None);
    };
    if !names.iter().any(|name| name == AE_FILE) {
        anyhow::bail!("`{}` has a FLUX GGUF but no `{AE_FILE}`", dir.display());
    }
    Ok(Some(FluxLocalFiles {
        transformer,
        ae: dir.join(AE_FILE),
        text_encoders: LocalTextEncoders {
            t5: only_match(&names, &Regex::new(T5_GGUF_PATTERN)?, "T5 GGUF", dir)?,
            clip: names
                .iter()
                .any(|name| name == CLIP_L_FILE)
                .then(|| dir.join(CLIP_L_FILE)),
        },
    }))
}

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Deserialize, serde::Serialize, PartialEq, strum::EnumIter)]
/// The architecture to load the diffusion model as.
pub enum DiffusionLoaderType {
    #[serde(rename = "flux")]
    Flux,
    #[serde(rename = "flux-offloaded")]
    FluxOffloaded,
}

impl FromStr for DiffusionLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "flux" => Ok(Self::Flux),
            "flux-offloaded" => Ok(Self::FluxOffloaded),
            a => Err(format!(
                "Unknown architecture `{a}`. Possible architectures: `flux`."
            )),
        }
    }
}

impl DiffusionLoaderType {
    /// Auto-detect diffusion loader type from a repo file listing.
    /// Extend this when adding new diffusion pipelines.
    pub fn auto_detect_from_files(files: &[String]) -> Option<Self> {
        if Self::matches_flux(files) {
            return Some(Self::Flux);
        }
        None
    }

    fn matches_flux(files: &[String]) -> bool {
        let flux_regex = Regex::new(FLUX_SAFETENSORS_PATTERN);
        let Ok(flux_regex) = flux_regex else {
            return false;
        };
        let has_transformer = files.iter().any(|f| f == "transformer/config.json");
        let has_vae = files.iter().any(|f| f == "vae/config.json");
        let has_ae = files.iter().any(|f| f == "ae.safetensors");
        let has_flux = files.iter().any(|f| {
            let name = f.rsplit('/').next().unwrap_or(f);
            flux_regex.is_match(name)
        });

        let gguf_regex = Regex::new(FLUX_GGUF_PATTERN);
        let has_flux_gguf = gguf_regex.is_ok_and(|regex| files.iter().any(|f| regex.is_match(f)));
        has_ae && ((has_transformer && has_vae && has_flux) || has_flux_gguf)
    }
}

#[derive(Clone, Debug)]
pub struct DiffusionModelPathsInner {
    pub config_filenames: Vec<PathBuf>,
    pub filenames: Vec<PathBuf>,
    pub text_encoders: LocalTextEncoders,
}

#[derive(Clone, Debug)]
pub struct DiffusionModelPaths(pub DiffusionModelPathsInner);

impl ModelPaths for DiffusionModelPaths {
    fn get_config_filename(&self) -> &PathBuf {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_tokenizer_filename(&self) -> &PathBuf {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_weight_filenames(&self) -> &[PathBuf] {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_template_filename(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_gen_conf_filename(&self) -> Option<&PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_preprocessor_config(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_processor_config(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_chat_template_explicit(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_adapter_paths(&self) -> &AdapterPaths {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_modules(&self) -> Option<&[EmbeddingModulePaths]> {
        unreachable!("Use `std::any::Any`.")
    }
}

// ======================== Flux loader

/// [`DiffusionLoader`] for a Flux Diffusion model.
///
/// [`DiffusionLoader`]: crate::pipeline::DiffusionLoader
pub struct FluxLoader {
    pub(crate) offload: bool,
}

impl DiffusionModelLoader for FluxLoader {
    fn get_model_paths(
        &self,
        api: &ApiRepo,
        model_id: &Path,
        revision: &str,
    ) -> Result<Vec<PathBuf>> {
        let regex = Regex::new(FLUX_SAFETENSORS_PATTERN)?;
        let flux_name = api_dir_list!(api, model_id, true, revision)
            .filter(|x| regex.is_match(x))
            .nth(0)
            .with_context(|| "Expected at least 1 .safetensors file matching the FLUX regex, please raise an issue.")?;
        let flux_file = api_get_file!(api, &flux_name, model_id, revision);
        let ae_file = api_get_file!(api, "ae.safetensors", model_id, revision);

        // NOTE(EricLBuehler): disgusting way of doing this but the 0th path is the flux, 1 is ae
        Ok(vec![flux_file, ae_file])
    }
    fn get_config_filenames(
        &self,
        api: &ApiRepo,
        model_id: &Path,
        revision: &str,
    ) -> Result<Vec<PathBuf>> {
        let flux_file = api_get_file!(api, "transformer/config.json", model_id, revision);
        let ae_file = api_get_file!(api, "vae/config.json", model_id, revision);

        // NOTE(EricLBuehler): disgusting way of doing this but the 0th path is the flux, 1 is ae
        Ok(vec![flux_file, ae_file])
    }
    fn local_paths(&self, model_id: &Path) -> Result<Option<DiffusionModelPathsInner>> {
        Ok(
            flux_local_files(model_id)?.map(|files| DiffusionModelPathsInner {
                config_filenames: Vec::new(),
                filenames: vec![files.transformer, files.ae],
                text_encoders: files.text_encoders,
            }),
        )
    }
    fn force_cpu_vb(&self) -> Vec<bool> {
        vec![self.offload, false]
    }
    fn load(&self, inputs: DiffusionLoad) -> Result<Box<dyn DiffusionModel + Send + Sync>> {
        let DiffusionLoad {
            configs,
            mut vbs,
            shapes,
            text_encoders,
            metadata: normal_loading_metadata,
            silent,
        } = inputs;
        let vae_vb = vbs.remove(1);
        let flux_vb = vbs.remove(0);
        if self.offload && flux_vb.weight_source().is_some() {
            anyhow::bail!(
                "a GGUF FLUX transformer is quantized and cannot be offloaded; use `flux`, not `flux-offloaded`"
            );
        }
        let (flux_cfg, vae_cfg) = match configs.as_slice() {
            [] => (
                flux::model::Config::from_weights(
                    shapes[0]
                        .as_ref()
                        .context("a FLUX checkpoint without configs must be a GGUF")?,
                )?,
                flux::autoencoder::Config::flux(),
            ),
            [flux_cfg, vae_cfg] => (
                flux::model::Config::from_json(flux_cfg)?,
                flux::autoencoder::Config::from_json(vae_cfg)?,
            ),
            other => anyhow::bail!("expected the FLUX and VAE configs, got {}", other.len()),
        };

        let flux_dtype = flux_vb.dtype();
        if flux_dtype != vae_vb.dtype() {
            anyhow::bail!(
                "Expected VAE and FLUX model VBs to be the same dtype, got {:?} and {flux_dtype:?}",
                vae_vb.dtype()
            );
        }

        Ok(Box::new(FluxStepper::new(
            FluxStepperConfig::default_for_guidance(flux_cfg.guidance_embeds),
            (flux_vb, &flux_cfg),
            (vae_vb, &vae_cfg),
            FluxStepperLoad {
                dtype: flux_dtype,
                device: &normal_loading_metadata.real_device,
                silent,
                offloaded: self.offload,
                fetch: hub_file_fetcher()?,
                text_encoders,
            },
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::{DiffusionLoaderType, flux_local_files};

    fn dir_with(files: &[&str]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for file in files {
            std::fs::write(dir.path().join(file), []).unwrap();
        }
        dir
    }

    #[test]
    fn a_single_file_flux_layout_is_found_with_its_text_encoders() {
        let dir = dir_with(&[
            "flux1-dev-Q8_0.gguf",
            "ae.safetensors",
            "t5-v1_1-xxl-encoder-Q8_0.gguf",
            "clip_l.safetensors",
            "unrelated-model.gguf",
        ]);
        let files = flux_local_files(dir.path()).unwrap().unwrap();
        assert_eq!(files.transformer, dir.path().join("flux1-dev-Q8_0.gguf"));
        assert_eq!(files.ae, dir.path().join("ae.safetensors"));
        assert_eq!(
            files.text_encoders.t5.as_deref(),
            Some(dir.path().join("t5-v1_1-xxl-encoder-Q8_0.gguf").as_path())
        );
        assert_eq!(
            files.text_encoders.clip.as_deref(),
            Some(dir.path().join("clip_l.safetensors").as_path())
        );
    }

    #[test]
    fn a_flux_gguf_layout_needs_the_autoencoder_and_one_transformer() {
        assert!(flux_local_files(dir_with(&["flux1-dev-Q8_0.gguf"]).path()).is_err());
        let two = dir_with(&[
            "flux1-dev-Q8_0.gguf",
            "flux1-schnell-Q4_0.gguf",
            "ae.safetensors",
        ]);
        assert!(flux_local_files(two.path()).is_err());
        assert!(
            flux_local_files(dir_with(&["ae.safetensors"]).path())
                .unwrap()
                .is_none()
        );
        assert!(
            flux_local_files(std::path::Path::new("black-forest-labs/FLUX.1-dev"))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn a_flux_repo_listing_is_detected() {
        let files = [
            "flux1-dev.safetensors",
            "ae.safetensors",
            "transformer/config.json",
            "vae/config.json",
        ]
        .map(String::from);
        assert!(matches!(
            DiffusionLoaderType::auto_detect_from_files(&files),
            Some(DiffusionLoaderType::Flux)
        ));
        assert!(DiffusionLoaderType::auto_detect_from_files(&files[1..]).is_none());
        let gguf = [
            "flux1-dev-Q8_0.gguf",
            "ae.safetensors",
            "clip_l.safetensors",
        ]
        .map(String::from);
        assert!(matches!(
            DiffusionLoaderType::auto_detect_from_files(&gguf),
            Some(DiffusionLoaderType::Flux)
        ));
        assert!(DiffusionLoaderType::auto_detect_from_files(&gguf[..1]).is_none());
    }
}
