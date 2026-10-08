use super::IsqOrganization;
use super::decoder::{DecoderModel, DecoderPipeline};
use super::decoder_core::{DecoderCore, DecoderCoreArgs, LoadedModelView};
use super::loaders::NormalLoaderTypeExt;
use super::{AutoNormalLoader, NormalLoaderType};
use super::{Loader, ModelKind, ModelPaths, NormalModelLoader, TokenSource};
use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::pipeline::LoadOptions;
use crate::pipeline::isq::{UqffFullSer, UqffWriteConfig};
use crate::pipeline::tokenizer::get_tokenizer;
use crate::pipeline::{Modalities, SupportedModality};
use crate::utils::progress::ProgressScopeGuard;
use crate::{
    DeviceMapSetting, LoraAdapterSpec, LoraRuntimeConfig, PagedAttentionConfig, Pipeline, Topology,
    TryIntoDType,
};
use anyhow::Result;
use inference_quant::IsqType;
use inference_tensor::Device;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use tokio::sync::Mutex;
use tracing::{debug, trace};

const ADJACENT_PARTIAL_ROTARY_LORA: &str = "LoRA adapters are not supported when Q/K use adjacent RoPE pairs over part of each head (MLA or partial rotary); load the original safetensors model or omit the adapter";

/// A loader for a "normal" (non-quantized) model.
pub struct NormalLoader {
    inner: Box<dyn NormalModelLoader>,
    model_id: String,
    config: NormalSpecificConfig,
    lora_adapters: Option<Vec<LoraAdapterSpec>>,
    lora_runtime_config: Option<LoraRuntimeConfig>,
    kind: ModelKind,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    from_uqff: RwLock<Option<Vec<PathBuf>>>,
    jinja_explicit: Option<String>,
    hf_cache_path: Option<PathBuf>,
    prepared_source: Option<super::loading::PreparedSource>,
    mtp: bool,
}

pub(crate) fn new_dynamic_lora_registry(
    config: &str,
    rope_pairing: Option<crate::gguf::normal_registry::RopePairing>,
) -> Result<Arc<inference_quant::LoraLayerRegistry>> {
    let config = serde_json::from_str::<serde_json::Value>(config)?;
    let qwen35_moe_identity = config
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .and_then(|architectures| architectures.first())
        .and_then(serde_json::Value::as_str)
        == Some("Qwen3NextForCausalLM")
        && config
            .get(crate::gdn::GDN_V_HEAD_LAYOUT_CONFIG_KEY)
            .and_then(serde_json::Value::as_str)
            == Some("tiled");
    let registry = if qwen35_moe_identity {
        inference_quant::LoraLayerRegistry::new_with_site_prefix_alias(
            "model",
            "model.language_model",
        )?
    } else {
        inference_quant::LoraLayerRegistry::new()
    };
    let registry = match rope_pairing {
        Some(crate::gguf::normal_registry::RopePairing::Adjacent) => {
            // MLA and partial rotary pair only part of each head, which the per-head row map does not describe
            let partial_rotary = config.get("qk_rope_head_dim").is_some()
                || config
                    .get("partial_rotary_factor")
                    .and_then(serde_json::Value::as_f64)
                    .is_some_and(|factor| factor < 1.0);
            if partial_rotary {
                anyhow::bail!(ADJACENT_PARTIAL_ROTARY_LORA);
            }
            registry.with_adjacent_qk_rope(attention_head_dim(&config)?)?
        }
        _ => registry,
    };
    Ok(Arc::new(registry))
}

fn attention_head_dim(config: &serde_json::Value) -> Result<usize> {
    let field = |name: &str| config.get(name).and_then(serde_json::Value::as_u64);
    let head_dim = match (
        field("head_dim"),
        field("hidden_size"),
        field("num_attention_heads"),
    ) {
        (Some(head_dim), _, _) => head_dim,
        (None, Some(hidden), Some(heads)) if heads > 0 => hidden / heads,
        _ => anyhow::bail!("the model config has no head_dim, hidden_size or num_attention_heads"),
    };
    Ok(usize::try_from(head_dim)?)
}

#[derive(Default)]
/// A builder for a loader for a "normal" (non-quantized) model.
pub struct NormalLoaderBuilder {
    model_id: Option<String>,
    config: NormalSpecificConfig,
    lora_adapters: Option<Vec<LoraAdapterSpec>>,
    lora_runtime_config: Option<LoraRuntimeConfig>,
    kind: ModelKind,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    jinja_explicit: Option<String>,
    hf_cache_path: Option<PathBuf>,
    mtp: bool,
}

#[derive(Clone, Default)]
/// Config specific to loading a normal model.
pub struct NormalSpecificConfig {
    pub topology: Option<Topology>,
    pub organization: IsqOrganization,
    pub write_uqff: Option<UqffWriteConfig>,
    pub from_uqff: Option<Vec<PathBuf>>,
    pub imatrix: Option<PathBuf>,
    pub calibration_file: Option<PathBuf>,
    pub hf_cache_path: Option<PathBuf>,
    pub hf_config_overrides: Option<super::HfConfigOverrides>,
    pub max_model_len: Option<usize>,
    pub matformer_config_path: Option<PathBuf>,
    pub matformer_slice_name: Option<String>,
}

impl NormalLoaderBuilder {
    pub fn new(
        config: NormalSpecificConfig,
        chat_template: Option<String>,
        tokenizer_json: Option<String>,
        model_id: Option<String>,
        no_kv_cache: bool,
        jinja_explicit: Option<String>,
    ) -> Self {
        let hf_cache_path = config.hf_cache_path.clone();
        Self {
            config,
            chat_template,
            tokenizer_json,
            model_id,
            kind: ModelKind::Normal,
            jinja_explicit,
            no_kv_cache,
            hf_cache_path,
            ..Default::default()
        }
    }

    /// Load the MTP head built into the checkpoint so it can drive speculative decoding.
    pub fn with_mtp(mut self, mtp: bool) -> Self {
        self.mtp = mtp;
        self
    }

    pub fn with_lora(
        mut self,
        adapters: Vec<LoraAdapterSpec>,
        runtime_config: LoraRuntimeConfig,
    ) -> Self {
        self.kind = ModelKind::Lora;
        self.lora_adapters = Some(adapters);
        self.lora_runtime_config = Some(runtime_config);
        self
    }

    pub fn hf_cache_path(mut self, hf_cache_path: PathBuf) -> Self {
        self.hf_cache_path = Some(hf_cache_path);
        self
    }

    /// If the loader type is not specified, loader type is automatically determined from the
    /// `architectures` array in the config.
    fn build_inner(
        self,
        loader_tp: Option<NormalLoaderType>,
        prepared_source: Option<super::loading::PreparedSource>,
    ) -> anyhow::Result<NormalLoader> {
        super::validate_lora_loader_config(
            self.lora_adapters.as_deref(),
            self.lora_runtime_config,
        )?;
        let loader: Box<dyn NormalModelLoader> = match loader_tp {
            Some(tp) => tp.loader()?,
            None => Box::new(AutoNormalLoader),
        };
        Ok(NormalLoader {
            inner: loader,
            model_id: self.model_id.unwrap(),
            config: self.config,
            lora_adapters: self.lora_adapters,
            lora_runtime_config: self.lora_runtime_config,
            kind: self.kind,
            no_kv_cache: self.no_kv_cache,
            chat_template: self.chat_template,
            tokenizer_json: self.tokenizer_json,
            jinja_explicit: self.jinja_explicit,
            from_uqff: RwLock::new(None),
            hf_cache_path: self.hf_cache_path,
            prepared_source,
            mtp: self.mtp,
        })
    }

    pub fn build(self, loader_tp: Option<NormalLoaderType>) -> anyhow::Result<Box<dyn Loader>> {
        Ok(Box::new(self.build_inner(loader_tp, None)?))
    }

    pub(crate) fn build_with_source(
        mut self,
        loader_tp: NormalLoaderType,
        source: super::loading::PreparedSource,
        kind: ModelKind,
    ) -> anyhow::Result<Box<dyn Loader>> {
        self.kind = kind;
        Ok(Box::new(self.build_inner(Some(loader_tp), Some(source))?))
    }
}

impl NormalLoader {
    #[allow(clippy::too_many_arguments)]
    fn load_from_paths(
        &self,
        paths: &dyn ModelPaths,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        mut paged_attn_config: Option<PagedAttentionConfig>,
        uqff: super::loading::UqffLoad<'_>,
    ) -> Result<super::loading::LoadOutcome> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let serve_written = uqff.serves_written(device);
        let in_situ_quant = in_situ_quant.filter(|_| !uqff.is_reload());
        // a reload reads the written UQFF: the imatrix and calibration were spent writing it
        let (imatrix, calibration_file) = match uqff {
            super::loading::UqffLoad::Reload(_) => (None, None),
            _ => (
                self.config.imatrix.as_ref(),
                self.config.calibration_file.as_ref(),
            ),
        };
        let from_uqff_files = self.from_uqff.read().unwrap();
        let uqff_files = uqff.reload_files().or(from_uqff_files.as_deref());
        let config = super::loading::prepare_model_config(
            self.prepared_source.as_ref(),
            paths.get_config_filename(),
            uqff.reads(),
            self.config.hf_config_overrides.as_ref(),
            self.mtp,
        )?;
        // The UQFF artifact keeps the checkpoint config; max_model_len and the like apply to this load only.
        let source_config = config;
        let config = self
            .inner
            .runtime_config(&source_config, self.config.max_model_len)?
            .into_owned();

        if !self.inner.supports_paged_attention(&config)? {
            paged_attn_config = None;
        }

        debug!("Prompt chunk size is {ATTENTION_CHUNK_SIZE}.");

        let matformer = super::loading::load_matformer_slice(
            self.config.matformer_config_path.as_deref(),
            self.config.matformer_slice_name.as_deref(),
        )?;
        let (session, mapper) = super::loading::open_load_session(
            super::loading::LoadSessionInputs {
                mapped: &*self.inner,
                isq: &*self.inner,
                config: &config,
                settings: super::loading::LoadSettings {
                    topology: self.config.topology.as_ref(),
                    organization: self.config.organization,
                    write_uqff: uqff.write(),
                    from_uqff: uqff.reads(),
                    has_imatrix: imatrix.is_some(),
                    has_calibration: calibration_file.is_some(),
                },
                paths,
                device,
                dtype,
                mapper,
                in_situ_quant,
                uqff_files,
                prepared: self.prepared_source.as_ref(),
                has_lora: self.lora_adapters.is_some(),
                matformer,
                matformer_sizing: false,
                non_mapped_unpacked: false,
                auto_device_map_params: None,
                weight_target: "model",
            },
            &mut paged_attn_config,
        )?;
        trace!("Model config: {:?}", self.inner.get_config_repr(&config)?);
        let (model, tracker, dynamic_lora) = super::loading::load_model(
            &*self.inner,
            &session,
            mapper,
            super::loading::ModelLoadInputs {
                config: &config,
                paths,
                silent,
                organization: self.config.organization,
                from_uqff: uqff.reads(),
                write_uqff: uqff.write().is_some(),
                prepared: self.prepared_source.as_ref(),
                lora: super::loading::lora_runtime(&self.kind, self.lora_runtime_config),
            },
        )?;
        let super::loading::LoadSession {
            device,
            weight_source,
            max_kv_tokens,
            pipeline_mapper,
            layer_devices,
            dtype,
            plan,
            ..
        } = session;
        let load_device = plan.load_device.clone();

        let tokenizer = match self.prepared_source.as_ref() {
            Some(source) => source.tokenizer.clone(),
            None => get_tokenizer(paths.get_tokenizer_filename(), None)?,
        };
        let gen_conf = super::loading::generation_config(
            self.prepared_source
                .as_ref()
                .map(|source| source.generation_config.clone()),
            paths,
            &config,
        );

        let chat_template = super::loading::load_chat_template(
            paths,
            self.jinja_explicit.as_ref(),
            self.chat_template.as_ref(),
            self.prepared_source.as_ref(),
        );

        // cloned out so the tracker lock is not held through calibration and the UQFF write
        let tracked = tracker.get().clone();
        let written = super::isq_flow::finish_isq_load(super::isq_flow::FinishIsqLoad {
            plan: &plan,
            modules: tracked,
            drive: &super::isq_flow::NormalCalibrationDrive(&*model),
            in_situ_quant,
            imatrix,
            calibration_file,
            calibration: super::isq_flow::CalibrationCtx {
                tokenizer: &tokenizer,
                bos_tok_id: chat_template
                    .bos_tok()
                    .as_deref()
                    .and_then(|tok| tokenizer.token_to_id(tok)),
                load_device: &load_device,
                mapper: Some(pipeline_mapper.as_ref()),
            },
            uqff: uqff.write().map(|write| super::isq_flow::UqffArtifact {
                config: write,
                residual: super::loading::uqff_residual_tensors(self.config.organization, &*model),
                full_ser: UqffFullSer {
                    tokenizer: &tokenizer,
                    template_filename: paths.get_template_filename(),
                    effective_chat_template: Some(&chat_template),
                    generation_config: super::loading::uqff_generation_config_file(
                        paths,
                        self.prepared_source.as_ref(),
                    ),
                    config: source_config.clone(),
                    processor_filename: &None,
                    preprocessor_filename: &None,
                    modules: None,
                    module_paths: None,
                },
            }),
        })?;
        if serve_written && let Some(files) = written.filter(|files| !files.is_empty()) {
            if self.prepared_source.is_some() {
                anyhow::bail!(
                    "Wrote the UQFF to `{}`; a model read from GGUF serves from it on a GPU only through a load with `from_uqff`.",
                    files[0].display()
                );
            }
            return Ok(super::loading::LoadOutcome::Written(files));
        }

        let tracked_modules = tracker.get().clone();
        let source_weight_files = super::loading::source_weight_files(
            self.prepared_source.as_ref(),
            uqff.reads(),
            paths.get_weight_filenames(),
        );

        let core = DecoderCore::new(DecoderCoreArgs {
            model: LoadedModelView {
                target: &*model,
                cache: model.cache(),
                config: model.model_config(),
                max_seq_len: model.max_seq_len(),
                sliding_window: model.config().sliding_window,
                block_diffusion: false,
            },
            tokenizer,
            chat_template,
            generation_config: gen_conf,
            paged_attn_config,
            dtype,
            layer_devices,
            device,
            mapper: pipeline_mapper,
            silent,
            max_kv_tokens,
            no_kv_cache: self.no_kv_cache,
            no_prefix_cache: false,
            kind: self.kind.clone(),
            model_id: self.model_id.clone(),
            modalities: Modalities {
                input: vec![SupportedModality::Text],
                output: vec![SupportedModality::Text],
            },
            loaded_for_uqff_write: uqff.write().is_some(),
            tracked_modules,
            source_weight_files,
            source_weight_source: weight_source,
            dynamic_lora,
        })?;
        Ok(super::loading::LoadOutcome::Pipeline(Arc::new(Mutex::new(
            DecoderPipeline::new(DecoderModel::Text(model), core, None),
        ))))
    }
}

impl Loader for NormalLoader {
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        options: LoadOptions<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let silent = options.silent;
        let _progress_guard = ProgressScopeGuard::new(silent);
        let paths = super::loading::hub_model_paths(
            super::loading::HubPathsRequest {
                hf_cache_path: self.hf_cache_path.clone(),
                model_id: &self.model_id,
                tokenizer_json: self.tokenizer_json.as_deref(),
                chat_template: self.chat_template.as_deref(),
                token_source: &token_source,
                revision,
                silent,
                from_uqff: self.config.from_uqff.as_deref(),
            },
            &self.from_uqff,
            |request| super::paths::get_paths(request, self.lora_adapters.as_deref()),
        )?;
        self.load_model_from_path(&paths, options)
    }

    fn load_model_from_path(
        &self,
        paths: &dyn ModelPaths,
        options: LoadOptions<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let LoadOptions {
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        } = options;
        let uqff = super::loading::UqffLoad::new(
            self.config.from_uqff.is_some(),
            self.config.write_uqff.as_ref(),
        )?;
        super::loading::load_serving_written(uqff, |uqff| {
            self.load_from_paths(
                paths,
                dtype,
                device,
                silent,
                mapper.clone(),
                in_situ_quant,
                paged_attn_config,
                uqff,
            )
        })
    }

    fn get_id(&self) -> String {
        self.model_id.clone()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::new_dynamic_lora_registry;
    use crate::LoraRuntimeConfig;
    use crate::pipeline::finish_dynamic_lora_runtime;
    use crate::pipeline::{AdapterPaths, LocalModelPaths};
    use inference_quant::{LoraLayerRegistry, LoraLinearSpec, LoraSiteKey};
    use inference_tensor::{DType, Device};
    use std::{path::PathBuf, sync::Arc};

    fn empty_lora_paths() -> LocalModelPaths<PathBuf> {
        LocalModelPaths {
            tokenizer_filename: PathBuf::new(),
            config_filename: PathBuf::new(),
            template_filename: None,
            filenames: Vec::new(),
            adapter_paths: AdapterPaths::Lora(Vec::new()),
            gen_conf: None,
            preprocessor_config: None,
            video_preprocessor_config: None,
            processor_config: None,
            chat_template_json_filename: None,
        }
    }

    #[test]
    fn prepared_lora_runtime_finalizes_sites_and_preserves_update_policy() {
        let paths = empty_lora_paths();
        for live_updates in [false, true] {
            let layers = Arc::new(LoraLayerRegistry::new());
            let runtime = finish_dynamic_lora_runtime(
                &paths,
                layers.clone(),
                LoraRuntimeConfig::default(),
                live_updates,
            )
            .unwrap();

            assert_eq!(runtime.supports_live_updates(), live_updates);
            let error = layers
                .register(
                    LoraSiteKey::new("model.layers.0.self_attn.q_proj"),
                    LoraLinearSpec::replicated(2, 2),
                    DType::F32,
                    Device::Cpu,
                )
                .unwrap_err();
            assert!(error.to_string().contains("after registry finalization"));
        }
    }

    #[test]
    fn persisted_qwen35_moe_config_restores_lora_namespace_alias() {
        let config = r#"{
            "architectures":["Qwen3NextForCausalLM"],
            "_inference_gdn_v_head_layout":"tiled"
        }"#;
        let registry = new_dynamic_lora_registry(config, None).unwrap();
        let site = registry
            .register(
                LoraSiteKey::new("model.layers.0.self_attn.q_proj"),
                LoraLinearSpec::replicated(2, 2),
                DType::F32,
                Device::Cpu,
            )
            .unwrap();

        assert_eq!(
            site.key().path(),
            "model.language_model.layers.0.self_attn.q_proj"
        );
    }

    #[test]
    fn dense_qwen35_config_does_not_alias_lora_namespace() {
        let registry = new_dynamic_lora_registry(
            r#"{
                "architectures":["Qwen3_5ForCausalLM"],
                "_inference_gdn_v_head_layout":"tiled"
            }"#,
            None,
        )
        .unwrap();
        let site = registry
            .register(
                LoraSiteKey::new("model.layers.0.self_attn.q_proj"),
                LoraLinearSpec::replicated(2, 2),
                DType::F32,
                Device::Cpu,
            )
            .unwrap();

        assert_eq!(site.key().path(), "model.layers.0.self_attn.q_proj");
    }

    #[test]
    fn adjacent_rope_lora_refuses_heads_rotated_only_in_part() {
        let adjacent = Some(crate::gguf::normal_registry::RopePairing::Adjacent);
        for config in [
            r#"{"head_dim":16,"qk_rope_head_dim":8}"#,
            r#"{"head_dim":16,"partial_rotary_factor":0.5}"#,
        ] {
            let error = new_dynamic_lora_registry(config, adjacent)
                .err()
                .unwrap()
                .to_string();
            assert!(error.contains("original safetensors model"), "{error}");
        }
        assert!(
            new_dynamic_lora_registry(r#"{"head_dim":16,"partial_rotary_factor":1.0}"#, adjacent)
                .is_ok()
        );
        assert!(new_dynamic_lora_registry(r#"{"qk_rope_head_dim":8}"#, None).is_ok());
    }
}
