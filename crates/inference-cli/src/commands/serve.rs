//! Server command implementation

use anyhow::{Context, Result};
use std::path::Path;
use tracing::info;

use inference_api::{
    Engine, EngineSpec,
    engine::{
        AdapterSpec, AgenticSpec, MtpSpec, PagedCacheSpec, RuntimeSpec, SandboxLimits, SearchSpec,
        SkillsSpec,
    },
    lora_adapters::LoraAdapterApiConfig,
    skill_store::SkillStore,
};
use inference_core::{
    AutoDeviceMapParams, DiffusionLoaderType, McpClientConfig, PagedCacheType, SpeechLoaderType,
    initialize_logging,
};
use inference_selection::{MmprojSelection, ModelSelected};
use inference_server_core::{
    inference_server_router_builder::InferenceRsServerRouterBuilder,
    metrics::install_prometheus_recorder,
    serve::{ServeOptions, serve},
};

#[cfg(test)]
use crate::args::MultimodalAdapterOptions;
use crate::args::{
    AdapterOptions, AgentCliOptions, CodeExecPermissionArg, DeviceOptions, FormatOptions,
    GlobalOptions, MatformerSelection, ModelFormat, ModelSourceOptions, ModelType,
    MultimodalOptions, QuantizationOptions, RuntimeOptions, SandboxOptions, ServerOptions,
};
use inference_webui::{UI_ROUTE, UiOptions};

const MEBIBYTE_BYTES: usize = 1024 * 1024;

/// Run the HTTP server with the specified model
#[allow(clippy::too_many_arguments)]
pub async fn run_server(
    mut model_type: ModelType,
    server: ServerOptions,
    mut runtime: RuntimeOptions,
    agent_options: AgentCliOptions,
    sandbox: SandboxOptions,
    global: GlobalOptions,
) -> Result<()> {
    initialize_logging();
    if server.observability_config().metrics {
        install_prometheus_recorder();
    }

    agent_options.apply_to(&mut runtime);
    apply_agent_mode(&mut runtime);
    validate_agent_options(&runtime)?;
    log_agent_runtime(&runtime, server.max_tool_rounds);

    // Convert our clean args to ModelSelected for the existing loader infrastructure
    let matformer = runtime.matformer_selection();
    normalize_quant_flags(&mut model_type)?;
    let spec = engine_spec(EngineSpecInputs {
        model_type: &model_type,
        matformer: &matformer,
        model_id: None,
        runtime: &runtime,
        sandbox,
        global: &global,
        max_tool_rounds: server.max_tool_rounds,
        tool_dispatch_url: server.tool_dispatch_url.clone(),
        skills_root: Some(skills_root(&runtime)),
        adapters: adapter_spec_from_env(),
        throughput_logging: true,
    })?;
    serve_engine(spec, &server).await
}

/// The model and the runtime settings that come with it, as `serve`, `run` and `bench` all resolve them.
pub(crate) fn model_spec(
    model_type: &ModelType,
    matformer: &MatformerSelection,
    global: &GlobalOptions,
) -> Result<(ModelSelected, RuntimeSpec)> {
    let model = convert_to_model_selected(model_type, matformer)?;
    let (max_model_len, hf_config_overrides) = extract_hf_config_settings(model_type);
    let (paged_attn, memory_mb, memory_fraction, context_len, block_size, cache_type) =
        extract_paged_attn_settings(model_type);
    let (cpu, device_layers) = extract_device_settings(model_type);
    let runtime = RuntimeSpec {
        device: cpu.then(|| "cpu".to_string()),
        seed: global.seed,
        max_model_len,
        isq: extract_isq_setting(model_type),
        paged_attn,
        token_source: Some(global.token_source.to_string()),
        device_layers,
        paged_cache: PagedCacheSpec {
            context_len,
            memory_mb,
            memory_fraction,
            block_size,
            cache_type,
        },
        encoder_cache_memory_bytes: extract_encoder_cache_memory_bytes(model_type)?,
        hf_config_overrides,
        log: global.log.clone(),
        ..Default::default()
    };
    Ok((model, runtime))
}

/// What `serve` and `run` need to describe their engine, resolved from the command line.
pub(crate) struct EngineSpecInputs<'a> {
    pub model_type: &'a ModelType,
    pub matformer: &'a MatformerSelection,
    pub model_id: Option<String>,
    pub runtime: &'a RuntimeOptions,
    pub sandbox: SandboxOptions,
    pub global: &'a GlobalOptions,
    pub max_tool_rounds: Option<usize>,
    pub tool_dispatch_url: Option<String>,
    /// `None` keeps skills in a directory of the engine's own.
    pub skills_root: Option<std::path::PathBuf>,
    pub adapters: AdapterSpec,
    pub throughput_logging: bool,
}

/// The engine `serve` and `run` load, as the spec the C ABI and bindings load from.
pub(crate) fn engine_spec(inputs: EngineSpecInputs) -> Result<EngineSpec> {
    let (model, model_runtime) = model_spec(inputs.model_type, inputs.matformer, inputs.global)?;
    Ok(EngineSpec {
        model: Some(model),
        model_id: inputs.model_id,
        runtime: runtime_options_spec(inputs.runtime, model_runtime, inputs.throughput_logging),
        agentic: agentic_spec(AgenticInputs {
            runtime: inputs.runtime,
            sandbox: inputs.sandbox,
            max_tool_rounds: inputs.max_tool_rounds,
            tool_dispatch_url: inputs.tool_dispatch_url,
        })?,
        adapters: inputs.adapters,
        skills: SkillsSpec {
            root: inputs.skills_root,
        },
        ..Default::default()
    })
}

/// `RuntimeOptions`' settings over the model-derived ones in `base`.
pub(crate) fn runtime_options_spec(
    runtime: &RuntimeOptions,
    base: RuntimeSpec,
    throughput_logging: bool,
) -> RuntimeSpec {
    RuntimeSpec {
        max_seqs: Some(runtime.max_seqs),
        prefix_cache_n: Some(runtime.prefix_cache_n),
        no_kv_cache: runtime.no_kv_cache,
        chat_template: path_string(runtime.chat_template.as_deref()),
        jinja_explicit: path_string(runtime.jinja_explicit.as_deref()),
        mtp: mtp_spec(
            runtime.mtp,
            runtime.mtp_model.clone(),
            runtime.mtp_n_predict,
            runtime.mtp_draft_sampling,
        ),
        max_num_batched_tokens: Some(runtime.max_num_batched_tokens.get()),
        max_prefill_chunk_tokens: Some(runtime.max_prefill_chunk_tokens.get()),
        max_decode_steps_before_prefill: Some(runtime.max_decode_steps_before_prefill.get()),
        throughput_logging: Some(throughput_logging),
        ..base
    }
}

pub(crate) struct AgenticInputs<'a> {
    pub runtime: &'a RuntimeOptions,
    pub sandbox: SandboxOptions,
    pub max_tool_rounds: Option<usize>,
    pub tool_dispatch_url: Option<String>,
}

/// The tool loop, search, MCP and code-running tools `RuntimeOptions` asks for.
pub(crate) fn agentic_spec(inputs: AgenticInputs) -> Result<AgenticSpec> {
    let runtime = inputs.runtime;
    Ok(AgenticSpec {
        max_tool_rounds: inputs.max_tool_rounds,
        tool_dispatch_url: inputs.tool_dispatch_url,
        agent_permission: Some(runtime.code_exec_permission.into()),
        search: runtime.enable_search.then(|| SearchSpec {
            embedding_model: runtime
                .search_embedding_model
                .map(Into::into)
                .unwrap_or_default(),
        }),
        mcp: load_mcp_config(runtime.mcp_config.as_deref())?,
        #[cfg(feature = "code-execution")]
        code_execution: build_code_exec_config(runtime),
        #[cfg(not(feature = "code-execution"))]
        code_execution: None,
        #[cfg(feature = "code-execution")]
        shell: build_shell_config(runtime),
        #[cfg(not(feature = "code-execution"))]
        shell: None,
        sandbox: inputs.sandbox.mode.into(),
        sandbox_profile: inputs.sandbox.profile.map(Into::into),
        sandbox_limits: SandboxLimits {
            max_memory_mb: inputs.sandbox.max_memory_mb,
            max_cpu_secs: inputs.sandbox.max_cpu_secs,
            max_procs: inputs.sandbox.max_procs,
            network: inputs.sandbox.network.map(Into::into),
        },
    })
}

fn path_string(path: Option<&Path>) -> Option<String> {
    path.map(|path| path.to_string_lossy().into_owned())
}

pub(crate) fn mtp_spec(
    builtin: bool,
    model: Option<String>,
    n_predict: Option<usize>,
    draft_sampling: crate::args::MtpDraftSamplingArg,
) -> Option<MtpSpec> {
    (builtin || model.is_some()).then(|| MtpSpec {
        model: if builtin { None } else { model },
        n_predict,
        draft_sampling: draft_sampling.into(),
    })
}

/// Runtime LoRA management from the `INFERENCE_RS_*` environment variables.
pub(crate) fn adapter_spec_from_env() -> AdapterSpec {
    let config = LoraAdapterApiConfig::from_env();
    AdapterSpec {
        runtime_updates: config.enabled(),
        root: config.allowed_root().map(Path::to_path_buf),
    }
}

pub(crate) fn skills_root(runtime: &RuntimeOptions) -> std::path::PathBuf {
    #[cfg(feature = "code-execution")]
    if let Some(dir) = &runtime.skills_dir {
        return dir.clone();
    }
    let _ = runtime;
    SkillStore::default_root()
}

/// Loads `spec` and serves it over HTTP, with the web UI and MCP server the options ask for.
pub(crate) async fn serve_engine(spec: EngineSpec, server: &ServerOptions) -> Result<()> {
    if server.mcp_port == Some(server.port) {
        anyhow::bail!(
            "--mcp-port must differ from the HTTP --port ({})",
            server.port
        );
    }
    let ui = (!server.no_ui).then(|| UiOptions::from_agentic(&spec.agentic));
    let engine = Engine::load(spec).await?;
    let mut app = InferenceRsServerRouterBuilder::new()
        .with_engine(&engine)
        .with_observability_config(server.observability_config())
        .build()
        .await?;
    if let Some(ui) = ui {
        app = inference_webui::mount(app, &engine, ui, server.observability_config()).await?;
        info!(
            "UI available at http://{}:{}{UI_ROUTE}",
            server.host, server.port
        );
    }
    let options = ServeOptions {
        host: &server.host,
        port: server.port,
        mcp_port: server.mcp_port,
    };
    serve(app, &engine, options).await
}

/// Convert our clean ModelType to the legacy ModelSelected enum
pub(crate) fn convert_to_model_selected(
    model_type: &ModelType,
    matformer: &MatformerSelection,
) -> Result<ModelSelected> {
    match model_type {
        ModelType::Auto {
            model,
            format,
            adapter,
            quantization,
            device,
            cache: _,
            multimodal,
        } => {
            // If user explicitly specified a quantized format, handle it
            let format_type = format.format.unwrap_or(ModelFormat::Plain);
            validate_mmproj_format(format)?;
            adapter.validate().map_err(anyhow::Error::msg)?;
            let has_lora = adapter.dynamic_lora_enabled();
            let has_legacy_lora = adapter.legacy_lora.is_some();
            let has_xlora = adapter.xlora.is_some();

            // For GGUF/GGML formats, delegate to text model conversion which has proper validation
            match format_type {
                ModelFormat::Gguf | ModelFormat::Ggml => {
                    let picks_by_quant =
                        matches!(format_type, ModelFormat::Gguf) && quantization.quant.is_some();
                    if format.quantized_file.is_none() && !picks_by_quant {
                        match format_type {
                            ModelFormat::Gguf => anyhow::bail!(
                                "GGUF format requires a model file. Pass `-f <model.gguf>`, or use \
                                 `-m <GGUF-repo> --quant <level>` to select one automatically."
                            ),
                            ModelFormat::Ggml => anyhow::bail!(
                                "GGML format requires a model file. Pass \
                                 `-m <model-repo-or-directory> -f <model.ggml>`."
                            ),
                            ModelFormat::Plain => unreachable!(),
                        }
                    }
                    // Use the text model conversion which handles GGUF/GGML properly
                    return convert_text_model(
                        model,
                        format,
                        adapter,
                        quantization,
                        device,
                        matformer,
                        Some(multimodal),
                    );
                }
                ModelFormat::Plain => {
                    // For plain format with adapters, also use text model conversion
                    if has_lora || has_legacy_lora || has_xlora {
                        return convert_text_model(
                            model,
                            format,
                            adapter,
                            quantization,
                            device,
                            matformer,
                            Some(multimodal),
                        );
                    }
                }
            }

            // Use Run (auto-loader) for auto mode without explicit quantized format
            Ok(ModelSelected::Run {
                model_id: model.model_id.clone(),
                quant: quantization.quant.clone(),
                tokenizer_json: model
                    .tokenizer
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string()),
                dtype: model.dtype,
                topology: device
                    .topology
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string()),
                organization: quantization.isq_organization,
                write_uqff: None,
                from_uqff: quantization.from_uqff.clone(),
                imatrix: quantization.imatrix.clone(),
                calibration_file: quantization.calibration_file.clone(),
                max_edge: multimodal.max_edge,
                max_seq_len: device.max_seq_len,
                max_batch_size: device.max_batch_size,
                max_num_images: multimodal.max_num_images,
                max_image_length: multimodal.max_image_length,
                hf_cache_path: device.hf_cache.clone(),
                matformer_config_path: matformer.config_path.clone(),
                matformer_slice_name: matformer.slice_name.clone(),
            })
        }

        ModelType::Text {
            model,
            format,
            adapter,
            quantization,
            device,
            cache: _,
        } => convert_text_model(
            model,
            format,
            adapter,
            quantization,
            device,
            matformer,
            None,
        ),

        ModelType::Multimodal {
            model,
            format,
            adapter,
            quantization,
            device,
            cache: _,
            multimodal,
        } => {
            validate_mmproj_format(format)?;
            adapter.validate().map_err(anyhow::Error::msg)?;
            let adapter = adapter.as_adapter_options();
            let mut model = model.clone();
            model.arch = None;
            match format.format.unwrap_or(ModelFormat::Plain) {
                ModelFormat::Gguf => convert_text_model(
                    &model,
                    format,
                    &adapter,
                    quantization,
                    device,
                    matformer,
                    Some(multimodal),
                )
                .map(require_projector),
                ModelFormat::Ggml => {
                    anyhow::bail!(
                        "GGML is not supported for multimodal models; use a GGUF model with \
                         `--mmproj`, or use plain safetensors"
                    )
                }
                ModelFormat::Plain if adapter.dynamic_lora_enabled() => convert_text_model(
                    &model,
                    format,
                    &adapter,
                    quantization,
                    device,
                    matformer,
                    Some(multimodal),
                )
                .map(require_projector),
                ModelFormat::Plain => Ok(ModelSelected::MultimodalPlain {
                    quant: quantization.quant.clone(),
                    model_id: model.model_id.clone(),
                    tokenizer_json: model
                        .tokenizer
                        .as_ref()
                        .map(|p| p.to_string_lossy().to_string()),
                    arch: None,
                    dtype: model.dtype,
                    topology: device
                        .topology
                        .as_ref()
                        .map(|p| p.to_string_lossy().to_string()),
                    write_uqff: None,
                    from_uqff: quantization.from_uqff.clone(),
                    max_edge: multimodal.max_edge,
                    calibration_file: quantization.calibration_file.clone(),
                    imatrix: quantization.imatrix.clone(),
                    max_seq_len: device.max_seq_len,
                    max_batch_size: device.max_batch_size,
                    max_num_images: multimodal
                        .max_num_images
                        .unwrap_or(AutoDeviceMapParams::DEFAULT_MAX_NUM_IMAGES),
                    max_image_length: multimodal
                        .max_image_length
                        .unwrap_or(AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH),
                    hf_cache_path: device.hf_cache.clone(),
                    matformer_config_path: matformer.config_path.clone(),
                    matformer_slice_name: matformer.slice_name.clone(),
                    organization: quantization.isq_organization,
                }),
            }
        }

        ModelType::Diffusion { model, device: _ } => Ok(ModelSelected::DiffusionPlain {
            model_id: model.model_id.clone(),
            arch: DiffusionLoaderType::Flux,
            dtype: model.dtype,
        }),

        ModelType::Speech { model, device: _ } => Ok(ModelSelected::Speech {
            model_id: model.model_id.clone(),
            dac_model_id: None,
            arch: SpeechLoaderType::Dia,
            dtype: model.dtype,
        }),

        ModelType::Embedding {
            model,
            format,
            quantization,
            device,
            cache: _,
        } => {
            validate_mmproj_format(format)?;
            if !matches!(format.format, None | Some(ModelFormat::Plain)) {
                anyhow::bail!("Embedding models do not support GGUF or GGML format");
            }
            Ok(ModelSelected::Embedding {
                quant: quantization.quant.clone(),
                model_id: model.model_id.clone(),
                tokenizer_json: model
                    .tokenizer
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string()),
                arch: None,
                dtype: model.dtype,
                topology: device
                    .topology
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string()),
                write_uqff: None,
                from_uqff: quantization.from_uqff.clone(),
                imatrix: quantization.imatrix.clone(),
                calibration_file: quantization.calibration_file.clone(),
                hf_cache_path: device.hf_cache.clone(),
            })
        }
    }
}

/// An explicit multimodal model needs its projector, whether named, found beside the file or found by `quant`.
pub(crate) fn require_projector(mut model: ModelSelected) -> ModelSelected {
    if let ModelSelected::GGUF {
        mmproj_selection, ..
    }
    | ModelSelected::Lora {
        mmproj_selection, ..
    } = &mut model
    {
        *mmproj_selection = MmprojSelection::Required;
    }
    model
}

pub(crate) fn gguf_mmproj_selection(direct_file_only: bool) -> MmprojSelection {
    // `-f` without `-m` points at a file: its directory is not a repository to judge
    if direct_file_only {
        MmprojSelection::Any
    } else {
        MmprojSelection::ArtifactRepo
    }
}

/// The GGUF file `-f` names, or none for `--quant` to pick.
pub(crate) fn gguf_filename(quantized_file: Option<&str>, quant: Option<&str>) -> Result<String> {
    match (quantized_file, quant) {
        (Some(file), _) => Ok(file.to_string()),
        (None, Some(_)) => Ok(String::new()),
        (None, None) => anyhow::bail!(
            "GGUF format requires a model file. Pass `-f <model.gguf>`, or use `-m <GGUF-repo> --quant <level>` \
             to select one automatically."
        ),
    }
}

fn validate_mmproj_format(format_opts: &FormatOptions) -> Result<()> {
    if format_opts.mmproj.is_some() && !matches!(format_opts.format, Some(ModelFormat::Gguf)) {
        anyhow::bail!("`--mmproj` requires GGUF format");
    }
    Ok(())
}

/// Convert text model with orthogonal format/adapter flags
fn convert_text_model(
    model: &ModelSourceOptions,
    format_opts: &FormatOptions,
    adapter: &AdapterOptions,
    quantization: &QuantizationOptions,
    device: &DeviceOptions,
    matformer: &MatformerSelection,
    multimodal: Option<&MultimodalOptions>,
) -> Result<ModelSelected> {
    validate_mmproj_format(format_opts)?;
    adapter.validate().map_err(anyhow::Error::msg)?;
    let format_type = format_opts.format.unwrap_or(ModelFormat::Plain);
    let has_lora = adapter.dynamic_lora_enabled();
    let has_legacy_lora = adapter.legacy_lora.is_some();
    let has_xlora = adapter.xlora.is_some();
    if format_opts.mmproj.is_some() && (has_legacy_lora || has_xlora) {
        anyhow::bail!("Multimodal GGUF does not support legacy LoRA or X-LoRA adapters");
    }

    match (format_type, has_lora, has_legacy_lora, has_xlora) {
        // Plain format
        (ModelFormat::Plain, false, false, false) => Ok(ModelSelected::Plain {
            quant: quantization.quant.clone(),
            model_id: model.model_id.clone(),
            tokenizer_json: model
                .tokenizer
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            arch: model.arch.clone(),
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            organization: quantization.isq_organization,
            write_uqff: None,
            from_uqff: quantization.from_uqff.clone(),
            imatrix: quantization.imatrix.clone(),
            calibration_file: quantization.calibration_file.clone(),
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
            hf_cache_path: device.hf_cache.clone(),
            matformer_config_path: matformer.config_path.clone(),
            matformer_slice_name: matformer.slice_name.clone(),
        }),

        (ModelFormat::Plain, true, false, false) => Ok(ModelSelected::Lora {
            mmproj_selection: gguf_mmproj_selection(format_opts.direct_file_only),
            quant: quantization.quant.clone(),
            model_id: model.model_id.clone(),
            tokenizer_json: model
                .tokenizer
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            adapters: adapter.lora.clone(),
            runtime_config: adapter.lora_runtime_config(),
            arch: model.arch.clone(),
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            organization: quantization.isq_organization,
            write_uqff: None,
            from_uqff: quantization.from_uqff.clone(),
            imatrix: quantization.imatrix.clone(),
            calibration_file: quantization.calibration_file.clone(),
            max_edge: multimodal.and_then(|options| options.max_edge),
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
            max_num_images: multimodal.and_then(|options| options.max_num_images),
            max_image_length: multimodal.and_then(|options| options.max_image_length),
            hf_cache_path: device.hf_cache.clone(),
            matformer_config_path: matformer.config_path.clone(),
            matformer_slice_name: matformer.slice_name.clone(),
        }),

        (ModelFormat::Plain, false, false, true) => Ok(ModelSelected::XLora {
            quant: quantization.quant.clone(),
            model_id: Some(model.model_id.clone()),
            tokenizer_json: model
                .tokenizer
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            xlora_model_id: adapter.xlora.clone().unwrap_or_default(),
            order: adapter
                .xlora_order
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
            tgt_non_granular_index: adapter.tgt_non_granular_index,
            arch: model.arch.clone(),
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            write_uqff: None,
            from_uqff: quantization.from_uqff.clone(),
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
            hf_cache_path: device.hf_cache.clone(),
            organization: quantization.isq_organization,
        }),

        (ModelFormat::Gguf, dynamic_lora, false, false) => Ok(ModelSelected::GGUF {
            quant: quantization.quant.clone(),
            mmproj_selection: gguf_mmproj_selection(format_opts.direct_file_only),
            tok_model_id: format_opts.tok_model_id.clone(),
            quantized_model_id: model.model_id.clone(),
            quantized_filename: gguf_filename(
                format_opts.quantized_file.as_deref(),
                quantization.quant.as_deref(),
            )?,
            tokenizer_json: model
                .tokenizer
                .as_ref()
                .map(|path| path.to_string_lossy().to_string()),
            mmproj_filename: format_opts.mmproj.clone(),
            lora_adapters: adapter.lora.clone(),
            lora_runtime_config: dynamic_lora.then(|| adapter.lora_runtime_config()),
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            organization: quantization.isq_organization,
            write_uqff: None,
            imatrix: quantization.imatrix.clone(),
            calibration_file: quantization.calibration_file.clone(),
            max_edge: multimodal.and_then(|options| options.max_edge),
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
            max_num_images: multimodal.and_then(|options| options.max_num_images),
            max_image_length: multimodal.and_then(|options| options.max_image_length),
            hf_cache_path: device.hf_cache.clone(),
            matformer_config_path: matformer.config_path.clone(),
            matformer_slice_name: matformer.slice_name.clone(),
        }),

        (ModelFormat::Gguf, false, true, false) => Ok(ModelSelected::LoraGGUF {
            quant: quantization.quant.clone(),
            tok_model_id: format_opts.tok_model_id.clone(),
            quantized_model_id: model.model_id.clone(),
            quantized_filename: gguf_filename(
                format_opts.quantized_file.as_deref(),
                quantization.quant.as_deref(),
            )?,
            adapters_model_id: adapter.legacy_lora.clone().unwrap_or_default(),
            order: adapter
                .legacy_lora_order
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
            tokenizer_json: model
                .tokenizer
                .as_ref()
                .map(|path| path.to_string_lossy().to_string()),
            organization: quantization.isq_organization,
            write_uqff: None,
            imatrix: quantization.imatrix.clone(),
            calibration_file: quantization.calibration_file.clone(),
            hf_cache_path: device.hf_cache.clone(),
            matformer_config_path: matformer.config_path.clone(),
            matformer_slice_name: matformer.slice_name.clone(),
        }),

        (ModelFormat::Gguf, false, false, true) => Ok(ModelSelected::XLoraGGUF {
            quant: quantization.quant.clone(),
            tok_model_id: format_opts.tok_model_id.clone(),
            quantized_model_id: model.model_id.clone(),
            quantized_filename: gguf_filename(
                format_opts.quantized_file.as_deref(),
                quantization.quant.as_deref(),
            )?,
            xlora_model_id: adapter.xlora.clone().unwrap_or_default(),
            order: adapter
                .xlora_order
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
            tgt_non_granular_index: adapter.tgt_non_granular_index,
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
            tokenizer_json: model
                .tokenizer
                .as_ref()
                .map(|path| path.to_string_lossy().to_string()),
            organization: quantization.isq_organization,
            write_uqff: None,
            imatrix: quantization.imatrix.clone(),
            calibration_file: quantization.calibration_file.clone(),
            hf_cache_path: device.hf_cache.clone(),
            matformer_config_path: matformer.config_path.clone(),
            matformer_slice_name: matformer.slice_name.clone(),
        }),

        // GGML format
        (ModelFormat::Ggml, false, false, false) => Ok(ModelSelected::GGML {
            tok_model_id: format_opts
                .tok_model_id
                .clone()
                .unwrap_or_else(|| model.model_id.clone()),
            tokenizer_json: model
                .tokenizer
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            quantized_model_id: model.model_id.clone(),
            quantized_filename: format_opts
                .quantized_file
                .clone()
                .context("GGML model type requires `--quantized-file`/`-f` to be specified")?,
            gqa: format_opts.gqa,
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
        }),

        (ModelFormat::Ggml, false, true, false) => Ok(ModelSelected::LoraGGML {
            tok_model_id: Some(
                format_opts
                    .tok_model_id
                    .clone()
                    .unwrap_or_else(|| model.model_id.clone()),
            ),
            tokenizer_json: model
                .tokenizer
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            quantized_model_id: model.model_id.clone(),
            quantized_filename: format_opts
                .quantized_file
                .clone()
                .context("GGML model type requires `--quantized-file`/`-f` to be specified")?,
            adapters_model_id: adapter.legacy_lora.clone().unwrap_or_default(),
            order: adapter
                .legacy_lora_order
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
            gqa: format_opts.gqa,
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
        }),

        (ModelFormat::Ggml, false, false, true) => Ok(ModelSelected::XLoraGGML {
            tok_model_id: Some(
                format_opts
                    .tok_model_id
                    .clone()
                    .unwrap_or_else(|| model.model_id.clone()),
            ),
            tokenizer_json: model
                .tokenizer
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            quantized_model_id: model.model_id.clone(),
            quantized_filename: format_opts
                .quantized_file
                .clone()
                .context("GGML model type requires `--quantized-file`/`-f` to be specified")?,
            xlora_model_id: adapter.xlora.clone().unwrap_or_default(),
            order: adapter
                .xlora_order
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
            tgt_non_granular_index: adapter.tgt_non_granular_index,
            gqa: format_opts.gqa,
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
        }),

        (ModelFormat::Plain, false, true, false) => {
            anyhow::bail!("--legacy-lora is only supported with raw GGUF or GGML models")
        }
        (ModelFormat::Ggml, true, false, false) => {
            anyhow::bail!(
                "dynamic --lora adapters are not supported with raw GGML models; use \
                 --legacy-lora with --legacy-lora-order"
            )
        }
        _ => anyhow::bail!("dynamic LoRA, legacy LoRA, and X-LoRA are mutually exclusive"),
    }
}

pub(crate) fn extract_paged_attn_settings(
    model_type: &ModelType,
) -> crate::args::PagedAttnBuilderFlags {
    let cache = match model_type {
        ModelType::Auto { cache, .. } => cache,
        ModelType::Text { cache, .. } => cache,
        ModelType::Multimodal { cache, .. } => cache,
        ModelType::Embedding { cache, .. } => cache,
        _ => return (None, None, None, None, None, PagedCacheType::Auto),
    };

    cache.paged_attn.clone().into_builder_flags()
}

pub(crate) fn extract_device_settings(model_type: &ModelType) -> (bool, Option<Vec<String>>) {
    let device = match model_type {
        ModelType::Auto { device, .. } => device,
        ModelType::Text { device, .. } => device,
        ModelType::Multimodal { device, .. } => device,
        ModelType::Diffusion { device, .. } => device,
        ModelType::Speech { device, .. } => device,
        ModelType::Embedding { device, .. } => device,
    };

    (device.cpu, device.device_layers.clone())
}

pub(crate) fn extract_isq_setting(model_type: &ModelType) -> Option<String> {
    model_type
        .quantization()
        .and_then(|q| q.in_situ_quant.clone())
}

pub(crate) fn extract_encoder_cache_memory_bytes(model_type: &ModelType) -> Result<Option<usize>> {
    let memory_mb = match model_type {
        ModelType::Auto { multimodal, .. } | ModelType::Multimodal { multimodal, .. } => {
            multimodal.encoder_cache_memory_mb
        }
        _ => None,
    };
    memory_mb
        .map(|memory_mb| {
            memory_mb
                .get()
                .checked_mul(MEBIBYTE_BYTES)
                .context("encoder cache memory capacity overflow")
        })
        .transpose()
}

pub(crate) fn extract_quant_flag(model_type: &ModelType) -> Option<String> {
    model_type.quantization().and_then(|q| q.quant.clone())
}

pub(crate) fn extract_hf_config_settings(
    model_type: &ModelType,
) -> (Option<usize>, Option<inference_core::HfConfigOverrides>) {
    let model = match model_type {
        ModelType::Auto { model, .. }
        | ModelType::Text { model, .. }
        | ModelType::Multimodal { model, .. }
        | ModelType::Diffusion { model, .. }
        | ModelType::Speech { model, .. }
        | ModelType::Embedding { model, .. } => model,
    };
    (model.max_model_len, model.hf_overrides.clone())
}

/// The `--quant` rules that are about the flags; what `--quant` picks is resolved when the engine loads.
pub(crate) fn normalize_quant_flags(model_type: &mut ModelType) -> Result<()> {
    let quant = model_type.quantization().and_then(|q| q.quant.clone());
    let legacy_lora = matches!(
        model_type,
        ModelType::Auto { adapter, .. } | ModelType::Text { adapter, .. } if adapter.legacy_lora.is_some()
    );
    let Some(format) = model_type.format_mut() else {
        return Ok(());
    };
    format.normalize()?;
    if quant.is_none() {
        return Ok(());
    }
    if format.quantized_file.is_some() {
        anyhow::bail!("`--quant` and `--quantized-file` are mutually exclusive");
    }
    match format.format {
        Some(ModelFormat::Ggml) => {
            anyhow::bail!("`--quant` cannot select a GGML file; pass one explicitly with `-f`")
        }
        // these only mean something for a quantized file, as `--mmproj` does
        None if format.tok_model_id.is_some() || legacy_lora => {
            format.format = Some(ModelFormat::Gguf)
        }
        _ => {}
    }
    Ok(())
}

/// Load an MCP client config from `--mcp-config` (or `MCP_CONFIG_PATH` if no path given).
pub(crate) fn load_mcp_config(path: Option<&Path>) -> Result<Option<McpClientConfig>> {
    let resolved = match path {
        Some(p) => Some(p.to_path_buf()),
        None => std::env::var("MCP_CONFIG_PATH").ok().map(Into::into),
    };
    let Some(path) = resolved else {
        return Ok(None);
    };
    let contents = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read MCP config {}", path.display()))?;
    let config: McpClientConfig = serde_json::from_str(&contents)
        .with_context(|| format!("Failed to parse MCP config {}", path.display()))?;
    info!(
        "Loaded MCP configuration from {} ({} servers)",
        path.display(),
        config.servers.len()
    );
    Ok(Some(config))
}

/// Build a `CodeExecutionConfig` from runtime options. Returns `None` when code execution is off.
#[cfg(feature = "code-execution")]
pub(crate) fn build_code_exec_config(
    runtime: &RuntimeOptions,
) -> Option<inference_core::CodeExecutionConfig> {
    if !runtime.enable_code_execution {
        return None;
    }
    let mut config = inference_core::CodeExecutionConfig::default();
    if let Some(python) = runtime.code_exec_python.clone() {
        config.python_path = python;
    }
    if let Some(timeout) = runtime.code_exec_timeout {
        config.timeout_secs = timeout;
    }
    config.working_directory = runtime.code_exec_workdir.clone();
    Some(config)
}

/// Build a `ShellConfig` from runtime options. Returns `None` when shell execution is off.
#[cfg(feature = "code-execution")]
pub(crate) fn build_shell_config(runtime: &RuntimeOptions) -> Option<inference_core::ShellConfig> {
    if !runtime.enable_shell {
        return None;
    }
    let mut config = inference_core::ShellConfig::default();
    if let Some(shell_path) = runtime.shell_path.clone() {
        config.shell_path = shell_path;
    }
    if let Some(timeout) = runtime.shell_timeout {
        config.timeout_secs = timeout;
    }
    config.working_directory = runtime.shell_workdir.clone();
    config.permission = runtime.code_exec_permission.into();
    Some(config)
}

pub(crate) fn apply_agent_mode(runtime: &mut RuntimeOptions) {
    if !runtime.agent {
        return;
    }
    runtime.enable_search = true;
    #[cfg(feature = "code-execution")]
    {
        runtime.enable_code_execution = true;
        runtime.enable_shell = true;
    }
}

pub(crate) fn validate_agent_options(runtime: &RuntimeOptions) -> Result<()> {
    if runtime.search_embedding_model.is_some() && !runtime.enable_search {
        anyhow::bail!(
            "`--search-embedding-model` requires `--enable-search` (or `--agent`/`--agentic`)"
        );
    }
    #[cfg(feature = "code-execution")]
    {
        let touches_code_exec = runtime.code_exec_python.is_some()
            || runtime.code_exec_timeout.is_some()
            || runtime.code_exec_workdir.is_some();
        if touches_code_exec && !runtime.enable_code_execution {
            anyhow::bail!(
                "`--code-exec-*` options require `--enable-code-execution` (or `--agent`/`--agentic`)"
            );
        }
        let touches_shell = runtime.shell_path.is_some()
            || runtime.shell_timeout.is_some()
            || runtime.shell_workdir.is_some()
            || runtime.skills_dir.is_some();
        if touches_shell && !runtime.enable_shell {
            anyhow::bail!(
                "`--shell-*` and `--skills-dir` options require `--enable-shell` (or `--agent`/`--agentic`)"
            );
        }
    }
    Ok(())
}

pub(crate) fn log_agent_runtime(runtime: &RuntimeOptions, max_tool_rounds: Option<usize>) {
    if !runtime.agent
        && !runtime.enable_search
        && !is_code_execution_enabled(runtime)
        && !is_shell_enabled(runtime)
    {
        return;
    }

    let rounds = max_tool_rounds.unwrap_or(inference_core::DEFAULT_MAX_TOOL_ROUNDS);
    let mode = if runtime.agent { "agent" } else { "tools" };
    tracing::info!(
        "{mode}: search {}, code execution {}, shell {}, approvals {}, max tool rounds {rounds}",
        search_summary(runtime),
        code_execution_summary(runtime),
        shell_summary(runtime),
        agent_permission_summary(runtime.code_exec_permission)
    );
    log_agent_runtime_details(runtime);
}

fn search_summary(runtime: &RuntimeOptions) -> String {
    if !runtime.enable_search {
        return "off".to_string();
    }
    let model = runtime
        .search_embedding_model
        .map(inference_core::SearchEmbeddingModel::from)
        .unwrap_or_default();
    format!("on (reranker {model})")
}

fn agent_permission_summary(permission: CodeExecPermissionArg) -> &'static str {
    match permission {
        CodeExecPermissionArg::Auto => "auto",
        CodeExecPermissionArg::Ask => "ask",
        CodeExecPermissionArg::Deny => "deny",
    }
}

#[cfg(feature = "code-execution")]
fn is_code_execution_enabled(runtime: &RuntimeOptions) -> bool {
    runtime.enable_code_execution
}
#[cfg(not(feature = "code-execution"))]
fn is_code_execution_enabled(_runtime: &RuntimeOptions) -> bool {
    false
}

#[cfg(feature = "code-execution")]
fn is_shell_enabled(runtime: &RuntimeOptions) -> bool {
    runtime.enable_shell
}
#[cfg(not(feature = "code-execution"))]
fn is_shell_enabled(_runtime: &RuntimeOptions) -> bool {
    false
}

#[cfg(feature = "code-execution")]
fn code_execution_summary(runtime: &RuntimeOptions) -> &'static str {
    if !runtime.enable_code_execution {
        "off"
    } else {
        "on"
    }
}

#[cfg(feature = "code-execution")]
fn shell_summary(runtime: &RuntimeOptions) -> &'static str {
    if runtime.enable_shell { "on" } else { "off" }
}

#[cfg(feature = "code-execution")]
fn log_agent_runtime_details(runtime: &RuntimeOptions) {
    if !runtime.enable_code_execution && !runtime.enable_shell {
        return;
    }
    if runtime.enable_code_execution {
        let python = runtime
            .code_exec_python
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "python3 (default)".to_string());
        let timeout = runtime.code_exec_timeout.map_or_else(
            || {
                format!(
                    "{}s (default)",
                    inference_core::DEFAULT_CODE_EXEC_TIMEOUT_SECS
                )
            },
            |t| format!("{t}s"),
        );
        let workdir = runtime
            .code_exec_workdir
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "per-session temp dir".to_string());
        tracing::info!(
            "code-exec: python={python}, timeout={timeout}, workdir={workdir}, permission={}",
            agent_permission_summary(runtime.code_exec_permission)
        );
    }
    if runtime.enable_shell {
        let shell = runtime
            .shell_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "/bin/sh (default)".to_string());
        let timeout = runtime.shell_timeout.map_or_else(
            || format!("{}s (default)", inference_core::DEFAULT_SHELL_TIMEOUT_SECS),
            |t| format!("{t}s"),
        );
        let workdir = runtime
            .shell_workdir
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "per-session temp dir".to_string());
        let skills_dir = runtime
            .skills_dir
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "system temp dir".to_string());
        tracing::info!(
            "shell: shell={shell}, timeout={timeout}, workdir={workdir}, skills_dir={skills_dir}, permission={}",
            agent_permission_summary(runtime.code_exec_permission)
        );
    }
}
#[cfg(not(feature = "code-execution"))]
fn code_execution_summary(runtime: &RuntimeOptions) -> &'static str {
    if runtime.agent {
        "not compiled in"
    } else {
        "off"
    }
}

#[cfg(not(feature = "code-execution"))]
fn shell_summary(runtime: &RuntimeOptions) -> &'static str {
    if runtime.agent {
        "not compiled in"
    } else {
        "off"
    }
}

#[cfg(not(feature = "code-execution"))]
fn log_agent_runtime_details(runtime: &RuntimeOptions) {
    if runtime.agent {
        tracing::warn!(
            "code-exec: not compiled in (build with `--features code-execution`); --agent enabled search only"
        );
    }
}

#[cfg(test)]
mod tests {
    use inference_core::{
        AutoDeviceMapParams, IsqOrganization, LoraAdapterSpec, ModelDType, NormalLoaderType,
    };
    use std::{num::NonZeroUsize, path::PathBuf};

    use super::*;

    fn test_model() -> ModelSourceOptions {
        ModelSourceOptions {
            model_id: "org/base".to_string(),
            tokenizer: None,
            arch: None,
            dtype: ModelDType::Auto,
            hf_overrides: None,
            max_model_len: None,
        }
    }

    #[test]
    fn extracts_runtime_hf_config_settings() {
        let mut model = test_model();
        model.max_model_len = Some(131072);
        model.hf_overrides = Some(
            r#"{"text_config":{"max_position_embeddings":131072}}"#
                .parse()
                .unwrap(),
        );
        let model_type = ModelType::Auto {
            model,
            format: FormatOptions::default(),
            adapter: AdapterOptions::default(),
            quantization: QuantizationOptions::default(),
            device: DeviceOptions::default(),
            cache: Default::default(),
            multimodal: MultimodalOptions::default(),
        };

        let (max_model_len, overrides) = extract_hf_config_settings(&model_type);
        assert_eq!(max_model_len, Some(131072));
        assert_eq!(
            overrides.unwrap().as_value()["text_config"]["max_position_embeddings"],
            131072
        );
    }

    fn auto_with_quant(format: FormatOptions, quant: &str) -> ModelType {
        ModelType::Auto {
            model: test_model(),
            format,
            adapter: AdapterOptions::default(),
            quantization: QuantizationOptions {
                quant: Some(quant.to_string()),
                ..QuantizationOptions::default()
            },
            device: DeviceOptions::default(),
            cache: crate::args::CacheOptions::default(),
            multimodal: MultimodalOptions::default(),
        }
    }

    fn converted(model_type: &mut ModelType) -> ModelSelected {
        normalize_quant_flags(model_type).unwrap();
        convert_to_model_selected(model_type, &MatformerSelection::default()).unwrap()
    }

    #[test]
    fn quant_is_left_for_the_engine_to_resolve() {
        let model = converted(&mut auto_with_quant(FormatOptions::default(), "4"));
        assert!(matches!(model, ModelSelected::Run { quant: Some(ref q), .. } if q == "4"));

        let gguf = FormatOptions {
            format: Some(ModelFormat::Gguf),
            ..FormatOptions::default()
        };
        let model = converted(&mut auto_with_quant(gguf, "4"));
        let ModelSelected::GGUF {
            quant,
            quantized_filename,
            mmproj_selection,
            ..
        } = &model
        else {
            panic!("expected GGUF, got {model:?}");
        };
        assert_eq!(quant.as_deref(), Some("4"));
        assert!(quantized_filename.is_empty());
        assert_eq!(*mmproj_selection, MmprojSelection::ArtifactRepo);
        assert!(model.needs_source_resolution());
    }

    #[test]
    fn tok_model_id_with_quant_means_gguf() {
        let format = FormatOptions {
            tok_model_id: Some("org/base".to_string()),
            ..FormatOptions::default()
        };
        let model = converted(&mut auto_with_quant(format, "4"));
        assert!(matches!(
            model,
            ModelSelected::GGUF { tok_model_id: Some(ref id), quant: Some(_), .. } if id == "org/base"
        ));
    }

    #[test]
    fn gguf_projector_selection_follows_how_the_file_was_given() {
        let selection = |format: FormatOptions, multimodal: bool| {
            let mut model_type = if multimodal {
                test_multimodal_model(format)
            } else {
                ModelType::Auto {
                    model: test_model(),
                    format,
                    adapter: AdapterOptions::default(),
                    quantization: QuantizationOptions::default(),
                    device: DeviceOptions::default(),
                    cache: crate::args::CacheOptions::default(),
                    multimodal: MultimodalOptions::default(),
                }
            };
            match converted(&mut model_type) {
                ModelSelected::GGUF {
                    mmproj_selection, ..
                } => mmproj_selection,
                other => panic!("expected GGUF, got {other:?}"),
            }
        };
        let file = |direct_file_only| FormatOptions {
            quantized_file: Some("model.gguf".to_string()),
            direct_file_only,
            ..FormatOptions::default()
        };
        assert_eq!(selection(file(false), false), MmprojSelection::ArtifactRepo);
        assert_eq!(selection(file(true), false), MmprojSelection::Any);
        assert_eq!(selection(file(false), true), MmprojSelection::Required);
    }

    #[test]
    fn multimodal_lora_requires_a_projector_once_quant_picks_a_gguf() {
        let mut model_type = ModelType::Multimodal {
            model: test_model(),
            format: FormatOptions::default(),
            adapter: MultimodalAdapterOptions {
                enable_lora: true,
                ..MultimodalAdapterOptions::default()
            },
            quantization: QuantizationOptions {
                quant: Some("4".to_string()),
                ..QuantizationOptions::default()
            },
            device: DeviceOptions::default(),
            cache: crate::args::CacheOptions::default(),
            multimodal: MultimodalOptions::default(),
        };
        let model = converted(&mut model_type);
        assert!(matches!(
            model,
            ModelSelected::Lora {
                quant: Some(_),
                mmproj_selection: MmprojSelection::Required,
                ..
            }
        ));
    }

    #[test]
    fn legacy_lora_with_quant_means_gguf() {
        let mut model_type = auto_with_quant(FormatOptions::default(), "4");
        let ModelType::Auto { adapter, .. } = &mut model_type else {
            unreachable!()
        };
        adapter.legacy_lora = Some("org/adapters".to_string());
        adapter.legacy_lora_order = Some(PathBuf::from("order.json"));
        let model = converted(&mut model_type);
        assert!(matches!(
            model,
            ModelSelected::LoraGGUF { ref quantized_filename, quant: Some(_), .. } if quantized_filename.is_empty()
        ));
    }

    #[test]
    fn multimodal_dynamic_lora_gguf_keeps_its_runtime_and_requires_a_projector() {
        let mut model_type = ModelType::Multimodal {
            model: test_model(),
            format: FormatOptions {
                quantized_file: Some("model.gguf".to_string()),
                ..FormatOptions::default()
            },
            adapter: MultimodalAdapterOptions {
                enable_lora: true,
                ..MultimodalAdapterOptions::default()
            },
            quantization: QuantizationOptions::default(),
            device: DeviceOptions::default(),
            cache: crate::args::CacheOptions::default(),
            multimodal: MultimodalOptions::default(),
        };
        let model = converted(&mut model_type);
        assert!(matches!(
            model,
            ModelSelected::GGUF {
                lora_runtime_config: Some(_),
                mmproj_selection: MmprojSelection::Required,
                ..
            }
        ));
    }

    #[test]
    fn quant_flag_conflicts_are_refused_before_loading() {
        let file = FormatOptions {
            quantized_file: Some("model.gguf".to_string()),
            ..FormatOptions::default()
        };
        let error = normalize_quant_flags(&mut auto_with_quant(file, "4")).unwrap_err();
        assert!(error.to_string().contains("mutually exclusive"), "{error}");

        let ggml = FormatOptions {
            format: Some(ModelFormat::Ggml),
            ..FormatOptions::default()
        };
        let error = normalize_quant_flags(&mut auto_with_quant(ggml, "4")).unwrap_err();
        assert!(
            error.to_string().contains("cannot select a GGML file"),
            "{error}"
        );
    }

    fn test_multimodal_model(format: FormatOptions) -> ModelType {
        ModelType::Multimodal {
            model: test_model(),
            format,
            adapter: MultimodalAdapterOptions::default(),
            quantization: QuantizationOptions::default(),
            device: DeviceOptions::default(),
            cache: crate::args::CacheOptions::default(),
            multimodal: MultimodalOptions::default(),
        }
    }

    #[test]
    fn encoder_cache_memory_converts_mib_to_bytes() {
        let mut model_type = test_multimodal_model(FormatOptions::default());
        let ModelType::Multimodal { multimodal, .. } = &mut model_type else {
            unreachable!()
        };
        multimodal.encoder_cache_memory_mb = NonZeroUsize::new(64);

        assert_eq!(
            extract_encoder_cache_memory_bytes(&model_type).unwrap(),
            Some(64 * MEBIBYTE_BYTES)
        );
    }

    const XDG_CACHE_HOME: &str = "XDG_CACHE_HOME";

    #[tokio::test(flavor = "multi_thread")]
    async fn the_ui_mounts_beside_the_api_with_the_engines_tools() -> anyhow::Result<()> {
        use axum::body::{Body, to_bytes};
        use tower::ServiceExt;

        let cache = tempfile::tempdir()?;
        let previous = std::env::var_os(XDG_CACHE_HOME);
        // nextest runs each test in its own process, so no other thread reads the environment meanwhile
        unsafe { std::env::set_var(XDG_CACHE_HOME, cache.path()) };
        let dir = crate::commands::tiny_support::tiny_checkpoint()?;
        let spec: EngineSpec = serde_json::from_value(serde_json::json!({
            "model": {"MultimodalPlain": {"model_id": dir.path().to_string_lossy(), "dtype": "f32"}},
            "runtime": {"device": "cpu"},
            "agentic": {"tool_dispatch_url": "http://127.0.0.1:9/tools"},
        }))?;
        let ui = UiOptions::from_agentic(&spec.agentic);
        let engine = Engine::load(spec).await?;
        let app = InferenceRsServerRouterBuilder::new()
            .with_engine(&engine)
            .build()
            .await?;
        let app = inference_webui::mount(app, &engine, ui, Default::default()).await?;

        let request = axum::http::Request::get(format!("{UI_ROUTE}/api/capabilities"));
        let response = app.oneshot(request.body(Body::empty())?).await?;
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await?)?;
        assert_eq!(body["tool_dispatch_url"], "http://127.0.0.1:9/tools");
        assert_eq!(body["search_enabled"], false);
        assert!(cache.path().join("inference-rs").join("chats").is_dir());
        match previous {
            Some(value) => unsafe { std::env::set_var(XDG_CACHE_HOME, value) },
            None => unsafe { std::env::remove_var(XDG_CACHE_HOME) },
        }
        Ok(())
    }

    #[test]
    fn enable_lora_builds_an_empty_dynamic_runtime() {
        let adapter = AdapterOptions {
            enable_lora: true,
            ..AdapterOptions::default()
        };
        let selected = convert_text_model(
            &test_model(),
            &FormatOptions::default(),
            &adapter,
            &QuantizationOptions::default(),
            &DeviceOptions::default(),
            &MatformerSelection::default(),
            None,
        )
        .unwrap();

        assert!(matches!(
            inference_selection::get_auto_device_map_params(&selected).unwrap(),
            AutoDeviceMapParams::Text { .. }
        ));
        match selected {
            ModelSelected::Lora {
                adapters,
                runtime_config,
                organization,
                imatrix,
                calibration_file,
                max_edge,
                max_num_images,
                max_image_length,
                matformer_config_path,
                matformer_slice_name,
                ..
            } => {
                assert!(adapters.is_empty());
                assert_eq!(runtime_config, adapter.lora_runtime_config());
                assert!(organization.is_none());
                assert!(imatrix.is_none());
                assert!(calibration_file.is_none());
                assert!(max_edge.is_none());
                assert!(max_num_images.is_none());
                assert!(max_image_length.is_none());
                assert!(matformer_config_path.is_none());
                assert!(matformer_slice_name.is_none());
            }
            _ => panic!("expected dynamic LoRA model"),
        }
    }

    #[test]
    fn auto_lora_preserves_multimodal_and_loading_options() {
        let adapter = AdapterOptions {
            enable_lora: true,
            ..AdapterOptions::default()
        };
        let quantization = QuantizationOptions {
            from_uqff: Some("q4k-0.uqff".to_string()),
            isq_organization: Some(IsqOrganization::MoeExpertsOnly),
            imatrix: Some(PathBuf::from("model.imatrix")),
            ..QuantizationOptions::default()
        };
        let device = DeviceOptions {
            max_seq_len: 8192,
            max_batch_size: 7,
            ..DeviceOptions::default()
        };
        let matformer = MatformerSelection {
            config_path: Some(PathBuf::from("matformer.csv")),
            slice_name: Some("slice".to_string()),
        };
        let multimodal = MultimodalOptions {
            encoder_cache_memory_mb: None,
            max_edge: Some(2048),
            max_num_images: Some(5),
            max_image_length: Some(1536),
        };
        let model_type = ModelType::Auto {
            model: test_model(),
            format: FormatOptions::default(),
            adapter,
            quantization,
            device,
            cache: crate::args::CacheOptions::default(),
            multimodal,
        };
        let selected = convert_to_model_selected(&model_type, &matformer).unwrap();

        match inference_selection::get_auto_device_map_params(&selected).unwrap() {
            AutoDeviceMapParams::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            } => {
                assert_eq!(max_seq_len, 8192);
                assert_eq!(max_batch_size, 7);
                assert_eq!(max_image_shape, (1536, 1536));
                assert_eq!(max_num_images, 5);
            }
            _ => panic!("expected multimodal device-map parameters"),
        }
        match selected {
            ModelSelected::Lora {
                organization,
                from_uqff,
                imatrix,
                max_edge,
                max_num_images,
                max_image_length,
                matformer_config_path,
                matformer_slice_name,
                ..
            } => {
                assert!(matches!(
                    organization,
                    Some(IsqOrganization::MoeExpertsOnly)
                ));
                assert_eq!(from_uqff.as_deref(), Some("q4k-0.uqff"));
                assert_eq!(imatrix, Some(PathBuf::from("model.imatrix")));
                assert_eq!(max_edge, Some(2048));
                assert_eq!(max_num_images, Some(5));
                assert_eq!(max_image_length, Some(1536));
                assert_eq!(matformer_config_path, Some(PathBuf::from("matformer.csv")));
                assert_eq!(matformer_slice_name.as_deref(), Some("slice"));
            }
            _ => panic!("expected dynamic LoRA model"),
        }
    }

    #[test]
    fn explicit_multimodal_lora_preserves_multimodal_device_mapping() {
        let mut model = test_model();
        model.arch = Some(NormalLoaderType::Qwen3);
        let model_type = ModelType::Multimodal {
            model,
            format: FormatOptions::default(),
            adapter: MultimodalAdapterOptions {
                enable_lora: true,
                ..MultimodalAdapterOptions::default()
            },
            quantization: QuantizationOptions::default(),
            device: DeviceOptions {
                max_seq_len: 8192,
                max_batch_size: 3,
                ..DeviceOptions::default()
            },
            cache: crate::args::CacheOptions::default(),
            multimodal: MultimodalOptions {
                encoder_cache_memory_mb: None,
                max_edge: Some(1280),
                max_num_images: Some(4),
                max_image_length: Some(1024),
            },
        };
        let selected =
            convert_to_model_selected(&model_type, &MatformerSelection::default()).unwrap();

        let ModelSelected::Lora { arch, .. } = &selected else {
            panic!("expected dynamic LoRA model")
        };
        assert!(arch.is_none());
        match inference_selection::get_auto_device_map_params(&selected).unwrap() {
            AutoDeviceMapParams::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            } => {
                assert_eq!(max_seq_len, 8192);
                assert_eq!(max_batch_size, 3);
                assert_eq!(max_image_shape, (1024, 1024));
                assert_eq!(max_num_images, 4);
            }
            _ => panic!("expected multimodal device-map parameters"),
        }
    }

    #[test]
    fn dynamic_lora_routes_native_text_gguf() {
        let preload = LoraAdapterSpec::new("code", "org/code-lora");
        let adapter = AdapterOptions {
            lora: vec![preload.clone()],
            ..AdapterOptions::default()
        };
        let format = FormatOptions {
            format: Some(ModelFormat::Gguf),
            quantized_file: Some("model.gguf".to_string()),
            ..FormatOptions::default()
        };
        let selected = convert_text_model(
            &test_model(),
            &format,
            &adapter,
            &QuantizationOptions::default(),
            &DeviceOptions::default(),
            &MatformerSelection::default(),
            None,
        )
        .unwrap();

        match selected {
            ModelSelected::GGUF {
                lora_adapters,
                lora_runtime_config,
                mmproj_filename,
                ..
            } => {
                assert_eq!(lora_adapters, vec![preload]);
                assert_eq!(lora_runtime_config, Some(adapter.lora_runtime_config()));
                assert!(mmproj_filename.is_none());
            }
            _ => panic!("expected GGUF model"),
        }
    }

    #[test]
    fn dynamic_lora_routes_native_multimodal_gguf() {
        let preload = LoraAdapterSpec::new("vision-chat", "org/language-lora");
        let adapter = AdapterOptions {
            lora: vec![preload.clone()],
            ..AdapterOptions::default()
        };
        let format = FormatOptions {
            format: Some(ModelFormat::Gguf),
            quantized_file: Some("model.gguf".to_string()),
            mmproj: Some("mmproj-BF16.gguf".to_string()),
            ..FormatOptions::default()
        };
        let selected = convert_text_model(
            &test_model(),
            &format,
            &adapter,
            &QuantizationOptions::default(),
            &DeviceOptions::default(),
            &MatformerSelection::default(),
            Some(&MultimodalOptions::default()),
        )
        .unwrap();

        match selected {
            ModelSelected::GGUF {
                lora_adapters,
                lora_runtime_config,
                mmproj_filename,
                ..
            } => {
                assert_eq!(lora_adapters, vec![preload]);
                assert_eq!(lora_runtime_config, Some(adapter.lora_runtime_config()));
                assert_eq!(mmproj_filename.as_deref(), Some("mmproj-BF16.gguf"));
            }
            _ => panic!("expected GGUF model"),
        }
    }

    #[test]
    fn enable_lora_routes_empty_native_text_gguf_runtime() {
        let adapter = AdapterOptions {
            enable_lora: true,
            ..AdapterOptions::default()
        };
        let selected = convert_text_model(
            &test_model(),
            &FormatOptions {
                format: Some(ModelFormat::Gguf),
                quantized_file: Some("model.gguf".to_string()),
                ..FormatOptions::default()
            },
            &adapter,
            &QuantizationOptions::default(),
            &DeviceOptions::default(),
            &MatformerSelection::default(),
            None,
        )
        .unwrap();

        match selected {
            ModelSelected::GGUF {
                lora_adapters,
                lora_runtime_config,
                ..
            } => {
                assert!(lora_adapters.is_empty());
                assert_eq!(lora_runtime_config, Some(adapter.lora_runtime_config()));
            }
            _ => panic!("expected GGUF model"),
        }
    }

    #[test]
    fn dynamic_lora_is_rejected_for_ggml() {
        let adapter = AdapterOptions {
            lora: vec![LoraAdapterSpec::new("code", "org/code-lora")],
            ..AdapterOptions::default()
        };
        let format = FormatOptions {
            format: Some(ModelFormat::Ggml),
            quantized_file: Some("model.ggml".to_string()),
            ..FormatOptions::default()
        };
        let error = convert_text_model(
            &test_model(),
            &format,
            &adapter,
            &QuantizationOptions::default(),
            &DeviceOptions::default(),
            &MatformerSelection::default(),
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("raw GGML"));
    }

    #[test]
    fn legacy_lora_keeps_legacy_gguf_selection() {
        let selected = convert_text_model(
            &test_model(),
            &FormatOptions {
                format: Some(ModelFormat::Gguf),
                quantized_file: Some("model.gguf".to_string()),
                ..FormatOptions::default()
            },
            &AdapterOptions {
                legacy_lora: Some("org/legacy-lora".to_string()),
                legacy_lora_order: Some(PathBuf::from("order.json")),
                ..AdapterOptions::default()
            },
            &QuantizationOptions::default(),
            &DeviceOptions::default(),
            &MatformerSelection::default(),
            None,
        )
        .unwrap();

        assert!(matches!(selected, ModelSelected::LoraGGUF { .. }));
    }

    #[test]
    fn xlora_keeps_legacy_gguf_selection() {
        let selected = convert_text_model(
            &test_model(),
            &FormatOptions {
                format: Some(ModelFormat::Gguf),
                quantized_file: Some("model.gguf".to_string()),
                ..FormatOptions::default()
            },
            &AdapterOptions {
                xlora: Some("org/xlora".to_string()),
                xlora_order: Some(PathBuf::from("order.json")),
                ..AdapterOptions::default()
            },
            &QuantizationOptions::default(),
            &DeviceOptions::default(),
            &MatformerSelection::default(),
            None,
        )
        .unwrap();

        assert!(matches!(selected, ModelSelected::XLoraGGUF { .. }));
    }

    #[test]
    fn auto_gguf_preserves_mmproj_files() {
        let mut model = test_model();
        model.tokenizer = Some(PathBuf::from("tokenizer.json"));
        let model_type = ModelType::Auto {
            model,
            format: FormatOptions {
                format: Some(ModelFormat::Gguf),
                quantized_file: Some("model.gguf".to_string()),
                mmproj: Some("mmproj-0.gguf;mmproj-1.gguf".to_string()),
                tok_model_id: Some("org/tokenizer".to_string()),
                ..FormatOptions::default()
            },
            adapter: AdapterOptions::default(),
            quantization: QuantizationOptions {
                isq_organization: Some(IsqOrganization::MoeExpertsOnly),
                imatrix: Some(PathBuf::from("model.imatrix")),
                ..QuantizationOptions::default()
            },
            device: DeviceOptions {
                hf_cache: Some(PathBuf::from("hf-cache")),
                max_seq_len: 8192,
                max_batch_size: 3,
                ..DeviceOptions::default()
            },
            cache: crate::args::CacheOptions::default(),
            multimodal: MultimodalOptions {
                encoder_cache_memory_mb: None,
                max_edge: Some(1280),
                max_num_images: Some(4),
                max_image_length: Some(1152),
            },
        };
        let selected = convert_to_model_selected(
            &model_type,
            &MatformerSelection {
                config_path: Some(PathBuf::from("matformer.csv")),
                slice_name: Some("small".to_string()),
            },
        )
        .unwrap();

        match inference_selection::get_auto_device_map_params(&selected).unwrap() {
            AutoDeviceMapParams::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            } => {
                assert_eq!(max_seq_len, 8192);
                assert_eq!(max_batch_size, 3);
                assert_eq!(max_image_shape, (1152, 1152));
                assert_eq!(max_num_images, 4);
            }
            _ => panic!("expected multimodal device-map parameters"),
        }
        match selected {
            ModelSelected::GGUF {
                tok_model_id,
                quantized_filename,
                tokenizer_json,
                mmproj_filename,
                organization,
                imatrix,
                max_edge,
                hf_cache_path,
                matformer_config_path,
                matformer_slice_name,
                ..
            } => {
                assert_eq!(tok_model_id.as_deref(), Some("org/tokenizer"));
                assert_eq!(quantized_filename, "model.gguf");
                assert_eq!(tokenizer_json.as_deref(), Some("tokenizer.json"));
                assert_eq!(
                    mmproj_filename.as_deref(),
                    Some("mmproj-0.gguf;mmproj-1.gguf")
                );
                assert!(matches!(
                    organization,
                    Some(IsqOrganization::MoeExpertsOnly)
                ));
                assert_eq!(imatrix, Some(PathBuf::from("model.imatrix")));
                assert_eq!(max_edge, Some(1280));
                assert_eq!(hf_cache_path, Some(PathBuf::from("hf-cache")));
                assert_eq!(matformer_config_path, Some(PathBuf::from("matformer.csv")));
                assert_eq!(matformer_slice_name.as_deref(), Some("small"));
            }
            _ => panic!("expected GGUF model"),
        }
    }

    #[test]
    fn multimodal_gguf_preserves_dynamic_lora() {
        let adapter = LoraAdapterSpec::new("code", "org/code-lora");
        let selected = convert_text_model(
            &test_model(),
            &FormatOptions {
                format: Some(ModelFormat::Gguf),
                quantized_file: Some("model.gguf".to_string()),
                mmproj: Some("mmproj.gguf".to_string()),
                ..FormatOptions::default()
            },
            &AdapterOptions {
                lora: vec![adapter.clone()],
                ..AdapterOptions::default()
            },
            &QuantizationOptions::default(),
            &DeviceOptions::default(),
            &MatformerSelection::default(),
            Some(&MultimodalOptions::default()),
        )
        .unwrap();

        match selected {
            ModelSelected::GGUF {
                mmproj_filename,
                lora_adapters,
                lora_runtime_config,
                ..
            } => {
                assert_eq!(mmproj_filename.as_deref(), Some("mmproj.gguf"));
                assert_eq!(lora_adapters, vec![adapter]);
                assert!(lora_runtime_config.is_some());
            }
            _ => panic!("expected GGUF model"),
        }
    }

    #[test]
    fn embedding_rejects_gguf_format() {
        let model_type = ModelType::Embedding {
            model: test_model(),
            format: FormatOptions {
                format: Some(ModelFormat::Gguf),
                quantized_file: Some("model.gguf".to_string()),
                mmproj: Some("mmproj.gguf".to_string()),
                ..FormatOptions::default()
            },
            quantization: QuantizationOptions::default(),
            device: DeviceOptions::default(),
            cache: crate::args::CacheOptions::default(),
        };
        let error =
            convert_to_model_selected(&model_type, &MatformerSelection::default()).unwrap_err();

        assert!(error.to_string().contains("Embedding models"));
    }

    #[test]
    fn text_gguf_keeps_text_device_map() {
        let selected = convert_text_model(
            &test_model(),
            &FormatOptions {
                format: Some(ModelFormat::Gguf),
                quantized_file: Some("model.gguf".to_string()),
                ..FormatOptions::default()
            },
            &AdapterOptions::default(),
            &QuantizationOptions::default(),
            &DeviceOptions::default(),
            &MatformerSelection::default(),
            None,
        )
        .unwrap();

        assert!(matches!(
            inference_selection::get_auto_device_map_params(&selected).unwrap(),
            AutoDeviceMapParams::Text { .. }
        ));
    }

    #[test]
    fn explicit_multimodal_gguf_routes_to_gguf() {
        let model_type = test_multimodal_model(FormatOptions {
            format: Some(ModelFormat::Gguf),
            quantized_file: Some("model.gguf".to_string()),
            mmproj: Some("mmproj.gguf".to_string()),
            ..FormatOptions::default()
        });
        let selected =
            convert_to_model_selected(&model_type, &MatformerSelection::default()).unwrap();

        match selected {
            ModelSelected::GGUF {
                quantized_model_id,
                quantized_filename,
                mmproj_filename,
                ..
            } => {
                assert_eq!(quantized_model_id, "org/base");
                assert_eq!(quantized_filename, "model.gguf");
                assert_eq!(mmproj_filename.as_deref(), Some("mmproj.gguf"));
            }
            _ => panic!("expected GGUF model"),
        }
    }

    #[test]
    fn explicit_multimodal_gguf_requires_model_file() {
        let model_type = test_multimodal_model(FormatOptions {
            format: Some(ModelFormat::Gguf),
            mmproj: Some("mmproj.gguf".to_string()),
            ..FormatOptions::default()
        });
        let error =
            convert_to_model_selected(&model_type, &MatformerSelection::default()).unwrap_err();

        assert!(
            error.to_string().contains("requires a model file"),
            "{error}"
        );
    }

    #[test]
    fn explicit_multimodal_ggml_is_rejected() {
        let model_type = test_multimodal_model(FormatOptions {
            format: Some(ModelFormat::Ggml),
            quantized_file: Some("model.ggml".to_string()),
            ..FormatOptions::default()
        });
        let error =
            convert_to_model_selected(&model_type, &MatformerSelection::default()).unwrap_err();

        assert!(error.to_string().contains("GGML is not supported"));
    }

    fn serve_spec(args: &[&str]) -> EngineSpec {
        use clap::Parser;
        let cli = crate::args::Cli::try_parse_from(args).unwrap();
        let crate::args::Command::Serve {
            model_type,
            default_model,
            server,
            mut runtime,
            agent_options,
            sandbox,
        } = cli.command
        else {
            panic!("not a serve command");
        };
        agent_options.apply_to(&mut runtime);
        let model_type = crate::args::resolve_model_type(model_type, default_model).unwrap();
        engine_spec(EngineSpecInputs {
            model_type: &model_type,
            matformer: &runtime.matformer_selection(),
            model_id: Some("alias".to_string()),
            runtime: &runtime,
            sandbox,
            global: &cli.global,
            max_tool_rounds: server.max_tool_rounds,
            tool_dispatch_url: server.tool_dispatch_url.clone(),
            skills_root: Some(skills_root(&runtime)),
            adapters: adapter_spec_from_env(),
            throughput_logging: true,
        })
        .unwrap()
    }

    #[test]
    fn serve_flags_become_the_engine_spec() {
        let spec = serve_spec(&[
            "inference",
            "--seed",
            "7",
            "--log",
            "requests.log",
            "serve",
            "-m",
            "org/model",
            "--cpu",
            "--max-seqs",
            "4",
            "--paged-attn",
            "off",
            "--pa-block-size",
            "32",
            "--isq",
            "q4k",
            "--mtp",
            "--max-tool-rounds",
            "3",
            "--enable-search",
            "--no-kv-cache",
            "--max-num-batched-tokens",
            "2048",
        ]);
        let rt = &spec.runtime;
        assert_eq!(spec.model_id.as_deref(), Some("alias"));
        assert_eq!(rt.device.as_deref(), Some("cpu"));
        assert_eq!(
            (rt.seed, rt.max_seqs, rt.no_kv_cache),
            (Some(7), Some(4), true)
        );
        assert_eq!(
            (rt.paged_attn, rt.paged_cache.block_size),
            (Some(false), Some(32))
        );
        assert_eq!(rt.isq.as_deref(), Some("q4k"));
        assert!(rt.mtp.as_ref().is_some_and(|mtp| mtp.model.is_none()));
        assert_eq!(rt.max_num_batched_tokens, Some(2048));
        assert_eq!(rt.log.as_deref(), Some(Path::new("requests.log")));
        assert_eq!(rt.token_source.as_deref(), Some("cache"));
        assert_eq!(spec.agentic.max_tool_rounds, Some(3));
        assert!(spec.agentic.search.is_some());
        assert!(spec.skills.root.is_some());
    }

    #[test]
    fn an_mtp_assistant_model_and_the_sandbox_options_carry_over() {
        let spec = serve_spec(&[
            "inference",
            "serve",
            "-m",
            "org/model",
            "--mtp-model",
            "org/draft",
            "--mtp-n-predict",
            "3",
        ]);
        let mtp = spec.runtime.mtp.unwrap();
        assert_eq!(
            (mtp.model.as_deref(), mtp.n_predict),
            (Some("org/draft"), Some(3))
        );
        assert_eq!(spec.agentic.sandbox, inference_core::SandboxMode::Auto);
        assert!(spec.agentic.sandbox_profile.is_none());

        let spec = serve_spec(&[
            "inference",
            "serve",
            "-m",
            "org/model",
            "--sandbox",
            "on",
            "--sandbox-profile",
            "restricted",
            "--sb-max-procs",
            "7",
            "--sandbox-network",
            "loopback",
        ]);
        let agentic = spec.agentic;
        assert_eq!(agentic.sandbox, inference_core::SandboxMode::On);
        assert_eq!(
            agentic.sandbox_profile,
            Some(inference_core::SandboxProfile::Restricted)
        );
        assert_eq!(agentic.sandbox_limits.max_procs, Some(7));
        assert_eq!(
            agentic.sandbox_limits.network,
            Some(inference_core::NetworkMode::Loopback)
        );
        // the engine applies the sandbox, so a tool config carries no policy of its own
        #[cfg(feature = "code-execution")]
        {
            let spec = serve_spec(&["inference", "serve", "-m", "org/model", "--enable-shell"]);
            assert!(spec.agentic.shell.unwrap().sandbox_policy.is_none());
        }
    }
}
