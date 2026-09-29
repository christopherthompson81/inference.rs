//! Run inference-cli from a full TOML configuration.

use anyhow::Result;
use tracing::info;

use inference_api::{
    Engine, EngineSpec,
    engine::{AdapterSpec, ModelSpec, PagedCacheSpec, RuntimeSpec, SkillsSpec},
};
use inference_core::initialize_logging;
use inference_server_core::metrics::install_prometheus_recorder;

use crate::args::{
    GlobalOptions, MatformerSelection, PagedAttentionOptions, RuntimeOptions, SandboxOptions,
};
use crate::commands::run::{InteractiveConfig, interactive_mode};
use crate::commands::serve::{
    AgenticInputs, adapter_spec_from_env, agentic_spec, apply_agent_mode,
    convert_to_model_selected, log_agent_runtime, runtime_options_spec, serve_engine, skills_root,
    validate_agent_options,
};
use crate::config::{CliConfig, load_cli_config};

/// Execute the CLI using a TOML configuration file.
pub async fn run_from_config(path: std::path::PathBuf) -> Result<()> {
    initialize_logging();

    let config = load_cli_config(&path)?;

    match config {
        CliConfig::Serve(cfg) => run_serve_config(cfg).await,
        CliConfig::Run(cfg) => run_run_config(cfg).await,
    }
}

async fn run_serve_config(cfg: crate::config::ServeConfig) -> Result<()> {
    let crate::config::ServeConfig {
        global,
        mut runtime,
        server,
        paged_attn,
        sandbox,
        models,
        default_model_id,
    } = cfg;

    if server.observability_config().metrics {
        install_prometheus_recorder();
    }
    let global = global.to_global_options()?;
    apply_agent_mode(&mut runtime);
    validate_agent_options(&runtime)?;
    log_agent_runtime(&runtime, server.max_tool_rounds);

    let spec = config_spec(ConfigSpecInputs {
        models: &models,
        default_model_id,
        runtime: &runtime,
        paged_attn,
        sandbox,
        global: &global,
        max_tool_rounds: server.max_tool_rounds,
        tool_dispatch_url: server.tool_dispatch_url.clone(),
        skills_root: Some(skills_root(&runtime)),
        adapters: adapter_spec_from_env(),
        throughput_logging: true,
    })
    .await?;
    serve_engine(spec, &server, &runtime).await
}

async fn run_run_config(cfg: crate::config::RunConfig) -> Result<()> {
    let crate::config::RunConfig {
        global,
        mut runtime,
        paged_attn,
        sandbox,
        models,
        thinking,
        reasoning_effort,
        adapter,
    } = cfg;

    inference_core::resolve_reasoning_controls(thinking, reasoning_effort)?;

    let global = global.to_global_options()?;
    apply_agent_mode(&mut runtime);
    validate_agent_options(&runtime)?;
    log_agent_runtime(&runtime, None);
    let spec = config_spec(ConfigSpecInputs {
        models: &models,
        default_model_id: None,
        runtime: &runtime,
        paged_attn,
        sandbox,
        global: &global,
        max_tool_rounds: None,
        tool_dispatch_url: None,
        skills_root: None,
        adapters: Default::default(),
        throughput_logging: false,
    })
    .await?;
    let engine = Engine::load(spec).await?;
    let inference = engine.state().clone();
    if let Some(alias) = adapter.as_deref() {
        let adapters = inference.list_lora_adapters(None).await?;
        if !adapters.iter().any(|loaded| loaded.alias == alias) {
            anyhow::bail!("LoRA adapter alias `{alias}` is not loaded");
        }
    }

    #[cfg(feature = "code-execution")]
    let do_code_exec = runtime.enable_code_execution;
    #[cfg(not(feature = "code-execution"))]
    let do_code_exec = false;
    #[cfg(feature = "code-execution")]
    let do_shell = runtime.enable_shell;
    #[cfg(not(feature = "code-execution"))]
    let do_shell = false;

    info!("Model(s) loaded, starting interactive mode...");

    interactive_mode(
        inference.clone(),
        InteractiveConfig {
            do_search: runtime.enable_search,
            do_code_exec,
            do_shell,
            agent_permission: runtime.code_exec_permission.into(),
            enable_thinking: thinking,
            reasoning_effort,
            adapter,
        },
    )
    .await;

    Ok(())
}

async fn build_model_specs(
    models: &[crate::config::ModelEntry],
    runtime: &RuntimeOptions,
    token_source: &inference_core::TokenSource,
) -> Result<(Vec<ModelSpec>, bool)> {
    let mut cpu_setting: Option<bool> = None;
    for entry in models {
        if let Some(cpu) = entry.device.cpu {
            match cpu_setting {
                None => cpu_setting = Some(cpu),
                Some(existing) if existing != cpu => {
                    anyhow::bail!(
                        "cpu must be consistent across all models (found both true and false)"
                    );
                }
                _ => {}
            }
        }
    }
    let cpu = cpu_setting.unwrap_or(false);

    let mut specs = Vec::new();
    for entry in models {
        let mut model_type = entry.to_model_type(cpu);
        let matformer = MatformerSelection {
            config_path: entry
                .matformer_config_path
                .clone()
                .or_else(|| runtime.matformer_config_path.clone()),
            slice_name: entry
                .matformer_slice_name
                .clone()
                .or_else(|| runtime.matformer_slice_name.clone()),
        };
        crate::commands::serve::apply_quant_resolution(&mut model_type, token_source, &matformer)
            .await?;
        let model = convert_to_model_selected(&model_type, &matformer)?;
        let resolved_loader_id = crate::commands::serve::model_id_of(&model_type);
        specs.push(ModelSpec {
            model,
            // An entry whose id resolved to another loader id keeps its own as the id requests use.
            model_id: (resolved_loader_id != entry.model_id).then(|| entry.model_id.clone()),
            chat_template: entry
                .chat_template
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned()),
            jinja_explicit: entry
                .jinja_explicit
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned()),
            max_model_len: entry.max_model_len,
            hf_config_overrides: entry.hf_overrides.clone(),
            device_layers: entry.device.device_layers.clone(),
            isq: crate::commands::serve::extract_isq_setting(&model_type),
            encoder_cache_memory_bytes: crate::commands::serve::extract_encoder_cache_memory_bytes(
                &model_type,
            )?,
        });
    }
    Ok((specs, cpu))
}

/// What a TOML config describes, before its models are resolved.
struct ConfigSpecInputs<'a> {
    models: &'a [crate::config::ModelEntry],
    default_model_id: Option<String>,
    runtime: &'a RuntimeOptions,
    paged_attn: PagedAttentionOptions,
    sandbox: SandboxOptions,
    global: &'a GlobalOptions,
    max_tool_rounds: Option<usize>,
    tool_dispatch_url: Option<String>,
    skills_root: Option<std::path::PathBuf>,
    adapters: AdapterSpec,
    throughput_logging: bool,
}

/// The engine a TOML config loads: every model it lists, sharing its runtime settings.
async fn config_spec(inputs: ConfigSpecInputs<'_>) -> Result<EngineSpec> {
    let (models, cpu) =
        build_model_specs(inputs.models, inputs.runtime, &inputs.global.token_source).await?;
    let (paged_attn, memory_mb, memory_fraction, context_len, block_size, cache_type) =
        inputs.paged_attn.into_builder_flags();
    let base = RuntimeSpec {
        device: cpu.then(|| "cpu".to_string()),
        seed: inputs.global.seed,
        token_source: Some(inputs.global.token_source.to_string()),
        log: inputs.global.log.clone(),
        paged_attn,
        paged_cache: PagedCacheSpec {
            context_len,
            memory_mb,
            memory_fraction,
            block_size,
            cache_type,
        },
        ..Default::default()
    };
    Ok(EngineSpec {
        models,
        default_model_id: inputs.default_model_id,
        runtime: runtime_options_spec(inputs.runtime, base, inputs.throughput_logging),
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

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[tokio::test]
    async fn from_config_infers_gguf_for_an_exact_file() {
        let root = std::env::temp_dir().join(format!("inference-gguf-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("model.gguf"), []).unwrap();
        fs::write(root.join("mmproj-BF16.gguf"), []).unwrap();
        let input = format!(
            r#"
command = "run"

[[models]]
model_id = "{}"

[models.format]
quantized_file = "model.gguf"
mmproj = "mmproj-BF16.gguf"
"#,
            root.display()
        );
        let config: CliConfig = toml::from_str(&input).unwrap();
        let CliConfig::Run(config) = config else {
            unreachable!()
        };

        let (models, cpu) = build_model_specs(
            &config.models,
            &config.runtime,
            &inference_core::TokenSource::None,
        )
        .await
        .unwrap();
        assert_eq!(models.len(), 1);
        assert!(!cpu);

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn from_config_accepts_dynamic_lora_for_text_gguf() {
        let root = std::env::temp_dir().join(format!("inference-gguf-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("model-Q4_K_M.gguf"), []).unwrap();
        let input = format!(
            r#"
command = "serve"

[[models]]
model_id = "{}"

[models.format]
quantized_file = "model-Q4_K_M.gguf"

[models.adapter]
lora = [
  {{ alias = "code", source = "org/code-lora" }},
]
"#,
            root.display()
        );
        let config: CliConfig = toml::from_str(&input).unwrap();
        let CliConfig::Serve(config) = config else {
            unreachable!()
        };
        assert!(config.models[0].adapter.dynamic_lora_enabled());

        let (models, cpu) = build_model_specs(
            &config.models,
            &config.runtime,
            &inference_core::TokenSource::None,
        )
        .await
        .unwrap();
        assert_eq!(models.len(), 1);
        assert!(!cpu);

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn a_config_serves_each_of_its_models_with_their_own_settings() {
        let dirs = [0, 1].map(|_| {
            let dir =
                std::env::temp_dir().join(format!("inference-config-{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(&dir).unwrap();
            dir
        });
        let input = format!(
            r#"
command = "serve"
default_model_id = "second"

[paged_attn]
mode = "off"
block_size = 32

[[models]]
model_id = "{}"
chat_template = "first.jinja"

[[models]]
model_id = "{}"
"#,
            dirs[0].display(),
            dirs[1].display(),
        );
        let CliConfig::Serve(config) = toml::from_str(&input).unwrap() else {
            unreachable!()
        };
        let global = config.global.to_global_options().unwrap();
        let spec = config_spec(ConfigSpecInputs {
            models: &config.models,
            default_model_id: Some("second".to_string()),
            runtime: &config.runtime,
            paged_attn: config.paged_attn,
            sandbox: config.sandbox,
            global: &global,
            max_tool_rounds: None,
            tool_dispatch_url: None,
            skills_root: None,
            adapters: Default::default(),
            throughput_logging: true,
        })
        .await
        .unwrap();
        assert!(spec.model.is_none());
        assert_eq!(spec.models.len(), 2);
        assert_eq!(spec.models[0].chat_template.as_deref(), Some("first.jinja"));
        assert!(spec.models[1].chat_template.is_none());
        assert_eq!(spec.default_model_id.as_deref(), Some("second"));
        assert_eq!(
            (spec.runtime.paged_attn, spec.runtime.paged_cache.block_size),
            (Some(false), Some(32))
        );
        for dir in dirs {
            fs::remove_dir_all(dir).unwrap();
        }
    }
}
