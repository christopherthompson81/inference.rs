//! Interactive mode command implementation

mod chat;
mod interactive;

use interactive::OneshotInput;
pub(crate) use interactive::{InteractiveConfig, interactive_mode};

use anyhow::Result;
use inference_core::{ReasoningEffort, resolve_reasoning_controls};
use tracing::info;

use inference_api::Engine;
use inference_core::initialize_logging;

use super::normalize_requested_adapter;
use super::serve::{
    EngineSpecInputs, apply_agent_mode, apply_quant_resolution, engine_spec, log_agent_runtime,
    validate_agent_options,
};
use crate::args::{AgentCliOptions, GlobalOptions, ModelType, RuntimeOptions, SandboxOptions};

/// Run the model in interactive or one-shot mode
#[allow(clippy::too_many_arguments)]
pub async fn run_interactive(
    mut model_type: ModelType,
    mut runtime: RuntimeOptions,
    agent_options: AgentCliOptions,
    sandbox: SandboxOptions,
    global: GlobalOptions,
    thinking: Option<bool>,
    reasoning_effort: Option<ReasoningEffort>,
    input: Option<String>,
    images: Vec<String>,
    videos: Vec<String>,
    audios: Vec<String>,
    request_adapter: Option<String>,
) -> Result<()> {
    initialize_logging();
    resolve_reasoning_controls(thinking, reasoning_effort)?;

    let request_adapter = normalize_requested_adapter(&model_type, request_adapter.as_deref())?;

    agent_options.apply_to(&mut runtime);
    apply_agent_mode(&mut runtime);
    validate_agent_options(&runtime)?;
    log_agent_runtime(&runtime, None);

    // Convert our clean args to ModelSelected
    let matformer = runtime.matformer_selection();
    apply_quant_resolution(&mut model_type, &global.token_source, &matformer).await?;
    let spec = run_spec(&model_type, &runtime, sandbox, &global)?;
    let engine = Engine::load(spec).await?;
    if let Some(alias) = request_adapter.as_deref() {
        require_adapter(&engine, alias).await?;
    }

    if let Some(text) = input {
        info!("Model loaded, running one-shot mode...");
        #[cfg(feature = "code-execution")]
        let do_code_exec = runtime.enable_code_execution;
        #[cfg(not(feature = "code-execution"))]
        let do_code_exec = false;
        #[cfg(feature = "code-execution")]
        let do_shell = runtime.enable_shell;
        #[cfg(not(feature = "code-execution"))]
        let do_shell = false;

        interactive::oneshot_mode(
            &engine,
            OneshotInput {
                text,
                images,
                videos,
                audios,
            },
            InteractiveConfig {
                do_search: runtime.enable_search,
                do_code_exec,
                do_shell,
                agent_permission: runtime.code_exec_permission.into(),
                enable_thinking: thinking,
                reasoning_effort,
                adapter: request_adapter,
            },
        )
        .await;
    } else {
        #[cfg(feature = "code-execution")]
        let do_code_exec = runtime.enable_code_execution;
        #[cfg(not(feature = "code-execution"))]
        let do_code_exec = false;
        #[cfg(feature = "code-execution")]
        let do_shell = runtime.enable_shell;
        #[cfg(not(feature = "code-execution"))]
        let do_shell = false;

        info!("Model loaded, starting interactive mode...");
        interactive::interactive_mode(
            &engine,
            InteractiveConfig {
                do_search: runtime.enable_search,
                do_code_exec,
                do_shell,
                agent_permission: runtime.code_exec_permission.into(),
                enable_thinking: thinking,
                reasoning_effort,
                adapter: request_adapter,
            },
        )
        .await;
    }

    Ok(())
}

/// Fails unless the engine has a LoRA adapter loaded under `alias`.
pub(crate) async fn require_adapter(engine: &Engine, alias: &str) -> Result<()> {
    let adapters = engine
        .lora_adapters(inference_api::lora_adapters::ListLoraAdaptersQuery::default())
        .await
        .map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        adapters.data.iter().any(|adapter| adapter.id == alias),
        "LoRA adapter alias `{alias}` is not loaded"
    );
    Ok(())
}

/// The engine `run` chats with: no tool loop limits, adapter management or shared skill store, and no throughput
/// lines between turns.
fn run_spec(
    model_type: &ModelType,
    runtime: &RuntimeOptions,
    sandbox: SandboxOptions,
    global: &GlobalOptions,
) -> Result<inference_api::EngineSpec> {
    engine_spec(EngineSpecInputs {
        model_type,
        matformer: &runtime.matformer_selection(),
        model_id: None,
        runtime,
        sandbox,
        global,
        max_tool_rounds: None,
        tool_dispatch_url: None,
        skills_root: None,
        adapters: Default::default(),
        throughput_logging: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_keeps_its_turns_free_of_throughput_lines() {
        use clap::Parser;
        let cli =
            crate::args::Cli::try_parse_from(["inference", "run", "-m", "org/model", "--cpu"])
                .unwrap();
        let crate::args::Command::Run {
            model_type,
            default_model,
            runtime,
            sandbox,
            ..
        } = cli.command
        else {
            panic!("not a run command");
        };
        let model_type = crate::args::resolve_model_type(model_type, default_model).unwrap();
        let spec = run_spec(&model_type, &runtime, sandbox, &cli.global).unwrap();
        assert_eq!(spec.runtime.throughput_logging, Some(false));
        assert!(spec.skills.root.is_none() && !spec.adapters.runtime_updates);
        assert!(spec.agentic.max_tool_rounds.is_none());
        assert_eq!(spec.runtime.device.as_deref(), Some("cpu"));
    }
}
