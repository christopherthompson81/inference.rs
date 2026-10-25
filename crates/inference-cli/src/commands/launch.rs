//! `inference launch`: a coding agent (Claude Code, opencode, Codex) run against an inference server, its provider set
//! through the agent's environment or command line so none of its configuration files are written.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use super::agent::read_api_key;
use crate::args::LaunchAgent;
use crate::config::{CliConfig, load_cli_config};

const API_KEY_ENV: &str = "INFERENCE_API_KEY";
const DEFAULT_URL: &str = "http://127.0.0.1:1234";
// sent when the server needs no key, as the agents insist on one
const PLACEHOLDER_KEY: &str = "inference-local";
// a server started here loads its models before it answers
const SERVER_READY_TIMEOUT: Duration = Duration::from_secs(900);
const SERVER_POLL: Duration = Duration::from_millis(500);
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const SERVER_LOG: &str = "launch-server.log";
// opencode's provider id, the most tokens it asks for in one reply, and the share of the window a reply may take (the
// rest is the prompt's)
const OPENCODE_PROVIDER: &str = "inference";
const OPENCODE_OUTPUT_TOKENS: u64 = 32_768;
const OPENCODE_OUTPUT_SHARE: u64 = 4;
// the id every server lists for its default model, used when that model's own id is a path
const DEFAULT_MODEL_ALIAS: &str = "default";
// Codex's provider id, and the variable its provider reads the key from
const CODEX_PROVIDER: &str = "inference";
const CODEX_KEY_ENV: &str = "INFERENCE_RS_API_KEY";
// Codex waits this long for the next streamed event; a long prefill streams nothing until its first token
const CODEX_STREAM_IDLE_MS: u64 = 300_000;

/// What `inference launch` was asked for.
pub struct LaunchRequest {
    pub agent: LaunchAgent,
    pub url: Option<String>,
    pub config: Option<PathBuf>,
    pub api_key_file: Option<PathBuf>,
    pub model: Option<String>,
    pub agent_args: Vec<String>,
}

/// The served model an agent is pointed at.
struct ServedModel {
    id: String,
    context: Option<u64>,
}

/// Runs the agent to completion and returns its exit code; a server this started is stopped afterwards.
pub async fn run_launch(request: LaunchRequest) -> Result<i32> {
    let config = request
        .config
        .as_deref()
        .map(|path| match load_cli_config(path)? {
            CliConfig::Serve(cfg) => Ok(*cfg),
            CliConfig::Run(_) => bail!("{} configures `run`, not a server", path.display()),
        })
        .transpose()?;
    let base = match (&request.url, &config) {
        (Some(url), _) => url.trim_end_matches('/').to_string(),
        (None, Some(cfg)) => format!(
            "http://{}:{}",
            local_host(&cfg.server.host),
            cfg.server.port
        ),
        (None, None) => DEFAULT_URL.to_string(),
    };
    let key = match &request.api_key_file {
        Some(path) => Some(key_from_file(path)?),
        None => match std::env::var(API_KEY_ENV)
            .ok()
            .filter(|key| !key.is_empty())
        {
            Some(key) => Some(key),
            None => config
                .as_ref()
                .and_then(|cfg| cfg.server.api_keys_file.as_deref())
                .map(key_from_file)
                .transpose()?,
        },
    };
    inference_mcp::tls::install_provider();
    let client = reqwest::Client::builder().build()?;
    let mut server = None;
    if models(&client, &base, key.as_deref()).await.is_err() {
        let Some(path) = &request.config else {
            bail!("no server answers at {base}; start one, or pass --config to start it here");
        };
        server = Some(start_server(path)?);
    }
    let result = async {
        if let Some(server) = server.as_mut() {
            wait_ready(&client, &base, key.as_deref(), server).await?;
        }
        let model = pick_model(
            &models(&client, &base, key.as_deref()).await?,
            request.model.as_deref(),
        )?;
        let status = run_agent(&request, &base, key.as_deref(), &model)?;
        Ok::<_, anyhow::Error>(status.code().unwrap_or(1))
    }
    .await;
    if let Some(server) = server {
        stop_server(server);
    }
    result
}

// a server bound to every interface is reached on loopback
fn local_host(host: &str) -> &str {
    match host {
        "0.0.0.0" | "::" | "" => "127.0.0.1",
        host => host,
    }
}

fn key_from_file(path: &Path) -> Result<String> {
    read_api_key(
        &std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?,
    )
}

async fn models(client: &reqwest::Client, base: &str, key: Option<&str>) -> Result<Value> {
    let mut request = client
        .get(format!("{base}/v1/models"))
        .timeout(PROBE_TIMEOUT);
    if let Some(key) = key {
        request = request.bearer_auth(key);
    }
    let response = request.send().await?;
    if !response.status().is_success() {
        bail!("{base}/v1/models answered {}", response.status());
    }
    Ok(response.json().await?)
}

// `--model`, else the server's default model (by its alias when its id is a path, which agents read as provider/model),
// else its first
fn pick_model(models: &Value, wanted: Option<&str>) -> Result<ServedModel> {
    let entries = models["data"].as_array().cloned().unwrap_or_default();
    let entry = match wanted {
        Some(id) => entries
            .iter()
            .find(|m| m["id"] == id)
            .with_context(|| format!("the server has no model `{id}`"))?,
        None => entries
            .iter()
            .find(|m| m["default"] == true)
            .or_else(|| entries.first())
            .context("the server lists no models")?,
    };
    let id = entry["id"].as_str().unwrap_or_default();
    let alias = entries.iter().any(|m| m["id"] == DEFAULT_MODEL_ALIAS);
    let id = if wanted.is_none() && id.contains('/') && alias {
        DEFAULT_MODEL_ALIAS
    } else {
        id
    };
    Ok(ServedModel {
        id: id.to_string(),
        context: entry["max_model_len"].as_u64(),
    })
}

fn start_server(config: &Path) -> Result<Child> {
    let log_dir = dirs::cache_dir()
        .context("no cache directory for the server log")?
        .join("inference-rs");
    std::fs::create_dir_all(&log_dir)?;
    let log_path = log_dir.join(SERVER_LOG);
    let log = std::fs::File::create(&log_path)?;
    eprintln!(
        "Starting the server from {} (log: {})",
        config.display(),
        log_path.display()
    );
    Ok(Command::new(std::env::current_exe()?)
        .arg("from-config")
        .arg("-f")
        .arg(config)
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log)
        .spawn()?)
}

async fn wait_ready(
    client: &reqwest::Client,
    base: &str,
    key: Option<&str>,
    server: &mut Child,
) -> Result<()> {
    let started = Instant::now();
    while started.elapsed() < SERVER_READY_TIMEOUT {
        if let Some(status) = server.try_wait()? {
            bail!("the server exited ({status}) before it answered; see its log");
        }
        if models(client, base, key).await.is_ok() {
            return Ok(());
        }
        tokio::time::sleep(SERVER_POLL).await;
    }
    bail!("the server did not answer at {base} within {SERVER_READY_TIMEOUT:?}")
}

// SIGTERM, so the server shuts down as it does on Ctrl-C
fn stop_server(mut server: Child) {
    let stopped = Command::new("kill")
        .arg(server.id().to_string())
        .status()
        .is_ok_and(|status| status.success());
    if !stopped {
        let _ = server.kill();
    }
    let _ = server.wait();
}

fn run_agent(
    request: &LaunchRequest,
    base: &str,
    key: Option<&str>,
    model: &ServedModel,
) -> Result<ExitStatus> {
    let key = key.unwrap_or(PLACEHOLDER_KEY);
    let mut command = match request.agent {
        LaunchAgent::Claude => {
            let mut command = Command::new("claude");
            command
                .env_remove("ANTHROPIC_API_KEY")
                .env("ANTHROPIC_BASE_URL", base)
                .env("ANTHROPIC_AUTH_TOKEN", key)
                .env("ANTHROPIC_MODEL", &model.id)
                .env("ANTHROPIC_DEFAULT_OPUS_MODEL", &model.id)
                .env("ANTHROPIC_DEFAULT_SONNET_MODEL", &model.id)
                .env("ANTHROPIC_DEFAULT_HAIKU_MODEL", &model.id)
                .env("ANTHROPIC_DEFAULT_FABLE_MODEL", &model.id)
                .env("CLAUDE_CODE_SUBAGENT_MODEL", &model.id)
                .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1");
            // an id Claude Code does not know gets a 200K window unless told the server's
            if let Some(context) = model.context {
                command.env("CLAUDE_CODE_MAX_CONTEXT_TOKENS", context.to_string());
            }
            command
        }
        LaunchAgent::Opencode => {
            let mut command = Command::new("opencode");
            command.env(
                "OPENCODE_CONFIG_CONTENT",
                opencode_config(base, key, model).to_string(),
            );
            command
        }
        LaunchAgent::Codex => {
            let mut command = Command::new("codex");
            command
                .env(CODEX_KEY_ENV, key)
                .args(
                    codex_overrides(base, model)
                        .iter()
                        .flat_map(|o| ["-c", o.as_str()]),
                )
                .args(["-m", &model.id]);
            command
        }
    };
    let name = request.agent.program();
    eprintln!("Running {name} against {base} with model `{}`", model.id);
    command
        .args(&request.agent_args)
        .status()
        .with_context(|| format!("running `{name}`; is it installed and on PATH?"))
}

// Codex's provider as `-c` overrides (TOML values): the Responses API under /v1, as the coding-agents guide sets it
fn codex_overrides(base: &str, model: &ServedModel) -> Vec<String> {
    let provider = format!("model_providers.{CODEX_PROVIDER}");
    let mut overrides = vec![
        format!("{provider}.name=\"inference.rs\""),
        format!("{provider}.base_url=\"{base}/v1\""),
        format!("{provider}.wire_api=\"responses\""),
        format!("{provider}.env_key=\"{CODEX_KEY_ENV}\""),
        format!("{provider}.request_max_retries=1"),
        format!("{provider}.stream_max_retries=0"),
        format!("{provider}.stream_idle_timeout_ms={CODEX_STREAM_IDLE_MS}"),
        format!("model_provider=\"{CODEX_PROVIDER}\""),
        "model_reasoning_summary=\"none\"".to_string(),
        "model_supports_reasoning_summaries=false".to_string(),
    ];
    if let Some(context) = model.context {
        overrides.push(format!("model_context_window={context}"));
    }
    overrides
}

fn opencode_config(base: &str, key: &str, model: &ServedModel) -> Value {
    let mut entry = json!({ "name": model.id });
    if let Some(context) = model.context {
        entry["limit"] = json!({ "context": context, "output": OPENCODE_OUTPUT_TOKENS.min(context / OPENCODE_OUTPUT_SHARE) });
    }
    json!({
        "$schema": "https://opencode.ai/config.json",
        "model": format!("{OPENCODE_PROVIDER}/{}", model.id),
        "provider": {
            OPENCODE_PROVIDER: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "inference.rs",
                "options": { "baseURL": format!("{base}/v1"), "apiKey": key, "timeout": false },
                "models": { model.id.clone(): entry },
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn listing() -> Value {
        json!({ "data": [
            { "id": "default", "max_model_len": 4096 },
            { "id": "chat", "max_model_len": 262_144, "default": true },
        ]})
    }

    #[test]
    fn the_servers_default_model_is_picked_unless_one_is_named() -> Result<()> {
        let picked = pick_model(&listing(), None)?;
        assert_eq!(
            (picked.id.as_str(), picked.context),
            ("chat", Some(262_144))
        );
        assert_eq!(pick_model(&listing(), Some("default"))?.id, "default");
        assert!(pick_model(&listing(), Some("missing")).is_err());
        let by_path = json!({ "data": [
            { "id": "default", "max_model_len": 2048 },
            { "id": "/srv/models", "max_model_len": 2048, "default": true },
        ]});
        assert_eq!(pick_model(&by_path, None)?.id, "default");
        Ok(())
    }

    #[test]
    fn opencode_gets_the_server_as_its_provider() {
        let model = ServedModel {
            id: "chat".into(),
            context: Some(8192),
        };
        let config = opencode_config("http://127.0.0.1:1234", "k", &model);
        assert_eq!(config["model"], "inference/chat");
        let provider = &config["provider"][OPENCODE_PROVIDER];
        assert_eq!(provider["options"]["baseURL"], "http://127.0.0.1:1234/v1");
        assert_eq!(provider["models"]["chat"]["limit"]["output"], 2048);
    }

    #[test]
    fn codex_gets_the_responses_api_as_its_provider() {
        let model = ServedModel {
            id: "chat".into(),
            context: Some(8192),
        };
        let overrides = codex_overrides("http://127.0.0.1:1234", &model);
        assert!(overrides.contains(
            &"model_providers.inference.base_url=\"http://127.0.0.1:1234/v1\"".to_string()
        ));
        assert!(
            overrides.contains(&"model_providers.inference.wire_api=\"responses\"".to_string())
        );
        assert!(overrides.contains(&"model_context_window=8192".to_string()));
    }

    #[test]
    fn a_server_on_every_interface_is_reached_on_loopback() {
        assert_eq!(local_host("0.0.0.0"), "127.0.0.1");
        assert_eq!(local_host("10.0.0.5"), "10.0.0.5");
    }
}
