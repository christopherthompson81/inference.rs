//! Chat turns through the engine API: requests, streamed output, agent panels and approvals, and stats.

use std::{
    io::{self, Write},
    sync::{LazyLock, Mutex},
    time::{Duration, Instant},
};

use inference_api::{
    Engine,
    agentic::{ApprovalDecision, ApprovalDecisionRequest},
    engine_chat::{
        AgentToolKind, AgenticToolApproval, AgenticToolCallData, AgenticToolCallPhase,
        BlockDenoisingProgress, ChatStreamEvent, RequestCancellation, Usage,
    },
    media_source::{MediaAttachment, MediaAttachments, MediaSourcePolicy, load_media_source},
    openai::{ChatCompletionRequest, GenerationDefaults, ModelObject},
};
use inference_core::{AgentPermission, ReasoningEffort, files::File};
use serde_json::{Value, json};
use tracing::info;

const AGENTIC_PANEL_WIDTH: usize = 50;
const DENOISING_BAR_WIDTH: usize = 28;
const FALLBACK_TEMPERATURE: f64 = 0.8;
const FALLBACK_TOP_K: usize = 40;
const FALLBACK_TOP_P: f64 = 0.95;
const FALLBACK_MIN_P: f64 = 0.05;
const TEMPERATURE_CMD: &str = "/temperature";
const TOPK_CMD: &str = "/topk";
const TOPP_CMD: &str = "/topp";
const MEDIA_SCHEME: &str = "media://";
pub(super) const IMAGE_PART: &str = "image_url";
pub(super) const AUDIO_PART: &str = "audio_url";
pub(super) const VIDEO_PART: &str = "video_url";
const GRAY: &str = "\x1b[90m";
const RESET: &str = "\x1b[0m";

// What Ctrl-C cancels: nothing between turns (it exits), or the turn being prepared or streamed.
enum InFlight {
    Idle,
    Preparing { canceled: bool },
    Streaming(RequestCancellation),
}

static IN_FLIGHT: LazyLock<Mutex<InFlight>> = LazyLock::new(|| Mutex::new(InFlight::Idle));

/// Cancels the turn in flight, which still ends with its stats; returns false when no turn is running.
pub(super) fn cancel_in_flight() -> bool {
    match &mut *IN_FLIGHT.lock().unwrap() {
        InFlight::Idle => false,
        InFlight::Preparing { canceled } => {
            *canceled = true;
            true
        }
        InFlight::Streaming(cancellation) => {
            cancellation.cancel();
            true
        }
    }
}

// Back to idle however the turn ends.
struct TurnGuard;

impl TurnGuard {
    fn preparing() -> Self {
        *IN_FLIGHT.lock().unwrap() = InFlight::Preparing { canceled: false };
        Self
    }

    fn streaming(&self, cancellation: Option<RequestCancellation>) {
        let Some(cancellation) = cancellation else {
            return;
        };
        let mut in_flight = IN_FLIGHT.lock().unwrap();
        if matches!(*in_flight, InFlight::Preparing { canceled: true }) {
            cancellation.cancel();
        }
        *in_flight = InFlight::Streaming(cancellation);
    }
}

impl Drop for TurnGuard {
    fn drop(&mut self) {
        *IN_FLIGHT.lock().unwrap() = InFlight::Idle;
    }
}

/// The loaded default model, as the engine describes it.
pub(super) fn default_model(engine: &Engine) -> anyhow::Result<ModelObject> {
    engine
        .models()
        .map_err(anyhow::Error::msg)?
        .data
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no model is loaded"))
}

/// What every turn of a session asks for besides its messages.
#[derive(Clone)]
pub(super) struct ChatOptions {
    pub do_search: bool,
    pub do_code_exec: bool,
    pub do_shell: bool,
    pub agent_permission: AgentPermission,
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub adapter: Option<String>,
    /// Shared by the session's turns so code execution and the shell keep their state.
    pub session_id: Option<String>,
}

/// The sampling a session sends: the model's generation defaults, edited by `/temperature`, `/topk` and `/topp`.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Sampling {
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f32>,
    pub max_tokens: Option<usize>,
}

impl Sampling {
    pub(super) fn for_model(defaults: Option<&GenerationDefaults>) -> Self {
        let Some(defaults) = defaults else {
            return Self::fallback();
        };
        let mut sampling = Self {
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            repetition_penalty: defaults.repetition_penalty,
            max_tokens: defaults.max_new_tokens,
        };
        if defaults.do_sample == Some(false) {
            sampling.top_k = Some(1);
        } else {
            sampling.temperature = defaults.temperature;
            sampling.top_k = defaults.top_k.filter(|&top_k| top_k != 0);
            sampling.top_p = defaults.top_p;
            sampling.min_p = defaults.min_p;
        }
        sampling
    }

    // A model without generation defaults gets broad sampling rather than the engine's greedy default.
    fn fallback() -> Self {
        Self {
            temperature: Some(FALLBACK_TEMPERATURE),
            top_k: Some(FALLBACK_TOP_K),
            top_p: Some(FALLBACK_TOP_P),
            min_p: Some(FALLBACK_MIN_P),
            repetition_penalty: None,
            max_tokens: None,
        }
    }

    pub(super) fn describe(&self) -> String {
        fn or_off<T: std::fmt::Display>(value: &Option<T>) -> String {
            value
                .as_ref()
                .map_or_else(|| "off".to_string(), T::to_string)
        }
        let mut parts = vec![
            format!("temp={}", or_off(&self.temperature)),
            format!("top_k={}", or_off(&self.top_k)),
            format!("top_p={}", or_off(&self.top_p)),
            format!("min_p={}", or_off(&self.min_p)),
        ];
        if self.repetition_penalty.is_some() {
            parts.push(format!("rep_pen={}", or_off(&self.repetition_penalty)));
        }
        parts.join(", ")
    }

    /// Applies a `/temperature`, `/topk` or `/topp` command; false when `prompt` is none of them.
    pub(super) fn apply_command(&mut self, prompt: &str) -> bool {
        let prompt = prompt.trim();
        let Some((command, value)) = [TEMPERATURE_CMD, TOPK_CMD, TOPP_CMD]
            .into_iter()
            .find(|command| prompt.starts_with(command))
            .map(|command| (command, prompt[command.len()..].trim()))
        else {
            return false;
        };
        match command {
            TEMPERATURE_CMD => match value.parse::<f64>() {
                Ok(v) if (0.0..=2.0).contains(&v) => {
                    self.temperature = Some(v);
                    info!("Set temperature to {v}");
                }
                Ok(_) => println!("Error: temperature must be in [0.0, 2.0]"),
                Err(_) => println!("Error: format is `{TEMPERATURE_CMD} <float>`"),
            },
            TOPK_CMD => match value.parse::<usize>() {
                Ok(v) if v > 0 => {
                    self.top_k = Some(v);
                    info!("Set top-k to {v}");
                }
                Ok(_) => println!("Error: top-k must be a positive integer"),
                Err(_) => println!("Error: format is `{TOPK_CMD} <int>`"),
            },
            _ => match value.parse::<f64>() {
                Ok(v) if v > 0.0 && v <= 1.0 => {
                    self.top_p = Some(v);
                    info!("Set top-p to {v}");
                }
                Ok(_) => println!("Error: top-p must be in (0.0, 1.0]"),
                Err(_) => println!("Error: format is `{TOPP_CMD} <float>`"),
            },
        }
        true
    }
}

/// The media a session has attached, in `media://<index>` order; every turn resends them all.
#[derive(Default)]
pub(super) struct SessionMedia {
    attachments: Vec<MediaAttachment>,
    references: Vec<String>,
    shared: MediaAttachments,
}

impl SessionMedia {
    /// Loads a path, `file://` or http(s) URL once, as a local caller may (private networks included), and
    /// returns the `media://` source that names it.
    pub(super) async fn source_for(
        &mut self,
        reference: &str,
        part: &str,
    ) -> anyhow::Result<String> {
        let kind = part.trim_end_matches("_url");
        let loaded = load_media_source(reference, MediaSourcePolicy::Local, kind)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to load {kind} `{reference}`: {e}"))?;
        self.attachments.push(MediaAttachment {
            bytes: loaded.bytes,
            mime_type: loaded.mime_type,
        });
        self.references.push(reference.to_string());
        self.shared = MediaAttachments::new(self.attachments.clone());
        Ok(format!("{MEDIA_SCHEME}{}", self.attachments.len() - 1))
    }

    pub(super) fn len(&self) -> usize {
        self.attachments.len()
    }

    /// Drops the media attached after the first `len`, e.g. by a turn that failed.
    pub(super) fn truncate(&mut self, len: usize) {
        if len < self.attachments.len() {
            self.attachments.truncate(len);
            self.references.truncate(len);
            self.shared = MediaAttachments::new(self.attachments.clone());
        }
    }

    pub(super) fn clear(&mut self) {
        self.truncate(0);
    }

    /// `message` with each `media://<index>` replaced by what the user named.
    pub(super) fn describe(&self, message: &str) -> String {
        self.references.iter().enumerate().rev().fold(
            message.to_string(),
            |message, (index, reference)| {
                message.replace(&format!("{MEDIA_SCHEME}{index}"), reference)
            },
        )
    }
}

pub(super) fn text_message(role: &str, text: &str) -> Value {
    json!({"role": role, "content": text})
}

/// A user message of media parts (`image_url`, `audio_url` or `video_url` with its source) followed by text.
pub(super) fn media_message(parts: Vec<(&str, String)>, text: &str) -> Value {
    let mut content = parts
        .into_iter()
        .map(|(kind, url)| {
            let mut part = serde_json::Map::new();
            part.insert("type".to_string(), json!(kind));
            part.insert(kind.to_string(), json!({"url": url}));
            Value::Object(part)
        })
        .collect::<Vec<_>>();
    content.push(json!({"type": "text", "text": text}));
    json!({"role": "user", "content": content})
}

fn chat_request(
    messages: &[Value],
    options: &ChatOptions,
    sampling: &Sampling,
) -> anyhow::Result<ChatCompletionRequest> {
    let mut request = json!({
        "messages": messages,
        "temperature": sampling.temperature,
        "top_k": sampling.top_k,
        "top_p": sampling.top_p,
        "min_p": sampling.min_p,
        "repetition_penalty": sampling.repetition_penalty,
        "max_tokens": sampling.max_tokens,
        "agent_permission": options.agent_permission,
        "enable_thinking": options.enable_thinking,
        "reasoning_effort": options.reasoning_effort.map(ReasoningEffort::as_str),
        "adapter": options.adapter,
        "session_id": options.session_id,
        "enable_shell": options.do_shell,
    });
    if options.do_search {
        request["web_search_options"] = json!({});
    }
    if options.do_code_exec {
        request["tools"] = json!([{"type": "code_interpreter", "container": {"type": "auto"}}]);
    }
    Ok(serde_json::from_value(request)?)
}

/// One streamed turn: the assistant's reply for the history, and what the stats need.
pub(super) struct Turn {
    pub message: Value,
    pub time_to_first_token: Option<Duration>,
    pub usage: Option<Usage>,
}

/// Streams one chat turn, printing its text, reasoning, tool panels and approvals as they arrive.
pub(super) async fn stream_turn(
    engine: &Engine,
    messages: &[Value],
    media: &SessionMedia,
    options: &ChatOptions,
    sampling: &Sampling,
) -> anyhow::Result<Turn> {
    let request = chat_request(messages, options, sampling)?;
    let guard = TurnGuard::preparing();
    let mut stream = engine
        .chat_stream(request, media.shared.clone())
        .await
        .map_err(|e| anyhow::anyhow!(media.describe(&e.to_string())))?
        .with_denoising_progress();
    guard.streaming(stream.cancellation());
    read_turn(engine, &mut stream, Instant::now()).await
}

async fn read_turn(
    engine: &Engine,
    stream: &mut inference_api::engine_chat::ChatStream,
    start: Instant,
) -> anyhow::Result<Turn> {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut time_to_first_token = None;
    let mut usage = None;
    let mut files = Vec::new();
    let mut denoising = DenoisingBar::default();
    let mut was_reasoning = false;

    while let Some(event) = stream.next_event().await {
        match event {
            ChatStreamEvent::Chunk(chunk) => {
                denoising.clear();
                usage = chunk.usage.clone();
                let Some(choice) = chunk.choices.first() else {
                    continue;
                };
                let delta = &choice.delta;
                if (delta.content.is_some() || delta.reasoning_content.is_some())
                    && time_to_first_token.is_none()
                {
                    time_to_first_token = Some(start.elapsed());
                }
                if let Some(text) = &delta.reasoning_content {
                    reasoning.push_str(text);
                    print!("{GRAY}{text}{RESET}");
                    was_reasoning = true;
                }
                if let Some(text) = &delta.content {
                    if was_reasoning {
                        println!();
                        was_reasoning = false;
                    }
                    content.push_str(text);
                    print!("{text}");
                }
                io::stdout().flush()?;
                if let Some(finish_reason) = &choice.finish_reason {
                    if was_reasoning {
                        println!();
                    }
                    if finish_reason == "length" {
                        print!("...");
                    }
                }
            }
            ChatStreamEvent::AgenticToolCallProgress(progress) => {
                denoising.clear();
                print_tool_progress(&progress.tool_name, &progress.phase, &files);
                if matches!(progress.phase, AgenticToolCallPhase::Complete(_)) {
                    files.clear();
                }
            }
            ChatStreamEvent::AgenticToolApprovalRequired(approval) => {
                denoising.clear();
                let decision = ask_approval(approval.clone()).await;
                // A late answer finds the approval already denied by the engine's timeout.
                if let Err(e) = engine.resolve_approval(&approval.approval_id, decision) {
                    println!("│ The answer was not recorded: {e}");
                }
            }
            ChatStreamEvent::FileProduced(file) => files.push(file),
            ChatStreamEvent::BlockDenoisingProgress(progress) => denoising.render(&progress),
            ChatStreamEvent::Error(error) => {
                denoising.clear();
                anyhow::bail!("{error}");
            }
        }
    }
    denoising.clear();

    let mut message = json!({"role": "assistant", "content": content});
    // Thinking models expect their own reasoning back in history (Qwen3.5 preserve_thinking).
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    Ok(Turn {
        message,
        time_to_first_token,
        usage,
    })
}

pub(super) fn print_stats(turn: &Turn, sampling: &Sampling) {
    let Some(usage) = &turn.usage else {
        return;
    };
    println!();
    println!();
    println!("Stats:");
    if let Some(ttft) = turn.time_to_first_token {
        println!("CLI time to first token: {:.2?}s", ttft.as_secs_f32());
    }
    println!(
        "Prompt: {} tokens, {:.2} T/s",
        usage.prompt_tokens, usage.avg_prompt_tok_per_sec
    );
    println!(
        "Decode: {} tokens, {:.2} T/s",
        usage.completion_tokens, usage.avg_compl_tok_per_sec
    );
    if let Some(details) = &usage.prompt_tokens_details
        && details.cached_tokens > 0
    {
        println!(
            "Prefix cache: {} prompt tokens reused",
            details.cached_tokens
        );
    }
    println!("Sampling: {}", sampling.describe());
}

// The terminal is the approval UI: the tool and its arguments, then y / n / a (approve for the rest of the session).
async fn ask_approval(approval: AgenticToolApproval) -> ApprovalDecisionRequest {
    let prompt = tokio::task::spawn_blocking(move || -> io::Result<ApprovalDecisionRequest> {
        print_divider("approval");
        println!("│ session: {}", approval.session_id);
        println!("│ tool: {}", approval.tool.label);
        if matches!(approval.tool.kind, AgentToolKind::Shell)
            && let Some(commands) = approval.arguments.get("commands").and_then(Value::as_array)
        {
            println!("│ commands:");
            for line in commands
                .iter()
                .filter_map(Value::as_str)
                .flat_map(str::lines)
            {
                println!("│   {line}");
            }
        }
        if let Some(outputs) = approval.arguments.get("outputs").and_then(Value::as_array) {
            let outputs = outputs.iter().filter_map(Value::as_str).collect::<Vec<_>>();
            if !outputs.is_empty() {
                println!("│ outputs: {}", outputs.join(", "));
            }
        }
        loop {
            print!("│ Approve action? [y]es / [n]o / [a]lways: ");
            io::stdout().flush()?;
            let mut input = String::new();
            if io::stdin().read_line(&mut input)? == 0 {
                return Ok(decision(ApprovalDecision::Deny, false));
            }
            match input.trim().to_ascii_lowercase().as_str() {
                "y" | "yes" => return Ok(decision(ApprovalDecision::Approve, false)),
                "a" | "always" => return Ok(decision(ApprovalDecision::Approve, true)),
                "" | "n" | "no" => return Ok(decision(ApprovalDecision::Deny, false)),
                _ => println!("│ Please enter y, n, or a."),
            }
        }
    });
    // Unreadable input denies, as an unanswered prompt would.
    match prompt.await {
        Ok(Ok(decision)) => decision,
        _ => decision(ApprovalDecision::Deny, false),
    }
}

fn decision(decision: ApprovalDecision, remember_for_session: bool) -> ApprovalDecisionRequest {
    ApprovalDecisionRequest {
        decision,
        remember_for_session,
        message: None,
    }
}

fn print_block(label: &str, text: Option<&str>) {
    println!("│ {label}:");
    match text.map(str::trim).filter(|text| !text.is_empty()) {
        Some(text) => text.lines().for_each(|line| println!("│   {line}")),
        None => println!("│   {GRAY}<none>{RESET}"),
    }
}

fn print_divider(label: &str) {
    let divider = format!("├─ {label} ");
    let pad = AGENTIC_PANEL_WIDTH.saturating_sub(divider.len());
    println!("{divider}{}", "─".repeat(pad));
}

fn print_tool_progress(tool_name: &str, phase: &AgenticToolCallPhase, files: &[File]) {
    match phase {
        AgenticToolCallPhase::Calling(data) => {
            let header = format!("╭─ tool call: {tool_name} ");
            let pad = AGENTIC_PANEL_WIDTH.saturating_sub(header.len());
            println!("\n{header}{}", "─".repeat(pad));
            match data {
                AgenticToolCallData::CodeExecution {
                    code: Some(code), ..
                } => code.lines().for_each(|line| println!("│ {line}")),
                AgenticToolCallData::WebSearch {
                    query: Some(query), ..
                } => println!("│ query: {query}"),
                AgenticToolCallData::Shell { commands, .. } => commands
                    .iter()
                    .flat_map(|command| command.lines())
                    .for_each(|line| println!("│ {line}")),
                AgenticToolCallData::Custom { arguments, .. } if !arguments.is_empty() => {
                    println!("│ {arguments}")
                }
                _ => {}
            }
        }
        AgenticToolCallPhase::Complete(data) => {
            match data {
                AgenticToolCallData::CodeExecution {
                    stdout,
                    stderr,
                    exception,
                    images,
                    video_frame_count,
                    working_directory,
                    execution_time_ms,
                    ..
                } => {
                    let timing = execution_time_ms
                        .map(|ms| format!(" ({ms}ms)"))
                        .unwrap_or_default();
                    let status = if exception.is_some() {
                        "error"
                    } else {
                        "result"
                    };
                    print_divider(&format!("{status}{timing}"));
                    if let Some(dir) = working_directory {
                        println!("│ workdir: {dir}");
                    }
                    print_block("stdout", stdout.as_deref());
                    print_block("stderr", stderr.as_deref());
                    if let Some(exception) = exception {
                        exception.lines().for_each(|line| println!("│ {line}"));
                    }
                    if !images.is_empty() {
                        println!("│ {} image(s) captured", images.len());
                    }
                    if let Some(n) = video_frame_count {
                        println!("│ {n} video frame(s) captured");
                    }
                    if !files.is_empty() {
                        println!("│ files:");
                        for file in files {
                            println!(
                                "│   {} ({}, {} bytes)",
                                file.name,
                                file.format.as_deref().unwrap_or(""),
                                file.bytes
                            );
                        }
                    }
                }
                AgenticToolCallData::WebSearch {
                    results_count,
                    sources,
                    ..
                } => {
                    print_divider("result");
                    if let Some(n) = results_count {
                        println!("│ {n} results found");
                    }
                    if !sources.is_empty() {
                        println!("│ sources:");
                        sources.iter().for_each(|source| println!("│   {source}"));
                    }
                }
                AgenticToolCallData::Shell {
                    stdout,
                    stderr,
                    exit_code,
                    status,
                    working_directory,
                    timed_out,
                    ..
                } => {
                    print_divider(status.as_deref().unwrap_or("result"));
                    if let Some(dir) = working_directory {
                        println!("│ workdir: {dir}");
                    }
                    if let Some(code) = exit_code {
                        println!("│ exit: {code}");
                    }
                    if matches!(timed_out, Some(true)) {
                        println!("│ timed out");
                    }
                    print_block("stdout", stdout.as_deref());
                    print_block("stderr", stderr.as_deref());
                }
                AgenticToolCallData::Custom { content, .. } if !content.is_empty() => {
                    print_divider("result");
                    content
                        .lines()
                        .take(5)
                        .for_each(|line| println!("│ {line}"));
                }
                _ => {}
            }
            println!("{}", "╰".to_string() + &"─".repeat(AGENTIC_PANEL_WIDTH));
        }
    }
    let _ = io::stdout().flush();
}

#[derive(Default)]
struct DenoisingBar {
    active: bool,
}

impl DenoisingBar {
    fn clear(&mut self) {
        if self.active {
            eprint!("\r\x1b[K");
            let _ = io::stderr().flush();
            self.active = false;
        }
    }

    fn render(&mut self, progress: &BlockDenoisingProgress) {
        // Only the first row draws the bar; the last block has nothing left to show.
        if progress.index != 0 || progress.final_block {
            self.clear();
            return;
        }
        let total_steps = progress.total_steps.max(1);
        let step = progress.step.min(total_steps);
        let status = if progress.finished {
            "stable"
        } else {
            "denoising"
        };
        let filled = (step * DENOISING_BAR_WIDTH / total_steps).min(DENOISING_BAR_WIDTH);
        eprint!(
            "\rblock diffusion [{}{}] {step}/{total_steps} {status}\x1b[K",
            "=".repeat(filled),
            " ".repeat(DENOISING_BAR_WIDTH - filled),
        );
        let _ = io::stderr().flush();
        self.active = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_model_without_generation_defaults_samples_broadly() {
        let sampling = Sampling::for_model(None);
        assert_eq!(sampling.temperature, Some(FALLBACK_TEMPERATURE));
        assert_eq!(sampling.top_k, Some(FALLBACK_TOP_K));
        assert_eq!(sampling.top_p, Some(FALLBACK_TOP_P));
        assert_eq!(sampling.min_p, Some(FALLBACK_MIN_P));
    }

    #[test]
    fn do_sample_false_means_greedy() {
        let defaults = GenerationDefaults {
            do_sample: Some(false),
            temperature: Some(0.7),
            max_new_tokens: Some(128),
            ..Default::default()
        };
        let sampling = Sampling::for_model(Some(&defaults));
        assert_eq!((sampling.temperature, sampling.top_k), (None, Some(1)));
        assert_eq!(sampling.max_tokens, Some(128));
    }

    #[test]
    fn sampling_commands_edit_the_session() {
        let mut sampling = Sampling::for_model(None);
        assert!(sampling.apply_command("/temperature 0.2"));
        assert!(sampling.apply_command("/topk 7"));
        assert!(!sampling.apply_command("hello"));
        assert_eq!((sampling.temperature, sampling.top_k), (Some(0.2), Some(7)));
    }

    #[test]
    fn a_request_carries_the_sessions_tools_and_sampling() {
        let options = ChatOptions {
            do_search: true,
            do_code_exec: true,
            do_shell: false,
            agent_permission: AgentPermission::Ask,
            enable_thinking: Some(false),
            reasoning_effort: Some(ReasoningEffort::High),
            adapter: Some("style".to_string()),
            session_id: Some("session".to_string()),
        };
        let messages = [text_message("user", "hi")];
        let request = chat_request(&messages, &options, &Sampling::for_model(None)).unwrap();
        let request = serde_json::to_value(request).unwrap();
        assert_eq!(request["agent_permission"], "ask");
        assert_eq!(request["reasoning_effort"], "high");
        assert_eq!(request["session_id"], "session");
        assert_eq!(request["tools"][0]["type"], "code_interpreter");
        assert!(request["web_search_options"].is_object());
        assert_eq!(request["top_k"], FALLBACK_TOP_K);
    }

    const MAX_TOKENS: usize = 4;

    #[tokio::test(flavor = "multi_thread")]
    async fn a_turn_streams_through_the_engine_api_and_reports_its_usage() -> anyhow::Result<()> {
        let dir = crate::commands::tiny_support::tiny_checkpoint()?;
        let spec = serde_json::from_value(json!({
            "model": {"MultimodalPlain": {"model_id": dir.path().to_string_lossy(), "dtype": "f32"}},
            "runtime": {"device": "cpu"},
        }))?;
        let engine = Engine::load(spec).await?;
        let model = default_model(&engine)?;
        let mut sampling = Sampling::for_model(model.generation_defaults.as_ref());
        sampling.max_tokens = Some(MAX_TOKENS);
        let options = ChatOptions {
            do_search: false,
            do_code_exec: false,
            do_shell: false,
            agent_permission: AgentPermission::Auto,
            enable_thinking: None,
            reasoning_effort: None,
            adapter: None,
            session_id: None,
        };
        let messages = [text_message("user", "Reply with the single word: ok")];
        let turn = stream_turn(
            &engine,
            &messages,
            &SessionMedia::default(),
            &options,
            &sampling,
        )
        .await?;
        assert_eq!(turn.message["role"], "assistant");
        let usage = turn.usage.expect("the final chunk carries usage");
        assert!(
            usage.completion_tokens <= MAX_TOKENS && usage.prompt_tokens > 0,
            "{usage:?}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_second_image_turn_resends_the_first_image_by_its_index() -> anyhow::Result<()> {
        let dir = crate::commands::tiny_support::tiny_checkpoint()?;
        let spec = serde_json::from_value(json!({
            "model": {"MultimodalPlain": {"model_id": dir.path().to_string_lossy(), "dtype": "f32"}},
            "runtime": {"device": "cpu"},
        }))?;
        let engine = Engine::load(spec).await?;
        let images = tempfile::tempdir()?;
        let mut media = SessionMedia::default();
        let mut messages = Vec::new();
        let mut sampling = Sampling::for_model(None);
        sampling.max_tokens = Some(MAX_TOKENS);
        let options = ChatOptions {
            do_search: false,
            do_code_exec: false,
            do_shell: false,
            agent_permission: AgentPermission::Auto,
            enable_thinking: None,
            reasoning_effort: None,
            adapter: None,
            session_id: None,
        };
        for name in ["first.png", "second.png"] {
            let path = images.path().join(name);
            image::DynamicImage::new_rgb8(28, 28).save(&path)?;
            let source = media
                .source_for(&path.to_string_lossy(), IMAGE_PART)
                .await?;
            messages.push(media_message(vec![(IMAGE_PART, source)], "OCR:"));
            let turn = stream_turn(&engine, &messages, &media, &options, &sampling).await?;
            messages.push(turn.message);
        }
        assert_eq!(media.len(), 2);
        assert_eq!(messages[2]["content"][0]["image_url"]["url"], "media://1");
        Ok(())
    }

    #[tokio::test]
    async fn media_loads_once_and_a_failed_turn_can_drop_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("picture.png");
        std::fs::write(&path, b"png").unwrap();
        let mut media = SessionMedia::default();
        let name = path.to_string_lossy().to_string();
        assert_eq!(
            media.source_for(&name, IMAGE_PART).await.unwrap(),
            "media://0"
        );
        let url = format!("file://{}", path.display());
        assert_eq!(
            media.source_for(&url, IMAGE_PART).await.unwrap(),
            "media://1"
        );
        assert_eq!(
            media.describe("Failed to parse image resource: media://1"),
            format!("Failed to parse image resource: {url}")
        );
        media.truncate(1);
        assert_eq!(media.len(), 1);
        let missing = dir.path().join("missing.png");
        let error = media
            .source_for(&missing.to_string_lossy(), IMAGE_PART)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("missing.png"), "{error}");
        let message = media_message(vec![(IMAGE_PART, "media://0".to_string())], "What is this?");
        assert_eq!(message["content"][0]["image_url"]["url"], "media://0");
        assert_eq!(message["content"][1]["text"], "What is this?");
    }
}
