use std::{path::Path, sync::Arc};

use either::Either;
use image::DynamicImage;
use indexmap::IndexMap;
use num_traits::ToPrimitive;

use serde_json::Value;

use inference_core::{
    AgentPermission, AgentToolApproval, AgentToolApprovalCallback, AgentToolApprovalDecision,
    AgentToolApprovalHandler, AgentToolKind, AgentToolMetadata, AgentToolSource,
    AgenticToolCallData, AgenticToolCallPhase, Engine, FINISH_REASON_CANCELED, MessageContent,
    NormalRequest, Request, RequestMessage, Response, SupportedModality, ToolCallResponse,
    ToolChoice, Usage, WebSearchOptions,
    agent::{
        AGENTIC_LOOP_REENTRY_SENTINEL, CODE_EXECUTION, DEFAULT_MAX_TOOL_ROUNDS, is_code_exec_tool,
        is_list_files_tool, is_read_file_tool, is_shell_tool, is_surface_outputs_tool,
    },
    files::{
        File, FileSource, RequestedFile, compose_tool_response_with_files, file_to_tool_input_file,
        merge_required_outputs_into_args, required_files_tool_addendum, tool_file_to_file,
    },
};

use crate::file_tools::{do_list_files, do_read_file};

const MAX_SKILL_TREE_ENTRIES: usize = 80;
const DENIED: &str = "Agent action was denied.";
const MAX_SKILL_TREE_DEPTH: usize = 4;

/// Turn = number of completed user messages.
fn count_user_messages(request: &NormalRequest) -> usize {
    request
        .chat_messages()
        .iter()
        .filter(|m| {
            matches!(
                m.get("role"),
                Some(Either::Left(s)) if s == "user"
            )
        })
        .count()
        .saturating_sub(1)
}

fn shell_skill_dir_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn append_skill_tree(content: &mut String, source_path: &Path, mounted_dir: &str) {
    content.push_str("  File tree:\n");
    let mut entries = 0;
    append_skill_tree_entries(content, source_path, mounted_dir, 2, &mut entries);
    if entries >= MAX_SKILL_TREE_ENTRIES {
        content.push_str("    ...\n");
    }
}

fn append_skill_tree_entries(
    content: &mut String,
    source_path: &Path,
    mounted_path: &str,
    depth: usize,
    entries: &mut usize,
) {
    if depth > MAX_SKILL_TREE_DEPTH || *entries >= MAX_SKILL_TREE_ENTRIES {
        return;
    }

    let Ok(read_dir) = std::fs::read_dir(source_path) else {
        content.push_str(&format!("    - {mounted_path}/ (unavailable)\n"));
        *entries += 1;
        return;
    };
    let mut children = read_dir.filter_map(Result::ok).collect::<Vec<_>>();
    children.sort_by_key(|entry| entry.file_name());

    for child in children {
        if *entries >= MAX_SKILL_TREE_ENTRIES {
            return;
        }
        let file_name = child.file_name().to_string_lossy().to_string();
        let child_mounted_path = format!("{mounted_path}/{file_name}");
        let Ok(file_type) = child.file_type() else {
            continue;
        };
        let indent = "  ".repeat(depth);
        if file_type.is_dir() {
            content.push_str(&format!("{indent}- {child_mounted_path}/\n"));
            *entries += 1;
            append_skill_tree_entries(
                content,
                &child.path(),
                &child_mounted_path,
                depth + 1,
                entries,
            );
        } else if file_type.is_file() {
            content.push_str(&format!("{indent}- {child_mounted_path}\n"));
            *entries += 1;
        }
    }
}

fn inject_shell_skills_message(request: &mut NormalRequest) {
    let Some(shell_options) = &request.shell_options else {
        return;
    };
    if shell_options.skills.is_empty() {
        return;
    }

    let mut content = String::from(
        "Uploaded skills are folders available to the shell tool in the session working directory.\n\
         Skills are not shell commands and are not installed on PATH. Do not invent commands named \
         after a skill.\n\
         Before running any command from a skill, you must read that skill's SKILL.md file. This is \
         required.\n\
         After reading SKILL.md, follow its workflow. If the skill uses bundled scripts, run them by \
         path under the skill folder, for example `python skills/<skill-name>/scripts/<script>.py ...`.\n",
    );
    for skill in &shell_options.skills {
        let mounted_dir = format!("skills/{}", shell_skill_dir_name(&skill.name));
        content.push_str(&format!("- {}: {}\n", skill.name, skill.description));
        content.push_str(&format!(
            "  Required first command: `cat {mounted_dir}/SKILL.md`\n"
        ));
        append_skill_tree(&mut content, &skill.source_path, &mounted_dir);
    }

    let messages = request.chat_messages_mut();
    let mut message: IndexMap<String, MessageContent> = IndexMap::new();
    message.insert("role".to_string(), Either::Left("system".to_string()));
    message.insert("content".to_string(), Either::Left(content));
    messages.insert(0, message);
}

/// Structured `tool_calls` field for the assistant message. Required by templates (Gemma 4 etc.) that render from `message.tool_calls`.
fn build_tool_calls_field(calls: &[ToolCallResponse]) -> MessageContent {
    let calls = calls
        .iter()
        .map(|tc| {
            let mut tc_map = IndexMap::new();
            tc_map.insert("id".to_string(), Value::String(tc.id.clone()));
            tc_map.insert("type".to_string(), Value::String("function".to_string()));
            let mut function_map = serde_json::Map::new();
            function_map.insert("name".to_string(), Value::String(tc.function.name.clone()));
            let args_value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(Value::String(tc.function.arguments.clone()));
            function_map.insert("arguments".to_string(), args_value);
            tc_map.insert("function".to_string(), Value::Object(function_map));
            tc_map
        })
        .collect();
    Either::Right(calls)
}

pub(crate) fn append_assistant_tool_calls(
    messages: &mut Vec<IndexMap<String, MessageContent>>,
    calls: &[ToolCallResponse],
) {
    let mut message: IndexMap<String, MessageContent> = IndexMap::new();
    message.insert("role".to_string(), Either::Left("assistant".to_string()));
    message.insert("content".to_string(), Either::Left(String::new()));
    message.insert("tool_calls".to_string(), build_tool_calls_field(calls));
    messages.push(message);
}

fn attach_reasoning_to_latest_assistant_tool_call(
    messages: &mut [IndexMap<String, MessageContent>],
    reasoning_content: Option<&str>,
) {
    let Some(reasoning_content) = reasoning_content.filter(|content| !content.is_empty()) else {
        return;
    };
    let Some(message) = messages.iter_mut().rev().find(|message| {
        message.contains_key("tool_calls")
            && matches!(message.get("role"), Some(Either::Left(role)) if role == "assistant")
    }) else {
        return;
    };
    message.insert(
        "reasoning_content".to_string(),
        Either::Left(reasoning_content.to_string()),
    );
}

fn tool_message(
    tc: &ToolCallResponse,
    content: MessageContent,
) -> IndexMap<String, MessageContent> {
    let mut message: IndexMap<String, MessageContent> = IndexMap::new();
    message.insert("role".to_string(), Either::Left("tool".to_string()));
    message.insert("tool_call_id".to_string(), Either::Left(tc.id.clone()));
    message.insert("name".to_string(), Either::Left(tc.function.name.clone()));
    message.insert("content".to_string(), content);
    message
}

/// Appends a call's answer, routing images and video to the request's media when the model takes them.
fn append_tool_outcome(
    request: &mut NormalRequest,
    tc: &ToolCallResponse,
    outcome: ToolOutcome,
    ctx: &DispatchCtx<'_>,
) -> (AgenticToolCallData, Vec<File>) {
    let ToolOutcome {
        mut content,
        images,
        video_frames,
        data,
        files,
    } = outcome;
    let inject_images = !images.is_empty() && ctx.supports_vision;
    let inject_video = !video_frames.is_empty() && ctx.supports_video;

    if !images.is_empty() && !ctx.supports_vision {
        content.push_str(&format!(
            "\n[ERROR: {} image(s) were generated but this model does not support vision input. Do not attempt to generate images.]",
            images.len()
        ));
    }
    if !video_frames.is_empty() && !ctx.supports_video {
        content.push_str(&format!(
            "\n[ERROR: {} video frame(s) were generated but this model does not support video input. Do not attempt to generate video.]",
            video_frames.len()
        ));
    }

    if !inject_images && !inject_video {
        let message = tool_message(tc, Either::Left(content));
        request.chat_messages_mut().push(message);
        return (data, files);
    }

    request.upgrade_to_multimodal();

    let mut parts: Vec<IndexMap<String, Value>> = Vec::new();

    if inject_images {
        let req_images = request.images_mut();
        for img in images {
            req_images.push(img);
            let mut part = IndexMap::new();
            part.insert("type".to_string(), Value::String("image".to_string()));
            parts.push(part);
        }
    }

    if inject_video {
        let video = inference_core::VideoInput::from_frames(video_frames, 1.0, None);
        request.videos_mut().push(video);
        let mut part = IndexMap::new();
        part.insert("type".to_string(), Value::String("video".to_string()));
        parts.push(part);
    }

    let mut text_part = IndexMap::new();
    text_part.insert("type".to_string(), Value::String("text".to_string()));
    text_part.insert("text".to_string(), Value::String(content));
    parts.push(text_part);

    let message = tool_message(tc, Either::Right(parts));
    request.chat_messages_mut().push(message);
    (data, files)
}

/// `Some(resp)` for `Done`/`Chunk` (caller handles); forwards everything else and returns `None`.
async fn forward_passthrough(
    resp: Response,
    user_sender: &tokio::sync::mpsc::Sender<Response>,
) -> Option<Response> {
    match resp {
        Response::Done(_) | Response::Chunk(_) => Some(resp),
        other => {
            let _ = user_sender.send(other).await;
            None
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct ClientDisconnected;

async fn recv_or_client_disconnect<T>(
    receiver: &mut tokio::sync::mpsc::Receiver<T>,
    user_sender: &tokio::sync::mpsc::Sender<Response>,
) -> Result<Option<T>, ClientDisconnected> {
    tokio::select! {
        biased;
        _ = user_sender.closed() => Err(ClientDisconnected),
        response = receiver.recv() => Ok(response),
    }
}

#[derive(Default)]
struct AgenticUsageAccumulator {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_prompt_time_sec: f32,
    total_completion_time_sec: f32,
    saw_usage: bool,
}

impl AgenticUsageAccumulator {
    fn add(&mut self, usage: &Usage) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(usage.prompt_tokens);
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(usage.completion_tokens);
        self.total_prompt_time_sec += usage.total_prompt_time_sec;
        self.total_completion_time_sec += usage.total_completion_time_sec;
        self.saw_usage = true;
    }

    fn aggregate(&self) -> Option<Usage> {
        self.saw_usage.then(|| {
            let total_tokens = self.prompt_tokens.saturating_add(self.completion_tokens);
            let total_time_sec = self.total_prompt_time_sec + self.total_completion_time_sec;
            Usage {
                completion_tokens: self.completion_tokens,
                prompt_tokens: self.prompt_tokens,
                total_tokens,
                prompt_tokens_details: None,
                avg_tok_per_sec: tps(total_tokens, total_time_sec),
                avg_prompt_tok_per_sec: tps(self.prompt_tokens, self.total_prompt_time_sec),
                avg_compl_tok_per_sec: tps(self.completion_tokens, self.total_completion_time_sec),
                total_time_sec,
                total_prompt_time_sec: self.total_prompt_time_sec,
                total_completion_time_sec: self.total_completion_time_sec,
            }
        })
    }
}

fn tps(tokens: usize, seconds: f32) -> f32 {
    if seconds > 0.0 {
        tokens.to_f32().unwrap_or(f32::MAX) / seconds
    } else {
        0.0
    }
}

/// Persist the conversation as-is. Refreshes file TTLs.
fn save_session(engine: &Arc<Engine>, session_id: &str, visible_req: &NormalRequest) {
    let messages = visible_req.chat_messages().clone();
    let (images, videos) = match &visible_req.messages {
        RequestMessage::MultimodalChat { images, videos, .. } => (images.clone(), videos.clone()),
        _ => (Vec::new(), Vec::new()),
    };
    let entry = inference_core::agentic_session::AgenticSessionEntry::new(messages, images, videos);
    engine.session_store().lock().unwrap().save(
        session_id.to_string(),
        entry,
        visible_req.owner.as_deref(),
    );
    engine.file_store().touch_session(session_id);
}

use crate::{search, tool_dispatch};

fn shell_commands_from_args(arguments: &str) -> Vec<String> {
    serde_json::from_str::<serde_json::Value>(arguments)
        .ok()
        .and_then(|v| {
            v.get("commands").and_then(|commands| {
                commands.as_array().map(|commands| {
                    commands
                        .iter()
                        .filter_map(|command| command.as_str().map(str::to_string))
                        .collect::<Vec<_>>()
                })
            })
        })
        .unwrap_or_default()
}

fn calling_data_for_tool(tc: &ToolCallResponse) -> AgenticToolCallData {
    if search::search_tool_called(&tc.function.name) {
        let query = serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
            .ok()
            .and_then(|v| {
                v.get("query")
                    .and_then(|q| q.as_str())
                    .map(|s| s.to_string())
            });
        AgenticToolCallData::WebSearch {
            query,
            results_count: None,
            sources: Vec::new(),
        }
    } else if is_read_file_tool(&tc.function.name)
        || is_list_files_tool(&tc.function.name)
        || is_surface_outputs_tool(&tc.function.name)
    {
        AgenticToolCallData::Custom {
            arguments: tc.function.arguments.clone(),
            content: String::new(),
        }
    } else if is_code_exec_tool(&tc.function.name) {
        let code = serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
            .ok()
            .and_then(|v| {
                v.get("code")
                    .and_then(|c| c.as_str())
                    .map(|s| s.to_string())
            });
        AgenticToolCallData::CodeExecution {
            code,
            stdout: None,
            stderr: None,
            exception: None,
            images: vec![],
            video_frame_count: None,
            video_frames: vec![],
            working_directory: None,
            execution_time_ms: None,
        }
    } else if is_shell_tool(&tc.function.name) {
        AgenticToolCallData::Shell {
            commands: shell_commands_from_args(&tc.function.arguments),
            stdout: None,
            stderr: None,
            exit_code: None,
            status: None,
            working_directory: None,
            timed_out: None,
        }
    } else {
        AgenticToolCallData::Custom {
            arguments: tc.function.arguments.clone(),
            content: String::new(),
        }
    }
}

fn tool_arguments(tc: &ToolCallResponse) -> Value {
    serde_json::from_str(&tc.function.arguments)
        .unwrap_or_else(|_| Value::String(tc.function.arguments.clone()))
}

fn tool_metadata_for(ctx: &DispatchCtx<'_>, tc: &ToolCallResponse) -> AgentToolMetadata {
    let name = &tc.function.name;
    if is_read_file_tool(name) || is_list_files_tool(name) {
        AgentToolMetadata {
            source: AgentToolSource::BuiltIn,
            kind: AgentToolKind::File,
            label: "File access".to_string(),
        }
    } else if is_surface_outputs_tool(name) {
        AgentToolMetadata {
            source: AgentToolSource::BuiltIn,
            kind: AgentToolKind::File,
            label: "File outputs".to_string(),
        }
    } else if is_code_exec_tool(name) {
        AgentToolMetadata {
            source: AgentToolSource::BuiltIn,
            kind: AgentToolKind::CodeExecution,
            label: "Python code".to_string(),
        }
    } else if search::search_tool_called(name) {
        AgentToolMetadata {
            source: AgentToolSource::BuiltIn,
            kind: AgentToolKind::WebSearch,
            label: if name == search::SEARCH_TOOL_NAME {
                "Web search".to_string()
            } else {
                "Web page extraction".to_string()
            },
        }
    } else if is_shell_tool(name) {
        AgentToolMetadata {
            source: AgentToolSource::BuiltIn,
            kind: AgentToolKind::Shell,
            label: "Shell command".to_string(),
        }
    } else if ctx.engine.tool_callbacks().contains_key(name) {
        AgentToolMetadata {
            source: AgentToolSource::User,
            kind: AgentToolKind::Custom,
            label: name.clone(),
        }
    } else if ctx.dispatch_url.is_some() {
        AgentToolMetadata {
            source: AgentToolSource::External,
            kind: AgentToolKind::External,
            label: name.clone(),
        }
    } else {
        AgentToolMetadata {
            source: AgentToolSource::User,
            kind: AgentToolKind::Custom,
            label: name.clone(),
        }
    }
}

async fn call_agent_approval_callback(
    callback: AgentToolApprovalCallback,
    approval: AgentToolApproval,
) -> AgentToolApprovalDecision {
    match tokio::task::spawn_blocking(move || callback(&approval)).await {
        Ok(decision) => decision,
        Err(_) => AgentToolApprovalDecision::deny_with_message(
            "Agent action requires approval, but the approval handler failed.",
        ),
    }
}

async fn call_agent_approval_handler(
    handler: AgentToolApprovalHandler,
    approval: AgentToolApproval,
) -> AgentToolApprovalDecision {
    match handler {
        AgentToolApprovalHandler::Sync(callback) => {
            call_agent_approval_callback(callback, approval).await
        }
        AgentToolApprovalHandler::Async(callback) => {
            match tokio::spawn(async move { callback(approval).await }).await {
                Ok(decision) => decision,
                Err(_) => AgentToolApprovalDecision::deny_with_message(
                    "Agent action requires approval, but the approval handler failed.",
                ),
            }
        }
    }
}

async fn approve_agent_tool(
    ctx: &DispatchCtx<'_>,
    tc: &ToolCallResponse,
    round: usize,
) -> AgentToolApprovalDecision {
    let tool = tool_metadata_for(ctx, tc);
    let message = match ctx.agent_permission {
        AgentPermission::Auto => return AgentToolApprovalDecision::approve(),
        AgentPermission::Deny => format!("{} was denied by policy.", tool.label),
        AgentPermission::Ask => {
            if ctx
                .engine
                .session_store()
                .lock()
                .unwrap()
                .agent_actions_approved(&ctx.tool_call_ctx.sandbox_key(ctx.session_id))
            {
                return AgentToolApprovalDecision::approve();
            }
            let Some(handler) = &ctx.agent_approval_handler else {
                return AgentToolApprovalDecision::deny_with_message(
                    "Agent action requires approval, but no approval handler is configured.",
                );
            };
            let approval = AgentToolApproval {
                approval_id: format!("appr_{}", uuid::Uuid::new_v4().simple()),
                session_id: ctx.session_id.to_string(),
                round,
                tool,
                arguments: tool_arguments(tc),
            };
            if let Some(notifier) = &ctx.tool_call_ctx.agent_approval_notifier {
                notifier(inference_mcp::AgentToolApprovalRequest {
                    approval_id: approval.approval_id.clone(),
                    session_id: approval.session_id.clone(),
                    round: approval.round,
                    tool: approval.tool.clone(),
                    arguments: approval.arguments.clone(),
                });
            }
            let decision = call_agent_approval_handler(handler.clone(), approval).await;
            if decision.approve && decision.remember_for_session {
                ctx.engine
                    .session_store()
                    .lock()
                    .unwrap()
                    .approve_agent_actions(ctx.tool_call_ctx.sandbox_key(ctx.session_id));
            }
            return decision;
        }
    };
    AgentToolApprovalDecision::deny_with_message(message)
}

fn denied_tool_result(tc: &ToolCallResponse, message: String) -> ToolOutcome {
    let content = serde_json::json!({
        "status": "denied",
        "exception": message,
    })
    .to_string();

    let data = if is_read_file_tool(&tc.function.name)
        || is_list_files_tool(&tc.function.name)
        || is_surface_outputs_tool(&tc.function.name)
    {
        AgenticToolCallData::Custom {
            arguments: String::new(),
            content: content.clone(),
        }
    } else if is_code_exec_tool(&tc.function.name) {
        AgenticToolCallData::CodeExecution {
            code: None,
            stdout: None,
            stderr: None,
            exception: Some(message),
            images: vec![],
            video_frame_count: None,
            video_frames: vec![],
            working_directory: None,
            execution_time_ms: None,
        }
    } else if is_shell_tool(&tc.function.name) {
        AgenticToolCallData::Shell {
            commands: shell_commands_from_args(&tc.function.arguments),
            stdout: None,
            stderr: Some(message),
            exit_code: None,
            status: Some("denied".to_string()),
            working_directory: None,
            timed_out: Some(false),
        }
    } else {
        AgenticToolCallData::Custom {
            arguments: String::new(),
            content: content.clone(),
        }
    };

    ToolOutcome::text(content, data)
}

fn shell_completion_data(arguments: &str, content: &str) -> AgenticToolCallData {
    let val = serde_json::from_str::<serde_json::Value>(content).ok();
    AgenticToolCallData::Shell {
        commands: val
            .as_ref()
            .and_then(|v| v.get("commands"))
            .and_then(|commands| {
                commands.as_array().map(|commands| {
                    commands
                        .iter()
                        .filter_map(|command| command.as_str().map(str::to_string))
                        .collect::<Vec<_>>()
                })
            })
            .filter(|commands| !commands.is_empty())
            .unwrap_or_else(|| shell_commands_from_args(arguments)),
        stdout: val
            .as_ref()
            .and_then(|v| v.get("stdout"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string),
        stderr: val
            .as_ref()
            .and_then(|v| v.get("stderr"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string),
        exit_code: val
            .as_ref()
            .and_then(|v| v.get("exit_code"))
            .and_then(|v| v.as_i64()),
        status: val
            .as_ref()
            .and_then(|v| v.get("status"))
            .and_then(|v| v.as_str())
            .map(str::to_string),
        working_directory: val
            .as_ref()
            .and_then(|v| v.get("working_directory"))
            .and_then(|v| v.as_str())
            .map(str::to_string),
        timed_out: val
            .as_ref()
            .and_then(|v| v.get("timed_out"))
            .and_then(|v| v.as_bool()),
    }
}

/// One call's answer, appended to the conversation once every call of its round has run.
struct ToolOutcome {
    content: String,
    images: Vec<DynamicImage>,
    video_frames: Vec<DynamicImage>,
    data: AgenticToolCallData,
    files: Vec<File>,
}

impl ToolOutcome {
    fn text(content: String, data: AgenticToolCallData) -> Self {
        Self {
            content,
            images: Vec::new(),
            video_frames: Vec::new(),
            data,
            files: Vec::new(),
        }
    }

    fn custom(content: String) -> Self {
        let data = AgenticToolCallData::Custom {
            arguments: String::new(),
            content: content.clone(),
        };
        Self::text(content, data)
    }
}

/// Per-loop dispatch context. Borrows data owned by the loop's task; the round and its calls are passed alongside.
struct DispatchCtx<'a> {
    engine: &'a Arc<Engine>,
    user_sender: &'a tokio::sync::mpsc::Sender<Response>,
    web_search_options: Option<&'a WebSearchOptions>,
    dispatch_url: Option<&'a str>,
    supports_vision: bool,
    supports_video: bool,
    tool_call_ctx: &'a inference_mcp::ToolCallContext,
    owner: Option<&'a str>,
    turn: usize,
    session_id: &'a str,
    required_files: &'a [RequestedFile],
    agent_permission: AgentPermission,
    agent_approval_handler: Option<AgentToolApprovalHandler>,
}

fn web_search_metadata(content: &str) -> (Option<usize>, Vec<String>) {
    let Ok(value) = serde_json::from_str::<Value>(content) else {
        return (None, Vec::new());
    };

    let results_count = value.get("output").and_then(|output| {
        if let Some(results) = output.as_array() {
            Some(results.len())
        } else if output.is_string() {
            Some(1)
        } else {
            None
        }
    });

    let sources = value
        .get("sources")
        .and_then(|sources| sources.as_array())
        .map(|sources| {
            sources
                .iter()
                .filter_map(|source| source.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_else(|| {
            value
                .get("output")
                .and_then(|output| output.as_array())
                .map(|results| {
                    search::source_domains(
                        results
                            .iter()
                            .filter_map(|result| result.get("url").and_then(|url| url.as_str())),
                    )
                })
                .unwrap_or_default()
        });

    (results_count, sources)
}

fn extraction_sources(tc: &ToolCallResponse) -> Vec<String> {
    serde_json::from_str::<Value>(&tc.function.arguments)
        .ok()
        .and_then(|value| {
            value
                .get("url")
                .and_then(|url| url.as_str())
                .map(|url| search::source_domains([url]))
        })
        .unwrap_or_default()
}

async fn do_search(
    engine: &Arc<Engine>,
    tc: &ToolCallResponse,
    opts: &WebSearchOptions,
) -> ToolOutcome {
    let result = tool_dispatch::execute_search(engine, tc, opts).await;
    let (results_count, sources) = web_search_metadata(&result.content);
    let data = AgenticToolCallData::WebSearch {
        query: None, // already sent in Calling phase
        results_count,
        sources,
    };
    ToolOutcome::text(result.content, data)
}

async fn do_extraction(
    engine: &Arc<Engine>,
    tc: &ToolCallResponse,
    opts: &WebSearchOptions,
) -> ToolOutcome {
    let result = tool_dispatch::execute_extraction(engine, tc, opts).await;
    let data = AgenticToolCallData::WebSearch {
        query: None,
        results_count: Some(1),
        sources: extraction_sources(tc),
    };
    ToolOutcome::text(result.content, data)
}

async fn do_custom_tool(ctx: &DispatchCtx<'_>, tc: &ToolCallResponse, round: usize) -> ToolOutcome {
    // Merge required files into `outputs` so the tool surfaces them even if the model omitted them.
    let dispatched = if (is_code_exec_tool(&tc.function.name) || is_shell_tool(&tc.function.name))
        && !ctx.required_files.is_empty()
    {
        merge_required_outputs_into_args(tc, ctx.required_files)
    } else {
        tc.clone()
    };

    let mut tool_ctx = ctx.tool_call_ctx.clone();
    tool_ctx.round = Some(round);
    tool_ctx.tool_name = Some(tc.function.name.clone());

    // On the blocking pool, so the round's other calls run while a host callback blocks.
    let engine = ctx.engine.clone();
    let result = tokio::task::spawn_blocking(move || {
        tool_dispatch::execute_custom_tool(&engine, &dispatched, &tool_ctx)
    })
    .await
    .unwrap_or_else(|error| tool_dispatch::ToolResult::failed(&tc.function.name, &error));

    let files: Vec<File> = result
        .files
        .iter()
        .map(|tf| {
            let source = FileSource {
                tool: tc.function.name.clone(),
                round,
                turn: ctx.turn,
                tool_call_id: Some(tc.id.clone()),
            };
            tool_file_to_file(tf, source)
        })
        .collect();

    let is_code_exec = is_code_exec_tool(&tc.function.name);
    let data = if is_code_exec {
        let val = serde_json::from_str::<serde_json::Value>(&result.content).ok();
        AgenticToolCallData::CodeExecution {
            code: None, // already sent in Calling phase
            stdout: val
                .as_ref()
                .and_then(|v| v.get("stdout"))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string()),
            stderr: val
                .as_ref()
                .and_then(|v| v.get("stderr"))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string()),
            exception: val
                .as_ref()
                .and_then(|v| v.get("exception"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            images: result.images.clone(),
            video_frame_count: if result.video_frames.is_empty() {
                None
            } else {
                Some(result.video_frames.len())
            },
            video_frames: result.video_frames.clone(),
            working_directory: val
                .as_ref()
                .and_then(|v| v.get("working_directory"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            execution_time_ms: val
                .as_ref()
                .and_then(|v| v.get("execution_time_ms"))
                .and_then(|v| v.as_u64()),
        }
    } else if is_surface_outputs_tool(&tc.function.name) {
        AgenticToolCallData::Custom {
            arguments: tc.function.arguments.clone(),
            content: result.content.clone(),
        }
    } else if is_shell_tool(&tc.function.name) {
        shell_completion_data(&tc.function.arguments, &result.content)
    } else {
        AgenticToolCallData::Custom {
            arguments: String::new(), // already sent in Calling phase
            content: result.content.clone(),
        }
    };

    let content = compose_tool_response_with_files(&result.content, &files);
    ToolOutcome {
        content,
        images: result.images,
        video_frames: result.video_frames,
        data,
        files,
    }
}

async fn do_http_tool(tc: &ToolCallResponse, url: &str) -> ToolOutcome {
    let (call, url) = (tc.clone(), url.to_string());
    let result = tokio::task::spawn_blocking(move || tool_dispatch::execute_http_tool(&call, &url))
        .await
        .unwrap_or_else(|error| tool_dispatch::ToolResult::failed(&tc.function.name, &error));
    let data = AgenticToolCallData::Custom {
        arguments: String::new(),
        content: result.content.clone(),
    };
    ToolOutcome::text(result.content, data)
}

/// What answers a call.
enum Dispatcher<'a> {
    ReadFile,
    ListFiles,
    Search(&'a WebSearchOptions),
    Extract(&'a WebSearchOptions),
    Custom,
    Http(&'a str),
}

/// `None` when nothing here can answer `name`.
fn dispatcher<'a>(ctx: &DispatchCtx<'a>, name: &str) -> Option<Dispatcher<'a>> {
    if is_read_file_tool(name) {
        return Some(Dispatcher::ReadFile);
    }
    if is_list_files_tool(name) {
        return Some(Dispatcher::ListFiles);
    }
    if search::search_tool_called(name) {
        let opts = ctx.web_search_options?;
        return Some(if name == search::SEARCH_TOOL_NAME {
            Dispatcher::Search(opts)
        } else {
            Dispatcher::Extract(opts)
        });
    }
    if ctx.engine.tool_callbacks().contains_key(name) {
        return Some(Dispatcher::Custom);
    }
    ctx.dispatch_url.map(Dispatcher::Http)
}

async fn dispatch_tool(
    ctx: &DispatchCtx<'_>,
    dispatcher: Dispatcher<'_>,
    tc: &ToolCallResponse,
    round: usize,
) -> ToolOutcome {
    let store = ctx.engine.file_store();
    match dispatcher {
        Dispatcher::ReadFile => ToolOutcome::custom(do_read_file(tc, store, ctx.owner)),
        Dispatcher::ListFiles => {
            ToolOutcome::custom(do_list_files(store, ctx.session_id, ctx.owner))
        }
        Dispatcher::Search(opts) => do_search(ctx.engine, tc, opts).await,
        Dispatcher::Extract(opts) => do_extraction(ctx.engine, tc, opts).await,
        Dispatcher::Custom => do_custom_tool(ctx, tc, round).await,
        Dispatcher::Http(url) => do_http_tool(tc, url).await,
    }
}

async fn run_call(
    ctx: &DispatchCtx<'_>,
    tc: &ToolCallResponse,
    dispatcher: Dispatcher<'_>,
    approval: AgentToolApprovalDecision,
    round: usize,
) -> ToolOutcome {
    if approval.approve {
        dispatch_tool(ctx, dispatcher, tc, round).await
    } else {
        let message = approval.message.unwrap_or_else(|| DENIED.to_string());
        denied_tool_result(tc, message)
    }
}

fn shares_sandbox(name: &str) -> bool {
    let python = is_code_exec_tool(name) && !is_read_file_tool(name) && !is_list_files_tool(name);
    python || is_shell_tool(name) || is_surface_outputs_tool(name)
}

fn progress(round: usize, tc: &ToolCallResponse, phase: AgenticToolCallPhase) -> Response {
    Response::AgenticToolCallProgress {
        round,
        tool_call_id: tc.id.clone(),
        tool_name: tc.function.name.clone(),
        phase,
    }
}

/// Runs every call of a round at once and appends them with their answers; `None` hands the round to the client.
async fn run_round(
    ctx: &DispatchCtx<'_>,
    mut request: NormalRequest,
    calls: &[ToolCallResponse],
    round: usize,
    reasoning: Option<&str>,
) -> Option<NormalRequest> {
    // One call nothing here can run makes the whole round the client's, so its results arrive together.
    let dispatchers = calls
        .iter()
        .map(|tc| dispatcher(ctx, &tc.function.name))
        .collect::<Option<Vec<_>>>()?;
    for tc in calls {
        let calling = AgenticToolCallPhase::Calling(calling_data_for_tool(tc));
        let _ = ctx.user_sender.send(progress(round, tc, calling)).await;
    }
    tokio::task::yield_now().await;

    // One at a time, since each may wait on the client's answer.
    let mut approvals = Vec::with_capacity(calls.len());
    for tc in calls {
        approvals.push(approve_agent_tool(ctx, tc, round).await);
    }
    let (in_order, alongside): (Vec<_>, Vec<_>) = calls
        .iter()
        .zip(dispatchers)
        .zip(approvals)
        .enumerate()
        .partition(|(_, ((tc, _), _))| shares_sandbox(&tc.function.name));
    // Calls into the session's sandbox keep the model's order, since one may build on what another left there.
    let sequential = async {
        let mut done = Vec::with_capacity(in_order.len());
        for (index, ((tc, dispatcher), approval)) in in_order {
            done.push((index, run_call(ctx, tc, dispatcher, approval, round).await));
        }
        done
    };
    let concurrent = futures::future::join_all(alongside.into_iter().map(
        |(index, ((tc, dispatcher), approval))| async move {
            (index, run_call(ctx, tc, dispatcher, approval, round).await)
        },
    ));
    let (sequential, concurrent) = futures::join!(sequential, concurrent);
    let mut outcomes: Vec<_> = sequential.into_iter().chain(concurrent).collect();
    outcomes.sort_by_key(|(index, _)| *index);
    let outcomes = outcomes.into_iter().map(|(_, outcome)| outcome);

    let messages = request.chat_messages_mut();
    append_assistant_tool_calls(messages, calls);
    attach_reasoning_to_latest_assistant_tool_call(messages, reasoning);
    let completed: Vec<_> = calls
        .iter()
        .zip(outcomes)
        .map(|(tc, outcome)| (tc, append_tool_outcome(&mut request, tc, outcome, ctx)))
        .collect();
    request.tool_choice = Some(ToolChoice::Auto);

    for (tc, (data, files)) in completed {
        emit_files(
            ctx.engine,
            ctx.session_id,
            ctx.owner,
            files,
            ctx.user_sender,
        )
        .await;
        let complete = AgenticToolCallPhase::Complete(data);
        let _ = ctx.user_sender.send(progress(round, tc, complete)).await;
    }
    Some(request)
}

/// Store full file bodies and emit wire-elided clones on the user channel. Truncated bodies stay fetchable via the store.
async fn emit_files(
    engine: &Engine,
    session_id: &str,
    owner: Option<&str>,
    files: Vec<File>,
    sender: &tokio::sync::mpsc::Sender<Response>,
) {
    for f in files {
        let wire = f.elide_for_wire();
        engine
            .file_store()
            .insert(f, Some(session_id.to_string()), owner);
        let _ = sender.send(Response::File(wire)).await;
    }
}

/// A streamed run's last chunk (its held final, or the tool call it stopped on), with the run's usage and session.
fn stopped_round_final_chunk(
    held: Option<inference_core::ChatCompletionChunkResponse>,
    usage: Option<Usage>,
    session_id: &str,
) -> Option<inference_core::ChatCompletionChunkResponse> {
    let mut chunk = held?;
    if usage.is_some() {
        chunk.usage = usage;
    }
    chunk.session_id = Some(session_id.to_string());
    Some(chunk)
}

/// Drive tool-use rounds (search, code exec, custom tools) without recursion. Forwards every reply except the first probe.
pub(crate) async fn agentic_loop(this: Arc<Engine>, mut request: NormalRequest) {
    let web_search_options = request.web_search_options.clone();
    let shell_options = request.shell_options.clone();
    let dispatch_url = request.tool_dispatch_url.clone();
    let code_execution_permission = request.code_execution_permission;
    let code_execution_approval_notifier = request.code_execution_approval_notifier.clone();
    let agent_permission = request.agent_permission.unwrap_or_default();
    let agent_approval_handler = request.agent_approval_handler.clone();
    let agent_approval_notifier = request.agent_approval_notifier.clone();
    let required_files: Vec<RequestedFile> = request.files.clone().unwrap_or_default();
    let input_files = request.input_files.clone();

    let mut session_id = request
        .session_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let owner = request.owner.clone();
    let in_use = {
        let mut store = this.session_store().lock().unwrap();
        let in_use = store.held_by_other(&session_id, owner.as_deref());
        let existing = if request.session_id.is_some() {
            store
                .get(&session_id, owner.as_deref())
                .map(|e| (session_id.clone(), e))
        } else {
            let msgs = request.chat_messages();
            store.find_by_messages(msgs, owner.as_deref())
        };
        if let Some((matched_id, entry)) = existing {
            session_id = matched_id;
            inference_core::agentic_session::splice_session_into_request(&mut request, &entry);
        }
        in_use
    };
    // another owner's session: the id can be neither read nor taken over
    if in_use {
        let message = format!("session_id `{session_id}` is in use");
        let _ = request
            .response
            .send(Response::ValidationError(message.into()))
            .await;
        return;
    }

    for file in &input_files {
        this.file_store()
            .insert(file.clone(), Some(session_id.clone()), owner.as_deref());
    }
    this.file_store().touch_session(&session_id);

    let turn = count_user_messages(&request);
    inject_shell_skills_message(&mut request);
    request.inject_input_files_message();

    let modalities = this.modalities();
    let supports_vision = modalities.input.contains(&SupportedModality::Vision);
    let supports_video = modalities.input.contains(&SupportedModality::Video);

    let user_sender = request.response.clone();
    let is_streaming = request.is_streaming;

    let mut probe = request.clone();
    if let Some(ref opts) = web_search_options {
        probe
            .tools
            .get_or_insert_with(Vec::new)
            .extend(search::get_search_tools(opts).unwrap());
    }

    if let Some(user_tools) = &probe.tools {
        for t in user_tools {
            if this.tool_callbacks().contains_key(&t.function.name) {
                let _ = user_sender
                    .send(Response::ValidationError(
                        format!(
                            "Tool '{}' conflicts with a registered internal tool. \
                             Internal tool names cannot be overridden.",
                            t.function.name
                        )
                        .into(),
                    ))
                    .await;
                return;
            }
        }
    }

    if !this.tool_callbacks().is_empty() {
        let tools = probe.tools.get_or_insert_with(Vec::new);

        for (name, callback_with_tool) in this.tool_callbacks() {
            if is_shell_tool(name) && !probe.enable_shell {
                continue;
            }
            if is_code_exec_tool(name)
                && !is_read_file_tool(name)
                && !is_list_files_tool(name)
                && !probe.enable_code_execution
            {
                continue;
            }
            if !tools.iter().any(|t| t.function.name == *name) {
                tools.push(callback_with_tool.tool.clone());
            }
        }
    }

    if CODE_EXECUTION && !input_files.is_empty() {
        let tools = probe.tools.get_or_insert_with(Vec::new);
        if !tools
            .iter()
            .any(|t| t.function.name == inference_code_exec::READ_FILE_TOOL_NAME)
        {
            tools.push(inference_code_exec::build_read_file_tool());
        }
        if !tools
            .iter()
            .any(|t| t.function.name == inference_code_exec::LIST_FILES_TOOL_NAME)
        {
            tools.push(inference_code_exec::build_list_files_tool());
        }
    }

    if let Some(addendum) = required_files_tool_addendum(&required_files)
        && let Some(tools) = probe.tools.as_mut()
    {
        for t in tools.iter_mut() {
            if is_code_exec_tool(&t.function.name) || is_shell_tool(&t.function.name) {
                let desc = t.function.description.get_or_insert_with(String::new);
                desc.push_str(&addendum);
            }
        }
    }

    if probe.tool_choice.is_none() {
        probe.tool_choice = Some(ToolChoice::Auto);
    }
    probe.web_search_options = None;

    let mut visible_req = probe.clone();
    visible_req.response = user_sender.clone();

    let this_clone = this.clone();
    let handle = tokio::spawn(async move {
        let tool_call_ctx = inference_mcp::ToolCallContext {
            session_id: Some(session_id.clone()),
            owner: owner.clone(),
            round: None,
            tool_name: None,
            agent_permission: Some(agent_permission),
            agent_approval_notifier,
            code_execution_permission,
            code_execution_approval_notifier,
            shell_options,
            input_files: input_files.iter().map(file_to_tool_input_file).collect(),
        };
        let dispatch_ctx = DispatchCtx {
            engine: &this_clone,
            user_sender: &user_sender,
            web_search_options: web_search_options.as_ref(),
            dispatch_url: dispatch_url.as_deref(),
            supports_vision,
            supports_video,
            tool_call_ctx: &tool_call_ctx,
            owner: owner.as_deref(),
            turn,
            session_id: &session_id,
            required_files: &required_files,
            agent_permission,
            agent_approval_handler,
        };

        let cancellation = probe.cancellation.clone();
        let mut current = probe;
        let max_rounds = current.max_tool_rounds.unwrap_or(DEFAULT_MAX_TOOL_ROUNDS);
        let mut round = 0;
        let mut usage_accumulator = AgenticUsageAccumulator::default();

        loop {
            let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
            current.response = sender;

            // Prevent the inner probe from re-entering the agentic loop or being rejected
            // by the files-without-agentic-surface guard in `add_request`.
            current.web_search_options = None;
            current.enable_code_execution = false;
            current.enable_shell = false;
            current.shell_options = None;
            current.max_tool_rounds = AGENTIC_LOOP_REENTRY_SENTINEL;
            current.tool_dispatch_url = None;
            current.files = None;
            current.input_files = Vec::new();
            let _ = this_clone
                .request_sender()
                .send(Request::Normal(Box::new(current)))
                .await;

            if !is_streaming {
                let resp = match recv_or_client_disconnect(&mut receiver, &user_sender).await {
                    Ok(Some(resp)) => resp,
                    Ok(None) => {
                        tracing::warn!("Engine closed without sending a response.");
                        return;
                    }
                    Err(_) => return,
                };
                let Some(resp) = forward_passthrough(resp, &user_sender).await else {
                    return;
                };
                let done = match resp {
                    Response::Done(done) => done,
                    _ => {
                        let _ = user_sender.send(resp).await;
                        return;
                    }
                };
                usage_accumulator.add(&done.usage);

                let calls = done.choices[0]
                    .message
                    .tool_calls
                    .clone()
                    .filter(|calls| !calls.is_empty());

                let canceled = cancellation.as_ref().is_some_and(|c| c.is_canceled());
                if calls.is_none() || round >= max_rounds || canceled {
                    save_session(&this_clone, &session_id, &visible_req);
                    let mut final_resp = done.clone();
                    if canceled {
                        for choice in &mut final_resp.choices {
                            choice.finish_reason = FINISH_REASON_CANCELED.to_string();
                            choice.message.tool_calls = None;
                        }
                    }
                    if let Some(usage) = usage_accumulator.aggregate() {
                        final_resp.usage = usage;
                    }
                    final_resp.session_id = Some(session_id.clone());
                    let _ = user_sender.send(Response::Done(final_resp)).await;
                    return;
                }

                let calls = calls.unwrap();
                let reasoning = done.choices[0].message.reasoning_content.as_deref();
                let next = run_round(&dispatch_ctx, visible_req.clone(), &calls, round, reasoning);
                let Some(next_visible) = next.await else {
                    save_session(&this_clone, &session_id, &visible_req);
                    let mut final_resp = done.clone();
                    if let Some(usage) = usage_accumulator.aggregate() {
                        final_resp.usage = usage;
                    }
                    final_resp.session_id = Some(session_id.clone());
                    let _ = user_sender.send(Response::Done(final_resp)).await;
                    return;
                };
                round += 1;

                visible_req = next_visible;
                visible_req.response = user_sender.clone();
                current = visible_req.clone();
            } else {
                // Hold the finish-reason chunk so we can stamp the session ID on it if this is the final round.
                let mut last_choice = None;
                let mut held_final_chunk: Option<inference_core::ChatCompletionChunkResponse> =
                    None;
                let mut tool_call_final_chunk = None;
                let mut round_reasoning_content = String::new();

                loop {
                    let resp = match recv_or_client_disconnect(&mut receiver, &user_sender).await {
                        Ok(Some(resp)) => resp,
                        Ok(None) => break,
                        Err(_) => return,
                    };
                    let Some(resp) = forward_passthrough(resp, &user_sender).await else {
                        return;
                    };
                    match resp {
                        Response::Chunk(chunk) => {
                            // Suppress tool-call chunks. Forwarding them would surface a premature finish_reason before the tool loop continues.
                            let first_choice = &chunk.choices[0];
                            if let Some(reasoning_content) = &first_choice.delta.reasoning_content {
                                round_reasoning_content.push_str(reasoning_content);
                            }
                            let is_final = first_choice.finish_reason.is_some();
                            if is_final && let Some(usage) = &chunk.usage {
                                usage_accumulator.add(usage);
                            }
                            if first_choice.delta.tool_calls.is_none() {
                                if is_final {
                                    held_final_chunk = Some(chunk.clone());
                                } else {
                                    let _ = user_sender.send(Response::Chunk(chunk.clone())).await;
                                }
                            } else if is_final {
                                tool_call_final_chunk = Some(chunk.clone());
                            }
                            last_choice = Some(first_choice.clone());

                            if is_final {
                                break;
                            }
                        }
                        other => {
                            let _ = user_sender.send(other).await;
                            return;
                        }
                    }
                }

                let Some(choice) = last_choice else {
                    tracing::warn!("Engine closed without sending any chunks.");
                    save_session(&this_clone, &session_id, &visible_req);
                    break;
                };

                let calls = choice
                    .delta
                    .tool_calls
                    .clone()
                    .filter(|calls| !calls.is_empty());

                let canceled = cancellation.as_ref().is_some_and(|c| c.is_canceled());
                if canceled {
                    // A tool call that finished on the canceled token is dropped, not run.
                    held_final_chunk =
                        held_final_chunk
                            .or(tool_call_final_chunk.take())
                            .map(|mut chunk| {
                                for choice in &mut chunk.choices {
                                    choice.finish_reason = Some(FINISH_REASON_CANCELED.to_string());
                                    choice.delta.tool_calls = None;
                                }
                                chunk
                            });
                }
                if calls.is_none() || round >= max_rounds || canceled {
                    save_session(&this_clone, &session_id, &visible_req);
                    let usage = usage_accumulator.aggregate();
                    let held = held_final_chunk.or(tool_call_final_chunk);
                    if let Some(terminal) = stopped_round_final_chunk(held, usage, &session_id) {
                        let _ = user_sender.send(Response::Chunk(terminal)).await;
                    }
                    break;
                }

                let calls = calls.unwrap();
                let reasoning = Some(round_reasoning_content.as_str());
                let next = run_round(&dispatch_ctx, visible_req.clone(), &calls, round, reasoning);
                let Some(next_visible) = next.await else {
                    save_session(&this_clone, &session_id, &visible_req);
                    let usage = usage_accumulator.aggregate();
                    let held = tool_call_final_chunk;
                    if let Some(terminal) = stopped_round_final_chunk(held, usage, &session_id) {
                        let _ = user_sender.send(Response::Chunk(terminal)).await;
                    }
                    break;
                };
                round += 1;

                visible_req = next_visible;
                visible_req.response = user_sender.clone();
                current = visible_req.clone();
            }
        }
    });

    this.track_task(handle);
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_core::{CalledFunction, ToolCallType};

    #[tokio::test]
    async fn client_disconnect_drops_the_internal_response_bridge() {
        let (user_sender, user_receiver) = tokio::sync::mpsc::channel(1);
        let (internal_sender, mut internal_receiver) = tokio::sync::mpsc::channel::<usize>(1);

        let task = tokio::spawn(async move {
            recv_or_client_disconnect(&mut internal_receiver, &user_sender).await
        });
        tokio::task::yield_now().await;
        assert!(!internal_sender.is_closed());

        drop(user_receiver);
        assert_eq!(task.await.unwrap(), Err(ClientDisconnected));
        assert!(internal_sender.is_closed());
    }

    #[test]
    fn assistant_tool_call_history_preserves_reasoning() {
        let mut messages = Vec::new();
        let tool_call = ToolCallResponse {
            index: 0,
            id: "call-1".to_string(),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: "get_weather".to_string(),
                arguments: r#"{"city":"Paris"}"#.to_string(),
            },
        };
        append_assistant_tool_calls(&mut messages, std::slice::from_ref(&tool_call));

        attach_reasoning_to_latest_assistant_tool_call(&mut messages, Some("Need weather"));

        assert_eq!(
            messages[0].get("reasoning_content"),
            Some(&Either::Left("Need weather".to_string()))
        );
    }

    #[test]
    fn a_round_stopped_on_its_tool_call_ends_the_stream_with_that_call() {
        let tool_call = ToolCallResponse {
            index: 0,
            id: "call-1".to_string(),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: "lookup".to_string(),
                arguments: "{}".to_string(),
            },
        };
        let chunk = inference_core::ChatCompletionChunkResponse {
            id: "chunk".to_string(),
            choices: vec![inference_core::ChunkChoice {
                finish_reason: Some("tool_calls".to_string()),
                stop_sequence: None,
                index: 0,
                delta: inference_core::Delta {
                    content: None,
                    role: "assistant".to_string(),
                    tool_calls: Some(vec![tool_call]),
                    reasoning_content: None,
                },
                logprobs: None,
            }],
            created: 0,
            model: "model".to_string(),
            system_fingerprint: "local".to_string(),
            object: "chat.completion.chunk".to_string(),
            usage: None,
            adapter_generation: None,
            session_id: None,
        };
        let mut usage = AgenticUsageAccumulator::default();
        usage.add(&Usage {
            completion_tokens: 3,
            prompt_tokens: 5,
            total_tokens: 8,
            prompt_tokens_details: None,
            avg_tok_per_sec: 0.0,
            avg_prompt_tok_per_sec: 0.0,
            avg_compl_tok_per_sec: 0.0,
            total_time_sec: 0.0,
            total_prompt_time_sec: 0.0,
            total_completion_time_sec: 0.0,
        });

        // No text chunk was held: the tool-call chunk is what ends the run.
        let last = stopped_round_final_chunk(Some(chunk), usage.aggregate(), "session").unwrap();
        assert_eq!(last.choices[0].finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(
            last.choices[0].delta.tool_calls.as_ref().unwrap()[0]
                .function
                .name,
            "lookup"
        );
        assert_eq!(last.usage.unwrap().completion_tokens, 3);
        assert_eq!(last.session_id.as_deref(), Some("session"));
        assert!(stopped_round_final_chunk(None, None, "session").is_none());
    }
}
