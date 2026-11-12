//! The tool loop: send the conversation, stream the reply, run the tools it calls, repeat until it answers.

use std::pin::pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Result, bail};
use futures::StreamExt;
use futures::future::BoxFuture;
use serde_json::{Map, Value, json};
use tokio::sync::Notify;

use crate::backend::ChatBackend;
use crate::changes::Changes;
use crate::permission::{Approval, Decision, Permissions};
use crate::session::{
    Compaction, SUMMARY_ACK, Session, ToolImage, is_tool_images, summary_message,
    tool_images_message, with_image_data, without_image_files,
};
use crate::tools::{Tool, ToolContext, ToolKind, ToolOutcome};
use crate::turn::{ToolCall, Turn};

// a runaway loop stops here; a real task rarely needs a tenth of it
pub const DEFAULT_MAX_ROUNDS: usize = 200;
// a reply that runs past this is cut: without a cap a looping reply fills the whole context (Qwen3's suggested length)
pub const DEFAULT_MAX_REPLY_TOKENS: usize = 32_768;
const LENGTH_FINISH: &str = "length";
const REPLY_CUT: &str = "the reply was cut at the token limit";
const INTERRUPTED: &str = "interrupted by the user";
const DECLINED: &str = "the user declined this call";
/// What a front end's user types to fold the session's history into a summary now.
pub const COMPACT_COMMAND: &str = "/compact";
// past this share of the model's context a round folds the older history into a summary first
const COMPACT_AT: f64 = 0.8;
// the recent history a fold keeps word for word, and the most one summary request reads, as shares of the context
const KEEP_SHARE: f64 = 0.1;
const CHUNK_SHARE: f64 = 0.5;
const SUMMARY_TOKENS: usize = 8192;
// a summary request shows a tool result's head and tail only past this many characters
const CLIPPED_RESULT_CHARS: usize = 16_000;
const CLIP_MARK: &str = "\n[... clipped ...]\n";
// JSON characters per prompt token until a reply's usage says (about what code and tool output run at)
const DEFAULT_CHARS_PER_TOKEN: f64 = 3.0;
// an image counts as this many characters instead of its encoded bytes (about 1.4K tokens at the default ratio)
const IMAGE_CHARS: usize = 4096;
// a summary request the server still refuses is retried over half as much history, this many times
const SUMMARY_RETRIES: usize = 3;
const COMPACT_PROMPT: &str = "Write a summary of this session so far for yourself: the earlier messages are about \
to leave your context, and you will continue the work from the summary and the most recent messages. Keep the \
user's goal and every instruction or preference they gave; what has been done and found (files, functions, \
commands, results, numbers); decisions made and why; what failed and why; the current state and the next steps. \
If an earlier summary opens the conversation, fold it in. Be specific (paths, names, values) and leave out \
pleasantries. Write only the summary.";

/// What the agent shows and asks while it works.
pub trait AgentUi: Send {
    fn text(&mut self, delta: &str);
    fn reasoning(&mut self, delta: &str);
    fn tool_start(&mut self, call: &ToolCall, args: &Value);
    fn tool_end(&mut self, call: &ToolCall, outcome: &ToolOutcome);
    fn approve<'a>(
        &'a mut self,
        call: &'a ToolCall,
        args: &'a Value,
        kind: ToolKind,
    ) -> BoxFuture<'a, Approval>;
    /// A request of about `prompt_tokens` (the harness's estimate) is going to the model.
    fn round_start(&mut self, prompt_tokens: usize) {
        let _ = prompt_tokens;
    }
    /// One model reply finished; its usage says how full the context is.
    fn round_end(&mut self, turn: &Turn) {
        let _ = turn;
    }
    /// The history up to message `end` is being folded into a summary.
    fn compacting(&mut self, end: usize) {
        let _ = end;
    }
    /// Text of the summary being written, as it streams.
    fn summary_text(&mut self, delta: &str) {
        let _ = delta;
    }
    /// One summary request finished (a long history takes several); its usage says how fast it ran.
    fn summary_done(&mut self, turn: &Turn) {
        let _ = turn;
    }
    /// It was; the model now sees the summary and the messages from `compaction.start`, about `context_tokens` in all.
    fn compacted(&mut self, compaction: &Compaction, context_tokens: usize) {
        let _ = (compaction, context_tokens);
    }
}

/// A Ctrl-C the UI raises and the loop answers at its next await.
#[derive(Clone, Default)]
pub struct Interrupt(Arc<(AtomicBool, Notify)>);

impl Interrupt {
    pub fn trigger(&self) {
        self.0.0.store(true, Ordering::SeqCst);
        self.0.1.notify_waiters();
    }

    pub fn reset(&self) {
        self.0.0.store(false, Ordering::SeqCst);
    }

    pub fn is_set(&self) -> bool {
        self.0.0.load(Ordering::SeqCst)
    }

    async fn wait(&self) {
        loop {
            let mut notified = pin!(self.0.1.notified());
            notified.as_mut().enable();
            if self.is_set() {
                return;
            }
            notified.await;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnEnd {
    Done,
    Interrupted,
    RoundLimit,
    /// A reply reached the per-reply token cap, usually a model stuck repeating itself.
    ReplyLimit,
}

pub struct AgentConfig {
    /// The model name requests carry.
    pub model: String,
    pub max_rounds: usize,
    /// The most tokens one model reply may take, reasoning included.
    pub max_reply_tokens: usize,
    /// Extra request fields (sampling, `enable_thinking`, `reasoning_effort`), the same on every request.
    pub request: Map<String, Value>,
}

pub struct Agent {
    backend: Arc<dyn ChatBackend>,
    tools: Vec<Arc<dyn Tool>>,
    ctx: ToolContext,
    pub permissions: Permissions,
    pub session: Session,
    config: AgentConfig,
    changes: Arc<Mutex<Changes>>,
    // the model's context and whether it takes images, asked of the backend once per model
    window: Option<(String, Option<usize>)>,
    vision: Option<(String, bool)>,
    chars_per_token: f64,
}

impl Agent {
    pub fn new(
        backend: Arc<dyn ChatBackend>,
        tools: Vec<Arc<dyn Tool>>,
        permissions: Permissions,
        session: Session,
        config: AgentConfig,
    ) -> Self {
        Self {
            changes: Arc::new(Mutex::new(Changes::open(&session.root, &session.path))),
            backend,
            tools,
            ctx: ToolContext {
                root: session.root.clone(),
                images: false,
            },
            permissions,
            session,
            config,
            window: None,
            vision: None,
            chars_per_token: DEFAULT_CHARS_PER_TOKEN,
        }
    }

    /// What this session has changed in its project, readable while a prompt runs.
    pub fn changes(&self) -> Arc<Mutex<Changes>> {
        self.changes.clone()
    }

    /// The model later requests name, for a front end whose model selection can change between prompts.
    pub fn set_model(&mut self, model: impl Into<String>) {
        self.config.model = model.into();
    }

    /// Sets (or with `None` removes) an extra field every later request carries.
    pub fn set_request_field(&mut self, key: &str, value: Option<Value>) {
        match value {
            Some(value) => self.config.request.insert(key.to_string(), value),
            None => self.config.request.remove(key),
        };
    }

    pub fn backend(&self) -> &dyn ChatBackend {
        self.backend.as_ref()
    }

    /// The backend, for a request a front end sends beside the agent's own, such as an activity label.
    pub fn shared_backend(&self) -> Arc<dyn ChatBackend> {
        self.backend.clone()
    }

    /// Runs one user prompt (text, or content parts from [`crate::session::user_content`]) to its answer. Messages
    /// reach the session a round at a time, each assistant reply with its tool results, so the history on disk never
    /// holds a call without a result.
    pub async fn send(
        &mut self,
        prompt: impl Into<Value>,
        ui: &mut dyn AgentUi,
        interrupt: &Interrupt,
    ) -> Result<TurnEnd> {
        let mut pending = vec![json!({"role": "user", "content": prompt.into()})];
        self.ctx.images = self.reads_images().await;
        for _ in 0..self.config.max_rounds {
            self.compact_if_full(&pending, ui).await?;
            let sent = self
                .session
                .context()
                .iter()
                .chain(&pending)
                .map(message_chars)
                .sum::<usize>();
            ui.round_start((sent as f64 / self.chars_per_token) as usize);
            let (turn, interrupted) = self.stream_turn(&pending, ui, interrupt).await?;
            self.learn_ratio(&turn, sent);
            ui.round_end(&turn);
            if interrupted {
                let mut message = turn.message();
                if let Some(message) = message.as_object_mut() {
                    message.remove("tool_calls");
                }
                if turn.content.is_empty() {
                    message["content"] = Value::String(format!("({INTERRUPTED})"));
                }
                pending.push(message);
                self.commit(&mut pending)?;
                return Ok(TurnEnd::Interrupted);
            }
            if turn.finish_reason.as_deref() == Some(LENGTH_FINISH) {
                // a cut call's arguments are partial, and a cut reasoning loop would only crowd the context
                let mut message = turn.message();
                if let Some(message) = message.as_object_mut() {
                    message.remove("tool_calls");
                    message.remove("reasoning_content");
                }
                if turn.content.trim().is_empty() {
                    message["content"] = Value::String(format!("({REPLY_CUT})"));
                }
                pending.push(message);
                self.commit(&mut pending)?;
                return Ok(TurnEnd::ReplyLimit);
            }
            pending.push(turn.message());
            if turn.tool_calls.is_empty() {
                self.commit(&mut pending)?;
                return Ok(TurnEnd::Done);
            }
            let mut images = Vec::new();
            for call in &turn.tool_calls {
                let outcome = if interrupt.is_set() {
                    ToolOutcome::error(INTERRUPTED)
                } else {
                    self.run_call(call, ui, interrupt).await
                };
                pending.push(
                    json!({"role": "tool", "tool_call_id": call.id, "content": outcome.text}),
                );
                images.extend(outcome.images.into_iter().map(|path| ToolImage {
                    call_id: call.id.clone(),
                    shown_as: self.ctx.display(&path),
                    path,
                }));
            }
            // tool results are text, so the images they returned follow as a turn of their own
            if !images.is_empty() {
                pending.push(tool_images_message(&images));
            }
            self.commit(&mut pending)?;
            if interrupt.is_set() {
                return Ok(TurnEnd::Interrupted);
            }
        }
        Ok(TurnEnd::RoundLimit)
    }

    /// Folds the history before the latest user prompt into a summary (as much of it as the context needs, when the
    /// recent part alone is too long); false when there was nothing before it to fold.
    pub async fn compact(&mut self, ui: &mut dyn AgentUi) -> Result<bool> {
        let start = self.session.context_start();
        let last_prompt = (start..self.session.messages.len())
            .rev()
            .find(|&i| {
                let message = &self.session.messages[i];
                message["role"] == "user" && !is_tool_images(message)
            })
            .unwrap_or(start);
        let window = self.window().await;
        let end = match window {
            Some(window) => last_prompt.max(self.fold_end(share(window, KEEP_SHARE))),
            None => last_prompt,
        };
        if end <= start {
            return Ok(false);
        }
        self.fold(end, window, ui).await?;
        Ok(true)
    }

    // past COMPACT_AT of the context, the older history is folded down to KEEP_SHARE
    async fn compact_if_full(&mut self, pending: &[Value], ui: &mut dyn AgentUi) -> Result<()> {
        let Some(window) = self.window().await else {
            return Ok(());
        };
        let context = self.session.context();
        let used = self.tokens(context.iter().chain(pending));
        if used < share(window, COMPACT_AT) {
            return Ok(());
        }
        let end = self.fold_end(share(window, KEEP_SHARE));
        if end > self.session.context_start() {
            self.fold(end, Some(window), ui).await?;
        }
        Ok(())
    }

    /// The most tokens a prompt to the current model may hold, where the backend can tell.
    pub async fn context_window(&mut self) -> Option<usize> {
        self.window().await
    }

    /// About how many tokens the model's view of the session holds now, by the harness's own count.
    pub fn context_estimate(&self) -> usize {
        self.tokens(&self.session.context())
    }

    async fn reads_images(&mut self) -> bool {
        match &self.vision {
            Some((model, vision)) if *model == self.config.model => *vision,
            _ => {
                let vision = self.backend.reads_images(&self.config.model).await;
                self.vision = Some((self.config.model.clone(), vision));
                vision
            }
        }
    }

    async fn window(&mut self) -> Option<usize> {
        match &self.window {
            Some((model, window)) if *model == self.config.model => *window,
            _ => {
                let window = self.backend.context_tokens(&self.config.model).await;
                self.window = Some((self.config.model.clone(), window));
                window
            }
        }
    }

    fn tokens<'a>(&self, messages: impl IntoIterator<Item = &'a Value>) -> usize {
        let chars = messages.into_iter().map(message_chars).sum::<usize>();
        (chars as f64 / self.chars_per_token) as usize
    }

    // where a run of tool results ends and the next message can open the kept history
    fn boundaries(&self) -> impl Iterator<Item = usize> + '_ {
        (self.session.context_start() + 1..self.session.messages.len())
            .filter(|&i| self.session.messages[i]["role"] != "tool")
    }

    // the earliest boundary whose history from there fits `keep` tokens: the messages before it fold
    fn fold_end(&self, keep: usize) -> usize {
        let messages = &self.session.messages;
        self.boundaries()
            .find(|&i| self.tokens(&messages[i..]) <= keep)
            .unwrap_or(messages.len())
    }

    // folds messages[context_start..end] a chunk at a time, each chunk's summary taking in the one before
    async fn fold(
        &mut self,
        end: usize,
        window: Option<usize>,
        ui: &mut dyn AgentUi,
    ) -> Result<()> {
        ui.compacting(end);
        let mut summary = self.session.compaction.as_ref().map(|c| c.summary.clone());
        let mut cursor = self.session.context_start();
        while cursor < end {
            let mut budget = window.map(|w| share(w, CHUNK_SHARE));
            let mut attempt = 0;
            let (next, text) = loop {
                let next = self.chunk_end(cursor, end, summary.as_deref(), budget);
                match self.summarize(summary.as_deref(), cursor, next, ui).await {
                    Ok(text) => break (next, text),
                    Err(_) if attempt < SUMMARY_RETRIES && budget.is_some() => {
                        budget = budget.map(|b| b / 2);
                        attempt += 1;
                    }
                    Err(error) => return Err(error.context("summarizing the earlier history")),
                }
            };
            summary = Some(text);
            cursor = next;
        }
        let summary = summary.unwrap_or_default();
        self.session.compact(summary, end)?;
        if let Some(compaction) = &self.session.compaction {
            ui.compacted(compaction, self.context_estimate());
        }
        Ok(())
    }

    // the furthest boundary up to `end` whose chunk fits `budget`, at least the first round past `cursor`
    fn chunk_end(
        &self,
        cursor: usize,
        end: usize,
        summary: Option<&str>,
        budget: Option<usize>,
    ) -> usize {
        let Some(budget) = budget else {
            return end;
        };
        let base = summary.map_or(0, |s| self.tokens([&summary_message(s)]));
        let messages = &self.session.messages;
        let mut last = None;
        for i in (cursor + 1..=end).filter(|&i| i == end || messages[i]["role"] != "tool") {
            let chunk = messages[cursor..i].iter().map(clipped);
            if base + self.tokens(chunk.collect::<Vec<_>>().iter()) > budget && last.is_some() {
                break;
            }
            last = Some(i);
        }
        last.unwrap_or(end)
    }

    async fn summarize(
        &mut self,
        summary: Option<&str>,
        from: usize,
        to: usize,
        ui: &mut dyn AgentUi,
    ) -> Result<String> {
        let mut messages = self
            .session
            .messages
            .iter()
            .take(1)
            .cloned()
            .collect::<Vec<_>>();
        if let Some(summary) = summary {
            messages.push(summary_message(summary));
        }
        let chunk = &self.session.messages[from..to];
        // two user turns in a row: the summary's and the chunk's first
        if summary.is_some() && chunk.first().is_some_and(|m| m["role"] == "user") {
            messages.push(json!({"role": "assistant", "content": SUMMARY_ACK}));
        }
        messages.extend(chunk.iter().map(clipped));
        messages.push(json!({"role": "user", "content": COMPACT_PROMPT}));
        let chars = messages.iter().map(message_chars).sum::<usize>();
        let mut request = self.config.request.clone();
        request.insert("model".into(), Value::String(self.config.model.clone()));
        request.insert("messages".into(), Value::Array(messages));
        request.insert("max_tokens".into(), json!(SUMMARY_TOKENS));
        request.insert("enable_thinking".into(), json!(false));
        request.insert("stream".into(), json!(true));
        let mut stream = self.backend.stream(Value::Object(request)).await?;
        let mut turn = Turn::default();
        while let Some(chunk) = stream.next().await {
            if let Some(text) = turn.apply(&chunk?).content {
                ui.summary_text(&text);
            }
        }
        ui.summary_done(&turn);
        self.learn_ratio(&turn, chars);
        let text = turn.content.trim();
        if text.is_empty() {
            bail!("the model wrote an empty summary");
        }
        Ok(text.to_string())
    }

    // characters per token, from what the server counted for a request of `chars` characters
    fn learn_ratio(&mut self, turn: &Turn, chars: usize) {
        let prompt = turn
            .usage
            .as_ref()
            .and_then(|u| u["prompt_tokens"].as_u64())
            .filter(|&p| p > 0);
        if let Some(prompt) = prompt {
            self.chars_per_token = chars as f64 / prompt as f64;
        }
    }

    fn commit(&mut self, pending: &mut Vec<Value>) -> Result<()> {
        for message in pending.drain(..) {
            self.session.append(message)?;
        }
        Ok(())
    }

    fn request(&self, pending: &[Value]) -> Value {
        let mut request = self.config.request.clone();
        // image files are read in here, so the session keeps only their paths
        let messages = self
            .session
            .context()
            .iter()
            .chain(pending)
            .map(|message| with_image_data(message, false))
            .collect();
        request
            .entry("max_tokens")
            .or_insert_with(|| json!(self.config.max_reply_tokens));
        request.insert("model".into(), Value::String(self.config.model.clone()));
        request.insert("messages".into(), Value::Array(messages));
        request.insert(
            "tools".into(),
            self.tools.iter().map(|t| t.definition()).collect(),
        );
        request.insert("tool_choice".into(), json!("auto"));
        request.insert("stream".into(), json!(true));
        Value::Object(request)
    }

    async fn stream_turn(
        &self,
        pending: &[Value],
        ui: &mut dyn AgentUi,
        interrupt: &Interrupt,
    ) -> Result<(Turn, bool)> {
        let mut turn = Turn::default();
        let mut stream = tokio::select! {
            stream = self.backend.stream(self.request(pending)) => stream?,
            () = interrupt.wait() => return Ok((turn, true)),
        };
        loop {
            tokio::select! {
                chunk = stream.next() => match chunk {
                    Some(chunk) => {
                        let delta = turn.apply(&chunk?);
                        if let Some(reasoning) = &delta.reasoning {
                            ui.reasoning(reasoning);
                        }
                        if let Some(content) = &delta.content {
                            ui.text(content);
                        }
                    }
                    None if turn.finish_reason.is_none() => bail!("the reply stream ended before the reply finished"),
                    None => return Ok((turn, false)),
                },
                () = interrupt.wait() => return Ok((turn, true)),
            }
        }
    }

    async fn run_call(
        &mut self,
        call: &ToolCall,
        ui: &mut dyn AgentUi,
        interrupt: &Interrupt,
    ) -> ToolOutcome {
        let Some(tool) = self.tools.iter().find(|t| t.name() == call.name).cloned() else {
            return ToolOutcome::error(format!("there is no tool `{}`", call.name));
        };
        let arguments = if call.arguments.trim().is_empty() {
            "{}"
        } else {
            &call.arguments
        };
        let args = match serde_json::from_str::<Value>(arguments) {
            Ok(args) if args.is_object() => args,
            Ok(_) => return ToolOutcome::error("the arguments must be a JSON object"),
            Err(error) => {
                return ToolOutcome::error(format!("the arguments are not valid JSON: {error}"));
            }
        };
        match self.permissions.check(tool.as_ref(), &args, &self.ctx) {
            Decision::Allow => {}
            Decision::Deny(reason) => return ToolOutcome::error(format!("not allowed: {reason}")),
            Decision::Ask => match ui.approve(call, &args, tool.kind()).await {
                Approval::Once => {}
                Approval::Always => self.permissions.allow_always(tool.name()),
                Approval::Deny => return ToolOutcome::error(DECLINED),
            },
        }
        {
            let mut changes = self.changes.lock().unwrap();
            // failing to keep a copy only costs that file's diff, so the call still runs
            let _ = match tool.kind() {
                ToolKind::Edit => tool
                    .paths(&args, &self.ctx)
                    .iter()
                    .try_for_each(|path| changes.record(path)),
                ToolKind::Execute => changes.note_command(),
                ToolKind::Read => Ok(()),
            };
        }
        ui.tool_start(call, &args);
        let outcome = tokio::select! {
            outcome = tool.run(args, &self.ctx) => outcome,
            () = interrupt.wait() => ToolOutcome::error(INTERRUPTED),
        };
        ui.tool_end(call, &outcome);
        outcome
    }
}

// a message's JSON length, with each image counted as IMAGE_CHARS rather than its URL or path
fn message_chars(message: &Value) -> usize {
    let len = message.to_string().len();
    let Some(parts) = message["content"].as_array() else {
        return len;
    };
    parts
        .iter()
        .filter_map(|part| {
            let reference = part["image_url"]["url"]
                .as_str()
                .or(part["path"].as_str())?;
            Some(reference.len())
        })
        .fold(len, |len, reference| {
            len.saturating_sub(reference) + IMAGE_CHARS
        })
}

fn share(window: usize, share: f64) -> usize {
    (window as f64 * share) as usize
}

// a summary request needs a long tool result's gist, not all of it, and an image's name, not its pixels
fn clipped(message: &Value) -> Value {
    let mut message = without_image_files(message);
    if message["role"] == "tool"
        && let Some(text) = message["content"].as_str()
        && text.len() > CLIPPED_RESULT_CHARS
    {
        let half = CLIPPED_RESULT_CHARS / 2;
        let head = text.floor_char_boundary(half);
        let tail = text.ceil_char_boundary(text.len() - half);
        message["content"] =
            Value::String(format!("{}{CLIP_MARK}{}", &text[..head], &text[tail..]));
    }
    message
}

#[cfg(test)]
mod tests {
    use super::{
        Agent, AgentConfig, AgentUi, CLIP_MARK, CLIPPED_RESULT_CHARS, COMPACT_PROMPT, IMAGE_CHARS,
        Interrupt, TurnEnd, clipped, message_chars,
    };
    use crate::backend::ChatBackend;
    use crate::permission::{Approval, PermissionMode, Permissions};
    use crate::session::{Compaction, Session, summary_message};
    use crate::tools::{ToolKind, ToolOutcome, default_tools};
    use crate::turn::{ToolCall, Turn};
    use anyhow::Result;
    use futures::future::BoxFuture;
    use futures::stream::BoxStream;
    use serde_json::{Map, Value, json};
    use std::sync::{Arc, Mutex};

    #[test]
    fn an_image_counts_as_a_fixed_share_of_the_context_not_its_bytes() {
        let image = format!("data:image/png;base64,{}", "A".repeat(1 << 20));
        let content = crate::session::user_content("look", &[image]);
        let message = json!({"role": "user", "content": content});
        let text = json!({"role": "user", "content": [{"type": "text", "text": "look"}]});
        let without_url = message.to_string().len() - (1 << 20) - "data:image/png;base64,".len();
        assert_eq!(message_chars(&message), without_url + IMAGE_CHARS);
        assert_eq!(message_chars(&text), text.to_string().len());
    }

    // replays canned replies and records the requests it was sent; a summary request gets `summary <n>`
    struct Scripted {
        replies: Mutex<Vec<Vec<Value>>>,
        requests: Mutex<Vec<Value>>,
        window: Option<usize>,
        vision: bool,
    }

    fn scripted(replies: Vec<Vec<Value>>, window: Option<usize>) -> Arc<Scripted> {
        Arc::new(Scripted {
            replies: Mutex::new(replies),
            requests: Mutex::new(Vec::new()),
            window,
            vision: false,
        })
    }

    fn is_summary_request(request: &Value) -> bool {
        request["messages"]
            .as_array()
            .and_then(|m| m.last())
            .is_some_and(|m| m["content"] == COMPACT_PROMPT)
    }

    impl ChatBackend for Scripted {
        fn stream(
            &self,
            request: Value,
        ) -> BoxFuture<'_, Result<BoxStream<'static, Result<Value>>>> {
            let summary = is_summary_request(&request);
            let mut requests = self.requests.lock().unwrap();
            requests.push(request);
            let chunks = if summary {
                let n = requests.iter().filter(|r| is_summary_request(r)).count();
                vec![
                    json!({"choices": [{"delta": {"content": format!("summary {n}")},
                    "finish_reason": "stop"}]}),
                ]
            } else {
                self.replies.lock().unwrap().remove(0)
            };
            Box::pin(async move {
                Ok(Box::pin(futures::stream::iter(chunks.into_iter().map(Ok))) as BoxStream<_>)
            })
        }

        fn describe(&self) -> String {
            "scripted".into()
        }

        fn context_tokens<'a>(&'a self, _: &'a str) -> BoxFuture<'a, Option<usize>> {
            let window = self.window;
            Box::pin(async move { window })
        }

        fn reads_images<'a>(&'a self, _: &'a str) -> BoxFuture<'a, bool> {
            let vision = self.vision;
            Box::pin(async move { vision })
        }
    }

    #[derive(Default)]
    struct Recorder {
        text: String,
        tools: Vec<(String, bool)>,
        approvals: usize,
        compactions: Vec<Compaction>,
    }

    impl AgentUi for Recorder {
        fn text(&mut self, delta: &str) {
            self.text.push_str(delta);
        }
        fn reasoning(&mut self, _: &str) {}
        fn tool_start(&mut self, _: &ToolCall, _: &Value) {}
        fn tool_end(&mut self, call: &ToolCall, outcome: &ToolOutcome) {
            self.tools.push((call.name.clone(), outcome.is_error));
        }
        fn approve<'a>(
            &'a mut self,
            _: &'a ToolCall,
            _: &'a Value,
            _: ToolKind,
        ) -> BoxFuture<'a, Approval> {
            self.approvals += 1;
            Box::pin(async { Approval::Once })
        }
        fn round_end(&mut self, _: &Turn) {}
        fn compacted(&mut self, compaction: &Compaction, _: usize) {
            self.compactions.push(compaction.clone());
        }
    }

    fn call(id: &str, name: &str, args: Value) -> Value {
        json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": id, "function": {"name": name, "arguments": args.to_string()}}]},
            "finish_reason": "tool_calls"}]})
    }

    #[tokio::test]
    async fn an_image_read_reaches_the_next_request_and_the_session_keeps_its_path() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("project");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("shot.png"), b"png bytes").unwrap();
        let backend = Arc::new(Scripted {
            vision: true,
            ..Arc::into_inner(scripted(
                vec![
                    vec![call("c1", "read", json!({"path": "shot.png"}))],
                    vec![json!({"choices": [{"delta": {"content": "Looks right."},
                        "finish_reason": "stop"}]})],
                ],
                None,
            ))
            .unwrap()
        });
        let session = Session::create(&dir.path().join("sessions"), &root, "m", "sys").unwrap();
        let config = AgentConfig {
            model: "m".into(),
            max_rounds: 10,
            max_reply_tokens: 64,
            request: Map::new(),
        };
        let mut agent = Agent::new(
            backend.clone(),
            default_tools(),
            Permissions::new(PermissionMode::Ask),
            session,
            config,
        );
        let mut ui = Recorder::default();
        agent
            .send("check the screenshot", &mut ui, &Interrupt::default())
            .await
            .unwrap();
        assert_eq!(ui.tools, [("read".to_string(), false)]);
        let requests = backend.requests.lock().unwrap();
        let last = requests[1]["messages"].as_array().unwrap();
        // system, prompt, the call, its result, then the image turn
        let image_turn = &last[4];
        assert_eq!(image_turn["role"], "user");
        let url = image_turn["content"][1]["image_url"]["url"]
            .as_str()
            .unwrap();
        assert!(url.starts_with("data:image/png;base64,"), "{url}");
        let saved = std::fs::read_to_string(&agent.session.path).unwrap();
        assert!(saved.contains("\"image_file\"") && !saved.contains("base64"));
    }

    #[tokio::test]
    async fn runs_tools_until_the_model_answers_and_keeps_the_history() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("project");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("notes.txt"), "old\n").unwrap();
        let backend = scripted(
            vec![
                vec![call("c1", "read", json!({"path": "notes.txt"}))],
                vec![call(
                    "c2",
                    "edit",
                    json!({"path": "notes.txt", "old_string": "old", "new_string": "new"}),
                )],
                vec![
                    json!({"choices": [{"delta": {"content": "Done."}, "finish_reason": "stop"}]}),
                ],
            ],
            None,
        );
        let session = Session::create(&dir.path().join("sessions"), &root, "m", "sys").unwrap();
        let config = AgentConfig {
            model: "m".into(),
            max_rounds: 10,
            max_reply_tokens: 64,
            request: Map::new(),
        };
        let mut agent = Agent::new(
            backend.clone(),
            default_tools(),
            Permissions::new(PermissionMode::Ask),
            session,
            config,
        );
        let mut ui = Recorder::default();
        let end = agent
            .send("update the notes", &mut ui, &Interrupt::default())
            .await
            .unwrap();
        assert_eq!(end, TurnEnd::Done);
        assert_eq!(ui.text, "Done.");
        assert_eq!(
            ui.tools,
            [("read".to_string(), false), ("edit".to_string(), false)]
        );
        // the read ran on its own; the edit asked
        assert_eq!(ui.approvals, 1);
        assert_eq!(
            std::fs::read_to_string(root.join("notes.txt")).unwrap(),
            "new\n"
        );
        let roles = agent
            .session
            .messages
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            roles,
            [
                "system",
                "user",
                "assistant",
                "tool",
                "assistant",
                "tool",
                "assistant"
            ]
        );
        assert_eq!(
            Session::open(&agent.session.path).unwrap().messages,
            agent.session.messages
        );
        let requests = backend.requests.lock().unwrap();
        // each request extends the previous one, so the cached prefix carries over
        let first = requests[0]["messages"].as_array().unwrap();
        let last = requests[2]["messages"].as_array().unwrap();
        assert_eq!(&last[..first.len()], first.as_slice());
        assert_eq!(requests[0]["tools"], requests[2]["tools"]);
    }

    #[tokio::test]
    async fn a_reply_cut_at_the_token_cap_ends_the_turn_without_its_partial_call() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("project");
        std::fs::create_dir_all(&root).unwrap();
        let backend = scripted(
            vec![vec![json!({"choices": [{"delta": {
                "reasoning_content": "let me think. let me think.",
                "tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read", "arguments": "{\"pa"}}]},
                "finish_reason": "length"}]})]],
            None,
        );
        let session = Session::create(&dir.path().join("sessions"), &root, "m", "sys").unwrap();
        let config = AgentConfig {
            model: "m".into(),
            max_rounds: 10,
            max_reply_tokens: 64,
            request: Map::new(),
        };
        let mut agent = Agent::new(
            backend.clone(),
            default_tools(),
            Permissions::new(PermissionMode::Auto),
            session,
            config,
        );
        let mut ui = Recorder::default();
        let end = agent
            .send("go", &mut ui, &Interrupt::default())
            .await
            .unwrap();
        assert_eq!(end, TurnEnd::ReplyLimit);
        assert!(ui.tools.is_empty());
        assert_eq!(backend.requests.lock().unwrap()[0]["max_tokens"], 64);
        let last = agent.session.messages.last().unwrap();
        assert_eq!(last["role"], "assistant");
        assert!(last.get("tool_calls").is_none() && last.get("reasoning_content").is_none());
        assert!(last["content"].as_str().unwrap().contains("token limit"));
    }

    // a session of `exchanges` user prompts and answers, each about `chars` long
    fn long_session(dir: &std::path::Path, exchanges: usize, chars: usize) -> Session {
        let mut session = Session::create(&dir.join("sessions"), dir, "m", "sys").unwrap();
        for i in 0..exchanges {
            let text = format!("{i} {}", "x".repeat(chars));
            session
                .append(json!({"role": "user", "content": text}))
                .unwrap();
            session
                .append(json!({"role": "assistant", "content": text}))
                .unwrap();
        }
        session
    }

    fn agent(backend: Arc<Scripted>, session: Session) -> Agent {
        let config = AgentConfig {
            model: "m".into(),
            max_rounds: 10,
            max_reply_tokens: 64,
            request: Map::new(),
        };
        Agent::new(
            backend,
            default_tools(),
            Permissions::new(PermissionMode::Auto),
            session,
            config,
        )
    }

    fn answer(text: &str) -> Vec<Value> {
        vec![json!({"choices": [{"delta": {"content": text}, "finish_reason": "stop"}]})]
    }

    // past most of the window the older history folds a chunk at a time (each summary taking in the last) before
    // the round, the transcript keeps everything, and the round is sent the summary and the recent messages
    #[tokio::test]
    async fn a_full_context_folds_the_older_history_before_the_round() {
        let dir = tempfile::tempdir().unwrap();
        // 12 messages of about 200 tokens against a 1000-token window: a chunk is a few messages, the kept tail one
        let session = long_session(dir.path(), 6, 600);
        let backend = scripted(vec![answer("ok")], Some(1000));
        let mut agent = agent(backend.clone(), session);
        let mut ui = Recorder::default();
        let end = agent
            .send("next", &mut ui, &Interrupt::default())
            .await
            .unwrap();
        assert_eq!(end, TurnEnd::Done);
        let requests = backend.requests.lock().unwrap();
        let summaries = requests.iter().filter(|r| is_summary_request(r)).count();
        assert!(summaries > 1, "{summaries} summary requests");
        // the second summary request opens with the first summary
        let second = requests
            .iter()
            .filter(|r| is_summary_request(r))
            .nth(1)
            .unwrap();
        assert_eq!(second["messages"][1], summary_message("summary 1"));
        let round = requests.last().unwrap()["messages"].as_array().unwrap();
        assert_eq!(round[1], summary_message(&format!("summary {summaries}")));
        assert_eq!(round.last().unwrap()["content"], "next");
        assert_eq!(ui.compactions.len(), 1);
        // 12 old messages, the prompt and its answer, all still in the transcript and on disk
        assert_eq!(agent.session.messages.len(), 15);
        let reopened = Session::open(&agent.session.path).unwrap();
        assert_eq!(reopened.compaction, agent.session.compaction);
        assert_eq!(reopened.context(), agent.session.context());
    }

    // asked to, it folds everything before the latest prompt even with no window known
    #[tokio::test]
    async fn compacting_on_request_folds_all_before_the_latest_prompt() {
        let dir = tempfile::tempdir().unwrap();
        let session = long_session(dir.path(), 3, 10);
        let mut agent = agent(scripted(Vec::new(), None), session);
        let mut ui = Recorder::default();
        assert!(agent.compact(&mut ui).await.unwrap());
        let context = agent.session.context();
        let roles = context
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(roles, ["system", "user", "assistant", "user", "assistant"]);
        assert_eq!(context[1], summary_message("summary 1"));
        assert!(context[3]["content"].as_str().unwrap().starts_with("2 "));
        // nothing before the latest prompt is left to fold
        assert!(!agent.compact(&mut ui).await.unwrap());
    }

    #[test]
    fn a_summary_request_clips_long_tool_results_only() {
        let long = "y".repeat(CLIPPED_RESULT_CHARS * 2);
        let tool = clipped(&json!({"role": "tool", "content": long}));
        let text = tool["content"].as_str().unwrap();
        assert!(
            text.contains(CLIP_MARK.trim())
                && text.len() < CLIPPED_RESULT_CHARS + CLIP_MARK.len() + 1
        );
        let user = json!({"role": "user", "content": long});
        assert_eq!(clipped(&user), user);
    }
}
