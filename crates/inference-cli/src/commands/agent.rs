//! `inference agent`: the coding agent's REPL and headless mode over `inference_coder`.

use std::future::Future;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use inference_api::engine_chat::{ReasoningEffort, resolve_reasoning_controls};
use inference_api::{Engine, initialize_logging};
use inference_coder::{
    Agent, AgentConfig, AgentUi, Approval, ChatBackend, EngineBackend, HttpBackend, Interrupt,
    PermissionMode, Permissions, Session, ToolCall, ToolKind, ToolOutcome, Turn, TurnEnd,
    default_tools, prompt::system_prompt,
};
use rustyline::{DefaultEditor, error::ReadlineError};
use serde_json::{Map, Value, json};

use super::run::run_spec;
use super::serve::normalize_quant_flags;
use crate::args::{CoderOptions, GlobalOptions, ModelType, RuntimeOptions, SandboxOptions};

const API_KEY_ENV: &str = "INFERENCE_API_KEY";
// the request model that targets the engine's only model
const ENGINE_MODEL: &str = "default";
const HISTORY_FILE: &str = "agent_history.txt";
const SUMMARY_CHARS: usize = 100;
const PREVIEW_LINES: usize = 4;
const APPROVAL_PREVIEW_LINES: usize = 20;
const DIM: &str = "\x1b[2m";
const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";
// exit status after a second Ctrl-C, as a shell reports SIGINT
const SIGINT_EXIT: i32 = 130;

const HELP: &str = "\
/help              this text
/clear             start a new session
/sessions          this project's sessions
/resume <id>       switch to a session
/mode <mode>       ask, auto or read-only
/context           how full the context is
/exit              quit (also Ctrl-D)
Ctrl-C stops the model or the running tool; a second one quits.";

/// Where the model comes from: loaded here from `run`'s model options, or a server.
pub enum AgentModel {
    Local {
        model_type: Box<ModelType>,
        runtime: Box<RuntimeOptions>,
        sandbox: SandboxOptions,
    },
    Server {
        url: String,
        model: String,
    },
}

pub async fn run_agent(
    model: AgentModel,
    coder: CoderOptions,
    global: GlobalOptions,
    thinking: Option<bool>,
    reasoning_effort: Option<ReasoningEffort>,
) -> Result<()> {
    let root = project_root(&coder)?;
    let sessions_dir = inference_coder::session::default_dir();
    resolve_reasoning_controls(thinking, reasoning_effort)?;
    let mut request = Map::new();
    if let Some(thinking) = thinking {
        request.insert("enable_thinking".into(), json!(thinking));
    }
    if let Some(effort) = reasoning_effort {
        request.insert("reasoning_effort".into(), json!(effort.as_str()));
    }

    let (backend, request_model): (Arc<dyn ChatBackend>, String) = match model {
        AgentModel::Server { url, model } => {
            let api_key = match &coder.api_key_file {
                Some(path) => Some(read_api_key(
                    &std::fs::read_to_string(path)
                        .with_context(|| format!("reading {}", path.display()))?,
                )?),
                None => std::env::var(API_KEY_ENV)
                    .ok()
                    .filter(|key| !key.is_empty()),
            };
            (
                Arc::new(HttpBackend::new(&url, api_key, model.clone())?),
                model,
            )
        }
        AgentModel::Local {
            mut model_type,
            runtime,
            sandbox,
        } => {
            initialize_logging();
            normalize_quant_flags(&mut model_type)?;
            let label = model_label(&model_type);
            let engine = Engine::load(run_spec(&model_type, &runtime, sandbox, &global)?).await?;
            (
                Arc::new(EngineBackend::new(engine, label)),
                ENGINE_MODEL.to_string(),
            )
        }
    };

    let session = if let Some(id) = &coder.resume {
        Session::open(&Session::find(&sessions_dir, id)?)?
    } else if coder.continue_session {
        match Session::latest_for(&sessions_dir, &root)? {
            Some(path) => Session::open(&path)?,
            None => bail!("no session to continue for {}", root.display()),
        }
    } else {
        Session::create(
            &sessions_dir,
            &root,
            &backend.describe(),
            &system_prompt(&root),
        )?
    };
    let config = AgentConfig {
        model: request_model,
        max_rounds: coder.max_rounds,
        max_reply_tokens: coder.max_reply_tokens,
        request,
    };
    let mut agent = Agent::new(
        backend,
        default_tools(),
        Permissions::new(coder.permission_mode),
        session,
        config,
    );

    match coder.prompt {
        Some(prompt) => headless(&mut agent, &prompt).await,
        None => repl(agent, &sessions_dir).await,
    }
}

/// A bare key, or the first key of a `serve` keys file (`name = key` lines).
fn read_api_key(text: &str) -> Result<String> {
    let line = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('#'))
        .context("the API key file is empty")?;
    Ok(line
        .split_once('=')
        .map_or(line, |(_, key)| key.trim())
        .to_string())
}

fn project_root(coder: &CoderOptions) -> Result<PathBuf> {
    let root = match &coder.cwd {
        Some(dir) => dir.clone(),
        None => std::env::current_dir()?,
    };
    root.canonicalize()
        .with_context(|| format!("project directory {}", root.display()))
}

/// `--sessions`: this project's sessions, newest first. No model is needed.
pub fn list_agent_sessions(coder: &CoderOptions) -> Result<()> {
    print_sessions(
        &inference_coder::session::default_dir(),
        &project_root(coder)?,
    )
}

fn model_label(model_type: &ModelType) -> String {
    match model_type {
        ModelType::Auto { model, .. } if !model.model_id.is_empty() => model.model_id.clone(),
        _ => "local model".to_string(),
    }
}

async fn headless(agent: &mut Agent, prompt: &str) -> Result<()> {
    let interrupt = Interrupt::default();
    let handler = interrupt.clone();
    ctrlc::set_handler(move || handler.trigger()).context("installing the Ctrl-C handler")?;
    let mut ui = Headless;
    let end = agent.send(prompt, &mut ui, &interrupt).await?;
    println!();
    match end {
        TurnEnd::Done => Ok(()),
        TurnEnd::Interrupted => bail!("interrupted"),
        TurnEnd::RoundLimit => bail!("stopped at the round limit (--max-rounds)"),
        TurnEnd::ReplyLimit => bail!("a reply hit the token limit (--max-reply-tokens)"),
    }
}

struct Headless;

impl AgentUi for Headless {
    fn text(&mut self, delta: &str) {
        print!("{delta}");
        let _ = std::io::stdout().flush();
    }

    fn reasoning(&mut self, _: &str) {}

    fn tool_start(&mut self, call: &ToolCall, args: &Value) {
        eprintln!("* {}", summarize(call, args));
    }

    fn tool_end(&mut self, _: &ToolCall, outcome: &ToolOutcome) {
        if outcome.is_error {
            eprintln!(
                "  error: {}",
                outcome.text.lines().next().unwrap_or_default()
            );
        }
    }

    fn approve<'a>(
        &'a mut self,
        call: &'a ToolCall,
        args: &'a Value,
        _: ToolKind,
    ) -> Pin<Box<dyn Future<Output = Approval> + Send + 'a>> {
        eprintln!(
            "* {} refused: it needs approval (see --permission-mode)",
            summarize(call, args)
        );
        Box::pin(async { Approval::Deny })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Block {
    Fresh,
    Reasoning,
    Text,
}

struct Repl {
    editor: DefaultEditor,
    block: Block,
    // newlines held back until more text follows, so a block never ends in a run of blank lines
    held_newlines: usize,
    last_usage: Option<Value>,
}

impl Repl {
    fn write(&mut self, block: Block, delta: &str) {
        let delta = if self.block == block {
            delta
        } else {
            let trimmed = delta.trim_start_matches('\n');
            if trimmed.is_empty() {
                return;
            }
            self.finish_block(self.block == Block::Reasoning);
            if block == Block::Reasoning {
                print!("{DIM}");
            }
            self.block = block;
            trimmed
        };
        let core = delta.trim_end_matches('\n');
        if core.is_empty() {
            self.held_newlines += delta.len();
        } else {
            print!("{}{core}", "\n".repeat(self.held_newlines));
            self.held_newlines = delta.len() - core.len();
        }
        let _ = std::io::stdout().flush();
    }

    /// Ends the block on its own line, with a blank line after it when `gap`.
    fn finish_block(&mut self, gap: bool) {
        if self.block == Block::Fresh {
            return;
        }
        print!("{RESET}\n{}", if gap { "\n" } else { "" });
        self.block = Block::Fresh;
        self.held_newlines = 0;
    }
}

impl AgentUi for Repl {
    fn text(&mut self, delta: &str) {
        self.write(Block::Text, delta);
    }

    fn reasoning(&mut self, delta: &str) {
        self.write(Block::Reasoning, delta);
    }

    fn tool_start(&mut self, call: &ToolCall, args: &Value) {
        self.finish_block(false);
        println!("{BOLD}* {}{RESET}", summarize(call, args));
    }

    fn tool_end(&mut self, _: &ToolCall, outcome: &ToolOutcome) {
        let color = if outcome.is_error { RED } else { DIM };
        let lines = outcome.text.lines().collect::<Vec<_>>();
        for line in lines.iter().take(PREVIEW_LINES) {
            println!("{color}  | {}{RESET}", cut(line, SUMMARY_CHARS));
        }
        if lines.len() > PREVIEW_LINES {
            println!(
                "{DIM}  | ... {} more lines{RESET}",
                lines.len() - PREVIEW_LINES
            );
        }
    }

    fn approve<'a>(
        &'a mut self,
        call: &'a ToolCall,
        args: &'a Value,
        _: ToolKind,
    ) -> Pin<Box<dyn Future<Output = Approval> + Send + 'a>> {
        self.finish_block(false);
        println!("{BOLD}? {}{RESET}", summarize(call, args));
        print_approval_preview(call, args);
        let question = format!(
            "allow? [y]es, [a]lways allow {} this session, [n]o > ",
            call.name
        );
        let answer = match self.editor.readline(&question) {
            Ok(answer) => match answer.trim() {
                "y" | "yes" => Approval::Once,
                "a" | "always" => Approval::Always,
                _ => Approval::Deny,
            },
            Err(_) => Approval::Deny,
        };
        Box::pin(async move { answer })
    }

    fn round_end(&mut self, turn: &Turn) {
        if turn.usage.is_some() {
            self.last_usage = turn.usage.clone();
        }
    }
}

fn print_approval_preview(call: &ToolCall, args: &Value) {
    let field = |key: &str| args.get(key).and_then(Value::as_str).unwrap_or_default();
    let block = |text: &str, sign: &str, color: &str| {
        let lines = text.lines().collect::<Vec<_>>();
        for line in lines.iter().take(APPROVAL_PREVIEW_LINES) {
            println!("{color}  {sign} {line}{RESET}");
        }
        if lines.len() > APPROVAL_PREVIEW_LINES {
            println!(
                "{DIM}  ... {} more lines{RESET}",
                lines.len() - APPROVAL_PREVIEW_LINES
            );
        }
    };
    match call.name.as_str() {
        "edit" => {
            block(field("old_string"), "-", RED);
            block(field("new_string"), "+", GREEN);
        }
        "write" => block(field("content"), "+", GREEN),
        "bash" if field("command").lines().count() > 1 => block(field("command"), "$", ""),
        _ => {}
    }
}

fn summarize(call: &ToolCall, args: &Value) -> String {
    let field = |key: &str| args.get(key).and_then(Value::as_str);
    let detail = match call.name.as_str() {
        "bash" => field("command").map(|c| c.lines().next().unwrap_or_default().to_string()),
        "grep" | "glob" => field("pattern").map(|p| match field("path") {
            Some(path) => format!("{p} in {path}"),
            None => p.to_string(),
        }),
        _ => field("path").map(str::to_string),
    }
    .unwrap_or_else(|| args.to_string());
    format!("{}({})", call.name, cut(&detail, SUMMARY_CHARS))
}

fn cut(text: &str, max: usize) -> String {
    match text.char_indices().nth(max) {
        Some((end, _)) => format!("{}...", &text[..end]),
        None => text.to_string(),
    }
}

fn history_path() -> Option<PathBuf> {
    let dir = dirs::config_dir()?.join("inference.rs");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join(HISTORY_FILE))
}

fn print_sessions(dir: &Path, root: &Path) -> Result<()> {
    let sessions = Session::list(dir)?
        .into_iter()
        .filter(|s| s.root == root)
        .collect::<Vec<_>>();
    if sessions.is_empty() {
        println!("no sessions for {}", root.display());
    }
    for session in sessions {
        println!("{}  {}", session.id, session.title);
    }
    Ok(())
}

fn print_context(agent: &Agent, usage: Option<&Value>) {
    println!(
        "session {} ({} messages)",
        agent.session.id,
        agent.session.messages.len()
    );
    if let Some(usage) = usage {
        println!("context: {} tokens", usage["total_tokens"]);
    }
}

fn print_turn_stats(usage: &Value) {
    let cached = usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0);
    println!(
        "{DIM}[ctx {} tokens, {} cached, prefill {:.1} s, {:.1} tok/s]{RESET}",
        usage["total_tokens"],
        cached,
        usage["total_prompt_time_sec"].as_f64().unwrap_or(0.0),
        usage["avg_compl_tok_per_sec"].as_f64().unwrap_or(0.0),
    );
}

async fn repl(mut agent: Agent, sessions_dir: &Path) -> Result<()> {
    let interrupt = Interrupt::default();
    let handler = interrupt.clone();
    ctrlc::set_handler(move || {
        if handler.is_set() {
            std::process::exit(SIGINT_EXIT);
        }
        handler.trigger();
    })
    .context("installing the Ctrl-C handler")?;
    let mut editor = DefaultEditor::new()?;
    let history = history_path();
    if let Some(path) = &history {
        let _ = editor.load_history(path);
    }
    let mut ui = Repl {
        editor,
        block: Block::Fresh,
        held_newlines: 0,
        last_usage: None,
    };
    println!(
        "{BOLD}inference agent{RESET}: {}",
        agent.backend().describe()
    );
    println!(
        "{DIM}{} | session {} | /help for commands{RESET}",
        agent.session.root.display(),
        agent.session.id
    );
    if agent.session.messages.len() > 1 {
        println!(
            "{DIM}resumed with {} messages{RESET}",
            agent.session.messages.len() - 1
        );
    }
    loop {
        let line = match ui.editor.readline("> ") {
            Ok(line) => line,
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
            Err(error) => return Err(error.into()),
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let _ = ui.editor.add_history_entry(line);
        if let Some(command) = line.strip_prefix('/') {
            let (name, arg) = command
                .split_once(' ')
                .map_or((command, ""), |(n, a)| (n, a.trim()));
            match name {
                "exit" | "quit" => break,
                "help" => println!("{HELP}"),
                "clear" => {
                    let root = agent.session.root.clone();
                    agent.session = Session::create(
                        sessions_dir,
                        &root,
                        &agent.backend().describe(),
                        &system_prompt(&root),
                    )?;
                    ui.last_usage = None;
                    println!("{DIM}new session {}{RESET}", agent.session.id);
                }
                "sessions" => print_sessions(sessions_dir, &agent.session.root)?,
                "resume" => {
                    match Session::find(sessions_dir, arg).and_then(|path| Session::open(&path)) {
                        Ok(session) if session.root == agent.session.root => {
                            println!(
                                "{DIM}resumed {} ({} messages){RESET}",
                                session.id,
                                session.messages.len() - 1
                            );
                            agent.session = session;
                            ui.last_usage = None;
                        }
                        Ok(session) => println!(
                            "{RED}{} belongs to {}{RESET}",
                            session.id,
                            session.root.display()
                        ),
                        Err(error) => println!("{RED}{error}{RESET}"),
                    }
                }
                "mode" => match arg.parse::<PermissionMode>() {
                    Ok(mode) => {
                        agent.permissions.mode = mode;
                        println!("{DIM}permission mode: {arg}{RESET}");
                    }
                    Err(error) => println!("{RED}{error}{RESET}"),
                },
                "context" => print_context(&agent, ui.last_usage.as_ref()),
                _ => println!("{RED}unknown command /{name} (try /help){RESET}"),
            }
            continue;
        }
        interrupt.reset();
        let previous_usage = ui.last_usage.take();
        let result = agent.send(line, &mut ui, &interrupt).await;
        ui.finish_block(false);
        match result {
            Ok(TurnEnd::Done) => {}
            Ok(TurnEnd::Interrupted) => println!("{DIM}[interrupted]{RESET}"),
            Ok(TurnEnd::RoundLimit) => println!("{RED}[stopped at the round limit]{RESET}"),
            Ok(TurnEnd::ReplyLimit) => {
                println!("{RED}[the reply hit the token limit, likely repeating itself]{RESET}")
            }
            Err(error) => println!("{RED}error: {error:#}{RESET}"),
        }
        match &ui.last_usage {
            Some(usage) => print_turn_stats(usage),
            None => ui.last_usage = previous_usage,
        }
        interrupt.reset();
    }
    if let Some(path) = &history {
        let _ = ui.editor.save_history(path);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::read_api_key;

    #[test]
    fn api_key_files_hold_a_bare_key_or_named_keys() {
        assert_eq!(read_api_key("sk-plain\n").unwrap(), "sk-plain");
        assert_eq!(
            read_api_key("# made by serve\nfirst = sk-one\nsecond = sk-two\n").unwrap(),
            "sk-one"
        );
        assert!(read_api_key("# only a comment\n").is_err());
    }
}
