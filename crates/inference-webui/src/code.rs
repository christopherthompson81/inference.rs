//! The Code tab: `inference_coder`'s agent over this server's engine, working in projects under the code roots, or in
//! a demo project it makes in the temp directory.

use std::collections::{BTreeMap, HashSet};
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, UNIX_EPOCH};

use axum::Json;
use inference_json::QueryParams;

use crate::json_body::JsonBody;
use axum::extract::Extension;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::future::BoxFuture;
use indexmap::IndexMap;
use inference_coder::{
    Agent, AgentConfig, AgentUi, Approval, Changes, ChatBackend, DEFAULT_MAX_REPLY_TOKENS,
    DEFAULT_MAX_ROUNDS, EngineBackend, Interrupt, PermissionMode, Permissions, Session, ToolCall,
    ToolKind, ToolOutcome, Turn, TurnEnd,
    activity::{Run, RunTracker},
    default_tools,
    prompt::system_prompt,
};
use serde_json::{Map, Value, json};
use tokio::sync::{mpsc, oneshot};

use crate::types::AppState;

// sessions kept loaded at once; the least recently opened idle one goes first
const MAX_LIVE_SESSIONS: usize = 32;
const KEEP_ALIVE: Duration = Duration::from_secs(15);
const DEMO_DIR_PREFIX: &str = "inference-code-demo-";
const DEMO_PROMPT: &str = "The test suite fails. Fix the stockroom package so \
`python3 -m unittest discover -s tests -t .` passes. The tests and README.md describe the intended behaviour; \
don't change the tests.";
// a three-module package with eight planted bugs and a missing `undo`, failing 14 of its 19 tests
macro_rules! demo_file {
    ($path:literal) => {
        ($path, include_str!(concat!("../demo/", $path)))
    };
}
const DEMO_FILES: [(&str, &str); 10] = [
    // so the test run's bytecode stays out of the Files tab
    demo_file!(".gitignore"),
    demo_file!("README.md"),
    demo_file!("stockroom/__init__.py"),
    demo_file!("stockroom/units.py"),
    demo_file!("stockroom/inventory.py"),
    demo_file!("stockroom/report.py"),
    demo_file!("tests/__init__.py"),
    demo_file!("tests/test_units.py"),
    demo_file!("tests/test_inventory.py"),
    demo_file!("tests/test_report.py"),
];

/// The projects the Code tab may open and the sessions it has loaded.
pub(crate) struct CodeState {
    roots: Vec<PathBuf>,
    /// Demo projects made since the server started, which may be opened although they are outside the roots.
    demos: Mutex<HashSet<PathBuf>>,
    sessions_dir: PathBuf,
    live: Mutex<IndexMap<String, Arc<LiveSession>>>,
}

struct LiveSession {
    agent: Arc<tokio::sync::Mutex<Agent>>,
    interrupt: Interrupt,
    approvals: Mutex<IndexMap<String, oneshot::Sender<Approval>>>,
    // outside the agent's lock, so the page can read it while a prompt runs
    changes: Arc<Mutex<Changes>>,
    // activity labels by their group's first call, also outside the lock, and the file they are kept in
    labels: Mutex<BTreeMap<String, String>>,
    path: PathBuf,
}

impl CodeState {
    pub(crate) fn new(roots: &[PathBuf]) -> anyhow::Result<Self> {
        let roots = roots
            .iter()
            .map(|root| {
                let path = expand_home(root);
                path.canonicalize()
                    .map_err(|error| anyhow::anyhow!("code root {}: {error}", path.display()))
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(Self {
            roots,
            demos: Mutex::new(HashSet::new()),
            sessions_dir: inference_coder::session::default_dir(),
            live: Mutex::new(IndexMap::new()),
        })
    }

    pub(crate) fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    fn allows(&self, path: &Path) -> bool {
        self.roots.iter().any(|root| path.starts_with(root))
            || self.demos.lock().unwrap().contains(path)
    }

    /// `project`, an absolute path (or one under `~/`), as a directory the Code tab may open.
    fn project(&self, project: &str) -> Result<PathBuf, Refusal> {
        let path = expand_home(Path::new(project.trim()));
        if !path.is_absolute() {
            return Err(bad_request(format!(
                "{}: give the project as an absolute path",
                path.display()
            )));
        }
        // canonical, so `..` and symlinks can't step outside a root
        let path = path
            .canonicalize()
            .map_err(|error| bad_request(format!("{}: {error}", path.display())))?;
        if !path.is_dir() {
            return Err(bad_request(format!(
                "{} is not a directory",
                path.display()
            )));
        }
        if !self.allows(&path) {
            let reason = if self.roots.is_empty() {
                "the server has no code roots (--code-root)".to_string()
            } else {
                let roots = self.roots.iter().map(|root| root.display().to_string());
                format!(
                    "it is not under a code root ({})",
                    roots.collect::<Vec<_>>().join(", ")
                )
            };
            return Err(bad_request(format!(
                "{} can't be opened: {reason}",
                path.display()
            )));
        }
        Ok(path)
    }

    fn live(&self, id: &str) -> Option<Arc<LiveSession>> {
        self.live.lock().unwrap().get(id).cloned()
    }

    fn insert(&self, id: String, session: Arc<LiveSession>) {
        let mut live = self.live.lock().unwrap();
        if live.len() >= MAX_LIVE_SESSIONS
            && let Some(idle) = live.iter().position(|(_, s)| s.agent.try_lock().is_ok())
        {
            live.shift_remove_index(idle);
        }
        live.insert(id, session);
    }
}

fn expand_home(path: &Path) -> PathBuf {
    match (path.strip_prefix("~"), dirs::home_dir()) {
        (Ok(rest), Some(home)) => home.join(rest),
        _ => path.to_path_buf(),
    }
}

/// Why a request was turned down, small enough to pass back through `Result`s.
type Refusal = (StatusCode, String);

fn bad_request(message: impl Into<String>) -> Refusal {
    (StatusCode::BAD_REQUEST, message.into())
}

fn not_found(id: &str) -> Response {
    (StatusCode::NOT_FOUND, format!("no session `{id}`")).into_response()
}

/// The model the page picked, else the owner's chat selection, else the first chat model loaded.
async fn chat_model(app: &AppState, requested: Option<&str>) -> Result<String, Refusal> {
    if let Some(model) = requested.filter(|model| !model.is_empty()) {
        return Ok(model.to_string());
    }
    if let Some(model) = app.current.read().await.clone() {
        return Ok(model);
    }
    crate::chat_models(&app.engine)
        .into_keys()
        .next()
        .ok_or_else(|| bad_request("no chat model is loaded"))
}

#[derive(inference_json::FromJson)]
pub(crate) struct SessionsQuery {
    #[serde(default)]
    project: String,
}

/// The project's sessions, newest first.
pub(crate) async fn list_sessions(
    Extension(code): Extension<Arc<CodeState>>,
    QueryParams(query): QueryParams<SessionsQuery>,
) -> Response {
    let project = match code.project(&query.project) {
        Ok(project) => project,
        Err(refusal) => return refusal.into_response(),
    };
    let sessions = match Session::list(&code.sessions_dir) {
        Ok(sessions) => sessions,
        Err(error) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response();
        }
    };
    let sessions = sessions
        .into_iter()
        .filter(|s| s.root == project)
        .map(|s| {
            let modified = s
                .modified
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
            json!({"id": s.id, "title": s.title, "modified": modified})
        })
        .collect::<Vec<_>>();
    Json(json!({"project": project, "sessions": sessions})).into_response()
}

#[derive(inference_json::FromJson)]
pub(crate) struct OpenRequest {
    #[serde(default)]
    project: String,
    /// Resume this session; without one a new session starts in `project`.
    session_id: Option<String>,
    #[serde(default)]
    model: Option<String>,
}

/// Opens a session, new or saved, and returns its history.
pub(crate) async fn open_session(
    Extension(code): Extension<Arc<CodeState>>,
    Extension(app): Extension<Arc<AppState>>,
    JsonBody(request): JsonBody<OpenRequest>,
) -> Response {
    let live = match &request.session_id {
        Some(id) => match code.live(id) {
            Some(live) => live,
            None => {
                let session = match Session::find(&code.sessions_dir, id)
                    .and_then(|path| Session::open(&path))
                {
                    Ok(session) => session,
                    Err(_) => return not_found(id),
                };
                if !code.allows(&session.root) {
                    return bad_request("that session's project is outside the code roots")
                        .into_response();
                }
                match load(&code, &app, session, request.model.as_deref()).await {
                    Ok(live) => live,
                    Err(refusal) => return refusal.into_response(),
                }
            }
        },
        None => {
            let project = match code.project(&request.project) {
                Ok(project) => project,
                Err(refusal) => return refusal.into_response(),
            };
            let model = match chat_model(&app, request.model.as_deref()).await {
                Ok(model) => model,
                Err(refusal) => return refusal.into_response(),
            };
            match Session::create(
                &code.sessions_dir,
                &project,
                &model,
                &system_prompt(&project),
            ) {
                Ok(session) => match load(&code, &app, session, request.model.as_deref()).await {
                    Ok(live) => live,
                    Err(refusal) => return refusal.into_response(),
                },
                Err(error) => {
                    return (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response();
                }
            }
        }
    };
    let Ok(agent) = live.agent.try_lock() else {
        return (StatusCode::CONFLICT, "a prompt is running in that session").into_response();
    };
    let session = &agent.session;
    Json(json!({
        "session_id": session.id,
        "project": session.root,
        // the system prompt is the harness's, not part of the conversation shown
        "messages": session.messages.iter().skip(1).collect::<Vec<_>>(),
        "labels": *live.labels.lock().unwrap(),
    }))
    .into_response()
}

/// Makes a demo project in the temp directory (a small package whose tests fail) and opens a session there; the page
/// then sends `prompt` like any other.
pub(crate) async fn start_demo(
    Extension(code): Extension<Arc<CodeState>>,
    Extension(app): Extension<Arc<AppState>>,
    JsonBody(request): JsonBody<DemoRequest>,
) -> Response {
    let project = match make_demo_project() {
        Ok(project) => project,
        Err(error) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("demo project: {error}"),
            )
                .into_response();
        }
    };
    code.demos.lock().unwrap().insert(project.clone());
    let model = match chat_model(&app, request.model.as_deref()).await {
        Ok(model) => model,
        Err(refusal) => return refusal.into_response(),
    };
    let session = match Session::create(
        &code.sessions_dir,
        &project,
        &model,
        &system_prompt(&project),
    ) {
        Ok(session) => session,
        Err(error) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response();
        }
    };
    let session_id = session.id.clone();
    if let Err(refusal) = load(&code, &app, session, Some(&model)).await {
        return refusal.into_response();
    }
    Json(json!({"session_id": session_id, "project": project, "prompt": DEMO_PROMPT}))
        .into_response()
}

#[derive(inference_json::FromJson)]
pub(crate) struct DemoRequest {
    #[serde(default)]
    model: Option<String>,
}

fn make_demo_project() -> std::io::Result<PathBuf> {
    let id = uuid::Uuid::new_v4().simple().to_string();
    let project = std::env::temp_dir().join(format!("{DEMO_DIR_PREFIX}{}", &id[..8]));
    std::fs::create_dir_all(&project)?;
    for (path, contents) in DEMO_FILES {
        let path = project.join(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, contents)?;
    }
    // a repo, so `git diff` shows what the agent changed; a machine without git still gets the files
    let committed = std::process::Command::new("sh")
        .args(["-c", "git init -q && git add . && git -c user.name=demo -c user.email=demo@example.invalid commit -qm demo"])
        .current_dir(&project)
        .status();
    if let Err(error) = committed {
        tracing::debug!("demo project without git: {error}");
    }
    project.canonicalize()
}

async fn load(
    code: &CodeState,
    app: &AppState,
    session: Session,
    requested: Option<&str>,
) -> Result<Arc<LiveSession>, Refusal> {
    let model = chat_model(app, requested).await?;
    let backend = Arc::new(EngineBackend::new(app.engine.clone(), model.clone()));
    let config = AgentConfig {
        model,
        max_rounds: DEFAULT_MAX_ROUNDS,
        max_reply_tokens: DEFAULT_MAX_REPLY_TOKENS,
        request: Map::new(),
    };
    let id = session.id.clone();
    let agent = Agent::new(
        backend,
        default_tools(),
        Permissions::new(PermissionMode::Ask),
        session,
        config,
    );
    let live = Arc::new(LiveSession {
        labels: Mutex::new(agent.session.labels.clone()),
        path: agent.session.path.clone(),
        changes: agent.changes(),
        agent: Arc::new(tokio::sync::Mutex::new(agent)),
        interrupt: Interrupt::default(),
        approvals: Mutex::new(IndexMap::new()),
    });
    code.insert(id, live.clone());
    Ok(live)
}

#[derive(inference_json::FromJson)]
pub(crate) struct SendRequest {
    session_id: String,
    prompt: String,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    enable_thinking: Option<bool>,
    #[serde(default)]
    model: Option<String>,
    /// Label each run of thinking and tool calls as it closes, with the session's model.
    #[serde(default)]
    describe_activity: bool,
}

/// Runs a prompt to its answer, streaming the agent's text, tool calls and approval requests as SSE.
pub(crate) async fn send(
    Extension(code): Extension<Arc<CodeState>>,
    Extension(app): Extension<Arc<AppState>>,
    JsonBody(request): JsonBody<SendRequest>,
) -> Response {
    let Some(live) = code.live(&request.session_id) else {
        return not_found(&request.session_id);
    };
    let mode = match request.mode.as_deref().map(str::parse::<PermissionMode>) {
        None => PermissionMode::Ask,
        Some(Ok(mode)) => mode,
        Some(Err(error)) => return bad_request(error).into_response(),
    };
    let model = match chat_model(&app, request.model.as_deref()).await {
        Ok(model) => model,
        Err(refusal) => return refusal.into_response(),
    };
    let Ok(mut agent) = live.agent.clone().try_lock_owned() else {
        return (
            StatusCode::CONFLICT,
            "a prompt is already running in this session",
        )
            .into_response();
    };
    agent.permissions.mode = mode;
    let labeller = request.describe_activity.then(|| Labeller {
        backend: agent.shared_backend(),
        model: model.clone(),
        task: request.prompt.clone(),
    });
    agent.set_model(model);
    agent.set_request_field("enable_thinking", request.enable_thinking.map(Value::Bool));
    live.interrupt.reset();
    let (events, receiver) = mpsc::unbounded_channel::<Value>();
    tokio::spawn(async move {
        let mut ui = WebUi {
            events: events.clone(),
            live: live.clone(),
            labeller,
            runs: RunTracker::default(),
        };
        let result = agent.send(&request.prompt, &mut ui, &live.interrupt).await;
        if let Some(run) = ui.runs.finish() {
            ui.label(run);
        }
        let event = match result {
            Ok(end) => json!({"type": "done", "end": match end {
                TurnEnd::Done => "done",
                TurnEnd::Interrupted => "interrupted",
                TurnEnd::RoundLimit => "round_limit",
                TurnEnd::ReplyLimit => "reply_limit",
            }}),
            Err(error) => json!({"type": "error", "message": format!("{error:#}")}),
        };
        let _ = events.send(event);
    });
    let stream = futures::stream::unfold(receiver, |mut receiver| async move {
        let event = receiver.recv().await?;
        Some((
            Ok::<_, Infallible>(Event::default().data(event.to_string())),
            receiver,
        ))
    });
    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(KEEP_ALIVE))
        .into_response()
}

#[derive(inference_json::FromJson)]
pub(crate) struct ApproveRequest {
    session_id: String,
    approval_id: String,
    /// once, always or deny
    decision: String,
}

pub(crate) async fn approve(
    Extension(code): Extension<Arc<CodeState>>,
    JsonBody(request): JsonBody<ApproveRequest>,
) -> Response {
    let Some(live) = code.live(&request.session_id) else {
        return not_found(&request.session_id);
    };
    let decision = match request.decision.as_str() {
        "once" => Approval::Once,
        "always" => Approval::Always,
        "deny" => Approval::Deny,
        other => {
            return bad_request(format!("unknown decision `{other}` (once, always, deny)"))
                .into_response();
        }
    };
    let Some(waiting) = live
        .approvals
        .lock()
        .unwrap()
        .shift_remove(&request.approval_id)
    else {
        return (StatusCode::NOT_FOUND, "no such pending approval").into_response();
    };
    let _ = waiting.send(decision);
    StatusCode::NO_CONTENT.into_response()
}

#[derive(inference_json::FromJson)]
pub(crate) struct InterruptRequest {
    session_id: String,
}

pub(crate) async fn interrupt(
    Extension(code): Extension<Arc<CodeState>>,
    JsonBody(request): JsonBody<InterruptRequest>,
) -> Response {
    let Some(live) = code.live(&request.session_id) else {
        return not_found(&request.session_id);
    };
    live.interrupt.trigger();
    // a prompt waiting on the user stops too
    for (_, waiting) in live.approvals.lock().unwrap().drain(..) {
        let _ = waiting.send(Approval::Deny);
    }
    StatusCode::NO_CONTENT.into_response()
}

/// Forwards the agent's progress to the SSE stream; a closed stream (the page went away) interrupts the prompt.
struct WebUi {
    events: mpsc::UnboundedSender<Value>,
    live: Arc<LiveSession>,
    labeller: Option<Labeller>,
    runs: RunTracker,
}

// what a run's label request needs, for a task of its own beside the agent's
struct Labeller {
    backend: Arc<dyn ChatBackend>,
    model: String,
    task: String,
}

impl WebUi {
    // the label reaches the page on the prompt's stream, which stays open until it does
    fn label(&self, run: Run) {
        let (Some(labeller), Some(key)) = (&self.labeller, run.key()) else {
            return;
        };
        let key = key.to_string();
        let (backend, model, task) = (
            labeller.backend.clone(),
            labeller.model.clone(),
            labeller.task.clone(),
        );
        let (events, live) = (self.events.clone(), self.live.clone());
        tokio::spawn(async move {
            match inference_coder::activity::label(backend.as_ref(), &model, &task, &run).await {
                Ok(Some(label)) => {
                    let labels = BTreeMap::from([(key.clone(), label.clone())]);
                    if let Err(error) = Session::append_labels(&live.path, &labels) {
                        tracing::warn!("Failed keeping an activity label: {error:#}");
                    }
                    live.labels.lock().unwrap().extend(labels);
                    let _ = events.send(json!({"type": "label", "id": key, "label": label}));
                }
                Ok(None) => {}
                Err(error) => tracing::debug!("No activity label: {error:#}"),
            }
        });
    }

    fn emit(&self, event: Value) -> bool {
        let sent = self.events.send(event).is_ok();
        if !sent {
            self.live.interrupt.trigger();
        }
        sent
    }
}

impl AgentUi for WebUi {
    fn text(&mut self, delta: &str) {
        if let Some(run) = self.runs.text(delta) {
            self.label(run);
        }
        self.emit(json!({"type": "text", "delta": delta}));
    }

    fn reasoning(&mut self, delta: &str) {
        self.runs.reasoning(delta);
        self.emit(json!({"type": "reasoning", "delta": delta}));
    }

    fn tool_start(&mut self, call: &ToolCall, args: &Value) {
        self.runs.call(&call.id, &call.name, args);
        self.emit(json!({"type": "tool_start", "id": call.id, "name": call.name, "args": args}));
    }

    fn tool_end(&mut self, call: &ToolCall, outcome: &ToolOutcome) {
        self.emit(
            json!({"type": "tool_end", "id": call.id, "text": outcome.text,
            "is_error": outcome.is_error}),
        );
    }

    fn approve<'a>(
        &'a mut self,
        call: &'a ToolCall,
        args: &'a Value,
        kind: ToolKind,
    ) -> BoxFuture<'a, Approval> {
        self.runs.call(&call.id, &call.name, args);
        Box::pin(async move {
            let approval_id = uuid::Uuid::new_v4().simple().to_string();
            let (sender, receiver) = oneshot::channel();
            self.live
                .approvals
                .lock()
                .unwrap()
                .insert(approval_id.clone(), sender);
            let kind = match kind {
                ToolKind::Read => "read",
                ToolKind::Edit => "edit",
                ToolKind::Execute => "execute",
            };
            let event = json!({"type": "approval", "approval_id": approval_id, "id": call.id,
                "name": call.name, "args": args, "kind": kind});
            if !self.emit(event) {
                self.live
                    .approvals
                    .lock()
                    .unwrap()
                    .shift_remove(&approval_id);
                return Approval::Deny;
            }
            receiver.await.unwrap_or(Approval::Deny)
        })
    }

    fn round_end(&mut self, turn: &Turn) {
        if let Some(usage) = &turn.usage {
            self.emit(json!({"type": "usage", "usage": usage}));
        }
    }
}

#[derive(inference_json::FromJson)]
pub(crate) struct ChangesQuery {
    session_id: String,
    #[serde(default)]
    path: String,
}

// file reads, a project scan and git run off the async threads
async fn with_changes<T: serde::Serialize + Send + 'static>(
    code: &CodeState,
    id: &str,
    read: impl FnOnce(&Changes) -> anyhow::Result<Option<T>> + Send + 'static,
) -> Response {
    let Some(live) = code.live(id) else {
        return not_found(id);
    };
    let changes = live.changes.clone();
    match tokio::task::spawn_blocking(move || read(&changes.lock().unwrap())).await {
        Ok(Ok(Some(value))) => Json(value).into_response(),
        Ok(Ok(None)) => bad_request("not a file or directory in this project").into_response(),
        Ok(Err(error)) => (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response(),
        Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response(),
    }
}

/// The files the session has changed, with their line counts.
pub(crate) async fn changes(
    Extension(code): Extension<Arc<CodeState>>,
    QueryParams(query): QueryParams<ChangesQuery>,
) -> Response {
    with_changes(&code, &query.session_id, |changes| {
        changes.list().map(|files| Some(json!({"files": files})))
    })
    .await
}

/// One file's diff against its copy from before the session changed it.
pub(crate) async fn diff(
    Extension(code): Extension<Arc<CodeState>>,
    QueryParams(query): QueryParams<ChangesQuery>,
) -> Response {
    with_changes(&code, &query.session_id, move |changes| {
        changes.diff(&query.path)
    })
    .await
}

/// One directory of the project, for the file tree.
pub(crate) async fn tree(
    Extension(code): Extension<Arc<CodeState>>,
    QueryParams(query): QueryParams<ChangesQuery>,
) -> Response {
    with_changes(&code, &query.session_id, move |changes| {
        Ok(changes
            .entries(&query.path)
            .map(|entries| json!({"entries": entries})))
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::make_demo_project;

    #[test]
    fn the_demo_project_fails_its_tests_as_planted() {
        let project = make_demo_project().unwrap();
        let suite = std::process::Command::new("python3")
            .args(["-m", "unittest", "discover", "-s", "tests", "-t", "."])
            .current_dir(&project)
            .output();
        std::fs::remove_dir_all(&project).unwrap();
        // a machine without python can still make the demo; it just can't grade it
        let Ok(suite) = suite else {
            return;
        };
        let report = String::from_utf8_lossy(&suite.stderr);
        assert!(report.contains("Ran 19 tests"), "{report}");
        assert!(report.contains("FAILED (failures=7, errors=7)"), "{report}");
    }
}
