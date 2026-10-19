//! The Code tab: `inference_coder`'s agent over this server's engine, working in projects under the code root, or in a
//! demo project it makes in the temp directory.

use std::collections::HashSet;
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, UNIX_EPOCH};

use axum::Json;
use axum::extract::{Extension, Query};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::future::BoxFuture;
use indexmap::IndexMap;
use inference_coder::{
    Agent, AgentConfig, AgentUi, Approval, DEFAULT_MAX_REPLY_TOKENS, DEFAULT_MAX_ROUNDS,
    EngineBackend, Interrupt, PermissionMode, Permissions, Session, ToolCall, ToolKind,
    ToolOutcome, Turn, TurnEnd, default_tools, prompt::system_prompt,
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tokio::sync::{mpsc, oneshot};

use crate::types::AppState;

// sessions kept loaded at once; the least recently opened idle one goes first
const MAX_LIVE_SESSIONS: usize = 32;
const KEEP_ALIVE: Duration = Duration::from_secs(15);
const DEMO_DIR_PREFIX: &str = "inference-code-demo-";
const DEMO_PROMPT: &str =
    "The tests in test_stats.py fail. Fix stats.py so they pass, then run them.";
// two planted bugs: mean is off by one, and median ignores even-length lists
const DEMO_STATS: &str = "def mean(values):\n    return sum(values) / len(values) + 1\n\n\n\
def median(values):\n    ordered = sorted(values)\n    mid = len(ordered) // 2\n    return ordered[mid]\n";
const DEMO_TESTS: &str = "from stats import mean, median\n\nassert mean([1, 2, 3]) == 2\n\
assert median([1, 2, 3, 4]) == 2.5\nassert median([3, 1, 2]) == 2\nprint(\"ok\")\n";

/// The projects the Code tab may open and the sessions it has loaded.
pub(crate) struct CodeState {
    root: Option<PathBuf>,
    /// Demo projects made since the server started, which may be opened although they are outside the root.
    demos: Mutex<HashSet<PathBuf>>,
    sessions_dir: PathBuf,
    live: Mutex<IndexMap<String, Arc<LiveSession>>>,
}

struct LiveSession {
    agent: Arc<tokio::sync::Mutex<Agent>>,
    interrupt: Interrupt,
    approvals: Mutex<IndexMap<String, oneshot::Sender<Approval>>>,
}

impl CodeState {
    pub(crate) fn new(root: Option<&Path>) -> std::io::Result<Self> {
        Ok(Self {
            root: root.map(Path::canonicalize).transpose()?,
            demos: Mutex::new(HashSet::new()),
            sessions_dir: inference_coder::session::default_dir(),
            live: Mutex::new(IndexMap::new()),
        })
    }

    pub(crate) fn root(&self) -> Option<&Path> {
        self.root.as_deref()
    }

    fn allows(&self, path: &Path) -> bool {
        self.root
            .as_ref()
            .is_some_and(|root| path.starts_with(root))
            || self.demos.lock().unwrap().contains(path)
    }

    /// `project` (relative to the code root, or absolute) as a directory the Code tab may open.
    fn project(&self, project: &str) -> Result<PathBuf, Refusal> {
        let path = match &self.root {
            Some(root) => root.join(project.trim()),
            None => PathBuf::from(project.trim()),
        };
        let path = path
            .canonicalize()
            .map_err(|error| bad_request(format!("{}: {error}", path.display())))?;
        if !path.is_dir() || !self.allows(&path) {
            let reason = match &self.root {
                Some(root) => format!("is not a directory inside the code root {}", root.display()),
                None => "can't be opened: the server has no code root (--code-root)".to_string(),
            };
            return Err(bad_request(format!("{} {reason}", path.display())));
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

#[derive(Deserialize)]
pub(crate) struct SessionsQuery {
    #[serde(default)]
    project: String,
}

/// The project's sessions, newest first.
pub(crate) async fn list_sessions(
    Extension(code): Extension<Arc<CodeState>>,
    Query(query): Query<SessionsQuery>,
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

#[derive(Deserialize)]
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
    Json(request): Json<OpenRequest>,
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
                    return bad_request("that session's project is outside the code root")
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
    }))
    .into_response()
}

/// Makes a demo project in the temp directory (two planted bugs and the tests that catch them) and opens a session
/// there; the page then sends `prompt` like any other.
pub(crate) async fn start_demo(
    Extension(code): Extension<Arc<CodeState>>,
    Extension(app): Extension<Arc<AppState>>,
    Json(request): Json<DemoRequest>,
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

#[derive(Deserialize)]
pub(crate) struct DemoRequest {
    #[serde(default)]
    model: Option<String>,
}

fn make_demo_project() -> std::io::Result<PathBuf> {
    let id = uuid::Uuid::new_v4().simple().to_string();
    let project = std::env::temp_dir().join(format!("{DEMO_DIR_PREFIX}{}", &id[..8]));
    std::fs::create_dir_all(&project)?;
    std::fs::write(project.join("stats.py"), DEMO_STATS)?;
    std::fs::write(project.join("test_stats.py"), DEMO_TESTS)?;
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
        agent: Arc::new(tokio::sync::Mutex::new(agent)),
        interrupt: Interrupt::default(),
        approvals: Mutex::new(IndexMap::new()),
    });
    code.insert(id, live.clone());
    Ok(live)
}

#[derive(Deserialize)]
pub(crate) struct SendRequest {
    session_id: String,
    prompt: String,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    enable_thinking: Option<bool>,
    #[serde(default)]
    model: Option<String>,
}

/// Runs a prompt to its answer, streaming the agent's text, tool calls and approval requests as SSE.
pub(crate) async fn send(
    Extension(code): Extension<Arc<CodeState>>,
    Extension(app): Extension<Arc<AppState>>,
    Json(request): Json<SendRequest>,
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
    agent.set_model(model);
    agent.set_request_field("enable_thinking", request.enable_thinking.map(Value::Bool));
    live.interrupt.reset();
    let (events, receiver) = mpsc::unbounded_channel::<Value>();
    tokio::spawn(async move {
        let mut ui = WebUi {
            events: events.clone(),
            live: live.clone(),
        };
        let result = agent.send(&request.prompt, &mut ui, &live.interrupt).await;
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

#[derive(Deserialize)]
pub(crate) struct ApproveRequest {
    session_id: String,
    approval_id: String,
    /// once, always or deny
    decision: String,
}

pub(crate) async fn approve(
    Extension(code): Extension<Arc<CodeState>>,
    Json(request): Json<ApproveRequest>,
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

#[derive(Deserialize)]
pub(crate) struct InterruptRequest {
    session_id: String,
}

pub(crate) async fn interrupt(
    Extension(code): Extension<Arc<CodeState>>,
    Json(request): Json<InterruptRequest>,
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
}

impl WebUi {
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
        self.emit(json!({"type": "text", "delta": delta}));
    }

    fn reasoning(&mut self, delta: &str) {
        self.emit(json!({"type": "reasoning", "delta": delta}));
    }

    fn tool_start(&mut self, call: &ToolCall, args: &Value) {
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
