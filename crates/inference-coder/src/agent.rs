//! The tool loop: send the conversation, stream the reply, run the tools it calls, repeat until it answers.

use std::pin::pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, bail};
use futures::StreamExt;
use futures::future::BoxFuture;
use serde_json::{Map, Value, json};
use tokio::sync::Notify;

use crate::backend::ChatBackend;
use crate::permission::{Approval, Decision, Permissions};
use crate::session::Session;
use crate::tools::{Tool, ToolContext, ToolKind, ToolOutcome};
use crate::turn::{ToolCall, Turn};

// a runaway loop stops here; a real task rarely needs a tenth of it
pub const DEFAULT_MAX_ROUNDS: usize = 200;
const INTERRUPTED: &str = "interrupted by the user";
const DECLINED: &str = "the user declined this call";

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
    /// One model reply finished; its usage says how full the context is.
    fn round_end(&mut self, turn: &Turn) {
        let _ = turn;
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
}

pub struct AgentConfig {
    /// The model name requests carry.
    pub model: String,
    pub max_rounds: usize,
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
            backend,
            tools,
            ctx: ToolContext {
                root: session.root.clone(),
            },
            permissions,
            session,
            config,
        }
    }

    pub fn backend(&self) -> &dyn ChatBackend {
        self.backend.as_ref()
    }

    /// Runs one user prompt to its answer. Messages reach the session a round at a time, each assistant reply with its
    /// tool results, so the history on disk never holds a call without a result.
    pub async fn send(
        &mut self,
        prompt: &str,
        ui: &mut dyn AgentUi,
        interrupt: &Interrupt,
    ) -> Result<TurnEnd> {
        let mut pending = vec![json!({"role": "user", "content": prompt})];
        for _ in 0..self.config.max_rounds {
            let (turn, interrupted) = self.stream_turn(&pending, ui, interrupt).await?;
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
            pending.push(turn.message());
            if turn.tool_calls.is_empty() {
                self.commit(&mut pending)?;
                return Ok(TurnEnd::Done);
            }
            for call in &turn.tool_calls {
                let outcome = if interrupt.is_set() {
                    ToolOutcome::error(INTERRUPTED)
                } else {
                    self.run_call(call, ui, interrupt).await
                };
                pending.push(
                    json!({"role": "tool", "tool_call_id": call.id, "content": outcome.text}),
                );
            }
            self.commit(&mut pending)?;
            if interrupt.is_set() {
                return Ok(TurnEnd::Interrupted);
            }
        }
        Ok(TurnEnd::RoundLimit)
    }

    fn commit(&mut self, pending: &mut Vec<Value>) -> Result<()> {
        for message in pending.drain(..) {
            self.session.append(message)?;
        }
        Ok(())
    }

    fn request(&self, pending: &[Value]) -> Value {
        let mut request = self.config.request.clone();
        let messages = self
            .session
            .messages
            .iter()
            .chain(pending)
            .cloned()
            .collect();
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
        ui.tool_start(call, &args);
        let outcome = tokio::select! {
            outcome = tool.run(args, &self.ctx) => outcome,
            () = interrupt.wait() => ToolOutcome::error(INTERRUPTED),
        };
        ui.tool_end(call, &outcome);
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::{Agent, AgentConfig, AgentUi, Interrupt, TurnEnd};
    use crate::backend::ChatBackend;
    use crate::permission::{Approval, PermissionMode, Permissions};
    use crate::session::Session;
    use crate::tools::{ToolKind, ToolOutcome, default_tools};
    use crate::turn::{ToolCall, Turn};
    use anyhow::Result;
    use futures::future::BoxFuture;
    use futures::stream::BoxStream;
    use serde_json::{Map, Value, json};
    use std::sync::{Arc, Mutex};

    // replays canned replies and records the requests it was sent
    struct Scripted {
        replies: Mutex<Vec<Vec<Value>>>,
        requests: Mutex<Vec<Value>>,
    }

    impl ChatBackend for Scripted {
        fn stream(
            &self,
            request: Value,
        ) -> BoxFuture<'_, Result<BoxStream<'static, Result<Value>>>> {
            self.requests.lock().unwrap().push(request);
            let chunks = self.replies.lock().unwrap().remove(0);
            Box::pin(async move {
                Ok(Box::pin(futures::stream::iter(chunks.into_iter().map(Ok))) as BoxStream<_>)
            })
        }

        fn describe(&self) -> String {
            "scripted".into()
        }
    }

    #[derive(Default)]
    struct Recorder {
        text: String,
        tools: Vec<(String, bool)>,
        approvals: usize,
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
    }

    fn call(id: &str, name: &str, args: Value) -> Value {
        json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": id, "function": {"name": name, "arguments": args.to_string()}}]},
            "finish_reason": "tool_calls"}]})
    }

    #[tokio::test]
    async fn runs_tools_until_the_model_answers_and_keeps_the_history() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("project");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("notes.txt"), "old\n").unwrap();
        let backend = Arc::new(Scripted {
            replies: Mutex::new(vec![
                vec![call("c1", "read", json!({"path": "notes.txt"}))],
                vec![call(
                    "c2",
                    "edit",
                    json!({"path": "notes.txt", "old_string": "old", "new_string": "new"}),
                )],
                vec![
                    json!({"choices": [{"delta": {"content": "Done."}, "finish_reason": "stop"}]}),
                ],
            ]),
            requests: Mutex::new(Vec::new()),
        });
        let session = Session::create(&dir.path().join("sessions"), &root, "m", "sys").unwrap();
        let config = AgentConfig {
            model: "m".into(),
            max_rounds: 10,
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
}
