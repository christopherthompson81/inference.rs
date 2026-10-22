//! Labels for the agent's activity: what a run of its thinking and tool calls was for, in a few words the session's
//! model writes as soon as the run closes. The label request is small (the task, the run's thinking and calls, no
//! history, no thinking of its own, a few tokens), so it shares only a handful of the agent's steps. A front end shows
//! its own summary of the calls until a label arrives, and after a reply it can't use.

use anyhow::{Result, bail};
use futures::StreamExt;
use serde_json::{Value, json};

use crate::backend::ChatBackend;
use crate::turn::Turn;

pub const LABEL_WORDS: usize = 10;
const LABEL_TOKENS: usize = 20;
// a reply's label past this is cut, so a rambling one can't crowd the line it heads
const LABEL_CHARS: usize = 80;
// the end of a run's thinking the request quotes, where the reasoning toward its calls is
const THINKING_CHARS: usize = 1500;
// how much of a call's main argument the request quotes
const ARG_CHARS: usize = 80;
const ARG_KEYS: [&str; 4] = ["path", "command", "pattern", "url"];
const LABEL_EXAMPLE: &str = "Checked how undo should behave";

/// A run of thinking and tool calls that visible text, the next prompt or the end of the turn closed.
#[derive(Debug, Default)]
pub struct Run {
    pub thinking: String,
    /// Each call's id and how the request names it, e.g. `read(src/undo.py)`.
    pub calls: Vec<(String, String)>,
}

impl Run {
    /// The id the run's label is kept and shown under: its first call's.
    pub fn key(&self) -> Option<&str> {
        self.calls.first().map(|(id, _)| id.as_str())
    }
}

/// Follows an agent's stream and hands back each run of thinking and tool calls as it closes, folded as the Code tab
/// folds them: a run takes the thinking and calls until text the user sees.
#[derive(Default)]
pub struct RunTracker {
    run: Run,
}

impl RunTracker {
    pub fn reasoning(&mut self, delta: &str) {
        self.run.thinking.push_str(delta);
    }

    pub fn call(&mut self, id: &str, name: &str, args: &Value) {
        if self.run.calls.iter().all(|(known, _)| known != id) {
            self.run.calls.push((id.to_string(), describe(name, args)));
        }
    }

    /// Visible text closes the run before it; thinking with no call after it belongs to no run.
    pub fn text(&mut self, delta: &str) -> Option<Run> {
        if delta.trim().is_empty() {
            return None;
        }
        self.finish()
    }

    /// The run still open when the turn ends.
    pub fn finish(&mut self) -> Option<Run> {
        let run = std::mem::take(&mut self.run);
        (!run.calls.is_empty()).then_some(run)
    }
}

fn describe(name: &str, args: &Value) -> String {
    let arg = ARG_KEYS
        .iter()
        .find_map(|key| args.get(*key)?.as_str())
        .map(|arg| {
            arg.lines()
                .next()
                .unwrap_or_default()
                .chars()
                .take(ARG_CHARS)
                .collect::<String>()
        });
    match arg {
        Some(arg) => format!("{name}({arg})"),
        None => name.to_string(),
    }
}

/// The request for `run`'s label: the task it serves, the end of its thinking and its calls, answered with thinking
/// off in at most a few tokens.
pub fn label_request(model: &str, task: &str, run: &Run) -> Value {
    let thinking = run.thinking.trim();
    let skip = thinking.chars().count().saturating_sub(THINKING_CHARS);
    let thinking = thinking.chars().skip(skip).collect::<String>();
    let mut parts = vec![format!("Task: {task}")];
    if !thinking.is_empty() {
        parts.push(format!("Reasoning (abridged): {thinking}"));
    }
    let calls = run
        .calls
        .iter()
        .map(|(_, call)| call.as_str())
        .collect::<Vec<_>>();
    parts.push(format!("Tool calls: {}", calls.join(", ")));
    // framed as text to label, so a model doesn't carry the task on instead
    let content = format!(
        "Here is a step of a coding agent's work:\n\n<activity>\n{}\n</activity>\n\nIn at most {LABEL_WORDS} words, in \
         the past tense and without \"I\", say what these tool calls were for (for example \"{LABEL_EXAMPLE}\"). \
         Reply with only the phrase.",
        parts.join("\n\n")
    );
    json!({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": LABEL_TOKENS,
        "temperature": 0,
        "enable_thinking": false,
        "stream": true,
    })
}

/// The label in a reply: its first line, without quotes or a closing period; None when nothing is left.
pub fn clean_label(reply: &str) -> Option<String> {
    let line = reply.trim().lines().next()?;
    let label = line
        .trim()
        .trim_matches(|c: char| c == '"' || c == '\'' || c == '*')
        .trim_end_matches('.')
        .trim();
    (!label.is_empty()).then(|| label.chars().take(LABEL_CHARS).collect())
}

/// The label for `run` from the session's model.
pub async fn label(
    backend: &dyn ChatBackend,
    model: &str,
    task: &str,
    run: &Run,
) -> Result<Option<String>> {
    let mut stream = backend.stream(label_request(model, task, run)).await?;
    let mut turn = Turn::default();
    while let Some(chunk) = stream.next().await {
        turn.apply(&chunk?);
    }
    if turn.finish_reason.is_none() {
        bail!("the label reply ended before it finished");
    }
    Ok(clean_label(&turn.content))
}

#[cfg(test)]
mod tests {
    use super::{RunTracker, clean_label, label_request};
    use serde_json::json;

    #[test]
    fn a_run_closes_at_visible_text_or_the_end_of_the_turn() {
        let mut tracker = RunTracker::default();
        tracker.reasoning("The README says undo pops the last change.");
        assert!(tracker.text("   ").is_none());
        tracker.call("c1", "read", &json!({"path": "src/undo.py"}));
        tracker.call("c1", "read", &json!({"path": "src/undo.py"}));
        tracker.call(
            "c2",
            "bash",
            &json!({"command": "python3 -m unittest\nmore"}),
        );
        let run = tracker.text("Found it.").unwrap();
        assert_eq!(run.key(), Some("c1"));
        assert_eq!(run.calls[1].1, "bash(python3 -m unittest)");
        assert!(run.thinking.starts_with("The README"));
        // thinking with no call after it is no run
        tracker.reasoning("Done, I think.");
        assert!(tracker.text("All tests pass.").is_none());
        tracker.call("c3", "glob", &json!({}));
        assert_eq!(tracker.finish().unwrap().calls[0].1, "glob");
        assert!(tracker.finish().is_none());
    }

    #[test]
    fn the_request_carries_the_run_and_asks_for_few_tokens_without_thinking() {
        let mut tracker = RunTracker::default();
        tracker.reasoning(&"x".repeat(5000));
        tracker.call("c1", "read", &json!({"path": "README.md"}));
        let request = label_request("m", "fix the tests", &tracker.finish().unwrap());
        let prompt = request["messages"][0]["content"].as_str().unwrap();
        assert!(prompt.contains("fix the tests") && prompt.contains("read(README.md)"));
        assert!(prompt.len() < 2000, "the thinking is trimmed");
        assert_eq!(request["enable_thinking"], json!(false));
        assert_eq!(request["max_tokens"], json!(20));
    }

    #[test]
    fn a_label_keeps_its_first_line_without_quotes_or_period() {
        assert_eq!(
            clean_label("\"Found the failing test.\"\nmore").as_deref(),
            Some("Found the failing test")
        );
        assert_eq!(clean_label("  \n"), None);
    }
}
