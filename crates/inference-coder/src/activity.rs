//! Labels for the agent's activity: what a run of its tool calls was for, in a few words the session's model writes
//! after a prompt is answered. A front end shows its own summary of the calls until a label arrives, and after a
//! reply it can't use.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use futures::StreamExt;
use serde_json::Value;

use crate::backend::ChatBackend;
use crate::turn::Turn;

pub const LABEL_WORDS: usize = 8;
// a reply's label past this is cut, so a rambling one can't crowd the line it heads
const LABEL_CHARS: usize = 80;
pub(crate) const TOKENS_PER_LABEL: usize = 24;
// room for a thinking model's reasoning before the labels; turning thinking off would reword its system prompt
pub(crate) const THINKING_TOKENS: usize = 1024;
// how much of a call's main argument the instruction quotes
const ARG_CHARS: usize = 80;
const ARG_KEYS: [&str; 4] = ["path", "command", "pattern", "url"];
const LABEL_EXAMPLE: &str = "Checked how undo should behave";

/// The groups whose calls are all in `messages`: an interrupted reply's calls never reach the session.
pub fn recorded_groups(messages: &[Value], groups: Vec<Vec<String>>) -> Vec<Vec<String>> {
    let ids = call_ids(messages);
    groups
        .into_iter()
        .filter(|group| !group.is_empty() && group.iter().all(|id| ids.contains(id.as_str())))
        .collect()
}

fn call_ids(messages: &[Value]) -> std::collections::HashSet<&str> {
    messages
        .iter()
        .filter_map(|message| message.get("tool_calls")?.as_array())
        .flatten()
        .filter_map(|call| call.get("id")?.as_str())
        .collect()
}

/// The user message asking for one label per group, each group the ids of a run of tool calls in `messages`; None
/// when a group is empty or names a call `messages` doesn't hold.
pub fn instruction(messages: &[Value], groups: &[Vec<String>]) -> Option<String> {
    let calls = messages
        .iter()
        .filter_map(|message| message.get("tool_calls")?.as_array())
        .flatten()
        .filter_map(|call| Some((call.get("id")?.as_str()?, call.get("function")?)))
        .collect::<BTreeMap<_, _>>();
    let mut lines = Vec::with_capacity(groups.len());
    for (n, group) in groups.iter().enumerate() {
        if group.is_empty() {
            return None;
        }
        let described = group
            .iter()
            .map(|id| calls.get(id.as_str()).map(|function| describe(function)))
            .collect::<Option<Vec<_>>>()?;
        lines.push(format!("{}. {}", n + 1, described.join(", ")));
    }
    Some(format!(
        "Label each numbered group of your tool calls above with what it was for, as a past-tense phrase of at most \
         {LABEL_WORDS} words (for example \"{LABEL_EXAMPLE}\"). Answer at once with only a JSON array of {} strings, in order.\n{}",
        groups.len(),
        lines.join("\n")
    ))
}

fn describe(function: &Value) -> String {
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("tool");
    let args = function
        .get("arguments")
        .and_then(Value::as_str)
        .and_then(|args| serde_json::from_str::<Value>(args).ok());
    let arg = ARG_KEYS
        .iter()
        .find_map(|key| args.as_ref()?.get(*key)?.as_str())
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

/// The labels in a reply: a JSON array of exactly `groups` strings, wherever it sits in the text.
pub fn parse(reply: &str, groups: usize) -> Option<Vec<String>> {
    let start = reply.find('[')?;
    let end = reply.rfind(']')?;
    let labels = serde_json::from_str::<Vec<String>>(reply.get(start..=end)?).ok()?;
    if labels.len() != groups {
        return None;
    }
    labels
        .into_iter()
        .map(|label| {
            let label = label.trim().trim_end_matches('.').trim();
            (!label.is_empty()).then(|| label.chars().take(LABEL_CHARS).collect())
        })
        .collect()
}

/// The text of a streamed reply to `request`.
pub async fn reply_text(backend: &dyn ChatBackend, request: Value) -> Result<String> {
    let mut stream = backend.stream(request).await?;
    let mut turn = Turn::default();
    while let Some(chunk) = stream.next().await {
        turn.apply(&chunk?);
    }
    if turn.finish_reason.is_none() {
        bail!("the label reply ended before it finished");
    }
    Ok(turn.content)
}

#[cfg(test)]
mod tests {
    use super::{instruction, parse, recorded_groups};
    use serde_json::json;

    fn call(id: &str, name: &str, args: serde_json::Value) -> serde_json::Value {
        json!({"id": id, "type": "function", "function": {"name": name, "arguments": args.to_string()}})
    }

    #[test]
    fn the_instruction_lists_each_group_by_its_calls() {
        let messages = vec![
            json!({"role": "assistant", "content": "", "tool_calls": [
                call("c1", "read", json!({"path": "src/undo.py"})),
                call("c2", "bash", json!({"command": "python3 -m unittest\nmore"})),
            ]}),
            json!({"role": "assistant", "content": "", "tool_calls": [call("c3", "glob", json!({}))]}),
        ];
        let groups = vec![
            vec!["c1".to_string(), "c2".to_string()],
            vec!["c3".to_string()],
        ];
        let text = instruction(&messages, &groups).unwrap();
        assert!(
            text.ends_with("\n1. read(src/undo.py), bash(python3 -m unittest)\n2. glob"),
            "{text}"
        );
        assert!(instruction(&messages, &[vec!["missing".to_string()]]).is_none());
        let kept = recorded_groups(
            &messages,
            vec![
                vec!["c3".to_string()],
                vec!["c1".to_string(), "gone".to_string()],
            ],
        );
        assert_eq!(kept, [vec!["c3".to_string()]]);
    }

    #[test]
    fn a_reply_parses_only_as_one_label_a_group() {
        assert_eq!(
            parse("Sure:\n[\"Found the failing test.\", \" Fixed undo \"]", 2),
            Some(vec![
                "Found the failing test".to_string(),
                "Fixed undo".to_string()
            ])
        );
        assert_eq!(parse("[\"one\"]", 2), None);
        assert_eq!(parse("no labels here", 1), None);
        assert_eq!(parse("[\"\"]", 1), None);
    }
}
