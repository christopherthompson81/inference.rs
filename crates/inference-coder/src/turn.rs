//! One streamed assistant turn, assembled from OpenAI chunk deltas.

use serde_json::{Value, json};

/// A tool call the model made: its id, the tool's name and the JSON arguments as streamed.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// The assistant's reply: text, reasoning, tool calls, why it stopped and the server's usage report.
#[derive(Default, Debug)]
pub struct Turn {
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<String>,
    pub usage: Option<Value>,
}

/// What one chunk added, for a UI that streams it.
#[derive(Default, Debug, PartialEq)]
pub struct Delta {
    pub content: Option<String>,
    pub reasoning: Option<String>,
}

impl Turn {
    /// Folds one chunk in. Tool call fragments merge by index: ids and names arrive once, arguments in pieces (a server
    /// may also send a whole call in one chunk).
    pub fn apply(&mut self, chunk: &Value) -> Delta {
        if let Some(usage) = chunk.get("usage").filter(|usage| !usage.is_null()) {
            self.usage = Some(usage.clone());
        }
        let Some(choice) = chunk.get("choices").and_then(|c| c.get(0)) else {
            return Delta::default();
        };
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish_reason = Some(reason.to_string());
        }
        let Some(delta) = choice.get("delta") else {
            return Delta::default();
        };
        let text = |key: &str| {
            delta
                .get(key)
                .and_then(Value::as_str)
                .filter(|t| !t.is_empty())
        };
        let content = text("content").map(str::to_string);
        let reasoning = text("reasoning_content")
            .or_else(|| text("reasoning"))
            .map(str::to_string);
        if let Some(content) = &content {
            self.content.push_str(content);
        }
        if let Some(reasoning) = &reasoning {
            self.reasoning.push_str(reasoning);
        }
        for call in delta
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let index = call
                .get("index")
                .and_then(Value::as_u64)
                .and_then(|i| usize::try_from(i).ok())
                .unwrap_or(self.tool_calls.len());
            while self.tool_calls.len() <= index {
                self.tool_calls.push(ToolCall {
                    id: String::new(),
                    name: String::new(),
                    arguments: String::new(),
                });
            }
            let slot = &mut self.tool_calls[index];
            if let Some(id) = call
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
            {
                slot.id = id.to_string();
            }
            if let Some(function) = call.get("function") {
                if let Some(name) = function
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|n| !n.is_empty())
                {
                    slot.name = name.to_string();
                }
                if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
                    slot.arguments.push_str(arguments);
                }
            }
        }
        Delta { content, reasoning }
    }

    /// The turn as a history message. Reasoning goes back in, so a template that renders it re-renders this turn as
    /// generated and the next request reuses its cached prefix.
    pub fn message(&self) -> Value {
        let mut message = json!({"role": "assistant", "content": self.content});
        if !self.reasoning.is_empty() {
            message["reasoning_content"] = Value::String(self.reasoning.clone());
        }
        if !self.tool_calls.is_empty() {
            message["tool_calls"] = self
                .tool_calls
                .iter()
                .map(|call| {
                    json!({"id": call.id, "type": "function",
                           "function": {"name": call.name, "arguments": call.arguments}})
                })
                .collect();
        }
        message
    }
}

#[cfg(test)]
mod tests {
    use super::{ToolCall, Turn};
    use serde_json::json;

    #[test]
    fn deltas_merge_into_text_reasoning_and_tool_calls() {
        let mut turn = Turn::default();
        turn.apply(&json!({"choices": [{"delta": {"reasoning_content": "think"}}]}));
        let delta = turn.apply(&json!({"choices": [{"delta": {"content": "Hi"}}]}));
        assert_eq!(delta.content.as_deref(), Some("Hi"));
        turn.apply(&json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "read", "arguments": "{\"pa"}}]}}]}));
        turn.apply(&json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "th\":\"a\"}"}}]}}]}));
        turn.apply(&json!({"choices": [{"delta": {"tool_calls": [
            {"index": 1, "id": "call_2", "function": {"name": "bash", "arguments": "{}"}}]},
            "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 7}}));
        assert_eq!(
            (turn.content.as_str(), turn.reasoning.as_str()),
            ("Hi", "think")
        );
        assert_eq!(
            turn.tool_calls,
            [
                ToolCall {
                    id: "call_1".into(),
                    name: "read".into(),
                    arguments: "{\"path\":\"a\"}".into()
                },
                ToolCall {
                    id: "call_2".into(),
                    name: "bash".into(),
                    arguments: "{}".into()
                },
            ]
        );
        assert_eq!(turn.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(turn.usage, Some(json!({"prompt_tokens": 7})));
        let message = turn.message();
        assert_eq!(message["reasoning_content"], "think");
        assert_eq!(message["tool_calls"][1]["function"]["name"], "bash");
    }
}
