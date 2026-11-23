//! Inkling 2 tool call parser.
//!
//! Format: `<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>`, one block
//! per call. The reasoning parser drops the `<|message_model|>` markers, so the name the block opens with runs on from
//! any text before it; the JSON names the tool again, which is where the name is read.

use llguidance::api::TopLevelGrammar;
use serde_json::Value;

use super::ToolFormatParser;
use crate::Tool;

pub(crate) const INVOKE: &str = "<|content_invoke_tool_json|>";
pub(crate) const MESSAGE_MODEL: &str = "<|message_model|>";
pub(crate) const END_MESSAGE: &str = "<|end_message|>";
const ARGS_KEY: &str = "args";

pub struct InklingParser;

/// One call's span in the text (its leading name through its JSON and closing marker) and its name and arguments.
struct Call {
    start: usize,
    end: usize,
    name: String,
    args: Value,
}

/// Every complete call in `message`, or None when a call is still open or isn't Inkling's JSON.
fn calls(message: &str) -> Option<Vec<Call>> {
    let mut calls = Vec::new();
    let mut from = 0;
    while let Some(found) = message[from..].find(INVOKE) {
        let invoke = from + found;
        let body = &message[invoke + INVOKE.len()..];
        let mut stream = serde_json::Deserializer::from_str(body).into_iter::<Value>();
        let value = stream.next()?.ok()?;
        let mut end = invoke + INVOKE.len() + stream.byte_offset();
        let name = value.get("name")?.as_str()?.to_string();
        let args = value
            .get(ARGS_KEY)
            .cloned()
            .unwrap_or(Value::Object(Default::default()));
        let before = &message[from..invoke];
        let before = before.strip_suffix(name.as_str()).unwrap_or(before);
        let before = before.strip_suffix(MESSAGE_MODEL).unwrap_or(before);
        let rest = &message[end..];
        if let Some(stripped) = rest.trim_start().strip_prefix(END_MESSAGE) {
            end = message.len() - stripped.len();
        }
        calls.push(Call {
            start: from + before.len(),
            end,
            name,
            args,
        });
        from = end;
    }
    (!calls.is_empty()).then_some(calls)
}

/// `message` without its calls, for the reply's text.
pub(crate) fn strip_calls(message: &str) -> String {
    let Some(calls) = calls(message) else {
        return message
            .find(INVOKE)
            .map_or_else(|| message.to_string(), |pos| message[..pos].to_string());
    };
    let mut out = String::new();
    let mut from = 0;
    for call in calls {
        out.push_str(&message[from..call.start]);
        from = call.end;
    }
    out.push_str(&message[from..]);
    out
}

fn tool_names(tools: &[Tool]) -> String {
    let names = tools
        .iter()
        .map(|tool| serde_json::to_string(&tool.function.name).expect("a string serializes"))
        .collect::<Vec<_>>();
    if names.is_empty() {
        r#""""#.to_string()
    } else {
        format!("({})", names.join(" | "))
    }
}

impl ToolFormatParser for InklingParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        text.contains(INVOKE)
    }

    fn format(&self) -> super::ToolCallFormat {
        super::ToolCallFormat::Inkling
    }

    fn tool_call_grammar(&self, tools: &[Tool], _text: &str) -> TopLevelGrammar {
        crate::tools::grammar::build_json_format_grammar(
            format!(r#"start: @json_body "{END_MESSAGE}""#),
            tools,
            ARGS_KEY,
            false,
        )
    }

    fn required_tool_call_grammar(&self, tools: &[Tool]) -> TopLevelGrammar {
        crate::tools::grammar::build_json_format_grammar(
            format!(
                r#"start: "{MESSAGE_MODEL}" {} "{INVOKE}" @json_body "{END_MESSAGE}""#,
                tool_names(tools)
            ),
            tools,
            ARGS_KEY,
            false,
        )
    }

    fn parse(&self, message: &str) -> anyhow::Result<Option<String>> {
        let Some(calls) = calls(message) else {
            return Ok(None);
        };
        let calls = calls
            .into_iter()
            .map(|call| serde_json::json!({"name": call.name, "arguments": call.args}))
            .collect::<Vec<_>>();
        Ok(Some(serde_json::to_string(&calls)?))
    }
}

#[cfg(test)]
mod tests {
    use super::InklingParser;
    use crate::tools::parsers::{ToolFormatParser, extract_model_specific_message};

    #[test]
    fn parses_a_call_whose_name_runs_on_from_the_text() {
        let message = r#"Let me look.bash<|content_invoke_tool_json|>{"name":"bash","args":{"command":"ls"}}<|end_message|>"#;
        let (calls, content) = extract_model_specific_message(message).unwrap().unwrap();
        assert_eq!(calls, r#"[{"name":"bash","arguments":{"command":"ls"}}]"#);
        assert_eq!(content, "Let me look.");
    }

    #[test]
    fn parses_consecutive_calls() {
        let message = r#"read<|content_invoke_tool_json|>{"name":"read","args":{"path":"a"}}<|end_message|><|message_model|>grep<|content_invoke_tool_json|>{"name":"grep","args":{"pattern":"b"}}"#;
        let parsed = InklingParser.parse(message).unwrap().unwrap();
        assert_eq!(
            parsed,
            r#"[{"name":"read","arguments":{"path":"a"}},{"name":"grep","arguments":{"pattern":"b"}}]"#
        );
    }

    #[test]
    fn leaves_an_open_call_unparsed() {
        let message = r#"bash<|content_invoke_tool_json|>{"name":"bash","args":{"command":"l"#;
        assert!(InklingParser.parse(message).unwrap().is_none());
        assert!(InklingParser.could_be_tool_call(message));
    }
}
