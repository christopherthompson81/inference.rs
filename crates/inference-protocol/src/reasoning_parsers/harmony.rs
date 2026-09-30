//! OpenAI Harmony format parsing for GPT-OSS models.
//!
//! The Harmony format uses channels to separate different types of content:
//! - `analysis`: Chain-of-thought reasoning (internal, not for end users)
//! - `commentary`: Tool call preambles and explanations
//! - `final`: User-facing response content
//!
//! Tool calls in Harmony are indicated by the `recipient` field being set to:
//! - `functions.tool_name` for user-defined tools
//! - `browser.search`, `browser.open`, `browser.find` for browser tool
//! - `python` for python tool
//!
//! This module provides incremental parsing of Harmony-formatted token streams.

use uuid::Uuid;

const START: &[u8] = b"<|start|>";
const MESSAGE: &[u8] = b"<|message|>";
const CHANNEL: &str = "<|channel|>";
const CONSTRAIN: &str = "<|constrain|>";
// `<|end|>` closes a message, `<|call|>` a tool call, `<|return|>` the turn.
const MESSAGE_ENDS: &[&[u8]] = &[b"<|end|>", b"<|call|>", b"<|return|>"];
const ASSISTANT_ROLE: &str = "assistant";
const TOOL_ROLE: &str = "tool";
const HARMONY_ROLES: &[&str] = &["system", "developer", "user", "assistant", "tool"];

/// Extract the tool name from a recipient string.
/// - "functions.my_tool" -> "my_tool"
/// - "browser.search" -> "browser.search"
/// - "python" -> "python"
fn extract_tool_name(recipient: &str) -> String {
    if let Some(name) = recipient.strip_prefix("functions.") {
        name.to_string()
    } else {
        // For builtin tools like "browser.search" or "python", use the full recipient
        recipient.to_string()
    }
}

/// Channel types in Harmony format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarmonyChannel {
    /// Chain-of-thought reasoning (internal, not for end users)
    Analysis,
    /// Tool call preambles and explanations
    Commentary,
    /// User-facing response content
    Final,
}

impl HarmonyChannel {
    /// Parse a channel name string into a HarmonyChannel
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "analysis" => Some(Self::Analysis),
            "commentary" => Some(Self::Commentary),
            "final" => Some(Self::Final),
            _ => None,
        }
    }
}

/// Incremental delta from Harmony parsing
#[derive(Debug, Clone, Default)]
pub struct HarmonyDelta {
    /// New analysis/reasoning content since last delta
    pub analysis_delta: Option<String>,
    /// New commentary content since last delta
    pub commentary_delta: Option<String>,
    /// New final response content since last delta
    pub final_delta: Option<String>,
    /// Currently active channel
    pub current_channel: Option<HarmonyChannel>,
}

impl HarmonyDelta {
    /// Check if this delta has any content
    pub fn has_content(&self) -> bool {
        self.analysis_delta.is_some()
            || self.commentary_delta.is_some()
            || self.final_delta.is_some()
    }

    /// Get reasoning content (analysis + commentary without tool calls)
    pub fn reasoning_content(&self) -> Option<String> {
        match (&self.analysis_delta, &self.commentary_delta) {
            (Some(a), Some(c)) => Some(format!("{}{}", a, c)),
            (Some(a), None) => Some(a.clone()),
            (None, Some(c)) => Some(c.clone()),
            (None, None) => None,
        }
    }
}

/// Accumulated content for each channel
#[derive(Debug, Clone, Default)]
pub struct HarmonyAccumulated {
    /// Accumulated analysis content
    pub analysis: String,
    /// Accumulated commentary content
    pub commentary: String,
    /// Accumulated final content
    pub final_content: String,
}

/// A tool call parsed from Harmony format
#[derive(Debug, Clone)]
pub struct HarmonyToolCall {
    /// Unique ID for this tool call
    pub id: String,
    /// The function name (extracted from recipient like "functions.tool_name")
    pub name: String,
    /// The JSON arguments as a string
    pub arguments: String,
}

impl HarmonyAccumulated {
    /// Get all reasoning content (analysis + commentary)
    pub fn reasoning_content(&self) -> Option<String> {
        let combined = format!("{}{}", self.analysis, self.commentary);
        if combined.is_empty() {
            None
        } else {
            Some(combined)
        }
    }
}

/// The routing header of the message being streamed.
#[derive(Debug, Clone, Default, PartialEq)]
struct MessageHeader {
    channel: Option<String>,
    recipient: Option<String>,
}

#[derive(Debug)]
enum ParserState {
    ExpectStart,
    Header {
        bytes: Vec<u8>,
    },
    Content {
        header: MessageHeader,
        text: String,
        pending: Vec<u8>,
    },
}

/// `openai-harmony`'s `StreamableParser` over decoded token bytes; a special token decodes to exactly its marker.
#[derive(Debug)]
struct HarmonyParser {
    state: ParserState,
    // The prompt ends in `<|start|>assistant`; the role stays pending until a header parses, as upstream's does.
    assistant_role_pending: bool,
    messages_started: usize,
}

impl HarmonyParser {
    fn new() -> Self {
        Self {
            state: ParserState::Header { bytes: Vec::new() },
            assistant_role_pending: true,
            messages_started: 0,
        }
    }

    fn process(&mut self, token: &[u8]) {
        match &mut self.state {
            ParserState::ExpectStart => {
                if token == START {
                    self.state = ParserState::Header { bytes: Vec::new() };
                }
            }
            ParserState::Header { bytes } => {
                if token == MESSAGE {
                    let header =
                        parse_header(&String::from_utf8_lossy(bytes), self.assistant_role_pending);
                    self.state = match header {
                        Some(header) => {
                            self.assistant_role_pending = false;
                            self.messages_started += 1;
                            ParserState::Content {
                                header,
                                text: String::new(),
                                pending: Vec::new(),
                            }
                        }
                        None => ParserState::ExpectStart,
                    };
                } else {
                    // Strict parsing, as `openai-harmony` defaults to: a stop token here is header text.
                    bytes.extend_from_slice(token);
                }
            }
            ParserState::Content { text, pending, .. } => {
                if MESSAGE_ENDS.contains(&token) {
                    self.state = ParserState::ExpectStart;
                } else {
                    pending.extend_from_slice(token);
                    push_complete_utf8(text, pending);
                }
            }
        }
    }

    fn process_eos(&mut self) {
        self.state = ParserState::ExpectStart;
    }

    fn header(&self) -> Option<&MessageHeader> {
        match &self.state {
            ParserState::Content { header, .. } => Some(header),
            _ => None,
        }
    }

    fn content(&self) -> Option<&str> {
        match &self.state {
            ParserState::Content { text, .. } => Some(text),
            _ => None,
        }
    }
}

// Moves the complete UTF-8 prefix of `pending` into `text`, replacing invalid bytes and keeping an incomplete tail.
fn push_complete_utf8(text: &mut String, pending: &mut Vec<u8>) {
    let mut consumed = 0;
    loop {
        match std::str::from_utf8(&pending[consumed..]) {
            Ok(valid) => {
                text.push_str(valid);
                consumed = pending.len();
                break;
            }
            Err(error) => {
                let valid_up_to = consumed + error.valid_up_to();
                text.push_str(
                    std::str::from_utf8(&pending[consumed..valid_up_to]).expect("validated"),
                );
                match error.error_len() {
                    Some(len) => {
                        text.push(char::REPLACEMENT_CHARACTER);
                        consumed = valid_up_to + len;
                    }
                    None => {
                        consumed = valid_up_to;
                        break;
                    }
                }
            }
        }
    }
    pending.drain(..consumed);
}

// The header rules of `openai-harmony`'s `parse_header_from_string`; `None` where it errors.
fn parse_header(header: &str, role_known: bool) -> Option<MessageHeader> {
    let mut header = header.to_string();
    let mut channel = None;
    if let Some(idx) = header.find(CHANNEL) {
        let after = &header[idx + CHANNEL.len()..];
        let end = after
            .find(|c: char| c.is_whitespace() || c == '<')
            .unwrap_or(after.len());
        if end == 0 {
            return None;
        }
        channel = Some(after[..end].to_string());
        header = format!("{}{}", &header[..idx], &after[end..]);
    }
    let mut header = header.trim().to_string();
    if header.contains(CONSTRAIN) {
        header = header
            .replace(CONSTRAIN, &format!(" {CONSTRAIN}"))
            .trim()
            .to_string();
    }
    let mut parts: Vec<&str> = header.split_ascii_whitespace().collect();
    let role = if role_known {
        ASSISTANT_ROLE
    } else {
        let first = *parts.first()?;
        if HARMONY_ROLES.contains(&first) {
            first
        } else if parts.len() > 1 || first.starts_with("to=") {
            // An unknown role with a recipient is a tool message.
            parts.remove(0);
            TOOL_ROLE
        } else {
            return None;
        }
    };
    if parts.first() == Some(&role) {
        parts.remove(0);
    }
    let mut recipient = None;
    if let Some(last) = parts.pop() {
        let num_parts = parts.len() + 1;
        if let Some(stripped) = last.strip_prefix("to=") {
            recipient = Some(stripped.to_string());
        } else if num_parts == 1 {
            recipient = Some(last.to_string());
        } else if let Some(raw) = parts.pop() {
            recipient = Some(raw.strip_prefix("to=").unwrap_or(raw).to_string());
        }
    }
    parts
        .is_empty()
        .then_some(MessageHeader { channel, recipient })
}

/// Context for tracking Harmony parsing state within a sequence.
///
/// Wraps the streaming parser with delta extraction for streaming responses.
pub struct HarmonyContext {
    parser: HarmonyParser,
    // The message the per-channel lengths below belong to.
    message: usize,
    // Track lengths for delta extraction (for parser content)
    last_analysis_len: usize,
    last_commentary_len: usize,
    last_final_len: usize,
    // Accumulated content
    accumulated: HarmonyAccumulated,
    // Track which channel we're currently in
    channel: Option<HarmonyChannel>,
    // Track positions for streaming deltas (what has been sent)
    sent_reasoning_len: usize,
    sent_final_len: usize,
    // Tool call tracking
    tool_calls: Vec<HarmonyToolCall>,
    // Track current tool call being built (recipient, accumulated_args)
    current_tool_call: Option<(String, String)>,
    // The message the current tool call is in; a later message to the same recipient is a new call.
    current_tool_message: usize,
    // Track how much of current tool call args have been sent
    sent_tool_args_len: usize,
    // Set when a new tool call starts, signaling that a JSON grammar
    // should be activated for the argument tokens.
    needs_grammar_activation: bool,
}

impl Default for HarmonyContext {
    fn default() -> Self {
        Self::new()
    }
}

impl HarmonyContext {
    /// Create a new Harmony parsing context
    pub fn new() -> Self {
        Self {
            parser: HarmonyParser::new(),
            message: 0,
            last_analysis_len: 0,
            last_commentary_len: 0,
            last_final_len: 0,
            accumulated: HarmonyAccumulated::default(),
            channel: None,
            sent_reasoning_len: 0,
            sent_final_len: 0,
            tool_calls: Vec::new(),
            current_tool_call: None,
            current_tool_message: 0,
            sent_tool_args_len: 0,
            needs_grammar_activation: false,
        }
    }

    /// Process one token's decoded bytes, special tokens included, and return any new delta content.
    pub fn process_token(&mut self, token: &[u8]) -> HarmonyDelta {
        self.parser.process(token);
        if self.parser.messages_started != self.message {
            self.message = self.parser.messages_started;
            self.last_analysis_len = 0;
            self.last_commentary_len = 0;
            self.last_final_len = 0;
        }
        self.extract_delta()
    }

    /// Extract delta since last call
    fn extract_delta(&mut self) -> HarmonyDelta {
        let mut delta = HarmonyDelta::default();

        // Get current channel from parser
        if let Some(channel) = self
            .parser
            .header()
            .and_then(|header| header.channel.as_deref())
            .and_then(HarmonyChannel::parse)
        {
            self.channel = Some(channel);
            delta.current_channel = Some(channel);
        }

        // Check for tool calls via recipient field
        // Recipient is set to "functions.tool_name" when making a tool call
        let current_recipient = self.current_recipient();

        if let Some(content) = self.parser.content() {
            // Check if this is a tool call
            // Tool calls have recipients like:
            // - "functions.tool_name" for user-defined tools
            // - "browser.search", "browser.open", "browser.find" for browser tool
            // - "python" for python tool
            if let Some(ref recipient) = current_recipient {
                let is_tool_call = recipient.starts_with("functions.")
                    || recipient.starts_with("browser.")
                    || recipient == "python";

                if is_tool_call {
                    // This is a tool call - track it
                    // Check if this is the same tool call or a different one
                    let is_same_tool_call = self.current_tool_message == self.message
                        && self
                            .current_tool_call
                            .as_ref()
                            .is_some_and(|(existing, _)| existing == recipient);

                    if is_same_tool_call {
                        // Same tool call, update arguments
                        if let Some((_, ref mut args)) = self.current_tool_call {
                            *args = content.to_string();
                        }
                    } else {
                        // Different tool call or no current tool call
                        // Finalize previous tool call if any
                        if let Some((prev_recipient, prev_args)) = self.current_tool_call.take() {
                            let prev_name = extract_tool_name(&prev_recipient);
                            self.tool_calls.push(HarmonyToolCall {
                                id: format!("call_{}", Uuid::new_v4()),
                                name: prev_name,
                                arguments: prev_args,
                            });
                        }
                        // Start new tool call
                        self.current_tool_call = Some((recipient.clone(), content.to_string()));
                        self.current_tool_message = self.message;
                        self.sent_tool_args_len = 0;
                        self.needs_grammar_activation = true;
                    }
                    // Don't accumulate tool call content to final_content
                    return delta;
                }
            }

            // Not a tool call, handle normally by channel
            match self.channel {
                Some(HarmonyChannel::Analysis) => {
                    if content.len() > self.last_analysis_len {
                        let new_content = content[self.last_analysis_len..].to_string();
                        self.accumulated.analysis.push_str(&new_content);
                        delta.analysis_delta = Some(new_content);
                        self.last_analysis_len = content.len();
                    }
                }
                Some(HarmonyChannel::Commentary) => {
                    if content.len() > self.last_commentary_len {
                        let new_content = content[self.last_commentary_len..].to_string();
                        self.accumulated.commentary.push_str(&new_content);
                        delta.commentary_delta = Some(new_content);
                        self.last_commentary_len = content.len();
                    }
                }
                Some(HarmonyChannel::Final) | None => {
                    // Final channel OR no channel marker - treat content as final.
                    // This handles cases where the model responds without Harmony
                    // channel markers (e.g., after tool call results).
                    if content.len() > self.last_final_len {
                        let new_content = content[self.last_final_len..].to_string();
                        self.accumulated.final_content.push_str(&new_content);
                        delta.final_delta = Some(new_content);
                        self.last_final_len = content.len();
                    }
                }
            }
        }

        delta
    }

    /// Check if grammar activation is needed for a new tool call,
    /// clearing the flag after reading.
    pub fn take_needs_grammar_activation(&mut self) -> bool {
        std::mem::replace(&mut self.needs_grammar_activation, false)
    }

    /// Get the currently active channel
    pub fn current_channel(&self) -> Option<HarmonyChannel> {
        self.channel
    }

    /// Get all accumulated content
    pub fn accumulated(&self) -> &HarmonyAccumulated {
        &self.accumulated
    }

    /// Get accumulated reasoning content (analysis + commentary)
    pub fn reasoning_content(&self) -> Option<String> {
        self.accumulated.reasoning_content()
    }

    /// Get accumulated final content
    pub fn final_content(&self) -> Option<String> {
        if self.accumulated.final_content.is_empty() {
            None
        } else {
            Some(self.accumulated.final_content.clone())
        }
    }

    /// Get the reasoning delta since last call (for streaming).
    /// Returns new reasoning content that hasn't been sent yet.
    pub fn get_reasoning_delta(&mut self) -> Option<String> {
        let reasoning = format!(
            "{}{}",
            self.accumulated.analysis, self.accumulated.commentary
        );
        if reasoning.len() > self.sent_reasoning_len {
            let delta = reasoning[self.sent_reasoning_len..].to_string();
            self.sent_reasoning_len = reasoning.len();
            if delta.is_empty() { None } else { Some(delta) }
        } else {
            None
        }
    }

    /// Get the final content delta since last call (for streaming).
    /// Returns new final content that hasn't been sent yet.
    pub fn get_final_delta(&mut self) -> Option<String> {
        if self.accumulated.final_content.len() > self.sent_final_len {
            let delta = self.accumulated.final_content[self.sent_final_len..].to_string();
            self.sent_final_len = self.accumulated.final_content.len();
            if delta.is_empty() { None } else { Some(delta) }
        } else {
            None
        }
    }

    /// Signal end of stream to the parser
    pub fn process_eos(&mut self) {
        self.parser.process_eos();

        // Finalize any pending tool call
        if let Some((recipient, args)) = self.current_tool_call.take() {
            let name = extract_tool_name(&recipient);
            self.tool_calls.push(HarmonyToolCall {
                id: format!("call_{}", Uuid::new_v4()),
                name,
                arguments: args,
            });
        }
    }

    /// Get the recipient (for tool calls) if any
    pub fn current_recipient(&self) -> Option<String> {
        self.parser.header()?.recipient.clone()
    }

    /// Check if there's a tool call in progress
    pub fn has_tool_call(&self) -> bool {
        self.current_tool_call.is_some() || !self.tool_calls.is_empty()
    }

    /// Get all completed tool calls
    pub fn get_tool_calls(&self) -> &[HarmonyToolCall] {
        &self.tool_calls
    }

    /// Get the current tool call being built (if any)
    /// Returns (recipient, arguments_so_far)
    pub fn get_current_tool_call(&self) -> Option<(&str, &str)> {
        self.current_tool_call
            .as_ref()
            .map(|(recipient, args)| (recipient.as_str(), args.as_str()))
    }

    /// Finalize any pending tool call and return all tool calls.
    /// This should be called when the sequence is done.
    /// Note: This takes ownership of the tool calls, so calling it twice
    /// will return an empty vector the second time.
    pub fn finalize_tool_calls(&mut self) -> Vec<HarmonyToolCall> {
        // Finalize any pending tool call
        if let Some((recipient, args)) = self.current_tool_call.take() {
            let name = extract_tool_name(&recipient);
            self.tool_calls.push(HarmonyToolCall {
                id: format!("call_{}", Uuid::new_v4()),
                name,
                arguments: args,
            });
        }
        // Take ownership to prevent duplicate returns if called multiple times
        std::mem::take(&mut self.tool_calls)
    }
}

/// Check if a chat template uses Harmony format by looking for Harmony markers.
///
/// Returns true if the template contains Harmony-specific tokens like
/// `<|channel|>`, `<|start|>`, `<|message|>`, or `<|end|>`.
pub fn is_harmony_template(template: &str) -> bool {
    // Check for the most distinctive Harmony marker
    if template.contains("<|channel|>") {
        return true;
    }

    // Check for the combination of start/message/end which is characteristic of Harmony
    template.contains("<|start|>")
        && template.contains("<|message|>")
        && template.contains("<|end|>")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(tokens: &[&[u8]]) -> HarmonyContext {
        let mut context = HarmonyContext::new();
        for token in tokens {
            context.process_token(token);
        }
        context
    }

    #[test]
    fn channels_split_reasoning_from_the_final_answer() {
        let mut context = feed(&[
            b"<|channel|>",
            b"analysis",
            b"<|message|>",
            b"Thinking",
            b" hard",
            b"<|end|>",
            b"<|start|>",
            b"assistant",
            b"<|channel|>",
            b"final",
            b"<|message|>",
            b"Hi",
            b"<|return|>",
        ]);
        context.process_eos();
        assert_eq!(
            context.reasoning_content().as_deref(),
            Some("Thinking hard")
        );
        assert_eq!(context.final_content().as_deref(), Some("Hi"));
        assert!(!context.has_tool_call());
    }

    #[test]
    fn a_recipient_after_the_channel_is_a_tool_call() {
        let mut context = feed(&[
            b"<|channel|>",
            b"commentary",
            b" to=functions.get_weather",
            b" ",
            b"<|constrain|>",
            b"json",
            b"<|message|>",
            br#"{"city":"#,
            br#""Paris"}"#,
        ]);
        assert!(context.take_needs_grammar_activation());
        assert_eq!(
            context.get_current_tool_call(),
            Some(("functions.get_weather", r#"{"city":"Paris"}"#))
        );
        context.process_token(b"<|call|>");
        let calls = context.finalize_tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Paris"}"#);
        assert_eq!(context.final_content(), None);
    }

    #[test]
    fn a_recipient_before_the_channel_is_a_tool_call() {
        let mut context = feed(&[
            b"<|channel|>",
            b"analysis",
            b"<|message|>",
            b"look it up",
            b"<|end|>",
            b"<|start|>",
            b"assistant",
            b" to=browser.search",
            b"<|channel|>",
            b"commentary json",
            b"<|message|>",
            b"{}",
            b"<|call|>",
        ]);
        let calls = context.finalize_tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "browser.search");
        assert_eq!(context.reasoning_content().as_deref(), Some("look it up"));
    }

    #[test]
    fn a_character_split_across_tokens_streams_once_complete() {
        let mut context = feed(&[b"<|channel|>", b"final", b"<|message|>", b"x", &[0xc3]]);
        assert_eq!(context.get_final_delta().as_deref(), Some("x"));
        context.process_token(&[0xa9]);
        assert_eq!(context.get_final_delta().as_deref(), Some("\u{e9}"));
    }

    #[test]
    fn a_second_message_on_a_channel_streams_from_its_start() {
        let context = feed(&[
            b"<|channel|>",
            b"analysis",
            b"<|message|>",
            b"abc",
            b"<|end|>",
            b"<|start|>",
            b"assistant",
            b"<|channel|>",
            b"analysis",
            b"<|message|>",
            b"de",
        ]);
        assert_eq!(context.reasoning_content().as_deref(), Some("abcde"));
    }

    #[test]
    fn the_gpt_oss_tool_call_header_names_the_function() {
        let calls = feed(&[
            b"<|channel|>",
            b"analysis",
            b"<|message|>",
            b"need weather",
            b"<|end|>",
            b"<|start|>",
            b"assistant",
            b" to=functions.get_weather",
            b"<|channel|>",
            b"commentary",
            b" ",
            b"<|constrain|>",
            b"json",
            b"<|message|>",
            br#"{"city":"Oslo"}"#,
            b"<|call|>",
        ])
        .finalize_tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Oslo"}"#);
    }

    #[test]
    fn a_recipient_alone_is_a_tool_call() {
        let calls = feed(&[
            b"<|channel|>",
            b"commentary",
            b" to=functions.lookup",
            b"<|message|>",
            b"{}",
            b"<|call|>",
        ])
        .finalize_tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "lookup");
    }

    #[test]
    fn back_to_back_calls_to_one_function_stay_separate() {
        let calls = feed(&[
            b"<|channel|>",
            b"commentary to=functions.f",
            b"<|message|>",
            br#"{"n":1}"#,
            b"<|call|>",
            b"<|start|>",
            b"assistant",
            b"<|channel|>",
            b"commentary to=functions.f",
            b"<|message|>",
            br#"{"n":2}"#,
            b"<|call|>",
        ])
        .finalize_tool_calls();
        let arguments: Vec<_> = calls.iter().map(|call| call.arguments.as_str()).collect();
        assert_eq!(arguments, [r#"{"n":1}"#, r#"{"n":2}"#]);
    }

    #[test]
    fn a_stop_token_inside_a_header_is_header_text() {
        let context = feed(&[b"<|end|>", b"<|channel|>", b"final", b"<|message|>", b"hi"]);
        assert_eq!(context.final_content().as_deref(), Some("hi"));
    }

    #[test]
    fn a_failed_header_leaves_the_assistant_role_pending() {
        // With the role pending, the next header's first word is a recipient, not a role.
        let calls = feed(&[
            b"<|channel|>",
            b"<|constrain|>",
            b"<|message|>",
            b"dropped",
            b"<|start|>",
            b" to=functions.x json",
            b"<|message|>",
            b"{}",
            b"<|call|>",
        ])
        .finalize_tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "x");
    }

    #[test]
    fn text_between_messages_and_malformed_headers_are_dropped() {
        let context = feed(&[
            b"<|message|>",
            b"plain",
            b"<|end|>",
            b"stray",
            b"<|start|>",
            b"narrator",
            b"<|message|>",
            b"ignored",
            b"<|end|>",
        ]);
        assert_eq!(context.final_content().as_deref(), Some("plain"));
    }

    #[test]
    fn test_is_harmony_template() {
        // Should detect Harmony templates
        assert!(is_harmony_template(
            "<|start|>system<|message|>content<|end|>"
        ));
        assert!(is_harmony_template(
            "some prefix <|channel|>analysis<|message|>thinking"
        ));

        // Should not detect non-Harmony templates
        assert!(!is_harmony_template("<|im_start|>system<|im_end|>"));
        assert!(!is_harmony_template("regular chat template"));
    }

    #[test]
    fn test_harmony_channel_from_str() {
        assert_eq!(
            HarmonyChannel::parse("analysis"),
            Some(HarmonyChannel::Analysis)
        );
        assert_eq!(
            HarmonyChannel::parse("commentary"),
            Some(HarmonyChannel::Commentary)
        );
        assert_eq!(HarmonyChannel::parse("final"), Some(HarmonyChannel::Final));
        assert_eq!(HarmonyChannel::parse("unknown"), None);
    }

    #[test]
    fn test_harmony_delta_has_content() {
        let empty = HarmonyDelta::default();
        assert!(!empty.has_content());

        let with_analysis = HarmonyDelta {
            analysis_delta: Some("thinking".to_string()),
            ..Default::default()
        };
        assert!(with_analysis.has_content());

        let with_final = HarmonyDelta {
            final_delta: Some("response".to_string()),
            ..Default::default()
        };
        assert!(with_final.has_content());
    }

    #[test]
    fn test_harmony_delta_reasoning_content() {
        let both = HarmonyDelta {
            analysis_delta: Some("thinking ".to_string()),
            commentary_delta: Some("about tools".to_string()),
            ..Default::default()
        };
        assert_eq!(
            both.reasoning_content(),
            Some("thinking about tools".to_string())
        );

        let only_analysis = HarmonyDelta {
            analysis_delta: Some("just thinking".to_string()),
            ..Default::default()
        };
        assert_eq!(
            only_analysis.reasoning_content(),
            Some("just thinking".to_string())
        );

        let none = HarmonyDelta::default();
        assert_eq!(none.reasoning_content(), None);
    }
}
