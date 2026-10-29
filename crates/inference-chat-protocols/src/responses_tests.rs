//! The Responses streamer over agentic rounds, and the tool calls a stored reply keeps for its follow-up.

use futures::StreamExt;
use inference_api::engine_chat::{AgenticToolProgress, ChatStreamEvent};
use inference_mcp::CalledFunction;
use inference_protocol::{
    openai::{Message, MessageContent, ToolCall},
    response::{
        ChatCompletionChunkResponse, ChatCompletionResponse, Choice, ChunkChoice, Delta,
        ResponseMessage, Usage,
    },
    responses_types::OutputItem,
    tools::{ToolCallResponse, ToolCallType},
};
use serde_json::json;

use super::{
    OpenResponsesStreamEvent, OpenResponsesStreamer, RequestContext, ResponseRun,
    ResponsesStreamItem, parse_openresponses_request, run_to_end,
};
use crate::cached_responses::get_response_cache;

fn events(events: Vec<ChatStreamEvent>) -> super::ChatEvents {
    Box::pin(futures::stream::iter(events))
}

fn reasoning_chunk(
    reasoning: Option<&str>,
    content: Option<&str>,
    finish_reason: Option<&str>,
) -> ChatStreamEvent {
    ChatStreamEvent::Chunk(ChatCompletionChunkResponse {
        id: "chunk".to_string(),
        choices: vec![ChunkChoice {
            finish_reason: finish_reason.map(str::to_string),
            stop_sequence: None,
            index: 0,
            delta: Delta {
                content: content.map(str::to_string),
                role: "assistant".to_string(),
                tool_calls: None,
                reasoning_content: reasoning.map(str::to_string),
            },
            logprobs: None,
        }],
        created: 0,
        model: "default".to_string(),
        system_fingerprint: "local".to_string(),
        object: "chat.completion.chunk".to_string(),
        usage: None,
        adapter_generation: None,
        session_id: None,
    })
}

fn tool_progress() -> ChatStreamEvent {
    let data = inference_api::engine_chat::AgenticToolCallData::Custom {
        arguments: "{}".to_string(),
        content: "found".to_string(),
    };
    ChatStreamEvent::AgenticToolCallProgress(AgenticToolProgress {
        round: 0,
        tool_call_id: "call_lookup".to_string(),
        tool_name: "lookup".to_string(),
        phase: inference_api::engine_chat::AgenticToolCallPhase::Complete(data),
    })
}

/// Streams `responses` through a Responses streamer and checks every item keeps one `output_index`, matching its
/// position in the final resource; returns each reasoning item's text, in output order.
async fn stream_rounds(rounds: Vec<ChatStreamEvent>) -> anyhow::Result<Vec<String>> {
    let run = ResponseRun {
        id: "resp_rounds".to_string(),
        model: "default".to_string(),
        metadata: None,
        store: false,
        history: Vec::new(),
        context: RequestContext::default(),
        session_id: None,
        owner: None,
    };
    let mut stream = OpenResponsesStreamer::new(run, events(rounds), None, None);
    let mut index_of = std::collections::HashMap::new();
    let mut last = None;
    while let Some(item) = stream.next().await {
        if let ResponsesStreamItem::Event(
            OpenResponsesStreamEvent::OutputItemAdded {
                output_index, item, ..
            }
            | OpenResponsesStreamEvent::OutputItemDone {
                output_index, item, ..
            },
        ) = &item
        {
            let seen = index_of
                .entry(item.id().to_string())
                .or_insert(*output_index);
            assert_eq!(seen, output_index, "{} moved", item.id());
        }
        last = Some(item);
    }
    let Some(ResponsesStreamItem::Event(OpenResponsesStreamEvent::ResponseCompleted {
        response,
        ..
    })) = last
    else {
        anyhow::bail!("expected response.completed last");
    };
    for (position, item) in response.output.iter().enumerate() {
        assert_eq!(
            index_of.get(item.id()),
            Some(&position),
            "{:?}",
            response.output
        );
    }
    Ok(response
        .output
        .iter()
        .filter_map(|item| match item {
            OutputItem::Reasoning { .. } => Some(
                serde_json::to_value(item).unwrap()["content"][0]["text"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
            ),
            _ => None,
        })
        .collect())
}

#[tokio::test]
async fn each_agentic_round_streams_its_reasoning_into_its_own_item() -> anyhow::Result<()> {
    let first = || reasoning_chunk(Some("first thought"), None, None);
    let second = || reasoning_chunk(Some("second thought"), None, None);
    let finish = || reasoning_chunk(None, Some("Found it."), Some("stop"));
    // A reply between the rounds, then a tool between them (the loop holds the tool call itself back).
    for rounds in [
        vec![
            first(),
            reasoning_chunk(None, Some("Let me look. "), None),
            second(),
            finish(),
        ],
        vec![first(), tool_progress(), second(), finish()],
    ] {
        let reasoning = stream_rounds(rounds).await?;
        assert_eq!(reasoning, ["first thought", "second thought"]);
    }
    Ok(())
}

const CLIENT_TOOL: &str = "lookup";
const SERVER_TOOL: &str = "search_index";
const RUN_SESSION: &str = "session_run";

fn tool_call(id: &str, name: &str) -> ToolCallResponse {
    ToolCallResponse {
        index: 0,
        id: id.to_string(),
        tp: ToolCallType::Function,
        function: CalledFunction {
            name: name.to_string(),
            arguments: "{\"word\":\"ok\"}".to_string(),
        },
    }
}

/// A client-defined call, and a server tool's call that a round-limit stop hands back.
fn returned_calls() -> Vec<ToolCallResponse> {
    vec![
        tool_call("call_lookup", CLIENT_TOOL),
        tool_call("call_search", SERVER_TOOL),
    ]
}

fn client_tools() -> serde_json::Value {
    json!([{"type": "function", "name": CLIENT_TOOL, "parameters": {"type": "object"}}])
}

fn run(id: &str) -> anyhow::Result<ResponseRun> {
    Ok(ResponseRun {
        id: id.to_string(),
        model: "default".to_string(),
        metadata: None,
        store: true,
        history: vec![Message {
            content: Some(MessageContent::from_text("Look up ok.".to_string())),
            role: "user".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }],
        context: RequestContext {
            tools: Some(serde_json::from_value(client_tools())?),
            ..Default::default()
        },
        session_id: None,
        owner: None,
    })
}

fn done_with_calls(calls: Vec<ToolCallResponse>) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: "chatcmpl".to_string(),
        choices: vec![Choice {
            finish_reason: "tool_calls".to_string(),
            stop_sequence: None,
            index: 0,
            message: ResponseMessage {
                content: None,
                role: "assistant".to_string(),
                tool_calls: Some(calls),
                reasoning_content: None,
            },
            logprobs: None,
        }],
        created: 0,
        model: "default".to_string(),
        system_fingerprint: "local".to_string(),
        object: "chat.completion".to_string(),
        usage: Usage {
            completion_tokens: 1,
            prompt_tokens: 1,
            total_tokens: 2,
            prompt_tokens_details: None,
            avg_tok_per_sec: 0.0,
            avg_prompt_tok_per_sec: 0.0,
            avg_compl_tok_per_sec: 0.0,
            total_time_sec: 0.0,
            total_prompt_time_sec: 0.0,
            total_completion_time_sec: 0.0,
        },
        adapter_generation: None,
        agentic_tool_calls: None,
        files: None,
        session_id: Some(RUN_SESSION.to_string()),
    }
}

/// The stored reply's tool calls, as `previous_response_id` will replay them in the run's session.
fn stored_tool_calls(id: &str) -> anyhow::Result<Vec<ToolCall>> {
    let history = get_response_cache()
        .get_conversation(id, None)?
        .expect("the reply was stored");
    assert_eq!(history.session_id.as_deref(), Some(RUN_SESSION));
    let reply = history
        .messages
        .last()
        .expect("the history holds the reply");
    assert_eq!(reply.role, "assistant");
    Ok(reply
        .tool_calls
        .clone()
        .expect("the reply keeps its tool call"))
}

fn assert_only_client_call(calls: &[ToolCall]) {
    let calls: Vec<_> = calls
        .iter()
        .map(|call| (call.id.as_deref(), call.function.name.as_str()))
        .collect();
    assert_eq!(calls, [(Some("call_lookup"), CLIENT_TOOL)]);
}

async fn stream_returned_calls(id: &str) -> anyhow::Result<()> {
    let mut chunk = reasoning_chunk(None, None, Some("tool_calls"));
    if let ChatStreamEvent::Chunk(chunk) = &mut chunk {
        chunk.choices[0].delta.tool_calls = Some(returned_calls());
        chunk.session_id = Some(RUN_SESSION.to_string());
    }
    let mut stream = OpenResponsesStreamer::new(run(id)?, events(vec![chunk]), None, None);
    while stream.next().await.is_some() {}
    Ok(())
}

/// Runs `done` as a collected (not streamed) reply, storing it as the request asked.
async fn collect(run: ResponseRun, done: ChatCompletionResponse) -> anyhow::Result<()> {
    let (result, stored) = run_to_end(
        run,
        events(vec![ChatStreamEvent::Done(Box::new(done))]),
        None,
    )
    .await;
    result.map_err(anyhow::Error::msg)?;
    stored.expect("the request asked to be stored").save();
    Ok(())
}

#[tokio::test]
async fn a_streamed_reply_is_stored_with_the_client_calls_it_returns() -> anyhow::Result<()> {
    stream_returned_calls("resp_streamed_call").await?;
    assert_only_client_call(&stored_tool_calls("resp_streamed_call")?);
    Ok(())
}

#[tokio::test]
async fn a_collected_reply_is_stored_with_the_client_calls_it_returns() -> anyhow::Result<()> {
    collect(
        run("resp_collected_call")?,
        done_with_calls(returned_calls()),
    )
    .await?;
    assert_only_client_call(&stored_tool_calls("resp_collected_call")?);
    Ok(())
}

/// The agent session a `previous_response_id` follow-up would send on its request.
fn follow_up_session(previous: &str) -> anyhow::Result<Option<String>> {
    let request =
        json!({"model": "default", "previous_response_id": previous, "input": "And then?"});
    let (chat, _, _) = parse_openresponses_request(serde_json::from_value(request)?, None)?;
    Ok(chat.session_id)
}

#[tokio::test]
async fn only_a_follow_up_on_the_latest_reply_continues_its_session() -> anyhow::Result<()> {
    stream_returned_calls("resp_first").await?;
    assert_eq!(
        follow_up_session("resp_first")?.as_deref(),
        Some(RUN_SESSION)
    );
    stream_returned_calls("resp_second").await?;
    // branching from the first reply must not rewrite the turns the second added to the session
    assert_eq!(follow_up_session("resp_first")?, None);
    assert_eq!(
        follow_up_session("resp_second")?.as_deref(),
        Some(RUN_SESSION)
    );
    Ok(())
}

#[tokio::test]
async fn a_reply_without_an_agent_run_keeps_the_session_it_continued() -> anyhow::Result<()> {
    let mut done = done_with_calls(returned_calls());
    done.session_id = None;
    let mut run = run("resp_plain_turn")?;
    run.session_id = Some(RUN_SESSION.to_string());
    collect(run, done).await?;
    assert_only_client_call(&stored_tool_calls("resp_plain_turn")?);
    Ok(())
}

#[tokio::test]
async fn a_follow_up_answers_the_stored_call_without_repeating_it() -> anyhow::Result<()> {
    stream_returned_calls("resp_answered_call").await?;
    let request = json!({
        "model": "default",
        "previous_response_id": "resp_answered_call",
        "tools": client_tools(),
        "input": [
            {"type": "function_call", "call_id": "call_lookup", "name": CLIENT_TOOL, "arguments": "{\"word\":\"ok\"}"},
            {"type": "function_call_output", "call_id": "call_lookup", "output": "found"},
        ],
    });
    let (_, history, _) = parse_openresponses_request(serde_json::from_value(request)?, None)?;
    let turns: Vec<_> = history
        .iter()
        .map(|message| {
            (
                message.role.as_str(),
                message.tool_calls.as_ref().map(Vec::len),
            )
        })
        .collect();
    assert_eq!(
        turns,
        [("user", None), ("assistant", Some(1)), ("tool", None)]
    );
    Ok(())
}
