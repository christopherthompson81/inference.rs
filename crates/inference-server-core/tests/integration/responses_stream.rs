//! The Responses streamer over agentic rounds, and the tool calls a stored reply keeps for its follow-up.

use futures::StreamExt;
use inference_api::{
    Engine,
    responses::{OpenResponsesStreamEvent, ResponsesStreamItem},
};
use serde_json::json;

use crate::cancel::tiny_engine;

fn reasoning_chunk(
    reasoning: Option<&str>,
    content: Option<&str>,
    finish_reason: Option<&str>,
) -> inference_core::Response {
    inference_core::Response::Chunk(inference_core::ChatCompletionChunkResponse {
        id: "chunk".to_string(),
        choices: vec![inference_core::ChunkChoice {
            finish_reason: finish_reason.map(str::to_string),
            stop_sequence: None,
            index: 0,
            delta: inference_core::Delta {
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

fn tool_progress() -> inference_core::Response {
    let data = inference_core::AgenticToolCallData::Custom {
        arguments: "{}".to_string(),
        content: "found".to_string(),
    };
    inference_core::Response::AgenticToolCallProgress {
        round: 0,
        tool_name: "lookup".to_string(),
        phase: inference_core::AgenticToolCallPhase::Complete(data),
    }
}

/// Streams `responses` through a Responses streamer and checks every item keeps one `output_index`, matching its
/// position in the final resource; returns each reasoning item's text, in output order.
async fn stream_rounds(
    engine: &Engine,
    responses: Vec<inference_core::Response>,
) -> anyhow::Result<Vec<String>> {
    use inference_api::responses::{OpenResponsesStreamer, PreparedResponse, RequestContext};
    use inference_api::responses_types::OutputItem;

    let (tx, rx) = tokio::sync::mpsc::channel(16);
    for response in responses {
        tx.send(response).await?;
    }
    drop(tx);
    let prepared = PreparedResponse {
        rx,
        id: "resp_rounds".to_string(),
        model: "default".to_string(),
        metadata: None,
        store: false,
        stream: true,
        background: false,
        history: Vec::new(),
        context: RequestContext::default(),
        cancellation: Default::default(),
        session_id: None,
        owner: None,
    };
    let mut stream = OpenResponsesStreamer::new(prepared, engine.state().clone(), None);
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

#[tokio::test(flavor = "multi_thread")]
async fn each_agentic_round_streams_its_reasoning_into_its_own_item() -> anyhow::Result<()> {
    let (_dir, engine) = tiny_engine().await?;
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
        let reasoning = stream_rounds(&engine, rounds).await?;
        assert_eq!(reasoning, ["first thought", "second thought"]);
    }
    Ok(())
}

const CLIENT_TOOL: &str = "lookup";
const SERVER_TOOL: &str = "search_index";
const RUN_SESSION: &str = "session_run";

fn tool_call(id: &str, name: &str) -> inference_core::ToolCallResponse {
    inference_core::ToolCallResponse {
        index: 0,
        id: id.to_string(),
        tp: inference_core::ToolCallType::Function,
        function: inference_core::CalledFunction {
            name: name.to_string(),
            arguments: "{\"word\":\"ok\"}".to_string(),
        },
    }
}

/// A client-defined call, and a server tool's call that a round-limit stop hands back.
fn returned_calls() -> Vec<inference_core::ToolCallResponse> {
    vec![
        tool_call("call_lookup", CLIENT_TOOL),
        tool_call("call_search", SERVER_TOOL),
    ]
}

fn client_tools() -> serde_json::Value {
    json!([{"type": "function", "name": CLIENT_TOOL, "parameters": {"type": "object"}}])
}

fn prepared(
    id: &str,
    stream: bool,
    rx: tokio::sync::mpsc::Receiver<inference_core::Response>,
) -> anyhow::Result<inference_api::responses::PreparedResponse> {
    Ok(inference_api::responses::PreparedResponse {
        rx,
        id: id.to_string(),
        model: "default".to_string(),
        metadata: None,
        store: true,
        stream,
        background: false,
        history: vec![inference_api::openai::Message {
            content: Some(inference_api::openai::MessageContent::from_text(
                "Look up ok.".to_string(),
            )),
            role: "user".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }],
        context: inference_api::responses::RequestContext {
            tools: Some(serde_json::from_value(client_tools())?),
            ..Default::default()
        },
        cancellation: Default::default(),
        session_id: None,
        owner: None,
    })
}

fn done_with_calls(calls: Vec<inference_core::ToolCallResponse>) -> inference_core::Response {
    inference_core::Response::Done(inference_core::ChatCompletionResponse {
        id: "chatcmpl".to_string(),
        choices: vec![inference_core::Choice {
            finish_reason: "tool_calls".to_string(),
            stop_sequence: None,
            index: 0,
            message: inference_core::ResponseMessage {
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
        usage: inference_core::Usage {
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
    })
}

/// The stored reply's tool calls, as `previous_response_id` will replay them in the run's session.
fn stored_tool_calls(id: &str) -> anyhow::Result<Vec<inference_api::openai::ToolCall>> {
    let history = inference_api::cached_responses::get_response_cache()
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

fn assert_only_client_call(calls: &[inference_api::openai::ToolCall]) {
    let calls: Vec<_> = calls
        .iter()
        .map(|call| (call.id.as_deref(), call.function.name.as_str()))
        .collect();
    assert_eq!(calls, [(Some("call_lookup"), CLIENT_TOOL)]);
}

async fn stream_returned_calls(engine: &Engine, id: &str) -> anyhow::Result<()> {
    let (tx, rx) = tokio::sync::mpsc::channel(4);
    let mut chunk = reasoning_chunk(None, None, Some("tool_calls"));
    if let inference_core::Response::Chunk(chunk) = &mut chunk {
        chunk.choices[0].delta.tool_calls = Some(returned_calls());
        chunk.session_id = Some(RUN_SESSION.to_string());
    }
    tx.send(chunk).await?;
    drop(tx);
    let mut stream = inference_api::responses::OpenResponsesStreamer::new(
        prepared(id, true, rx)?,
        engine.state().clone(),
        None,
    );
    while stream.next().await.is_some() {}
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_streamed_reply_is_stored_with_the_client_calls_it_returns() -> anyhow::Result<()> {
    let (_dir, engine) = tiny_engine().await?;
    stream_returned_calls(&engine, "resp_streamed_call").await?;
    assert_only_client_call(&stored_tool_calls("resp_streamed_call")?);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_collected_reply_is_stored_with_the_client_calls_it_returns() -> anyhow::Result<()> {
    let (_dir, engine) = tiny_engine().await?;
    let (tx, rx) = tokio::sync::mpsc::channel(4);
    tx.send(done_with_calls(returned_calls())).await?;
    drop(tx);
    inference_api::responses::collect_response(
        prepared("resp_collected_call", false, rx)?,
        engine.state(),
    )
    .await
    .map_err(anyhow::Error::msg)?;
    assert_only_client_call(&stored_tool_calls("resp_collected_call")?);
    Ok(())
}

/// The agent session a `previous_response_id` follow-up would send on its request.
async fn follow_up_session(engine: &Engine, previous: &str) -> anyhow::Result<Option<String>> {
    let request =
        json!({"model": "default", "previous_response_id": previous, "input": "And then?"});
    let prepared = inference_api::responses::prepare_response(
        engine.chat_engine(),
        serde_json::from_value(request)?,
    )
    .await
    .map_err(|error| anyhow::Error::msg(error.into_api_error(engine.state().clone())))?;
    prepared.cancellation.cancel();
    Ok(prepared.session_id)
}

#[tokio::test(flavor = "multi_thread")]
async fn only_a_follow_up_on_the_latest_reply_continues_its_session() -> anyhow::Result<()> {
    let (_dir, engine) = tiny_engine().await?;
    stream_returned_calls(&engine, "resp_first").await?;
    assert_eq!(
        follow_up_session(&engine, "resp_first").await?.as_deref(),
        Some(RUN_SESSION)
    );
    stream_returned_calls(&engine, "resp_second").await?;
    // branching from the first reply must not rewrite the turns the second added to the session
    assert_eq!(follow_up_session(&engine, "resp_first").await?, None);
    assert_eq!(
        follow_up_session(&engine, "resp_second").await?.as_deref(),
        Some(RUN_SESSION)
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_reply_without_an_agent_run_keeps_the_session_it_continued() -> anyhow::Result<()> {
    let (_dir, engine) = tiny_engine().await?;
    let (tx, rx) = tokio::sync::mpsc::channel(4);
    let inference_core::Response::Done(mut done) = done_with_calls(returned_calls()) else {
        unreachable!()
    };
    done.session_id = None;
    tx.send(inference_core::Response::Done(done)).await?;
    drop(tx);
    let mut prepared = prepared("resp_plain_turn", false, rx)?;
    prepared.session_id = Some(RUN_SESSION.to_string());
    inference_api::responses::collect_response(prepared, engine.state())
        .await
        .map_err(anyhow::Error::msg)?;
    assert_only_client_call(&stored_tool_calls("resp_plain_turn")?);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_follow_up_answers_the_stored_call_without_repeating_it() -> anyhow::Result<()> {
    let (_dir, engine) = tiny_engine().await?;
    stream_returned_calls(&engine, "resp_answered_call").await?;
    let request = json!({
        "model": "default",
        "previous_response_id": "resp_answered_call",
        "tools": client_tools(),
        "input": [
            {"type": "function_call", "call_id": "call_lookup", "name": CLIENT_TOOL, "arguments": "{\"word\":\"ok\"}"},
            {"type": "function_call_output", "call_id": "call_lookup", "output": "found"},
        ],
    });
    let prepared = inference_api::responses::prepare_response(
        engine.chat_engine(),
        serde_json::from_value(request)?,
    )
    .await
    .map_err(|error| anyhow::Error::msg(error.into_api_error(engine.state().clone())))?;
    prepared.cancellation.cancel();
    let turns: Vec<_> = prepared
        .history
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
