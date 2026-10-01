//! The Responses streamer over agentic rounds: per-round reasoning items and stable output indices.

use futures::StreamExt;
use inference_api::{
    Engine,
    responses::{OpenResponsesStreamEvent, ResponsesStreamItem},
};

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
