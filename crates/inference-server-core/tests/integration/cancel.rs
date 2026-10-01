//! Cancelling a request through the engine API ends it early and still delivers its final event.

use std::time::Duration;

use futures::StreamExt;
use inference_api::{
    Engine,
    responses::{OpenResponsesStreamEvent, ResponsesStreamItem},
};
use serde_json::{Value, json};

use crate::support;

const PROMPT: &str = "Reply with the single word: ok";
// Enough to outlast the few steps a cancel takes to land, on a model whose random weights never stop on their own.
const LONG_COMPLETION: usize = 512;
const BACKGROUND_POLL: Duration = Duration::from_millis(20);
const BACKGROUND_DEADLINE: Duration = Duration::from_secs(60);

pub(crate) async fn tiny_engine() -> anyhow::Result<(tempfile::TempDir, Engine)> {
    let dir = support::tiny_checkpoint()?;
    let spec = serde_json::from_value(json!({
        "model": {"MultimodalPlain": {"model_id": dir.path().to_string_lossy(), "dtype": "f32"}},
        "runtime": {"device": "cpu"},
    }))?;
    let engine = Engine::load(spec).await?;
    Ok((dir, engine))
}

#[tokio::test(flavor = "multi_thread")]
async fn a_cancelled_completion_stream_finishes_as_canceled() -> anyhow::Result<()> {
    use inference_api::engine_completion::CompletionStreamEvent;

    let (_dir, engine) = tiny_engine().await?;
    let request = json!({"model": "default", "prompt": PROMPT, "max_tokens": LONG_COMPLETION, "ignore_eos": true});
    let mut stream = engine
        .completion_stream(serde_json::from_value(request)?)
        .await
        .map_err(anyhow::Error::msg)?;
    let (mut chunks, mut finish) = (0, None);
    while let Some(event) = stream.next().await {
        match event {
            CompletionStreamEvent::Chunk(chunk) => {
                stream.cancel();
                chunks += 1;
                finish = chunk.choices[0].finish_reason.clone().or(finish);
            }
            CompletionStreamEvent::Error(error) => anyhow::bail!("{error:?}"),
        }
    }
    assert_eq!(finish.as_deref(), Some("canceled"));
    assert!(chunks < LONG_COMPLETION, "{chunks} chunks");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_cancelled_anthropic_stream_still_stops_with_its_usage() -> anyhow::Result<()> {
    let (_dir, engine) = tiny_engine().await?;
    let request = json!({
        "model": "default",
        "max_tokens": LONG_COMPLETION,
        "messages": [{"role": "user", "content": PROMPT}],
    });
    let mut stream = engine
        .anthropic_messages_stream(serde_json::from_value(request)?)
        .await
        .map_err(anyhow::Error::msg)?;
    let mut events: Vec<(&'static str, Value)> = Vec::new();
    while let Some(event) = stream.next().await {
        stream.cancel();
        events.push((event.name, event.payload));
    }
    assert_eq!(events.last().map(|(name, _)| *name), Some("message_stop"));
    let (_, delta) = events
        .iter()
        .find(|(name, _)| *name == "message_delta")
        .expect("a stopped message sends its delta");
    let output_tokens = delta["usage"]["output_tokens"].as_u64().unwrap() as usize;
    assert!(output_tokens < LONG_COMPLETION, "{delta}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_cancelled_response_stream_ends_as_cancelled_with_its_usage() -> anyhow::Result<()> {
    let (_dir, engine) = tiny_engine().await?;
    let request = json!({
        "model": "default",
        "input": PROMPT,
        "max_output_tokens": LONG_COMPLETION,
        "ignore_eos": true,
    });
    let mut stream = engine
        .responses_stream(serde_json::from_value(request)?)
        .await
        .map_err(anyhow::Error::msg)?;
    let mut last = None;
    while let Some(event) = stream.next().await {
        stream.cancel();
        last = Some(event);
    }
    let Some(ResponsesStreamItem::Event(OpenResponsesStreamEvent::ResponseCancelled {
        response,
        ..
    })) = last
    else {
        anyhow::bail!(
            "expected response.cancelled last, got {:?}",
            last.map(|item| item.name())
        );
    };
    assert_eq!(response.status.as_str(), "cancelled");
    assert_cut_short(&response);
    let usage = response
        .usage
        .clone()
        .expect("a cancelled response reports its usage");
    assert!(usage.output_tokens < LONG_COMPLETION, "{usage:?}");

    // Stored for fetching, but its reply was cut short, so it is not a conversation to continue.
    let stored = engine.response(&response.id).map_err(anyhow::Error::msg)?;
    assert_eq!(stored.status.as_str(), "cancelled");
    let follow_up =
        json!({"model": "default", "input": PROMPT, "previous_response_id": response.id});
    let refused = engine
        .responses(serde_json::from_value(follow_up)?)
        .await
        .expect_err("a cancelled response cannot be continued");
    assert!(refused.message.contains(&response.id), "{refused:?}");
    Ok(())
}

fn assert_cut_short(response: &inference_api::responses_types::ResponseResource) {
    use inference_api::responses_types::{ItemStatus, OutputItem};
    for item in &response.output {
        if matches!(item, OutputItem::Message { .. }) {
            assert_eq!(item.status(), ItemStatus::Incomplete, "{item:?}");
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn cancelling_a_background_response_stops_its_generation() -> anyhow::Result<()> {
    let (_dir, engine) = tiny_engine().await?;
    let request = json!({
        "model": "default",
        "input": PROMPT,
        "max_output_tokens": LONG_COMPLETION,
        "ignore_eos": true,
        "background": true,
        "store": false,
    });
    let queued = engine
        .responses(serde_json::from_value(request)?)
        .await
        .map_err(anyhow::Error::msg)?;
    let cancelled = engine
        .cancel_response(&queued.id)
        .map_err(anyhow::Error::msg)?;
    assert_eq!(cancelled.status.as_str(), "cancelled");
    // The stopped request comes back with what it generated.
    let deadline = tokio::time::Instant::now() + BACKGROUND_DEADLINE;
    let partial = loop {
        let response = engine.response(&queued.id).map_err(anyhow::Error::msg)?;
        if response.usage.is_some() {
            break response;
        }
        anyhow::ensure!(
            tokio::time::Instant::now() < deadline,
            "the cancelled request never came back"
        );
        tokio::time::sleep(BACKGROUND_POLL).await;
    };
    assert_eq!(partial.status.as_str(), "cancelled");
    assert_cut_short(&partial);
    assert!(partial.usage.unwrap().output_tokens < LONG_COMPLETION);
    Ok(())
}
