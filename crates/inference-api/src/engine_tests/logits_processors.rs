use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};

use super::tiny_engine;
use crate::{logits_processors::in_place, media_source::MediaAttachments};

const MAX_TOKENS: usize = 6;
// Late enough that both requests are decoding in the same batch when it fails.
const FAILING_STEP: usize = 3;

fn request(processors: &[&str]) -> Vec<u8> {
    json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Reply with ok"}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "logits_processors": processors,
    })
    .to_string()
    .into_bytes()
}

fn text_and_tokens(response: &str) -> anyhow::Result<(Value, Value)> {
    let response: Value = serde_json::from_str(response)?;
    let text = response["choices"][0]["message"]["content"].clone();
    Ok((text, response["usage"]["completion_tokens"].clone()))
}

#[tokio::test(flavor = "multi_thread")]
async fn a_failing_processor_fails_its_request_and_spares_the_batch() -> anyhow::Result<()> {
    let (_dir, engine) = tiny_engine().await?;
    let steps = std::sync::Arc::new(AtomicUsize::new(0));
    let counted = steps.clone();
    engine.register_logits_processor(
        "fails-late",
        in_place(
            move |_, _| match counted.fetch_add(1, Ordering::SeqCst) + 1 {
                FAILING_STEP => Err("refused".to_string()),
                _ => Ok(()),
            },
        ),
    )?;

    let healthy_steps = std::sync::Arc::new(AtomicUsize::new(0));
    let seen = healthy_steps.clone();
    engine.register_logits_processor(
        "counts",
        in_place(move |_, _| {
            seen.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }),
    )?;
    let plain = request(&[]);
    let (alone, _) = text_and_tokens(
        &engine
            .chat_json(&plain, MediaAttachments::default())
            .await?,
    )?;
    let failing = request(&["fails-late"]);
    let healthy = request(&["counts"]);
    let (failed, beside) = tokio::join!(
        engine.chat_json(&failing, MediaAttachments::default()),
        engine.chat_json(&healthy, MediaAttachments::default()),
    );
    assert!(failed.is_err());
    assert_eq!(steps.load(Ordering::SeqCst), FAILING_STEP);
    let (text, tokens) = text_and_tokens(&beside?)?;
    assert_eq!(text, alone);
    // A step whose token went missing is sampled once more than the tokens it produced.
    assert_eq!(json!(healthy_steps.load(Ordering::SeqCst)), tokens);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn responses_and_anthropic_requests_resolve_their_processors() -> anyhow::Result<()> {
    let (_dir, engine) = tiny_engine().await?;
    let unknown = ["never-registered"];
    let responses = json!({"model": "default", "input": "hi", "max_output_tokens": 2, "logits_processors": unknown});
    let anthropic = json!({
        "model": "default",
        "max_tokens": 2,
        "messages": [{"role": "user", "content": "hi"}],
        "logits_processors": unknown,
    });
    let refused = engine
        .responses_json(responses.to_string().as_bytes())
        .await;
    assert_eq!(
        refused.err().map(|error| error.param),
        Some(Some("logits_processors".into()))
    );
    let refused = engine
        .anthropic_messages_json(anthropic.to_string().as_bytes())
        .await;
    assert_eq!(
        refused.err().map(|error| error.param),
        Some(Some("logits_processors".into()))
    );
    Ok(())
}
