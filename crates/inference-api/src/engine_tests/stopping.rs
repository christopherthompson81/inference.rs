use serde_json::{Value, json};

use super::{tiny_engine, tool_loop::scripted};
use crate::{media_source::MediaAttachments, operations::TokenizeRequest};

const SCRIPT: &str = "stopping-script";
const SCRIPTED_TEXT: &str = "abcdefgh";
// The script's third token ends the generation, so it is the last one produced.
const STOP_AT: usize = 2;
const OUT_OF_VOCABULARY: u32 = 1_000_000;

#[tokio::test(flavor = "multi_thread")]
async fn a_stop_token_id_ends_the_generation_where_it_is_produced() -> anyhow::Result<()> {
    let (_dir, engine) = tiny_engine().await?;
    let tokenized = TokenizeRequest {
        model: None,
        text: SCRIPTED_TEXT.to_string(),
        add_special_tokens: false,
    };
    let tokens = engine.tokenize(tokenized).await?.tokens;
    assert!(tokens.len() > STOP_AT + 1, "{tokens:?}");
    engine.register_logits_processor(SCRIPT, scripted(tokens.clone()))?;

    let request = |stop: Value| {
        json!({
            "model": "default",
            "messages": [{"role": "user", "content": "go"}],
            "max_tokens": tokens.len(),
            "logits_processors": [SCRIPT],
            "stop_token_ids": stop,
        })
        .to_string()
    };
    let run = |body: String| {
        let engine = engine.clone();
        async move {
            let response = engine
                .chat_json(body.as_bytes(), MediaAttachments::default())
                .await?;
            anyhow::Ok(serde_json::from_str::<Value>(&response)?)
        }
    };
    let unstopped = run(request(json!(null))).await?;
    assert_eq!(unstopped["usage"]["completion_tokens"], tokens.len());
    let stopped = run(request(json!([tokens[STOP_AT]]))).await?;
    assert_eq!(
        stopped["usage"]["completion_tokens"],
        STOP_AT + 1,
        "{stopped}"
    );
    assert_eq!(stopped["choices"][0]["finish_reason"], "stop", "{stopped}");
    let beyond = run(request(json!([OUT_OF_VOCABULARY])))
        .await
        .err()
        .map(|error| error.to_string());
    assert!(beyond.is_some_and(|error| error.contains("outside the model's vocabulary")));
    Ok(())
}
