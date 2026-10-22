//! A tool call that starts mid-reply under plain CUDA decode, on a real Qwen3.5 GGUF; skips unless
//! `INFERENCE_TEST_QWEN3_5_GGUF` is a local file and the build has CUDA. The reply runs on the resident CUDA sampler
//! until the call begins and its grammar needs the host sampler.

use std::collections::HashMap;
use std::path::Path;

use inference::{Function, GgufModelBuilder, RequestBuilder, TextMessageRole, Tool, ToolType};
use serde_json::{Value, json};

const GGUF_ENV: &str = "INFERENCE_TEST_QWEN3_5_GGUF";
const MAX_TOKENS: usize = 768;
const PROMPT: &str = "What is the weather in Paris right now? Look it up with the tool.";

fn weather_tool() -> anyhow::Result<Tool> {
    let parameters: HashMap<String, Value> = serde_json::from_value(json!({
        "type": "object",
        "properties": { "city": { "type": "string", "description": "The city to look up." } },
        "required": ["city"],
    }))?;
    Ok(Tool {
        tp: ToolType::Function,
        function: Function {
            description: Some("Get the current weather for a city.".to_string()),
            name: "get_weather".to_string(),
            parameters: Some(parameters),
            strict: None,
        },
    })
}

#[tokio::test]
async fn a_tool_call_mid_reply_completes_under_plain_cuda_decode() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        eprintln!("SKIP: resident CUDA decode needs the cuda feature");
        return Ok(());
    }
    let Some(file) = std::env::var(GGUF_ENV)
        .ok()
        .filter(|f| Path::new(f).is_file())
    else {
        eprintln!("SKIP: {GGUF_ENV} is not a local GGUF file");
        return Ok(());
    };
    let file = Path::new(&file);
    let dir = file.parent().unwrap_or(Path::new("."));
    let name = file.file_name().unwrap_or_default().to_string_lossy();
    let model = GgufModelBuilder::new(dir.to_string_lossy(), vec![name])
        .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?)
        .build()
        .await?;
    let response = model
        .send_chat_request(
            RequestBuilder::new()
                .add_message(TextMessageRole::User, PROMPT)
                .set_tools(vec![weather_tool()?])
                .set_sampler_max_len(MAX_TOKENS),
        )
        .await?;
    let choice = &response.choices[0];
    let calls = choice.message.tool_calls.as_deref().unwrap_or_default();
    assert_eq!(
        choice.finish_reason, "tool_calls",
        "reply: {:?}",
        choice.message.content
    );
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    let arguments: Value = serde_json::from_str(&calls[0].function.arguments)?;
    assert!(
        arguments["city"]
            .as_str()
            .is_some_and(|city| city.contains("Paris")),
        "{arguments}"
    );
    Ok(())
}
