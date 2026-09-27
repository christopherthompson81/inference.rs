//! Cost of the JSON the engine surface adds per streamed chunk; run with `--ignored --nocapture`.

use std::time::Instant;

use inference_api::engine_chat::ChatStreamEvent;
use inference_core::{ChatCompletionChunkResponse, ChunkChoice, Delta};

const ITERATIONS: u32 = 100_000;

#[test]
#[ignore = "benchmark"]
fn per_chunk_json_cost() {
    let event = ChatStreamEvent::Chunk(ChatCompletionChunkResponse {
        id: "chatcmpl-0".to_string(),
        choices: vec![ChunkChoice {
            finish_reason: None,
            stop_sequence: None,
            index: 0,
            delta: Delta {
                content: Some(" token".to_string()),
                role: "assistant".to_string(),
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs: None,
        }],
        created: 1_700_000_000,
        model: "org/model".to_string(),
        system_fingerprint: "local".to_string(),
        object: "chat.completion.chunk".to_string(),
        usage: None,
        adapter_generation: None,
        session_id: None,
    });
    let start = Instant::now();
    let mut bytes = 0;
    for _ in 0..ITERATIONS {
        let json = event.to_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        bytes += json.len() + usize::from(parsed.is_object());
    }
    let per_chunk = start.elapsed() / ITERATIONS;
    println!(
        "serialize + client parse: {per_chunk:?} per chunk ({} bytes)",
        bytes / ITERATIONS as usize
    );
}
