use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use serde_json::{Value, json};

use super::tiny_engine_with;
use crate::{
    engine::{EngineCallbacks, Tool, ToolCallbackKind, ToolCallbackWithTool},
    logits_processors::in_place,
    media_source::MediaAttachments,
    operations::TokenizeRequest,
};

const TOOLS: [&str; 2] = ["first_lookup", "second_lookup"];
// Long enough that two calls run one after the other could not overlap by accident.
const TOOL_TIME: Duration = Duration::from_millis(300);
// The tiny tokenizer's `</s>`.
const EOS: u32 = 2;
const SCRIPT: &str = "script";
#[cfg(feature = "code-execution")]
const EXECUTE_PYTHON: &str = "inference_execute_python";
// Byte-level tokens: the two calls' JSON runs to about 180.
#[cfg(feature = "code-execution")]
const SANDBOX_SCRIPT_TOKENS: usize = 512;

#[derive(Default)]
struct Concurrency {
    running: AtomicUsize,
    most: AtomicUsize,
    calls: AtomicUsize,
}

fn host_tool(name: &str, seen: Arc<Concurrency>) -> anyhow::Result<ToolCallbackWithTool> {
    let tool: Tool = serde_json::from_value(json!({
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object", "properties": {}}},
    }))?;
    let run = move |_: &crate::engine::CalledFunction, _: &crate::engine::ToolCallContext| {
        let now = seen.running.fetch_add(1, Ordering::SeqCst) + 1;
        seen.most.fetch_max(now, Ordering::SeqCst);
        std::thread::sleep(TOOL_TIME);
        seen.running.fetch_sub(1, Ordering::SeqCst);
        seen.calls.fetch_add(1, Ordering::SeqCst);
        Ok("found".to_string())
    };
    Ok(ToolCallbackWithTool {
        callback: ToolCallbackKind::Text(Arc::new(run)),
        tool,
    })
}

// Forces `tokens` then EOS at the start of every generation, which a context that did not grow by one marks.
fn scripted(tokens: Vec<u32>) -> Arc<dyn inference_core::CustomLogitsProcessor> {
    let state = Mutex::new((0_usize, 0_usize));
    in_place(move |logits, context| {
        let mut state = state.lock().unwrap();
        let (position, last_len) = &mut *state;
        if context.len() != *last_len + 1 {
            *position = 0;
        }
        *last_len = context.len();
        let next = tokens.get(*position).copied().unwrap_or(EOS);
        *position += 1;
        logits.fill(f32::NEG_INFINITY);
        logits[next as usize] = 0.0;
        Ok(())
    })
}

async fn script(engine: &crate::Engine, calls: Value) -> anyhow::Result<()> {
    let tokenized = TokenizeRequest {
        model: None,
        text: calls.to_string(),
        add_special_tokens: false,
    };
    let tokens = engine.tokenize(tokenized).await?.tokens;
    engine.register_logits_processor(SCRIPT, scripted(tokens))?;
    Ok(())
}

// An engine whose model calls both tools in one round, every round.
async fn calling_both(
    seen: &Arc<Concurrency>,
) -> anyhow::Result<(tempfile::TempDir, crate::Engine)> {
    let tools = TOOLS
        .iter()
        .map(|name| host_tool(name, seen.clone()))
        .collect::<anyhow::Result<_>>()?;
    let callbacks = EngineCallbacks {
        tools,
        search: None,
    };
    let (dir, engine) = tiny_engine_with(callbacks).await?;
    script(
        &engine,
        json!(TOOLS.map(|name| json!({"name": name, "arguments": {}}))),
    )
    .await?;
    Ok((dir, engine))
}

// One round runs; the second, scripted the same way, stops at the limit and comes back with its calls.
fn request(stream: bool) -> Value {
    json!({
        "model": "default",
        "messages": [{"role": "user", "content": "look both up"}],
        "max_tokens": 128,
        "max_tool_rounds": 1,
        "logits_processors": [SCRIPT],
        "stream": stream,
    })
}

#[tokio::test(flavor = "multi_thread")]
async fn every_call_of_a_round_runs_at_once_and_reports_its_own_id() -> anyhow::Result<()> {
    let seen = Arc::new(Concurrency::default());
    let (_dir, engine) = calling_both(&seen).await?;
    let request = request(false).to_string();
    let response = engine
        .chat_json(request.as_bytes(), MediaAttachments::default())
        .await?;
    let response: Value = serde_json::from_str(&response)?;

    assert_eq!(seen.calls.load(Ordering::SeqCst), TOOLS.len());
    assert_eq!(
        seen.most.load(Ordering::SeqCst),
        TOOLS.len(),
        "the calls ran one at a time"
    );
    let records = response["agentic_tool_calls"]
        .as_array()
        .expect("tool call records");
    let names: Vec<_> = records
        .iter()
        .map(|record| record["name"].clone())
        .collect();
    assert_eq!(names, TOOLS.map(Value::from));
    let ids: std::collections::HashSet<_> = records
        .iter()
        .map(|record| record["tool_call_id"].as_str().unwrap())
        .collect();
    assert_eq!(ids.len(), TOOLS.len(), "{response}");
    assert!(
        records
            .iter()
            .all(|record| record["result_content"] == "found"),
        "{response}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_streamed_round_pairs_each_calls_phases_by_its_id() -> anyhow::Result<()> {
    use futures::StreamExt;

    let seen = Arc::new(Concurrency::default());
    let (_dir, engine) = calling_both(&seen).await?;
    let request = serde_json::from_value(request(true))?;
    let mut stream = engine
        .chat_stream(request, MediaAttachments::default())
        .await?;
    let mut phases = Vec::new();
    while let Some(event) = stream.next().await {
        if let crate::engine_chat::ChatStreamEvent::AgenticToolCallProgress(progress) = event {
            let phase = progress.to_json()["phase"].as_str().unwrap().to_string();
            phases.push((progress.tool_call_id, progress.tool_name, phase));
        }
    }
    assert_eq!(
        seen.most.load(Ordering::SeqCst),
        TOOLS.len(),
        "the calls ran one at a time"
    );
    assert_eq!(phases.len(), 2 * TOOLS.len(), "{phases:?}");
    let ids: std::collections::HashSet<_> = phases.iter().map(|(id, ..)| id).collect();
    assert_eq!(ids.len(), TOOLS.len(), "{phases:?}");
    for id in ids {
        let mine: Vec<_> = phases.iter().filter(|(other, ..)| other == id).collect();
        assert_eq!(mine.len(), 2, "{phases:?}");
        assert_eq!(
            (mine[0].2.as_str(), mine[1].2.as_str()),
            ("calling", "complete")
        );
        assert_eq!(mine[0].1, mine[1].1);
    }
    Ok(())
}

// No spaces: the tiny tokenizer has no token for its space marker.
#[cfg(feature = "code-execution")]
#[tokio::test(flavor = "multi_thread")]
async fn calls_into_the_sandbox_keep_the_models_order() -> anyhow::Result<()> {
    let python = EXECUTE_PYTHON;
    let extra = json!({"agentic": {"code_execution": {"timeout_secs": 30}}});
    let (_dir, engine) = super::tiny_engine_from(extra, EngineCallbacks::default()).await?;
    let calls = json!([
        {"name": python, "arguments": {"code": "__import__('time').sleep(0.3);x=41"}},
        {"name": python, "arguments": {"code": "print(x+1)"}},
    ]);
    script(&engine, calls).await?;
    let mut request = request(false);
    request["max_tokens"] = json!(SANDBOX_SCRIPT_TOKENS);
    request["tools"] = json!([{"type": "code_interpreter", "container": {"type": "auto"}}]);
    let response = engine
        .chat_json(request.to_string().as_bytes(), MediaAttachments::default())
        .await?;
    let response: Value = serde_json::from_str(&response)?;
    let records = response["agentic_tool_calls"]
        .as_array()
        .expect("tool call records");
    assert_eq!(records.len(), 2, "{response}");
    let second = records[1]["result_content"].as_str().unwrap_or_default();
    assert!(
        second.contains("42"),
        "the second call ran before the first: {response}"
    );
    Ok(())
}
