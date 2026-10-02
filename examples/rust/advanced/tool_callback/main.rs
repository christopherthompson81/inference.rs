//! Tool callbacks for automatic server-side tool execution.
//!
//! Run with: `cargo run --release --example tool_callback -p inference-examples`

use anyhow::Result;
use inference::{
    CalledFunction, Function, IsqBits, ModelBuilder, SearchResult, TextMessageRole, TextMessages,
    Tool, ToolCallContext, ToolType,
};
use std::fs;
use std::sync::Arc;
use walkdir::WalkDir;

fn local_search(query: &str) -> Result<Vec<SearchResult>> {
    let mut results = Vec::new();
    for entry in WalkDir::new(".") {
        let entry = entry?;
        if entry.file_type().is_file() {
            let name = entry.file_name().to_string_lossy();
            if name.contains(query) {
                let path = entry.path().display().to_string();
                let content = fs::read_to_string(entry.path()).unwrap_or_default();
                results.push(SearchResult {
                    title: name.into_owned(),
                    description: path.clone(),
                    url: path,
                    content,
                });
            }
        }
    }
    results.sort_by_key(|r| r.title.clone());
    results.reverse();
    Ok(results)
}

#[tokio::main]
async fn main() -> Result<()> {
    let parameters = std::collections::HashMap::from([(
        "query".to_string(),
        serde_json::json!({"type": "string", "description": "Query"}),
    )]);
    let tool = Tool {
        tp: ToolType::Function,
        function: Function {
            description: Some("Local filesystem search".to_string()),
            name: "local_search".to_string(),
            parameters: Some(parameters),
            strict: None,
        },
    };

    // Every request offers the tool, and the engine runs the callback when the model calls it.
    let model = ModelBuilder::new("google/gemma-4-E4B-it")
        .with_auto_isq(IsqBits::Four)
        .with_logging()
        .with_tool_callback_and_tool(
            Arc::new(|f: &CalledFunction, _ctx: &ToolCallContext| {
                let args: serde_json::Value = serde_json::from_str(&f.arguments)?;
                let query = args["query"].as_str().unwrap_or("");
                Ok(serde_json::to_string(&local_search(query)?)?)
            }),
            tool,
        )
        .build()
        .await?;

    let messages =
        TextMessages::new().add_message(TextMessageRole::User, "Where is Cargo.toml in this repo?");

    let response = model.send_chat_request(messages).await?;
    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    Ok(())
}
