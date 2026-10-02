//! Code execution producing first-class output files.
//!
//! Declare required output files via `RequestBuilder::require_file`. The
//! engine surfaces them in `response.files`, including an error placeholder
//! if the model never wrote one.
//!
//! Run with: `cargo run --release --features code-execution --example code_execution_files -p inference-examples`

use anyhow::Result;
use inference::{
    CodeExecutionConfig, IsqBits, ModelBuilder, RequestBuilder, TextMessageRole, TextMessages,
    api::sdk::SandboxPolicy,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Enable the OS-level sandbox (Linux/macOS) with the default policy.
    let model = ModelBuilder::new("google/gemma-4-E4B-it")
        .with_auto_isq(IsqBits::Four)
        .with_logging()
        .with_code_execution(CodeExecutionConfig {
            sandbox_policy: Some(SandboxPolicy::default()),
            ..CodeExecutionConfig::default()
        })
        .build()
        .await?;

    let messages = TextMessages::new().add_message(
        TextMessageRole::User,
        "Use Python and matplotlib to plot y=sin(x) for x in [0, 2*pi] and save to plot.png. Also write the data to data.csv.",
    );
    let request = RequestBuilder::from(messages)
        .require_file_described("plot.png", "png", "Sine plot saved as PNG")
        .require_file("data.csv");

    let response = model.send_chat_request(request).await?;

    if let Some(files) = response.files.as_ref() {
        for f in files {
            println!(
                "file id={} name={} bytes={} error={}",
                f.id,
                f.name,
                f.bytes,
                f.is_error()
            );
            if f.is_error() {
                continue;
            }
            if f.is_truncated() {
                // Wire body was elided; fetch the full body from the engine's file store.
                std::fs::write(&f.name, model.file_content(&f.id)?.bytes)?;
                println!("  fetched truncated body via file_content");
            } else {
                f.save(&f.name)?;
            }
        }
    }

    Ok(())
}
