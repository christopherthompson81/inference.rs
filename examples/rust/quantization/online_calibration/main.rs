//! Online calibration: serve an ISQ model, collect activation statistics from real traffic,
//! then requantize from the source weights and hot-swap the layers, all without a restart.
//!
//! Run with: `cargo run --release --example online_calibration -p inference-examples`

use anyhow::Result;
use inference::{
    IsqBits, ModelBuilder, TextMessageRole, TextMessages, api::operations::CalibrationAction,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new("google/gemma-4-E4B-it")
        .with_auto_isq(IsqBits::Four)
        .with_logging()
        .build()
        .await?;

    let messages = TextMessages::new().add_message(
        TextMessageRole::User,
        "Explain how a hash map works, briefly.",
    );

    // Collect activation statistics while serving normally (~15% decode overhead while on).
    model.calibration(CalibrationAction::Start, None).await?;
    for _ in 0..8 {
        model.send_chat_request(messages.clone()).await?;
    }

    let status = model.calibration(CalibrationAction::Status, None).await?;
    println!(
        "Collecting on {}/{} layers, {} token rows seen",
        status.layers_tracking, status.layers, status.total_rows
    );

    // Requantize from the source weights with the traffic-derived importance matrix and
    // hot-swap each layer. The optional path also saves the imatrix for reuse via --imatrix.
    let apply = CalibrationAction::Apply {
        save_cimatrix: Some("traffic.cimatrix".into()),
    };
    model.calibration(apply, None).await?;

    let response = model.send_chat_request(messages).await?;
    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
