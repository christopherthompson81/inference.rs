//! Speaker diarization with Nemotron-3 Diarization: who speaks when, as RTTM.
//!
//! Run with: `cargo run --release --example diarization -p inference-examples -- <audio file>`

use anyhow::Result;
use inference::{DiarizationModelBuilder, DiarizationRequest, DiarizationResponseFormat};

#[tokio::main]
async fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("pass the path of an audio file to diarize"))?;
    let model = DiarizationModelBuilder::new("nvidia/Nemotron-3-Diarization")
        .with_logging()
        .build()
        .await?;
    let mut request = DiarizationRequest::new();
    request.response_format = DiarizationResponseFormat::Rttm;
    let output = model.diarization(request, &std::fs::read(path)?).await?;
    print!("{}", output.body);
    Ok(())
}
