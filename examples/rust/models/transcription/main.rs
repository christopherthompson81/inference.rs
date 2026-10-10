//! Speech recognition with timestamps using a Parakeet model.
//!
//! Run with: `cargo run --release --example transcription -p inference-examples -- <audio file>`

use anyhow::Result;
use inference::{
    TimestampGranularity, TranscriptionModelBuilder, TranscriptionRequest,
    TranscriptionResponseFormat,
};

#[tokio::main]
async fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("pass the path of an audio file to transcribe"))?;
    let model = TranscriptionModelBuilder::new("nvidia/parakeet-tdt-0.6b-v3")
        .with_logging()
        .build()
        .await?;

    let mut request = TranscriptionRequest::new(TranscriptionResponseFormat::VerboseJson);
    request.timestamp_granularities = vec![TimestampGranularity::Word];
    let transcript = model.transcription(request, &std::fs::read(path)?).await?;
    println!("{}", transcript.body);

    Ok(())
}
