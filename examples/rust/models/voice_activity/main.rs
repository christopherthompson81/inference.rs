//! Voice activity detection with Silero VAD.
//!
//! Run with: `cargo run --release --example voice_activity -p inference-examples -- <silero gguf> <audio file>`

use anyhow::Result;
use inference::{VoiceActivityModelBuilder, VoiceActivityRequest};

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let (Some(gguf), Some(audio)) = (args.next(), args.next()) else {
        anyhow::bail!("pass a Silero VAD GGUF and an audio file");
    };
    let model = VoiceActivityModelBuilder::new(gguf)
        .with_logging()
        .build()
        .await?;
    let activity = model
        .voice_activity(VoiceActivityRequest::new(), &std::fs::read(audio)?)
        .await?;
    for segment in &activity.segments {
        println!("{:>9.2}s - {:>9.2}s", segment.start, segment.end);
    }
    println!(
        "{} segments in {:.2}s",
        activity.segments.len(),
        activity.duration
    );
    Ok(())
}
