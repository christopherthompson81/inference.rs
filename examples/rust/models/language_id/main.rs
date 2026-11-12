//! Spoken language identification with VoxLingua107 (`speechbrain/lang-id-voxlingua107-ecapa`).
//!
//! Run with: `cargo run --release --example language_id -p inference-examples -- <model> <audio file>`

use anyhow::Result;
use inference::{LanguageIdModelBuilder, LanguageIdRequest};

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let (Some(model_id), Some(audio)) = (args.next(), args.next()) else {
        anyhow::bail!("pass a language identification model and an audio file");
    };
    let model = LanguageIdModelBuilder::new(model_id)
        .with_logging()
        .build()
        .await?;
    let identified = model
        .language_id(LanguageIdRequest::new(), &std::fs::read(audio)?)
        .await?;
    for scored in &identified.languages {
        println!(
            "{:>4} {:<16} {:.3}",
            scored.language, scored.name, scored.probability
        );
    }
    if identified.ambiguous {
        println!("ambiguous: the likeliest language is under even odds");
    }
    Ok(())
}
