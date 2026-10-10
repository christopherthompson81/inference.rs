//! Convert Silero VAD's 16 kHz weights to the GGUF `SileroVad` loads: mlx-community/silero-vad-v6's
//! `model.safetensors` (the v6 release) or the PyPI package's `silero_vad_16k.safetensors`.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use inference_models_speech::silero::write_gguf;

#[derive(Parser)]
struct Args {
    /// The safetensors file to convert.
    #[arg(long)]
    weights: PathBuf,
    out: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args.out)?);
    write_gguf(&args.weights, &mut out)?;
    Ok(())
}
