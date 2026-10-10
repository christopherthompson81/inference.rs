//! Writes a recogniser's `.nemo` as an `asr` GGUF under a precision policy: `--matrices` for the large matrices,
//! `--keep fragment=type` for the tensors whose names contain `fragment`, as a sensitivity sweep compares them.

use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::Parser;
use inference_models_speech::parakeet::{QuantPolicy, write_asr_gguf};
use inference_tensor::quantized::GgmlDType;

const TYPES: [(&str, GgmlDType); 9] = [
    ("f32", GgmlDType::F32),
    ("f16", GgmlDType::F16),
    ("q8_0", GgmlDType::Q8_0),
    ("q6_k", GgmlDType::Q6K),
    ("q5_k", GgmlDType::Q5K),
    ("q5_0", GgmlDType::Q5_0),
    ("q4_k", GgmlDType::Q4K),
    ("q4_0", GgmlDType::Q4_0),
    ("q3_k", GgmlDType::Q3K),
];

fn dtype(name: &str) -> Result<GgmlDType> {
    match TYPES.iter().find(|(n, _)| *n == name) {
        Some((_, d)) => Ok(*d),
        None => bail!("`{name}` is not one of {:?}", TYPES.map(|(n, _)| n)),
    }
}

#[derive(Parser)]
struct Args {
    /// The recogniser's `.nemo`, its weights in full precision.
    #[arg(long)]
    nemo: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// A named policy: `q8_0` (NVIDIA's), `q8_0-decoder` (decoder at F16), `q6_k-mix` (smaller, more accurate).
    #[arg(long, conflicts_with = "matrices")]
    preset: Option<String>,
    #[arg(long, default_value = "q8_0")]
    matrices: String,
    /// `fragment=type`, repeatable; the first whose fragment the name contains wins.
    #[arg(long)]
    keep: Vec<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let overrides = args
        .keep
        .iter()
        .map(|k| match k.split_once('=') {
            Some((fragment, d)) => Ok((fragment.to_string(), dtype(d)?)),
            None => bail!("`{k}` is not fragment=type"),
        })
        .collect::<Result<Vec<_>>>()?;
    let mut policy = match args.preset.as_deref() {
        None => QuantPolicy::uniform(dtype(&args.matrices)?),
        Some("q8_0") => QuantPolicy::uniform(GgmlDType::Q8_0),
        Some("q8_0-decoder") => QuantPolicy::q8_0_decoder(),
        Some("q6_k-mix") => QuantPolicy::q6_k_mix(),
        Some(other) => bail!("`{other}` is not a preset: q8_0, q8_0-decoder, q6_k-mix"),
    };
    // a `--keep` wins over the preset's own overrides
    policy.overrides.splice(0..0, overrides);
    let chosen = write_asr_gguf(&args.nemo, &args.out, &policy)?;
    let mut counts: Vec<(String, usize)> = Vec::new();
    for (_, d) in &chosen {
        let name = format!("{d:?}");
        match counts.iter_mut().find(|(n, _)| *n == name) {
            Some((_, c)) => *c += 1,
            None => counts.push((name, 1)),
        }
    }
    let bytes = std::fs::metadata(&args.out)?.len();
    println!(
        "{}: {:.1} MiB, {counts:?}",
        args.out.display(),
        bytes as f64 / (1 << 20) as f64
    );
    Ok(())
}
