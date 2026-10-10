//! Replays `scripts/parakeet_parity.sh`'s reference dumps: features and encoder output must reach `--min-cosine`, and
//! the emissions (token, frame, span) and transcript must match exactly.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use inference_models_speech::parakeet::{Parakeet, ParakeetFiles};
use inference_tensor::{DType, Device};

#[derive(Parser)]
struct Args {
    /// Directory with `config.json`, `processor_config.json`, `tokenizer.json` and `model.safetensors`.
    #[arg(long)]
    model: PathBuf,
    /// `<n>.safetensors` and `<n>.json` dumps from the reference.
    #[arg(long)]
    dumps: PathBuf,
    #[arg(long, default_value_t = 0.999)]
    min_cosine: f64,
    #[arg(long)]
    cpu: bool,
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    let files = ParakeetFiles {
        config: args.model.join("config.json"),
        processor_config: args.model.join("processor_config.json"),
        tokenizer: args.model.join("tokenizer.json"),
        weights: vec![args.model.join("model.safetensors")],
    };
    let model = Parakeet::load(&files, &device, DType::F32)?;
    let mut dumps = std::fs::read_dir(&args.dumps)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    dumps.retain(|p| p.extension().is_some_and(|e| e == "safetensors"));
    dumps.sort();
    let mut failed = false;
    for path in &dumps {
        let t = inference_tensor::safetensors::load(path, &Device::Cpu)?;
        let get = |name: &str| {
            t.get(name)
                .ok_or_else(|| anyhow::anyhow!("{} has no {name}", path.display()))
        };
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path.with_extension("json"))?)?;
        let pcm = get("pcm")?.to_vec1::<f32>()?;
        let want_features = get("features")?.flatten_all()?.to_vec1::<f32>()?;
        let want_encoded = get("encoded")?.flatten_all()?.to_vec1::<f32>()?;
        let want_emissions: Vec<[i64; 3]> = get("emissions")?
            .to_vec2::<i64>()?
            .into_iter()
            .map(|r| [r[0], r[1], r[2]])
            .collect();

        let start = std::time::Instant::now();
        let (features, _) = model.features(&pcm)?;
        let encoded = model.encode(&pcm, None)?;
        let emissions: Vec<[i64; 3]> = model
            .emissions(&encoded)?
            .iter()
            .map(|e| [i64::from(e.token), e.frame as i64, e.frames as i64])
            .collect();
        let text = model
            .transcribe(&pcm, model.sample_rate(), &Default::default())?
            .text;
        let elapsed = start.elapsed().as_secs_f64();
        // warm: the weights are paged in and the kernels loaded by the run above
        let warm = std::time::Instant::now();
        model.transcribe(&pcm, model.sample_rate(), &Default::default())?;
        let warm = warm.elapsed().as_secs_f64();

        let encoded = encoded.squeeze(0)?.flatten_all()?.to_vec1::<f32>()?;
        let feature_cos = if features.len() == want_features.len() {
            cosine(&features, &want_features)
        } else {
            0.0
        };
        let encoded_cos = if encoded.len() == want_encoded.len() {
            cosine(&encoded, &want_encoded)
        } else {
            0.0
        };
        let same_emissions = emissions == want_emissions;
        let first_diff = emissions
            .iter()
            .zip(&want_emissions)
            .position(|(a, b)| a != b);
        let same_text = meta["text"].as_str() == Some(text.as_str());
        let ok = feature_cos >= args.min_cosine
            && encoded_cos >= args.min_cosine
            && same_emissions
            && same_text;
        failed |= !ok;
        println!(
            "{} {}: features cos {feature_cos:.6} ({} vs {}), encoder cos {encoded_cos:.6}, emissions {}/{}{} {}, text {}, {elapsed:.2}s (one warm transcribe {warm:.2}s)",
            if ok { "ok  " } else { "FAIL" },
            meta["wav"]
                .as_str()
                .unwrap_or_default()
                .rsplit('/')
                .next()
                .unwrap_or_default(),
            features.len(),
            want_features.len(),
            emissions.len(),
            want_emissions.len(),
            first_diff
                .map(|i| format!(" (first diff at {i})"))
                .unwrap_or_default(),
            if same_emissions { "match" } else { "differ" },
            if same_text {
                "matches".to_string()
            } else {
                format!("differs:\n  ours {text:?}\n  want {:?}", meta["text"])
            },
        );
    }
    anyhow::ensure!(!failed, "parity failed");
    Ok(())
}
