//! Replays `scripts/kokoro_parity.sh`'s reference dumps: durations must match and the waveform reach `--min-snr` dB.
//! (The source's sine runs on ~1e5 rad phases, so f32 op order alone caps agreement; torch CPU vs CUDA gets 25-27 dB.)

use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::Parser;
use inference_models_speech::kokoro::{KokoroModel, SourceNoise};
use inference_tensor::{DType, Device};

#[derive(Parser)]
struct Args {
    /// Directory with `config.json` and `kokoro-v1_0.pth`.
    #[arg(long)]
    model: PathBuf,
    /// `*.safetensors` dumps from the reference.
    #[arg(long)]
    dumps: PathBuf,
    #[arg(long, default_value_t = 30.)]
    min_snr: f64,
    #[arg(long)]
    cpu: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    let model = KokoroModel::from_pth(
        &args.model.join("config.json"),
        &args.model.join("kokoro-v1_0.pth"),
        &device,
    )?;
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
        let ids = get("ids")?
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let style = get("ref_s")?.flatten_all()?.to_vec1::<f32>()?;
        let noise = get("noise")?.flatten_all()?.to_vec1::<f32>()?;
        let speed = get("speed")?.flatten_all()?.to_vec1::<f32>()?[0];
        let want_dur = get("durations")?
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let want = get("audio")?.flatten_all()?.to_vec1::<f32>()?;

        let start = std::time::Instant::now();
        let out = model.synthesize_ids(&ids, &style, speed, &mut SourceNoise::Given(noise))?;
        let elapsed = start.elapsed().as_secs_f32();
        let name = path.file_stem().unwrap_or_default().to_string_lossy();
        if out.durations != want_dur {
            let first = out
                .durations
                .iter()
                .zip(&want_dur)
                .position(|(a, b)| a != b);
            println!("{name}: durations differ (first at token {first:?})");
            failed = true;
            continue;
        }
        if out.audio.len() != want.len() {
            bail!(
                "{name}: {} samples, reference {}",
                out.audio.len(),
                want.len()
            );
        }
        let peak = want.iter().fold(0f32, |m, x| m.max(x.abs()));
        let max_diff = out
            .audio
            .iter()
            .zip(&want)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let err: f64 = out
            .audio
            .iter()
            .zip(&want)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum();
        let sig: f64 = want.iter().map(|x| f64::from(*x).powi(2)).sum();
        let snr = 10. * (sig / err.max(f64::MIN_POSITIVE)).log10();
        let rel = max_diff / peak;
        println!(
            "{name}: {} tokens, {} samples, max |diff| {max_diff:.2e} ({rel:.2e} of peak), SNR {snr:.1} dB, {elapsed:.2}s",
            ids.len(),
            want.len()
        );
        failed |= snr < args.min_snr;
    }
    if failed {
        bail!("parity failed");
    }
    Ok(())
}
