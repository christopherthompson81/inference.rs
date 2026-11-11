//! Replays `scripts/qwen3_tts_parity.sh`'s reference dumps: the prompt ids must match, the greedy codes are compared
//! frame by frame, and the codec decoder must reach `--min-snr` dB on the reference's own codes.

use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::Parser;
use inference_models_speech::SpeechOptions;
use inference_models_speech::qwen3_tts::{Qwen3Tts, Sampling};

const SAMPLE_RATE: u32 = 24_000;
use inference_tensor::{DType, Device};

#[derive(Parser)]
struct Args {
    /// A Qwen3-TTS checkpoint directory.
    #[arg(long)]
    model: PathBuf,
    /// The reference's `greedy.safetensors`.
    #[arg(long)]
    dump: PathBuf,
    #[arg(long)]
    text: String,
    #[arg(long)]
    voice: Option<String>,
    #[arg(long)]
    language: Option<String>,
    #[arg(long)]
    instructions: Option<String>,
    #[arg(long, default_value_t = 30.)]
    min_snr: f64,
    /// Fewest leading frames whose codes must equal the reference's; all of them when unset.
    #[arg(long)]
    min_equal_frames: Option<usize>,
    #[arg(long)]
    cpu: bool,
    #[arg(long)]
    bf16: bool,
    /// Writes our greedy audio here.
    #[arg(long)]
    wav: Option<PathBuf>,
    /// Also speaks the text with the default sampling and this seed, written to `--sampled-wav`.
    #[arg(long, requires = "sampled_wav")]
    seed: Option<u64>,
    #[arg(long)]
    sampled_wav: Option<PathBuf>,
}

fn write_wav(path: &std::path::Path, pcm: &[f32], rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in pcm {
        writer.write_sample(s)?;
    }
    writer.finalize()?;
    Ok(())
}

fn snr_db(reference: &[f32], ours: &[f32]) -> f64 {
    let (mut signal, mut noise) = (0f64, 0f64);
    for (r, o) in reference.iter().zip(ours) {
        signal += f64::from(*r).powi(2);
        noise += f64::from(r - o).powi(2);
    }
    10. * (signal / noise.max(f64::MIN_POSITIVE)).log10()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    let dtype = if args.bf16 { DType::BF16 } else { DType::F32 };
    let model = Qwen3Tts::load(&args.model, dtype, &device)?;
    let dump = inference_tensor::safetensors::load(&args.dump, &Device::Cpu)?;
    let get = |name: &str| {
        dump.get(name)
            .ok_or_else(|| anyhow::anyhow!("dump has no `{name}`"))
    };
    let mut failed = false;

    let ref_ids: Vec<i64> = get("ids")?.flatten_all()?.to_vec1()?;
    let ids: Vec<i64> = model
        .prompt_ids(&args.text)?
        .into_iter()
        .map(i64::from)
        .collect();
    println!(
        "prompt ids: {} ours, {} reference, equal {}",
        ids.len(),
        ref_ids.len(),
        ids == ref_ids
    );
    failed |= ids != ref_ids;

    let ref_codes = get("codes")?.to_dtype(DType::U32)?.to_vec2::<u32>()?;
    let ref_wav: Vec<f32> = get("wav")?.to_vec1()?;
    let wav = model.decode(&ref_codes)?;
    let snr = snr_db(&ref_wav, &wav);
    println!(
        "codec on reference codes: {} vs {} samples, SNR {snr:.1} dB",
        wav.len(),
        ref_wav.len()
    );
    failed |= wav.len() != ref_wav.len() || snr < args.min_snr;

    let options = SpeechOptions {
        voice: args.voice.clone(),
        language: args.language.clone(),
        instructions: args.instructions.clone(),
        ..SpeechOptions::default()
    };
    let start = std::time::Instant::now();
    let codes = model.greedy_codes(&args.text, &options, ref_codes.len() + 50)?;
    let elapsed = start.elapsed().as_secs_f64();
    let equal = codes
        .iter()
        .zip(&ref_codes)
        .take_while(|(a, b)| a == b)
        .count();
    let first_equal = codes
        .iter()
        .zip(&ref_codes)
        .take_while(|(a, b)| a[0] == b[0])
        .count();
    println!(
        "greedy: {} frames ours, {} reference; first {equal} frames equal, first {first_equal} first codes equal ({elapsed:.1} s)",
        codes.len(),
        ref_codes.len()
    );
    if let Some(row) = codes.get(equal) {
        println!("  frame {equal}: ours {row:?}");
        println!("  frame {equal}: ref  {:?}", ref_codes.get(equal));
    }
    failed |= equal < args.min_equal_frames.unwrap_or(ref_codes.len());
    if let Some(path) = &args.wav {
        write_wav(path, &model.decode(&codes)?, SAMPLE_RATE)?;
    }
    if let (Some(seed), Some(path)) = (args.seed, &args.sampled_wav) {
        let start = std::time::Instant::now();
        let out = model.generate(&args.text, &options, &Sampling::default(), seed)?;
        println!(
            "sampled (seed {seed}): {:.2} s of audio in {:.2} s",
            out.pcm.len() as f64 / f64::from(SAMPLE_RATE),
            start.elapsed().as_secs_f64()
        );
        write_wav(path, &out.pcm, SAMPLE_RATE)?;
    }
    if failed {
        bail!("Qwen3-TTS parity failed");
    }
    Ok(())
}
