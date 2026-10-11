//! Replays `scripts/parakeet_parity.sh`'s reference dumps: features and encoder output must reach `--min-cosine`, and
//! the emissions (token, frame, span) and transcript must match exactly.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use inference_models_speech::TranscriptionOptions;
use inference_models_speech::parakeet::{HeadKind, Parakeet, ParakeetFiles};
use inference_tensor::{DType, Device};

// the `inference` binary's allocator, so CPU timings match it
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

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
    /// Load ours from this `.nemo` instead, against the same transformers reference.
    #[arg(long)]
    nemo: Option<PathBuf>,
    /// Load ours from this `asr` GGUF instead. Its 8-bit weights move the encoder and the odd word, so a clip
    /// passes on `--min-cosine` and `--max-wer` rather than on identical emissions.
    #[arg(long, conflicts_with = "nemo")]
    gguf: Option<PathBuf>,
    /// Beam-search over this many hypotheses instead of greedily: MALSD for TDT, the default search for RNN-T.
    #[arg(long)]
    beam: Option<usize>,
    /// The WER a quantized checkpoint may reach: 8-bit weights swap a spelling and flip words where voices overlap.
    #[arg(long, default_value_t = 0.05)]
    max_wer: f64,
    /// Also stream each clip in pieces of this many milliseconds; the streamed transcription must equal the offline one.
    #[arg(long)]
    stream_ms: Option<f64>,
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

// word-level edit distance, and the reference's word count
fn word_edits(ours: &str, reference: &str) -> (usize, usize) {
    let (a, b): (Vec<&str>, Vec<&str>) = (
        ours.split_whitespace().collect(),
        reference.split_whitespace().collect(),
    );
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for x in &a {
        let mut diagonal = row[0];
        row[0] += 1;
        for (j, y) in b.iter().enumerate() {
            let next = (diagonal + usize::from(x != y))
                .min(row[j] + 1)
                .min(row[j + 1] + 1);
            diagonal = row[j + 1];
            row[j + 1] = next;
        }
    }
    (row[b.len()], b.len())
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
    let model = match (&args.nemo, &args.gguf) {
        (Some(nemo), _) => Parakeet::load_nemo(nemo, &device, DType::F32)?,
        (None, Some(gguf)) => Parakeet::load_gguf(gguf, &device, DType::F32)?,
        (None, None) => Parakeet::load(&files, &device, DType::F32)?,
    };
    anyhow::ensure!(
        args.stream_ms.is_none() || model.can_stream(),
        "--stream-ms needs a cache-aware streaming checkpoint"
    );
    let mut dumps = std::fs::read_dir(&args.dumps)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    dumps.retain(|p| p.extension().is_some_and(|e| e == "safetensors"));
    dumps.sort();
    let options = TranscriptionOptions {
        beam_size: args.beam,
        ..Default::default()
    };
    let mut failed = false;
    // across the clips: word edits and reference words, and each clip's encoder cosine
    let (mut edits, mut words, mut cosines) = (0, 0, Vec::new());
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
        let encode_start = std::time::Instant::now();
        let encoded = model.encode(&pcm, None)?;
        let encode_seconds = encode_start.elapsed().as_secs_f64();
        let emissions: Vec<[i64; 3]> = model
            .emissions(&encoded, args.beam)?
            .iter()
            .map(|e| [i64::from(e.token), e.frame as i64, e.frames as i64])
            .collect();
        let text = model.transcribe(&pcm, model.sample_rate(), &options)?.text;
        let elapsed = start.elapsed().as_secs_f64();
        // warm: the weights are paged in and the kernels loaded by the run above
        let warm = std::time::Instant::now();
        model.transcribe(&pcm, model.sample_rate(), &options)?;
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
        cosines.push(encoded_cos);
        // NeMo's TDT MALSD stamps a token at its frame plus its duration, its RNN-T beam at its frame
        let same_emissions = if args.beam.is_some() {
            let stamp = |e: &[i64; 3]| match model.head() {
                HeadKind::Tdt => e[1] + e[2],
                _ => e[1],
            };
            emissions.len() == want_emissions.len()
                && emissions
                    .iter()
                    .zip(&want_emissions)
                    .all(|(e, w)| e[0] == w[0] && stamp(e) == w[1])
        } else {
            emissions == want_emissions
        };
        let first_diff = emissions
            .iter()
            .zip(&want_emissions)
            .position(|(a, b)| a != b);
        let same_text = meta["text"].as_str() == Some(text.as_str());
        let (clip_edits, clip_words) = word_edits(&text, meta["text"].as_str().unwrap_or_default());
        let wer = clip_edits as f64 / clip_words.max(1) as f64;
        (edits, words) = (edits + clip_edits, words + clip_words);
        let matches = if args.gguf.is_some() {
            wer <= args.max_wer
        } else {
            same_emissions && same_text
        };
        let mut streamed_ok = true;
        if let Some(ms) = args.stream_ms {
            let piece = ((ms / 1000.0) * f64::from(model.sample_rate())).max(1.0) as usize;
            let offline = model.transcribe(&pcm, model.sample_rate(), &options)?;
            let start = std::time::Instant::now();
            let mut stream = model.stream(&options)?;
            let (mut deltas, mut slowest) = (String::new(), 0f64);
            for chunk in pcm.chunks(piece) {
                let push = std::time::Instant::now();
                deltas.push_str(&model.stream_push(&mut stream, chunk)?);
                slowest = slowest.max(push.elapsed().as_secs_f64());
            }
            let (last, streamed) = model.stream_finish(stream)?;
            deltas.push_str(&last);
            let audio = pcm.len() as f64 / f64::from(model.sample_rate());
            streamed_ok = streamed == offline && deltas.trim_end() == offline.text;
            println!(
                "     streamed in {ms} ms pieces: {}, {:.2}x real time, slowest push {:.0} ms",
                if streamed_ok {
                    "equals offline"
                } else {
                    "DIFFERS from offline"
                },
                audio / start.elapsed().as_secs_f64(),
                slowest * 1000.0
            );
        }
        let ok = feature_cos >= args.min_cosine
            && encoded_cos >= args.min_cosine
            && matches
            && streamed_ok;
        failed |= !ok;
        println!(
            "{} {}: features cos {feature_cos:.6} ({} vs {}), encoder cos {encoded_cos:.6}, emissions {}/{}{} {}, text {}, {elapsed:.2}s (one warm transcribe {warm:.2}s, first encode {encode_seconds:.2}s)",
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
            } else if args.gguf.is_some() {
                format!(
                    "WER {:.2}%:\n  ours {text:?}\n  want {:?}",
                    wer * 100.0,
                    meta["text"]
                )
            } else {
                format!("differs:\n  ours {text:?}\n  want {:?}", meta["text"])
            },
        );
    }
    let mean = cosines.iter().sum::<f64>() / cosines.len().max(1) as f64;
    let min = cosines.iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "summary: WER {:.3}% ({edits} edits over {words} words), encoder cos min {min:.6} mean {mean:.6}",
        edits as f64 / words.max(1) as f64 * 100.0
    );
    anyhow::ensure!(!failed, "parity failed");
    Ok(())
}
