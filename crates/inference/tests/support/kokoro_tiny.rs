//! Builds a tiny random-weight Kokoro checkpoint (safetensors, config, raw voice packs) at test time.
#![allow(dead_code)]

use std::path::Path;

use inference_models_speech::kokoro::{KokoroConfig, KokoroModel};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};

#[path = "recording.rs"]
mod recording;

const CONFIG: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/kokoro_tiny/config.json"
);
// invented voice names; the first sorts first, so it is the default when the reference's default is absent. Their
// prefixes pick the language text is read in: American and British English, and Japanese, which has no text input yet.
pub const VOICES: [&str; 3] = ["af_tiny_one", "bm_tiny_two", "jf_tiny_three"];
const VOICE_ROWS: usize = 510;
const VOICE_DIM: usize = 256;
const VOICE_STD: f32 = 0.3;
const VOICE_SEED: u64 = 0x0C0C_0A1D;
// past the reader's inline limit (1 << 20 values), as audio.cpp's G2P resources are, and not a file the model reads
const FILLER_LEN: usize = (1 << 20) + 1;
const FILLER_BYTE: u8 = 0x5a;

/// The committed tiny config, random weights for every tensor the model loads, and two raw voice packs.
pub fn tiny_kokoro_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let cfg: KokoroConfig = serde_json::from_str(&std::fs::read_to_string(CONFIG)?)?;
    let dir = recording::record_plain_checkpoint(&[Path::new(CONFIG)], |vb| {
        KokoroModel::new(&cfg, vb).map(|_| ())
    })?;
    let voices = dir.path().join("voices");
    std::fs::create_dir(&voices)?;
    let normal = Normal::new(0f32, VOICE_STD)?;
    for (i, name) in VOICES.iter().enumerate() {
        let mut rng = StdRng::seed_from_u64(VOICE_SEED + i as u64);
        let bytes = (0..VOICE_ROWS * VOICE_DIM)
            .flat_map(|_| normal.sample(&mut rng).to_le_bytes())
            .collect::<Vec<_>>();
        std::fs::write(voices.join(format!("{name}.bin")), bytes)?;
    }
    Ok(dir)
}

/// The checkpoint as an audio.cpp-layout F32 GGUF, with a filler file large enough that the embedded data is skipped.
pub fn tiny_kokoro_gguf(checkpoint: &Path, out_dir: &Path) -> anyhow::Result<std::path::PathBuf> {
    use inference_tensor::quantized::{GgmlDType, QTensor, gguf_file::Value};

    let mut tensors = inference_tensor::safetensors::load(
        checkpoint.join("model.safetensors"),
        &inference_tensor::Device::Cpu,
    )?
    .into_iter()
    .collect::<Vec<_>>();
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    let mut metadata = vec![
        (
            "general.architecture".to_string(),
            Value::String("kokoro_tts".into()),
        ),
        (
            "kokoro.tensor_names".to_string(),
            Value::Array(
                tensors
                    .iter()
                    .map(|(n, _)| Value::String(n.clone()))
                    .collect(),
            ),
        ),
    ];
    for (name, t) in &tensors {
        let dims = t.dims().iter().map(|&d| Value::I64(d as i64)).collect();
        metadata.push((format!("kokoro.tensor_shape.{name}"), Value::Array(dims)));
    }
    let mut files = vec![
        (
            "config.json".to_string(),
            std::fs::read(checkpoint.join("config.json"))?,
        ),
        ("g2p/filler.bin".to_string(), vec![FILLER_BYTE; FILLER_LEN]),
    ];
    for name in VOICES {
        files.push((
            format!("voices/{name}.bin"),
            std::fs::read(checkpoint.join("voices").join(format!("{name}.bin")))?,
        ));
    }
    let mut offsets = vec![Value::U64(0)];
    let mut data = Vec::new();
    for (_, bytes) in &files {
        data.extend(bytes.iter().map(|&b| Value::U8(b)));
        offsets.push(Value::U64(data.len() as u64));
    }
    metadata.push((
        "audiocpp.embedded_files.names".to_string(),
        Value::Array(
            files
                .iter()
                .map(|(n, _)| Value::String(n.clone()))
                .collect(),
        ),
    ));
    metadata.push((
        "audiocpp.embedded_files.offsets".to_string(),
        Value::Array(offsets),
    ));
    metadata.push((
        "audiocpp.embedded_files.data".to_string(),
        Value::Array(data),
    ));

    let quantized = tensors
        .iter()
        .enumerate()
        .map(|(i, (_, t))| Ok((format!("kokoro.{i}"), QTensor::quantize(t, GgmlDType::F32)?)))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let path = out_dir.join("kokoro-tiny-f32.gguf");
    let mut out = std::io::BufWriter::new(std::fs::File::create(&path)?);
    let metadata = metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();
    let quantized = quantized
        .iter()
        .map(|(k, q)| (k.as_str(), q))
        .collect::<Vec<_>>();
    inference_tensor::quantized::gguf_file::write(&mut out, &metadata, &quantized)?;
    Ok(path)
}
