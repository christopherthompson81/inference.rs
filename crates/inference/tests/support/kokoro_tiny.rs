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
// invented voice names; the first sorts first, so it is the default when the reference's default is absent
pub const VOICES: [&str; 2] = ["tiny_one", "tiny_two"];
const VOICE_ROWS: usize = 510;
const VOICE_DIM: usize = 256;
const VOICE_STD: f32 = 0.3;
const VOICE_SEED: u64 = 0x0C0C_0A1D;

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
