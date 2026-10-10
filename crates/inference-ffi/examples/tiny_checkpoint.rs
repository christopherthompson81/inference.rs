//! Writes the tiny random-weight PaddleOCR-VL checkpoint the ABI tests use, and a tiny Parakeet under `parakeet/`,
//! for binding tests in other languages.
//! Usage: cargo run -p inference-ffi --example tiny_checkpoint -- <directory>

use std::path::Path;

#[path = "../../inference/tests/support/paddleocr_vl_tiny.rs"]
mod support;

#[path = "../../inference/tests/support/parakeet_tiny.rs"]
// each support file includes the recorder it needs, so both bring a copy
#[allow(clippy::duplicate_mod)]
mod parakeet_support;

const PARAKEET_DIR: &str = "parakeet";

fn copy_files(from: &Path, to: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let path = entry?.path();
        std::fs::copy(&path, to.join(path.file_name().unwrap()))?;
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let target = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: tiny_checkpoint <directory>"))?;
    let checkpoint = support::tiny_checkpoint()?;
    copy_files(checkpoint.path(), Path::new(&target))?;
    let parakeet = parakeet_support::tiny_parakeet_checkpoint(parakeet_support::HEADS[0])?;
    copy_files(parakeet.path(), &Path::new(&target).join(PARAKEET_DIR))?;
    println!("{target}");
    Ok(())
}
