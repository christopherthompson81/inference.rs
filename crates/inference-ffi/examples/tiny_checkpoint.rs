//! Writes the tiny random-weight PaddleOCR-VL checkpoint the ABI tests use, for binding tests in other languages.
//! Usage: cargo run -p inference-ffi --example tiny_checkpoint -- <directory>

#[path = "../../inference/tests/support/paddleocr_vl_tiny.rs"]
mod support;

fn main() -> anyhow::Result<()> {
    let target = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: tiny_checkpoint <directory>"))?;
    let checkpoint = support::tiny_checkpoint()?;
    std::fs::create_dir_all(&target)?;
    for entry in std::fs::read_dir(checkpoint.path())? {
        let path = entry?.path();
        std::fs::copy(
            &path,
            std::path::Path::new(&target).join(path.file_name().unwrap()),
        )?;
    }
    println!("{target}");
    Ok(())
}
