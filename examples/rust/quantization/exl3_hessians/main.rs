//! Each layer's input Hessian from a calibration file, written as exllamav3's converter takes them (`--hessians`).
//!
//! Run with: `cargo run --release --example exl3_hessians -p inference-examples -- <model> <calibration.txt> <dir>`

use anyhow::{Context, Result};
use inference::ModelBuilder;

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: exl3_hessians <model> <calibration.txt> <out dir>";
    let model = args.next().context(usage)?;
    let calibration = args.next().context(usage)?;
    let out = args.next().context(usage)?;
    ModelBuilder::new(model)
        .with_calibration_file(calibration.into())
        .with_hessians_out(out.into())
        .with_logging()
        .build()
        .await?;
    Ok(())
}
