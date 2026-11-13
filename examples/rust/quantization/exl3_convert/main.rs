//! A model converted to EXL3 from a calibration file, each layer encoded against the Hessian of the inputs the layers
//! before it, already encoded, give it; written as exllamav3's converter lays a model out.
//!
//! Run with: `cargo run --release --example exl3_convert -p inference-examples -- <model> <calibration.txt> <out dir>
//! <bits> [3inst|mcg|mul1]`

use anyhow::{Context, Result};
use inference::{Exl3WriteConfig, ModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: exl3_convert <model> <calibration.txt> <out dir> <bits> [3inst|mcg|mul1]";
    let model = args.next().context(usage)?;
    let calibration = args.next().context(usage)?;
    let output = args.next().context(usage)?.into();
    let bits = args.next().context(usage)?.parse()?;
    let codebook = args.next();
    ModelBuilder::new(model)
        .with_calibration_file(calibration.into())
        .with_exl3_out(Exl3WriteConfig {
            output,
            bits,
            head_bits: None,
            codebook,
        })
        .with_logging()
        .build()
        .await?;
    Ok(())
}
