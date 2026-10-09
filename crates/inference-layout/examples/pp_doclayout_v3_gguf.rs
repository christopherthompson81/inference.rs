//! Convert a PP-DocLayoutV3 HF directory to GGUF; `PPDocLayoutV3Detector::load` and the C ABI take the result.

use anyhow::{Result, bail};
use clap::Parser;
use inference_layout::pp_doclayout_v3::gguf::write_gguf;
use inference_tensor::quantized::GgmlDType;

#[derive(Parser)]
struct Args {
    /// HF `PP-DocLayoutV3_safetensors` directory.
    #[arg(long)]
    model: String,
    /// f32, f16 or q8_0 (bf16 and the 4/5-bit types move detections); vectors stay f32 either way.
    #[arg(long, default_value = "f16")]
    dtype: String,
    out: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dtype = match args.dtype.as_str() {
        "f32" => GgmlDType::F32,
        "f16" => GgmlDType::F16,
        "q8_0" => GgmlDType::Q8_0,
        other => bail!("unsupported dtype {other}"),
    };
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args.out)?);
    write_gguf(args.model.as_ref(), &mut out, dtype)?;
    Ok(())
}
