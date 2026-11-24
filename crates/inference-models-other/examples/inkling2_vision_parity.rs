//! Writes Inkling 2's image rows for `scripts/inkling2_vision_parity.sh`: the tower from the official checkpoint's
//! `model.visual.*`, a raw RGB picture in, `[cells, 2304]` f32 rows out.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use inference_json::FromJson;
use inference_models_other::inkling2_vision::{VisionConfig, VisionTower};
use inference_quant::ShardedSafeTensors;
use inference_tensor::{DType, Device, Tensor};

const SHARD_EXTENSION: &str = "safetensors";
const TOWER_PREFIX: &str = "model.visual.";

#[derive(Parser)]
struct Args {
    /// The official checkpoint directory.
    #[arg(long)]
    checkpoint: PathBuf,
    /// `height` rows of `width` RGB bytes.
    #[arg(long)]
    rgb: PathBuf,
    #[arg(long)]
    width: usize,
    #[arg(long)]
    height: usize,
    /// Where the rows go, f32 little endian.
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    cpu: bool,
    /// The tower's weights dtype (bf16 or f32).
    #[arg(long, default_value = "bf16")]
    dtype: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    let dtype = match args.dtype.as_str() {
        "f32" => DType::F32,
        _ => DType::BF16,
    };
    let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        args.checkpoint.join("config.json"),
    )?)?;
    let cfg = VisionConfig::from_json(&config["vision_config"])?;
    let mut shards = std::fs::read_dir(&args.checkpoint)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    shards.retain(|p| {
        p.extension().is_some_and(|e| e == SHARD_EXTENSION)
            && p.file_stem().is_some_and(|s| s != "mtp")
    });
    let keep = std::sync::Arc::new(|name: String| name.starts_with(TOWER_PREFIX));
    let vb = unsafe { ShardedSafeTensors::sharded(&shards, dtype, &device, None, keep)? };
    let tower = VisionTower::new(&cfg, vb.pp("model").pp("visual"))?;
    let rgb = std::fs::read(&args.rgb)?;
    anyhow::ensure!(
        rgb.len() == args.width * args.height * 3,
        "not a {}x{} RGB file",
        args.width,
        args.height
    );
    let patches = cfg.patches(&rgb, args.width, args.height);
    let rows = patches.values.len() / cfg.patch_features();
    let input = Tensor::from_vec(patches.values, (rows, cfg.patch_features()), &device)?;
    let start = std::time::Instant::now();
    let out = tower
        .forward(&input, patches.gh, patches.gw)?
        .to_dtype(DType::F32)?;
    let out = out.to_device(&Device::Cpu)?;
    let elapsed = start.elapsed();
    let values = out.flatten_all()?.to_vec1::<f32>()?;
    std::fs::write(
        &args.out,
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    println!(
        "rows {} width {} in {:.1} ms",
        out.dim(0)?,
        out.dim(1)?,
        elapsed.as_secs_f64() * 1e3
    );
    Ok(())
}
