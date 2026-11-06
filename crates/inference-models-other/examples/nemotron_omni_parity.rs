//! Replays `scripts/nemotron_omni_parity.sh`'s transformers dumps: our image patches against NVIDIA's processor, then
//! C-RADIO's features and the projected embeddings, from the BF16 checkpoint's vision shards or from the text GGUF and
//! its mmproj (so the bindings and the synthesized config are checked too).

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use inference_models_other::nemotron_omni::{
    Config, inputs_processor::NemotronOmniInputsProcessor, radio::RadioTower,
};
use inference_quant::{GgufArchive, GgufWeightSource, ShardedSafeTensors, ShardedVarBuilder};
use inference_tensor::{DType, Device, Tensor};

const SHARD_EXTENSION: &str = "safetensors";

#[derive(Parser)]
struct Args {
    /// The checkpoint directory: `config.json` and the shards that hold `vision_model.*` and `mlp1.*`.
    #[arg(long, required_unless_present = "gguf")]
    checkpoint: Option<PathBuf>,
    /// The text GGUF and its mmproj, in place of the checkpoint.
    #[arg(long, num_args = 2)]
    gguf: Vec<PathBuf>,
    /// `<name>.png` images and their `<name>.safetensors` dumps from the reference.
    #[arg(long)]
    dumps: PathBuf,
    #[arg(long)]
    cpu: bool,
    /// The tower's dtype (f32 or bf16).
    #[arg(long, default_value = "f32")]
    dtype: String,
    /// The lowest embeddings cosine that passes; BF16 attention over 13 K patches reaches about 0.9992.
    #[arg(long)]
    min_cosine: Option<f64>,
}

const F32_MIN_COSINE: f64 = 0.99999;
const BF16_MIN_COSINE: f64 = 0.999;

// in f64: an f32 cosine over millions of values drifts past 1
fn compare(name: &str, ours: &Tensor, reference: &Tensor) -> Result<f64> {
    let a = ours
        .to_dtype(DType::F64)?
        .to_device(&Device::Cpu)?
        .flatten_all()?;
    let b = reference.to_dtype(DType::F64)?.flatten_all()?;
    anyhow::ensure!(
        a.dims() == b.dims(),
        "{name}: {:?} != {:?}",
        ours.dims(),
        reference.dims()
    );
    let max = (&a - &b)?.abs()?.max(0)?.to_scalar::<f64>()?;
    let dot = (&a * &b)?.sum_all()?.to_scalar::<f64>()?;
    let norms = a.sqr()?.sum_all()?.sqrt()?.to_scalar::<f64>()?
        * b.sqr()?.sum_all()?.sqrt()?.to_scalar::<f64>()?;
    let scale = b.abs()?.max(0)?.to_scalar::<f64>()?;
    println!(
        "  {name}: max diff {max:.3e} (of {scale:.3e}), cosine {:.7}",
        dot / norms
    );
    Ok(dot / norms)
}

fn checkpoint_tower(
    checkpoint: &std::path::Path,
    dtype: DType,
    device: &Device,
) -> Result<(Config, ShardedVarBuilder)> {
    let cfg = Config::from_json(&std::fs::read_to_string(checkpoint.join("config.json"))?)?;
    let mut shards = std::fs::read_dir(checkpoint)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    shards.retain(|p| p.extension().is_some_and(|e| e == SHARD_EXTENSION));
    let vb = unsafe {
        ShardedSafeTensors::sharded(&shards, dtype, device, None, std::sync::Arc::new(|_| true))?
    };
    Ok((cfg, vb))
}

fn gguf_tower(
    paths: &[PathBuf],
    dtype: DType,
    device: &Device,
) -> Result<(Config, ShardedVarBuilder)> {
    let mut archive = GgufArchive::open(&paths[..1])?;
    archive.merge_components(GgufArchive::open_components(&paths[1..])?)?;
    let resolved =
        inference_gguf::multimodal_vision_registry::resolve_native_multimodal_gguf(&archive)?
            .ok_or_else(|| anyhow::anyhow!("not a native multimodal GGUF pair"))?;
    let config = resolved
        .config
        .ok_or_else(|| anyhow::anyhow!("the GGUFs describe no config"))?;
    let cfg = Config::from_json(&config)?;
    let source = GgufWeightSource::new(std::sync::Arc::new(archive), &resolved.bindings, dtype)?;
    Ok((
        cfg,
        std::sync::Arc::new(source).sharded_var_builder(device.clone()),
    ))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    let dtype = match args.dtype.as_str() {
        "bf16" => DType::BF16,
        _ => DType::F32,
    };
    let (cfg, vb) = match &args.checkpoint {
        Some(checkpoint) => checkpoint_tower(checkpoint, dtype, &device)?,
        None => gguf_tower(&args.gguf, dtype, &device)?,
    };
    let tower = RadioTower::new(&cfg, vb, cfg.llm_config.hidden_size)?;
    let processor = NemotronOmniInputsProcessor::new(&cfg, None)?;
    let mut dumps = std::fs::read_dir(&args.dumps)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    dumps.retain(|p| p.extension().is_some_and(|e| e == SHARD_EXTENSION));
    dumps.sort();
    let mut worst = 1f64;
    for path in &dumps {
        let t = inference_tensor::safetensors::load(path, &Device::Cpu)?;
        let get = |name: &str| {
            t.get(name)
                .ok_or_else(|| anyhow::anyhow!("{} has no {name}", path.display()))
        };
        let image = image::open(path.with_extension("png"))?;
        // Bolt's budget layout gives each image one tile
        let (patches, (rows, cols)) = processor
            .patches(&image)
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("no tile for {}", path.display()))?;
        println!("{}: {rows} x {cols} patches", path.display());
        let patches = Tensor::from_vec(patches, (rows * cols, ()), &Device::Cpu)?;
        compare("patches", &patches, get("patches")?)?;
        // the tower on the reference's own patches, so a resize difference does not hide the tower's
        let reference_patches = get("patches")?.to_device(&device)?;
        let features = tower.features(&reference_patches, rows, cols)?;
        worst = worst.min(compare("features", &features, get("features")?)?);
        let embeds = tower.forward(&reference_patches, rows, cols)?;
        worst = worst.min(compare("embeds", &embeds, get("embeds")?)?);
        let patches = patches.to_device(&device)?;
        let start = std::time::Instant::now();
        let embeds = tower.forward(&patches, rows, cols)?;
        device.synchronize()?;
        let elapsed = start.elapsed();
        worst = worst.min(compare("embeds from our patches", &embeds, get("embeds")?)?);
        println!("  tower: {:.0} ms", elapsed.as_secs_f64() * 1e3);
    }
    let floor = args.min_cosine.unwrap_or(if dtype == DType::F32 {
        F32_MIN_COSINE
    } else {
        BF16_MIN_COSINE
    });
    println!("lowest cosine {worst:.7} (passes at {floor})");
    anyhow::ensure!(worst >= floor, "parity failed");
    Ok(())
}
