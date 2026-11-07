//! Replays `scripts/nemotron_omni_parity.sh`'s transformers dumps: our image patches against NVIDIA's processor, then
//! C-RADIO's features and the projected embeddings, from the BF16 checkpoint's vision shards or from the text GGUF and
//! its mmproj (so the bindings and the synthesized config are checked too); or `nemotron_omni_video_parity.sh`'s, a
//! video's tubelets and its prompt text.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use inference_models_other::nemotron_omni::{
    Config, ImagePatches,
    inputs_processor::{NemotronOmniInputsProcessor, tubelet_labels, video_replacement},
    radio::RadioTower,
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
    #[arg(long, required_unless_present = "video")]
    dumps: Option<PathBuf>,
    /// A video's dump: `frames/`, `video.json` (fps, sampled indices, the reference's `<video>` text) and
    /// `video.safetensors` (each tubelet's patches, features and embeddings).
    #[arg(long)]
    video: Option<PathBuf>,
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
    let cfg = Config::parse(&std::fs::read_to_string(checkpoint.join("config.json"))?)?;
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
    let floor = args.min_cosine.unwrap_or(if dtype == DType::F32 {
        F32_MIN_COSINE
    } else {
        BF16_MIN_COSINE
    });
    if let Some(dir) = &args.video {
        let worst = video(&cfg, &tower, &processor, dir, &device)?;
        println!("lowest cosine {worst:.7} (passes at {floor})");
        anyhow::ensure!(worst >= floor, "parity failed");
        return Ok(());
    }
    let dumps_dir = args
        .dumps
        .as_ref()
        .expect("clap requires --dumps without --video");
    let mut dumps = std::fs::read_dir(dumps_dir)?
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
        let image = |patches: Tensor| ImagePatches {
            patches,
            rows,
            cols,
            video: false,
        };
        let reference = image(get("patches")?.to_device(&device)?);
        let features = tower.features(&reference)?;
        worst = worst.min(compare("features", &features, get("features")?)?);
        let embeds = tower.forward(&reference)?;
        worst = worst.min(compare("embeds", &embeds, get("embeds")?)?);
        let ours = image(patches.to_device(&device)?);
        let start = std::time::Instant::now();
        let embeds = tower.forward(&ours)?;
        device.synchronize()?;
        let elapsed = start.elapsed();
        worst = worst.min(compare("embeds from our patches", &embeds, get("embeds")?)?);
        println!("  tower: {:.0} ms", elapsed.as_secs_f64() * 1e3);
    }
    println!("lowest cosine {worst:.7} (passes at {floor})");
    anyhow::ensure!(worst >= floor, "parity failed");
    Ok(())
}

// one video: our tubelets' patches, the tower on the reference's and on ours, and the `<video>` text
fn video(
    cfg: &Config,
    tower: &RadioTower,
    processor: &NemotronOmniInputsProcessor,
    dir: &std::path::Path,
    device: &Device,
) -> Result<f64> {
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("video.json"))?)?;
    let mut names = std::fs::read_dir(dir.join("frames"))?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    names.sort();
    let frames = names
        .iter()
        .map(image::open)
        .collect::<image::ImageResult<Vec<_>>>()?;
    let indices = meta["indices"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("video.json has no indices"))?
        .iter()
        .map(|i| i.as_u64().map(|i| i as usize))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| anyhow::anyhow!("video.json indices are not integers"))?;
    let fps = meta["fps"]
        .as_f64()
        .ok_or_else(|| anyhow::anyhow!("video.json has no fps"))?;
    let input =
        inference_nn::media_inputs::video::VideoInput::from_frames(frames, fps, Some(indices));
    let (patches, tubelets, (rows, cols)) = processor.video_patches(&input)?;
    let temporal = cfg
        .vision_config
        .video_layout()
        .ok_or_else(|| anyhow::anyhow!("the checkpoint's tower takes no video"))?
        .temporal;
    println!(
        "{} frames: {tubelets} tubelets of {rows} x {cols} patches",
        input.frames.len()
    );
    let labels = tubelet_labels(&input, temporal);
    let tokens = rows * cols / (cfg.merge() * cfg.merge());
    let text = video_replacement(&labels, tokens);
    let reference_text = meta["replacement"].as_str().unwrap_or_default();
    anyhow::ensure!(
        text == reference_text,
        "the `<video>` text differs:\n ours {:?}\n  ref {:?}",
        &text[..text.len().min(200)],
        &reference_text[..reference_text.len().min(200)]
    );
    println!(
        "  `<video>` text: identical ({} bytes, labels {:?})",
        text.len(),
        labels
    );
    let t = inference_tensor::safetensors::load(dir.join("video.safetensors"), &Device::Cpu)?;
    let get = |name: &str| {
        t.get(name)
            .ok_or_else(|| anyhow::anyhow!("video.safetensors has no {name}"))
    };
    let ours = Tensor::from_vec(patches, (tubelets, rows * cols, ()), &Device::Cpu)?;
    compare("patches", &ours, get("patches")?)?;
    let mut worst = 1f64;
    for tubelet in 0..tubelets {
        let tubelet_patches = |patches: &Tensor| -> Result<ImagePatches> {
            Ok(ImagePatches {
                patches: patches.get(tubelet)?.to_device(device)?,
                rows,
                cols,
                video: true,
            })
        };
        println!(" tubelet {tubelet}");
        let reference = tubelet_patches(get("patches")?)?;
        worst = worst.min(compare(
            "features",
            &tower.features(&reference)?,
            &get("features")?.get(tubelet)?,
        )?);
        worst = worst.min(compare(
            "embeds",
            &tower.forward(&reference)?,
            &get("embeds")?.get(tubelet)?,
        )?);
        let embeds = tower.forward(&tubelet_patches(&ours)?)?;
        worst = worst.min(compare(
            "embeds from our patches",
            &embeds,
            &get("embeds")?.get(tubelet)?,
        )?);
    }
    Ok(worst)
}
