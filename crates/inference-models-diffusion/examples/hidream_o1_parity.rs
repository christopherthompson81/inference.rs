//! Replays `scripts/hidream_o1_parity.sh`'s dumps: per-forward cosine from the reference's inputs, then a pixel diff.
//! With reference images, our preprocessing, prompt tokens and positions are first held to the reference's.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use inference_models_diffusion::hidream::model::PATCH_SIZE;
use inference_models_diffusion::hidream::references::References;
use inference_models_diffusion::hidream::sampling::NUM_TRAIN_TIMESTEPS;
use inference_models_diffusion::hidream::{
    HiDreamO1Pipeline, Noise, UNCONDITIONAL_PROMPT, Variant, patches_to_image,
};
use inference_tensor::{D, DType, Device, Tensor};

// how much further from the higher-precision run than the reference a forward may land
const AGAINST_SLACK: f64 = 1e-3;
// an F32 run of the reference's ViT, compared with when present in the dumps directory
const VISION_F32: &str = "vision_f32.safetensors";
// the reference's flow times are F32 renderings of the same schedule
const FLOW_TIME_TOLERANCE: f32 = 1e-6;

#[derive(Parser)]
struct Args {
    /// Directory with `config.json`, `tokenizer.json` and the `model*.safetensors` shards.
    #[arg(long)]
    model: PathBuf,
    /// Our weights from this GGUF instead, the config and tokenizer still from `--model`.
    #[arg(long)]
    gguf: Option<PathBuf>,
    /// How much further from the F32 run than the reference a forward may land.
    #[arg(long, default_value_t = AGAINST_SLACK)]
    slack: f64,
    /// `reference.safetensors` and `reference.json` from the reference.
    #[arg(long)]
    dumps: PathBuf,
    #[arg(long, default_value_t = 0.999)]
    min_cosine: f64,
    /// `f32.<forward>` x_pred of an F32 run: those forwards must sit at least as close to it as the BF16 reference
    /// does, in place of `--min-cosine`, which still holds every other forward.
    #[arg(long)]
    against: Option<PathBuf>,
    /// Feed the reference's own preprocessed ViT pixels and patches, so the model is checked apart from our resizing.
    #[arg(long)]
    their_inputs: bool,
    /// Feed the reference's own ViT outputs to our denoiser, so it is checked apart from our vision tower.
    #[arg(long)]
    their_vision: bool,
    /// Only the forwards an F32 run covers, and no sampling: a quick look at a precision change.
    #[arg(long)]
    quick: bool,
}

/// The reference's noise: its initial draw, then the draw of each Dev step.
struct ReplayNoise<'a> {
    tensors: &'a std::collections::HashMap<String, Tensor>,
    drawn: Option<usize>,
}

impl Noise for ReplayNoise<'_> {
    fn sample(&mut self, _patches: usize, device: &Device) -> inference_tensor::Result<Tensor> {
        let name = match self.drawn {
            None => "noise.initial".to_string(),
            Some(i) => format!("noise.{i}"),
        };
        self.drawn = Some(self.drawn.map_or(0, |i| i + 1));
        self.tensors
            .get(&name)
            .ok_or_else(|| inference_tensor::Error::msg(format!("the dumps have no {name}")))?
            .unsqueeze(0)?
            .to_device(device)
    }
}

fn cosine(a: &Tensor, b: &Tensor) -> Result<f64> {
    let (a, b) = (
        a.flatten_all()?.to_dtype(DType::F64)?,
        b.flatten_all()?.to_dtype(DType::F64)?,
    );
    let dot = (&a * &b)?.sum_all()?.to_scalar::<f64>()?;
    let norm = |v: &Tensor| -> Result<f64> { Ok(v.sqr()?.sum_all()?.to_scalar::<f64>()?.sqrt()) };
    Ok(dot / (norm(&a)? * norm(&b)?))
}

/// Our reference inputs against the reference's own: the output size, the ViT grids, and how far the pixels differ.
fn check_references(
    ours: &References,
    size: (usize, usize),
    want_size: (usize, usize),
    tensors: &std::collections::HashMap<String, Tensor>,
) -> Result<bool> {
    let get = |name: &str| {
        tensors
            .get(name)
            .with_context(|| format!("the dumps have no {name}"))
    };
    let grids: Vec<Vec<u32>> = get("ref.image_grid_thw")?.to_dtype(DType::U32)?.to_vec2()?;
    let our_grids: Vec<Vec<u32>> = ours.vit_grids.iter().map(|g| g.to_vec()).collect();
    let mut ok = size == want_size && grids == our_grids;
    println!(
        "{} references: output {}x{} (reference {}x{}), ViT grids {our_grids:?} (reference {grids:?})",
        if ok { "ok  " } else { "FAIL" },
        size.0,
        size.1,
        want_size.0,
        want_size.1
    );
    for (name, ours, want) in [
        ("ViT pixels", &ours.vit_pixels, get("ref.pixel_values")?),
        ("patches", &ours.patches.squeeze(0)?, get("ref.patches")?),
    ] {
        // the reference feeds both in BF16, which is how they are dumped
        let ours = ours
            .to_device(&Device::Cpu)?
            .to_dtype(DType::BF16)?
            .to_dtype(DType::F32)?;
        if ours.dims() != want.dims() {
            println!(
                "FAIL {name}: shape {:?}, reference {:?}",
                ours.dims(),
                want.dims()
            );
            ok = false;
            continue;
        }
        let diff = (&ours - want)?.abs()?.flatten_all()?;
        let max = diff.max(0)?.to_scalar::<f32>()?;
        let differing = diff
            .gt(1e-6)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        // one 8-bit level is 2 / 255 after normalizing to [-1, 1]
        println!(
            "     {name}: max |diff| {max:.6} ({:.2} levels), {differing} of {} values differ",
            max * 255.0 / 2.0,
            diff.elem_count()
        );
    }
    Ok(ok)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::cuda_if_available(0)?;
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(args.dumps.join("reference.json"))?)?;
    let tensors = inference_tensor::safetensors::load(
        args.dumps.join("reference.safetensors"),
        &Device::Cpu,
    )?;
    let get = |name: &str| {
        tensors
            .get(name)
            .with_context(|| format!("the dumps have no {name}"))
    };
    let variant = match meta["variant"].as_str() {
        Some("dev") => Variant::Dev,
        _ => Variant::Full,
    };
    let prompt = meta["prompt"].as_str().context("no prompt")?;
    let (width, height) = (
        meta["width"].as_u64().context("no width")? as usize,
        meta["height"].as_u64().context("no height")? as usize,
    );
    let guidance = meta["guidance"].as_bool().unwrap_or(false);
    let calls = meta["calls"].as_array().context("no calls")?;
    let reference_paths: Vec<String> = meta["references"]
        .as_array()
        .map(|refs| {
            refs.iter()
                .filter_map(|r| r.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let keep_aspect = meta["keep_aspect"].as_bool().unwrap_or(false);

    let start = std::time::Instant::now();
    let pipeline = match &args.gguf {
        Some(gguf) => {
            HiDreamO1Pipeline::load_gguf(gguf, &args.model, variant, &device, DType::BF16)?
        }
        None => HiDreamO1Pipeline::load(&args.model, variant, &device, DType::BF16)?,
    };
    println!("loaded in {:.1}s", start.elapsed().as_secs_f64());
    let model = pipeline.model();
    let (rows, cols) = (height / PATCH_SIZE, width / PATCH_SIZE);
    let mut failed = false;
    let references = if reference_paths.is_empty() {
        None
    } else {
        let images = reference_paths
            .iter()
            .map(|p| image::open(p).with_context(|| format!("reading {p}")))
            .collect::<Result<Vec<_>>>()?;
        let size = (!keep_aspect).then_some((width, height));
        let (mut references, size) =
            References::prepare(&images, size, pipeline.model().image_edge(), &device)?;
        failed |= !check_references(&references, size, (width, height), &tensors)?;
        if args.their_inputs {
            references.vit_pixels = get("ref.pixel_values")?.to_device(&device)?;
            references.patches = get("ref.patches")?.unsqueeze(0)?.to_device(&device)?;
        }
        Some(references)
    };
    let (cond, uncond) = pipeline.sequences(prompt, (rows, cols), guidance, references.as_ref())?;
    if references.is_some() {
        for (k, seq) in std::iter::once(&cond).chain(uncond.as_ref()).enumerate() {
            let ours = pipeline.prompt_tokens(
                if k == 0 { prompt } else { UNCONDITIONAL_PROMPT },
                references.as_ref(),
            )?;
            let want: Vec<u32> = get(&format!("input_ids.{k}"))?
                .to_dtype(DType::U32)?
                .to_vec1()?;
            let positions = seq
                .positions()
                .squeeze(1)?
                .to_dtype(DType::U32)?
                .to_vec2::<u32>()?;
            let want_positions = get(&format!("position_ids.{k}"))?
                .to_dtype(DType::U32)?
                .to_vec2::<u32>()?;
            let same = ours == want && positions == want_positions;
            failed |= !same;
            println!(
                "{} sample {k}: {} tokens, {} rows; ids {}, positions {}",
                if same { "ok  " } else { "FAIL" },
                ours.len(),
                positions[0].len(),
                if ours == want { "match" } else { "differ" },
                if positions == want_positions {
                    "match"
                } else {
                    "differ"
                },
            );
        }
    }
    let conditioning = references
        .as_ref()
        .map(|r| model.condition(r))
        .transpose()?;
    if let (Some(conditioning), Some(want)) = (&conditioning, tensors.get("ref.vit")) {
        let (vit, deepstack) = conditioning.vision();
        let vit = vit.to_device(&Device::Cpu)?;
        let mut line = format!("ViT output cos {:.6}", cosine(&vit, want)?);
        let mut row = 0;
        let per_image = references
            .as_ref()
            .map(References::vit_tokens)
            .unwrap_or_default()
            .into_iter()
            .map(|n| {
                let cos = cosine(&vit.narrow(0, row, n)?, &want.narrow(0, row, n)?);
                row += n;
                cos.map(|c| format!("{c:.4}"))
            })
            .collect::<Result<Vec<_>>>()?;
        line += &format!(" (by image {})", per_image.join(" "));
        // an F32 run of the reference's ViT on its own inputs, written beside the dumps by hand
        let truth_path = args.dumps.join(VISION_F32);
        if truth_path.exists() {
            let truth = inference_tensor::safetensors::load(truth_path, &Device::Cpu)?;
            let t = truth.get("vit").context("no vit in the F32 vision dump")?;
            line += &format!(
                "; vs F32: ours {:.6}, reference {:.6}",
                cosine(&vit, t)?,
                cosine(want, t)?
            );
        }
        for (i, layer) in deepstack.iter().enumerate() {
            let want = get(&format!("ref.deepstack.{i}"))?;
            line += &format!(
                ", deepstack {i} {:.6}",
                cosine(&layer.to_device(&Device::Cpu)?, want)?
            );
        }
        println!("     {line}");
    }
    let conditioning = match conditioning {
        Some(c) if args.their_vision => {
            let dtype = model.dtype();
            let load = |name: &str| -> Result<Tensor> {
                Ok(get(name)?.to_device(&device)?.to_dtype(dtype)?)
            };
            let deepstack = (0..c.vision().1.len())
                .map(|i| load(&format!("ref.deepstack.{i}")))
                .collect::<Result<Vec<_>>>()?;
            Some(c.with_vision(load("ref.vit")?, deepstack))
        }
        c => c,
    };
    let steps = match variant {
        Variant::Full => Some(meta["steps"].as_u64().context("no steps")? as usize),
        Variant::Dev => None,
    };
    let settings = variant.settings(steps, None, reference_paths.len())?;

    let against = match &args.against {
        Some(path) => inference_tensor::safetensors::load(path, &Device::Cpu)?,
        None => Default::default(),
    };
    let mut cosines = Vec::with_capacity(calls.len());
    let (mut gated, mut gated_worst) = (f64::INFINITY, 0);
    let per_step = if guidance { 2 } else { 1 };
    for (k, call) in calls.iter().enumerate() {
        if args.quick && !against.contains_key(&format!("f32.{k}")) {
            continue;
        }
        let t = call["timestep"].as_f64().context("no timestep")? as f32;
        let step = k / per_step;
        let want_t = 1.0 - settings.schedule.timesteps[step] / NUM_TRAIN_TIMESTEPS as f32;
        if (t - want_t).abs() > FLOW_TIME_TOLERANCE {
            println!("FAIL step {step}: the reference's flow time {t} is not ours {want_t}");
            failed = true;
        }
        let seq = match &uncond {
            Some(uncond) if k % 2 == 1 => uncond,
            _ => &cond,
        };
        let z = get(&format!("z_in.{k}"))?
            .unsqueeze(0)?
            .to_device(&device)?
            .to_dtype(DType::BF16)?;
        let ours = model
            .forward(
                seq,
                &z,
                &model.timestep_embeddings(&[t])?,
                conditioning.as_ref(),
            )?
            .squeeze(0)?
            .to_device(&Device::Cpu)?;
        let want = get(&format!("x_pred.{k}"))?;
        let cos = cosine(&ours, want)?;
        cosines.push(cos);
        match against.get(&format!("f32.{k}")) {
            Some(truth) => {
                let (ours_cos, reference_cos) = (cosine(&ours, truth)?, cosine(want, truth)?);
                // the reference's own BF16 noise sets the bar: ours must sit at least as close to the truth
                let closer = ours_cos >= reference_cos - args.slack;
                failed |= !closer;
                println!(
                    "{} forward {k}: ours vs f32 {ours_cos:.6}, reference vs f32 {reference_cos:.6}",
                    if closer { "ok  " } else { "FAIL" }
                );
            }
            None if cos < gated => (gated, gated_worst) = (cos, k),
            None => {}
        }
    }
    let min = cosines.iter().copied().fold(f64::INFINITY, f64::min);
    let mean = cosines.iter().sum::<f64>() / cosines.len().max(1) as f64;
    let worst = cosines
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(k, _)| k);
    let ok = gated >= args.min_cosine;
    failed |= !ok;
    println!(
        "{} teacher forced: {} forwards, x_pred cos min {min:.6} (forward {worst}) mean {mean:.6}; without an F32 run, min \
         {gated:.6} (forward {gated_worst})",
        if ok { "ok  " } else { "FAIL" },
        cosines.len()
    );
    if args.quick {
        anyhow::ensure!(!failed, "parity failed");
        return Ok(());
    }

    let mut noise = ReplayNoise {
        tensors: &tensors,
        drawn: None,
    };
    let mut drift = Vec::new();
    let start = std::time::Instant::now();
    let z = pipeline.generate(
        prompt,
        (width, height),
        &settings,
        &mut noise,
        references.as_ref(),
        &mut |i, x_pred| {
            let want = get(&format!("x_pred.{}", i * per_step)).ok();
            let got = x_pred.squeeze(0).and_then(|x| x.to_device(&Device::Cpu));
            if let (Some(want), Ok(got)) = (want, got) {
                drift.push(cosine(&got, want).unwrap_or(f64::NAN));
            }
        },
    )?;
    let elapsed = start.elapsed().as_secs_f64();
    let ours = patches_to_image(&z, width, height)?
        .squeeze(0)?
        .permute((1, 2, 0))?
        .to_device(&Device::Cpu)?;
    let want = get("image")?.to_dtype(DType::F32)?;
    let diff = (&ours - &want)?.abs()?;
    let mean_abs = diff.mean_all()?.to_scalar::<f32>()?;
    let mse = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let psnr = 10.0 * (255.0f32 * 255.0 / mse).log10();
    let max = diff
        .max_keepdim(D::Minus1)?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    println!(
        "sampled {width}x{height} in {elapsed:.1}s: mean |diff| {mean_abs:.3}, max {max}, PSNR {psnr:.2} dB; x_pred cos by step {}",
        drift
            .iter()
            .map(|c| format!("{c:.4}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let pixels: Vec<u8> = ours
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(|v| v as u8)
        .collect();
    let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
    ppm.extend(pixels);
    std::fs::write(args.dumps.join("ours.ppm"), ppm)?;
    println!("ours: {}", args.dumps.join("ours.ppm").display());
    anyhow::ensure!(!failed, "parity failed");
    Ok(())
}
