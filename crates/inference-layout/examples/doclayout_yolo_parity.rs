//! Compare DocLayout-YOLO's input, head features and boxes against a dump of the reference predictor
//! (`scripts/doclayout_yolo_parity.sh` writes it).

use anyhow::Result;
use clap::Parser;
use inference_layout::doclayout_yolo::{DEFAULT_THRESHOLD, DocLayoutYoloDetector, preprocess};
use inference_tensor::{DType, Device, Tensor};

#[derive(Parser)]
struct Args {
    /// The `.pt` checkpoint or its directory.
    #[arg(long)]
    model: String,
    /// Reference dump with `input`, `feat0..2` and `boxes` (`[x1, y1, x2, y2, score, class]` rows).
    #[arg(long)]
    reference: String,
    /// The page the reference read (the dump's PNG).
    #[arg(long)]
    image: String,
    #[arg(long)]
    cpu: bool,
}

fn compare(name: &str, ours: &Tensor, theirs: &Tensor) -> Result<f32> {
    let a = ours
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?;
    let b = theirs.to_dtype(DType::F32)?.flatten_all()?;
    anyhow::ensure!(
        a.dims() == b.dims(),
        "{name}: ours {:?}, reference {:?}",
        ours.dims(),
        theirs.dims()
    );
    let max_abs = (&a - &b)?.abs()?.max(0)?.to_scalar::<f32>()?;
    let scale = b.abs()?.max(0)?.to_scalar::<f32>()?;
    println!("{name:8} max_abs={max_abs:.3e} ref_max={scale:.3e}");
    Ok(max_abs / scale.max(1e-6))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dev = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    let det = DocLayoutYoloDetector::load(&args.model, &dev)?;
    let reference = inference_tensor::safetensors::load(&args.reference, &Device::Cpu)?;
    let image = image::open(&args.image)?.to_rgb8();
    let (input, _) = preprocess::preprocess(&image, det.input_size(), &dev)?;
    compare("input", &input, &reference["input"])?;
    let feats = det.install(|| {
        det.model()
            .features(&reference["input"].unsqueeze(0)?.to_device(&dev)?)
    })?;
    for (i, f) in feats.iter().enumerate() {
        compare(
            &format!("feat{i}"),
            &f.squeeze(0)?,
            &reference[&format!("feat{i}")],
        )?;
    }
    let ours = det.detect(&image, DEFAULT_THRESHOLD)?;
    let theirs = reference["boxes"].to_vec2::<f32>()?;
    println!("boxes: ours {}, reference {}", ours.len(), theirs.len());
    let mut worst = 0f32;
    for r in &theirs {
        let m = ours
            .iter()
            .filter(|d| d.class_id == r[5] as usize)
            .map(|d| {
                let shift = (0..4).map(|k| (d.bbox[k] - r[k]).abs()).fold(0., f32::max);
                (shift, (d.score - r[4]).abs())
            })
            .min_by(|a, b| a.0.total_cmp(&b.0));
        match m {
            Some((shift, score)) => {
                worst = worst.max(shift);
                println!(
                    "  {} {:.4}: box off {shift:.2} px, score off {score:.2e}",
                    det.labels()[r[5] as usize],
                    r[4]
                );
            }
            None => println!("  {} {:.4}: MISSING", det.labels()[r[5] as usize], r[4]),
        }
    }
    println!("worst box offset {worst:.2} px");
    Ok(())
}
