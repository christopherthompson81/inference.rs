//! Compare PP-DocLayoutV3 intermediates against a safetensors dump produced by the HF reference.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use clap::Parser;
use mistralrs_layout::pp_doclayout_v3::{
    postprocess, PPDocLayoutV3Detector, PostprocessArgs, DEFAULT_THRESHOLD,
};

#[derive(Parser)]
struct Args {
    /// HF `PP-DocLayoutV3_safetensors` directory.
    #[arg(long)]
    model: String,
    /// Reference dump with `pixel_values`, `backbone.*`, `encoder.*`, `logits`, ... tensors.
    #[arg(long)]
    reference: String,
    /// Source image, to also check preprocessing.
    #[arg(long)]
    image: Option<String>,
    #[arg(long)]
    cpu: bool,
}

fn compare(name: &str, ours: &Tensor, theirs: &Tensor) -> Result<()> {
    let a = ours
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?;
    let b = theirs.to_dtype(DType::F32)?.flatten_all()?;
    if a.dims() != b.dims() {
        println!(
            "{name:32} SHAPE MISMATCH ours={:?} ref={:?}",
            ours.dims(),
            theirs.dims()
        );
        return Ok(());
    }
    let diff = (&a - &b)?.abs()?;
    let max_abs = diff.max(0)?.to_scalar::<f32>()?;
    let mean_abs = diff.mean_all()?.to_scalar::<f32>()?;
    let ref_scale = b.abs()?.mean_all()?.to_scalar::<f32>()?;
    let dot = (&a * &b)?.sum_all()?.to_scalar::<f32>()?;
    let na = a.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let nb = b.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    println!(
        "{name:32} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} ref_mean_abs={ref_scale:.3e} cos={:.7}",
        dot / (na * nb)
    );
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dev = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    let det = PPDocLayoutV3Detector::load(&args.model, &dev)?;
    let r = candle_core::safetensors::load(&args.reference, &Device::Cpu)?;

    if let Some(img) = &args.image {
        let im = image::open(img)?.to_rgb8();
        let pv = det
            .preprocessor()
            .preprocess(&im, &Device::Cpu)?
            .unsqueeze(0)?;
        compare("pixel_values", &pv, &r["pixel_values"])?;
        let n_diff = (pv - &r["pixel_values"])?
            .abs()?
            .gt(1e-6)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        println!(
            "{:32} {n_diff} of {} values differ",
            "pixel_values",
            3 * 800 * 800
        );
    }

    let pv = r["pixel_values"].to_device(&dev)?;
    let t0 = std::time::Instant::now();
    let (out, inter) = det.model().forward_with_intermediates(&pv)?;
    println!("forward (with intermediates) {:?}", t0.elapsed());

    for (i, t) in inter.backbone.iter().enumerate() {
        compare(&format!("backbone.{i}"), t, &r[&format!("backbone.{i}")])?;
    }
    for (i, t) in inter.pan.iter().enumerate() {
        compare(
            &format!("encoder.pan.{i}"),
            t,
            &r[&format!("encoder.pan.{i}")],
        )?;
    }
    compare(
        "encoder.mask_feat",
        &inter.mask_feat,
        &r["encoder.mask_feat"],
    )?;
    compare("enc_score", &inter.enc_score, &r["enc_score"])?;
    compare(
        "init_reference_points",
        &inter.init_ref,
        &r["init_reference_points"],
    )?;
    let ref_hidden = &r["intermediate_hidden_states"];
    for (i, t) in inter.hidden.iter().enumerate() {
        compare(
            &format!("decoder.hidden.{i}"),
            t,
            &ref_hidden.narrow(1, i, 1)?.squeeze(1)?,
        )?;
    }
    compare("logits", &out.logits, &r["logits"])?;
    compare("pred_boxes", &out.pred_boxes, &r["pred_boxes"])?;
    compare("order_logits", &out.order_logits, &r["order_logits"])?;
    if let Some(masks) = &out.masks {
        compare("out_masks", masks, &r["out_masks"])?;
    }

    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        let o = det.model().forward(&pv, false)?;
        o.logits.to_device(&Device::Cpu)?;
        println!("forward {:?}", t0.elapsed());
    }

    if let Some(img) = &args.image {
        let (w, h) = image::image_dimensions(img)?;
        let dets = postprocess::postprocess(
            &out.logits.get(0)?,
            &out.pred_boxes.get(0)?,
            &out.order_logits.get(0)?,
            &PostprocessArgs {
                threshold: DEFAULT_THRESHOLD,
                orig_size: (w, h),
            },
        )?;
        for d in dets {
            println!(
                "{:2} {:18} {:.4} [{:.1}, {:.1}, {:.1}, {:.1}]",
                d.reading_order, d.label, d.score, d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]
            );
        }
    }
    Ok(())
}
