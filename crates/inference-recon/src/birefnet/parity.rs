// Parity against the reference's tensors on one image (`ref.py`, scratch): BIREFNET_DIR is the checkpoint directory,
// BIREFNET_REF the dump, BIREFNET_IMAGE the image it was made from.

use std::path::PathBuf;

use inference_tensor::{Device, Result};

use super::BiRefNet;
use crate::worldmirror::parity::compare;

fn paths() -> Option<(PathBuf, PathBuf, PathBuf)> {
    Some((
        std::env::var_os("BIREFNET_DIR")?.into(),
        std::env::var_os("BIREFNET_REF")?.into(),
        std::env::var_os("BIREFNET_IMAGE")?.into(),
    ))
}

#[test]
#[ignore = "needs the BiRefNet checkpoint, a reference dump and a GPU"]
fn features_and_mask_match_the_reference() -> Result<()> {
    let Some((dir, reference, image)) = paths() else {
        return Ok(());
    };
    let device = Device::new_cuda(0)?;
    let refs = inference_tensor::safetensors::load(&reference, &Device::Cpu)?;
    let model = BiRefNet::load(&dir, &device)?;
    let input = refs["input"].to_device(&device)?;
    let report = |name: &str, got: &inference_tensor::Tensor| -> Result<f32> {
        let (max, rms) = compare(got, &refs[name])?;
        println!("{name:>24}: max {max:.3e} rms {rms:.3e}");
        Ok(rms)
    };
    let bb = model.encoder.forward(&input)?;
    for (i, t) in bb.iter().enumerate() {
        report(&format!("bb.{i}"), t)?;
    }
    let enc = model.encode(&input)?;
    for (i, t) in enc.iter().enumerate() {
        report(&format!("enc.{i}"), t)?;
    }
    report("squeeze_module", &model.squeeze.forward(&enc[3])?)?;
    let start = std::time::Instant::now();
    let logits = model.logits(&input)?;
    let rms = report("logits", &logits)?;
    println!("forward {:.3} s", start.elapsed().as_secs_f64());
    assert!(rms < 1e-2, "logits relative rms {rms}");

    let rgb = crate::read_image(&image)
        .map_err(inference_tensor::Error::wrap)?
        .to_rgb8();
    let size = (rgb.width() as usize, rgb.height() as usize);
    let mask = model.mask(rgb.as_raw(), size)?;
    let want = refs["mask"].flatten_all()?.to_vec1::<u8>()?;
    let diff: Vec<i32> = mask
        .iter()
        .zip(&want)
        .map(|(&a, &b)| i32::from(a) - i32::from(b))
        .collect();
    let off = diff.iter().filter(|&&d| d != 0).count();
    let worst = diff.iter().map(|d| d.abs()).max().unwrap_or(0);
    println!(
        "mask: {off} of {} pixels differ, worst by {worst}",
        diff.len()
    );
    assert!(off * 100 < diff.len(), "{off} mask pixels differ");
    Ok(())
}
