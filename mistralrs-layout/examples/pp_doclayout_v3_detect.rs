//! Run PP-DocLayoutV3 on images and print detections as JSON lines (`{"image": ..., "detections": [...]}`).

use anyhow::Result;
use candle_core::Device;
use clap::Parser;
use mistralrs_layout::pp_doclayout_v3::{PPDocLayoutV3Detector, DEFAULT_THRESHOLD};

#[derive(Parser)]
struct Args {
    /// HF `PP-DocLayoutV3_safetensors` directory.
    #[arg(long)]
    model: String,
    #[arg(long, default_value_t = DEFAULT_THRESHOLD)]
    threshold: f32,
    /// Run all images as a single batch instead of one at a time.
    #[arg(long)]
    batch: bool,
    #[arg(long)]
    cpu: bool,
    images: Vec<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dev = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    let det = PPDocLayoutV3Detector::load(&args.model, &dev)?;
    let images = args
        .images
        .iter()
        .map(|p| {
            Ok(image::ImageReader::open(p)?
                .with_guessed_format()?
                .decode()?
                .to_rgb8())
        })
        .collect::<Result<Vec<_>>>()?;

    let t0 = std::time::Instant::now();
    let results = if args.batch {
        det.detect_batch(&images, args.threshold)?
    } else {
        images
            .iter()
            .map(|im| det.detect(im, args.threshold))
            .collect::<candle_core::Result<Vec<_>>>()?
    };
    eprintln!("{} images in {:?}", images.len(), t0.elapsed());

    for (path, dets) in args.images.iter().zip(results) {
        println!(
            "{}",
            serde_json::json!({ "image": path, "detections": dets })
        );
    }
    Ok(())
}
