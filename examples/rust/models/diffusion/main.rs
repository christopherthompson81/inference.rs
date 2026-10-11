//! Image generation using a diffusion model.
//!
//! Run with: `cargo run --release --example diffusion -p inference-examples`

use std::time::Instant;

use anyhow::Result;
use inference::{
    DiffusionLoaderType, DiffusionModelBuilder, ImageGenerationRequest,
    ImageGenerationResponseFormat,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = DiffusionModelBuilder::new(
        "black-forest-labs/FLUX.1-schnell",
        DiffusionLoaderType::FluxOffloaded,
    )
    .with_logging()
    .build()
    .await?;

    let start = Instant::now();

    let mut request =
        ImageGenerationRequest::new("A vibrant sunset in the mountains, 4k, high quality.");
    request.response_format = ImageGenerationResponseFormat::Url;
    let response = model.generate_image(request).await?;

    let finished = Instant::now();

    // the url serves the PNG from the engine's file store; its id is the second-to-last segment
    let url = response.data[0].url.as_deref().unwrap();
    let file_id = url.rsplit('/').nth(1).unwrap();
    std::fs::write("image.png", model.file_content(file_id)?.bytes)?;
    println!(
        "Done! Took {} s. Image saved at: image.png",
        finished.duration_since(start).as_secs_f32()
    );

    Ok(())
}
