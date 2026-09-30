//! Generated images as the OpenAI image response: the engine returns pixels, and the surface encodes them.

use std::{fs::File, io::BufWriter, path::Path};

use anyhow::Result;
use base64::{Engine, engine::general_purpose::STANDARD};
use image::{DynamicImage, codecs::png::PngEncoder};

use crate::{
    request::ImageGenerationResponseFormat,
    response::{ImageChoice, ImageGenerationResponse},
};

const DEFAULT_FILE_PREFIX: &str = "image-generation-";

pub fn encode_png(image: &DynamicImage) -> Result<Vec<u8>> {
    let mut buffer = Vec::new();
    image.write_with_encoder(PngEncoder::new(&mut buffer))?;
    Ok(buffer)
}

/// Url writes each PNG to `save_file` (or a fresh name in the cwd) and returns its path; B64Json inlines it.
pub fn image_generation_response(
    created: u128,
    images: &[DynamicImage],
    format: ImageGenerationResponseFormat,
    save_file: Option<&Path>,
) -> Result<ImageGenerationResponse> {
    let data = images
        .iter()
        .map(|image| image_choice(image, format, save_file))
        .collect::<Result<_>>()?;
    Ok(ImageGenerationResponse { created, data })
}

fn image_choice(
    image: &DynamicImage,
    format: ImageGenerationResponseFormat,
    save_file: Option<&Path>,
) -> Result<ImageChoice> {
    Ok(match format {
        ImageGenerationResponseFormat::Url => {
            let path = match save_file {
                Some(path) => path.to_string_lossy().into_owned(),
                None => format!("{DEFAULT_FILE_PREFIX}{}.png", uuid::Uuid::new_v4()),
            };
            image.write_with_encoder(PngEncoder::new(BufWriter::new(File::create(&path)?)))?;
            ImageChoice {
                url: Some(path),
                b64_json: None,
            }
        }
        ImageGenerationResponseFormat::B64Json => ImageChoice {
            url: None,
            b64_json: Some(STANDARD.encode(encode_png(image)?)),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b64_json_round_trips_the_png() {
        let image = DynamicImage::new_rgb8(3, 2);
        let response =
            image_generation_response(7, &[image], ImageGenerationResponseFormat::B64Json, None)
                .unwrap();
        assert_eq!(response.created, 7);
        let png = STANDARD
            .decode(response.data[0].b64_json.as_ref().unwrap())
            .unwrap();
        let decoded = image::load_from_memory(&png).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (3, 2));
    }

    #[test]
    fn url_writes_the_png_to_the_save_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.png");
        let response = image_generation_response(
            0,
            &[DynamicImage::new_rgb8(4, 4)],
            ImageGenerationResponseFormat::Url,
            Some(&path),
        )
        .unwrap();
        assert_eq!(
            response.data[0].url.as_deref(),
            Some(&*path.to_string_lossy())
        );
        assert_eq!(image::open(&path).unwrap().width(), 4);
    }
}
