use fast_image_resize::images::{TypedImage, TypedImageRef};
use fast_image_resize::pixels::U8x3;
use fast_image_resize::{FilterType as FirFilter, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, RgbImage, imageops::FilterType};

/// `DynamicImage::resize_exact(..).to_rgb8()` through a SIMD resizer, many times faster for the convolution filters.
pub fn resize_rgb8(img: &DynamicImage, width: u32, height: u32, filter: FilterType) -> RgbImage {
    let alg = match filter {
        FilterType::Nearest => ResizeAlg::Nearest,
        FilterType::Triangle => ResizeAlg::Convolution(FirFilter::Bilinear),
        FilterType::CatmullRom => ResizeAlg::Convolution(FirFilter::CatmullRom),
        FilterType::Gaussian => ResizeAlg::Convolution(FirFilter::Gaussian),
        FilterType::Lanczos3 => ResizeAlg::Convolution(FirFilter::Lanczos3),
    };
    // RGB8 only: the untyped entry point compiles a resizer for every pixel type
    let src = img.to_rgb8();
    let src = TypedImageRef::<U8x3>::from_buffer(src.width(), src.height(), src.as_raw())
        .expect("an RgbImage holds width * height RGB pixels");
    let mut out = vec![0u8; width as usize * height as usize * 3];
    let mut dst = TypedImage::<U8x3>::from_buffer(width, height, &mut out)
        .expect("the buffer holds width * height RGB pixels");
    Resizer::new()
        .resize_typed(&src, &mut dst, &ResizeOptions::new().resize_alg(alg))
        .expect("RGB8 images of these sizes always resize");
    RgbImage::from_raw(width, height, out).expect("the buffer holds width * height RGB pixels")
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    const MAX_MEAN_DIFF: f64 = 0.2;
    const MAX_PIXEL_DIFF: u8 = 16;

    // the same filter as the image crate's, so pixels agree but for rounding at hard edges
    #[test]
    fn catmull_rom_matches_the_image_crate() {
        let src = DynamicImage::ImageRgb8(RgbImage::from_fn(117, 35, |x, y| {
            Rgb([
                (x * 7 % 256) as u8,
                (y * 13 % 256) as u8,
                ((x * y) % 256) as u8,
            ])
        }));
        for (w, h) in [(616, 196), (56, 28), (117, 35)] {
            let ours = resize_rgb8(&src, w, h, FilterType::CatmullRom);
            let theirs = src.resize_exact(w, h, FilterType::CatmullRom).to_rgb8();
            assert_eq!(ours.dimensions(), (w, h));
            let diffs: Vec<u8> = ours
                .as_raw()
                .iter()
                .zip(theirs.as_raw())
                .map(|(&a, &b)| a.abs_diff(b))
                .collect();
            let mean = diffs.iter().map(|&d| d as f64).sum::<f64>() / diffs.len() as f64;
            assert!(
                mean < MAX_MEAN_DIFF,
                "{w}x{h}: mean pixel difference {mean}"
            );
            assert!(
                diffs.iter().all(|&d| d <= MAX_PIXEL_DIFF),
                "{w}x{h}: a pixel differs by more than {MAX_PIXEL_DIFF}"
            );
        }
    }
}
