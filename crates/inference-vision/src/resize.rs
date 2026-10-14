use fast_image_resize::images::{TypedImage, TypedImageRef};
use fast_image_resize::pixels::{U8x3, U8x4};
use fast_image_resize::{FilterType as FirFilter, PixelTrait, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, RgbImage, RgbaImage, imageops::FilterType};

/// `DynamicImage::resize_exact` through a SIMD resizer for RGB8 and RGBA8, many times faster for the convolution filters.
pub fn resize_exact(
    img: &DynamicImage,
    width: u32,
    height: u32,
    filter: FilterType,
) -> DynamicImage {
    let Some(filter) = convolution(filter) else {
        return img.resize_exact(width, height, filter);
    };
    match img {
        DynamicImage::ImageRgb8(src) => DynamicImage::ImageRgb8(
            RgbImage::from_raw(
                width,
                height,
                resize_typed::<U8x3>(src.dimensions(), src.as_raw(), width, height, filter),
            )
            .expect("the buffer holds width * height pixels"),
        ),
        DynamicImage::ImageRgba8(src) => DynamicImage::ImageRgba8(
            RgbaImage::from_raw(
                width,
                height,
                resize_typed::<U8x4>(src.dimensions(), src.as_raw(), width, height, filter),
            )
            .expect("the buffer holds width * height pixels"),
        ),
        _ => img.resize_exact(width, height, filter_back(filter)),
    }
}

/// `resize_exact(..).to_rgb8()`.
pub fn resize_rgb8(img: &DynamicImage, width: u32, height: u32, filter: FilterType) -> RgbImage {
    match img {
        DynamicImage::ImageRgb8(_) => resize_exact(img, width, height, filter).into_rgb8(),
        _ => resize_exact(
            &DynamicImage::ImageRgb8(img.to_rgb8()),
            width,
            height,
            filter,
        )
        .into_rgb8(),
    }
}

// nearest is cheap already, and the two libraries pick different source pixels for it
fn convolution(filter: FilterType) -> Option<FirFilter> {
    match filter {
        FilterType::Nearest => None,
        FilterType::Triangle => Some(FirFilter::Bilinear),
        FilterType::CatmullRom => Some(FirFilter::CatmullRom),
        FilterType::Gaussian => Some(FirFilter::Gaussian),
        FilterType::Lanczos3 => Some(FirFilter::Lanczos3),
    }
}

fn filter_back(filter: FirFilter) -> FilterType {
    match filter {
        FirFilter::Bilinear => FilterType::Triangle,
        FirFilter::Gaussian => FilterType::Gaussian,
        FirFilter::Lanczos3 => FilterType::Lanczos3,
        _ => FilterType::CatmullRom,
    }
}

// typed per pixel type: the untyped entry point compiles a resizer for every one
fn resize_typed<P: PixelTrait>(
    (src_w, src_h): (u32, u32),
    src: &[u8],
    width: u32,
    height: u32,
    filter: FirFilter,
) -> Vec<u8> {
    let src = TypedImageRef::<P>::from_buffer(src_w, src_h, src)
        .expect("the image holds width * height pixels");
    let mut out = vec![0u8; width as usize * height as usize * P::size()];
    let mut dst = TypedImage::<P>::from_buffer(width, height, &mut out)
        .expect("the buffer holds width * height pixels");
    // the image crate filters alpha like any channel, with no premultiplication
    let options = ResizeOptions::new()
        .resize_alg(ResizeAlg::Convolution(filter))
        .use_alpha(false);
    Resizer::new()
        .resize_typed(&src, &mut dst, &options)
        .expect("8-bit images of these sizes always resize");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, Rgba};

    const MAX_MEAN_DIFF: f64 = 0.3;
    // the ringing filters clamp overshoot at a pattern's 255 -> 0 wraps differently
    const MIN_SHARE_WITHIN_2: f64 = 0.98;

    fn pattern(x: u32, y: u32) -> [u8; 4] {
        [
            (x * 7 % 256) as u8,
            (y * 13 % 256) as u8,
            ((x * y) % 256) as u8,
            ((x + y) * 5 % 256) as u8,
        ]
    }

    // the same filters as the image crate's, so pixels agree but for rounding and clamping at hard edges
    #[test]
    fn convolution_filters_match_the_image_crate() {
        let rgb = DynamicImage::ImageRgb8(RgbImage::from_fn(117, 35, |x, y| {
            let [r, g, b, _] = pattern(x, y);
            Rgb([r, g, b])
        }));
        let rgba =
            DynamicImage::ImageRgba8(RgbaImage::from_fn(117, 35, |x, y| Rgba(pattern(x, y))));
        for filter in [
            FilterType::Triangle,
            FilterType::CatmullRom,
            FilterType::Gaussian,
            FilterType::Lanczos3,
        ] {
            for src in [&rgb, &rgba] {
                for (w, h) in [(616, 196), (56, 28), (117, 35)] {
                    let ours = resize_exact(src, w, h, filter);
                    let theirs = src.resize_exact(w, h, filter);
                    assert_eq!(ours.color(), theirs.color());
                    assert_eq!((ours.width(), ours.height()), (w, h));
                    let diffs: Vec<u8> = ours
                        .as_bytes()
                        .iter()
                        .zip(theirs.as_bytes())
                        .map(|(&a, &b)| a.abs_diff(b))
                        .collect();
                    let mean = diffs.iter().map(|&d| d as f64).sum::<f64>() / diffs.len() as f64;
                    let within2 =
                        diffs.iter().filter(|&&d| d <= 2).count() as f64 / diffs.len() as f64;
                    assert!(
                        mean < MAX_MEAN_DIFF && within2 >= MIN_SHARE_WITHIN_2,
                        "{filter:?} {:?} {w}x{h}: mean {mean}, {within2} within 2",
                        src.color()
                    );
                }
            }
        }
    }
}
