//! The reference's image views: the global view padded to a square, and above one tile's size the image resized to
//! its best-fitting grid of tiles (`dynamic_preprocess`), all PIL bicubic.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use image::{DynamicImage, RgbImage};
use inference_vision::pil_resize_bicubic;

use super::config::{MIN_TILES, PAD_GRAY};

/// How an image splits: the global view's side, a tile's side and the most tiles.
#[derive(Clone, Copy)]
pub struct Tiling {
    pub base: usize,
    pub tile: usize,
    pub max_tiles: usize,
}

/// One image's views as RGB bytes: the global view, then its tiles row by row.
pub struct ImageViews {
    pub global: RgbImage,
    pub tiles: Vec<RgbImage>,
    /// The tile grid, `(columns, rows)`; `(1, 1)` when the image has no tiles.
    pub grid: (usize, usize),
}

fn resize(image: &RgbImage, width: usize, height: usize) -> RgbImage {
    let (w, h) = image.dimensions();
    let pixels = pil_resize_bicubic(image.as_raw(), (w as usize, h as usize), (width, height));
    RgbImage::from_raw(width as u32, height as u32, pixels).expect("resized pixels fill the image")
}

// Python's `round`: halves to even
fn round_half_even(x: f64) -> usize {
    x.round_ties_even() as usize
}

/// `ImageOps.pad(image, (size, size), color)`: fit inside the square, centered on the gray.
fn pad_square(image: &RgbImage, size: usize) -> RgbImage {
    let (w, h) = (image.width() as usize, image.height() as usize);
    let mut out = RgbImage::from_pixel(size as u32, size as u32, image::Rgb([PAD_GRAY; 3]));
    let (fitted, x, y) = match w.cmp(&h) {
        std::cmp::Ordering::Equal => return resize(image, size, size),
        std::cmp::Ordering::Greater => {
            let height = round_half_even(h as f64 / w as f64 * size as f64);
            let y = round_half_even((size - height) as f64 * 0.5);
            (resize(image, size, height), 0, y)
        }
        std::cmp::Ordering::Less => {
            let width = round_half_even(w as f64 / h as f64 * size as f64);
            let x = round_half_even((size - width) as f64 * 0.5);
            (resize(image, width, size), x, 0)
        }
    };
    image::imageops::replace(&mut out, &fitted, x as i64, y as i64);
    out
}

/// The `(columns, rows)` grids of `MIN_TILES..=max_tiles` tiles, fewest tiles first.
fn target_ratios(max_tiles: usize) -> Vec<(usize, usize)> {
    let mut ratios = Vec::new();
    for cols in 1..=max_tiles {
        for rows in 1..=max_tiles {
            if (MIN_TILES..=max_tiles).contains(&(cols * rows)) {
                ratios.push((cols, rows));
            }
        }
    }
    ratios.sort_by_key(|&(cols, rows)| cols * rows);
    ratios
}

/// `find_closest_aspect_ratio`: the grid nearest the image's aspect, a tie going to the larger grid while the image
/// covers half its area.
fn closest_grid(width: usize, height: usize, tiling: Tiling) -> (usize, usize) {
    let Tiling {
        tile, max_tiles, ..
    } = tiling;
    let aspect = width as f64 / height as f64;
    let area = (width * height) as f64;
    let mut best = (1, 1);
    let mut best_diff = f64::INFINITY;
    for (cols, rows) in target_ratios(max_tiles) {
        let diff = (aspect - cols as f64 / rows as f64).abs();
        if diff < best_diff {
            best_diff = diff;
            best = (cols, rows);
        } else if diff == best_diff && area > 0.5 * (tile * tile * cols * rows) as f64 {
            best = (cols, rows);
        }
    }
    best
}

pub fn image_views(image: &DynamicImage, tiling: Tiling) -> ImageViews {
    let tile = tiling.tile;
    let image = image.to_rgb8();
    let (width, height) = (image.width() as usize, image.height() as usize);
    let global = pad_square(&image, tiling.base);
    if width <= tile && height <= tile {
        return ImageViews {
            global,
            tiles: Vec::new(),
            grid: (1, 1),
        };
    }
    let (cols, rows) = closest_grid(width, height, tiling);
    let resized = resize(&image, tile * cols, tile * rows);
    let tiles = (0..rows)
        .flat_map(|row| (0..cols).map(move |col| (col, row)))
        .map(|(col, row)| {
            image::imageops::crop_imm(
                &resized,
                (col * tile) as u32,
                (row * tile) as u32,
                tile as u32,
                tile as u32,
            )
            .to_image()
        })
        .collect();
    ImageViews {
        global,
        tiles,
        grid: (cols, rows),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OCR: Tiling = Tiling {
        base: 1024,
        tile: 640,
        max_tiles: 9,
    };

    #[test]
    fn grids_follow_the_aspect_and_small_images_keep_one_view() {
        assert_eq!(closest_grid(1240, 1600, OCR), (2, 3));
        // 2x1 ties 4x2, which an image covering half of 4x2 takes
        assert_eq!(closest_grid(2000, 1000, OCR), (4, 2));
        assert_eq!(closest_grid(1500, 750, OCR), (2, 1));
        // a square ties 2x2 with 3x3, which it takes only while covering half of 3x3's area
        assert_eq!(closest_grid(1300, 1300, OCR), (2, 2));
        assert_eq!(closest_grid(1400, 1400, OCR), (3, 3));
        let small = DynamicImage::ImageRgb8(RgbImage::new(600, 400));
        let views = image_views(&small, OCR);
        assert!(views.tiles.is_empty());
        assert_eq!(views.grid, (1, 1));
        let page = DynamicImage::ImageRgb8(RgbImage::new(1240, 1600));
        let views = image_views(&page, OCR);
        assert_eq!(views.tiles.len(), 6);
        assert_eq!(views.global.dimensions(), (1024, 1024));
    }

    #[test]
    fn padding_centers_with_python_rounding() {
        // 1240x1600 fits as 794x1024 (793.6 rounds up), offset round(115) = 115
        let page = RgbImage::from_pixel(1240, 1600, image::Rgb([255, 255, 255]));
        let padded = pad_square(&page, 1024);
        assert_eq!(padded.get_pixel(114, 500).0, [PAD_GRAY; 3]);
        assert_eq!(padded.get_pixel(115, 500).0, [255; 3]);
        assert_eq!(padded.get_pixel(115 + 793, 500).0, [255; 3]);
        assert_eq!(padded.get_pixel(115 + 794, 500).0, [PAD_GRAY; 3]);
    }
}
