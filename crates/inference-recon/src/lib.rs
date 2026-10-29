//! 3D reconstruction and world generation: WorldMirror's feed-forward cameras, depth, normals, points and Gaussian
//! splats from images, and WorldStereo-2's generation of new views from a panorama.

mod conv;
mod dinov2;
mod geometry;
pub mod moge;
pub mod progress;
pub mod splatting;
pub mod worldmirror;
pub mod worldstereo;

/// An image file decoded from its bytes: `image::open` would compile every decoder again for a file reader.
pub fn read_image(path: &std::path::Path) -> image::ImageResult<image::DynamicImage> {
    image::load_from_memory(&std::fs::read(path)?)
}
