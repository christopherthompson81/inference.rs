//! 3D reconstruction and generation: WorldMirror's feed-forward cameras, depth, normals, points and Gaussian splats
//! from images, WorldStereo-2's generation of new views from a panorama, and JoVox's voxel structure from one image.

pub mod birefnet;
mod conv;
mod dinov2;
mod dinov3;
mod geometry;
pub mod jovox;
pub mod moge;
pub mod progress;
pub mod splatting;
pub mod trellis2;
pub mod worldmirror;
pub mod worldstereo;

/// An image file decoded from its bytes: `image::open` would compile every decoder again for a file reader.
pub fn read_image(path: &std::path::Path) -> image::ImageResult<image::DynamicImage> {
    image::load_from_memory(&std::fs::read(path)?)
}
