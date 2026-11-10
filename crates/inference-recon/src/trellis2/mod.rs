//! TRELLIS.2's second stage: the shape flow that fills a voxel structure with latents, and their decoding into a mesh.

pub mod decoder;
pub mod dit;
pub mod mesh;
#[cfg(test)]
mod parity;
pub mod shape_flow;
pub mod sparse;
