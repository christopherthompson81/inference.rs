//! Hi3D v1.0 (Sparc3D): a coarse mesh refined under an image. The mesh is voxelized, latents on its cells are sampled
//! by a sparse flow transformer conditioned on DINOv2 and CLIP, decoded by a sparse VAE to signed distances and
//! deformations at 8x the resolution, and meshed by FlexiCubes; up to three passes at 512, 1024 and 1536.
//!
//! Adapted from hi3d-official/Hi3Dv1.0 (`sparc3d/`), Copyright OCEANO HK HOLDING LIMITED, under the Hi3D Open License
//! Agreement v1.0 (LICENSE and NOTICE beside this file, the Agreement's Acceptable Use Policy included); modified:
//! reimplemented in Rust. Not an official Hi3D product.

pub mod conditioner;
pub mod extract;
mod flexicubes;
mod flexicubes_tables;
pub mod pipeline;
pub mod schedule;
pub mod transformer;
pub mod vae;
pub mod voxelize;

#[cfg(all(test, feature = "cuda"))]
mod parity;

/// The latents' per-channel statistics, the same in both transformers' configs.
pub const LATENT_MEAN: [f32; 32] = [
    -0.00155994,
    0.03314649,
    -0.00307796,
    0.00630751,
    -0.00835457,
    -0.03860179,
    0.02528263,
    0.0374062,
    -0.06209833,
    -0.00539356,
    -0.02110271,
    0.05126001,
    0.00753371,
    0.02866821,
    0.02772475,
    -0.02131735,
    -0.02799912,
    0.01084617,
    0.03433723,
    -0.009031,
    -0.00121899,
    -0.01944244,
    0.01641873,
    -0.01063657,
    -0.01990618,
    0.04856062,
    0.01747232,
    -0.02387775,
    0.03216169,
    -0.03732105,
    -0.03730354,
    -0.01948937,
];
pub const LATENT_STD: [f32; 32] = [
    0.61686146, 0.8011532, 0.854892, 0.96384716, 0.7700591, 0.6285985, 0.61541015, 0.7436831,
    0.6276261, 1.0184228, 0.77113867, 0.6262799, 0.68749195, 0.62698966, 0.7549124, 0.615774,
    0.6378066, 0.80406, 0.83846325, 0.6912039, 0.6922758, 0.79321617, 0.6347593, 0.638483,
    0.67194235, 0.7351938, 0.84781533, 0.9301291, 0.6269144, 0.6790608, 0.6519877, 0.7769204,
];
