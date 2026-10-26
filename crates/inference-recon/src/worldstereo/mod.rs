//! WorldStereo-2 orchestration: camera and render inputs for the transformer, memory frames, and the scene loop.

pub mod align;
pub mod cameras;
pub mod gs_data;
pub use inference_models_diffusion::wan::gguf_writer;
pub mod memory;
pub mod panorama;
pub mod render;
pub mod run;
pub mod scene;
pub mod trajectory;
