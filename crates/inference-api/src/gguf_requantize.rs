//! A float GGUF rewritten with per-tensor types and an imatrix, a slice at a time.

pub use inference_quant::requantize::{
    RequantizePlan, RequantizeSummary, load_imatrix, requantize_gguf, writable_type,
};
