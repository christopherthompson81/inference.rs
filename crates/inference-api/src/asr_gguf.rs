//! Writing a speech recogniser's `.nemo` as an `asr` GGUF under a per-tensor precision policy.

pub use inference_core::{
    ASR_QUANT_PRESETS as PRESETS, ASR_QUANT_TYPES as TYPE_NAMES, AsrGgufSummary,
    AsrQuantPolicy as QuantPolicy, write_asr_gguf,
};
