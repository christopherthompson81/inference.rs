//! The integration tests, one binary so the stack is monomorphized and linked once.

// The tiny random-weight checkpoint the CPU tests load.
#[path = "../../../inference/tests/support/paddleocr_vl_tiny.rs"]
mod support;

// The tiny random-weight Parakeet the transcription route loads.
#[path = "../../../inference/tests/support/parakeet_tiny.rs"]
// each support file includes the recorder it needs, so both bring a copy
#[allow(clippy::duplicate_mod)]
mod parakeet_support;

// The tiny random-weight Silero VAD the voice activity route loads.
#[path = "../../../inference/tests/support/silero_tiny.rs"]
#[allow(clippy::duplicate_mod)]
mod silero_support;

// The tiny random-weight Nemotron-3 Diarization the diarization route loads.
#[path = "../../../inference/tests/support/nemotron3_diarization_tiny.rs"]
#[allow(clippy::duplicate_mod)]
mod diarization_support;

mod cancel;
mod chat_route;
mod flux;
mod keyed;
mod mcp;
mod transcription_route;
