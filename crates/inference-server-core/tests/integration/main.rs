//! The integration tests, one binary so the stack is monomorphized and linked once.

// The tiny random-weight checkpoint the CPU tests load.
#[path = "../../../inference/tests/support/paddleocr_vl_tiny.rs"]
mod support;

mod cancel;
mod chat_route;
mod flux;
mod mcp;
