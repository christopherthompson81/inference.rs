//! The integration tests, one binary so the stack is monomorphized and linked once.
// Each test file includes the tiny-checkpoint helpers itself, as the SDK's tests do.
#![allow(clippy::duplicate_mod)]

mod jobs;
mod world_job;
mod worldmirror_job;
