//! External speculative drafters, separate models that draft for a target from its tapped hidden states: DFlash,
//! DFlash 2 and DSpark. Target-agnostic; a target model supplies the taps, its embeddings and its lm_head.
#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub mod dflash;
