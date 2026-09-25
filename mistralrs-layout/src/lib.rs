//! Document layout detection models built on candle.

mod layers;
pub mod pp_doclayout_v3;

pub use pp_doclayout_v3::{LayoutDetection, PPDocLayoutV3Detector};
