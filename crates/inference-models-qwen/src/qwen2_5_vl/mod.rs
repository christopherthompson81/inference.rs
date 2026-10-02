#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use candle_core::{Result, Tensor};
use inference_quant::ShardedVarBuilder;
use vision::Qwen2_5VLVisionModel;

use crate::qwen2vl::{QwenVlModel, QwenVlVision};

pub mod config;
pub mod vision;

pub use config::Config;

pub type Qwen2_5VLModel = QwenVlModel<Qwen2_5VLVisionModel>;

impl QwenVlVision for Qwen2_5VLVisionModel {
    type Config = config::VisionConfig;
    fn new(
        cfg: &Self::Config,
        vb: ShardedVarBuilder,
        comm: &Arc<inference_quant::Comm>,
    ) -> Result<Self> {
        Qwen2_5VLVisionModel::new(cfg, vb, comm)
    }
    fn spatial_merge_size(cfg: &Self::Config) -> usize {
        cfg.spatial_merge_size
    }
    fn forward(&self, xs: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        Qwen2_5VLVisionModel::forward(self, xs, grid_thw)
    }
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        Qwen2_5VLVisionModel::residual_tensors(self)
    }
}
