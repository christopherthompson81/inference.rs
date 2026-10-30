use std::sync::Arc;

use inference_quant::ShardedVarBuilder;
use inference_tensor::{D, Result, Tensor};

use crate::qwen_vl_vision::{
    MlpKind, NormKind, PatchEmbed, PatchMerger, TowerShape, VisionBlock, VisionRotaryEmbedding,
    block_diagonal_mask, item_cu_seqlens, tower_residual_tensors,
};

use super::config::VisionConfig;

pub struct Qwen2VLVisionModel {
    blocks: Vec<VisionBlock>,
    patch_merger: PatchMerger,
    patch_embed: PatchEmbed,
    rotary_pos_emb: VisionRotaryEmbedding,
}

impl Qwen2VLVisionModel {
    pub fn new(
        cfg: &VisionConfig,
        vb: ShardedVarBuilder,
        comm: &Arc<inference_quant::Comm>,
    ) -> Result<Self> {
        let shape = TowerShape {
            in_channels: cfg.in_channels,
            hidden_size: cfg.embed_dim,
            num_heads: cfg.num_heads,
            patch_size: cfg.patch_size,
            temporal_patch_size: cfg.temporal_patch_size,
            spatial_merge_size: cfg.spatial_merge_size,
        };
        let mlp = MlpKind::Plain {
            hidden_dim: (cfg.embed_dim as f64 * cfg.mlp_ratio) as usize,
            act: cfg.hidden_act,
        };
        let blocks = (0..cfg.depth)
            .map(|i| {
                VisionBlock::new(
                    NormKind::Layer,
                    shape,
                    mlp,
                    vb.pp(format!("blocks.{i}")),
                    comm,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            blocks,
            patch_merger: PatchMerger::new(
                NormKind::Layer,
                cfg.hidden_size,
                cfg.embed_dim,
                cfg.spatial_merge_size,
                vb.pp("merger"),
            )?,
            patch_embed: PatchEmbed::new(shape, vb.pp("patch_embed"))?,
            rotary_pos_emb: VisionRotaryEmbedding::new(shape, vb.device())?,
        })
    }

    pub fn forward(&self, xs: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        let mut xs = self
            .patch_embed
            .forward(&xs.to_dtype(self.patch_merger.dtype())?)?;
        let angles = self.rotary_pos_emb.angles(grid_thw, xs.device())?;
        let rotary_pos_emb = Tensor::cat(&[&angles; 2], D::Minus1)?.to_dtype(xs.dtype())?;
        let attention_mask = block_diagonal_mask(
            &item_cu_seqlens(grid_thw)?,
            xs.dim(0)?,
            xs.dtype(),
            xs.device(),
        )?;
        for blk in &self.blocks {
            xs = blk.forward(&xs, attention_mask.as_ref(), &rotary_pos_emb)?;
        }
        self.patch_merger.forward(&xs)
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        tower_residual_tensors(&self.patch_embed, &self.blocks, &self.patch_merger)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use inference_tensor::{DType, Device};

    use super::*;
    use crate::layers::Activation;

    #[test]
    fn residual_tensors_cover_the_complete_vision_checkpoint() -> Result<()> {
        let cfg = VisionConfig {
            depth: 1,
            embed_dim: 4,
            hidden_size: 4,
            hidden_act: Activation::QuickGelu,
            mlp_ratio: 2.0,
            num_heads: 1,
            in_channels: 1,
            patch_size: 1,
            spatial_merge_size: 1,
            temporal_patch_size: 2,
        };
        let shapes = [
            ("patch_embed.proj.weight", vec![4, 1, 2, 1, 1]),
            ("blocks.0.norm1.weight", vec![4]),
            ("blocks.0.norm1.bias", vec![4]),
            ("blocks.0.norm2.weight", vec![4]),
            ("blocks.0.norm2.bias", vec![4]),
            ("blocks.0.mlp.fc1.weight", vec![8, 4]),
            ("blocks.0.mlp.fc2.weight", vec![4, 8]),
            ("blocks.0.attn.qkv.weight", vec![12, 4]),
            ("blocks.0.attn.qkv.bias", vec![12]),
            ("blocks.0.attn.proj.weight", vec![4, 4]),
            ("blocks.0.attn.proj.bias", vec![4]),
            ("merger.ln_q.weight", vec![4]),
            ("merger.ln_q.bias", vec![4]),
            ("merger.mlp.0.weight", vec![4, 4]),
            ("merger.mlp.0.bias", vec![4]),
            ("merger.mlp.2.weight", vec![4, 4]),
            ("merger.mlp.2.bias", vec![4]),
        ];
        let expected = shapes
            .iter()
            .map(|(name, _)| (*name).to_string())
            .collect::<HashSet<_>>();
        let tensors = shapes
            .into_iter()
            .map(|(name, shape)| {
                Ok((
                    name.to_string(),
                    Tensor::zeros(shape, DType::F32, &Device::Cpu)?,
                ))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        let vb = inference_quant::ShardedSafeTensors::wrap(tensors, DType::F32, Device::Cpu);
        let comm = Arc::new(inference_quant::Comm::from_device(
            inference_quant::Id::new(),
            &Device::Cpu,
            0,
            1,
        )?);
        let model = Qwen2VLVisionModel::new(&cfg, vb, &comm)?;
        let actual = model
            .residual_tensors()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<HashSet<_>>();

        assert_eq!(actual, expected);
        Ok(())
    }
}
