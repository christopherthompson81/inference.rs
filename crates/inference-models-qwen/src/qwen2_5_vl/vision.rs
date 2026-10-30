use std::sync::Arc;

use inference_quant::ShardedVarBuilder;
use inference_tensor::{D, DType, Device, Result, Tensor};

use crate::qwen_vl_vision::{
    MlpKind, NormKind, PatchEmbed, PatchMerger, TowerShape, VisionBlock, VisionRotaryEmbedding,
    block_diagonal_mask, item_cu_seqlens, tower_residual_tensors,
};

use super::config::VisionConfig;

pub struct Qwen2_5VLVisionModel {
    blocks: Vec<VisionBlock>,
    patch_merger: PatchMerger,
    patch_embed: PatchEmbed,
    rotary_pos_emb: VisionRotaryEmbedding,
    spatial_merge_size: usize,
    spatial_merge_unit: usize,
    window_size: usize,
    patch_size: usize,
    fullatt_block_indices: Vec<usize>,
}

impl Qwen2_5VLVisionModel {
    pub fn new(
        cfg: &VisionConfig,
        vb: ShardedVarBuilder,
        comm: &Arc<inference_quant::Comm>,
    ) -> Result<Self> {
        let shape = TowerShape {
            in_channels: cfg.in_chans,
            hidden_size: cfg.hidden_size,
            num_heads: cfg.num_heads,
            patch_size: cfg.patch_size,
            temporal_patch_size: cfg.temporal_patch_size,
            spatial_merge_size: cfg.spatial_merge_size,
        };
        let mlp = MlpKind::Gated {
            hidden_dim: cfg.intermediate_size,
            act: cfg.hidden_act,
        };
        let blocks = (0..cfg.depth)
            .map(|i| {
                VisionBlock::new(
                    NormKind::Rms,
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
                NormKind::Rms,
                cfg.out_hidden_size,
                cfg.hidden_size,
                cfg.spatial_merge_size,
                vb.pp("merger"),
            )?,
            patch_embed: PatchEmbed::new(shape, vb.pp("patch_embed"))?,
            rotary_pos_emb: VisionRotaryEmbedding::new(shape, vb.device())?,
            spatial_merge_size: cfg.spatial_merge_size,
            spatial_merge_unit: cfg.spatial_merge_size * cfg.spatial_merge_size,
            window_size: cfg.window_size,
            patch_size: cfg.patch_size,
            fullatt_block_indices: cfg.fullatt_block_indexes.clone(),
        })
    }

    fn get_window_index(&self, grid_thw: &Tensor, device: &Device) -> Result<(Tensor, Vec<i64>)> {
        const PADDING_VALUE: i32 = -100;
        let mut window_index = Vec::new();
        let mut cu_window_seqlens = vec![0];
        let mut window_index_id = 0;
        let vit_merger_window_size = self.window_size / self.spatial_merge_size / self.patch_size;

        for i_thw in grid_thw.to_vec2::<u32>()? {
            let (t, h, w) = (i_thw[0] as usize, i_thw[1] as usize, i_thw[2] as usize);
            let llm_grid_h = h / self.spatial_merge_size;
            let llm_grid_w = w / self.spatial_merge_size;
            let index = Tensor::arange(0i32, (t * llm_grid_h * llm_grid_w) as i32, &Device::Cpu)?
                .reshape((t, llm_grid_h, llm_grid_w))?;
            let pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size;
            let pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size;
            let num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
            let num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;
            let index_padded = {
                let h = Tensor::full(PADDING_VALUE, (t, pad_h, llm_grid_w), &Device::Cpu)?;
                let w = Tensor::full(PADDING_VALUE, (t, pad_h + llm_grid_h, pad_w), &Device::Cpu)?;
                let mut index = Tensor::cat(&[index, h], D::Minus2)?;
                index = Tensor::cat(&[index, w], D::Minus1)?;
                index = index.reshape((
                    t,
                    num_windows_h,
                    vit_merger_window_size,
                    num_windows_w,
                    vit_merger_window_size,
                ))?;
                index = index.permute((0, 1, 3, 2, 4))?.reshape((
                    t,
                    num_windows_h * num_windows_w,
                    vit_merger_window_size,
                    vit_merger_window_size,
                ))?;
                index
            };
            let seqlens = index_padded
                .ne(PADDING_VALUE)?
                .to_dtype(index_padded.dtype())?
                .sum((2, 3))?
                .flatten_all()?;
            let index_new = index_padded
                .flatten_all()?
                .to_vec1::<i32>()?
                .into_iter()
                .filter(|x| *x != PADDING_VALUE)
                .collect::<Vec<_>>();
            window_index.push(Tensor::new(
                index_new
                    .iter()
                    .map(|x| (x + window_index_id) as u32)
                    .collect::<Vec<_>>(),
                device,
            )?);
            let cu_seqlens_tmp = ((seqlens
                .to_dtype(DType::F32)?
                .cumsum(0)?
                .to_dtype(seqlens.dtype())?
                * self.spatial_merge_unit as f64)?
                + cu_window_seqlens[cu_window_seqlens.len() - 1] as f64)?;
            cu_window_seqlens.extend(
                cu_seqlens_tmp
                    .to_vec1::<i32>()?
                    .into_iter()
                    .map(|x| x as i64)
                    .collect::<Vec<_>>(),
            );
            window_index_id += (t * llm_grid_h * llm_grid_w) as i32;
        }

        Ok((Tensor::cat(&window_index, 0)?, cu_window_seqlens))
    }

    pub fn forward(&self, xs: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        let xs = self
            .patch_embed
            .forward(&xs.to_dtype(self.patch_merger.dtype())?)?;
        let rotary_pos_emb = self.rotary_pos_emb.angles(grid_thw, xs.device())?;
        let (window_index, mut cu_window_seqlens) = self.get_window_index(grid_thw, xs.device())?;
        cu_window_seqlens.dedup();

        let seq_len = xs.dims2()?.0;
        let by_window = |xs: &Tensor| -> Result<Tensor> {
            xs.reshape((
                seq_len / self.spatial_merge_unit,
                self.spatial_merge_unit,
                (),
            ))?
            .index_select(&window_index, 0)?
            .reshape((seq_len, ()))
        };
        let mut xs = by_window(&xs)?;
        let rotary_pos_emb = by_window(&rotary_pos_emb)?;
        let rotary_pos_emb = Tensor::cat(&[&rotary_pos_emb; 2], D::Minus1)?.to_dtype(xs.dtype())?;

        let full_mask = block_diagonal_mask(
            &item_cu_seqlens(grid_thw)?,
            seq_len,
            xs.dtype(),
            xs.device(),
        )?;
        let cu_window_seqlens = cu_window_seqlens
            .into_iter()
            .map(|len| len as usize)
            .collect::<Vec<_>>();
        let window_mask =
            block_diagonal_mask(&cu_window_seqlens, seq_len, xs.dtype(), xs.device())?;

        for (i, blk) in self.blocks.iter().enumerate() {
            let attention_mask = if self.fullatt_block_indices.contains(&i) {
                full_mask.as_ref()
            } else {
                window_mask.as_ref()
            };
            xs = blk.forward(&xs, attention_mask, &rotary_pos_emb)?;
        }

        xs = self.patch_merger.forward(&xs)?;
        let reverse_indices = crate::ops::ArgSortOp::arg_sort(&window_index, true)?;
        xs.index_select(&reverse_indices, 0)
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        tower_residual_tensors(&self.patch_embed, &self.blocks, &self.patch_merger)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::*;
    use crate::layers::Activation;

    #[test]
    fn residual_tensors_cover_the_complete_vision_checkpoint() -> Result<()> {
        let cfg = VisionConfig {
            depth: 1,
            hidden_size: 4,
            out_hidden_size: 4,
            hidden_act: Activation::QuickGelu,
            intermediate_size: 8,
            num_heads: 1,
            in_chans: 1,
            patch_size: 1,
            spatial_merge_size: 1,
            temporal_patch_size: 2,
            window_size: 4,
            fullatt_block_indexes: vec![0],
        };
        let shapes = [
            ("patch_embed.proj.weight", vec![4, 1, 2, 1, 1]),
            ("blocks.0.norm1.weight", vec![4]),
            ("blocks.0.norm2.weight", vec![4]),
            ("blocks.0.mlp.gate_proj.weight", vec![8, 4]),
            ("blocks.0.mlp.up_proj.weight", vec![8, 4]),
            ("blocks.0.mlp.down_proj.weight", vec![4, 8]),
            ("blocks.0.attn.qkv.weight", vec![12, 4]),
            ("blocks.0.attn.qkv.bias", vec![12]),
            ("blocks.0.attn.proj.weight", vec![4, 4]),
            ("blocks.0.attn.proj.bias", vec![4]),
            ("merger.ln_q.weight", vec![4]),
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
        let model = Qwen2_5VLVisionModel::new(&cfg, vb, &comm)?;
        let actual = model
            .residual_tensors()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<HashSet<_>>();

        assert_eq!(actual, expected);
        Ok(())
    }
}
