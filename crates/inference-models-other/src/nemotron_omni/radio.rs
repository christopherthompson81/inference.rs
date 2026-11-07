//! C-RADIO, the ViT Nemotron-H Omni sees images with: patches embedded, the CPE position grid cropped to the image,
//! cls and register tokens before them, pre-norm blocks; then InternVL's pixel shuffle and `mlp1` into the decoder.

use std::sync::Arc;

use inference_quant::{QuantMethod, ShardedVarBuilder};
use inference_tensor::nn::{LayerNorm, Module};
use inference_tensor::{D, DType, Device, Result, Tensor};

use super::{Config, ImagePatches, VisionConfig};
use crate::attention::{AttentionMask, FlashParams, SdpaParams};
use crate::layers::{RmsNorm, Sdpa, layer_norm};

// the projector's RMSNorm epsilon (NVIDIA's MLP default)
const PROJECTOR_EPS: f64 = 1e-5;
const CHANNELS: usize = 3;
pub const FINAL_NORM_PREFIX: &str = "vision_projector";
pub const FINAL_NORM: &str = "vision_final_layernorm";

struct Block {
    norm1: LayerNorm,
    qkv: Arc<dyn QuantMethod>,
    proj: Arc<dyn QuantMethod>,
    norm2: LayerNorm,
    fc1: Arc<dyn QuantMethod>,
    fc2: Arc<dyn QuantMethod>,
    heads: usize,
}

impl Block {
    fn new(cfg: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let ffn = cfg.intermediate_size();
        let linear = |i, o, vb| inference_quant::linear_b(i, o, true, &None, vb);
        Ok(Self {
            norm1: layer_norm(h, cfg.layer_norm_eps, vb.pp("norm1"))?,
            qkv: linear(h, 3 * h, vb.pp("attn").pp("qkv"))?,
            proj: linear(h, h, vb.pp("attn").pp("proj"))?,
            norm2: layer_norm(h, cfg.layer_norm_eps, vb.pp("norm2"))?,
            fc1: linear(h, ffn, vb.pp("mlp").pp("fc1"))?,
            fc2: linear(ffn, h, vb.pp("mlp").pp("fc2"))?,
            heads: cfg.num_attention_heads,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, h) = xs.dims3()?;
        let head_dim = h / self.heads;
        let qkv = self
            .qkv
            .forward(&self.norm1.forward(xs)?)?
            .reshape((b, n, 3, self.heads, head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let (q, k, v) = (
            qkv.get(0)?.contiguous()?,
            qkv.get(1)?.contiguous()?,
            qkv.get(2)?.contiguous()?,
        );
        let params = SdpaParams {
            n_kv_groups: 1,
            sliding_window: None,
            softcap: None,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
            sinks: None,
            chunk: None,
        };
        let attn = Sdpa.run_attention(
            &q,
            &k,
            &v,
            &AttentionMask::None,
            Some(&FlashParams::empty(false)),
            &params,
        )?;
        let attn = attn.transpose(1, 2)?.reshape((b, n, h))?;
        let xs = (xs + self.proj.forward(&attn)?)?;
        let mlp = self
            .fc2
            .forward(&self.fc1.forward(&self.norm2.forward(&xs)?)?.gelu_erf()?)?;
        xs + mlp
    }
}

pub struct RadioTower {
    embedder: Arc<dyn QuantMethod>,
    // a tubelet's stacked frames in place of `embedder`, when the checkpoint takes video
    video_embedder: Option<Arc<dyn QuantMethod>>,
    // (1, hidden, rows, cols), f32
    pos_grid: Tensor,
    prefix: Tensor,
    blocks: Vec<Block>,
    // on the features before the pixel shuffle, when the decoder has MTP layers (Super VL)
    final_norm: Option<LayerNorm>,
    norm: RmsNorm,
    linear1: Arc<dyn QuantMethod>,
    linear2: Arc<dyn QuantMethod>,
    merge: usize,
    dtype: DType,
}

impl RadioTower {
    pub fn new(cfg: &Config, vb: ShardedVarBuilder, lm_hidden: usize) -> Result<Self> {
        let v = &cfg.vision_config;
        let h = v.hidden_size;
        let patch_in = CHANNELS * cfg.patch_size * cfg.patch_size;
        let model = vb.pp("vision_model").pp("radio_model").pp("model");
        let patches = model.pp("patch_generator");
        let side = v.max_img_size / cfg.patch_size;
        let pos = patches.get_unchecked_dtype("pos_embed", DType::F32)?;
        let pos_grid = pos
            .reshape((side, side, h))?
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .contiguous()?;
        let prefix = patches
            .pp("cls_token")
            .get((v.num_cls_tokens + v.num_registers, h), "token")?;
        let blocks = (0..v.num_hidden_layers)
            .map(|i| Block::new(v, model.pp("blocks").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let merge = cfg.merge();
        let shuffled = h * merge * merge;
        let mlp = vb.pp("mlp1");
        let video_embedder = v
            .video_layout()
            .map(|video| {
                inference_quant::linear_b(
                    video.temporal * patch_in,
                    h,
                    false,
                    &None,
                    patches.pp("video_embedder"),
                )
            })
            .transpose()?;
        let final_norm = cfg
            .final_layernorm()
            .then(|| layer_norm(h, v.layer_norm_eps, vb.pp(FINAL_NORM_PREFIX).pp(FINAL_NORM)))
            .transpose()?;
        Ok(Self {
            embedder: inference_quant::linear_b(patch_in, h, false, &None, patches.pp("embedder"))?,
            video_embedder,
            pos_grid,
            prefix,
            blocks,
            final_norm,
            norm: RmsNorm::new(shuffled, PROJECTOR_EPS, mlp.pp("0"))?,
            linear1: inference_quant::linear_b(
                shuffled,
                cfg.projector_hidden_size,
                false,
                &None,
                mlp.pp("1"),
            )?,
            linear2: inference_quant::linear_b(
                cfg.projector_hidden_size,
                lm_hidden,
                false,
                &None,
                mlp.pp("3"),
            )?,
            merge,
            dtype: vb.dtype(),
        })
    }

    pub fn device(&self) -> &Device {
        self.prefix.device()
    }

    // the grid resized to a square on the image's longer side, then cropped to it, as RADIO's CPE does
    fn positions(&self, rows: usize, cols: usize) -> Result<Tensor> {
        let (_, h, side, _) = self.pos_grid.dims4()?;
        let longest = rows.max(cols);
        let grid = if longest == side {
            self.pos_grid.clone()
        } else {
            self.pos_grid.upsample_bilinear2d(longest, longest, false)?
        };
        grid.narrow(2, 0, rows)?
            .narrow(3, 0, cols)?
            .reshape((h, rows * cols))?
            .t()?
            .to_dtype(self.dtype)
    }

    /// RADIO's patch features for one image or tubelet, its prefix tokens dropped.
    pub fn features(&self, image: &ImagePatches) -> Result<Tensor> {
        let (rows, cols) = (image.rows, image.cols);
        let embedder = match (&self.video_embedder, image.video) {
            (_, false) => &self.embedder,
            (Some(embedder), true) => embedder,
            (None, true) => inference_tensor::bail!("this vision tower has no `video_embedder`"),
        };
        let patches = image
            .patches
            .to_device(self.device())?
            .to_dtype(self.dtype)?;
        let embedded = (embedder.forward(&patches)? + self.positions(rows, cols)?)?;
        let mut xs =
            Tensor::cat(&[&self.prefix.to_dtype(self.dtype)?, &embedded], 0)?.unsqueeze(0)?;
        for block in &self.blocks {
            xs = block.forward(&xs)?;
        }
        let prefix = self.prefix.dim(0)?;
        xs.squeeze(0)?.narrow(0, prefix, rows * cols)
    }

    /// The decoder embeddings of one image or tubelet: `(rows * cols / merge^2, lm_hidden)`.
    pub fn forward(&self, image: &ImagePatches) -> Result<Tensor> {
        let mut features = self.features(image)?;
        if let Some(norm) = &self.final_norm {
            features = norm.forward(&features)?;
        }
        let shuffled = pixel_shuffle(&features, image.rows, image.cols, self.merge)?;
        let xs = self.norm.forward(&shuffled)?;
        let xs = self.linear1.forward(&xs)?.relu()?.sqr()?;
        self.linear2.forward(&xs)
    }
}

/// InternVL's `pixel_shuffle` (`ps_version` v2) on `(rows * cols, c)` features: `(rows * cols / m^2, c * m^2)`, each
/// output token an `m x m` neighborhood, row offset outermost in its channels.
pub fn pixel_shuffle(features: &Tensor, rows: usize, cols: usize, merge: usize) -> Result<Tensor> {
    let c = features.dim(D::Minus1)?;
    features
        .reshape((rows, cols / merge, c * merge))?
        .transpose(0, 1)?
        .reshape((cols / merge, rows / merge, c * merge * merge))?
        .transpose(0, 1)?
        .reshape(((rows / merge) * (cols / merge), c * merge * merge))
}

#[cfg(test)]
mod tests {
    use super::*;

    // tokens as `100 * row + 10 * col + channel`; the expected rows are NVIDIA's `pixel_shuffle` on the same grid
    #[test]
    fn pixel_shuffle_matches_internvl_v2() -> Result<()> {
        let (rows, cols, c) = (4, 6, 2);
        let values = (0..rows)
            .flat_map(|r| {
                (0..cols).flat_map(move |q| (0..c).map(move |k| (100 * r + 10 * q + k) as f32))
            })
            .collect::<Vec<_>>();
        let features = Tensor::from_vec(values, (rows * cols, c), &Device::Cpu)?;
        let shuffled = pixel_shuffle(&features, rows, cols, 2)?.to_vec2::<f32>()?;
        assert_eq!(shuffled.len(), 6);
        assert_eq!(shuffled[0], [0., 1., 10., 11., 100., 101., 110., 111.]);
        assert_eq!(shuffled[1], [20., 21., 30., 31., 120., 121., 130., 131.]);
        assert_eq!(
            shuffled[3],
            [200., 201., 210., 211., 300., 301., 310., 311.]
        );
        Ok(())
    }
}
