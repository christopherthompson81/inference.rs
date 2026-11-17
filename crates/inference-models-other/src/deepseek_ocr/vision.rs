//! DeepSeek-OCR's DeepEncoder: SAM ViT-B, a CLIP-L that reads SAM's features in place of patch embeddings, and a
//! linear projector over both, laid out as the reference's `DeepseekOCRModel.forward` does.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use inference_nn::vision::clip::{Activation, ClipConfig, ClipEncoder};
use inference_nn::vision::sam::{SamEncoder, resample_grid};
use inference_quant::{QuantMethod, ShardedVarBuilder};
use inference_tensor::nn::{LayerNorm, Module};
use inference_tensor::{D, Result, Tensor};

use crate::attention::AttentionMask;
use crate::layers::layer_norm;
use crate::utils::unvarbuilder::UnVarBuilder;

use super::config::{Config, TILE_SIZE};

const CLIP_EPS: f64 = 1e-5;
const CLIP_MLP_RATIO: usize = 4;

/// CLIP-L/14 over SAM's (B, C, G, G) map: a class token, the position table resampled to G, `pre_layrnorm`, the layers.
struct ClipOverSam {
    class_embedding: Tensor,
    position_embedding: Tensor,
    pre_layrnorm: LayerNorm,
    encoder: ClipEncoder,
    // per grid side, the class position and the patch positions resampled to it
    positions: Mutex<HashMap<usize, Tensor>>,
}

impl ClipOverSam {
    fn load(vb: ShardedVarBuilder, cfg: &Config) -> Result<Self> {
        let clip = &cfg.vision_config.width.clip;
        let grid = clip.image_size / clip.patch_size;
        let encoder_cfg = ClipConfig {
            hidden_size: clip.width,
            intermediate_size: CLIP_MLP_RATIO * clip.width,
            num_hidden_layers: clip.layers,
            num_attention_heads: clip.heads,
            num_channels: 3,
            image_size: clip.image_size,
            patch_size: clip.patch_size,
            hidden_act: Activation::QuickGelu,
            fused_qkv: true,
        };
        let embeddings = vb.pp("embeddings");
        Ok(Self {
            class_embedding: embeddings.get(clip.width, "class_embedding")?,
            position_embedding: embeddings
                .pp("position_embedding")
                .get((grid * grid + 1, clip.width), "weight")?,
            pre_layrnorm: layer_norm(clip.width, CLIP_EPS, vb.pp("pre_layrnorm"))?,
            encoder: ClipEncoder::new(vb.pp("transformer"), &encoder_cfg)?,
            positions: Mutex::new(HashMap::new()),
        })
    }

    // `get_abs_pos`: the class position kept, the patch grid resampled by antialiased bicubic
    fn positions(&self, grid: usize) -> Result<Tensor> {
        let (len, dim) = self.position_embedding.dims2()?;
        let side = (len - 1).isqrt();
        if grid == side {
            return Ok(self.position_embedding.clone());
        }
        let mut cache = self.positions.lock().expect("CLIP position cache poisoned");
        if let Some(positions) = cache.get(&grid) {
            return Ok(positions.clone());
        }
        let patches = self
            .position_embedding
            .narrow(0, 1, side * side)?
            .reshape((side, side, dim))?;
        let resampled = resample_grid(&patches, grid)?.reshape((grid * grid, dim))?;
        let positions = Tensor::cat(&[&self.position_embedding.narrow(0, 0, 1)?, &resampled], 0)?;
        cache.insert(grid, positions.clone());
        Ok(positions)
    }

    // (B, G * G, C) patch features to the last layer's patch outputs, the class token dropped
    fn forward(&self, patches: &Tensor) -> Result<Tensor> {
        let (b, n, dim) = patches.dims3()?;
        let class = self
            .class_embedding
            .to_dtype(patches.dtype())?
            .reshape((1, 1, dim))?
            .broadcast_as((b, 1, dim))?;
        let xs = Tensor::cat(&[&class, patches], 1)?;
        let positions = self.positions(n.isqrt())?.to_dtype(xs.dtype())?;
        let xs = self.pre_layrnorm.forward(&xs.broadcast_add(&positions)?)?;
        let states = self
            .encoder
            .forward_get_hidden_states(&xs, &AttentionMask::None)?;
        states.last().expect("CLIP has layers").narrow(1, 1, n)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let embeddings = uvb.pp("embeddings");
        embeddings.add_tensor("class_embedding", self.class_embedding.clone());
        embeddings
            .pp("position_embedding")
            .add_tensor("weight", self.position_embedding.clone());
        uvb.pp("pre_layrnorm").add(&self.pre_layrnorm);
        uvb.pp("transformer")
            .extend(self.encoder.residual_tensors());
        uvb.to_safetensors()
    }
}

pub struct DeepEncoder {
    sam: SamEncoder,
    clip: ClipOverSam,
    projector: Arc<dyn QuantMethod>,
    image_newline: Tensor,
    view_separator: Tensor,
    global_size: usize,
}

impl DeepEncoder {
    /// `vb` is the checkpoint's `model` prefix.
    pub fn load(vb: ShardedVarBuilder, cfg: &Config) -> Result<Self> {
        let projector = &cfg.projector_config;
        Ok(Self {
            sam: SamEncoder::load(vb.pp("sam_model"), &cfg.sam_config()?)?,
            clip: ClipOverSam::load(vb.pp("vision_model"), cfg)?,
            projector: inference_quant::linear(
                projector.input_dim,
                projector.n_embed,
                &None,
                vb.pp("projector").pp("layers"),
            )?,
            image_newline: vb.get(projector.n_embed, "image_newline")?,
            view_separator: vb.get(projector.n_embed, "view_seperator")?,
            global_size: cfg.vision_config.image_size,
        })
    }

    pub fn dtype(&self) -> inference_tensor::DType {
        self.image_newline.dtype()
    }

    // (B, 3, S, S) views to their (B, G * G, hidden) projected tokens
    fn encode(&self, pixels: &Tensor) -> Result<Tensor> {
        let sam = self.sam.forward(&pixels.to_dtype(self.dtype())?)?;
        let sam = sam.flatten_from(2)?.transpose(1, 2)?.contiguous()?;
        let clip = self.clip.forward(&sam)?;
        self.projector
            .forward(&Tensor::cat(&[&clip, &sam], D::Minus1)?)
    }

    // (rows, cols, hidden) with a newline after every row, flattened
    fn with_newlines(&self, grid: &Tensor) -> Result<Tensor> {
        let (rows, _, hidden) = grid.dims3()?;
        let newline = self
            .image_newline
            .to_dtype(grid.dtype())?
            .reshape((1, 1, hidden))?
            .broadcast_as((rows, 1, hidden))?;
        Tensor::cat(&[grid, &newline.contiguous()?], 1)?.reshape(((), hidden))
    }

    /// One image's tokens: the tiles' grid (each row ending in a newline), the global view's, then the separator.
    pub fn image_features(
        &self,
        global: &Tensor,
        tiles: Option<&Tensor>,
        (cols, rows): (usize, usize),
    ) -> Result<Tensor> {
        let side = Config::grid(self.global_size);
        let global = self.encode(global)?;
        let hidden = global.dim(D::Minus1)?;
        let mut parts = Vec::with_capacity(3);
        if let Some(tiles) = tiles {
            let tile = Config::grid(TILE_SIZE);
            let local = self
                .encode(tiles)?
                .reshape((rows, cols, tile, tile, hidden))?
                .permute((0, 2, 1, 3, 4))?
                .reshape((rows * tile, cols * tile, hidden))?;
            parts.push(self.with_newlines(&local)?);
        }
        parts.push(self.with_newlines(&global.reshape((side, side, hidden))?)?);
        parts.push(
            self.view_separator
                .to_dtype(global.dtype())?
                .reshape((1, hidden))?,
        );
        Tensor::cat(&parts, 0)
    }

    /// Tensors under the checkpoint's `model` prefix.
    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("sam_model").extend(self.sam.residual_tensors());
        uvb.pp("vision_model").extend(self.clip.residual_tensors());
        uvb.pp("projector").pp("layers").add(&self.projector);
        uvb.add_tensor("image_newline", self.image_newline.clone());
        uvb.add_tensor("view_seperator", self.view_separator.clone());
        uvb.to_safetensors()
    }
}
