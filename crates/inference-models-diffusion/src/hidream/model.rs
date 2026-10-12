//! HiDream-O1's pixel-space transformer: Qwen3-VL over the prompt (reference images in it), a timestep, 32 px patches.

use std::sync::OnceLock;

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::Linear;
use inference_tensor::{DType, Device, Module, Result, Tensor};
use serde::Deserialize;

use super::references::References;
use crate::attention::{AttentionMask, FlashParams};
use crate::layers;
use inference_nn::decoder::qwen3_vl::{TextConfig, deepstack_add};
use inference_nn::decoder::{DecoderStack, LayerMasks};
use inference_nn::device_map::DeviceMappedMask;
use inference_nn::model::{ModelForwardContext, NormalLoadingMetadata};
use inference_nn::paged_attention::AttentionImplementation;
use inference_nn::vision::qwen3_vl::{Qwen3VLVisionModel, VisionConfig};

pub const PATCH_SIZE: usize = 32;
pub const CHANNELS: usize = 3;
pub const PATCH_DIM: usize = PATCH_SIZE * PATCH_SIZE * CHANNELS;
/// The target patches' M-RoPE offset on every axis; a prompt must end before it.
pub const FIX_POINT: u32 = 4096;
const FREQUENCY_EMBEDDING_SIZE: usize = 256;
const MAX_PERIOD: f64 = 10000.0;
// the ViT's first tensor, which a GGUF written for text-to-image alone leaves out
const VISION_TENSOR: &str = "model.visual.patch_embed.proj.weight";
/// Why weights without the ViT refuse reference images.
pub const NO_VISION_TOWER: &str = "these HiDream-O1 weights have no vision tower (`model.visual.*`), which \
    reference images need; rewrite the GGUF from the checkpoint";

/// The fields of the checkpoint's `Qwen3VLForConditionalGeneration` config the denoiser reads.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_id: u32,
}

/// One prompt laid out for the denoiser: its tokens, every row's M-RoPE position and the attention mask.
pub struct Sequence {
    input_ids: Tensor,
    positions: Tensor,
    masks: LayerMasks,
    text_len: usize,
    image_len: usize,
    reference_len: usize,
    // each run of image pads in the prompt, as (first token, length)
    pad_runs: Vec<(usize, usize)>,
    // the pad rows, each index repeated across the hidden size, where deepstack adds its features
    pad_rows: Option<Tensor>,
}

impl Sequence {
    /// Every row's (t, h, w) M-RoPE position, (3, 1, rows).
    pub fn positions(&self) -> &Tensor {
        &self.positions
    }
}

/// The reference images through the ViT and the patch embedder, once per request.
pub struct Conditioning {
    vit: Tensor,
    deepstack: Vec<Tensor>,
    patches: Tensor,
}

impl Conditioning {
    /// The ViT's output for the prompt's pads and each deepstack layer's, both (pads, hidden).
    pub fn vision(&self) -> (&Tensor, &[Tensor]) {
        (&self.vit, &self.deepstack)
    }

    /// The same conditioning with another ViT output and deepstack features, as a parity check substitutes them.
    pub fn with_vision(self, vit: Tensor, deepstack: Vec<Tensor>) -> Self {
        Self {
            vit,
            deepstack,
            ..self
        }
    }
}

// the ViT, built from its weights on the first request with references, so text-to-image never holds it on the device
struct LazyVision {
    cfg: VisionConfig,
    vb: ShardedVarBuilder,
    model: OnceLock<Qwen3VLVisionModel>,
}

impl LazyVision {
    fn get(&self) -> Result<&Qwen3VLVisionModel> {
        if let Some(model) = self.model.get() {
            return Ok(model);
        }
        let model = Qwen3VLVisionModel::new(&self.cfg, self.vb.clone())?;
        Ok(self.model.get_or_init(|| model))
    }
}

pub struct HiDreamO1 {
    stack: DecoderStack,
    vision: Option<LazyVision>,
    image_token_id: u32,
    hidden: usize,
    x_proj1: Linear,
    x_proj2: Linear,
    t_mlp0: Linear,
    t_mlp2: Linear,
    final_layer: Linear,
    dtype: DType,
}

impl HiDreamO1 {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        metadata: NormalLoadingMetadata,
    ) -> Result<Self> {
        let hidden = cfg.text_config.hidden_size;
        let vb_m = vb.pp("model");
        let dtype = vb.dtype();
        let x = vb_m.pp("x_embedder");
        let t = vb_m.pp("t_embedder1").pp("mlp");
        let vision = vb.contains_tensor(VISION_TENSOR).then(|| LazyVision {
            cfg: cfg.vision_config.clone(),
            vb: vb_m.pp("visual"),
            model: OnceLock::new(),
        });
        Ok(Self {
            vision,
            image_token_id: cfg.image_token_id,
            hidden,
            x_proj1: layers::linear_no_bias(PATCH_DIM, hidden / 4, x.pp("proj1"))?,
            x_proj2: layers::linear(hidden / 4, hidden, x.pp("proj2"))?,
            t_mlp0: layers::linear(FREQUENCY_EMBEDDING_SIZE, hidden, t.pp("0"))?,
            t_mlp2: layers::linear(hidden, hidden, t.pp("2"))?,
            final_layer: layers::linear(hidden, PATCH_DIM, vb_m.pp("final_layer2").pp("linear"))?,
            stack: DecoderStack::new(
                &cfg.text_config.decoder_spec(false),
                vb_m.pp("language_model"),
                None,
                true,
                metadata,
                &AttentionImplementation::Eager,
            )?,
            dtype,
        })
    }

    pub fn device(&self) -> &Device {
        &self.stack.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn image_token_id(&self) -> u32 {
        self.image_token_id
    }

    /// Whether the weights carry the ViT that reference images need.
    pub fn sees_images(&self) -> bool {
        self.vision.is_some()
    }

    /// Lays out `tokens` (ending in the timestep token), a `rows` x `cols` target grid, then `references`' patches.
    pub fn sequence(
        &self,
        tokens: &[u32],
        rows: usize,
        cols: usize,
        references: Option<&References>,
    ) -> Result<Sequence> {
        let text_len = tokens.len();
        let (vit_tokens, merged, patch_grids) = match references {
            Some(r) => (r.vit_tokens(), r.merged_grids(), r.patch_grids.clone()),
            None => Default::default(),
        };
        let image_len = rows * cols;
        let reference_len: usize = patch_grids.iter().map(|(r, c)| r * c).sum();
        let total = text_len + image_len + reference_len;
        let device = self.device();
        let mut positions = vec![0u32; 3 * total];
        let mut set = |i: usize, (t, h, w): (u32, u32, u32)| {
            positions[i] = t;
            positions[total + i] = h;
            positions[2 * total + i] = w;
        };
        // Qwen3-VL's prompt positions, each image's pads over its merged grid; a pad past the images is caption text
        let (mut next, mut i, mut pad_runs) = (0u32, 0, Vec::with_capacity(merged.len()));
        while i < text_len {
            let k = pad_runs.len();
            if tokens[i] != self.image_token_id || k == vit_tokens.len() {
                set(i, (next, next, next));
                (next, i) = (next + 1, i + 1);
                continue;
            }
            let (n, (gh, gw)) = (vit_tokens[k], merged[k]);
            for j in 0..n {
                let (r, c) = ((j / gw) as u32, (j % gw) as u32);
                set(i + j, (next, next + r, next + c));
            }
            pad_runs.push((i, n));
            next += gh.max(gw) as u32;
            i += n;
        }
        if next > FIX_POINT {
            inference_tensor::bail!(
                "the prompt reaches position {next}; HiDream-O1 places its image at position {FIX_POINT}"
            );
        }
        for r in 0..rows {
            for c in 0..cols {
                let (r32, c32) = (r as u32, c as u32);
                set(
                    text_len + r * cols + c,
                    (FIX_POINT, FIX_POINT + r32, FIX_POINT + c32),
                );
            }
        }
        // each reference continues past the previous grid's largest position
        let mut start = FIX_POINT + rows.max(cols) as u32;
        let mut row = text_len + image_len;
        for &(rr, rc) in &patch_grids {
            for r in 0..rr {
                for c in 0..rc {
                    set(
                        row + r * rc + c,
                        (start, start + r as u32, start + c as u32),
                    );
                }
            }
            row += rr * rc;
            start += rr.max(rc) as u32;
        }
        let hidden = self.hidden;
        let pad_rows = (!pad_runs.is_empty())
            .then(|| {
                let rows: Vec<u32> = pad_runs
                    .iter()
                    .flat_map(|&(s, n)| (s..s + n).map(|r| r as u32))
                    .collect();
                let n = rows.len();
                Tensor::from_vec(rows, (n, 1), device)?.repeat((1, hidden))
            })
            .transpose()?;
        // the prompt is causal; the timestep token (the prompt's last) and every patch see everything
        let gen_start = text_len - 1;
        let mut mask = vec![0f32; total * total];
        for (q, row) in mask.chunks_mut(total).enumerate().take(gen_start) {
            row[q + 1..].fill(f32::NEG_INFINITY);
        }
        let mask = Tensor::from_vec(mask, (total, total), device)?.to_dtype(self.dtype)?;
        Ok(Sequence {
            input_ids: Tensor::new(tokens, device)?.unsqueeze(0)?,
            positions: Tensor::from_vec(positions, (3, 1, total), device)?,
            masks: LayerMasks::new(
                Some(DeviceMappedMask::new(
                    AttentionMask::Custom(mask),
                    &*self.stack.mapper,
                )?),
                None,
                Some(FlashParams::empty(false)),
            ),
            text_len,
            image_len,
            reference_len,
            pad_runs,
            pad_rows,
        })
    }

    /// `references` through the ViT, for the prompt's pads, and through the patch embedder.
    pub fn condition(&self, references: &References) -> Result<Conditioning> {
        let Some(vision) = &self.vision else {
            inference_tensor::bail!("{NO_VISION_TOWER}");
        };
        let device = self.device();
        let (vit, deepstack) = vision.get()?.forward(
            &references
                .vit_pixels
                .to_device(device)?
                .to_dtype(self.dtype)?,
            &references.vit_grid_tensor(device)?,
        )?;
        Ok(Conditioning {
            vit: vit.to_dtype(self.dtype)?,
            deepstack,
            patches: self.embed_patches(&references.patches.to_device(device)?)?,
        })
    }

    fn embed_patches(&self, z: &Tensor) -> Result<Tensor> {
        self.x_proj2
            .forward(&self.x_proj1.forward(&z.to_dtype(self.dtype)?)?)
    }

    /// The timestep tokens (n, 1, hidden) for flow times `ts`, built once so a sampling loop uploads nothing per step.
    pub fn timestep_embeddings(&self, ts: &[f32]) -> Result<Tensor> {
        let half = FREQUENCY_EMBEDDING_SIZE / 2;
        let mut emb = vec![0f32; ts.len() * FREQUENCY_EMBEDDING_SIZE];
        for (row, &t) in emb.chunks_mut(FREQUENCY_EMBEDDING_SIZE).zip(ts) {
            let scaled = t * super::sampling::NUM_TRAIN_TIMESTEPS as f32;
            for i in 0..half {
                let freq = (-(MAX_PERIOD.ln() as f32) * i as f32 / half as f32).exp();
                let arg = scaled * freq;
                row[i] = arg.cos();
                row[half + i] = arg.sin();
            }
        }
        let emb = Tensor::from_vec(emb, (ts.len(), FREQUENCY_EMBEDDING_SIZE), self.device())?
            .to_dtype(self.dtype)?;
        self.t_mlp2
            .forward(&self.t_mlp0.forward(&emb)?.silu()?)?
            .unsqueeze(1)
    }

    /// The clean-image prediction for patches `z` (1, patches, 3072) at `t_emb`, with the sequence's `conditioning`.
    pub fn forward(
        &self,
        seq: &Sequence,
        z: &Tensor,
        t_emb: &Tensor,
        conditioning: Option<&Conditioning>,
    ) -> Result<Tensor> {
        let conditioning = match (seq.reference_len > 0, conditioning) {
            (false, _) => None,
            (true, Some(conditioning)) => Some(conditioning),
            (true, None) => {
                inference_tensor::bail!("the sequence has reference images but no conditioning")
            }
        };
        let embedded = self.stack.embed(&seq.input_ids)?;
        let mut parts = Vec::with_capacity(2 * seq.pad_runs.len() + 4);
        let mut at = 0;
        // a sequence's pads come only with its references, which the match above holds to conditioning
        if let Some(conditioning) = conditioning {
            let mut vit_row = 0;
            for &(start, n) in &seq.pad_runs {
                parts.push(embedded.narrow(1, at, start - at)?);
                parts.push(conditioning.vit.narrow(0, vit_row, n)?.unsqueeze(0)?);
                (at, vit_row) = (start + n, vit_row + n);
            }
        }
        parts.push(embedded.narrow(1, at, seq.text_len - 1 - at)?);
        parts.push(t_emb.clone());
        parts.push(self.embed_patches(z)?);
        if let Some(conditioning) = conditioning {
            parts.push(conditioning.patches.clone());
        }
        let xs = Tensor::cat(&parts, 1)?;
        let total = seq.text_len + seq.image_len + seq.reference_len;
        let (offsets, context_lens, position_ids) = ([0], [(0, total)], [total]);
        let flash = FlashParams::empty(false);
        let mut ctx =
            ModelForwardContext::new(&offsets, &context_lens, &position_ids, None, &flash);
        let (cos, sin) = self.stack.mrope_tables(&seq.positions, self.dtype)?;
        ctx.set_rope_tables(cos, sin);
        // deepstack adds the ViT's earlier layers at the pads after the first few decoder layers
        let deepstack = |i: usize, xs: Tensor| match (&seq.pad_rows, conditioning) {
            (Some(rows), Some(c)) if i < c.deepstack.len() => {
                deepstack_add(xs, rows, &c.deepstack[i])
            }
            _ => Ok(xs),
        };
        let hidden = self
            .stack
            .forward_hooked(xs, &seq.masks, None, &mut ctx, &deepstack)?;
        self.final_layer
            .forward(&hidden.narrow(1, seq.text_len, seq.image_len)?)
    }
}
