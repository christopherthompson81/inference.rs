//! HiDream-O1's pixel-space transformer: Qwen3-VL's text stack over the prompt, a timestep token and 32 px patches.

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::Linear;
use inference_tensor::{DType, Device, Module, Result, Tensor};
use serde::Deserialize;

use crate::attention::{AttentionMask, FlashParams};
use crate::layers;
use inference_nn::decoder::qwen3_vl::TextConfig;
use inference_nn::decoder::{DecoderStack, LayerMasks};
use inference_nn::device_map::DeviceMappedMask;
use inference_nn::model::{ModelForwardContext, NormalLoadingMetadata};
use inference_nn::paged_attention::AttentionImplementation;

pub const PATCH_SIZE: usize = 32;
pub const CHANNELS: usize = 3;
pub const PATCH_DIM: usize = PATCH_SIZE * PATCH_SIZE * CHANNELS;
/// The target patches' M-RoPE offset on every axis; a prompt must end before it.
pub const FIX_POINT: u32 = 4096;
const FREQUENCY_EMBEDDING_SIZE: usize = 256;
const MAX_PERIOD: f64 = 10000.0;

/// The fields of the checkpoint's `Qwen3VLForConditionalGeneration` config that text-to-image reads.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
}

/// One prompt laid out for the denoiser: its tokens, every row's M-RoPE position and the attention mask.
pub struct Sequence {
    input_ids: Tensor,
    positions: Tensor,
    masks: LayerMasks,
    text_len: usize,
    image_len: usize,
}

pub struct HiDreamO1 {
    stack: DecoderStack,
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
        Ok(Self {
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

    /// Lays out `tokens` (ending in the timestep token) before a `rows` x `cols` grid of target patches.
    pub fn sequence(&self, tokens: &[u32], rows: usize, cols: usize) -> Result<Sequence> {
        let text_len = tokens.len();
        if text_len > FIX_POINT as usize {
            inference_tensor::bail!(
                "the prompt is {text_len} tokens; HiDream-O1 places its image at position {FIX_POINT}"
            );
        }
        let image_len = rows * cols;
        let total = text_len + image_len;
        let device = self.device();
        let mut positions = vec![0u32; 3 * total];
        for i in 0..text_len {
            positions[i] = i as u32;
            positions[total + i] = i as u32;
            positions[2 * total + i] = i as u32;
        }
        for r in 0..rows {
            for c in 0..cols {
                let i = text_len + r * cols + c;
                positions[i] = FIX_POINT;
                positions[total + i] = FIX_POINT + r as u32;
                positions[2 * total + i] = FIX_POINT + c as u32;
            }
        }
        // the prompt is causal; the timestep token (the prompt's last) and the patches see everything
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
        })
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

    /// The clean-image prediction for patches `z` (1, patches, 3072), `t_emb` a row of `timestep_embeddings`.
    pub fn forward(&self, seq: &Sequence, z: &Tensor, t_emb: &Tensor) -> Result<Tensor> {
        let text = self.stack.embed(&seq.input_ids)?;
        let text = Tensor::cat(&[&text.narrow(1, 0, seq.text_len - 1)?, t_emb], 1)?;
        let patches = self
            .x_proj2
            .forward(&self.x_proj1.forward(&z.to_dtype(self.dtype)?)?)?;
        let xs = Tensor::cat(&[text, patches], 1)?;
        let total = seq.text_len + seq.image_len;
        let (offsets, context_lens, position_ids) = ([0], [(0, total)], [total]);
        let flash = FlashParams::empty(false);
        let mut ctx =
            ModelForwardContext::new(&offsets, &context_lens, &position_ids, None, &flash);
        let (cos, sin) = self.stack.mrope_tables(&seq.positions, self.dtype)?;
        ctx.set_rope_tables(cos, sin);
        let hidden = self.stack.forward(xs, &seq.masks, None, &mut ctx)?;
        self.final_layer
            .forward(&hidden.narrow(1, seq.text_len, seq.image_len)?)
    }
}
