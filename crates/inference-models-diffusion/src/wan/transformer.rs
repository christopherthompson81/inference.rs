//! WorldStereo-2's transformer: Wan 2.1 I2V-14B whose self-attention runs jointly over the video tokens and memory
//! (reference) frame tokens, plus a per-frame camera embedding and ControlNet residuals from point-cloud renders.

use inference_quant::ShardedVarBuilder;
use inference_tensor::{DType, Device, Result, Tensor};
use serde::Deserialize;

use super::controlnet::{ControlNet, ControlNetConfig};
use super::ops::{
    ACT_DTYPE, LayerNorm, Linear, PatchConv, RmsNorm, RopeTables, attention, gated_residual,
    gelu_tanh, heads, modulate,
};

const TIME_FREQ_LN_BASE: f64 = 10_000.0;
// The image embedder's LayerNorms use torch's default eps
const IMAGE_NORM_EPS: f64 = 1e-5;
// Modulation rows per block: shift, scale, gate for attention, then for the feed-forward
const BLOCK_MODULATION: usize = 6;

/// The Wan base's transformer config (`transformer/config.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct WanConfig {
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub num_layers: usize,
    pub ffn_dim: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub text_dim: usize,
    pub freq_dim: usize,
    pub image_dim: usize,
    pub eps: f64,
}

/// The WorldStereo checkpoint's `config.json` parts the forward needs.
#[derive(Debug, Clone, Deserialize)]
pub struct WorldStereoConfig {
    pub controlnet_cfg: ControlNetConfig,
}

impl WanConfig {
    pub(crate) fn dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }
}

/// Timestep, text and image conditioning.
pub(crate) struct ConditionEmbedder {
    time_1: Linear,
    time_2: Linear,
    time_proj: Linear,
    text_1: Linear,
    text_2: Linear,
    image_norm1: LayerNorm,
    image_1: Linear,
    image_2: Linear,
    image_norm2: LayerNorm,
    freq_dim: usize,
}

/// What every block reads besides the tokens: `temb` `[1, dim]`, the per-block modulation input `[6, dim]` (f32) and
/// the cross-attention context, image tokens then text tokens.
pub struct Conditions {
    pub temb: Tensor,
    pub timestep_proj: Tensor,
    pub image_context: Tensor,
    pub text_context: Tensor,
}

impl ConditionEmbedder {
    fn load(cfg: &WanConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let dim = cfg.dim();
        Ok(Self {
            time_1: Linear::load(cfg.freq_dim, dim, vb.pp("time_embedder").pp("linear_1"))?,
            time_2: Linear::load(dim, dim, vb.pp("time_embedder").pp("linear_2"))?,
            time_proj: Linear::load(dim, BLOCK_MODULATION * dim, vb.pp("time_proj"))?,
            text_1: Linear::load(cfg.text_dim, dim, vb.pp("text_embedder").pp("linear_1"))?,
            text_2: Linear::load(dim, dim, vb.pp("text_embedder").pp("linear_2"))?,
            image_norm1: LayerNorm::load(
                cfg.image_dim,
                IMAGE_NORM_EPS,
                vb.pp("image_embedder").pp("norm1"),
            )?,
            image_1: Linear::load(
                cfg.image_dim,
                cfg.image_dim,
                vb.pp("image_embedder").pp("ff").pp("net").pp(0).pp("proj"),
            )?,
            image_2: Linear::load(
                cfg.image_dim,
                dim,
                vb.pp("image_embedder").pp("ff").pp("net").pp(2),
            )?,
            image_norm2: LayerNorm::load(dim, IMAGE_NORM_EPS, vb.pp("image_embedder").pp("norm2"))?,
            freq_dim: cfg.freq_dim,
        })
    }

    // diffusers' Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0): [cos, sin] of t * exp(-ln(1e4) i / half)
    fn sinusoid(&self, t: f64) -> Vec<f32> {
        let half = self.freq_dim / 2;
        let angles = (0..half)
            .map(|i| t * (-TIME_FREQ_LN_BASE.ln() * i as f64 / half as f64).exp())
            .collect::<Vec<_>>();
        angles
            .iter()
            .map(|a| a.cos() as f32)
            .chain(angles.iter().map(|a| a.sin() as f32))
            .collect()
    }

    /// Conditions for timestep `t` (0..1000), umT5 states `[1, L, text_dim]` and CLIP states `[1, I, image_dim]`.
    pub(crate) fn forward(&self, t: f64, text: &Tensor, image: &Tensor) -> Result<Conditions> {
        let device = text.device();
        let freq = Tensor::from_vec(self.sinusoid(t), (1, self.freq_dim), device)?;
        let temb = self.time_2.forward(&self.time_1.forward(&freq)?.silu()?)?;
        let timestep_proj = self
            .time_proj
            .forward(&temb.silu()?)?
            .to_dtype(DType::F32)?
            .reshape((BLOCK_MODULATION, ()))?;
        let text_context = self
            .text_2
            .forward(&gelu_tanh(&self.text_1.forward(text)?)?)?;
        let image = self.image_norm1.forward(image)?;
        let image = self
            .image_2
            .forward(&self.image_1.forward(&image)?.gelu_erf()?)?;
        let image_context = self.image_norm2.forward(&image)?.to_dtype(ACT_DTYPE)?;
        Ok(Conditions {
            temb,
            timestep_proj,
            image_context,
            text_context,
        })
    }
}

struct SelfAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    heads: usize,
}

impl SelfAttention {
    fn load(dim: usize, heads: usize, eps: f64, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            to_q: Linear::load(dim, dim, vb.pp("to_q"))?,
            to_k: Linear::load(dim, dim, vb.pp("to_k"))?,
            to_v: Linear::load(dim, dim, vb.pp("to_v"))?,
            to_out: Linear::load(dim, dim, vb.pp("to_out").pp(0))?,
            norm_q: RmsNorm::load(dim, eps, vb.pp("norm_q"))?,
            norm_k: RmsNorm::load(dim, eps, vb.pp("norm_k"))?,
            heads,
        })
    }

    fn to_device(&self, d: &Device) -> Result<Self> {
        Ok(Self {
            to_q: self.to_q.to_device(d)?,
            to_k: self.to_k.to_device(d)?,
            to_v: self.to_v.to_device(d)?,
            to_out: self.to_out.to_device(d)?,
            norm_q: self.norm_q.to_device(d)?,
            norm_k: self.norm_k.to_device(d)?,
            heads: self.heads,
        })
    }

    /// Attention over all of `xs` `[1, tokens, dim]` with `rope` for every token.
    fn forward(&self, xs: &Tensor, rope: &RopeTables) -> Result<Tensor> {
        let q = rope.apply(&heads(
            &self.norm_q.forward(&self.to_q.forward(xs)?)?,
            self.heads,
        )?)?;
        let k = rope.apply(&heads(
            &self.norm_k.forward(&self.to_k.forward(xs)?)?,
            self.heads,
        )?)?;
        let v = heads(&self.to_v.forward(xs)?, self.heads)?;
        self.to_out.forward(&attention(&q, &k, &v)?)
    }
}

// Cross-attention to the text tokens plus, through its own key and value projections, the CLIP image tokens
struct CrossAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    add_k: Linear,
    add_v: Linear,
    to_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    norm_added_k: RmsNorm,
    heads: usize,
}

impl CrossAttention {
    fn load(dim: usize, heads: usize, eps: f64, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            to_q: Linear::load(dim, dim, vb.pp("to_q"))?,
            to_k: Linear::load(dim, dim, vb.pp("to_k"))?,
            to_v: Linear::load(dim, dim, vb.pp("to_v"))?,
            add_k: Linear::load(dim, dim, vb.pp("add_k_proj"))?,
            add_v: Linear::load(dim, dim, vb.pp("add_v_proj"))?,
            to_out: Linear::load(dim, dim, vb.pp("to_out").pp(0))?,
            norm_q: RmsNorm::load(dim, eps, vb.pp("norm_q"))?,
            norm_k: RmsNorm::load(dim, eps, vb.pp("norm_k"))?,
            norm_added_k: RmsNorm::load(dim, eps, vb.pp("norm_added_k"))?,
            heads,
        })
    }

    fn to_device(&self, d: &Device) -> Result<Self> {
        Ok(Self {
            to_q: self.to_q.to_device(d)?,
            to_k: self.to_k.to_device(d)?,
            to_v: self.to_v.to_device(d)?,
            add_k: self.add_k.to_device(d)?,
            add_v: self.add_v.to_device(d)?,
            to_out: self.to_out.to_device(d)?,
            norm_q: self.norm_q.to_device(d)?,
            norm_k: self.norm_k.to_device(d)?,
            norm_added_k: self.norm_added_k.to_device(d)?,
            heads: self.heads,
        })
    }

    fn forward(&self, xs: &Tensor, cond: &Conditions) -> Result<Tensor> {
        let q = heads(&self.norm_q.forward(&self.to_q.forward(xs)?)?, self.heads)?;
        let k = heads(
            &self
                .norm_k
                .forward(&self.to_k.forward(&cond.text_context)?)?,
            self.heads,
        )?;
        let v = heads(&self.to_v.forward(&cond.text_context)?, self.heads)?;
        let k_img = heads(
            &self
                .norm_added_k
                .forward(&self.add_k.forward(&cond.image_context)?)?,
            self.heads,
        )?;
        let v_img = heads(&self.add_v.forward(&cond.image_context)?, self.heads)?;
        let text = attention(&q, &k, &v)?;
        let image = attention(&q, &k_img, &v_img)?;
        self.to_out.forward(&(text + image)?)
    }
}

struct FeedForward {
    up: Linear,
    down: Linear,
}

impl FeedForward {
    fn load(dim: usize, hidden: usize, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            up: Linear::load(dim, hidden, vb.pp("net").pp(0).pp("proj"))?,
            down: Linear::load(hidden, dim, vb.pp("net").pp(2))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.down.forward(&gelu_tanh(&self.up.forward(xs)?)?)
    }

    fn to_device(&self, d: &Device) -> Result<Self> {
        Ok(Self {
            up: self.up.to_device(d)?,
            down: self.down.to_device(d)?,
        })
    }
}

/// One block. Memory tokens share the self-attention (jointly with the video tokens) and the feed-forward, but skip the
/// cross-attention.
pub(crate) struct Block {
    attn1: SelfAttention,
    attn2: CrossAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    ffn: FeedForward,
    scale_shift_table: Tensor,
}

impl Block {
    pub(crate) fn load(cfg: &WanConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let (dim, h, eps) = (cfg.dim(), cfg.num_attention_heads, cfg.eps);
        Ok(Self {
            attn1: SelfAttention::load(dim, h, eps, vb.pp("attn1"))?,
            attn2: CrossAttention::load(dim, h, eps, vb.pp("attn2"))?,
            norm1: LayerNorm::plain(eps),
            norm2: LayerNorm::load(dim, eps, vb.pp("norm2"))?,
            norm3: LayerNorm::plain(eps),
            ffn: FeedForward::load(dim, cfg.ffn_dim, vb.pp("ffn"))?,
            scale_shift_table: vb
                .get((1, BLOCK_MODULATION, dim), "scale_shift_table")?
                .to_dtype(DType::F32)?
                .squeeze(0)?,
        })
    }

    /// A copy on `device`, for blocks kept in host memory and streamed in one at a time.
    pub(crate) fn to_device(&self, d: &Device) -> Result<Self> {
        Ok(Self {
            attn1: self.attn1.to_device(d)?,
            attn2: self.attn2.to_device(d)?,
            norm1: self.norm1.to_device(d)?,
            norm2: self.norm2.to_device(d)?,
            norm3: self.norm3.to_device(d)?,
            ffn: self.ffn.to_device(d)?,
            scale_shift_table: self.scale_shift_table.to_device(d)?,
        })
    }

    /// Video tokens `[1, N, dim]` and memory tokens `[1, M, dim]` (M may be 0); `rope` covers memory then video.
    pub(crate) fn forward(
        &self,
        video: &Tensor,
        memory: &Tensor,
        cond: &Conditions,
        rope: &RopeTables,
    ) -> Result<(Tensor, Tensor)> {
        let m = (self.scale_shift_table.clone() + &cond.timestep_proj)?;
        let row = |i: usize| m.narrow(0, i, 1);
        let (shift, scale, gate) = (row(0)?, row(1)?, row(2)?);
        let (c_shift, c_scale, c_gate) = (row(3)?, row(4)?, row(5)?);
        let n_mem = memory.dim(1)?;
        let tokens = Tensor::cat(&[memory, video], 1)?;
        let normed = modulate(&self.norm1.forward(&tokens)?, &shift, &scale)?;
        let attn = self.attn1.forward(&normed, rope)?;
        let tokens = gated_residual(&tokens, &attn, &gate)?;
        let memory = tokens.narrow(1, 0, n_mem)?;
        let video = tokens.narrow(1, n_mem, tokens.dim(1)? - n_mem)?;
        let cross = self
            .attn2
            .forward(&self.norm2.forward(&video)?.to_dtype(ACT_DTYPE)?, cond)?;
        let video = (video + cross)?;
        let tokens = Tensor::cat(&[&memory, &video], 1)?;
        let normed = modulate(&self.norm3.forward(&tokens)?, &c_shift, &c_scale)?;
        let tokens = gated_residual(&tokens, &self.ffn.forward(&normed)?, &c_gate)?;
        Ok((
            tokens.narrow(1, n_mem, tokens.dim(1)? - n_mem)?,
            tokens.narrow(1, 0, n_mem)?,
        ))
    }
}

/// The inputs of one denoising step, all for batch size 1 and latent frames F at latent size H x W.
pub struct StepInputs<'a> {
    /// `[36, F, H, W]`: the noisy latents (16 ch), then the first-frame mask (4) and condition latents (16)
    pub latents: &'a Tensor,
    /// `[16, F, H, W]` normalized latents of the point-cloud renders
    pub render_latent: &'a Tensor,
    /// `[7, F, 8H, 8W]`: the render mask and the per-pixel camera embedding at pixel resolution
    pub render_conditions: &'a Tensor,
    /// `[F, 7]` and `[R, 7]`: each frame's and memory frame's camera as quaternion and translation
    pub camera_qt: &'a Tensor,
    pub camera_qt_ref: Option<&'a Tensor>,
    /// `[36, R, H, W]` memory frames (latents, a ones mask, latents again) and the video frame each sits beside
    pub reference_latent: Option<&'a Tensor>,
    pub ref_index: &'a [usize],
}

pub struct WorldStereoTransformer {
    patch_embedding: PatchConv,
    camera_embedding: [Linear; 3],
    condition_embedder: ConditionEmbedder,
    blocks: Vec<Block>,
    controlnet: ControlNet,
    norm_out: LayerNorm,
    proj_out: Linear,
    scale_shift_table: Tensor,
    cfg: WanConfig,
    device: Device,
}

impl WorldStereoTransformer {
    /// Every weight on the builder's device.
    pub fn new(cfg: &WanConfig, ws: &WorldStereoConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Self::load(cfg, ws, vb.clone(), vb)
    }

    /// The 40 main blocks in host memory, uploaded one at a time as the forward reaches them; everything else (2 GB)
    /// on the builder's device. The bf16 checkpoint then needs about one block of device memory for weights.
    pub fn new_streamed(
        cfg: &WanConfig,
        ws: &WorldStereoConfig,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let host = vb.clone().set_device(Device::Cpu);
        Self::load(cfg, ws, vb, host)
    }

    fn load(
        cfg: &WanConfig,
        ws: &WorldStereoConfig,
        vb: ShardedVarBuilder,
        blocks_vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let dim = cfg.dim();
        let device = vb.device().clone();
        let cam = vb.pp("camera_embedding");
        let cam_dim = ws.controlnet_cfg.camera_embedding_dim;
        Ok(Self {
            patch_embedding: PatchConv::load(&vb.pp("patch_embedding"))?,
            camera_embedding: [
                Linear::load(cam_dim, dim / 2, cam.pp(0))?,
                Linear::load(dim / 2, dim, cam.pp(2))?,
                Linear::load(dim, dim, cam.pp(4))?,
            ],
            condition_embedder: ConditionEmbedder::load(cfg, vb.pp("condition_embedder"))?,
            blocks: (0..cfg.num_layers)
                .map(|i| Block::load(cfg, blocks_vb.pp("blocks").pp(i)))
                .collect::<Result<_>>()?,
            controlnet: ControlNet::load(&ws.controlnet_cfg, dim, vb.pp("controlnet"))?,
            norm_out: LayerNorm::plain(cfg.eps),
            proj_out: Linear::load(dim, cfg.out_channels * 4, vb.pp("proj_out"))?,
            scale_shift_table: vb
                .get((1, 2, dim), "scale_shift_table")?
                .to_dtype(DType::F32)?
                .squeeze(0)?,
            cfg: cfg.clone(),
            device,
        })
    }

    pub fn conditions(&self, t: f64, text: &Tensor, image: &Tensor) -> Result<Conditions> {
        self.condition_embedder.forward(t, text, image)
    }

    // the camera MLP of `[F, 7]` broadcast over each frame's `tokens_per_frame` tokens
    fn camera_tokens(&self, qt: &Tensor, tokens_per_frame: usize) -> Result<Tensor> {
        let [l0, l1, l2] = &self.camera_embedding;
        let e = l2.forward(&l1.forward(&l0.forward(qt)?.silu()?)?.silu()?)?;
        let (f, dim) = e.dims2()?;
        e.unsqueeze(1)?
            .broadcast_as((f, tokens_per_frame, dim))?
            .reshape((1, f * tokens_per_frame, dim))
    }

    /// The flow prediction `[16, F, H, W]` (f32) for one step.
    pub fn forward(&self, inputs: &StepInputs, cond: &Conditions) -> Result<Tensor> {
        let (_, f, h, w) = inputs.latents.dims4()?;
        let (ph, pw) = (h / 2, w / 2);
        let head_dim = self.cfg.attention_head_dim;
        let device = inputs.latents.device();
        let per_frame = ph * pw;
        // RoPE on a grid twice as wide: video tokens take columns 0..pw, memory tokens pw..2pw at frame ref_index + 1
        let video_rope = RopeTables::grid(head_dim, (f, ph, pw), device)?;
        let video = (self.patch_embedding.forward(inputs.latents)?.unsqueeze(0)?
            + self.camera_tokens(inputs.camera_qt, per_frame)?)?;
        let (memory, rope) = match (inputs.reference_latent, inputs.camera_qt_ref) {
            (Some(reference), Some(qt_ref)) => {
                let positions = inputs
                    .ref_index
                    .iter()
                    .flat_map(|&r| {
                        (0..ph).flat_map(move |y| (0..pw).map(move |x| [r + 1, y, pw + x]))
                    })
                    .collect::<Vec<_>>();
                let memory_rope = RopeTables::new(head_dim, &positions, device)?;
                let memory = (self.patch_embedding.forward(reference)?.unsqueeze(0)?
                    + self.camera_tokens(qt_ref, per_frame)?)?;
                (memory, RopeTables::cat(&[&memory_rope, &video_rope])?)
            }
            _ => (video.narrow(1, 0, 0)?, video_rope),
        };
        let control = self.controlnet.forward(inputs, &cond.temb, (f, h, w))?;
        let (mut video, mut memory) = (video, memory);
        for (i, block) in self.blocks.iter().enumerate() {
            (video, memory) = if block.scale_shift_table.device().same_device(&self.device) {
                block.forward(&video, &memory, cond, &rope)?
            } else {
                block
                    .to_device(&self.device)?
                    .forward(&video, &memory, cond, &rope)?
            };
            if let Some(c) = control.get(i) {
                video = (video + c)?;
            }
        }
        let ss = self
            .scale_shift_table
            .broadcast_add(&cond.temb.to_dtype(DType::F32)?)?;
        let (shift, scale) = (ss.narrow(0, 0, 1)?, ss.narrow(0, 1, 1)?);
        let out = self
            .proj_out
            .forward(&modulate(&self.norm_out.forward(&video)?, &shift, &scale)?)?
            .to_dtype(DType::F32)?;
        // tokens (f, ph, pw) x (pt=1, kh, kw, c) to [c, f, h, w]
        let c = self.cfg.out_channels;
        out.reshape((f, ph, pw, 2, 2, c))?
            .permute((5, 0, 1, 3, 2, 4))?
            .reshape((c, f, h, w))
    }
}
