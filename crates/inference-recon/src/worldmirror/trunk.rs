//! The visual geometry transformer: per-frame tokens (camera, registers, pose and ray prior tokens, patches) through 24
//! pairs of frame attention (within a frame) and global attention (across all frames), with normalized 2D RoPE.

use inference_tensor::nn::VarBuilder;
use inference_tensor::{DType, Result, Tensor};

use super::dinov2::Dinov2;
use super::layers::{Block, BlockSpec};
use super::rope::{RopeTable, frame_cells};

// The trunk's LayerNorm epsilon
const TRUNK_EPS: f64 = 1e-5;
// Layers whose frame and global outputs, concatenated, feed the heads
const HEAD_LAYERS: [usize; 4] = [4, 11, 17, 23];
// Tokens before a frame's patches: camera, the registers, then the pose and ray prior tokens
const CAMERA_TOKENS: usize = 1;
const PRIOR_TOKENS: usize = 2;

/// What the trunk hands the heads: the four `[frames, tokens, 2 * dim]` feature maps and where the patches start.
pub(crate) struct TrunkOutput {
    pub features: Vec<Tensor>,
    pub patch_start: usize,
    pub hp: usize,
    pub wp: usize,
}

pub(crate) struct Trunk {
    dinov2: Dinov2,
    cam_token: Tensor,
    reg_token: Tensor,
    frame_blocks: Vec<Block>,
    global_blocks: Vec<Block>,
    heads: usize,
    patch: usize,
    rope_base: f64,
    dtype: DType,
}

impl Trunk {
    pub(crate) fn load(cfg: &super::WorldMirrorConfig, dtype: DType, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.embed_dim;
        let spec = BlockSpec {
            dim,
            heads: cfg.num_heads,
            mlp_ratio: 4,
            eps: TRUNK_EPS,
            qk_norm: true,
            dtype,
        };
        let blocks = |name: &str| {
            (0..cfg.depth)
                .map(|i| Block::load(spec, vb.pp(name).pp(i)))
                .collect::<Result<Vec<_>>>()
        };
        let registers = cfg.num_register_tokens;
        Ok(Self {
            dinov2: Dinov2::load(
                dim,
                cfg.depth,
                cfg.num_heads,
                cfg.patch_size,
                registers,
                dtype,
                vb.pp("patch_embed"),
            )?,
            cam_token: vb.get((1, 2, 1, dim), "cam_token")?,
            reg_token: vb.get((1, 2, registers, dim), "reg_token")?,
            frame_blocks: blocks("frame_blocks")?,
            global_blocks: blocks("global_blocks")?,
            heads: cfg.num_heads,
            patch: cfg.patch_size,
            rope_base: cfg.rope_base,
            dtype,
        })
    }

    /// The head features of normalized images `[frames, 3, H, W]`, with no priors (their tokens stay zero).
    // The first frame block on `xs` `[frames, tokens, dim]` of an `hp x wp` grid, its insides to `inspect`
    pub(crate) fn first_frame_block(
        &self,
        xs: &Tensor,
        hp: usize,
        wp: usize,
        inspect: &mut dyn FnMut(&str, &Tensor),
    ) -> Result<Tensor> {
        let dim = xs.dim(2)?;
        let specials = CAMERA_TOKENS + self.reg_token.dim(2)? + PRIOR_TOKENS;
        let rope = RopeTable::new(hp + 1, wp + 1, dim / self.heads, self.rope_base, self.dtype, xs.device())?;
        let cells = Tensor::new(frame_cells(specials, hp, wp).as_slice(), xs.device())?;
        self.frame_blocks[0].forward_inspect(xs, Some((&rope, &cells)), inspect)
    }

    pub(crate) fn forward(&self, images: &Tensor) -> Result<TrunkOutput> {
        self.forward_inspect(images, &mut |_, _, _| {})
    }

    // `forward`, handing `inspect` every layer's frame and global output
    pub(crate) fn forward_inspect(
        &self,
        images: &Tensor,
        inspect: &mut dyn FnMut(usize, &Tensor, &Tensor),
    ) -> Result<TrunkOutput> {
        let (frames, _, h, w) = images.dims4()?;
        let (hp, wp) = (h / self.patch, w / self.patch);
        let device = images.device();
        let patches = self.dinov2.forward(images)?;
        let dim = patches.dim(2)?;
        let registers = self.reg_token.dim(2)?;
        let specials = CAMERA_TOKENS + registers + PRIOR_TOKENS;
        // frame 0 takes the first camera and register tokens, every other frame the second
        let per_frame = |token: &Tensor, n: usize| -> Result<Tensor> {
            let first = token.narrow(1, 0, 1)?.reshape((1, n, dim))?;
            let rest = token.narrow(1, 1, 1)?.reshape((1, n, dim))?;
            if frames == 1 {
                return Ok(first);
            }
            Tensor::cat(&[&first, &rest.broadcast_as((frames - 1, n, dim))?], 0)
        };
        let priors = Tensor::zeros((frames, PRIOR_TOKENS, dim), DType::F32, device)?;
        let mut xs = Tensor::cat(
            &[
                &per_frame(&self.cam_token, CAMERA_TOKENS)?,
                &per_frame(&self.reg_token, registers)?,
                &priors,
                &patches,
            ],
            1,
        )?;
        let tokens = xs.dim(1)?;
        let head_dim = dim / self.heads;
        let rope = RopeTable::new(hp + 1, wp + 1, head_dim, self.rope_base, self.dtype, device)?;
        let cells = frame_cells(specials, hp, wp);
        let frame_cells = Tensor::new(cells.as_slice(), device)?;
        let global_cells = Tensor::new(cells.repeat(frames).as_slice(), device)?;
        let mut features = Vec::with_capacity(HEAD_LAYERS.len());
        for (i, (frame_block, global_block)) in self.frame_blocks.iter().zip(&self.global_blocks).enumerate() {
            let local = frame_block.forward(&xs, Some((&rope, &frame_cells)))?;
            let global = global_block
                .forward(&local.reshape((1, frames * tokens, dim))?, Some((&rope, &global_cells)))?
                .reshape((frames, tokens, dim))?;
            inspect(i, &local, &global);
            if HEAD_LAYERS.contains(&i) {
                features.push(Tensor::cat(&[&local, &global], 2)?);
            }
            xs = global;
        }
        Ok(TrunkOutput {
            features,
            patch_start: specials,
            hp,
            wp,
        })
    }

}
