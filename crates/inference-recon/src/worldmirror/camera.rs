//! The camera head: each frame's camera token, refined over a few steps through transformer blocks that attend across
//! frames, to `[tx, ty, tz, qx, qy, qz, qw, fov_v, fov_u]`; and the decode to OpenCV world-to-camera and intrinsics.

use inference_tensor::nn::{Module, VarBuilder};
use inference_tensor::{D, DType, Device, Result, Tensor};

use super::layers::{AmpLinear, Block, BlockSpec, F32LayerNorm, Mlp};

// Parameters a camera is encoded as: translation, quaternion (scalar last), vertical and horizontal field of view
const CAMERA_PARAMS: usize = 9;
const CAMERA_EPS: f64 = 1e-5;
// the modulation's own LayerNorm: no affine, a smaller epsilon
const MODULATION_EPS: f32 = 1e-6;

pub(crate) struct CameraHead {
    token_norm: F32LayerNorm,
    out_norm: F32LayerNorm,
    init_token: Tensor,
    param_embed: AmpLinear,
    adapt_norm_gen: AmpLinear,
    refine_net: Vec<Block>,
    param_predictor: Mlp,
    steps: usize,
}

impl CameraHead {
    pub(crate) fn load(
        dim: usize,
        heads: usize,
        depth: usize,
        steps: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let spec = BlockSpec {
            dim,
            heads,
            mlp_ratio: 4,
            eps: CAMERA_EPS,
            qk_norm: false,
            dtype: DType::F32,
        };
        Ok(Self {
            token_norm: F32LayerNorm::load(dim, CAMERA_EPS, vb.pp("token_norm"))?,
            out_norm: F32LayerNorm::load(dim, CAMERA_EPS, vb.pp("out_norm"))?,
            init_token: vb.get((1, 1, CAMERA_PARAMS), "init_token")?,
            param_embed: AmpLinear::load(
                CAMERA_PARAMS,
                dim,
                true,
                DType::F32,
                vb.pp("param_embed"),
            )?,
            adapt_norm_gen: AmpLinear::load(
                dim,
                3 * dim,
                true,
                DType::F32,
                vb.pp("adapt_norm_gen").pp(1),
            )?,
            refine_net: (0..depth)
                .map(|i| Block::load(spec, vb.pp("refine_net").pp(i)))
                .collect::<Result<_>>()?,
            param_predictor: Mlp::load_with_hidden(
                dim,
                dim / 2,
                CAMERA_PARAMS,
                DType::F32,
                vb.pp("param_predictor"),
            )?,
            steps,
        })
    }

    /// The activated camera encodings `[frames, 9]` from the trunk's last features `[frames, tokens, 2 * dim]`.
    pub(crate) fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let frames = features.dim(0)?;
        let cam = self
            .token_norm
            .forward(&features.narrow(1, 0, 1)?.reshape((1, frames, ()))?)?;
        let normed = modulation_norm(&cam)?;
        let mut pred: Option<Tensor> = None;
        for _ in 0..self.steps {
            let input = match &pred {
                None => self.param_embed.forward(&self.init_token.broadcast_as((
                    1,
                    frames,
                    CAMERA_PARAMS,
                ))?)?,
                Some(p) => self.param_embed.forward(p)?,
            };
            let modulation = self
                .adapt_norm_gen
                .forward(&inference_tensor::nn::ops::silu(&input)?)?;
            let chunks = modulation.chunk(3, D::Minus1)?;
            let (shift, scale, gate) = (&chunks[0], &chunks[1], &chunks[2]);
            let modulated = (normed.broadcast_mul(&(scale + 1.0)?)? + shift)?;
            let mut xs = (gate * modulated)?.add(&cam)?;
            for block in &self.refine_net {
                xs = block.forward(&xs, None)?;
            }
            let delta = self.param_predictor.forward(&self.out_norm.forward(&xs)?)?;
            pred = Some(match pred {
                None => delta,
                Some(p) => (p + delta)?,
            });
        }
        let pred = pred.expect("at least one refinement step").squeeze(0)?;
        let fov = pred.narrow(1, 7, 2)?.relu()?;
        Tensor::cat(&[&pred.narrow(1, 0, 7)?, &fov], 1)
    }
}

// LayerNorm over the last dimension with no affine
fn modulation_norm(xs: &Tensor) -> Result<Tensor> {
    let dim = xs.dim(D::Minus1)?;
    let ones = Tensor::ones(dim, DType::F32, xs.device())?;
    let zeros = Tensor::zeros(dim, DType::F32, xs.device())?;
    inference_tensor::nn::LayerNorm::new(ones, zeros, f64::from(MODULATION_EPS)).forward(xs)
}

/// One frame's camera: OpenCV convention (x right, y down, z forward), camera-to-world and pixel intrinsics of the
/// processed image.
#[derive(Clone, Debug)]
pub struct Camera {
    pub camera_to_world: [[f32; 4]; 4],
    pub intrinsics: [[f32; 3]; 3],
}

/// The cameras the encodings `[frames, 9]` decode to for an `height x width` image: the encoding's rotation and
/// translation are world-to-camera, and its quaternion need not be unit.
pub fn decode_cameras(params: &Tensor, height: usize, width: usize) -> Result<Vec<Camera>> {
    let rows = params
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;
    Ok(rows
        .iter()
        .map(|p| {
            let (t, q) = ([p[0], p[1], p[2]], [p[3], p[4], p[5], p[6]]);
            let r = quaternion_to_rotation(q);
            // the inverse of [R | t] for a rotation R: [R^T | -R^T t]
            let mut c2w = [[0f32; 4]; 4];
            for i in 0..3 {
                for j in 0..3 {
                    c2w[i][j] = r[j][i];
                }
                c2w[i][3] = -(0..3).map(|j| r[j][i] * t[j]).sum::<f32>();
            }
            c2w[3][3] = 1.0;
            let (h, w) = (height as f32, width as f32);
            let fy = (h / 2.0) / (p[7] / 2.0).tan();
            let fx = (w / 2.0) / (p[8] / 2.0).tan();
            Camera {
                camera_to_world: c2w,
                intrinsics: [[fx, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]],
            }
        })
        .collect())
}

// The rotation of a quaternion (x, y, z, w), scaled by 2 / |q|^2 so an unnormalized one still gives a rotation
fn quaternion_to_rotation([x, y, z, w]: [f32; 4]) -> [[f32; 3]; 3] {
    let s = 2.0 / (x * x + y * y + z * z + w * w);
    [
        [
            1.0 - s * (y * y + z * z),
            s * (x * y - z * w),
            s * (x * z + y * w),
        ],
        [
            s * (x * y + z * w),
            1.0 - s * (x * x + z * z),
            s * (y * z - x * w),
        ],
        [
            s * (x * z - y * w),
            s * (y * z + x * w),
            1.0 - s * (x * x + y * y),
        ],
    ]
}
