//! The plain (unmodulated) sparse transformers LATO.2's VAEs stack: `SparseTransformerBase` (self-attention, full or
//! in shifted 8^3 windows) and `SparseTransformerCrossBase` (plus cross-attention to a context), positions as RoPE on
//! the cells' coordinates or as absolute sinusoids added to the features. The residual stream stays in f32 with the
//! norms (as autocast leaves them), the linears and attention run in the given dtype.

use inference_tensor::nn::VarBuilder;
use inference_tensor::{D, DType, Device, Result, Tensor};

use super::ape;
use crate::trellis2::dit::{Rope, sdpa};
use crate::worldmirror::layers::AmpLinear;

// LayerNorm32(elementwise_affine=False, eps=1e-6)
const NORM_EPS: f64 = 1e-6;
const MLP_RATIO: usize = 4;

/// How a stack's tokens learn where they are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Positions {
    Rope,
    Absolute,
}

/// A stack's self-attention: over every token, or within 8^3 windows shifted by half a window every other block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnMode {
    Full,
    Swin { window: i32 },
}

#[derive(Clone, Copy, Debug)]
pub struct StackSpec {
    pub input: usize,
    pub channels: usize,
    /// Cross-attention context width, none for a self-attention stack
    pub context: Option<usize>,
    pub blocks: usize,
    pub heads: usize,
    pub positions: Positions,
    pub attn: AttnMode,
}

/// Affine-free LayerNorm over the last dim in f32.
pub fn plain_norm(xs: &Tensor, eps: f64) -> Result<Tensor> {
    let xs = xs.to_dtype(DType::F32)?;
    let mean = xs.mean_keepdim(D::Minus1)?;
    let centered = xs.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
    centered.broadcast_div(&(var + eps)?.sqrt()?)
}

/// The tokens' grouping for one attention pass: all together, or by window.
pub(crate) struct Groups {
    // window order: `order[i]` is the token at packed row i, `cu` the windows' bounds; none for one group
    windows: Option<(Tensor, Tensor, Vec<u32>)>,
}

impl Groups {
    fn full() -> Self {
        Self { windows: None }
    }

    /// `calc_window_partition`: tokens grouped by the window their shifted coordinate falls in.
    fn windowed(coords: &[[i32; 3]], window: i32, shift: i32, device: &Device) -> Result<Self> {
        let key = |c: &[i32; 3]| c.map(|v| (v + shift).div_euclid(window));
        let mut order: Vec<u32> = (0..coords.len() as u32).collect();
        order.sort_by_key(|&i| key(&coords[i as usize]));
        let mut cu = vec![0u32];
        for (i, pair) in order.windows(2).enumerate() {
            if key(&coords[pair[0] as usize]) != key(&coords[pair[1] as usize]) {
                cu.push(i as u32 + 1);
            }
        }
        cu.push(order.len() as u32);
        let mut back = vec![0u32; order.len()];
        for (i, &o) in order.iter().enumerate() {
            back[o as usize] = i as u32;
        }
        let n = order.len();
        Ok(Self {
            windows: Some((
                Tensor::from_vec(order, n, device)?,
                Tensor::from_vec(back, n, device)?,
                cu,
            )),
        })
    }
}

/// `[n, heads, d]` attention within packed groups `cu` (bounds over dim 0): one varlen pass on CUDA in 16-bit, one
/// attention per group otherwise.
pub(crate) fn varlen(q: &Tensor, k: &Tensor, v: &Tensor, cu: &[u32]) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if q.device().is_cuda() && q.dtype() != DType::F32 {
        let max_len = cu
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0);
        let cu_t = Tensor::from_slice(cu, cu.len(), q.device())?;
        let seqs = inference_fattn::Packed {
            cu_seqlens: &cu_t,
            max_len,
        };
        let opts = inference_fattn::FattnOptions {
            scale: 1.0 / (q.dim(D::Minus1)? as f32).sqrt(),
            f32_values: true,
            ..Default::default()
        };
        let (q, k, v) = (q.contiguous()?, k.contiguous()?, v.contiguous()?);
        if inference_fattn::supported_varlen(&q, &k, &v, &seqs, &seqs, &opts)? {
            return inference_fattn::flash_attn_varlen(&q, &k, &v, &seqs, &seqs, &opts);
        }
    }
    let parts = cu
        .windows(2)
        .map(|w| {
            let (s, n) = (w[0] as usize, (w[1] - w[0]) as usize);
            let heads = |x: &Tensor| x.narrow(0, s, n)?.transpose(0, 1)?.unsqueeze(0);
            sdpa(&heads(q)?, &heads(k)?, &heads(v)?)?
                .squeeze(0)?
                .transpose(0, 1)
        })
        .collect::<Result<Vec<_>>>()?;
    Tensor::cat(&parts, 0)
}

struct Mlp {
    fc1: AmpLinear,
    fc2: AmpLinear,
}

impl Mlp {
    fn load(c: usize, hidden: usize, dtype: DType, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: AmpLinear::load(c, hidden, true, dtype, vb.pp(0))?,
            fc2: AmpLinear::load(hidden, c, true, dtype, vb.pp(2))?,
        })
    }

    /// tanh-GELU between the two
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(xs)?.gelu()?)
    }
}

struct Block {
    qkv: AmpLinear,
    self_out: AmpLinear,
    cross: Option<(AmpLinear, AmpLinear, AmpLinear)>,
    mlp: Mlp,
    heads: usize,
}

impl Block {
    fn load(spec: &StackSpec, dtype: DType, vb: VarBuilder) -> Result<Self> {
        let c = spec.channels;
        let cross = spec.context.is_some();
        let sa = vb.pp(if cross { "self_attn" } else { "attn" });
        Ok(Self {
            qkv: AmpLinear::load(c, 3 * c, true, dtype, sa.pp("to_qkv"))?,
            self_out: AmpLinear::load(c, c, true, dtype, sa.pp("to_out"))?,
            cross: spec
                .context
                .map(|ctx| {
                    let ca = vb.pp("cross_attn");
                    Ok::<_, inference_tensor::Error>((
                        AmpLinear::load(c, c, true, dtype, ca.pp("to_q"))?,
                        AmpLinear::load(ctx, 2 * c, true, dtype, ca.pp("to_kv"))?,
                        AmpLinear::load(c, c, true, dtype, ca.pp("to_out"))?,
                    ))
                })
                .transpose()?,
            mlp: Mlp::load(c, MLP_RATIO * c, dtype, vb.pp("mlp").pp("mlp"))?,
            heads: spec.heads,
        })
    }

    // `[n, c]` f32 stream; `groups` partitions the tokens for self-attention
    fn forward(
        &self,
        xs: &Tensor,
        rope: Option<&Rope>,
        groups: &Groups,
        context: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (n, c) = xs.dims2()?;
        let qkv = self.qkv.forward(&plain_norm(xs, NORM_EPS)?)?.reshape((
            n,
            3,
            self.heads,
            c / self.heads,
        ))?;
        let part = |i: usize| -> Result<Tensor> {
            let x = qkv.narrow(1, i, 1)?.squeeze(1)?;
            match (rope, i < 2) {
                (Some(rope), true) => rope.apply(&x.unsqueeze(0)?)?.squeeze(0),
                _ => Ok(x),
            }
        };
        let (q, k, v) = (part(0)?, part(1)?, part(2)?);
        let attn = match &groups.windows {
            None => {
                let heads = |x: &Tensor| x.transpose(0, 1)?.unsqueeze(0);
                sdpa(&heads(&q)?, &heads(&k)?, &heads(&v)?)?
                    .squeeze(0)?
                    .transpose(0, 1)?
            }
            Some((order, back, cu)) => {
                let packed = |x: &Tensor| x.contiguous()?.index_select(order, 0);
                varlen(&packed(&q)?, &packed(&k)?, &packed(&v)?, cu)?.index_select(back, 0)?
            }
        };
        let attn = self.self_out.forward(&attn.reshape((n, c))?)?;
        let mut xs = (xs + attn.to_dtype(DType::F32)?)?;
        if let (Some((to_q, to_kv, to_out)), Some(context)) = (&self.cross, context) {
            let m = context.dim(0)?;
            let q = to_q
                .forward(&plain_norm(&xs, NORM_EPS)?)?
                .reshape((n, self.heads, ()))?;
            let kv = to_kv.forward(context)?.reshape((m, 2, self.heads, ()))?;
            let heads = |x: &Tensor| x.transpose(0, 1)?.unsqueeze(0);
            let attn = sdpa(
                &heads(&q)?,
                &heads(&kv.narrow(1, 0, 1)?.squeeze(1)?)?,
                &heads(&kv.narrow(1, 1, 1)?.squeeze(1)?)?,
            )?
            .squeeze(0)?
            .transpose(0, 1)?
            .reshape((n, c))?;
            xs = (xs + to_out.forward(&attn)?.to_dtype(DType::F32)?)?;
        }
        let mlp = self.mlp.forward(&plain_norm(&xs, NORM_EPS)?)?;
        xs + mlp.to_dtype(DType::F32)?
    }
}

/// `SparseTransformerBase` or, with a context, `SparseTransformerCrossBase`: an input layer, then the blocks.
pub struct Stack {
    spec: StackSpec,
    input: AmpLinear,
    blocks: Vec<Block>,
    device: Device,
}

impl Stack {
    pub fn load(spec: StackSpec, dtype: DType, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            spec,
            input: AmpLinear::load(spec.input, spec.channels, true, dtype, vb.pp("input_layer"))?,
            blocks: (0..spec.blocks)
                .map(|i| Block::load(&spec, dtype, vb.pp("blocks").pp(i)))
                .collect::<Result<_>>()?,
            device: vb.device().clone(),
        })
    }

    /// The features `[n, channels]` (f32) of tokens `x` `[n, input]` at integer `coords`; a cross stack attends to
    /// `context` `[m, context]` whose tokens sit at `context_coords`.
    pub fn forward(
        &self,
        x: &Tensor,
        coords: &[[i32; 3]],
        context: Option<(&Tensor, &[[i32; 3]])>,
    ) -> Result<Tensor> {
        let position = |coords: &[[i32; 3]]| -> Vec<[f32; 3]> {
            coords.iter().map(|c| c.map(|v| v as f32)).collect()
        };
        let mut h = self.input.forward(x)?.to_dtype(DType::F32)?;
        let mut context = context
            .map(|(c, at)| Ok::<_, inference_tensor::Error>((c.to_dtype(DType::F32)?, at)))
            .transpose()?;
        let rope = match self.spec.positions {
            Positions::Rope => Some(Rope::new(
                &position(coords),
                self.spec.channels / self.spec.heads,
                &self.device,
            )?),
            Positions::Absolute => {
                h = h.broadcast_add(&ape(&position(coords), self.spec.channels, &self.device)?)?;
                if let Some((ctx, at)) = context.as_mut() {
                    *ctx = ctx.broadcast_add(&ape(&position(at), ctx.dim(1)?, &self.device)?)?;
                }
                None
            }
        };
        let context = context.map(|(c, _)| c);
        // windowed stacks shift their windows by half every other block
        let groups = match self.spec.attn {
            AttnMode::Full => vec![Groups::full()],
            AttnMode::Swin { window } => vec![
                Groups::windowed(coords, window, 0, &self.device)?,
                Groups::windowed(coords, window, window / 2, &self.device)?,
            ],
        };
        for (i, block) in self.blocks.iter().enumerate() {
            let groups = &groups[i % groups.len()];
            h = block.forward(&h, rope.as_ref(), groups, context.as_ref())?;
        }
        Ok(h)
    }
}
