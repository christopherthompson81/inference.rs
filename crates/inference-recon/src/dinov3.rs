//! DINOv3 ViTs as Hugging Face's `DINOv3ViTModel` lays them out: a class token and registers before the patches, 2D
//! RoPE (half-split, on the patches only) from each patch center normalized to [-1, 1], LayerScale on both branches.

use inference_nn::attention::{AttentionMask, FlashParams, Sdpa, SdpaParams};
use inference_tensor::nn::{LayerNorm, Module, VarBuilder};
use inference_tensor::{D, DType, Device, Result, Tensor};

use crate::worldmirror::layers::AmpLinear;

const EPS: f64 = 1e-5;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Dinov3Spec {
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub mlp: usize,
    pub patch: usize,
    pub registers: usize,
    pub rope_theta: f64,
    pub dtype: DType,
}

impl Dinov3Spec {
    /// `facebook/dinov3-vitl16-pretrain-lvd1689m`
    pub(crate) fn vitl16(dtype: DType) -> Self {
        Self {
            dim: 1024,
            depth: 24,
            heads: 16,
            mlp: 4096,
            patch: 16,
            registers: 4,
            rope_theta: 100.0,
            dtype,
        }
    }
}

struct Layer {
    norm1: LayerNorm,
    q: AmpLinear,
    k: AmpLinear,
    v: AmpLinear,
    o: AmpLinear,
    scale1: Tensor,
    norm2: LayerNorm,
    up: AmpLinear,
    down: AmpLinear,
    scale2: Tensor,
    heads: usize,
}

impl Layer {
    fn load(spec: Dinov3Spec, vb: VarBuilder) -> Result<Self> {
        let (c, dtype) = (spec.dim, spec.dtype);
        let attn = vb.pp("attention");
        Ok(Self {
            norm1: inference_tensor::nn::layer_norm(c, EPS, vb.pp("norm1"))?,
            q: AmpLinear::load(c, c, true, dtype, attn.pp("q_proj"))?,
            k: AmpLinear::load(c, c, false, dtype, attn.pp("k_proj"))?,
            v: AmpLinear::load(c, c, true, dtype, attn.pp("v_proj"))?,
            o: AmpLinear::load(c, c, true, dtype, attn.pp("o_proj"))?,
            scale1: vb.pp("layer_scale1").get(c, "lambda1")?,
            norm2: inference_tensor::nn::layer_norm(c, EPS, vb.pp("norm2"))?,
            up: AmpLinear::load(c, spec.mlp, true, dtype, vb.pp("mlp").pp("up_proj"))?,
            down: AmpLinear::load(spec.mlp, c, true, dtype, vb.pp("mlp").pp("down_proj"))?,
            scale2: vb.pp("layer_scale2").get(c, "lambda1")?,
            heads: spec.heads,
        })
    }

    fn forward(&self, xs: &Tensor, rope: &(Tensor, Tensor), prefix: usize) -> Result<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let h = self.norm1.forward(xs)?;
        let heads = |t: Tensor| t.reshape((b, n, self.heads, ()))?.transpose(1, 2);
        let q = rotate(&heads(self.q.forward(&h)?)?, rope, prefix)?;
        let k = rotate(&heads(self.k.forward(&h)?)?, rope, prefix)?;
        let v = heads(self.v.forward(&h)?)?;
        let params = SdpaParams {
            n_kv_groups: 1,
            softcap: None,
            softmax_scale: 1.0 / ((c / self.heads) as f32).sqrt(),
            sliding_window: None,
            sinks: None,
            chunk: None,
        };
        let attn = Sdpa.run_attention(
            &q.contiguous()?,
            &k.contiguous()?,
            &v.contiguous()?,
            &AttentionMask::None,
            Some(&FlashParams::empty(false)),
            &params,
        )?;
        let attn = self.o.forward(&attn.transpose(1, 2)?.reshape((b, n, c))?)?;
        let xs = (xs + attn.to_dtype(DType::F32)?.broadcast_mul(&self.scale1)?)?;
        let h = self.norm2.forward(&xs)?;
        let mlp = self.down.forward(&self.up.forward(&h)?.gelu_erf()?)?;
        xs + mlp.to_dtype(DType::F32)?.broadcast_mul(&self.scale2)?
    }
}

// `x * cos + rotate_half(x) * sin` on the patch tokens of `[b, heads, n, head_dim]`, the prefix tokens left alone
fn rotate(xs: &Tensor, (cos, sin): &(Tensor, Tensor), prefix: usize) -> Result<Tensor> {
    let (_, _, n, d) = xs.dims4()?;
    let patches = xs.narrow(2, prefix, n - prefix)?;
    let dtype = patches.dtype();
    let patches = patches.to_dtype(DType::F32)?;
    let half = d / 2;
    let rotated = Tensor::cat(
        &[
            patches.narrow(D::Minus1, half, half)?.neg()?,
            patches.narrow(D::Minus1, 0, half)?,
        ],
        D::Minus1,
    )?;
    let patches = (patches.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?)?.to_dtype(dtype)?;
    Tensor::cat(&[xs.narrow(2, 0, prefix)?, patches], 2)
}

pub(crate) struct Dinov3 {
    patch_weight: Tensor,
    patch_bias: Tensor,
    cls_token: Tensor,
    register_tokens: Tensor,
    layers: Vec<Layer>,
    spec: Dinov3Spec,
}

impl Dinov3 {
    pub(crate) fn load(spec: Dinov3Spec, vb: VarBuilder) -> Result<Self> {
        let (c, p) = (spec.dim, spec.patch);
        let emb = vb.pp("embeddings");
        Ok(Self {
            patch_weight: emb
                .pp("patch_embeddings")
                .get((c, 3, p, p), "weight")?
                .to_dtype(spec.dtype)?,
            patch_bias: emb
                .pp("patch_embeddings")
                .get(c, "bias")?
                .to_dtype(spec.dtype)?,
            cls_token: emb.get((1, 1, c), "cls_token")?,
            register_tokens: emb.get((1, spec.registers, c), "register_tokens")?,
            layers: (0..spec.depth)
                .map(|i| Layer::load(spec, vb.pp("layer").pp(i)))
                .collect::<Result<_>>()?,
            spec,
        })
    }

    // cos and sin `[patches, head_dim]` for an `hp x wp` patch grid
    fn rope(&self, hp: usize, wp: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        let head_dim = self.spec.dim / self.spec.heads;
        let quarter = head_dim / 4;
        let inv: Vec<f32> = (0..quarter)
            .map(|i| 1.0 / self.spec.rope_theta.powf(i as f64 * 4.0 / head_dim as f64) as f32)
            .collect();
        let two_pi = 2.0 * std::f32::consts::PI;
        let mut cos = Vec::with_capacity(hp * wp * head_dim);
        let mut sin = Vec::with_capacity(hp * wp * head_dim);
        for y in 0..hp {
            for x in 0..wp {
                let center = |i: usize, n: usize| 2.0 * ((i as f32 + 0.5) / n as f32) - 1.0;
                let angles: Vec<f32> = [center(y, hp), center(x, wp)]
                    .iter()
                    .flat_map(|&c| inv.iter().map(move |&f| two_pi * c * f))
                    .collect();
                for _ in 0..2 {
                    cos.extend(angles.iter().map(|a| a.cos()));
                    sin.extend(angles.iter().map(|a| a.sin()));
                }
            }
        }
        Ok((
            Tensor::from_vec(cos, (hp * wp, head_dim), device)?,
            Tensor::from_vec(sin, (hp * wp, head_dim), device)?,
        ))
    }

    /// Every token after the last layer, class token and registers first, through a LayerNorm with no affine
    /// (`[frames, 1 + registers + patches, dim]`, f32), for normalized images `[frames, 3, H, W]`.
    pub(crate) fn forward(&self, images: &Tensor) -> Result<Tensor> {
        let (frames, _, h, w) = images.dims4()?;
        let p = self.spec.patch;
        let (hp, wp) = (h / p, w / p);
        let c = self.spec.dim;
        let patches = images
            .to_dtype(self.spec.dtype)?
            .conv2d(&self.patch_weight, 0, p, 1, 1)?
            .broadcast_add(&self.patch_bias.reshape((1, (), 1, 1))?)?
            .flatten_from(2)?
            .transpose(1, 2)?
            .to_dtype(DType::F32)?;
        let prefix = 1 + self.spec.registers;
        let mut xs = Tensor::cat(
            &[
                self.cls_token.broadcast_as((frames, 1, c))?,
                self.register_tokens
                    .broadcast_as((frames, self.spec.registers, c))?,
                patches,
            ],
            1,
        )?;
        let rope = self.rope(hp, wp, images.device())?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &rope, prefix)?;
        }
        let ones = Tensor::ones(c, DType::F32, xs.device())?;
        let zeros = Tensor::zeros(c, DType::F32, xs.device())?;
        LayerNorm::new(ones, zeros, EPS).forward(&xs)
    }
}
