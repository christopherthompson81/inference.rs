#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Inkling 2's vision tower: a ViT with a CLS row, a learned position grid and 2D RoPE, then a 2 x 2 merge projected to
//! the text model's width (vLLM `inkling2/nvidia/vision.py`, llama.cpp `inkling2v`).

use crate::attention::{AttentionMask, FlashParams, SdpaParams};
use crate::layers::{Sdpa, layer_norm, linear, linear_no_bias};
use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{LayerNorm, Linear, Module};
use inference_tensor::{DType, Device, Result, Tensor};
use serde::Serialize;

const ROPE_THETA: f32 = 10000.0;
const CHANNELS: usize = 3;
// the canvas past the picture: a raw value, normalized like the pixels
const PAD_PIXEL: f32 = -1.0;
#[allow(clippy::excessive_precision)]
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
// the processor's float32-rounded CLIP std divides, the model's own multiplies back (vLLM `gather_patches`)
#[allow(clippy::excessive_precision)]
const PROCESSOR_CLIP_STD: [f32; 3] = [0.26862954, 0.2613026, 0.2757771];
#[allow(clippy::excessive_precision)]
const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

crate::serde_default_fn!(usize, default_hidden_size, 1024);
crate::serde_default_fn!(usize, default_num_hidden_layers, 23);
crate::serde_default_fn!(usize, default_num_attention_heads, 16);
crate::serde_default_fn!(f64, default_mlp_ratio, 4.0);
crate::serde_default_fn!(usize, default_native_patch_size, 20);
crate::serde_default_fn!(usize, default_adapter_merge_size, 2);
crate::serde_default_fn!(usize, default_posemb_grid_size, 32);
crate::serde_default_fn!(usize, default_projection_hidden_dim, 4096);
crate::serde_default_fn!(usize, default_text_hidden_size, 2304);
crate::serde_default_fn!(f64, default_layer_norm_eps, 1e-5);

#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct VisionConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f64,
    #[serde(default = "default_native_patch_size")]
    pub native_patch_size: usize,
    #[serde(default = "default_adapter_merge_size")]
    pub adapter_merge_size: usize,
    #[serde(default = "default_posemb_grid_size")]
    pub posemb_grid_size: usize,
    #[serde(default = "default_projection_hidden_dim")]
    pub projection_hidden_dim: usize,
    #[serde(default)]
    pub projector_bias: bool,
    #[serde(default = "default_text_hidden_size")]
    pub text_hidden_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
}

impl VisionConfig {
    pub fn intermediate_size(&self) -> usize {
        (self.hidden_size as f64 * self.mlp_ratio) as usize
    }
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    /// Pixels of a cell's side: one text row per cell.
    pub fn cell_size(&self) -> usize {
        self.native_patch_size * self.adapter_merge_size
    }
    pub fn patch_features(&self) -> usize {
        CHANNELS * self.native_patch_size * self.native_patch_size
    }
    /// Weight elements, for sizing the load.
    pub fn num_weights(&self) -> usize {
        let (d, f) = (self.hidden_size, self.intermediate_size());
        let layer = 4 * (d * d + d) + 2 * d * f + f + d + 6 * d;
        let grid = self.posemb_grid_size * self.posemb_grid_size;
        let merged = d * self.adapter_merge_size * self.adapter_merge_size;
        let projector = merged * self.projection_hidden_dim
            + self.projection_hidden_dim * self.text_hidden_size;
        self.patch_features() * d + (grid + 4) * d + self.num_hidden_layers * layer + projector
    }
}

// to the nearest bf16, ties to even (finite values only)
fn round_bf16(v: f32) -> f32 {
    let bits = v.to_bits();
    f32::from_bits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

/// A picture's native patches: the grid (`gh`, `gw`, even) and `[gh * gw, 1200]` values in [-1, 1] ordered
/// (py, px, c), the picture at the top left of a canvas of whole cells.
pub struct Patches {
    pub values: Vec<f32>,
    pub gh: usize,
    pub gw: usize,
}

impl VisionConfig {
    /// Cells across and down for a `width` x `height` picture.
    pub fn cells(&self, width: usize, height: usize) -> (usize, usize) {
        let c = self.cell_size();
        (width.div_ceil(c), height.div_ceil(c))
    }

    /// `rgb` is `height` rows of `width` RGB bytes.
    pub fn patches(&self, rgb: &[u8], width: usize, height: usize) -> Patches {
        let p = self.native_patch_size;
        let (cols, rows) = self.cells(width, height);
        let (gw, gh) = (
            cols * self.adapter_merge_size,
            rows * self.adapter_merge_size,
        );
        let features = self.patch_features();
        let mut values = vec![0f32; gh * gw * features];
        for (i, patch) in values.chunks_exact_mut(features).enumerate() {
            let (py0, px0) = ((i / gw) * p, (i % gw) * p);
            for (j, v) in patch.iter_mut().enumerate() {
                let (y, x, c) = (
                    py0 + j / (p * CHANNELS),
                    px0 + (j / CHANNELS) % p,
                    j % CHANNELS,
                );
                let raw = if y < height && x < width {
                    rgb[(y * width + x) * CHANNELS + c] as f32
                } else {
                    PAD_PIXEL
                };
                let clip = round_bf16((raw / 255.0 - CLIP_MEAN[c]) / PROCESSOR_CLIP_STD[c]);
                *v = 2.0 * (clip * CLIP_STD[c] + CLIP_MEAN[c]) - 1.0;
            }
        }
        Patches { values, gh, gw }
    }
}

// cos/sin for the CLS row then the patches: pairs [0, quarter) turn by x + 1, the rest by y + 1, CLS at 0
fn rope_tables(gh: usize, gw: usize, head_dim: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let pairs = head_dim / 2;
    let quarter = pairs / 2;
    let inv_freq: Vec<f32> = (0..quarter)
        .map(|j| ROPE_THETA.powf(-(j as f32) / quarter as f32))
        .collect();
    let rows = gh * gw + 1;
    let mut cos = vec![1f32; rows * pairs];
    let mut sin = vec![0f32; rows * pairs];
    for p in 0..gh * gw {
        let (y, x) = ((p / gw + 1) as f32, (p % gw + 1) as f32);
        let base = (p + 1) * pairs;
        for (j, f) in inv_freq.iter().enumerate() {
            let (ax, ay) = (x * f, y * f);
            cos[base + j] = ax.cos();
            sin[base + j] = ax.sin();
            cos[base + quarter + j] = ay.cos();
            sin[base + quarter + j] = ay.sin();
        }
    }
    Ok((
        Tensor::from_vec(cos, (rows, pairs), dev)?,
        Tensor::from_vec(sin, (rows, pairs), dev)?,
    ))
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(vb: ShardedVarBuilder, cfg: &VisionConfig) -> Result<Self> {
        let d = cfg.hidden_size;
        Ok(Self {
            q_proj: linear(d, d, vb.pp("q_proj"))?,
            k_proj: linear(d, d, vb.pp("k_proj"))?,
            v_proj: linear(d, d, vb.pp("v_proj"))?,
            out_proj: linear(d, d, vb.pp("out_proj"))?,
            heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let n = x.dim(0)?;
        let heads = |t: Tensor| {
            t.reshape((1, n, self.heads, self.head_dim))?
                .transpose(1, 2)
        };
        let (q, k) = inference_quant::rotary::apply_rotary_qk_preselected(
            &heads(self.q_proj.forward(x)?)?,
            &heads(self.k_proj.forward(x)?)?,
            cos,
            sin,
            false,
        )?;
        let v = heads(self.v_proj.forward(x)?)?;
        let params = SdpaParams {
            n_kv_groups: 1,
            sliding_window: None,
            softcap: None,
            softmax_scale: (self.head_dim as f32).powf(-0.5),
            sinks: None,
            chunk: None,
        };
        let out = Sdpa.run_attention(
            &q,
            &k,
            &v,
            &AttentionMask::None,
            Some(&FlashParams::empty(false)),
            &params,
        )?;
        let out = out
            .squeeze(0)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((n, self.heads * self.head_dim))?;
        self.out_proj.forward(&out)
    }
}

struct Layer {
    ln_1: LayerNorm,
    self_attn: Attention,
    ln_2: LayerNorm,
    c_fc: Linear,
    c_proj: Linear,
    ls_1: Tensor,
    ls_2: Tensor,
}

impl Layer {
    // `f32` is `vb` loading f32: the norms and layer scales work on the f32 residual
    fn load(vb: ShardedVarBuilder, f32: ShardedVarBuilder, cfg: &VisionConfig) -> Result<Self> {
        let (d, f, eps) = (cfg.hidden_size, cfg.intermediate_size(), cfg.layer_norm_eps);
        let mlp = vb.pp("mlp");
        Ok(Self {
            ln_1: layer_norm(d, eps, f32.pp("ln_1"))?,
            self_attn: Attention::load(vb.pp("self_attn"), cfg)?,
            ln_2: layer_norm(d, eps, f32.pp("ln_2"))?,
            c_fc: linear(d, f, mlp.pp("c_fc"))?,
            c_proj: linear(f, d, mlp.pp("c_proj"))?,
            ls_1: f32.pp("ls_1").get(d, "gamma")?,
            ls_2: f32.pp("ls_2").get(d, "gamma")?,
        })
    }

    // the residual stays f32; the norms hand the projections the weights' dtype
    fn forward(&self, h: &Tensor, cos: &Tensor, sin: &Tensor, dtype: DType) -> Result<Tensor> {
        let a = self
            .self_attn
            .forward(&self.ln_1.forward(h)?.to_dtype(dtype)?, cos, sin)?;
        let h = (h + a.to_dtype(DType::F32)?.broadcast_mul(&self.ls_1)?)?;
        let m = self
            .c_fc
            .forward(&self.ln_2.forward(&h)?.to_dtype(dtype)?)?;
        let m = self.c_proj.forward(&m.gelu_erf()?)?;
        &h + m.to_dtype(DType::F32)?.broadcast_mul(&self.ls_2)?
    }
}

pub struct VisionTower {
    patch_embedding: Linear,
    position_grid: Tensor,
    cls: Tensor,
    ln_pre: LayerNorm,
    layers: Vec<Layer>,
    proj_in: Linear,
    proj_out: Linear,
    cfg: VisionConfig,
    dtype: DType,
}

impl VisionTower {
    /// `vb` at `model.visual`.
    pub fn new(cfg: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let d = cfg.hidden_size;
        let dtype = vb.dtype();
        let f32_vb = vb.clone().set_dtype(DType::F32);
        let backbone = f32_vb.pp("backbone");
        let emb = backbone.pp("embeddings");
        let g = cfg.posemb_grid_size;
        let cls = (emb.get(d, "class_embedding")? + emb.get(d, "cls_position_embedding")?)?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                Layer::load(
                    vb.pp("backbone").pp("layers").pp(i),
                    backbone.pp("layers").pp(i),
                    cfg,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let merged = d * cfg.adapter_merge_size * cfg.adapter_merge_size;
        let projector = |i, o, name| -> Result<Linear> {
            if cfg.projector_bias {
                linear(i, o, vb.pp(name))
            } else {
                linear_no_bias(i, o, vb.pp(name))
            }
        };
        Ok(Self {
            patch_embedding: linear_no_bias(
                cfg.patch_features(),
                d,
                vb.pp("backbone").pp("embeddings").pp("patch_embedding"),
            )?,
            position_grid: emb.get((g * g, d), "position_embedding")?,
            cls: cls.reshape((1, d))?,
            ln_pre: layer_norm(d, cfg.layer_norm_eps, backbone.pp("ln_pre"))?,
            layers,
            proj_in: projector(merged, cfg.projection_hidden_dim, "proj_in")?,
            proj_out: projector(cfg.projection_hidden_dim, cfg.text_hidden_size, "proj_out")?,
            cfg: cfg.clone(),
            dtype,
        })
    }

    pub fn config(&self) -> &VisionConfig {
        &self.cfg
    }

    // the learned grid resized to the picture's patches (bilinear, half-pixel centers), as is at its own size
    fn positions(&self, gh: usize, gw: usize) -> Result<Tensor> {
        let (g, d) = (self.cfg.posemb_grid_size, self.cfg.hidden_size);
        if (gh, gw) == (g, g) {
            return Ok(self.position_grid.clone());
        }
        self.position_grid
            .reshape((1, g, g, d))?
            .permute((0, 3, 1, 2))?
            .contiguous()?
            .upsample_bilinear2d(gh, gw, false)?
            .permute((0, 2, 3, 1))?
            .contiguous()?
            .reshape((gh * gw, d))
    }

    /// One picture's text rows, `[cells, text_hidden_size]` in raster order. `patches` is `[gh * gw, 1200]` f32,
    /// normalized pixels ordered (py, px, c), the patches in raster order; `gh`, `gw` are even.
    pub fn forward(&self, patches: &Tensor, gh: usize, gw: usize) -> Result<Tensor> {
        let (d, m) = (self.cfg.hidden_size, self.cfg.adapter_merge_size);
        let dev = patches.device();
        let x = self
            .patch_embedding
            .forward(&patches.to_dtype(self.dtype)?)?;
        let x = (x.to_dtype(DType::F32)? + self.positions(gh, gw)?)?;
        let x = Tensor::cat(&[&self.cls, &x], 0)?;
        let mut h = self.ln_pre.forward(&x)?;
        let (cos, sin) = rope_tables(gh, gw, self.cfg.head_dim(), dev)?;
        let (cos, sin) = (cos.to_dtype(self.dtype)?, sin.to_dtype(self.dtype)?);
        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin, self.dtype)?;
        }
        let cells = h
            .narrow(0, 1, gh * gw)?
            .reshape((gh / m, m, gw / m, m, d))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?
            .reshape((gh * gw / (m * m), m * m * d))?
            .to_dtype(self.dtype)?;
        self.proj_out
            .forward(&self.proj_in.forward(&cells)?.gelu_erf()?)
    }
}
