//! SAM's ViT image encoder (segment-anything `ImageEncoderViT`) as DeepSeek-OCR's vision front end: windowed blocks
//! with decomposed relative positions beside a few global ones, the conv neck, then two stride-2 convs.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use inference_quant::{QuantMethod, ShardedVarBuilder};
use inference_tensor::nn::{Conv2d, Conv2dConfig, LayerNorm, Module};
use inference_tensor::{D, DType, Device, Result, Tensor};

use crate::attention::{AttentionMask, FlashParams, SdpaParams};
use crate::layers::{Sdpa, conv2d, conv2d_no_bias, layer_norm};
use crate::utils::unvarbuilder::UnVarBuilder;

const SAM_EPS: f64 = 1e-6;
const MLP_RATIO: usize = 4;
const NECK_KERNEL: usize = 3;
const DOWNSAMPLE_STRIDE: usize = 2;

#[derive(Clone, Debug)]
pub struct SamConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub window_size: usize,
    pub global_attn_indexes: Vec<usize>,
    /// The neck's width (`prompt_embed_dim`).
    pub out_chans: usize,
    /// The two stride-2 convs' widths after the neck (`net_2`, `net_3`).
    pub downsample_channels: [usize; 2],
}

impl SamConfig {
    /// SAM ViT-B as DeepSeek-OCR builds it (`build_sam_vit_b`).
    pub fn vit_b(global_attn_indexes: Vec<usize>, downsample_channels: [usize; 2]) -> Self {
        Self {
            image_size: 1024,
            patch_size: 16,
            embed_dim: 768,
            depth: 12,
            num_heads: 12,
            window_size: 14,
            global_attn_indexes,
            out_chans: 256,
            downsample_channels,
        }
    }

    fn grid(&self) -> usize {
        self.image_size / self.patch_size
    }

    pub fn num_elems(&self) -> usize {
        let dim = self.embed_dim;
        let head_dim = dim / self.num_heads;
        let block = 4 * dim
            + dim * 3 * dim
            + 3 * dim
            + dim * dim
            + dim
            + 2 * dim * MLP_RATIO * dim
            + MLP_RATIO * dim
            + dim;
        let rel_pos = (0..self.depth)
            .map(|layer| 2 * 2 * (self.attention_size(layer) * 2 - 1) * head_dim)
            .sum::<usize>();
        let patch = 3 * dim * self.patch_size * self.patch_size + dim;
        let pos = self.grid() * self.grid() * dim;
        let [mid, out] = self.downsample_channels;
        let neck = dim * self.out_chans
            + self.out_chans * self.out_chans * NECK_KERNEL * NECK_KERNEL
            + 4 * self.out_chans;
        let downsample = (self.out_chans * mid + mid * out) * NECK_KERNEL * NECK_KERNEL;
        patch + pos + block * self.depth + rel_pos + neck + downsample
    }

    fn attention_size(&self, layer: usize) -> usize {
        if self.global_attn_indexes.contains(&layer) {
            self.grid()
        } else {
            self.window_size
        }
    }
}

// LayerNorm over the channels of an NCHW map
fn layer_norm_2d(norm: &LayerNorm, xs: &Tensor) -> Result<Tensor> {
    norm.forward(&xs.permute((0, 2, 3, 1))?)?
        .permute((0, 3, 1, 2))
}

// `F.interpolate(mode="linear", align_corners=False)` of a (L, C) table to `size` rows, once per size on the host
fn resize_rel_pos(table: &Tensor, size: usize) -> Result<Tensor> {
    let (len, channels) = table.dims2()?;
    let rows = table.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let scale = len as f64 / size as f64;
    let mut out = Vec::with_capacity(size * channels);
    for i in 0..size {
        let src = ((i as f64 + 0.5) * scale - 0.5).max(0.0);
        let lo = (src.floor() as usize).min(len - 1);
        let hi = (lo + 1).min(len - 1);
        let t = (src - lo as f64) as f32;
        out.extend(rows[lo].iter().zip(&rows[hi]).map(|(a, b)| a + (b - a) * t));
    }
    Tensor::from_vec(out, (size, channels), &Device::Cpu)?
        .to_dtype(table.dtype())?
        .to_device(table.device())
}

// The (size, size, C) table of embeddings for every query/key offset along one axis
fn rel_pos_table(table: &Tensor, size: usize) -> Result<Tensor> {
    let max_rel = 2 * size - 1;
    let table = if table.dim(0)? == max_rel {
        table.clone()
    } else {
        resize_rel_pos(table, max_rel)?
    };
    let indices: Vec<u32> = (0..size)
        .flat_map(|q| (0..size).map(move |k| (q + size - 1 - k) as u32))
        .collect();
    let indices = Tensor::from_vec(indices, size * size, table.device())?;
    table
        .index_select(&indices, 0)?
        .reshape((size, size, table.dim(1)?))
}

struct SamAttention {
    qkv: Arc<dyn QuantMethod>,
    proj: Arc<dyn QuantMethod>,
    rel_pos_h: Tensor,
    rel_pos_w: Tensor,
    num_heads: usize,
    head_dim: usize,
    // per (H, W), the gathered (H, H, C) and (W, W, C) tables
    tables: Mutex<HashMap<(usize, usize), (Tensor, Tensor)>>,
}

impl SamAttention {
    fn load(vb: ShardedVarBuilder, cfg: &SamConfig, size: usize) -> Result<Self> {
        let dim = cfg.embed_dim;
        let head_dim = dim / cfg.num_heads;
        Ok(Self {
            qkv: inference_quant::linear(dim, 3 * dim, &None, vb.pp("qkv"))?,
            proj: inference_quant::linear(dim, dim, &None, vb.pp("proj"))?,
            rel_pos_h: vb.get((2 * size - 1, head_dim), "rel_pos_h")?,
            rel_pos_w: vb.get((2 * size - 1, head_dim), "rel_pos_w")?,
            num_heads: cfg.num_heads,
            head_dim,
            tables: Mutex::new(HashMap::new()),
        })
    }

    fn tables(&self, h: usize, w: usize) -> Result<(Tensor, Tensor)> {
        let mut tables = self.tables.lock().expect("SAM rel-pos tables poisoned");
        if let Some(entry) = tables.get(&(h, w)) {
            return Ok(entry.clone());
        }
        let entry = (
            rel_pos_table(&self.rel_pos_h, h)?,
            rel_pos_table(&self.rel_pos_w, w)?,
        );
        tables.insert((h, w), entry.clone());
        Ok(entry)
    }

    // xs: (B, H, W, C)
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, h, w, dim) = xs.dims4()?;
        let (heads, hd) = (self.num_heads, self.head_dim);
        let qkv = self
            .qkv
            .forward(&xs.reshape((b, h * w, dim))?)?
            .reshape((b, h * w, 3, heads, hd))?
            .permute((2, 0, 3, 1, 4))?;
        let q = qkv.get(0)?.contiguous()?;
        let k = qkv.get(1)?.contiguous()?;
        let v = qkv.get(2)?.contiguous()?;

        // decomposed relative positions from the unscaled queries: bias[(qh, qw), (kh, kw)] = q.Rh[qh, kh] + q.Rw[qw, kw]
        let (rh, rw) = self.tables(h, w)?;
        let rq = q.reshape((b * heads, h, w, hd))?;
        let rel_h = rq
            .permute((1, 0, 2, 3))?
            .reshape((h, b * heads * w, hd))?
            .matmul(&rh.transpose(1, 2)?.contiguous()?)?
            .reshape((h, b * heads, w, h))?
            .permute((1, 0, 2, 3))?;
        let rel_w = rq
            .permute((2, 0, 1, 3))?
            .reshape((w, b * heads * h, hd))?
            .matmul(&rw.transpose(1, 2)?.contiguous()?)?
            .reshape((w, b * heads, h, w))?
            .permute((1, 2, 0, 3))?;
        let bias = rel_h
            .unsqueeze(D::Minus1)?
            .broadcast_add(&rel_w.unsqueeze(3)?)?
            .reshape((b, heads, h * w, h * w))?;

        let params = SdpaParams {
            n_kv_groups: 1,
            sliding_window: None,
            softcap: None,
            softmax_scale: (hd as f32).powf(-0.5),
            sinks: None,
            chunk: None,
        };
        let out = Sdpa.run_attention(
            &q,
            &k,
            &v,
            &AttentionMask::Custom(bias),
            Some(&FlashParams::empty(false)),
            &params,
        )?;
        let out = out.transpose(1, 2)?.reshape((b, h * w, dim))?;
        self.proj.forward(&out)?.reshape((b, h, w, dim))
    }
}

// (B, H, W, C) to (B * windows, window, window, C), zero-padding H and W up to the window
fn window_partition(xs: &Tensor, window: usize) -> Result<(Tensor, usize, usize)> {
    let (b, h, w, c) = xs.dims4()?;
    let (hp, wp) = (h.div_ceil(window) * window, w.div_ceil(window) * window);
    let xs = xs
        .pad_with_zeros(1, 0, hp - h)?
        .pad_with_zeros(2, 0, wp - w)?;
    let windows = xs
        .reshape((b, hp / window, window, wp / window, window, c))?
        .permute((0, 1, 3, 2, 4, 5))?
        .reshape((b * (hp / window) * (wp / window), window, window, c))?;
    Ok((windows, hp, wp))
}

fn window_unpartition(
    windows: &Tensor,
    window: usize,
    (hp, wp): (usize, usize),
    (h, w): (usize, usize),
) -> Result<Tensor> {
    let c = windows.dim(D::Minus1)?;
    let b = windows.dim(0)? / ((hp / window) * (wp / window));
    windows
        .reshape((b, hp / window, wp / window, window, window, c))?
        .permute((0, 1, 3, 2, 4, 5))?
        .reshape((b, hp, wp, c))?
        .narrow(1, 0, h)?
        .narrow(2, 0, w)
}

struct SamBlock {
    norm1: LayerNorm,
    attn: SamAttention,
    norm2: LayerNorm,
    lin1: Arc<dyn QuantMethod>,
    lin2: Arc<dyn QuantMethod>,
    window: Option<usize>,
}

impl SamBlock {
    fn load(vb: ShardedVarBuilder, cfg: &SamConfig, layer: usize) -> Result<Self> {
        let dim = cfg.embed_dim;
        let global = cfg.global_attn_indexes.contains(&layer);
        Ok(Self {
            norm1: layer_norm(dim, SAM_EPS, vb.pp("norm1"))?,
            attn: SamAttention::load(vb.pp("attn"), cfg, cfg.attention_size(layer))?,
            norm2: layer_norm(dim, SAM_EPS, vb.pp("norm2"))?,
            lin1: inference_quant::linear(dim, MLP_RATIO * dim, &None, vb.pp("mlp").pp("lin1"))?,
            lin2: inference_quant::linear(MLP_RATIO * dim, dim, &None, vb.pp("mlp").pp("lin2"))?,
            window: (!global).then_some(cfg.window_size),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, h, w, _) = xs.dims4()?;
        let normed = self.norm1.forward(xs)?;
        let attended = match self.window {
            Some(window) => {
                let (windows, hp, wp) = window_partition(&normed, window)?;
                window_unpartition(&self.attn.forward(&windows)?, window, (hp, wp), (h, w))?
            }
            None => self.attn.forward(&normed)?,
        };
        let xs = (xs + attended)?;
        let mlp = self
            .lin2
            .forward(&self.lin1.forward(&self.norm2.forward(&xs)?)?.gelu_erf()?)?;
        xs + mlp
    }
}

pub struct SamEncoder {
    patch_embed: Conv2d,
    pos_embed: Tensor,
    blocks: Vec<SamBlock>,
    neck_0: Conv2d,
    neck_1: LayerNorm,
    neck_2: Conv2d,
    neck_3: LayerNorm,
    net_2: Conv2d,
    net_3: Conv2d,
    // per patch grid side, the position embedding resampled to it
    pos_cache: Mutex<HashMap<usize, Tensor>>,
}

impl SamEncoder {
    pub fn load(vb: ShardedVarBuilder, cfg: &SamConfig) -> Result<Self> {
        let dim = cfg.embed_dim;
        let grid = cfg.grid();
        let patch_embed = conv2d(
            3,
            dim,
            cfg.patch_size,
            Conv2dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            },
            vb.pp("patch_embed").pp("proj"),
        )?;
        let blocks = (0..cfg.depth)
            .map(|layer| SamBlock::load(vb.pp("blocks").pp(layer), cfg, layer))
            .collect::<Result<Vec<_>>>()?;
        let neck = vb.pp("neck");
        let same = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let down = Conv2dConfig {
            padding: 1,
            stride: DOWNSAMPLE_STRIDE,
            ..Default::default()
        };
        let [mid, out] = cfg.downsample_channels;
        Ok(Self {
            patch_embed,
            pos_embed: vb.get((1, grid, grid, dim), "pos_embed")?,
            blocks,
            neck_0: conv2d_no_bias(dim, cfg.out_chans, 1, Conv2dConfig::default(), neck.pp(0))?,
            neck_1: layer_norm(cfg.out_chans, SAM_EPS, neck.pp(1))?,
            neck_2: conv2d_no_bias(cfg.out_chans, cfg.out_chans, NECK_KERNEL, same, neck.pp(2))?,
            neck_3: layer_norm(cfg.out_chans, SAM_EPS, neck.pp(3))?,
            net_2: conv2d_no_bias(cfg.out_chans, mid, NECK_KERNEL, down, vb.pp("net_2"))?,
            net_3: conv2d_no_bias(mid, out, NECK_KERNEL, down, vb.pp("net_3"))?,
            pos_cache: Mutex::new(HashMap::new()),
        })
    }

    // `get_abs_pos_sam`: the (1, G, G, C) grid resampled to `size` by antialiased bicubic
    fn pos_embed(&self, size: usize) -> Result<Tensor> {
        let grid = self.pos_embed.dim(1)?;
        if size == grid {
            return Ok(self.pos_embed.clone());
        }
        let mut cache = self.pos_cache.lock().expect("SAM position cache poisoned");
        if let Some(pos) = cache.get(&size) {
            return Ok(pos.clone());
        }
        let pos = resample_grid(&self.pos_embed.squeeze(0)?, size)?.unsqueeze(0)?;
        cache.insert(size, pos.clone());
        Ok(pos)
    }

    /// (B, 3, S, S) pixels to the (B, C, S / 64, S / 64) feature map.
    pub fn forward(&self, pixels: &Tensor) -> Result<Tensor> {
        let xs = self.patch_embed.forward(pixels)?.permute((0, 2, 3, 1))?;
        let mut xs = xs.broadcast_add(&self.pos_embed(xs.dim(1)?)?)?;
        for block in &self.blocks {
            xs = block.forward(&xs)?;
        }
        let xs = self
            .neck_0
            .forward(&xs.permute((0, 3, 1, 2))?.contiguous()?)?;
        let xs = layer_norm_2d(&self.neck_1, &xs)?;
        let xs = self.neck_2.forward(&xs.contiguous()?)?;
        let xs = layer_norm_2d(&self.neck_3, &xs)?;
        let xs = self.net_2.forward(&xs.contiguous()?)?;
        self.net_3.forward(&xs)
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("patch_embed").pp("proj").add(&self.patch_embed);
        uvb.add_tensor("pos_embed", self.pos_embed.clone());
        for (layer, block) in self.blocks.iter().enumerate() {
            let b = uvb.pp("blocks").pp(layer);
            b.pp("norm1").add(&block.norm1);
            b.pp("norm2").add(&block.norm2);
            let attn = b.pp("attn");
            attn.pp("qkv").add(&block.attn.qkv);
            attn.pp("proj").add(&block.attn.proj);
            attn.add_tensor("rel_pos_h", block.attn.rel_pos_h.clone());
            attn.add_tensor("rel_pos_w", block.attn.rel_pos_w.clone());
            b.pp("mlp").pp("lin1").add(&block.lin1);
            b.pp("mlp").pp("lin2").add(&block.lin2);
        }
        let neck = uvb.pp("neck");
        neck.pp(0).add(&self.neck_0);
        neck.pp(1).add(&self.neck_1);
        neck.pp(2).add(&self.neck_2);
        neck.pp(3).add(&self.neck_3);
        uvb.pp("net_2").add(&self.net_2);
        uvb.pp("net_3").add(&self.net_3);
        uvb.to_safetensors()
    }
}

/// A (G, G, C) grid of embeddings resampled to (S, S, C) as `F.interpolate(mode="bicubic", antialias=True)` does.
pub fn resample_grid(grid: &Tensor, size: usize) -> Result<Tensor> {
    let (side, _, channels) = grid.dims3()?;
    let table = grid
        .reshape((side * side, channels))?
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .to_vec2::<f32>()?;
    let taps = inference_vision::aa_bicubic_weights(side, size);
    let mut out = vec![0f32; size * size * channels];
    for (oy, row) in taps.iter().enumerate() {
        for (ox, col) in taps.iter().enumerate() {
            let cell = &mut out[(oy * size + ox) * channels..][..channels];
            for &(iy, wy) in row {
                for &(ix, wx) in col {
                    let weight = (wy * wx) as f32;
                    for (o, v) in cell.iter_mut().zip(&table[iy * side + ix]) {
                        *o += weight * v;
                    }
                }
            }
        }
    }
    Tensor::from_vec(out, (size, size, channels), &Device::Cpu)?
        .to_dtype(grid.dtype())?
        .to_device(grid.device())
}
