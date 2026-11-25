//! Hi3D's image condition: the subject framed on white, DINOv2-large's normed tokens beside CLIP ViT-L/14's pooled
//! embedding, each L2-normed, `[1, 2917, 2048]`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use image::{DynamicImage, RgbImage, RgbaImage};
use inference_nn::utils::varbuilder_utils::{DeviceForLoadTensor, from_mmaped_safetensors};
use inference_nn::vision::clip::{ClipConfig, ClipVisionTransformer};
use inference_tensor::nn::VarBuilder;
use inference_tensor::{D, DType, Device, Result, Tensor};
use inference_vision::PilFilter;

use crate::dinov2::{Dinov2, Dinov2Spec, PosInterp};

// the subject fills 1 / 1.2 of the square it is framed in
const SUBJECT_PADDING: f64 = 1.2;
const DINO_SIZE: usize = 756;
const CLIP_SIZE: u32 = 224;
const DINO_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const DINO_STD: [f32; 3] = [0.229, 0.224, 0.225];
const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];
// F.normalize's floor
const NORM_EPS: f64 = 1e-12;
const DINO_LAYERS: usize = 24;
const DINO_SPEC: Dinov2Spec = Dinov2Spec {
    dim: 1024,
    depth: DINO_LAYERS,
    heads: 16,
    patch: 14,
    registers: 0,
    // transformers resamples the table to the patch grid by size: the scale `grid / n` with no offset
    pos_interp: PosInterp::Bicubic { offset: 0.0 },
    dtype: DType::F32,
};

/// `prepare_image`: the alpha bounding box squared about its centre (PIL's crop pads past the edges with transparent
/// black), composited on white, then centred on a white square `1.2x` its side.
pub fn prepare(image: &DynamicImage) -> RgbImage {
    let rgba = image.to_rgba8();
    let subject = crop_subject(&rgba);
    let (w, h) = subject.dimensions();
    let side = (w.max(h) as f64 * SUBJECT_PADDING) as u32;
    let mut out = RgbImage::from_pixel(side, side, image::Rgb([255, 255, 255]));
    let (ox, oy) = ((side - w) / 2, (side - h) / 2);
    for (x, y, p) in subject.enumerate_pixels() {
        let a = p[3] as u32;
        // PIL's paste through a mask: (src * a + dst * (255 - a)) / 255, rounded
        let blend = |s: u8| ((s as u32 * a + 255 * (255 - a) + 127) / 255) as u8;
        out.put_pixel(
            x + ox,
            y + oy,
            image::Rgb([blend(p[0]), blend(p[1]), blend(p[2])]),
        );
    }
    out
}

fn crop_subject(rgba: &RgbaImage) -> RgbaImage {
    let (w, h) = rgba.dimensions();
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for (x, y, p) in rgba.enumerate_pixels() {
        if p[3] != 0 {
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
        }
    }
    if x1 == 0 {
        return rgba.clone();
    }
    let (bw, bh) = (x1 - x0, y1 - y0);
    let side = bw.max(bh);
    let left = x0 as i64 - ((side - bw) / 2) as i64;
    let top = y0 as i64 - ((side - bh) / 2) as i64;
    RgbaImage::from_fn(side, side, |x, y| {
        let (sx, sy) = (left + x as i64, top + y as i64);
        if sx < 0 || sy < 0 || sx >= w as i64 || sy >= h as i64 {
            image::Rgba([0, 0, 0, 0])
        } else {
            *rgba.get_pixel(sx as u32, sy as u32)
        }
    })
}

// `[1, 3, h, w]` in [0, 1] (f32)
fn pixels(image: &RgbImage, device: &Device) -> Result<Tensor> {
    let (w, h) = image.dimensions();
    let data: Vec<f32> = image.as_raw().iter().map(|&v| v as f32 / 255.0).collect();
    Tensor::from_vec(data, (1, h as usize, w as usize, 3), device)?.permute((0, 3, 1, 2))
}

fn normalize(x: &Tensor, mean: [f32; 3], std: [f32; 3]) -> Result<Tensor> {
    let device = x.device();
    let mean = Tensor::from_slice(&mean, (1, 3, 1, 1), device)?;
    let std = Tensor::from_slice(&std, (1, 3, 1, 1), device)?;
    x.broadcast_sub(&mean)?.broadcast_div(&std)
}

fn l2(x: &Tensor) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?.maximum(NORM_EPS)?;
    x.broadcast_div(&norm)
}

/// transformers' DINOv2 weights under the torch hub names [`Dinov2`] reads: `query`, `key` and `value` as one `qkv`.
fn hub_dinov2(path: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    let mut hf = inference_tensor::safetensors::load(path, &Device::Cpu)?;
    let mut take = |k: &str| {
        hf.remove(k)
            .ok_or_else(|| inference_tensor::Error::Msg(format!("DINOv2 checkpoint lacks {k}")))
    };
    let mut out: HashMap<String, Tensor> = HashMap::new();
    out.insert("cls_token".into(), take("embeddings.cls_token")?);
    out.insert("pos_embed".into(), take("embeddings.position_embeddings")?);
    for p in ["weight", "bias"] {
        out.insert(
            format!("patch_embed.proj.{p}"),
            take(&format!("embeddings.patch_embeddings.projection.{p}"))?,
        );
        out.insert(format!("norm.{p}"), take(&format!("layernorm.{p}"))?);
    }
    for i in 0..DINO_LAYERS {
        let (src, dst) = (format!("encoder.layer.{i}"), format!("blocks.{i}"));
        for p in ["weight", "bias"] {
            let attn = format!("{src}.attention.attention");
            let qkv = Tensor::cat(
                &[
                    take(&format!("{attn}.query.{p}"))?,
                    take(&format!("{attn}.key.{p}"))?,
                    take(&format!("{attn}.value.{p}"))?,
                ],
                0,
            )?;
            out.insert(format!("{dst}.attn.qkv.{p}"), qkv);
            out.insert(
                format!("{dst}.attn.proj.{p}"),
                take(&format!("{src}.attention.output.dense.{p}"))?,
            );
            for n in ["norm1", "norm2", "mlp.fc1", "mlp.fc2"] {
                out.insert(format!("{dst}.{n}.{p}"), take(&format!("{src}.{n}.{p}"))?);
            }
        }
        out.insert(
            format!("{dst}.ls1.gamma"),
            take(&format!("{src}.layer_scale1.lambda1"))?,
        );
        out.insert(
            format!("{dst}.ls2.gamma"),
            take(&format!("{src}.layer_scale2.lambda1"))?,
        );
    }
    Ok(VarBuilder::from_tensors(out, DType::F32, device))
}

pub struct Conditioner {
    dino: Dinov2,
    clip: ClipVisionTransformer,
    device: Device,
}

impl Conditioner {
    /// DINOv2-large and CLIP ViT-L/14 from their transformers folders (`model.safetensors` and `config.json`).
    pub fn load(dino_dir: &Path, clip_dir: &Path, device: &Device) -> Result<Self> {
        let dino = Dinov2::load(
            DINO_SPEC,
            hub_dinov2(&dino_dir.join("model.safetensors"), device)?,
        )?;
        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(clip_dir.join("config.json"))?)
                .map_err(|e| inference_tensor::Error::Msg(e.to_string()))?;
        let config: ClipConfig = inference_json::from_value(&config["vision_config"])
            .map_err(|e| inference_tensor::Error::Msg(e.to_string()))?;
        let vb = from_mmaped_safetensors(
            vec![clip_dir.join("model.safetensors")],
            Some(DType::F32),
            device,
            Vec::new(),
            true,
            None,
            |name| name.starts_with("vision_model."),
            Arc::new(|_| DeviceForLoadTensor::Base),
        )?;
        let clip = ClipVisionTransformer::new(vb.pp("vision_model"), &config)?;
        Ok(Self {
            dino,
            clip,
            device: device.clone(),
        })
    }

    /// The condition `[1, 2917, 2048]` (f32) for a [`prepare`]d image.
    pub fn forward(&self, prepared: &RgbImage) -> Result<Tensor> {
        let base = pixels(prepared, &self.device)?;
        let dino_in = normalize(
            &base.upsample_bilinear2d(DINO_SIZE, DINO_SIZE, false)?,
            DINO_MEAN,
            DINO_STD,
        )?;
        let mut layers = self.dino.forward_layers(&dino_in, &[DINO_LAYERS - 1])?;
        let (patches, cls) = layers.remove(0);
        let tokens = l2(&Tensor::cat(&[&cls.unsqueeze(1)?, &patches], 1)?)?;
        let pooled = self.clip_embedding(&clip_image(prepared))?;
        let n = tokens.dim(1)?;
        let clip_tokens = pooled.unsqueeze(1)?.broadcast_as((1, n, pooled.dim(1)?))?;
        Tensor::cat(&[&tokens, &clip_tokens], 2)
    }

    /// CLIP's pooled embedding `[1, 1024]` (f32, L2-normed) of a 224 x 224 image.
    pub(crate) fn clip_embedding(&self, small: &RgbImage) -> Result<Tensor> {
        let clip_in = normalize(&pixels(small, &self.device)?, CLIP_MEAN, CLIP_STD)?;
        let pooled = self
            .clip
            .forward_get_hidden_states(&clip_in)?
            .pop()
            .expect("the pooled output is last");
        l2(&pooled.to_dtype(DType::F32)?)
    }
}

/// The prepared image at CLIP's 224 x 224, resampled as PIL's `resize(LANCZOS)`.
pub(crate) fn clip_image(prepared: &RgbImage) -> RgbImage {
    let (w, h) = prepared.dimensions();
    let size = CLIP_SIZE as usize;
    let pixels = inference_vision::pil_resize(
        prepared.as_raw(),
        3,
        (w as usize, h as usize),
        (size, size),
        PilFilter::Lanczos,
    );
    RgbImage::from_raw(CLIP_SIZE, CLIP_SIZE, pixels).expect("224 x 224 RGB")
}
