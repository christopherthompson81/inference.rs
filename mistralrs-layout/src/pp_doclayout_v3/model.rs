use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{LayerNorm, Linear, VarBuilder};

use super::backbone::HGNetV2Backbone;
use super::config::PPDocLayoutV3Config;
use super::decoder::{inverse_sigmoid, DecodeCtx, DecoderLayer, LevelGeom};
use super::encoder::HybridEncoder;
use crate::layers::{ConvNorm, ConvNormSpec, MlpHead};

const SEQ_CONV: (&str, &str) = ("0", "1");
const ANCHOR_EPS: f32 = 1e-2;
const ANCHOR_GRID: f32 = 0.05;
const MASK_MIN_FILL: f64 = 1e9;
const GP_MASK_FILL: f64 = -1e4;
/// Backbone stride of the x4 feature map that the mask prototypes live on.
const MASK_STRIDE: usize = 4;

pub struct RawOutputs {
    /// `(b, q, num_labels)` class logits.
    pub logits: Tensor,
    /// `(b, q, 4)` normalized cxcywh.
    pub pred_boxes: Tensor,
    /// `(b, q, q)` pairwise reading-order logits.
    pub order_logits: Tensor,
    /// `(b, q, H/4, W/4)` mask logits.
    pub masks: Tensor,
}

pub struct Intermediates {
    pub backbone: Vec<Tensor>,
    pub pan: Vec<Tensor>,
    pub mask_feat: Tensor,
    pub enc_score: Tensor,
    pub init_ref: Tensor,
    pub hidden: Vec<Tensor>,
}

pub struct PPDocLayoutV3 {
    cfg: PPDocLayoutV3Config,
    backbone: HGNetV2Backbone,
    encoder_input_proj: Vec<ConvNorm>,
    encoder: HybridEncoder,
    decoder_input_proj: Vec<ConvNorm>,
    enc_output: (Linear, LayerNorm),
    enc_score_head: Linear,
    enc_bbox_head: MlpHead,
    layers: Vec<DecoderLayer>,
    query_pos_head: MlpHead,
    order_head: Linear,
    global_pointer: Linear,
    decoder_norm: LayerNorm,
    mask_query_head: MlpHead,
    geom: LevelGeom,
    input_hw: (usize, usize),
    /// `(1, S, 1)` 1.0 where the level anchor lies strictly inside the image.
    anchor_valid: Tensor,
    /// `(1, 1, Hm*Wm)` pixel column / row indices of the mask grid.
    mask_x: Tensor,
    mask_y: Tensor,
    mask_norm: Tensor,
    gp_keep: Tensor,
    gp_fill: Tensor,
}

fn anchor_valid_mask(geom: &LevelGeom, dev: &Device) -> Result<Tensor> {
    let mut v = Vec::with_capacity(geom.total);
    for (lvl, &(h, w)) in geom.shapes.iter().enumerate() {
        let wh = ANCHOR_GRID * 2f32.powi(lvl as i32);
        let wh_ok = wh > ANCHOR_EPS && wh < 1. - ANCHOR_EPS;
        for y in 0..h {
            for x in 0..w {
                let cx = (x as f32 + 0.5) / w as f32;
                let cy = (y as f32 + 0.5) / h as f32;
                let ok = wh_ok
                    && [cx, cy]
                        .iter()
                        .all(|&c| c > ANCHOR_EPS && c < 1. - ANCHOR_EPS);
                v.push(if ok { 1f32 } else { 0f32 });
            }
        }
    }
    Tensor::from_vec(v, (1, geom.total, 1), dev)
}

impl PPDocLayoutV3 {
    /// The model is shape-specialized to `input_hw` so all positional/anchor tensors are built once here.
    pub fn new(cfg: PPDocLayoutV3Config, input_hw: (usize, usize), vb: VarBuilder) -> Result<Self> {
        if cfg.backbone_config.arch != "L" {
            candle_core::bail!("unsupported HGNetV2 arch {}", cfg.backbone_config.arch);
        }
        if cfg.decoder_in_channels.len() != cfg.num_feature_levels {
            candle_core::bail!("extra strided decoder levels are not supported");
        }
        let dev = vb.device().clone();
        let vbm = vb.pp("model");
        let d = cfg.d_model;
        let (ih, iw) = input_hw;

        let bb_out = HGNetV2Backbone::out_channels();
        let encoder_input_proj = bb_out[1..]
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                ConvNormSpec::new(c, cfg.encoder_hidden_dim, 1)
                    .names(SEQ_CONV)
                    .load(vbm.pp("encoder_input_proj").pp(i))
            })
            .collect::<Result<Vec<_>>>()?;

        let shapes: Vec<(usize, usize)> =
            cfg.feat_strides.iter().map(|s| (ih / s, iw / s)).collect();
        let aifi_hw: Vec<(usize, usize)> =
            cfg.encode_proj_layers.iter().map(|&l| shapes[l]).collect();
        let encoder = HybridEncoder::new(&cfg, &aifi_hw, vbm.pp("encoder"))?;

        let decoder_input_proj = cfg
            .decoder_in_channels
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                ConvNormSpec::new(c, d, 1)
                    .names(SEQ_CONV)
                    .load(vbm.pp("decoder_input_proj").pp(i))
            })
            .collect::<Result<Vec<_>>>()?;

        let eps = cfg.layer_norm_eps;
        let nl = cfg.num_labels();
        let layers = (0..cfg.decoder_layers)
            .map(|i| DecoderLayer::new(&cfg, vbm.pp("decoder").pp("layers").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        // only the final decoder layer's order head feeds the output
        let order_head = candle_nn::linear(
            d,
            d,
            vbm.pp("decoder_order_head").pp(cfg.decoder_layers - 1),
        )?;

        let geom = LevelGeom::new(shapes);
        let anchor_valid = anchor_valid_mask(&geom, &dev)?;

        let (mh, mw) = (ih / MASK_STRIDE, iw / MASK_STRIDE);
        let mask_x = Tensor::arange(0u32, mw as u32, &dev)?
            .to_dtype(DType::F32)?
            .reshape((1, mw))?
            .broadcast_as((mh, mw))?
            .contiguous()?
            .reshape((1, 1, mh * mw))?;
        let mask_y = Tensor::arange(0u32, mh as u32, &dev)?
            .to_dtype(DType::F32)?
            .reshape((mh, 1))?
            .broadcast_as((mh, mw))?
            .contiguous()?
            .reshape((1, 1, mh * mw))?;
        let mask_norm = Tensor::new(&[mw as f32, mh as f32, mw as f32, mh as f32], &dev)?;

        let q = cfg.num_queries;
        // reference masks the lower triangle incl. the diagonal
        let gp_keep =
            Tensor::triu2(q, DType::F32, &dev)?.sub(&Tensor::eye(q, DType::F32, &dev)?)?;
        let gp_fill = gp_keep.affine(-GP_MASK_FILL, GP_MASK_FILL)?;

        Ok(Self {
            backbone: HGNetV2Backbone::new(vbm.pp("backbone").pp("model"))?,
            encoder_input_proj,
            encoder,
            decoder_input_proj,
            enc_output: (
                candle_nn::linear(d, d, vbm.pp("enc_output").pp(0))?,
                candle_nn::layer_norm(d, eps, vbm.pp("enc_output").pp(1))?,
            ),
            // decoder.class_embed / bbox_embed are tied to these in the checkpoint
            enc_score_head: candle_nn::linear(d, nl, vbm.pp("enc_score_head"))?,
            enc_bbox_head: MlpHead::new(d, d, 4, 3, vbm.pp("enc_bbox_head"))?,
            layers,
            query_pos_head: MlpHead::new(4, 2 * d, d, 2, vbm.pp("decoder").pp("query_pos_head"))?,
            order_head,
            global_pointer: candle_nn::linear(
                d,
                cfg.global_pointer_head_size * 2,
                vbm.pp("decoder_global_pointer").pp("dense"),
            )?,
            decoder_norm: candle_nn::layer_norm(d, eps, vbm.pp("decoder_norm"))?,
            mask_query_head: MlpHead::new(d, d, cfg.num_prototypes, 3, vbm.pp("mask_query_head"))?,
            geom,
            input_hw,
            anchor_valid,
            mask_x,
            mask_y,
            mask_norm,
            gp_keep,
            gp_fill,
            cfg,
        })
    }

    pub fn config(&self) -> &PPDocLayoutV3Config {
        &self.cfg
    }

    pub fn input_hw(&self) -> (usize, usize) {
        self.input_hw
    }

    pub fn forward(&self, pixel_values: &Tensor) -> Result<RawOutputs> {
        self.forward_inner(pixel_values, None)
    }

    pub fn forward_with_intermediates(
        &self,
        pixel_values: &Tensor,
    ) -> Result<(RawOutputs, Intermediates)> {
        let mut inter = None;
        let out = self.forward_inner(pixel_values, Some(&mut inter))?;
        Ok((out, inter.expect("intermediates recorded")))
    }

    /// `(b, q, Hm*Wm)` mask logits -> `(b, q, 4)` normalized cxcywh of each mask's `> 0` bounding box.
    fn mask_to_box(&self, masks: &Tensor) -> Result<Tensor> {
        let m = masks.gt(0.)?.to_dtype(DType::F32)?;
        let not_m = m.affine(-1., 1.)?;
        let x_max = (m.broadcast_mul(&self.mask_x)?.max_keepdim(D::Minus1)? + 1.)?;
        let y_max = (m.broadcast_mul(&self.mask_y)?.max_keepdim(D::Minus1)? + 1.)?;
        let x_min =
            (m.broadcast_mul(&self.mask_x)? + (&not_m * MASK_MIN_FILL)?)?.min_keepdim(D::Minus1)?;
        let y_min =
            (m.broadcast_mul(&self.mask_y)? + (&not_m * MASK_MIN_FILL)?)?.min_keepdim(D::Minus1)?;
        let non_empty = m.max_keepdim(D::Minus1)?;
        let xyxy = Tensor::cat(&[x_min, y_min, x_max, y_max], D::Minus1)?
            .broadcast_mul(&non_empty)?
            .broadcast_div(&self.mask_norm)?;
        let x0 = xyxy.narrow(D::Minus1, 0, 1)?;
        let y0 = xyxy.narrow(D::Minus1, 1, 1)?;
        let x1 = xyxy.narrow(D::Minus1, 2, 1)?;
        let y1 = xyxy.narrow(D::Minus1, 3, 1)?;
        Tensor::cat(
            &[
                ((&x0 + &x1)? / 2.)?,
                ((&y0 + &y1)? / 2.)?,
                (x1 - x0)?,
                (y1 - y0)?,
            ],
            D::Minus1,
        )
    }

    /// Top-k over `S` runs on the host: candle's CUDA arg-sort keeps a whole row in shared memory.
    fn topk_rows(&self, scores: &Tensor, k: usize) -> Result<Tensor> {
        let (b, s) = scores.dims2()?;
        let rows = scores.to_vec2::<f32>()?;
        let mut flat = Vec::with_capacity(b * k);
        for (bi, row) in rows.iter().enumerate() {
            let mut order: Vec<u32> = (0..s as u32).collect();
            order.select_nth_unstable_by(k - 1, |&a, &c| {
                row[c as usize].total_cmp(&row[a as usize])
            });
            let top = &mut order[..k];
            top.sort_unstable_by(|&a, &c| row[c as usize].total_cmp(&row[a as usize]));
            flat.extend(top.iter().map(|&i| i + (bi * s) as u32));
        }
        Tensor::from_vec(flat, b * k, scores.device())
    }

    fn global_pointer(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, q, _) = xs.dims3()?;
        let hs = self.cfg.global_pointer_head_size;
        let qk = self.global_pointer.forward(xs)?.reshape((b, q, 2, hs))?;
        let queries = qk.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?;
        let keys = qk.narrow(2, 1, 1)?.squeeze(2)?.contiguous()?;
        let logits = (queries.matmul(&keys.t()?)? / (hs as f64).sqrt())?;
        logits
            .broadcast_mul(&self.gp_keep)?
            .broadcast_add(&self.gp_fill)
    }

    fn forward_inner(
        &self,
        pixel_values: &Tensor,
        inter: Option<&mut Option<Intermediates>>,
    ) -> Result<RawOutputs> {
        let (b, _, h, w) = pixel_values.dims4()?;
        if (h, w) != self.input_hw {
            candle_core::bail!("expected {:?} input, got {:?}", self.input_hw, (h, w));
        }
        let dev = pixel_values.device();
        let d = self.cfg.d_model;
        let q = self.cfg.num_queries;
        let heads = self.cfg.decoder_attention_heads;

        let mut feats = self.backbone.forward(pixel_values)?;
        let backbone_feats = inter.is_some().then(|| feats.clone());
        let x4 = feats.remove(0);
        let proj = feats
            .iter()
            .zip(&self.encoder_input_proj)
            .map(|(f, p)| p.forward(f))
            .collect::<Result<Vec<_>>>()?;
        let enc = self.encoder.forward(proj, &x4)?;

        let memory = enc
            .feats
            .iter()
            .zip(&self.decoder_input_proj)
            .map(|(f, p)| p.forward(f)?.flatten_from(2)?.transpose(1, 2))
            .collect::<Result<Vec<_>>>()?;
        let memory = Tensor::cat(&memory, 1)?.contiguous()?;
        let s = self.geom.total;

        let output_memory = self.enc_output.1.forward(
            &self
                .enc_output
                .0
                .forward(&memory.broadcast_mul(&self.anchor_valid)?)?,
        )?;
        let enc_score = self.enc_score_head.forward(&output_memory)?;
        let topk = self.topk_rows(&enc_score.max(D::Minus1)?, q)?;
        let target = output_memory
            .reshape((b * s, d))?
            .index_select(&topk, 0)?
            .reshape((b, q, d))?;

        let (_, np, mh, mw) = enc.mask_feat.dims4()?;
        let mask_feat = enc.mask_feat.reshape((b, np, mh * mw))?;
        let mask_embed = self
            .mask_query_head
            .forward(&self.decoder_norm.forward(&target)?)?;
        let enc_masks = mask_embed.matmul(&mask_feat)?;
        let init_ref = inverse_sigmoid(&self.mask_to_box(&enc_masks)?)?;

        let bh_offset = (Tensor::arange(0u32, (b * heads) as u32, dev)?.to_dtype(DType::F32)?
            * s as f64)?
            .reshape((b * heads, 1))?;
        let ctx = DecodeCtx {
            memory: &memory,
            geom: &self.geom,
            bh_offset: &bh_offset,
        };

        let mut hs = target;
        let mut ref_boxes = candle_nn::ops::sigmoid(&init_ref)?;
        let mut hidden = Vec::new();
        for layer in &self.layers {
            let pos = self.query_pos_head.forward(&ref_boxes)?;
            hs = layer.forward(&hs, &pos, &ref_boxes, &ctx)?;
            ref_boxes = candle_nn::ops::sigmoid(
                &(self.enc_bbox_head.forward(&hs)? + inverse_sigmoid(&ref_boxes)?)?,
            )?;
            if inter.is_some() {
                hidden.push(hs.clone());
            }
        }

        let out_query = self.decoder_norm.forward(&hs)?;
        let logits = self.enc_score_head.forward(&out_query)?;
        let masks = self
            .mask_query_head
            .forward(&out_query)?
            .matmul(&mask_feat)?
            .reshape((b, q, mh, mw))?;
        let order_logits = self.global_pointer(&self.order_head.forward(&out_query)?)?;

        if let Some(slot) = inter {
            *slot = Some(Intermediates {
                backbone: backbone_feats.unwrap_or_default(),
                pan: enc.feats.clone(),
                mask_feat: enc.mask_feat.clone(),
                enc_score,
                init_ref: init_ref.clone(),
                hidden,
            });
        }
        Ok(RawOutputs {
            logits,
            pred_boxes: ref_boxes,
            order_logits,
            masks,
        })
    }
}
