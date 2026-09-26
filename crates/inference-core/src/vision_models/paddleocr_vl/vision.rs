//! SigLIP/NaViT vision tower (transformers `PaddleOCR*`, interpolate_pos_encoding + use_rope path).
//! Uses both an interpolated learned pos table and 2D axial RoPE; the pooling `head.*` is unused.

use super::config::VisionConfig;
use crate::attention::FlashParams;
use crate::attention::{AttentionMask, SdpaParams};
use crate::layers::{layer_norm, linear, Sdpa};
use crate::utils::unvarbuilder::UnVarBuilder;
use candle_core::{Device, Result, Tensor, D};
use candle_nn::{LayerNorm, Linear, Module};
use inference_quant::ShardedVarBuilder;

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let hd = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, hd / 2)?;
    let x2 = x.narrow(D::Minus1, hd / 2, hd / 2)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let cos = cos.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?;
    x.broadcast_mul(&cos)? + rotate_half(x)?.broadcast_mul(&sin)?
}

// `apply_rotary_pos_emb_vision`: height freqs then width freqs, repeated neox-style to fill head_dim.
fn vision_rope(
    h: usize,
    w: usize,
    head_dim: usize,
    theta: f64,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let rope_dim = head_dim / 2;
    let half = rope_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|k| 1f32 / (theta as f32).powf((2 * k) as f32 / rope_dim as f32))
        .collect();
    let n = h * w;
    let mut cos = vec![0f32; n * head_dim];
    let mut sin = vec![0f32; n * head_dim];
    for j in 0..n {
        let h_id = (j / w) as f32;
        let w_id = (j % w) as f32;
        for k in 0..half {
            let ah = h_id * inv_freq[k];
            let aw = w_id * inv_freq[k];
            let base = j * head_dim;
            cos[base + k] = ah.cos();
            cos[base + half + k] = aw.cos();
            cos[base + rope_dim + k] = ah.cos();
            cos[base + rope_dim + half + k] = aw.cos();
            sin[base + k] = ah.sin();
            sin[base + half + k] = aw.sin();
            sin[base + rope_dim + k] = ah.sin();
            sin[base + rope_dim + half + k] = aw.sin();
        }
    }
    Ok((
        Tensor::from_vec(cos, (n, head_dim), dev)?,
        Tensor::from_vec(sin, (n, head_dim), dev)?,
    ))
}

// kernel == stride == patch on pre-patchified input, so the conv is exactly flatten + matmul.
struct PatchEmbed {
    weight: Tensor, // [hidden, 3*patch*patch]
    bias: Tensor,
}

impl PatchEmbed {
    fn load(vb: ShardedVarBuilder, cfg: &VisionConfig) -> Result<Self> {
        let flat = cfg.num_channels * cfg.patch_size * cfg.patch_size;
        let w = vb.get(
            (
                cfg.hidden_size,
                cfg.num_channels,
                cfg.patch_size,
                cfg.patch_size,
            ),
            "weight",
        )?;
        Ok(Self {
            weight: w.reshape((cfg.hidden_size, flat))?,
            bias: vb.get(cfg.hidden_size, "bias")?,
        })
    }

    fn forward(&self, pv: &Tensor) -> Result<Tensor> {
        let n = pv.dim(0)?;
        let flat = pv.dim(1)? * pv.dim(2)? * pv.dim(3)?;
        // preprocess yields f32 but the weights may be bf16.
        let x = pv.reshape((n, flat))?.to_dtype(self.weight.dtype())?;
        x.matmul(&self.weight.t()?)?.broadcast_add(&self.bias)
    }
}

struct VisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl VisionAttention {
    fn load(vb: ShardedVarBuilder, cfg: &VisionConfig) -> Result<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            q_proj: linear(h, h, vb.pp("q_proj"))?,
            k_proj: linear(h, h, vb.pp("k_proj"))?,
            v_proj: linear(h, h, vb.pp("v_proj"))?,
            out_proj: linear(h, h, vb.pp("out_proj"))?,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim,
            scale: (cfg.head_dim as f64).powf(-0.5),
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let n = x.dim(0)?;
        let hd = self.head_dim;
        let reshape = |t: Tensor| -> Result<Tensor> {
            t.reshape((n, self.num_heads, hd))?
                .transpose(0, 1)?
                .contiguous()
        };
        let q = apply_rope(&reshape(self.q_proj.forward(x)?)?, cos, sin)?;
        let k = apply_rope(&reshape(self.k_proj.forward(x)?)?, cos, sin)?;
        let v = reshape(self.v_proj.forward(x)?)?;

        // One image per call, so full bidirectional attention with no cu_seqlens mask.
        let sdpa_params = SdpaParams {
            n_kv_groups: 1,
            sliding_window: None,
            softcap: None,
            softmax_scale: self.scale as f32,
            sinks: None,
        };
        let flash_params = FlashParams::empty(false);
        let ctx = Sdpa.run_attention(
            &q.unsqueeze(0)?,
            &k.unsqueeze(0)?,
            &v.unsqueeze(0)?,
            &AttentionMask::None,
            Some(&flash_params),
            &sdpa_params,
        )?;

        let ctx = ctx
            .squeeze(0)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((n, self.num_heads * hd))?;
        self.out_proj.forward(&ctx)
    }
}

struct VisionMlp {
    fc1: Linear,
    fc2: Linear,
}

impl VisionMlp {
    fn load(vb: ShardedVarBuilder, cfg: &VisionConfig) -> Result<Self> {
        Ok(Self {
            fc1: linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?,
            fc2: linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(x)?.gelu()?) // gelu_pytorch_tanh
    }
}

struct EncoderLayer {
    layer_norm1: LayerNorm,
    self_attn: VisionAttention,
    layer_norm2: LayerNorm,
    mlp: VisionMlp,
}

impl EncoderLayer {
    fn load(vb: ShardedVarBuilder, cfg: &VisionConfig) -> Result<Self> {
        let eps = cfg.layer_norm_eps;
        Ok(Self {
            layer_norm1: layer_norm(cfg.hidden_size, eps, vb.pp("layer_norm1"))?,
            self_attn: VisionAttention::load(vb.pp("self_attn"), cfg)?,
            layer_norm2: layer_norm(cfg.hidden_size, eps, vb.pp("layer_norm2"))?,
            mlp: VisionMlp::load(vb.pp("mlp"), cfg)?,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let h = (x + self
            .self_attn
            .forward(&self.layer_norm1.forward(x)?, cos, sin)?)?;
        &h + self.mlp.forward(&self.layer_norm2.forward(&h)?)?
    }
}

pub struct VisionModel {
    patch_embed: PatchEmbed,
    position_embedding: Tensor, // [num_positions, hidden]
    layers: Vec<EncoderLayer>,
    post_layernorm: LayerNorm,
    cfg: VisionConfig,
}

impl VisionModel {
    pub fn load(vb: ShardedVarBuilder, cfg: &VisionConfig) -> Result<Self> {
        let emb = vb.pp("embeddings");
        let patch_embed = PatchEmbed::load(emb.pp("patch_embedding"), cfg)?;
        let position_embedding = emb
            .pp("position_embedding")
            .get((cfg.num_positions, cfg.hidden_size), "weight")?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let enc = vb.pp("encoder").pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(EncoderLayer::load(enc.pp(i), cfg)?);
        }
        let post_layernorm =
            layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("post_layernorm"))?;
        Ok(Self {
            patch_embed,
            position_embedding,
            layers,
            post_layernorm,
            cfg: cfg.clone(),
        })
    }

    // align_corners=True matches transformers' linspace sampling, not the custom_code F.interpolate(False).
    fn interpolate_pos(&self, h: usize, w: usize) -> Result<Tensor> {
        let g = self.cfg.pos_grid;
        let d = self.cfg.hidden_size;
        let p = self
            .position_embedding
            .reshape((1, g, g, d))?
            .permute((0, 3, 1, 2))?
            .contiguous()?;
        let up = p.upsample_bilinear2d(h, w, true)?;
        up.permute((0, 2, 3, 1))?.contiguous()?.reshape((h * w, d))
    }

    pub fn forward(&self, pixel_values: &Tensor, t: usize, h: usize, w: usize) -> Result<Tensor> {
        assert_eq!(
            t, 1,
            "video temporal grid (t>1) not in scope for the OCR path"
        );
        let dev = pixel_values.device();
        let patch_embed = self.patch_embed.forward(pixel_values)?;
        let pos = self.interpolate_pos(h, w)?;
        let mut x = (&patch_embed + &pos)?;

        let (cos, sin) = vision_rope(h, w, self.cfg.head_dim, self.cfg.rope_theta, dev)?;
        let (cos, sin) = (cos.to_dtype(x.dtype())?, sin.to_dtype(x.dtype())?);

        for layer in self.layers.iter() {
            x = layer.forward(&x, &cos, &sin)?;
        }
        self.post_layernorm.forward(&x)
    }

    // Patch-embed weight goes back to its 4D conv shape so UQFF round-trips.
    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_v = uvb.pp("visual").pp("vision_model");

        let emb = uvb_v.pp("embeddings");
        let conv_w = self
            .patch_embed
            .weight
            .reshape((
                self.cfg.hidden_size,
                self.cfg.num_channels,
                self.cfg.patch_size,
                self.cfg.patch_size,
            ))
            .expect("patch embed weight reshape");
        let pe = emb.pp("patch_embedding");
        pe.add_tensor("weight", conv_w);
        pe.add_tensor("bias", self.patch_embed.bias.clone());
        emb.pp("position_embedding")
            .add_tensor("weight", self.position_embedding.clone());

        let enc = uvb_v.pp("encoder").pp("layers");
        for (i, layer) in self.layers.iter().enumerate() {
            let uvb_l = enc.pp(i);
            uvb_l.pp("layer_norm1").add(&layer.layer_norm1);
            uvb_l.pp("layer_norm2").add(&layer.layer_norm2);
            let attn = uvb_l.pp("self_attn");
            attn.pp("q_proj").add(&layer.self_attn.q_proj);
            attn.pp("k_proj").add(&layer.self_attn.k_proj);
            attn.pp("v_proj").add(&layer.self_attn.v_proj);
            attn.pp("out_proj").add(&layer.self_attn.out_proj);
            let mlp = uvb_l.pp("mlp");
            mlp.pp("fc1").add(&layer.mlp.fc1);
            mlp.pp("fc2").add(&layer.mlp.fc2);
        }
        uvb_v.pp("post_layernorm").add(&self.post_layernorm);
        uvb.to_safetensors()
    }
}
