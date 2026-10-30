//! Gemma4 audio encoder.
//!
//! Gemma4's audio stack is similar to Gemma3n's, but it is not drop-in
//! compatible. In particular, the SSCP projection uses semicausal padding and
//! layer norm, the conformer blocks use standard RMSNorm without Gemma's +1
//! scale offset, the attention uses an extra `per_dim_key_scale`, and several
//! projections use optional clipping buffers from the checkpoint.

use inference_quant::{Convolution, QuantMethod, ShardedVarBuilder};
use inference_tensor::nn::{Conv1d, Conv2d, Conv2dConfig, LayerNorm, LayerNormConfig, ModuleT};
use inference_tensor::{D, DType, Module, Result, Tensor, bail};
use std::sync::Arc;

use crate::audio_attention::{LocalAttention, RelativePositionEmbedding};
use crate::{
    layers::{RmsNorm, conv1d_no_bias, conv2d_no_bias, layer_norm},
    utils::unvarbuilder::UnVarBuilder,
};

use super::config::Gemma4AudioConfig;

// HF's _init_weights overwrites the initial ones buffer with the standard sinusoidal timescales (min 1, max 10000)
fn gemma4_inv_timescales(channels: usize, device: &inference_tensor::Device) -> Result<Tensor> {
    let num_timescales = channels / 2;
    let min_timescale = 1.0_f64;
    let max_timescale = 10_000.0_f64;
    let log_timescale_increment =
        (max_timescale / min_timescale).ln() / num_timescales.saturating_sub(1).max(1) as f64;
    Tensor::from_vec(
        (0..num_timescales)
            .map(|i| (min_timescale * (-log_timescale_increment * i as f64).exp()) as f32)
            .collect::<Vec<_>>(),
        (1, 1, num_timescales),
        device,
    )
}

struct ClippableLinear {
    inner: Arc<dyn QuantMethod>,
    input_min: Option<f64>,
    input_max: Option<f64>,
    output_min: Option<f64>,
    output_max: Option<f64>,
    has_linear_prefix: bool,
}

impl ClippableLinear {
    fn new_no_bias(
        cfg: &Gemma4AudioConfig,
        in_features: usize,
        out_features: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let has_linear_prefix = super::has_clippable_linear_prefix(&vb);
        let linear_vb = if has_linear_prefix {
            vb.pp("linear")
        } else {
            vb.clone()
        };
        let inner = inference_quant::linear_no_bias(in_features, out_features, &None, linear_vb)?;
        let (input_min, input_max, output_min, output_max) = if cfg.use_clipped_linears {
            (
                Self::load_clip_scalar(&vb, "input_min"),
                Self::load_clip_scalar(&vb, "input_max"),
                Self::load_clip_scalar(&vb, "output_min"),
                Self::load_clip_scalar(&vb, "output_max"),
            )
        } else {
            (None, None, None, None)
        };

        Ok(Self {
            inner,
            input_min,
            input_max,
            output_min,
            output_max,
            has_linear_prefix,
        })
    }

    fn load_clip_scalar(vb: &ShardedVarBuilder, name: &str) -> Option<f64> {
        vb.get((), name)
            .ok()
            .and_then(|t| t.to_dtype(DType::F32).ok())
            .and_then(|t| t.to_scalar::<f32>().ok())
            .map(f64::from)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();
        if let (Some(lo), Some(hi)) = (self.input_min, self.input_max) {
            x = x.clamp(lo, hi)?;
        }
        let mut out = self.inner.forward(&x)?;
        if let (Some(lo), Some(hi)) = (self.output_min, self.output_max) {
            out = out.clamp(lo, hi)?;
        }
        Ok(out)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        if self.has_linear_prefix {
            uvb.pp("linear").add(&self.inner);
        } else {
            uvb.add(&self.inner);
        }
        if let Some(v) = self.input_min {
            uvb.add_tensor(
                "input_min",
                Tensor::new(v as f32, &inference_tensor::Device::Cpu).unwrap(),
            );
        }
        if let Some(v) = self.input_max {
            uvb.add_tensor(
                "input_max",
                Tensor::new(v as f32, &inference_tensor::Device::Cpu).unwrap(),
            );
        }
        if let Some(v) = self.output_min {
            uvb.add_tensor(
                "output_min",
                Tensor::new(v as f32, &inference_tensor::Device::Cpu).unwrap(),
            );
        }
        if let Some(v) = self.output_max {
            uvb.add_tensor(
                "output_max",
                Tensor::new(v as f32, &inference_tensor::Device::Cpu).unwrap(),
            );
        }
        uvb.to_safetensors()
    }
}

fn subsample_mask(mask: &Tensor, stride: usize, target_len: usize) -> Result<Tensor> {
    let indices = Tensor::arange(0f32, target_len as f32, mask.device())?
        .affine(stride as f64, 0.0)?
        .to_dtype(DType::I64)?;
    let max_idx = mask.dim(1)? as i64 - 1;
    let indices = indices.clamp(0i64, max_idx)?;

    if indices.dims().len() == 1 {
        if mask.dim(0)? == 1 {
            return mask.index_select(&indices, 1);
        }

        let batch_size = mask.dim(0)?;
        let mut out = Vec::with_capacity(batch_size);
        for batch_idx in 0..batch_size {
            out.push(
                mask.get(batch_idx)?
                    .index_select(&indices, 0)?
                    .unsqueeze(0)?,
            );
        }
        Tensor::cat(&out, 0)
    } else {
        bail!("Expected 1D indices for mask subsampling")
    }
}

struct Gemma4AudioSSCPConvBlock {
    conv: Conv2d,
    norm: LayerNorm,
    manual_padding: (usize, usize, usize, usize),
    time_stride: usize,
}

impl Gemma4AudioSSCPConvBlock {
    fn new(
        cfg: &Gemma4AudioConfig,
        idx: usize,
        input_freq_dim: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        if cfg.sscp_conv_norm_type != "layer_norm" {
            bail!(
                "Unsupported Gemma4 SSCP norm type `{}`; expected `layer_norm`.",
                cfg.sscp_conv_norm_type
            );
        }

        let in_channels = if idx == 0 {
            1
        } else {
            cfg.sscp_conv_channel_size[idx - 1]
        };
        let out_channels = cfg.sscp_conv_channel_size[idx];
        let kernel_t = cfg.sscp_conv_kernel_size[idx][0];
        let kernel_f = cfg.sscp_conv_kernel_size[idx][1];
        let stride_t = cfg.sscp_conv_stride_size[idx][0];
        let stride_f = cfg.sscp_conv_stride_size[idx][1];

        let (pad_t_top, pad_t_bottom) = if let (Some(top), Some(bottom)) =
            (cfg.sscp_conv_time_pad_top, cfg.sscp_conv_time_pad_bottom)
        {
            (top, bottom)
        } else if cfg.sscp_conv_padding_type == "semicausal" {
            let half = kernel_t / 2;
            (half, if cfg.streaming { 0 } else { half })
        } else {
            (0, if cfg.streaming { 0 } else { kernel_t - 1 })
        };
        let pad_f_left = 1;
        let pad_f_right = 1;

        assert_eq!(
            kernel_t, kernel_f,
            "Gemma4 SSCP conv2d requires square kernels (candle limitation), got ({kernel_t}, {kernel_f})"
        );
        assert_eq!(
            stride_t, stride_f,
            "Gemma4 SSCP conv2d requires square strides (candle limitation), got ({stride_t}, {stride_f})"
        );
        let conv = conv2d_no_bias(
            in_channels,
            out_channels,
            kernel_t,
            Conv2dConfig {
                stride: stride_t,
                padding: 0,
                dilation: 1,
                groups: 1,
            },
            vb.pp("conv"),
        )?;
        let norm = layer_norm(
            out_channels,
            LayerNormConfig {
                eps: cfg.rms_norm_eps,
                affine: false,
                ..Default::default()
            },
            vb.pp("norm"),
        )?;

        let _f_in_padded = input_freq_dim + pad_f_left + pad_f_right;

        Ok(Self {
            conv,
            norm,
            manual_padding: (pad_f_left, pad_f_right, pad_t_top, pad_t_bottom),
            time_stride: stride_t,
        })
    }

    fn forward(
        &self,
        audio_encodings: &Tensor,
        audio_mel_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let valid_mask = audio_mel_mask
            .eq(0.0)?
            .unsqueeze(1)?
            .unsqueeze(D::Minus1)?
            .to_dtype(audio_encodings.dtype())?;
        let audio_encodings = audio_encodings.broadcast_mul(&valid_mask)?;

        let audio_encodings = audio_encodings
            .pad_with_zeros(D::Minus1, self.manual_padding.0, self.manual_padding.1)?
            .pad_with_zeros(D::Minus2, self.manual_padding.2, self.manual_padding.3)?;
        let audio_encodings_conv = self.conv.forward_t(&audio_encodings, false)?;
        let output_mask = subsample_mask(
            audio_mel_mask,
            self.time_stride,
            audio_encodings_conv.dim(2)?,
        )?;

        let x = audio_encodings_conv.permute((0, 2, 3, 1))?;
        let x = self.norm.forward(&x)?;
        let x = x.permute((0, 3, 1, 2))?.relu()?;
        Ok((x, output_mask))
    }
}

pub struct Gemma4AudioSubSampleConvProjection {
    conv_0: Gemma4AudioSSCPConvBlock,
    conv_1: Gemma4AudioSSCPConvBlock,
    input_proj_linear: Arc<dyn QuantMethod>,
}

impl Gemma4AudioSubSampleConvProjection {
    fn new(cfg: &Gemma4AudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let input_feat_size = cfg.input_feat_size();
        let mut current_f_for_block_input = input_feat_size;
        let mut calculated_f_out_dims = Vec::new();

        for i in 0..2 {
            let kernel_w = cfg.sscp_conv_kernel_size[i][1];
            let stride_w = cfg.sscp_conv_stride_size[i][1];
            let f_in_padded = current_f_for_block_input + 2;
            let f_out_after_conv = (f_in_padded - kernel_w) / stride_w + 1;
            calculated_f_out_dims.push(f_out_after_conv);
            current_f_for_block_input = f_out_after_conv;
        }

        let conv_0 = Gemma4AudioSSCPConvBlock::new(cfg, 0, input_feat_size, vb.pp("layer0"))?;
        let conv_1 =
            Gemma4AudioSSCPConvBlock::new(cfg, 1, calculated_f_out_dims[0], vb.pp("layer1"))?;
        let final_c_out = cfg.sscp_conv_channel_size[1];
        let final_f_out = calculated_f_out_dims[1];
        let input_proj_in_features = final_c_out * final_f_out;
        let input_proj_linear = inference_quant::linear_no_bias(
            input_proj_in_features,
            cfg.hidden_size,
            &None,
            vb.pp("input_proj_linear"),
        )?;

        Ok(Self {
            conv_0,
            conv_1,
            input_proj_linear,
        })
    }

    pub fn forward(&self, audio_mel: &Tensor, audio_mel_mask: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = audio_mel.unsqueeze(1)?;
        let (x, mask) = self.conv_0.forward(&x, audio_mel_mask)?;
        let (x, mask) = self.conv_1.forward(&x, &mask)?;

        let (b, c_out, t_out, f_out) = x.dims4()?;
        let x = x
            .transpose(1, 2)?
            .transpose(2, 3)?
            .reshape((b, t_out, f_out * c_out))?;
        Ok((self.input_proj_linear.forward(&x)?, mask))
    }
}

pub struct Gemma4AudioAttention {
    num_heads: usize,
    head_dim: usize,
    local: LocalAttention,
    pub relative_position_embedding: RelativePositionEmbedding,
    _per_dim_scale: Tensor,
    q_proj: ClippableLinear,
    k_proj: ClippableLinear,
    v_proj: ClippableLinear,
    q_scale: f64,
    k_scale: f64,
    per_dim_scale_softplus: Tensor,
}

impl Gemma4AudioAttention {
    fn new(cfg: &Gemma4AudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let num_heads = cfg.conf_num_attention_heads;
        let hidden_size = cfg.hidden_size;
        let head_dim = hidden_size / num_heads;

        let pos_proj = inference_quant::linear_no_bias(
            hidden_size,
            num_heads * head_dim,
            &None,
            vb.pp("relative_k_proj"),
        )?;
        let relative_position_embedding = RelativePositionEmbedding::new(
            num_heads,
            head_dim,
            pos_proj,
            gemma4_inv_timescales(hidden_size, vb.device())?,
            cfg.conf_attention_context_left.saturating_sub(1),
            cfg.conf_attention_context_right,
        )?;
        let per_dim_scale = vb.get(head_dim, "per_dim_scale")?;
        let q_proj =
            ClippableLinear::new_no_bias(cfg, hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj =
            ClippableLinear::new_no_bias(cfg, hidden_size, num_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj =
            ClippableLinear::new_no_bias(cfg, hidden_size, num_heads * head_dim, vb.pp("v_proj"))?;

        let q_scale = (head_dim as f64).powf(-0.5) / 2.0_f64.ln();
        let k_scale = (1.0_f64 + std::f64::consts::E).ln() / 2.0_f64.ln();

        let local = LocalAttention::new(
            num_heads,
            head_dim,
            cfg.conf_attention_chunk_size,
            cfg.conf_attention_context_left,
            cfg.conf_attention_context_right,
            cfg.conf_attention_logit_cap,
            cfg.conf_attention_invalid_logits_value,
            vb.device(),
        )?;
        let per_dim_scale_softplus = {
            let ones = Tensor::ones_like(&per_dim_scale)?.to_dtype(DType::F32)?;
            let exp_scale = per_dim_scale.to_dtype(DType::F32)?.exp()?;
            ones.broadcast_add(&exp_scale)?.log()?
        };

        Ok(Self {
            num_heads,
            head_dim,
            local,
            relative_position_embedding,
            _per_dim_scale: per_dim_scale,
            q_proj,
            k_proj,
            v_proj,
            q_scale,
            k_scale,
            per_dim_scale_softplus,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let query_states = self.q_proj.forward(x)?.to_dtype(DType::F32)?;
        let key_states = self.k_proj.forward(x)?.to_dtype(DType::F32)?;
        let value_states = self.v_proj.forward(x)?.to_dtype(DType::F32)?;

        let (b, t) = match x.dims() {
            &[b, t, _] => (b, t),
            _ => bail!("Expected input to have 3 dimensions"),
        };

        let query_states = query_states.reshape((b, t, self.num_heads, self.head_dim))?;
        let key_states = key_states.reshape((b, t, self.num_heads, self.head_dim))?;
        let value_states = value_states.reshape((b, t, self.num_heads, self.head_dim))?;

        // HF: query_states = query_states * q_scale * softplus(per_dim_scale)
        let query_states = query_states.affine(self.q_scale, 0.0)?.broadcast_mul(
            &self
                .per_dim_scale_softplus
                .to_device(x.device())?
                .reshape((1, 1, 1, self.head_dim))?
                .to_dtype(DType::F32)?,
        )?;
        // HF: key_states = key_states * k_scale
        let key_states = key_states.affine(self.k_scale, 0.0)?;

        self.local.forward(
            &self.relative_position_embedding,
            &query_states,
            &key_states,
            &value_states,
            mask,
        )
    }
}

pub struct Gemma4AudioConformerAttention {
    pre_attn_norm: RmsNorm,
    attn: Gemma4AudioAttention,
    post: ClippableLinear,
    post_norm: RmsNorm,
    gradient_clipping: f64,
    hidden_size: usize,
}

impl Gemma4AudioConformerAttention {
    fn new(
        cfg: &Gemma4AudioConfig,
        attn_vb: ShardedVarBuilder,
        pre_attn_norm_vb: ShardedVarBuilder,
        post_norm_vb: ShardedVarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            pre_attn_norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, pre_attn_norm_vb)?,
            attn: Gemma4AudioAttention::new(cfg, attn_vb.clone())?,
            post: ClippableLinear::new_no_bias(
                cfg,
                cfg.hidden_size,
                cfg.hidden_size,
                attn_vb.pp("post"),
            )?,
            post_norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, post_norm_vb)?,
            gradient_clipping: cfg.gradient_clipping,
            hidden_size: cfg.hidden_size,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = x.clamp(-self.gradient_clipping, self.gradient_clipping)?;
        let x = self.pre_attn_norm.forward(&x)?;
        let x = self.attn.forward(&x, mask)?;
        let (b, t, _, _) = x.dims4()?;
        let x = x.reshape((b, t, self.hidden_size))?;
        let x = self
            .post
            .forward(&x)?
            .clamp(-self.gradient_clipping, self.gradient_clipping)?;
        residual.broadcast_add(&self.post_norm.forward(&x)?)
    }
}

pub struct Gemma4AudioConformerFeedForward {
    scale: f64,
    pre_layer_norm: RmsNorm,
    ffw_layer_1: ClippableLinear,
    ffw_layer_2: ClippableLinear,
    post_layer_norm: RmsNorm,
    gradient_clipping: f64,
}

impl Gemma4AudioConformerFeedForward {
    fn new(cfg: &Gemma4AudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            scale: cfg.conf_residual_weight,
            pre_layer_norm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("pre_layer_norm"),
            )?,
            ffw_layer_1: ClippableLinear::new_no_bias(
                cfg,
                cfg.hidden_size,
                cfg.hidden_size * 4,
                vb.pp("ffw_layer_1"),
            )?,
            ffw_layer_2: ClippableLinear::new_no_bias(
                cfg,
                cfg.hidden_size * 4,
                cfg.hidden_size,
                vb.pp("ffw_layer_2"),
            )?,
            post_layer_norm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_layer_norm"),
            )?,
            gradient_clipping: cfg.gradient_clipping,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = x.clamp(-self.gradient_clipping, self.gradient_clipping)?;
        let x = self.pre_layer_norm.forward(&x)?;
        let x = inference_tensor::nn::ops::silu(&self.ffw_layer_1.forward(&x)?)?;
        let x = self
            .ffw_layer_2
            .forward(&x)?
            .clamp(-self.gradient_clipping, self.gradient_clipping)?;
        let x = self.post_layer_norm.forward(&x)?;
        residual.broadcast_add(&(x * self.scale)?)
    }
}

pub struct Gemma4AudioConformerLightConv1d {
    pre_layer_norm: RmsNorm,
    depthwise_conv1d: Conv1d,
    conv_norm: RmsNorm,
    linear_start: ClippableLinear,
    linear_end: ClippableLinear,
    causal_padding: usize,
    gradient_clipping: f64,
}

impl Gemma4AudioConformerLightConv1d {
    fn new(cfg: &Gemma4AudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            pre_layer_norm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("pre_layer_norm"),
            )?,
            linear_start: ClippableLinear::new_no_bias(
                cfg,
                cfg.hidden_size,
                cfg.hidden_size * 2,
                vb.pp("linear_start"),
            )?,
            depthwise_conv1d: conv1d_no_bias(
                cfg.hidden_size,
                cfg.hidden_size,
                cfg.conf_conv_kernel_size,
                inference_tensor::nn::Conv1dConfig {
                    stride: 1,
                    padding: 0,
                    dilation: 1,
                    groups: cfg.hidden_size,
                },
                vb.pp("depthwise_conv1d").set_dtype(DType::F32),
            )?,
            conv_norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("conv_norm"))?,
            linear_end: ClippableLinear::new_no_bias(
                cfg,
                cfg.hidden_size,
                cfg.hidden_size,
                vb.pp("linear_end"),
            )?,
            causal_padding: cfg.conf_conv_kernel_size - 1,
            gradient_clipping: cfg.gradient_clipping,
        })
    }

    fn forward(&self, audio_encodings: &Tensor) -> Result<Tensor> {
        let residual = audio_encodings;
        let audio_encodings = self.pre_layer_norm.forward(audio_encodings)?;
        let audio_encodings = self.linear_start.forward(&audio_encodings)?;
        let chunks = audio_encodings.chunk(2, D::Minus1)?;
        let audio_encodings =
            chunks[0].broadcast_mul(&inference_tensor::nn::ops::sigmoid(&chunks[1])?)?;
        let audio_encodings = audio_encodings.transpose(D::Minus1, D::Minus2)?;
        let audio_encodings = audio_encodings.pad_with_zeros(D::Minus1, self.causal_padding, 0)?;
        let audio_encodings = Convolution
            .forward_1d(
                &self.depthwise_conv1d,
                &audio_encodings.to_dtype(DType::F32)?,
            )?
            .to_dtype(audio_encodings.dtype())?
            .transpose(D::Minus2, D::Minus1)?
            .clamp(-self.gradient_clipping, self.gradient_clipping)?;
        let audio_encodings = self.conv_norm.forward(&audio_encodings)?;
        let audio_encodings = inference_tensor::nn::ops::silu(&audio_encodings)?;
        let audio_encodings = self.linear_end.forward(&audio_encodings)?;
        residual.broadcast_add(&audio_encodings)
    }
}

pub struct Gemma4AudioConformerBlock {
    pub ffw_layer_start: Gemma4AudioConformerFeedForward,
    pub attention: Gemma4AudioConformerAttention,
    pub lconv1d: Gemma4AudioConformerLightConv1d,
    pub ffw_layer_end: Gemma4AudioConformerFeedForward,
    pub norm: RmsNorm,
    gradient_clipping: f64,
}

impl Gemma4AudioConformerBlock {
    fn new(cfg: &Gemma4AudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            ffw_layer_start: Gemma4AudioConformerFeedForward::new(cfg, vb.pp("feed_forward1"))?,
            attention: Gemma4AudioConformerAttention::new(
                cfg,
                vb.pp("self_attn"),
                vb.pp("norm_pre_attn"),
                vb.pp("norm_post_attn"),
            )?,
            lconv1d: Gemma4AudioConformerLightConv1d::new(cfg, vb.pp("lconv1d"))?,
            ffw_layer_end: Gemma4AudioConformerFeedForward::new(cfg, vb.pp("feed_forward2"))?,
            norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm_out"))?,
            gradient_clipping: cfg.gradient_clipping,
        })
    }

    pub fn forward(&self, audio_encodings: &Tensor, audio_mel_mask: &Tensor) -> Result<Tensor> {
        let audio_encodings = self.ffw_layer_start.forward(audio_encodings)?;
        let audio_encodings = self.attention.forward(&audio_encodings, audio_mel_mask)?;
        let audio_encodings = self.lconv1d.forward(&audio_encodings)?;
        let audio_encodings = self
            .ffw_layer_end
            .forward(&audio_encodings)?
            .clamp(-self.gradient_clipping, self.gradient_clipping)?;
        self.norm.forward(&audio_encodings)
    }
}

pub struct AudioModel {
    pub subsample_conv_projection: Gemma4AudioSubSampleConvProjection,
    pub conformer: Vec<Gemma4AudioConformerBlock>,
    conf_reduction_factor: usize,
    pub output_proj: Option<Arc<dyn QuantMethod>>,
}

impl AudioModel {
    pub fn new(cfg: &Gemma4AudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let subsample_conv_projection =
            Gemma4AudioSubSampleConvProjection::new(cfg, vb.pp("subsample_conv_projection"))?;
        let mut conformer = Vec::with_capacity(cfg.conf_num_hidden_layers);
        let vb_layers = vb.pp("layers");
        for i in 0..cfg.conf_num_hidden_layers {
            conformer.push(Gemma4AudioConformerBlock::new(cfg, vb_layers.pp(i))?);
        }
        let output_proj = if let Some(output_dim) = cfg.output_proj_dims {
            Some(inference_quant::linear(
                cfg.hidden_size,
                output_dim,
                &None,
                vb.pp("output_proj"),
            )?)
        } else {
            None
        };

        Ok(Self {
            subsample_conv_projection,
            conformer,
            conf_reduction_factor: cfg.conf_reduction_factor,
            output_proj,
        })
    }

    pub fn forward(&self, audio_mel: &Tensor, audio_mel_mask: &Tensor) -> Result<(Tensor, Tensor)> {
        let (mut audio_encodings, mut current_mask) = self
            .subsample_conv_projection
            .forward(audio_mel, audio_mel_mask)?;

        for block in &self.conformer {
            audio_encodings = block.forward(&audio_encodings, &current_mask)?;
        }

        if self.conf_reduction_factor > 1 {
            let stride = self.conf_reduction_factor;
            let reduced_len = audio_encodings.dim(1)?.div_ceil(stride);
            let indices = Tensor::arange(
                0f32,
                audio_encodings.dim(1)? as f32,
                audio_encodings.device(),
            )?
            .affine(stride as f64, 0.0)?
            .to_dtype(DType::I64)?;
            let max_idx = audio_encodings.dim(1)? as i64 - 1;
            let indices = indices
                .narrow(0, 0, reduced_len.min(indices.dim(0)?))?
                .clamp(0, max_idx)?;

            audio_encodings = audio_encodings.index_select(&indices, 1)?;
            current_mask = current_mask.index_select(&indices, 1)?;
        }

        if let Some(ref output_proj) = self.output_proj {
            audio_encodings = output_proj.forward(&audio_encodings)?;
        }

        let enc_len = audio_encodings.dim(1)?;
        let mask_len = current_mask.dim(1)?;
        if mask_len != enc_len {
            if enc_len < mask_len {
                current_mask = current_mask.narrow(1, 0, enc_len)?;
            } else {
                current_mask = current_mask.pad_with_zeros(1, 0, enc_len - mask_len)?;
            }
        }

        let valid_mask = current_mask.eq(0.0)?;
        let zeros = Tensor::zeros_like(&audio_encodings)?;
        let audio_encodings = valid_mask
            .unsqueeze(D::Minus1)?
            .broadcast_as(audio_encodings.shape())?
            .where_cond(&audio_encodings, &zeros)?;

        Ok((audio_encodings, current_mask))
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        // SSCP
        let uvb_sscp = uvb.pp("subsample_conv_projection");
        let uvb_l0 = uvb_sscp.pp("layer0");
        uvb_l0
            .pp("conv")
            .add(&self.subsample_conv_projection.conv_0.conv);
        uvb_l0
            .pp("norm")
            .add(&self.subsample_conv_projection.conv_0.norm);
        let uvb_l1 = uvb_sscp.pp("layer1");
        uvb_l1
            .pp("conv")
            .add(&self.subsample_conv_projection.conv_1.conv);
        uvb_l1
            .pp("norm")
            .add(&self.subsample_conv_projection.conv_1.norm);
        uvb_sscp
            .pp("input_proj_linear")
            .add(&self.subsample_conv_projection.input_proj_linear);

        // Conformer blocks
        let uvb_layers = uvb.pp("layers");
        for (i, block) in self.conformer.iter().enumerate() {
            let uvb_block = uvb_layers.pp(i);

            // Attention
            let uvb_attn = uvb_block.pp("self_attn");
            uvb_attn
                .pp("q_proj")
                .extend(block.attention.attn.q_proj.residual_tensors());
            uvb_attn
                .pp("k_proj")
                .extend(block.attention.attn.k_proj.residual_tensors());
            uvb_attn
                .pp("v_proj")
                .extend(block.attention.attn.v_proj.residual_tensors());
            uvb_attn.add_tensor("per_dim_scale", block.attention.attn._per_dim_scale.clone());
            uvb_attn
                .pp("relative_k_proj")
                .add(&block.attention.attn.relative_position_embedding.pos_proj);
            uvb_attn
                .pp("post")
                .extend(block.attention.post.residual_tensors());

            // Attention norms
            uvb_block
                .pp("norm_pre_attn")
                .add(&block.attention.pre_attn_norm);
            uvb_block
                .pp("norm_post_attn")
                .add(&block.attention.post_norm);

            // Feed-forward 1
            let uvb_ff1 = uvb_block.pp("feed_forward1");
            uvb_ff1
                .pp("pre_layer_norm")
                .add(&block.ffw_layer_start.pre_layer_norm);
            uvb_ff1
                .pp("ffw_layer_1")
                .extend(block.ffw_layer_start.ffw_layer_1.residual_tensors());
            uvb_ff1
                .pp("ffw_layer_2")
                .extend(block.ffw_layer_start.ffw_layer_2.residual_tensors());
            uvb_ff1
                .pp("post_layer_norm")
                .add(&block.ffw_layer_start.post_layer_norm);

            // Light conv 1d
            let uvb_lconv = uvb_block.pp("lconv1d");
            uvb_lconv
                .pp("pre_layer_norm")
                .add(&block.lconv1d.pre_layer_norm);
            uvb_lconv
                .pp("linear_start")
                .extend(block.lconv1d.linear_start.residual_tensors());
            uvb_lconv
                .pp("depthwise_conv1d")
                .add(&block.lconv1d.depthwise_conv1d);
            uvb_lconv.pp("conv_norm").add(&block.lconv1d.conv_norm);
            uvb_lconv
                .pp("linear_end")
                .extend(block.lconv1d.linear_end.residual_tensors());

            // Feed-forward 2
            let uvb_ff2 = uvb_block.pp("feed_forward2");
            uvb_ff2
                .pp("pre_layer_norm")
                .add(&block.ffw_layer_end.pre_layer_norm);
            uvb_ff2
                .pp("ffw_layer_1")
                .extend(block.ffw_layer_end.ffw_layer_1.residual_tensors());
            uvb_ff2
                .pp("ffw_layer_2")
                .extend(block.ffw_layer_end.ffw_layer_2.residual_tensors());
            uvb_ff2
                .pp("post_layer_norm")
                .add(&block.ffw_layer_end.post_layer_norm);

            // Block output norm
            uvb_block.pp("norm_out").add(&block.norm);
        }

        // Output projection
        if let Some(ref proj) = self.output_proj {
            uvb.pp("output_proj").add(proj);
        }

        uvb.to_safetensors()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use inference_nn::testing::load_synthesized;

    use super::*;
    use crate::audio_attention::pin::{Expected, assert_padded_clip};

    // blocked local attention with its relative position bias over a padded clip, on made-up weights
    #[test]
    fn attention_over_a_padded_clip_is_pinned() -> anyhow::Result<()> {
        let cfg: Gemma4AudioConfig = inference_json::from_str(
            r#"{"hidden_size": 16, "conf_num_attention_heads": 2, "conf_attention_chunk_size": 4,
                "conf_attention_context_left": 5, "conf_attention_context_right": 1,
                "use_clipped_linears": false}"#,
        )?;
        let (attn, _) = load_synthesized(&[], HashMap::new(), DType::F32, |vb| {
            Ok(Gemma4AudioAttention::new(&cfg, vb)?)
        })?;
        let expected = Expected {
            probes: [-0.14363255, 0.025407953, -0.03793859, -0.09744155],
            sum: 2.6534584,
            l2: 0.7963559,
        };
        assert_padded_clip(cfg.hidden_size, |x, mask| attn.forward(x, mask), &expected)
    }
}
