#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_quant::{Convolution, QuantMethod, ShardedVarBuilder};
use inference_tensor::nn::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, ModuleT};
use inference_tensor::{D, DType, Module, Result, Tensor, bail};
use std::sync::Arc;

use crate::audio_attention::{LocalAttention, RelativePositionEmbedding};
use crate::{
    layers::{RmsNorm, conv1d_no_bias, conv2d_no_bias},
    utils::unvarbuilder::UnVarBuilder,
};

use super::config::Gemma3nAudioConfig;

/// Gemma3n Cumulative Group Normalization
/// Applies Group Normalization cumulatively over the time dimension.
pub struct Gemma3nCumulativeGroupNorm {
    num_channels: usize,
    feature_dims: Vec<usize>,
    eps: f64,
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    reduction_axes: Vec<usize>,
}

impl Gemma3nCumulativeGroupNorm {
    pub fn new(
        num_channels: usize,
        feature_dims: Vec<usize>,
        eps: f64,
        use_scale: bool,
        use_bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let weight = if use_scale {
            Some(vb.get(num_channels, "weight")?)
        } else {
            None
        };

        let bias = if use_bias {
            Some(vb.get(num_channels, "bias")?)
        } else {
            None
        };

        let reduction_axes: Vec<usize> = (2..2 + feature_dims.len() + 1).collect();

        Ok(Self {
            num_channels,
            feature_dims,
            eps,
            weight,
            bias,
            reduction_axes,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // Pre-computed expected suffix to avoid repeated allocations
        let expected_suffix_len = self.feature_dims.len() + 1;
        let expected_suffix = self
            .feature_dims
            .iter()
            .cloned()
            .chain(std::iter::once(self.num_channels))
            .collect::<Vec<_>>();
        let actual_suffix = x.dims()[2..].to_vec();
        if actual_suffix != expected_suffix {
            bail!(
                "Input tensor shape suffix {:?} does not match expected suffix {:?}",
                actual_suffix,
                expected_suffix
            );
        }

        let input_dtype = x.dtype();
        let calc_dtype = DType::F32;
        let x_calc = x.to_dtype(calc_dtype)?;

        // Prepare broadcastable mask
        let mask_calc = if let Some(mask) = mask {
            // Mask should be [B, T]
            if mask.dims() != [x.dims()[0], x.dims()[1]] {
                bail!(
                    "Mask shape {:?} must match input Batch/Time dimensions {:?}",
                    mask.dims(),
                    &[x.dims()[0], x.dims()[1]]
                );
            }
            // Expand mask from [B, T] to [B, T, 1, ..., 1] for broadcasting
            let mut mask_shape = mask.dims().to_vec();
            mask_shape.extend(std::iter::repeat_n(1, expected_suffix_len));
            mask.reshape(mask_shape)?.to_dtype(calc_dtype)?
        } else {
            Tensor::ones_like(&x_calc)?
        };

        // Mask the input for sum calculation: only valid elements contribute
        let x_masked_for_sum = x_calc.broadcast_mul(&mask_calc)?;

        // Reduction operations
        let sum_values_at_t = self
            .reduction_axes
            .iter()
            .rev()
            .try_fold(x_masked_for_sum, |acc, &axis| acc.sum_keepdim(axis))?;

        let cum_sum_values = sum_values_at_t.cumsum(1)?;

        // Reduction operations for mask
        let elements_in_group_at_t = self
            .reduction_axes
            .iter()
            .rev()
            .try_fold(mask_calc.clone(), |acc, &axis| acc.sum_keepdim(axis))?;

        let cum_count_elements = elements_in_group_at_t.cumsum(1)?;
        let safe_cum_count_elements = cum_count_elements.clamp(1.0, f64::INFINITY)?;

        let cum_mean = cum_sum_values.div(&safe_cum_count_elements)?;

        let squared_diff_from_mean = (x_calc.broadcast_sub(&cum_mean))?.sqr()?;
        // Fuse reduction operations for squared differences
        let sum_sq_diff_at_t = self.reduction_axes.iter().rev().try_fold(
            squared_diff_from_mean.broadcast_mul(&mask_calc)?,
            |acc, &axis| acc.sum_keepdim(axis),
        )?;

        let cum_sum_sq_diff = sum_sq_diff_at_t.cumsum(1)?;

        let cum_variance = cum_sum_sq_diff.div(&safe_cum_count_elements)?;

        let normalized_x = x_calc.broadcast_sub(&cum_mean)?.broadcast_mul(
            &(cum_variance
                .broadcast_add(
                    &Tensor::new(self.eps as f32, cum_variance.device())?.to_dtype(calc_dtype)?,
                )?
                .recip()?
                .sqrt())?,
        )?;

        let mut result = normalized_x;
        let dims_len = x.dims().len();

        if let Some(weight) = &self.weight {
            let scale = weight.to_dtype(calc_dtype)?;
            // Optimize reshape by reusing shape vector
            let mut scale_shape = vec![1; dims_len];
            scale_shape[dims_len - 1] = self.num_channels;
            let scale = scale.reshape(scale_shape)?;
            result = result.broadcast_mul(&scale)?;
        }

        if let Some(bias) = &self.bias {
            let bias = bias.to_dtype(calc_dtype)?;
            // Reuse dims_len from earlier
            let mut bias_shape = vec![1; dims_len];
            bias_shape[dims_len - 1] = self.num_channels;
            let bias = bias.reshape(bias_shape)?;
            result = result.broadcast_add(&bias)?;
        }

        let final_output = result.broadcast_mul(&mask_calc)?;

        final_output.to_dtype(input_dtype)
    }
}

// Gemma 3n's timescales, computed in f32 on the device as HF does
fn gemma3n_inv_timescales(channels: usize, device: &inference_tensor::Device) -> Result<Tensor> {
    let min_timescale = 1.0_f32;
    let max_timescale = 1.0e4_f32;
    let num_timescales = channels / 2;
    let log_timescale_increment =
        (max_timescale / min_timescale).ln() / (num_timescales as f32 - 1.0).max(1.0);
    Tensor::arange(0, num_timescales as i64, device)?
        .to_dtype(DType::F32)?
        .affine(-log_timescale_increment as f64, 0.0)?
        .exp()?
        .affine(min_timescale as f64, 0.0)?
        .unsqueeze(0)?
        .unsqueeze(0)
}

/// Gemma3n Audio Attention
pub struct Gemma3nAudioAttention {
    num_heads: usize,
    head_dim: usize,
    local: LocalAttention,
    pub relative_position_embedding: RelativePositionEmbedding,
    _per_dim_scale: Tensor,
    pub q_proj: Arc<dyn QuantMethod>,
    pub k_proj: Arc<dyn QuantMethod>,
    pub v_proj: Arc<dyn QuantMethod>,
    q_scale: f64,
    per_dim_scale_softplus: Tensor,
}

impl Gemma3nAudioAttention {
    pub fn new(config: &Gemma3nAudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let num_heads = config.conf_num_attention_heads;
        let hidden_size = config.hidden_size;
        let head_dim = hidden_size / num_heads;

        let pos_proj = inference_quant::linear_no_bias(
            hidden_size,
            num_heads * head_dim,
            &None,
            vb.pp("relative_position_embedding").pp("pos_proj"),
        )?;
        let relative_position_embedding = RelativePositionEmbedding::new(
            num_heads,
            head_dim,
            pos_proj,
            gemma3n_inv_timescales(hidden_size, vb.device())?,
            config.conf_attention_context_left.saturating_sub(1),
            config.conf_attention_context_right,
        )?;
        // per_dim_scale is a learnable parameter, not zeros
        let per_dim_scale = vb.get(head_dim, "per_dim_scale")?;

        let q_proj = inference_quant::linear_no_bias(
            hidden_size,
            num_heads * head_dim,
            &None,
            vb.pp("q_proj"),
        )?;
        let k_proj = inference_quant::linear_no_bias(
            hidden_size,
            num_heads * head_dim,
            &None,
            vb.pp("k_proj"),
        )?;
        let v_proj = inference_quant::linear_no_bias(
            hidden_size,
            num_heads * head_dim,
            &None,
            vb.pp("v_proj"),
        )?;

        let q_scale = (head_dim as f64).powf(-0.5);
        let r_softplus_0 = 1.0 / (1.0_f64 + 0.0_f64.exp()).ln(); // 1 / ln(2)
        let q_scale = q_scale * r_softplus_0;

        let local = LocalAttention::new(
            num_heads,
            head_dim,
            config.conf_attention_chunk_size,
            config.conf_attention_context_left,
            config.conf_attention_context_right,
            config.conf_attention_logit_cap,
            config.conf_attention_invalid_logits_value,
            vb.device(),
        )?;

        let per_dim_scale_softplus = {
            let ones = Tensor::ones_like(&per_dim_scale)?.to_dtype(DType::F32)?;
            let exp_scale: Tensor = per_dim_scale.to_dtype(DType::F32)?.exp()?;
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
            per_dim_scale_softplus,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (query_states, key_states, value_states) =
            crate::ops::qkv_projections(x, &*self.q_proj, &*self.k_proj, &*self.v_proj)?;

        let (b, t) = match x.dims() {
            &[b, t, _] => (b, t),
            _ => bail!("Expected input to have 3 dimensions"),
        };

        let query_states = query_states.reshape((b, t, self.num_heads, self.head_dim))?;
        let key_states = key_states.reshape((b, t, self.num_heads, self.head_dim))?;
        let value_states = value_states.reshape((b, t, self.num_heads, self.head_dim))?;

        let per_dim_scale_sp_broadcast = self
            .per_dim_scale_softplus
            .to_device(x.device())?
            .reshape((1, 1, 1, self.head_dim))?
            .to_dtype(query_states.dtype())?;
        let query_states = query_states
            .affine(self.q_scale, 0.0)?
            .broadcast_mul(&per_dim_scale_sp_broadcast)?;

        self.local.forward(
            &self.relative_position_embedding,
            &query_states,
            &key_states,
            &value_states,
            mask,
        )
    }
}

/// SSCP Convolution Block
pub struct Gemma3nAudioSSCPConvBlock {
    pub conv: Conv2d,
    pub norm: Gemma3nCumulativeGroupNorm,
    manual_padding: (usize, usize, usize, usize),
}

impl Gemma3nAudioSSCPConvBlock {
    pub fn new(
        idx: usize,
        input_freq_dim: usize,
        config: &Gemma3nAudioConfig,
        manual_padding: (usize, usize, usize, usize),
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let in_channels = if idx == 0 {
            1
        } else {
            config.sscp_conv_channel_size[idx - 1]
        };
        let out_channels = config.sscp_conv_channel_size[idx];
        let kernel_h = config.sscp_conv_kernel_size[idx][0];
        let kernel_w = config.sscp_conv_kernel_size[idx][1];
        let stride_h = config.sscp_conv_stride_size[idx][0];
        let stride_w = config.sscp_conv_stride_size[idx][1];

        let conv = conv2d_no_bias(
            in_channels,
            out_channels,
            kernel_h.min(kernel_w),
            Conv2dConfig {
                stride: stride_h.min(stride_w),
                padding: 0,
                dilation: 1,
                groups: 1,
            },
            vb.pp("conv"),
        )?;

        // Calculate output frequency dimension after convolution
        let f_in_padded = input_freq_dim + manual_padding.0 + manual_padding.1;
        // Use ceil-division to exactly match PyTorch's convolution size formula
        // out = floor((in + 2*pad - kernel) / stride) + 1  - with manual padding.
        // Because we materialise the padding ourselves, the equivalent integer
        // expression is ceil((f_in_padded - kernel_w) / stride_w).
        let f_out_conv = (f_in_padded + stride_w - kernel_w) / stride_w;

        let norm = Gemma3nCumulativeGroupNorm::new(
            out_channels,
            vec![f_out_conv],
            config.sscp_conv_group_norm_eps,
            true,
            false,
            vb.pp("norm"),
        )?;

        Ok(Self {
            conv,
            norm,
            manual_padding,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Apply manual padding
        let x_padded = x
            .pad_with_zeros(D::Minus1, self.manual_padding.0, self.manual_padding.1)?
            .pad_with_zeros(D::Minus2, self.manual_padding.2, self.manual_padding.3)?;

        // Apply Conv2d - expecting NCHW format
        let x_conv = self.conv.forward_t(&x_padded, false)?;

        // Reshape for normalization: [B, C_out, T_out, F_out] -> [B, T_out, F_out, C_out]
        let x_for_norm = x_conv.permute((0, 2, 3, 1))?;

        // Apply normalization
        let x_normed = self.norm.forward(&x_for_norm, None)?;

        // Reshape back to [B, C_out, T_out, F_out] and apply ReLU
        x_normed.permute((0, 3, 1, 2))?.relu()
    }
}

/// Sub-sample Convolution Projection
pub struct Gemma3nAudioSubSampleConvProjection {
    pub conv_0: Gemma3nAudioSSCPConvBlock,
    pub conv_1: Gemma3nAudioSSCPConvBlock,
    pub input_proj_linear: Arc<dyn QuantMethod>,
}

impl Gemma3nAudioSubSampleConvProjection {
    pub fn new(config: &Gemma3nAudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let mut current_f_for_block_input = config.input_feat_size;
        let mut calculated_block_padding = Vec::new();
        let mut calculated_f_out_dims = Vec::new();

        for i in 0..2 {
            let kernel_h = config.sscp_conv_kernel_size[i][0];
            let kernel_w = config.sscp_conv_kernel_size[i][1];
            let _stride_h = config.sscp_conv_stride_size[i][0];
            let stride_w = config.sscp_conv_stride_size[i][1];

            // Padding for Time (reverse causal)
            let pad_t_top = 0;
            let pad_t_bottom = kernel_h - 1;

            // Frequency padding
            let pad_f_left = 1;
            let pad_f_right = 1;

            let manual_padding_tuple = (pad_f_left, pad_f_right, pad_t_top, pad_t_bottom);
            calculated_block_padding.push(manual_padding_tuple);

            // Calculate output frequency dimension
            let f_in_padded = current_f_for_block_input + pad_f_left + pad_f_right;
            let f_out_after_conv = (f_in_padded + stride_w - kernel_w) / stride_w;
            calculated_f_out_dims.push(f_out_after_conv);
            current_f_for_block_input = f_out_after_conv;
        }

        let conv_0 = Gemma3nAudioSSCPConvBlock::new(
            0,
            config.input_feat_size,
            config,
            calculated_block_padding[0],
            vb.pp("conv_0"),
        )?;

        let conv_1 = Gemma3nAudioSSCPConvBlock::new(
            1,
            calculated_f_out_dims[0],
            config,
            calculated_block_padding[1],
            vb.pp("conv_1"),
        )?;

        let final_c_out = config.sscp_conv_channel_size[1];
        let final_f_out = calculated_f_out_dims[1];
        let input_proj_in_features = final_c_out * final_f_out;

        let input_proj_linear = inference_quant::linear_no_bias(
            input_proj_in_features,
            config.hidden_size,
            &None,
            vb.pp("input_proj_linear"),
        )?;

        Ok(Self {
            conv_0,
            conv_1,
            input_proj_linear,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, T, F_in]
        // Reshape to [B, 1, T, F_in]
        let x_reshaped = x.unsqueeze(1)?;

        let x = self.conv_0.forward(&x_reshaped)?;
        let x = self.conv_1.forward(&x)?;

        // x: [B, C_out, T_out, F_out]
        let (b, c_out, t_out, f_out) = match x.dims() {
            &[b, c, t, f] => (b, c, t, f),
            _ => bail!("Expected conv output to have 4 dimensions"),
        };

        // Permute to [B, T_out, F_out, C_out] then flatten F_out and C_out
        let x_transposed = x.transpose(1, 2)?.transpose(2, 3)?;
        let output_flattened = x_transposed.reshape((b, t_out, f_out * c_out))?;

        self.input_proj_linear.forward(&output_flattened)
    }
}

/// Conformer Attention Module
pub struct Gemma3nAudioConformerAttention {
    pub pre_attn_norm: RmsNorm,
    pub attn: Gemma3nAudioAttention,
    pub post: Arc<dyn QuantMethod>,
    pub post_norm: RmsNorm,
    hidden_size: usize, // Cache for reshape operations
}

impl Gemma3nAudioConformerAttention {
    pub fn new(config: &Gemma3nAudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let pre_attn_norm = RmsNorm::new_gemma_3n(
            config.hidden_size,
            config.rms_norm_eps,
            true,
            vb.pp("pre_attn_norm"),
        )?;
        let attn = Gemma3nAudioAttention::new(config, vb.pp("attn"))?;
        let post = inference_quant::linear_no_bias(
            config.hidden_size,
            config.hidden_size,
            &None,
            vb.pp("post"),
        )?;
        let post_norm = RmsNorm::new_gemma_3n(
            config.hidden_size,
            config.rms_norm_eps,
            true,
            vb.pp("post_norm"),
        )?;

        Ok(Self {
            pre_attn_norm,
            attn,
            post,
            post_norm,
            hidden_size: config.hidden_size,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let audio_encodings_input_to_attn = x;
        let audio_encodings_norm = self.pre_attn_norm.forward(x)?;

        // attn output: [B, T, NumHeads, HeadDim]
        let audio_encodings_attn_out = self.attn.forward(&audio_encodings_norm, mask)?;

        // Reshape to [B, T, NumHeads * HeadDim]
        let (b, t, _, _) = audio_encodings_attn_out.dims4()?;
        let audio_encodings_reshaped =
            audio_encodings_attn_out.reshape((b, t, self.hidden_size))?;

        let x = self.post.forward(&audio_encodings_reshaped)?;

        audio_encodings_input_to_attn.broadcast_add(&self.post_norm.forward(&x)?)
    }
}

/// Conformer Feed-Forward Module
pub struct Gemma3nAudioConformerFeedForward {
    scale: f64,
    pub pre_layer_norm: RmsNorm,
    pub ffw_layer_1: Arc<dyn QuantMethod>,
    pub ffw_layer_2: Arc<dyn QuantMethod>,
    pub post_layer_norm: RmsNorm,
}

impl Gemma3nAudioConformerFeedForward {
    pub fn new(config: &Gemma3nAudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let pre_layer_norm = RmsNorm::new_gemma_3n(
            config.hidden_size,
            config.rms_norm_eps,
            true,
            vb.pp("pre_layer_norm"),
        )?;
        let ffw_layer_1 = inference_quant::linear_no_bias(
            config.hidden_size,
            config.hidden_size * 4,
            &None,
            vb.pp("ffw_layer_1"),
        )?;
        let ffw_layer_2 = inference_quant::linear_no_bias(
            config.hidden_size * 4,
            config.hidden_size,
            &None,
            vb.pp("ffw_layer_2"),
        )?;
        let post_layer_norm = RmsNorm::new_gemma_3n(
            config.hidden_size,
            config.rms_norm_eps,
            true,
            vb.pp("post_layer_norm"),
        )?;

        Ok(Self {
            scale: config.conf_residual_weight,
            pre_layer_norm,
            ffw_layer_1,
            ffw_layer_2,
            post_layer_norm,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.pre_layer_norm.forward(x)?;
        let x = self.ffw_layer_1.forward(&x)?;
        let x = inference_tensor::nn::ops::silu(&x)?;
        let x = self.ffw_layer_2.forward(&x)?;
        let x = self.post_layer_norm.forward(&x)?;

        residual.broadcast_add(&(x * self.scale)?)
    }
}

/// Lightweight 1D Convolution Module
pub struct Gemma3nAudioConformerLightConv1d {
    pub pre_layer_norm: RmsNorm,
    pub linear_start: Arc<dyn QuantMethod>,
    pub depthwise_conv1d: Conv1d,
    pub conv_norm: RmsNorm,
    pub linear_end: Arc<dyn QuantMethod>,
    causal_padding: usize,
}

impl Gemma3nAudioConformerLightConv1d {
    pub fn new(config: &Gemma3nAudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let pre_layer_norm = RmsNorm::new_gemma_3n(
            config.hidden_size,
            config.rms_norm_eps,
            true,
            vb.pp("pre_layer_norm"),
        )?;
        let linear_start = inference_quant::linear_no_bias(
            config.hidden_size,
            config.hidden_size * 2,
            &None,
            vb.pp("linear_start"),
        )?;

        let depthwise_conv1d = conv1d_no_bias(
            config.hidden_size,
            config.hidden_size,
            config.conf_conv_kernel_size,
            Conv1dConfig {
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: config.hidden_size,
            },
            vb.pp("depthwise_conv1d").set_dtype(DType::F32),
        )?;

        let conv_norm = RmsNorm::new_gemma_3n(
            config.hidden_size,
            config.rms_norm_eps,
            true,
            vb.pp("conv_norm"),
        )?;
        let linear_end = inference_quant::linear_no_bias(
            config.hidden_size,
            config.hidden_size,
            &None,
            vb.pp("linear_end"),
        )?;
        let causal_padding = config.conf_conv_kernel_size - 1;

        Ok(Self {
            pre_layer_norm,
            linear_start,
            depthwise_conv1d,
            conv_norm,
            linear_end,
            causal_padding,
        })
    }

    pub fn forward(&self, audio_encodings: &Tensor) -> Result<Tensor> {
        let audio_encodings_residual = audio_encodings;

        let audio_encodings = self.pre_layer_norm.forward(audio_encodings)?;
        let audio_encodings = self.linear_start.forward(&audio_encodings)?;
        // Implement GLU manually: split tensor in half and apply gating
        let chunks = audio_encodings.chunk(2, D::Minus1)?;
        let audio_encodings =
            chunks[0].broadcast_mul(&inference_tensor::nn::ops::sigmoid(&chunks[1])?)?;

        // Permute for Conv1d: [B, T, D] -> [B, D, T]
        let audio_encodings_transposed = audio_encodings.transpose(D::Minus1, D::Minus2)?;

        // Apply manual causal padding
        let audio_encodings_padded =
            audio_encodings_transposed.pad_with_zeros(D::Minus1, self.causal_padding, 0)?;

        // Conv1d expects NCW format, apply directly
        let conv_input = audio_encodings_padded.to_dtype(DType::F32)?;
        let audio_encodings_conv = Convolution
            .forward_1d(&self.depthwise_conv1d, &conv_input)?
            .to_dtype(audio_encodings_padded.dtype())?;

        // Permute back: [B, D, T] -> [B, T, D]
        let audio_encodings = audio_encodings_conv.transpose(D::Minus2, D::Minus1)?;

        let audio_encodings = self.conv_norm.forward(&audio_encodings)?;
        let audio_encodings = inference_tensor::nn::ops::silu(&audio_encodings)?;
        let audio_encodings = self.linear_end.forward(&audio_encodings)?;

        audio_encodings_residual.broadcast_add(&audio_encodings)
    }
}

/// Conformer Block
pub struct Gemma3nAudioConformerBlock {
    pub ffw_layer_start: Gemma3nAudioConformerFeedForward,
    pub attention: Gemma3nAudioConformerAttention,
    pub lconv1d: Gemma3nAudioConformerLightConv1d,
    pub ffw_layer_end: Gemma3nAudioConformerFeedForward,
    pub norm: RmsNorm,
}

impl Gemma3nAudioConformerBlock {
    pub fn new(config: &Gemma3nAudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let ffw_layer_start =
            Gemma3nAudioConformerFeedForward::new(config, vb.pp("ffw_layer_start"))?;
        let attention = Gemma3nAudioConformerAttention::new(config, vb.pp("attention"))?;
        let lconv1d = Gemma3nAudioConformerLightConv1d::new(config, vb.pp("lconv1d"))?;
        let ffw_layer_end = Gemma3nAudioConformerFeedForward::new(config, vb.pp("ffw_layer_end"))?;
        let norm =
            RmsNorm::new_gemma_3n(config.hidden_size, config.rms_norm_eps, true, vb.pp("norm"))?;

        Ok(Self {
            ffw_layer_start,
            attention,
            lconv1d,
            ffw_layer_end,
            norm,
        })
    }

    pub fn forward(&self, audio_encodings: &Tensor, audio_mel_mask: &Tensor) -> Result<Tensor> {
        let audio_encodings = self.ffw_layer_start.forward(audio_encodings)?;
        let audio_encodings = self.attention.forward(&audio_encodings, audio_mel_mask)?;

        // Apply mask for lconv1d
        let validity_mask_for_lconv = audio_mel_mask.eq(0.0)?; // True for valid
        let audio_encodings_for_lconv_input = audio_encodings.broadcast_mul(
            &validity_mask_for_lconv
                .unsqueeze(D::Minus1)?
                .to_dtype(audio_encodings.dtype())?,
        )?;

        let audio_encodings = self.lconv1d.forward(&audio_encodings_for_lconv_input)?;
        let audio_encodings = self.ffw_layer_end.forward(&audio_encodings)?;

        self.norm.forward(&audio_encodings)
    }
}

/// Main Audio Model
pub struct AudioModel {
    pub subsample_conv_projection: Gemma3nAudioSubSampleConvProjection,
    pub conformer: Vec<Gemma3nAudioConformerBlock>,
    sscp_conv_stride_size: Vec<Vec<usize>>,
    conf_reduction_factor: usize,
}

impl AudioModel {
    pub fn new(config: &Gemma3nAudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let subsample_conv_projection =
            Gemma3nAudioSubSampleConvProjection::new(config, vb.pp("subsample_conv_projection"))?;

        let mut conformer = Vec::with_capacity(config.conf_num_hidden_layers);
        for i in 0..config.conf_num_hidden_layers {
            conformer.push(Gemma3nAudioConformerBlock::new(
                config,
                vb.pp(format!("conformer.{i}")),
            )?);
        }

        Ok(Self {
            subsample_conv_projection,
            conformer,
            sscp_conv_stride_size: config.sscp_conv_stride_size.clone(),
            conf_reduction_factor: config.conf_reduction_factor,
        })
    }

    pub fn forward(&self, audio_mel: &Tensor, audio_mel_mask: &Tensor) -> Result<(Tensor, Tensor)> {
        // audio_mel: [B, T, F]
        let mut audio_encodings = self.subsample_conv_projection.forward(audio_mel)?;

        // Subsample the mask to match audio_encodings time dimension
        let t_sub = audio_encodings.dim(1)?;

        let mut time_stride_product = 1;
        for i in 0..self.sscp_conv_stride_size.len() {
            time_stride_product *= self.sscp_conv_stride_size[i][0];
        }

        // Create indices for gathering from the original mask
        // Use f32 for the affine operation, then convert to i64 for indexing
        let indices = Tensor::arange(0f32, t_sub as f32, audio_mel_mask.device())?
            .affine(time_stride_product as f64, 0.0)?
            .to_dtype(DType::I64)?;
        let max_idx = audio_mel_mask.dim(1)? as i64 - 1;
        let indices = indices.clamp(0i64, max_idx)?;

        // Expand indices for batch compatibility
        let indices = if audio_mel_mask.dims().len() > 1 && indices.dims().len() == 1 {
            indices
                .unsqueeze(0)?
                .broadcast_as((audio_mel_mask.dim(0)?, indices.dim(0)?))?
        } else if audio_mel_mask.dims().len() == indices.dims().len()
            && audio_mel_mask.dim(0)? == 1
            && indices.dim(0)? != 1
            && t_sub == indices.dim(0)?
        {
            indices.unsqueeze(0)?
        } else {
            indices
        };

        // Use index_select instead of gather for Metal compatibility
        let mut current_mask = if indices.dims().len() == 1 {
            // 1D indices case
            audio_mel_mask.index_select(&indices, 1)?
        } else {
            // 2D indices case - need to handle batch dimension
            let batch_size = audio_mel_mask.dim(0)?;
            let mut masks = Vec::with_capacity(batch_size);
            let indices_single_batch = indices.dim(0)? == 1;
            let indices_squeezed = if indices_single_batch {
                Some(indices.squeeze(0)?)
            } else {
                None
            };

            for b in 0..batch_size {
                let batch_mask = audio_mel_mask.get(b)?;
                let batch_indices = if indices_single_batch {
                    indices_squeezed.as_ref().unwrap()
                } else {
                    &indices.get(b)?
                };
                let selected = batch_mask.index_select(batch_indices, 0)?;
                masks.push(selected.unsqueeze(0)?);
            }
            Tensor::cat(&masks, 0)?
        };

        // Ensure mask length matches feature length
        if current_mask.dim(1)? != t_sub {
            if current_mask.dim(1)? > t_sub {
                current_mask = current_mask.narrow(1, 0, t_sub)?;
            } else {
                let padding_needed = t_sub - current_mask.dim(1)?;
                current_mask = current_mask.pad_with_zeros(1, 0, padding_needed)?;
            }
        }

        // Apply conformer blocks
        for block in &self.conformer {
            audio_encodings = block.forward(&audio_encodings, &current_mask)?;
        }

        // Apply reduction factor if specified
        if self.conf_reduction_factor > 1 {
            let stride = self.conf_reduction_factor;
            let indices = Tensor::arange(
                0f32,
                audio_encodings.dim(1)? as f32,
                audio_encodings.device(),
            )?
            .affine(stride as f64, 0.0)?
            .to_dtype(DType::I64)?;
            let max_idx = audio_encodings.dim(1)? as i64 - 1;
            let indices = indices
                .narrow(
                    0,
                    0,
                    (audio_encodings.dim(1)? / stride).min(indices.dim(0)?),
                )?
                .clamp(0, max_idx)?;

            audio_encodings = audio_encodings.index_select(&indices, 1)?;
            current_mask = current_mask.index_select(&indices, 1)?;
        }

        // Final masking - mask is 1 for invalid positions, 0 for valid
        // We want to zero out invalid positions
        let valid_mask = current_mask.eq(0.0)?; // True for valid positions
        let zeros = Tensor::zeros_like(&audio_encodings)?;
        let audio_encodings = valid_mask
            .unsqueeze(D::Minus1)?
            .broadcast_as(audio_encodings.shape())?
            .where_cond(&audio_encodings, &zeros)?;

        Ok((audio_encodings, current_mask))
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        // Add subsample conv projection residual tensors
        let uvb_sscp = uvb.pp("subsample_conv_projection");

        // Add conv blocks
        let uvb_conv0 = uvb_sscp.pp("conv_0");
        uvb_conv0
            .pp("conv")
            .add(&self.subsample_conv_projection.conv_0.conv);
        if let Some(weight) = &self.subsample_conv_projection.conv_0.norm.weight {
            uvb_conv0.pp("norm").add_tensor("weight", weight.clone());
        }
        if let Some(bias) = &self.subsample_conv_projection.conv_0.norm.bias {
            uvb_conv0.pp("norm").add_tensor("bias", bias.clone());
        }

        let uvb_conv1 = uvb_sscp.pp("conv_1");
        uvb_conv1
            .pp("conv")
            .add(&self.subsample_conv_projection.conv_1.conv);
        if let Some(weight) = &self.subsample_conv_projection.conv_1.norm.weight {
            uvb_conv1.pp("norm").add_tensor("weight", weight.clone());
        }
        if let Some(bias) = &self.subsample_conv_projection.conv_1.norm.bias {
            uvb_conv1.pp("norm").add_tensor("bias", bias.clone());
        }

        // Add conformer block residual tensors
        for (i, block) in self.conformer.iter().enumerate() {
            let uvb_block = uvb.pp("conformer").pp(i);

            // Add all the norm layers
            uvb_block
                .pp("attention")
                .pp("pre_attn_norm")
                .add(&block.attention.pre_attn_norm);
            uvb_block
                .pp("attention")
                .pp("post_norm")
                .add(&block.attention.post_norm);

            // Add attention residual tensors
            let uvb_attn = uvb_block.pp("attention").pp("attn");

            // Add per_dim_scale tensor
            uvb_attn.add_tensor("per_dim_scale", block.attention.attn._per_dim_scale.clone());

            // Add relative position embedding tensors
            let uvb_rel_pos = uvb_attn.pp("relative_position_embedding");
            uvb_rel_pos.add_tensor(
                "inv_timescales",
                block
                    .attention
                    .attn
                    .relative_position_embedding
                    .inv_timescales
                    .clone(),
            );
            uvb_rel_pos.add_tensor(
                "pos_indices",
                block
                    .attention
                    .attn
                    .relative_position_embedding
                    .pos_indices
                    .clone(),
            );

            // Add attention mask tensors
            uvb_attn.add_tensor(
                "local_causal_valid_mask",
                block.attention.attn.local.local_causal_valid_mask.clone(),
            );
            uvb_attn.add_tensor(
                "invalid_logits_tensor",
                block.attention.attn.local.invalid_logits_tensor.clone(),
            );
            uvb_attn.add_tensor(
                "per_dim_scale_softplus",
                block.attention.attn.per_dim_scale_softplus.clone(),
            );

            // Add FFW layer norms
            uvb_block
                .pp("ffw_layer_start")
                .pp("pre_layer_norm")
                .add(&block.ffw_layer_start.pre_layer_norm);
            uvb_block
                .pp("ffw_layer_start")
                .pp("post_layer_norm")
                .add(&block.ffw_layer_start.post_layer_norm);
            uvb_block
                .pp("ffw_layer_end")
                .pp("pre_layer_norm")
                .add(&block.ffw_layer_end.pre_layer_norm);
            uvb_block
                .pp("ffw_layer_end")
                .pp("post_layer_norm")
                .add(&block.ffw_layer_end.post_layer_norm);

            // Add lconv1d tensors
            uvb_block
                .pp("lconv1d")
                .pp("pre_layer_norm")
                .add(&block.lconv1d.pre_layer_norm);
            uvb_block
                .pp("lconv1d")
                .pp("conv_norm")
                .add(&block.lconv1d.conv_norm);
            uvb_block
                .pp("lconv1d")
                .pp("depthwise_conv1d")
                .add(&block.lconv1d.depthwise_conv1d);

            // Add final norm
            uvb_block.pp("norm").add(&block.norm);
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
        let cfg: Gemma3nAudioConfig = inference_json::from_str(
            r#"{"hidden_size": 16, "conf_num_attention_heads": 2, "conf_attention_chunk_size": 4,
                "conf_attention_context_left": 5, "conf_attention_context_right": 1}"#,
        )?;
        let (attn, _) = load_synthesized(&[], HashMap::new(), DType::F32, |vb| {
            Ok(Gemma3nAudioAttention::new(&cfg, vb)?)
        })?;
        let expected = Expected {
            probes: [0.040065933, 0.011715145, -0.06335059, -0.048223395],
            sum: -0.6183169,
            l2: 0.7466255,
        };
        assert_padded_clip(cfg.hidden_size, |x, mask| attn.forward(x, mask), &expected)
    }
}
