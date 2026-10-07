//! Gemma 3's text model on the shared decoder; it builds its own masks so image tokens attend to each other both ways.

use inference_tensor::{Device, Result, Tensor};

use crate::{
    amoe::{AnyMoeBaseModelMixin, AnyMoeLoraTarget, MlpLayer},
    attention::{AttentionMask, FlashParams},
    decoder::{CausalLm, DecoderSpec, LayerMasks, NormKind, QkNorm, RopeKind},
    device_map::DeviceMappedMask,
    gemma2::SANDWICH_NORMS,
    kv_cache::EitherCache,
    layers::{CausalMaskConfig, CausalMasker},
    model::{IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata, NormalModel},
    paged_attention::{
        AttentionImplementation, ModelConfigMetadata, block_hash::MultimodalAttentionPolicy,
    },
};
use inference_quant::ShardedVarBuilder;

use super::config::Gemma3TextConfig;

impl Gemma3TextConfig {
    /// Every `sliding_window_pattern`-th layer is global with the global RoPE; the rest slide with the local one.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn decoder_spec(&self) -> DecoderSpec {
        DecoderSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            hidden_act: self.hidden_activation,
            rms_norm_eps: self.rms_norm_eps,
            rope: RopeKind::Gemma3 {
                theta: self.rope_theta,
                scaling: self.rope_scaling.clone(),
            },
            local_rope: Some(RopeKind::Default {
                theta: self.rope_local_base_freq as f32,
            }),
            max_position_embeddings: self.max_position_embeddings,
            qkv_bias: self.attention_bias,
            o_bias: self.attention_bias,
            qk_norm: Some(QkNorm::BeforeRope),
            layer_windows: (0..self.num_hidden_layers)
                .map(|layer_idx| {
                    (!(layer_idx + 1).is_multiple_of(self.sliding_window_pattern))
                        .then_some(self.sliding_window)
                })
                .collect(),
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
            norm: NormKind::Gemma,
            norm_names: SANDWICH_NORMS,
            attn_softcap: self.attn_logit_softcapping.map(|cap| cap as f32),
            softmax_scale: Some(1.0 / (self.query_pre_attn_scalar as f32).sqrt()),
            final_logit_softcap: self.final_logit_softcapping.map(|cap| cap as f32),
            embed_scale: Some((self.hidden_size as f64).sqrt()),
            ..Default::default()
        }
    }
}

fn select_paged_mm_prefix_path(
    requires_noncausal: bool,
    is_paged: bool,
    is_cuda: bool,
    flash_attn: bool,
    packed: bool,
    has_range_metadata: bool,
) -> Result<bool> {
    if requires_noncausal && packed && !has_range_metadata {
        inference_tensor::bail!(
            "packed Gemma 3 multimodal prefill is missing noncausal range metadata"
        );
    }
    Ok(requires_noncausal && is_paged && is_cuda && flash_attn && has_range_metadata)
}

pub struct TextModel {
    lm: CausalLm,
    sliding_window: usize,
    image_token_index: Option<usize>,
}

impl TextModel {
    pub fn new(
        cfg: &Gemma3TextConfig,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
        image_token_index: Option<usize>,
    ) -> Result<Self> {
        Ok(Self {
            lm: CausalLm::new(
                &cfg.decoder_spec(),
                vb,
                is_gptx,
                normal_loading_metadata,
                attention_mechanism,
            )?,
            sliding_window: cfg.sliding_window,
            image_token_index,
        })
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.lm.get_input_embeddings(input_ids)
    }

    pub fn supports_packed_prefill(&self) -> bool {
        NormalModel::supports_packed_prefill(&self.lm)
    }

    pub fn forward_embeds(
        &self,
        input_ids: &Tensor,
        xs: Tensor,
        ctx: &mut ModelForwardContext<'_>,
        has_images: bool,
    ) -> Result<Tensor> {
        let cache = NormalModel::cache(&self.lm).normal();
        let mask_cache = ctx.mask_cache(&cache.0);

        // Non-paged backends materialize the bidirectional image-token mask.
        let q_len = input_ids.dim(1)?;
        let is_non_causal_media_chunk = ctx.prompt_chunk_attention_policy()
            == MultimodalAttentionPolicy::NonCausal
            && q_len > 1;
        let has_bidirectional = has_images && self.image_token_index.is_some() && q_len > 1;
        let requires_noncausal =
            has_bidirectional || (is_non_causal_media_chunk && self.image_token_index.is_some());
        let has_range_metadata = ctx
            .paged_input_metadata()
            .is_some_and(|metadata| metadata.has_noncausal_mm_context);
        let use_paged_mm_prefix_path = select_paged_mm_prefix_path(
            requires_noncausal,
            ctx.is_paged(),
            xs.device().is_cuda(),
            crate::utils::using_flash_attn(),
            ctx.flash_params().packed,
            has_range_metadata,
        )?;

        let (attention_mask, sliding_attention_mask, layer_flash_params) = if (has_bidirectional
            || (is_non_causal_media_chunk && self.image_token_index.is_some()))
            && !use_paged_mm_prefix_path
        {
            // Build real masks (not flash dummies) with bidirectional regions for image tokens
            let image_token_index = self.image_token_index.unwrap();
            let causal_mask = CausalMasker.make_causal_mask(
                input_ids,
                &mask_cache,
                xs.dtype(),
                &CausalMaskConfig {
                    force_custom: true,
                    ..Default::default()
                },
            )?;
            let sliding_mask = CausalMasker.make_causal_mask(
                input_ids,
                &mask_cache,
                xs.dtype(),
                &CausalMaskConfig {
                    sliding_window: Some(self.sliding_window),
                    force_custom: true,
                },
            )?;

            // Apply bidirectional override for image tokens
            let attention_mask = match causal_mask {
                AttentionMask::Custom(m) => AttentionMask::Custom(
                    Self::apply_image_bidirectional_mask(&m, input_ids, image_token_index, None)?,
                ),
                other => other,
            };
            let sliding_attention_mask = match sliding_mask {
                AttentionMask::Custom(m) => {
                    AttentionMask::Custom(Self::apply_image_bidirectional_mask(
                        &m,
                        input_ids,
                        image_token_index,
                        Some(self.sliding_window),
                    )?)
                }
                other => other,
            };

            // Move to CPU (same optimization as normal path)
            let attention_mask = match attention_mask {
                AttentionMask::Custom(m) => AttentionMask::Custom(m.to_device(&Device::Cpu)?),
                other => other,
            };
            let sliding_attention_mask = match sliding_attention_mask {
                AttentionMask::Custom(m) => AttentionMask::Custom(m.to_device(&Device::Cpu)?),
                other => other,
            };

            // PagedAttention prompt chunking filter
            let keep_mask = ctx.is_first_prompt_chunk() || is_non_causal_media_chunk;
            let attention_mask = if keep_mask {
                attention_mask
            } else {
                AttentionMask::None
            };
            let sliding_attention_mask = if keep_mask {
                sliding_attention_mask
            } else {
                AttentionMask::None
            };

            // non-causal flash params, so the paged gather path keeps the bidirectional overrides in these masks
            (
                attention_mask,
                sliding_attention_mask,
                Some(FlashParams::empty(false)),
            )
        } else {
            // Standard path: use CausalMasker (returns dummy (1,1) with flash attention on CUDA)
            let attention_mask = CausalMasker.make_causal_mask(
                input_ids,
                &mask_cache,
                xs.dtype(),
                &CausalMaskConfig::default(),
            )?;
            let attention_mask = match attention_mask {
                AttentionMask::Custom(m) => AttentionMask::Custom(m.to_device(&Device::Cpu)?),
                other => other,
            };
            let is_first = ctx.is_first_prompt_chunk();
            let attention_mask = if is_first {
                attention_mask
            } else {
                AttentionMask::None
            };
            let sliding_attention_mask = CausalMasker.make_causal_mask(
                input_ids,
                &mask_cache,
                xs.dtype(),
                &CausalMaskConfig {
                    sliding_window: Some(self.sliding_window),
                    ..Default::default()
                },
            )?;
            let sliding_attention_mask = match sliding_attention_mask {
                AttentionMask::Custom(m) => AttentionMask::Custom(m.to_device(&Device::Cpu)?),
                other => other,
            };
            let sliding_attention_mask = if is_first {
                sliding_attention_mask
            } else {
                AttentionMask::None
            };

            (attention_mask, sliding_attention_mask, None)
        };
        drop(cache);

        let mapper = self.lm.stack_mapper();
        let masks = LayerMasks::new(
            Some(DeviceMappedMask::new(attention_mask, mapper)?),
            Some(DeviceMappedMask::new(sliding_attention_mask, mapper)?),
            layer_flash_params,
        );
        self.lm.forward_with_masks(xs, &masks, ctx)
    }

    /// Apply bidirectional attention override for image tokens within the same image group.
    /// Where both query and key positions are image tokens in the same contiguous group,
    /// the mask value is set to 0.0 (attend) instead of -inf (mask).
    fn apply_image_bidirectional_mask(
        causal_mask: &Tensor,
        input_ids: &Tensor,
        image_token_index: usize,
        sliding_window: Option<usize>,
    ) -> Result<Tensor> {
        // input_ids: (1, seq_len), causal_mask: (seq_len, total_len) where total_len = seq_len + past_kv_len
        let (_, seq_len) = input_ids.dims2()?;
        let total_len = causal_mask.dim(1)?;
        let past_kv_len = total_len - seq_len;

        // Flatten input_ids to 1D: (seq_len,)
        let input_ids_1d = input_ids.squeeze(0)?;

        // is_image: (seq_len,) boolean - true where token is an image token
        let is_image = input_ids_1d
            .eq(image_token_index as f64)?
            .to_dtype(inference_tensor::DType::U32)?;

        // Compute image group IDs via contiguous block detection
        // is_prev_image: shift right by 1, pad left with 0
        let is_image_vec: Vec<u32> = is_image.to_vec1()?;
        let mut group_ids = vec![-1i64; seq_len];
        let mut current_group: i64 = -1;
        for i in 0..seq_len {
            if is_image_vec[i] == 1 {
                // Start new group if previous token is not an image token
                if i == 0 || is_image_vec[i - 1] == 0 {
                    current_group += 1;
                }
                group_ids[i] = current_group;
            }
        }

        // Build the bidirectional override mask on CPU as f32
        // For efficiency, we compute this as a Vec and create the tensor once
        let device = causal_mask.device();
        let dtype = causal_mask.dtype();

        // The mask covers (seq_len, total_len). Positions 0..past_kv_len are past KV cache
        // tokens (no image tokens there during image prefill since past_kv_len=0 typically).
        // Positions past_kv_len..total_len correspond to current input_ids.
        let mut override_vals = vec![0f32; seq_len * total_len];
        for qi in 0..seq_len {
            if group_ids[qi] < 0 {
                continue; // Not an image token query
            }
            for ki in 0..seq_len {
                let within_window = sliding_window.is_none_or(|window| qi.abs_diff(ki) < window);
                if group_ids[ki] >= 0 && group_ids[qi] == group_ids[ki] && within_window {
                    // Both are image tokens in the same group: mark for bidirectional override
                    let col = ki + past_kv_len;
                    override_vals[qi * total_len + col] = 1.0;
                }
            }
        }

        let override_mask = Tensor::from_vec(override_vals, (seq_len, total_len), device)?;

        // Where override is 1, set mask to 0.0 (attend); otherwise keep original causal mask.
        // We use where_cond instead of multiplication to avoid NaN from -inf * 0.
        let zero = Tensor::zeros((seq_len, total_len), dtype, device)?;
        let override_bool = override_mask.to_dtype(inference_tensor::DType::U8)?;
        override_bool.where_cond(&zero, causal_mask)
    }
}

impl IsqModel for TextModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        self.lm.residual_tensors()
    }
}

impl crate::speculative::SpeculativeTargetMixin for TextModel {}

impl crate::model::BlockDiffusionMixin for TextModel {}

impl MultimodalModel for TextModel {
    fn forward(
        &self,
        _input_ids: &Tensor,
        _pixel_values: Option<Tensor>,
        _model_specific_args: Box<dyn std::any::Any>,
        _ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        unreachable!()
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn std::any::Any> {
        unreachable!()
    }
    fn cache(&self) -> &EitherCache {
        NormalModel::cache(&self.lm)
    }
    fn device(&self) -> &Device {
        NormalModel::device(&self.lm)
    }
    fn max_seq_len(&self) -> usize {
        NormalModel::max_seq_len(&self.lm)
    }
    fn config(&self) -> &ModelConfigMetadata {
        NormalModel::config(&self.lm)
    }
}

impl AnyMoeBaseModelMixin for TextModel {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        self.lm.get_mlps()
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        self.lm.get_mlps_mut()
    }
    fn amoe_lora_targets(&self) -> &'static [AnyMoeLoraTarget] {
        self.lm.amoe_lora_targets()
    }
    fn amoe_fine_tuned_expert(
        &self,
        layer: usize,
        base: &dyn MlpLayer,
        vb: ShardedVarBuilder,
    ) -> Result<Box<dyn MlpLayer>> {
        self.lm.amoe_fine_tuned_expert(layer, base, vb)
    }
    fn amoe_supported(&self) -> bool {
        self.lm.amoe_supported()
    }
}

#[cfg(test)]
mod tests {
    use inference_tensor::{Device, Tensor};

    use super::{TextModel, select_paged_mm_prefix_path};

    #[test]
    fn paged_mm_prefix_requires_range_metadata() {
        assert!(!select_paged_mm_prefix_path(true, true, true, true, false, false).unwrap());
        assert!(select_paged_mm_prefix_path(true, true, true, true, true, false).is_err());
        assert!(select_paged_mm_prefix_path(true, true, true, true, true, true).unwrap());
        assert!(!select_paged_mm_prefix_path(false, true, true, true, true, false).unwrap());
    }

    #[test]
    fn image_attention_is_global_only_on_full_layers() {
        let causal = Tensor::from_vec(
            vec![
                0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.,
                0.,
                f32::NEG_INFINITY,
                0.,
                0.,
                0.,
            ],
            (3, 3),
            &Device::Cpu,
        )
        .unwrap();
        let input_ids = Tensor::from_vec(vec![9u32, 9, 9], (1, 3), &Device::Cpu).unwrap();
        let full = TextModel::apply_image_bidirectional_mask(&causal, &input_ids, 9, None)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let sliding = TextModel::apply_image_bidirectional_mask(&causal, &input_ids, 9, Some(2))
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        assert_eq!(full[0][2], 0.);
        assert_eq!(sliding[0][1], 0.);
        assert!(sliding[0][2].is_infinite() && sliding[0][2].is_sign_negative());
    }
}
