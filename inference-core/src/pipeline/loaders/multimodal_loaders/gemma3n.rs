use super::*;

/// [`MultimodalLoader`] for an Gemma 3n model.
///
/// [`MultimodalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.MultimodalLoader.html
pub struct Gemma3nLoader;

impl MultimodalModelLoader for Gemma3nLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg: Gemma3nConfig = serde_json::from_str(config)?;
        Ok(Box::new(Gemma3nModel::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self, _config: &str) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let config: Gemma3nConfig = serde_json::from_str(config)?;
        Ok(Box::new(config))
    }
    fn get_processor(
        &self,
        _config: &str,
        processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        // Handle the Gemma 3 1b case here
        Arc::new(Gemma3nProcessor::new(
            processor_config.unwrap_or_default(),
            true,
        ))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn supports_encoder_cache(&self, _config: &str) -> bool {
        true
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        true
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(Gemma3Prefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![
                SupportedModality::Text,
                SupportedModality::Vision,
                SupportedModality::Audio,
            ],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for Gemma3nLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.language_model\.embed_tokens\.weight$")?,
            Regex::new(r"^model\.language_model\.embed_tokens_per_layer\.weight$")?,
            Regex::new(r"^model\.language_model\.lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Language model attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // Language model MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
            // Audio conformer attention layers
            Regex::new(r"conformer\.(\d+)\.attention\.attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"conformer\.(\d+)\.attention\.attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"conformer\.(\d+)\.attention\.attn\.v_proj\.(weight|bias)$")?,
            Regex::new(
                r"conformer\.(\d+)\.attention\.attn\.relative_position_embedding\.pos_proj\.(weight|bias)$",
            )?,
            Regex::new(r"conformer\.(\d+)\.attention\.post\.(weight|bias)$")?,
            // Audio conformer FFW layers
            Regex::new(r"conformer\.(\d+)\.ffw_layer_start\.ffw_layer_1\.(weight|bias)$")?,
            Regex::new(r"conformer\.(\d+)\.ffw_layer_start\.ffw_layer_2\.(weight|bias)$")?,
            Regex::new(r"conformer\.(\d+)\.ffw_layer_end\.ffw_layer_1\.(weight|bias)$")?,
            Regex::new(r"conformer\.(\d+)\.ffw_layer_end\.ffw_layer_2\.(weight|bias)$")?,
            // Audio conformer conv1d layers
            Regex::new(r"conformer\.(\d+)\.lconv1d\.linear_start\.(weight|bias)$")?,
            Regex::new(r"conformer\.(\d+)\.lconv1d\.linear_end\.(weight|bias)$")?,
            // Audio subsample projection
            Regex::new(r"subsample_conv_projection\.input_proj_linear\.(weight|bias)$")?,
            // Multimodal embedders
            Regex::new(r"embed_vision\.embedding_projection\.(weight|bias)$")?,
            Regex::new(r"embed_audio\.embedding_projection\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"model\.language_model\.embed_tokens_per_layer\.weight$")?,
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Language model attention
            Regex::new(r"model\.language_model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // Language model MLP
            Regex::new(r"model\.language_model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
            // Projections
            Regex::new(r"model\.language_model\.per_layer_model_projection\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.altup_projections\.(\d+)\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.altup_unembed_projections\.(\d+)\.(weight|bias)$")?,
            // Audio conformer attention layers
            Regex::new(
                r"model\.audio_tower\.conformer\.(\d+)\.attention\.attn\.q_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"model\.audio_tower\.conformer\.(\d+)\.attention\.attn\.k_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"model\.audio_tower\.conformer\.(\d+)\.attention\.attn\.v_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"model\.audio_tower\.conformer\.(\d+)\.attention\.attn\.relative_position_embedding\.pos_proj\.(weight|bias)$",
            )?,
            Regex::new(r"model\.audio_tower\.conformer\.(\d+)\.attention\.post\.(weight|bias)$")?,
            // Audio conformer FFW layers
            Regex::new(
                r"model\.audio_tower\.conformer\.(\d+)\.ffw_layer_start\.ffw_layer_1\.(weight|bias)$",
            )?,
            Regex::new(
                r"model\.audio_tower\.conformer\.(\d+)\.ffw_layer_start\.ffw_layer_2\.(weight|bias)$",
            )?,
            Regex::new(
                r"model\.audio_tower\.conformer\.(\d+)\.ffw_layer_end\.ffw_layer_1\.(weight|bias)$",
            )?,
            Regex::new(
                r"model\.audio_tower\.conformer\.(\d+)\.ffw_layer_end\.ffw_layer_2\.(weight|bias)$",
            )?,
            // Audio conformer conv1d layers
            Regex::new(
                r"model\.audio_tower\.conformer\.(\d+)\.lconv1d\.linear_start\.(weight|bias)$",
            )?,
            Regex::new(
                r"model\.audio_tower\.conformer\.(\d+)\.lconv1d\.linear_end\.(weight|bias)$",
            )?,
            // Audio subsample projection
            Regex::new(
                r"model\.audio_tower\.subsample_conv_projection\.input_proj_linear\.(weight|bias)$",
            )?,
            // Multimodal embedders
            Regex::new(r"model\.embed_vision\.embedding_projection\.(weight|bias)$")?,
            Regex::new(r"model\.embed_audio\.embedding_projection\.(weight|bias)$")?,
        ])
    }
}

impl DeviceMappedModelLoader for Gemma3nLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Multimodal {
            max_seq_len,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };

        let cfg: Gemma3nConfig = serde_json::from_str(config)?;
        let text_cfg = &cfg.text_config;

        // Gemma3n is an "inject into the prompt" model, similar to Gemma3
        // We need to account for vision and audio tokens in the sequence length

        let mut total_seq_len = *max_seq_len.min(&ATTENTION_CHUNK_SIZE);

        // Add vision tokens
        {
            // Vision tokens are injected into the prompt
            // MSFA outputs fixed 16x16 features regardless of input size
            let msfa_spatial_size = 16; // Fixed from vision.rs line 1115
            let vision_tokens_per_image = msfa_spatial_size * msfa_spatial_size; // 256 tokens
            total_seq_len += vision_tokens_per_image * max_num_images;
        }

        // Add audio tokens
        {
            // Audio tokens are injected into the prompt
            // From config field audio_soft_tokens_per_image (typically 188)
            let audio_tokens = cfg.audio_soft_tokens_per_image;
            total_seq_len += audio_tokens;
        }

        // Calculate max attention size for text model with all injected tokens
        let max_text_attn =
            max_batch_size * text_cfg.num_attention_heads * total_seq_len * total_seq_len;

        Ok(max_text_attn)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Multimodal {
            max_seq_len: _,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };

        let cfg: Gemma3nConfig = serde_json::from_str(config)?;

        // Calculate max activation sizes for each modality
        let mut max_activation = 0;

        // Vision activation size
        {
            // Vision is Gemma3n's MobileNetV5 architecture with Multi-Query Attention
            // The peak activation is in the Multi-Query Attention layers

            // From the architecture: stages 3 and 4 have MMQA blocks
            // Input images are 768x768 (from inputs_processor.rs)
            // Stage 3: 640 channels at 48x48 (768/16 downsampling), MMQA with num_heads=12, kv_dim=64
            // Stage 4: 1280 channels at 24x24 (768/32 downsampling), MMQA with num_heads=16, kv_dim=96
            // MSFA output: 2048 channels at fixed 16x16

            let vision_tower_act = {
                // Peak is during MMQA attention computation in stage 4
                // Stage 4 has higher memory usage than Stage 3 due to more heads (16 vs 12)
                // From vision.rs: Stage 4 has num_heads=16, kv_dim=96, kv_stride=1
                let num_heads = 16; // Stage 4 configuration
                let spatial_size = 24; // 768 / 32 = 24 (input 768x768, stage 4 has 32x downsampling)
                let seq_len = spatial_size * spatial_size;

                // Attention scores: [B * num_images, num_heads, seq_len, seq_len]
                max_batch_size * max_num_images * num_heads * seq_len * seq_len
            };

            // Vision embedder activations
            let vision_embed_act = {
                // MSFA output: 2048 channels at fixed 16x16 spatial (from vision.rs line 1115)
                let msfa_channels = 2048; // MSFA_OUT_CHANNELS from vision.rs
                let spatial_size = 16; // Fixed output resolution from MSFA
                let vision_features =
                    max_batch_size * max_num_images * msfa_channels * spatial_size * spatial_size;

                // After embedding projection to text hidden size
                let projected = max_batch_size
                    * max_num_images
                    * spatial_size
                    * spatial_size
                    * cfg.text_config.hidden_size;

                vision_features.max(projected)
            };

            max_activation = max_activation.max(vision_tower_act).max(vision_embed_act);
        }

        // Audio activation size
        {
            let audio_cfg = &cfg.audio_config;

            // Calculate max audio sequence length based on config
            // Audio uses conformer with subsampling and reduction

            // A rough estimate of max_audio_frames
            let max_audio_frames = 1280;

            let subsample_factor: usize = audio_cfg
                .sscp_conv_stride_size
                .iter()
                .map(|stride| stride[0]) // Time dimension stride
                .product();
            let audio_seq_after_subsample = max_audio_frames / subsample_factor;

            // Audio encoder activations
            let audio_encoder_act = {
                // Conformer FFW layers have expansion factor from config
                let intermediate_size = audio_cfg.hidden_size * 4; // FFW expansion factor

                // Peak is in the FFW layers before reduction
                max_batch_size * audio_seq_after_subsample * intermediate_size
            };

            // Audio attention activations
            let audio_attn_act = {
                // Attention uses chunked processing with specific context sizes
                let chunk_size = audio_cfg.conf_attention_chunk_size;
                let context_size = chunk_size + audio_cfg.conf_attention_context_left - 1
                    + audio_cfg.conf_attention_context_right;

                // Peak is attention scores: [B, num_heads, num_chunks, chunk_size, context_size]
                let num_chunks = audio_seq_after_subsample.div_ceil(chunk_size);

                max_batch_size
                    * audio_cfg.conf_num_attention_heads
                    * num_chunks
                    * chunk_size
                    * context_size
            };

            max_activation = max_activation.max(audio_encoder_act).max(audio_attn_act);
        }

        Ok(max_activation)
    }
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: Gemma3nConfig = serde_json::from_str(config)?;

        // Apply matformer slicing if configured
        let text_cfg = if let Some(matformer_cfg) = matformer_config {
            use crate::device_map::DummyDeviceMapper;
            use crate::vision_models::gemma3n::text::handle_matformer_slicing;

            let dummy_mapper = DummyDeviceMapper {
                nm_device: Device::Cpu,
            };
            let (adjusted_cfg, _, _, _, _) = handle_matformer_slicing(
                &cfg.text_config,
                &Some(matformer_cfg.clone()),
                &dummy_mapper,
            )?;
            adjusted_cfg
        } else {
            cfg.text_config.clone()
        };

        let text_cfg = &text_cfg;

        // Text components that are not device-mapped
        let text_elems = {
            // Embeddings
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    quantization,
                    "model.language_model.embed_tokens.weight",
                    "model.language_model.lm_head.weight",
                    text_cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens =
                text_cfg.hidden_size * text_cfg.vocab_size / embed_tokens_pack_factor;
            let ple_embedding_pack_factor = if matformer_config.is_some() {
                1
            } else {
                super::promoted_tensor_pack_factor(
                    quantization,
                    "model.language_model.embed_tokens_per_layer.weight",
                    dtype,
                    weight_pack_factor,
                )?
            };
            let embed_tokens_per_layer = text_cfg.num_hidden_layers
                * text_cfg.hidden_size_per_layer_input
                * text_cfg.vocab_size_per_layer_input
                / ple_embedding_pack_factor;

            // LM head (if not tied)
            let lm_head = if !text_cfg.tie_word_embeddings {
                text_cfg.hidden_size * text_cfg.vocab_size / lm_head_pack_factor
            } else {
                0
            };

            // Final layer norm
            let norm = text_cfg.hidden_size;

            // AltUp projections (not device-mapped)
            let altup_projections =
                (text_cfg.altup_num_inputs - 1) * text_cfg.hidden_size * text_cfg.hidden_size
                    / weight_pack_factor;
            let altup_unembed_projections =
                (text_cfg.altup_num_inputs - 1) * text_cfg.hidden_size * text_cfg.hidden_size
                    / weight_pack_factor;

            // Per-layer model projection
            let per_layer_model_projection = text_cfg.num_hidden_layers
                * text_cfg.hidden_size
                * text_cfg.hidden_size_per_layer_input
                / weight_pack_factor;
            let per_layer_projection_norm = text_cfg.hidden_size;

            embed_tokens
                + embed_tokens_per_layer
                + lm_head
                + norm
                + altup_projections
                + altup_unembed_projections
                + per_layer_model_projection
                + per_layer_projection_norm
        };

        // Vision components
        let vision_elems = {
            let multimodal_cfg = &cfg.vision_config;
            // Vision tower - calculated from actual Gemma3n architecture
            // NOTE: Vision tower uses only Conv2d layers, NOT Arc<dyn QuantMethod>,
            // so NONE of these should be divided by weight_pack_factor
            let vision_tower_elems = {
                use crate::vision_models::gemma3n::vision::{
                    gemma3n_mobilenet_def, make_divisible, BlockType, INPUT_CHANNELS,
                    MSFA_EXPANSION_RATIO, MSFA_IN_CHANNELS, MSFA_OUT_CHANNELS, STEM_KERNEL_SIZE,
                    STEM_OUT_CHANNELS,
                };

                // Stem: ConvNormAct (Conv2d + RMSNorm)
                let stem_conv =
                    INPUT_CHANNELS * STEM_OUT_CHANNELS * STEM_KERNEL_SIZE * STEM_KERNEL_SIZE;
                let stem_norm = STEM_OUT_CHANNELS; // RMSNorm weight

                // Track input channels through the network
                let mut in_chs = STEM_OUT_CHANNELS;
                let mut total_elems = stem_conv + stem_norm;

                // Process all stages from gemma3n_mobilenet_def
                let block_defs = gemma3n_mobilenet_def();

                for stage_blocks in block_defs.iter() {
                    for block_type in stage_blocks.iter() {
                        match block_type {
                            BlockType::EdgeResidual {
                                out_channels,
                                kernel_size,
                                stride: _,
                                expand_ratio,
                                ..
                            } => {
                                #[allow(clippy::cast_precision_loss)]
                                let mid_chs = make_divisible(in_chs as f64 * expand_ratio, 8);
                                // EdgeResidual: all Conv2d layers, not quantizable
                                total_elems += in_chs * mid_chs * kernel_size * kernel_size; // conv_exp (Conv2d)
                                total_elems += mid_chs; // bn1 weight
                                total_elems += mid_chs * out_channels; // conv_pwl (Conv2d)
                                total_elems += out_channels; // bn2 weight
                                in_chs = *out_channels;
                            }
                            BlockType::UniversalInvertedResidual {
                                out_channels,
                                start_kernel_size,
                                mid_kernel_size,
                                stride: _,
                                expand_ratio,
                                ..
                            } => {
                                #[allow(clippy::cast_precision_loss)]
                                let mid_chs = make_divisible(in_chs as f64 * expand_ratio, 8);
                                // UniversalInvertedResidual: all Conv2d layers, not quantizable
                                if *expand_ratio != 1.0 {
                                    total_elems += in_chs * mid_chs; // expand conv (Conv2d)
                                    total_elems += mid_chs; // expand norm
                                }
                                if *start_kernel_size > 0 {
                                    total_elems += mid_chs * start_kernel_size * start_kernel_size; // depthwise start (Conv2d)
                                    total_elems += mid_chs; // norm
                                }
                                if *mid_kernel_size > 0 {
                                    total_elems += mid_chs * mid_kernel_size * mid_kernel_size; // depthwise mid (Conv2d)
                                    total_elems += mid_chs; // norm
                                }
                                total_elems += mid_chs * out_channels; // project conv (Conv2d)
                                total_elems += out_channels; // project norm
                                total_elems += out_channels; // layer scale gamma
                                in_chs = *out_channels;
                            }
                            BlockType::MultiQueryAttention {
                                num_heads, kv_dim, ..
                            } => {
                                // MMQA: all Conv2d layers, not quantizable
                                let dw_kernel_size = 3; // Default dw_kernel_size for MMQA
                                total_elems += in_chs; // norm weight
                                total_elems += in_chs * num_heads * kv_dim; // query_proj (Conv2d)
                                total_elems += in_chs * kv_dim; // key_proj (Conv2d)
                                total_elems += in_chs * dw_kernel_size * dw_kernel_size; // key_dw_conv (Conv2d)
                                total_elems += *kv_dim; // value_down_conv (Conv2d)
                                total_elems += 1; // value_norm weight
                                total_elems += *kv_dim; // value_proj (Conv2d)
                                total_elems += num_heads * kv_dim * in_chs; // output_proj (Conv2d)
                                total_elems += in_chs; // layer scale
                            }
                        }
                    }
                }

                // Multi-scale fusion adapter (msfa) - also uses Conv2d layers
                let msfa_in = MSFA_IN_CHANNELS.iter().sum::<usize>();
                let msfa_out = MSFA_OUT_CHANNELS;
                #[allow(clippy::cast_precision_loss)]
                let msfa_mid = make_divisible(msfa_in as f64 * MSFA_EXPANSION_RATIO, 8);

                // MSFA FFN (UIR with expansion_ratio) - Conv2d layers, not quantizable
                total_elems += msfa_in * msfa_mid; // expand (Conv2d)
                total_elems += msfa_mid; // expand norm
                total_elems += msfa_mid * msfa_out; // project (Conv2d)
                total_elems += msfa_out; // project norm
                total_elems += msfa_out; // final norm

                total_elems
            };

            // Vision multimodal embedder components
            let embed_vision_elems = {
                // Embedding layer (not quantizable)
                let embedding = multimodal_cfg.vocab_size * multimodal_cfg.hidden_size;

                // Normalization layers (not quantizable)
                let hard_norm = multimodal_cfg.hidden_size;
                let soft_norm = multimodal_cfg.hidden_size;

                // Projection from vision to text hidden size (IS Arc<dyn QuantMethod>, so quantizable)
                let projection =
                    multimodal_cfg.hidden_size * text_cfg.hidden_size / weight_pack_factor;

                // Post-projection norm (not quantizable)
                let post_norm = text_cfg.hidden_size;

                embedding + hard_norm + soft_norm + projection + post_norm
            };

            vision_tower_elems + embed_vision_elems
        };

        // Audio components - based on actual audio.rs structure
        let audio_elems = {
            let audio_cfg = &cfg.audio_config;

            // SubSampleConvProjection components
            let subsample_conv_projection_elems = {
                // Conv blocks (Conv2d layers - NOT quantizable)
                let mut conv_elems = 0;

                // conv_0: Conv2d from 1 channel to first channel size
                let in_ch_0 = 1;
                let out_ch_0 = audio_cfg.sscp_conv_channel_size[0];
                let kernel_0 = &audio_cfg.sscp_conv_kernel_size[0];
                conv_elems += in_ch_0 * out_ch_0 * kernel_0[0] * kernel_0[1];

                // conv_1: Conv2d from first to second channel size
                let in_ch_1 = out_ch_0;
                let out_ch_1 = audio_cfg.sscp_conv_channel_size[1];
                let kernel_1 = &audio_cfg.sscp_conv_kernel_size[1];
                conv_elems += in_ch_1 * out_ch_1 * kernel_1[0] * kernel_1[1];

                // CumulativeGroupNorm for each conv block (weight only, no bias by default)
                let norm_0 = out_ch_0; // norm weight for conv_0
                let norm_1 = out_ch_1; // norm weight for conv_1

                // input_proj_linear (Arc<dyn QuantMethod> - IS quantizable)
                let mut f_out = audio_cfg.input_feat_size;
                for i in 0..2 {
                    let kernel_w = audio_cfg.sscp_conv_kernel_size[i][1];
                    let stride_w = audio_cfg.sscp_conv_stride_size[i][1];
                    let pad_left = 1;
                    let pad_right = 1;
                    f_out = (f_out + pad_left + pad_right + stride_w - kernel_w) / stride_w;
                }
                let input_proj_in_features = out_ch_1 * f_out;
                let input_proj_linear =
                    input_proj_in_features * audio_cfg.hidden_size / weight_pack_factor;

                conv_elems + norm_0 + norm_1 + input_proj_linear
            };

            // Conformer blocks
            let conformer_elems = {
                let mut total = 0;

                for _ in 0..audio_cfg.conf_num_hidden_layers {
                    // ConformerAttention
                    let attention_elems = {
                        // Norms (NOT quantizable)
                        let pre_attn_norm = audio_cfg.hidden_size;
                        let post_norm = audio_cfg.hidden_size;

                        // Attention projections (Arc<dyn QuantMethod> - IS quantizable)
                        let q_proj =
                            audio_cfg.hidden_size * audio_cfg.hidden_size / weight_pack_factor;
                        let k_proj =
                            audio_cfg.hidden_size * audio_cfg.hidden_size / weight_pack_factor;
                        let v_proj =
                            audio_cfg.hidden_size * audio_cfg.hidden_size / weight_pack_factor;
                        let post =
                            audio_cfg.hidden_size * audio_cfg.hidden_size / weight_pack_factor;

                        // RelativePositionEmbedding
                        let pos_proj =
                            audio_cfg.hidden_size * audio_cfg.hidden_size / weight_pack_factor;
                        let per_dim_scale =
                            audio_cfg.hidden_size / audio_cfg.conf_num_attention_heads; // head_dim
                        let inv_timescales = audio_cfg.hidden_size / 2; // num_timescales
                        let pos_indices = audio_cfg.conf_attention_context_left
                            + audio_cfg.conf_attention_context_right
                            + 1;

                        // Local causal masks (precomputed tensors)
                        let chunk_size = audio_cfg.conf_attention_chunk_size;
                        let context_size = chunk_size + audio_cfg.conf_attention_context_left - 1
                            + audio_cfg.conf_attention_context_right;
                        let local_causal_valid_mask = chunk_size * context_size; // U8 tensor
                        let invalid_logits_tensor = 1; // single f32 value

                        pre_attn_norm
                            + post_norm
                            + q_proj
                            + k_proj
                            + v_proj
                            + post
                            + pos_proj
                            + per_dim_scale
                            + inv_timescales
                            + pos_indices
                            + local_causal_valid_mask
                            + invalid_logits_tensor
                    };

                    // ConformerFeedForward (start and end)
                    let ffw_elems = {
                        // Each FFW has:
                        // - pre_layer_norm (NOT quantizable)
                        // - ffw_layer_1 (Arc<dyn QuantMethod> - IS quantizable)
                        // - ffw_layer_2 (Arc<dyn QuantMethod> - IS quantizable)
                        // - post_layer_norm (NOT quantizable)
                        let intermediate_size = audio_cfg.hidden_size * 4;

                        let ffw_start = {
                            let pre_norm = audio_cfg.hidden_size;
                            let layer_1 =
                                audio_cfg.hidden_size * intermediate_size / weight_pack_factor;
                            let layer_2 =
                                intermediate_size * audio_cfg.hidden_size / weight_pack_factor;
                            let post_norm = audio_cfg.hidden_size;
                            pre_norm + layer_1 + layer_2 + post_norm
                        };

                        let ffw_end = ffw_start; // Same structure

                        ffw_start + ffw_end
                    };

                    // ConformerLightConv1d
                    let lconv1d_elems = {
                        // Norms (NOT quantizable)
                        let pre_layer_norm = audio_cfg.hidden_size;
                        let conv_norm = audio_cfg.hidden_size;

                        // Linear layers (Arc<dyn QuantMethod> - IS quantizable)
                        let linear_start = audio_cfg.hidden_size * (audio_cfg.hidden_size * 2)
                            / weight_pack_factor;
                        let linear_end =
                            audio_cfg.hidden_size * audio_cfg.hidden_size / weight_pack_factor;

                        // depthwise_conv1d (Conv1d - NOT quantizable)
                        let depthwise = audio_cfg.hidden_size * audio_cfg.conf_conv_kernel_size;

                        pre_layer_norm + conv_norm + linear_start + linear_end + depthwise
                    };

                    // Final norm for conformer block (NOT quantizable)
                    let block_norm = audio_cfg.hidden_size;

                    total += attention_elems + ffw_elems + lconv1d_elems + block_norm;
                }

                total
            };

            // Audio multimodal embedder (embed_audio)
            let embed_audio_elems = {
                // Embedding layer (ScaledEmbedding - NOT quantizable)
                let embedding = audio_cfg.vocab_size * audio_cfg.hidden_size;

                // RMS norms (NOT quantizable)
                let hard_embedding_norm = audio_cfg.hidden_size; // with scale
                let soft_embedding_norm = audio_cfg.hidden_size; // with scale
                let embedding_post_projection_norm = text_cfg.hidden_size; // without scale

                // Projection (Arc<dyn QuantMethod> - IS quantizable)
                let embedding_projection =
                    audio_cfg.hidden_size * text_cfg.hidden_size / weight_pack_factor;

                embedding
                    + hard_embedding_norm
                    + soft_embedding_norm
                    + embedding_post_projection_norm
                    + embedding_projection
            };

            subsample_conv_projection_elems + conformer_elems + embed_audio_elems
        };

        let vision_dtype = if dtype == DType::F16 {
            // f16 -> f32 for vision model in particular.
            DType::F32
        } else {
            dtype
        };

        let total_elems = text_elems * dtype.size_in_bytes()
            + vision_elems * vision_dtype.size_in_bytes()
            + audio_elems * dtype.size_in_bytes();

        Ok(total_elems)
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: Gemma3nConfig = serde_json::from_str(config)?;

        // Apply matformer slicing if configured
        let (text_cfg, _layer_rename_map, _layers_skipped) = if let Some(matformer_cfg) =
            matformer_config
        {
            use crate::device_map::DummyDeviceMapper;
            use crate::vision_models::gemma3n::text::handle_matformer_slicing;

            let dummy_mapper = DummyDeviceMapper {
                nm_device: Device::Cpu,
            };
            let (adjusted_cfg, _, _, layer_rename_map, layers_skipped) = handle_matformer_slicing(
                &cfg.text_config,
                &Some(matformer_cfg.clone()),
                &dummy_mapper,
            )?;
            (adjusted_cfg, layer_rename_map, layers_skipped)
        } else {
            (cfg.text_config.clone(), None, None)
        };

        let text_cfg = &text_cfg;

        // When matformer slicing is applied, we only include the layers that are kept
        let mut layer_sizes = Vec::new();

        // Note: We don't need orig_intermediate_sizes anymore since the adjusted config
        // already has the correct intermediate sizes after matformer slicing

        for layer_idx in 0..text_cfg.num_hidden_layers {
            let per_layer_elems = {
                // Layer norms
                let input_layernorm = text_cfg.hidden_size;
                let post_attention_layernorm = text_cfg.hidden_size;
                let pre_feedforward_layernorm = text_cfg.hidden_size;
                let post_feedforward_layernorm = text_cfg.hidden_size;
                let post_per_layer_input_norm = text_cfg.hidden_size;

                // Attention components
                let size_in = text_cfg.hidden_size;
                let size_q = text_cfg.num_attention_heads * text_cfg.head_dim;
                let size_kv = text_cfg.num_key_value_heads * text_cfg.head_dim;

                let q_proj = size_in * size_q / weight_pack_factor;
                let k_proj = size_in * size_kv / weight_pack_factor;
                let v_proj = size_in * size_kv / weight_pack_factor;
                let o_proj = size_q * size_in / weight_pack_factor;

                // Q, K, V norms
                let q_norm = text_cfg.head_dim;
                let k_norm = text_cfg.head_dim;
                let v_norm = text_cfg.head_dim; // No bias for v_norm

                // MLP components - use the adjusted intermediate sizes from matformer
                let intermediate_size = match &text_cfg.intermediate_size {
                    IntermediateSize::Single(size) => *size,
                    IntermediateSize::PerLayer(sizes) => sizes[layer_idx],
                    IntermediateSize::Matformer(sizes, _) => sizes[layer_idx],
                };
                let gate_proj = text_cfg.hidden_size * intermediate_size / weight_pack_factor;
                let up_proj = text_cfg.hidden_size * intermediate_size / weight_pack_factor;
                let down_proj = intermediate_size * text_cfg.hidden_size / weight_pack_factor;

                // AltUp components (per layer)
                let altup_elems = {
                    let correct_output_scale = text_cfg.hidden_size;
                    let correction_coefs = text_cfg.altup_num_inputs * text_cfg.altup_num_inputs;
                    let prediction_coefs =
                        text_cfg.altup_num_inputs * text_cfg.altup_num_inputs.pow(2);
                    let modality_router = text_cfg.hidden_size * text_cfg.altup_num_inputs;
                    let router_norm = text_cfg.hidden_size;

                    correct_output_scale
                        + correction_coefs
                        + prediction_coefs
                        + modality_router
                        + router_norm
                };

                // Laurel block components
                let laurel_elems = {
                    let left = text_cfg.hidden_size * text_cfg.laurel_rank;
                    let right = text_cfg.laurel_rank * text_cfg.hidden_size;
                    let post_norm = text_cfg.hidden_size;

                    left + right + post_norm
                };

                // Per-layer input components
                let per_layer_input_gate =
                    text_cfg.hidden_size * text_cfg.hidden_size_per_layer_input;
                let per_layer_projection =
                    text_cfg.hidden_size_per_layer_input * text_cfg.hidden_size;

                input_layernorm
                    + post_attention_layernorm
                    + pre_feedforward_layernorm
                    + post_feedforward_layernorm
                    + post_per_layer_input_norm
                    + q_proj
                    + k_proj
                    + v_proj
                    + o_proj
                    + q_norm
                    + k_norm
                    + v_norm
                    + gate_proj
                    + up_proj
                    + down_proj
                    + altup_elems
                    + laurel_elems
                    + per_layer_input_gate
                    + per_layer_projection
            };

            layer_sizes.push(per_layer_elems * dtype.size_in_bytes());
        }

        Ok(layer_sizes)
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: Gemma3nConfig = serde_json::from_str(config)?;
        Ok(cfg.text_config.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: Gemma3nConfig = serde_json::from_str(config)?;
        let cfg = cfg.text_config;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None, // None to be more forgiving, some do not
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision, NonMappedSubModel::Audio])
    }
}

#[allow(dead_code)]
pub struct Gemma3nPrefixer;

impl MultimodalPromptPrefixer for Gemma3nPrefixer {
    fn prefix_image(&self, _image_indexes: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
}
