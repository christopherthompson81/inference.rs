use super::*;

use crate::granite_speech::{GraniteSpeechModel, config::Config as GraniteSpeechConfig};

const TEXT_LAYER_INDEX_PATTERN: &str = r"language_model\.model\.layers\.(\d+)\.";
// a 30 s clip budgets the encoder's activations
const MAX_CLIP_SECONDS: usize = 30;

/// `MultimodalLoader` for IBM Granite Speech (and Granite Speech Plus).
pub struct GraniteSpeechLoader;

pub struct GraniteSpeechPrefixer;

impl MultimodalPromptPrefixer for GraniteSpeechPrefixer {
    // its chat template renders text only; the model card puts each clip's placeholder ahead of the question
    fn prefix_audio(&self, audio_indexes: Vec<usize>, prompt: &str) -> String {
        let clips =
            crate::granite_speech::inputs_processor::AUDIO_TOKEN.repeat(audio_indexes.len());
        format!("{clips}{prompt}")
    }
}

impl GraniteSpeechLoader {
    fn text_config(cfg: &GraniteSpeechConfig) -> Result<String> {
        Ok(serde_json::to_string(&cfg.text_config)?)
    }

    fn max_clip_tokens(cfg: &GraniteSpeechConfig) -> usize {
        crate::granite_speech::features::audio_token_count(
            MAX_CLIP_SECONDS * crate::granite_speech::features::SAMPLE_RATE as usize,
            cfg.window_size,
            cfg.downsample_rate,
        )
    }
}

impl MultimodalModelLoader for GraniteSpeechLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg = GraniteSpeechConfig::from_json(config)?;
        Ok(Box::new(GraniteSpeechModel::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(GraniteSpeechConfig::from_json(config)?))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        true
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Audio],
            output: vec![SupportedModality::Text],
        })
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(GraniteSpeechPrefixer)
    }
    fn get_device_for_tensor(
        &self,
        config: &str,
        _mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        let cfg = GraniteSpeechConfig::from_json(config)?;
        layer_indexed_device(
            TEXT_LAYER_INDEX_PATTERN,
            cfg.text_config.num_hidden_layers,
            loading_isq,
        )
    }
}

impl IsqModelLoader for GraniteSpeechLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^language_model\.model\.embed_tokens\.weight$",
            r"^language_model\.lm_head\.(weight|bias)$",
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"language_model\.lm_head\.(weight|bias)$",
            r"language_model\.model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$",
            r"language_model\.model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
            r"language_model\.model\.layers\.(\d+)\.shared_mlp\.(input_linear|output_linear)\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for GraniteSpeechLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Multimodal {
            max_seq_len,
            max_batch_size,
            max_num_images,
            ..
        } = params
        else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };
        let cfg = GraniteSpeechConfig::from_json(config)?;
        let seq_len = max_seq_len.min(&ATTENTION_CHUNK_SIZE)
            + Self::max_clip_tokens(&cfg) * max_num_images.max(&1);
        Ok(max_batch_size * cfg.text_config.num_attention_heads * seq_len * seq_len)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Multimodal { max_batch_size, .. } = params else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };
        let cfg = GraniteSpeechConfig::from_json(config)?;
        let ec = &cfg.encoder_config;
        let frames = MAX_CLIP_SECONDS * crate::granite_speech::features::SAMPLE_RATE as usize
            / crate::granite_speech::features::HOP_LENGTH
            / crate::granite_speech::features::STACKED_FRAMES;
        let blocks = frames.div_ceil(ec.context_size);
        let attention = blocks * ec.num_heads * ec.context_size * ec.context_size;
        let hidden = frames * ec.hidden_dim * ec.feedforward_mult.max(2 * ec.conv_expansion_factor);
        Ok(max_batch_size * attention.max(hidden))
    }
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = GraniteSpeechConfig::from_json(config)?;
        let tc = &cfg.text_config;
        let (embed_pack, head_pack) = super::language_model_pack_factors(
            quantization,
            "language_model.model.embed_tokens.weight",
            "language_model.lm_head.weight",
            tc.tie_word_embeddings,
            dtype,
            weight_pack_factor,
        )?;
        let text = tc.hidden_size * tc.vocab_size / embed_pack
            + if tc.tie_word_embeddings {
                0
            } else {
                tc.hidden_size * tc.vocab_size / head_pack
            }
            + tc.hidden_size;
        let ec = &cfg.encoder_config;
        let encoder = {
            let (h, inner) = (ec.hidden_dim, ec.dim_head * ec.num_heads);
            let ff = 2 * (2 * h + h * h * ec.feedforward_mult + h * ec.feedforward_mult + h);
            let attn =
                2 * h + h * inner * 3 + inner * h + h + (2 * ec.max_pos_emb + 1) * ec.dim_head;
            let conv_inner = h * ec.conv_expansion_factor;
            let conv = 2 * h
                + h * 2 * conv_inner
                + 2 * conv_inner
                + conv_inner * ec.conv_kernel_size
                + 4 * conv_inner
                + conv_inner * h
                + h;
            ec.input_dim * h
                + h
                + ec.num_layers * (ff + attn + conv + 2 * h)
                + 2 * h * ec.output_dim
                + h
                + ec.output_dim
        };
        let p = &cfg.projector_config;
        let projector = {
            let attention = |context: usize| {
                2 * (p.hidden_size * p.hidden_size + p.hidden_size)
                    + 2 * (context * p.hidden_size + p.hidden_size)
                    + 2 * p.hidden_size
            };
            let layer = attention(p.hidden_size)
                + attention(p.encoder_hidden_size)
                + 2 * p.hidden_size * p.intermediate_size
                + p.intermediate_size
                + 3 * p.hidden_size;
            cfg.window_size / cfg.downsample_rate * p.hidden_size
                + 2 * p.hidden_size
                + p.num_hidden_layers * layer
                + p.hidden_size * tc.hidden_size
                + tc.hidden_size
        };
        Ok((text + encoder + projector) * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = GraniteSpeechConfig::from_json(config)?;
        GraniteMoeHybridLoader.layer_sizes_in_bytes(
            &Self::text_config(&cfg)?,
            dtype,
            weight_pack_factor,
            matformer_config,
        )
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Ok(GraniteSpeechConfig::from_json(config)?
            .text_config
            .num_hidden_layers)
    }
    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Audio])
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = GraniteSpeechConfig::from_json(config)?;
        GraniteMoeHybridLoader.model_config(&Self::text_config(&cfg)?)
    }
}
