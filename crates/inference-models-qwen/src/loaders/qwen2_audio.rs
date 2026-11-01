use super::*;

use crate::qwen2_audio::{
    Qwen2AudioConfig, Qwen2AudioModel, audio_token_count,
    inputs_processor::{AUDIO_BOS, AUDIO_EOS, AUDIO_TOKEN, clip_positions},
};

const TEXT_LAYER_INDEX_PATTERN: &str = r"language_model\.model\.layers\.(\d+)\.";
const CONV_KERNEL: usize = 3;
// one 30 s window, the most a clip fills
const MAX_CLIP_SAMPLES: usize = 480_000;

/// `MultimodalLoader` for Qwen2-Audio (`Qwen2AudioForConditionalGeneration`).
pub struct Qwen2AudioLoader;

pub struct Qwen2AudioPrefixer;

impl MultimodalPromptPrefixer for Qwen2AudioPrefixer {
    // the repo's template renders text only; the processor's numbers each clip on its own line before the text
    fn prefix_audio(&self, audio_indexes: Vec<usize>, prompt: &str) -> String {
        let clips = audio_indexes
            .iter()
            .map(|index| format!("Audio {}: {AUDIO_BOS}{AUDIO_TOKEN}{AUDIO_EOS}\n", index + 1))
            .collect::<String>();
        format!("{clips}{prompt}")
    }
}

impl Qwen2AudioLoader {
    fn text_config(cfg: &Qwen2AudioConfig) -> Result<String> {
        Ok(serde_json::to_string(&cfg.text())?)
    }
}

impl MultimodalModelLoader for Qwen2AudioLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg = Qwen2AudioConfig::from_json(config)?;
        Ok(Box::new(Qwen2AudioModel::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(Qwen2AudioConfig::from_json(config)?))
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
        Arc::new(Qwen2AudioPrefixer)
    }
    fn get_device_for_tensor(
        &self,
        config: &str,
        _mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        let cfg = Qwen2AudioConfig::from_json(config)?;
        layer_indexed_device(
            TEXT_LAYER_INDEX_PATTERN,
            cfg.text_config.num_hidden_layers,
            loading_isq,
        )
    }
}

impl IsqModelLoader for Qwen2AudioLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^language_model\.model\.embed_tokens\.weight$",
            r"^language_model\.lm_head\.(weight|bias)$",
        ])
    }

    // anchored to the decoder: the audio tower's layers share the `layers.N.self_attn` names
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^language_model\.lm_head\.(weight|bias)$",
            r"^language_model\.model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$",
            r"^language_model\.model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Qwen2AudioLoader {
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
        let cfg = Qwen2AudioConfig::from_json(config)?;
        let clip_tokens = audio_token_count(clip_positions(MAX_CLIP_SAMPLES));
        let seq_len = max_seq_len.min(&ATTENTION_CHUNK_SIZE) + clip_tokens * max_num_images.max(&1);
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
        let cfg = Qwen2AudioConfig::from_json(config)?;
        let ac = &cfg.audio_config;
        // one window's attention scores over its encoder positions
        let positions = ac.max_source_positions;
        Ok(max_batch_size * ac.encoder_attention_heads * positions * positions)
    }
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = Qwen2AudioConfig::from_json(config)?;
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
        let ac = &cfg.audio_config;
        let (d, ffn) = (ac.d_model, ac.encoder_ffn_dim);
        let convs = (ac.num_mel_bins * d * CONV_KERNEL + d) + (d * d * CONV_KERNEL + d);
        // q, v and out carry biases, k does not; two layer norms
        let layer = 4 * d * d + 3 * d + 2 * d * ffn + ffn + d + 4 * d;
        let audio = convs + ac.max_source_positions * d + ac.encoder_layers * layer + 2 * d;
        let projector = d * tc.hidden_size + tc.hidden_size;
        Ok((text + audio + projector) * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = Qwen2AudioConfig::from_json(config)?;
        Qwen2Loader.layer_sizes_in_bytes(
            &Self::text_config(&cfg)?,
            dtype,
            weight_pack_factor,
            matformer_config,
        )
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Ok(Qwen2AudioConfig::from_json(config)?
            .text_config
            .num_hidden_layers)
    }
    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Audio])
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = Qwen2AudioConfig::from_json(config)?;
        Qwen2Loader.model_config(&Self::text_config(&cfg)?)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use inference_nn::audio::whisper::WhisperEncoder;
    use inference_nn::layers::linear_b;
    use inference_nn::testing::load_synthesized;

    use super::*;

    const TINY: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../inference/tests/fixtures/qwen2_audio/tiny/config.json"
    );

    // the device map sizes the audio tower, the projector and the embeddings by non_mapped_size_in_bytes
    #[test]
    fn non_mapped_size_counts_the_loaded_audio_tower_and_projector() -> anyhow::Result<()> {
        let config = std::fs::read_to_string(TINY)?;
        let cfg = Qwen2AudioConfig::from_json(&config)?;
        let (_, names) = load_synthesized(&[], HashMap::new(), DType::F32, |vb| {
            WhisperEncoder::new(&cfg.encoder(), vb.pp("audio_tower"))?;
            linear_b(
                cfg.audio_config.d_model,
                cfg.text_config.hidden_size,
                true,
                vb.pp("multi_modal_projector").pp("linear"),
            )?;
            Ok(())
        })?;
        let tc = &cfg.text_config;
        let text = 2 * tc.vocab_size * tc.hidden_size + tc.hidden_size;
        let towers = names.values().sum::<usize>();
        let bytes =
            Qwen2AudioLoader.non_mapped_size_in_bytes(&config, DType::F32, 1, None, None)?;
        assert_eq!(bytes, (text + towers) * DType::F32.size_in_bytes());
        Ok(())
    }
}
