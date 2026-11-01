use super::*;

use inference_nn::device_map::DeviceMapper;
use inference_nn::utils::varbuilder_utils::DeviceForLoadTensor;

use crate::voxtral_chat::{
    VoxtralChatConfig, VoxtralChatModel,
    inputs_processor::{AUDIO_TOKEN, BEGIN_AUDIO, VoxtralChatInputsProcessor},
};

const TEXT_LAYER_INDEX_PATTERN: &str = r"language_model\.model\.layers\.(\d+)\.";
const CONV_KERNEL: usize = 3;
// mistral_common's layout: no spaces around the control tokens, a system prompt in its own block
const CHAT_TEMPLATE: &str = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ '[SYSTEM_PROMPT]' + message['content'] + '[/SYSTEM_PROMPT]' }}{% elif message['role'] == 'user' %}{{ '[INST]' + message['content'] + '[/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}";

/// `MultimodalLoader` for Voxtral Mini and Small (`VoxtralForConditionalGeneration`).
pub struct VoxtralChatLoader;

pub struct VoxtralChatPrefixer;

impl MultimodalPromptPrefixer for VoxtralChatPrefixer {
    // the chat template renders text only; each clip opens the user turn, as mistral_common places it
    fn prefix_audio(&self, audio_indexes: Vec<usize>, prompt: &str) -> String {
        let clips = format!("{BEGIN_AUDIO}{AUDIO_TOKEN}").repeat(audio_indexes.len());
        format!("{clips}{prompt}")
    }
}

impl MultimodalModelLoader for VoxtralChatLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg = VoxtralChatConfig::from_json(config)?;
        Ok(Box::new(VoxtralChatModel::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(VoxtralChatConfig::from_json(config)?))
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
        Arc::new(VoxtralChatPrefixer)
    }
    fn default_chat_template(&self, _config: &str) -> Option<String> {
        Some(CHAT_TEMPLATE.to_string())
    }
    // llama.cpp's Voxtral GGUFs embed another Mistral model's template (Devstral's agent prompt among them)
    fn template_overrides_gguf(&self, _config: &str) -> bool {
        true
    }
    fn default_bos_eos(&self, _config: &str) -> Option<(String, String)> {
        Some(("<s>".to_string(), "</s>".to_string()))
    }
    fn get_device_for_tensor(
        &self,
        config: &str,
        _mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        let cfg = VoxtralChatConfig::from_json(config)?;
        layer_indexed_device(
            TEXT_LAYER_INDEX_PATTERN,
            cfg.text_config.num_hidden_layers,
            loading_isq,
        )
    }
}

impl IsqModelLoader for VoxtralChatLoader {
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

impl DeviceMappedModelLoader for VoxtralChatLoader {
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
        let cfg = VoxtralChatConfig::from_json(config)?;
        let seq_len = max_seq_len.min(&ATTENTION_CHUNK_SIZE)
            + VoxtralChatInputsProcessor::tokens_per_chunk(&cfg) * max_num_images.max(&1);
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
        let cfg = VoxtralChatConfig::from_json(config)?;
        let ac = &cfg.audio_config;
        // one 30 s chunk's attention scores over its encoder positions
        let positions = ac.max_source_positions;
        Ok(max_batch_size * ac.num_attention_heads * positions * positions)
    }
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = VoxtralChatConfig::from_json(config)?;
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
        let (d, ffn) = (ac.hidden_size, ac.intermediate_size);
        let convs = (ac.num_mel_bins * d * CONV_KERNEL + d) + (d * d * CONV_KERNEL + d);
        // q, v and out carry biases, k does not; two layer norms
        let layer = 4 * d * d + 3 * d + 2 * d * ffn + ffn + d + 4 * d;
        let audio = convs + ac.max_source_positions * d + ac.num_hidden_layers * layer + 2 * d;
        let projector = cfg.stacked_width() * tc.hidden_size + tc.hidden_size * tc.hidden_size;
        Ok((text + audio + projector) * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = VoxtralChatConfig::from_json(config)?.text_config;
        Ok(super::mistral3::decoder_shape(&cfg).layer_sizes_in_bytes(
            cfg.num_hidden_layers,
            dtype,
            weight_pack_factor,
        ))
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Ok(VoxtralChatConfig::from_json(config)?
            .text_config
            .num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = VoxtralChatConfig::from_json(config)?.text_config;
        Ok(Box::new(super::mistral3::decoder_shape(&cfg).model_config(
            cfg.num_hidden_layers,
            cfg.max_position_embeddings,
            cfg.sliding_window,
        )))
    }
    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Audio])
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use inference_nn::audio::whisper::WhisperEncoder;
    use inference_nn::layers::linear_no_bias;
    use inference_nn::testing::load_synthesized;

    use super::*;

    const TINY: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../inference/tests/fixtures/voxtral_chat/tiny/config.json"
    );

    // the device map sizes the audio tower, the projector and the embeddings by non_mapped_size_in_bytes
    #[test]
    fn non_mapped_size_counts_the_loaded_audio_tower_and_projector() -> anyhow::Result<()> {
        let config = std::fs::read_to_string(TINY)?;
        let cfg = VoxtralChatConfig::from_json(&config)?;
        let hidden = cfg.text_config.hidden_size;
        let (_, names) = load_synthesized(&[], HashMap::new(), DType::F32, |vb| {
            WhisperEncoder::new(&cfg.encoder(), vb.pp("audio_tower"))?;
            let projector = vb.pp("multi_modal_projector");
            linear_no_bias(cfg.stacked_width(), hidden, projector.pp("linear_1"))?;
            linear_no_bias(hidden, hidden, projector.pp("linear_2"))?;
            Ok(())
        })?;
        let tc = &cfg.text_config;
        let text = 2 * tc.vocab_size * tc.hidden_size + tc.hidden_size;
        let towers = names.values().sum::<usize>();
        let bytes =
            VoxtralChatLoader.non_mapped_size_in_bytes(&config, DType::F32, 1, None, None)?;
        assert_eq!(bytes, (text + towers) * DType::F32.size_in_bytes());
        Ok(())
    }
}
