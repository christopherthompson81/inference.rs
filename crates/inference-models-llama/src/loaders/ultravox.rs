use super::*;

use inference_nn::device_map::DeviceMapper;
use inference_nn::utils::varbuilder_utils::DeviceForLoadTensor;

use crate::ultravox::{
    UltravoxConfig, UltravoxModel,
    config::with_text_config,
    inputs_processor::{AUDIO_TOKEN, UltravoxInputsProcessor},
};

const TEXT_LAYER_INDEX_PATTERN: &str = r"^model\.layers\.(\d+)\.";
const CONV_KERNEL: usize = 3;
// one 30 s chunk, what a clip's attention budget is sized by
const CHUNK_FRAMES: usize = 3000;

/// `MultimodalLoader` for Ultravox (`UltravoxModel`): its GGUFs, or a safetensors repo with the text model it names.
pub struct UltravoxLoader;

pub struct UltravoxPrefixer;

impl MultimodalPromptPrefixer for UltravoxPrefixer {
    // each clip opens the user turn, where the reference's `<|audio|>` sits
    fn prefix_audio(&self, audio_indexes: Vec<usize>, prompt: &str) -> String {
        format!("{}{prompt}", AUDIO_TOKEN.repeat(audio_indexes.len()))
    }
}

impl MultimodalModelLoader for UltravoxLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg = UltravoxConfig::from_json(config)?;
        Ok(Box::new(UltravoxModel::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(UltravoxConfig::from_json(config)?))
    }
    fn with_companion_config(
        &self,
        config: &str,
        companion: &str,
        tokenizer: &str,
    ) -> Result<String> {
        with_text_config(config, companion, tokenizer)
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
        Arc::new(UltravoxPrefixer)
    }
    fn get_device_for_tensor(
        &self,
        config: &str,
        _mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        let cfg = UltravoxConfig::from_json(config)?;
        layer_indexed_device(
            TEXT_LAYER_INDEX_PATTERN,
            cfg.text_config.num_hidden_layers,
            loading_isq,
        )
    }
}

impl IsqModelLoader for UltravoxLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^model\.embed_tokens\.weight$",
            r"^lm_head\.(weight|bias)$",
        ])
    }

    // anchored to the decoder: the audio tower's layers share the `layers.N.self_attn` names
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^lm_head\.(weight|bias)$",
            r"^model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$",
            r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for UltravoxLoader {
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
        let cfg = UltravoxConfig::from_json(config)?;
        let chunk = UltravoxInputsProcessor::new(cfg.clone()).chunk_tokens(CHUNK_FRAMES);
        let seq_len = max_seq_len.min(&ATTENTION_CHUNK_SIZE) + chunk * max_num_images.max(&1);
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
        let cfg = UltravoxConfig::from_json(config)?;
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
        let cfg = UltravoxConfig::from_json(config)?;
        let tc = &cfg.text_config;
        let (embed_pack, head_pack) = super::language_model_pack_factors(
            quantization,
            "model.embed_tokens.weight",
            "lm_head.weight",
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
        let (stacked, inner) = (cfg.stacked_width(), cfg.hidden_size);
        let projector = stacked + stacked * inner + inner / 2 + inner / 2 * tc.hidden_size;
        Ok((text + audio + projector) * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = UltravoxConfig::from_json(config)?;
        LlamaLoader.layer_sizes_in_bytes(
            &serde_json::to_string(&cfg.text_config)?,
            dtype,
            weight_pack_factor,
            _matformer_config,
        )
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Ok(UltravoxConfig::from_json(config)?
            .text_config
            .num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = UltravoxConfig::from_json(config)?;
        LlamaLoader.model_config(&serde_json::to_string(&cfg.text_config)?)
    }
    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Audio])
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use inference_nn::audio::whisper::WhisperEncoder;
    use inference_nn::layers::{RmsNorm, linear_no_bias};
    use inference_nn::testing::load_synthesized;

    use super::*;

    const TINY: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../inference/tests/fixtures/ultravox/tiny/config.json"
    );

    // the device map sizes the audio tower, the projector and the embeddings by non_mapped_size_in_bytes
    #[test]
    fn non_mapped_size_counts_the_loaded_audio_tower_and_projector() -> anyhow::Result<()> {
        let config = std::fs::read_to_string(TINY)?;
        let cfg = UltravoxConfig::from_json(&config)?;
        let (stacked, inner, hidden) = (
            cfg.stacked_width(),
            cfg.hidden_size,
            cfg.text_config.hidden_size,
        );
        let (_, names) = load_synthesized(&[], HashMap::new(), DType::F32, |vb| {
            WhisperEncoder::new(&cfg.encoder(), vb.pp("audio_tower"))?;
            let projector = vb.pp("multi_modal_projector");
            RmsNorm::new(stacked, cfg.norm_eps, projector.pp("ln_pre"))?;
            linear_no_bias(stacked, inner, projector.pp("linear_1"))?;
            RmsNorm::new(inner / 2, cfg.norm_eps, projector.pp("ln_mid"))?;
            linear_no_bias(inner / 2, hidden, projector.pp("linear_2"))?;
            Ok(())
        })?;
        let tc = &cfg.text_config;
        let head = if tc.tie_word_embeddings {
            0
        } else {
            tc.vocab_size * tc.hidden_size
        };
        let text = tc.vocab_size * tc.hidden_size + head + tc.hidden_size;
        let towers = names.values().sum::<usize>();
        let bytes = UltravoxLoader.non_mapped_size_in_bytes(&config, DType::F32, 1, None, None)?;
        assert_eq!(bytes, (text + towers) * DType::F32.size_in_bytes());
        Ok(())
    }
}
