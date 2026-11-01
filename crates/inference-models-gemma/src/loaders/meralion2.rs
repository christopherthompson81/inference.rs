use super::*;

use inference_nn::device_map::DeviceMapper;
use inference_nn::utils::varbuilder_utils::DeviceForLoadTensor;

use crate::meralion2::{
    MERaLiON2Config, MERaLiON2Model,
    inputs_processor::{CHUNK_TOKENS, instruction},
};

const TEXT_LAYER_INDEX_PATTERN: &str = r"text_decoder\.model\.layers\.(\d+)\.";
const CONV_KERNEL: usize = 3;

/// `MultimodalLoader` for MERaLiON-2 (`MERaLiON2ForConditionalGeneration`).
pub struct MERaLiON2Loader;

pub struct MERaLiON2Prefixer;

// the repo's template renders text only; a turn with audio takes the model card's instruction template around its text
impl MultimodalPromptPrefixer for MERaLiON2Prefixer {
    fn prefix_audio(&self, audio_indexes: Vec<usize>, prompt: &str) -> String {
        instruction(prompt, audio_indexes.len())
    }
}

impl MultimodalModelLoader for MERaLiON2Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg = MERaLiON2Config::from_json(config)?;
        Ok(Box::new(MERaLiON2Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(MERaLiON2Config::from_json(config)?))
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
        Arc::new(MERaLiON2Prefixer)
    }
    fn audio_transcriber(&self, _config: &str) -> Option<AudioTranscriber> {
        Some(crate::meralion2::transcription::TRANSCRIBER)
    }
    fn get_device_for_tensor(
        &self,
        config: &str,
        _mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        let cfg = MERaLiON2Config::from_json(config)?;
        layer_indexed_device(
            TEXT_LAYER_INDEX_PATTERN,
            cfg.text_config.num_hidden_layers,
            loading_isq,
        )
    }
}

impl IsqModelLoader for MERaLiON2Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[r"^text_decoder\.model\.embed_tokens\.weight$"])
    }

    // anchored to the decoder: the speech encoder's layers share the `layers.N.self_attn` names
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^text_decoder\.model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$",
            r"^text_decoder\.model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for MERaLiON2Loader {
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
        let cfg = MERaLiON2Config::from_json(config)?;
        // one chunk per clip, what a clip's attention budget is sized by
        let seq_len =
            max_seq_len.min(&ATTENTION_CHUNK_SIZE) + CHUNK_TOKENS * max_num_images.max(&1);
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
        let sc = MERaLiON2Config::from_json(config)?.speech_config;
        // one chunk's attention scores over its encoder positions
        let positions = sc.max_source_positions;
        Ok(max_batch_size * sc.encoder_attention_heads * positions * positions)
    }
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = MERaLiON2Config::from_json(config)?;
        let tc = cfg.text_config();
        let embed_pack = super::tied_promoted_tensor_pack_factor(
            quantization,
            "text_decoder.model.embed_tokens.weight",
            "text_decoder.lm_head.weight",
            dtype,
            weight_pack_factor,
        )?;
        let text = tc.hidden_size * tc.vocab_size / embed_pack + tc.hidden_size;
        let sc = &cfg.speech_config;
        let (d, ffn) = (sc.d_model, sc.encoder_ffn_dim);
        let convs = (sc.num_mel_bins * d * CONV_KERNEL + d) + (d * d * CONV_KERNEL + d);
        // q, v and out carry biases, k does not; two layer norms
        let layer = 4 * d * d + 3 * d + 2 * d * ffn + ffn + d + 4 * d;
        let audio = convs + sc.max_source_positions * d + sc.encoder_layers * layer + 2 * d;
        let wide = cfg.adapter_width();
        let adapter = 2 * d
            + (cfg.stacked_width() * wide + wide)
            + 2 * (wide * wide + wide)
            + (wide * tc.hidden_size + tc.hidden_size);
        Ok((text + audio + adapter) * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = MERaLiON2Config::from_json(config)?.text_config();
        Ok(super::gemma2::decoder_shape(&cfg).layer_sizes_in_bytes(
            cfg.num_hidden_layers,
            dtype,
            weight_pack_factor,
        ))
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Ok(MERaLiON2Config::from_json(config)?
            .text_config
            .num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = MERaLiON2Config::from_json(config)?.text_config();
        Ok(Box::new(super::gemma2::decoder_shape(&cfg).model_config(
            cfg.num_hidden_layers,
            cfg.max_position_embeddings,
            cfg.decoder_spec()?.sliding_window(),
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
    use inference_nn::layers::{layer_norm, linear_b};
    use inference_nn::testing::load_synthesized;

    use super::*;

    const TINY: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../inference/tests/fixtures/meralion2/tiny/config.json"
    );

    // the device map sizes the speech encoder, the adapter and the embeddings by non_mapped_size_in_bytes
    #[test]
    fn non_mapped_size_counts_the_loaded_speech_encoder_and_adapter() -> anyhow::Result<()> {
        let config = std::fs::read_to_string(TINY)?;
        let cfg = MERaLiON2Config::from_json(&config)?;
        let (d, wide, hidden) = (
            cfg.speech_config.d_model,
            cfg.adapter_width(),
            cfg.text_config.hidden_size,
        );
        let (_, names) = load_synthesized(&[], HashMap::new(), DType::F32, |vb| {
            WhisperEncoder::new(&cfg.encoder(), vb.pp("speech_encoder"))?;
            layer_norm(d, 1e-5, vb.pp("ln_speech"))?;
            let adapter = vb.pp("speech_audio_adapter");
            linear_b(
                cfg.stacked_width(),
                wide,
                true,
                adapter.pp("mlp_adapter").pp("0"),
            )?;
            linear_b(wide, wide, true, adapter.pp("gate_proj"))?;
            linear_b(wide, wide, true, adapter.pp("pool_proj"))?;
            linear_b(wide, hidden, true, adapter.pp("out_proj"))?;
            Ok(())
        })?;
        let tc = &cfg.text_config;
        let text = tc.vocab_size * tc.hidden_size + tc.hidden_size;
        let towers = names.values().sum::<usize>();
        let bytes = MERaLiON2Loader.non_mapped_size_in_bytes(&config, DType::F32, 1, None, None)?;
        assert_eq!(bytes, (text + towers) * DType::F32.size_in_bytes());
        Ok(())
    }
}
