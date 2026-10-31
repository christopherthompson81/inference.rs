use super::*;

use crate::qwen3_asr::{
    Config as Qwen3AsrConfig, Qwen3AsrModel,
    audio::audio_token_count,
    inputs_processor::{AUDIO_END, AUDIO_START, AUDIO_TOKEN, SAMPLE_RATE},
};

const TEXT_LAYER_INDEX_PATTERN: &str = r"(?:thinker\.model|model\.language_model)\.layers\.(\d+)\.";
// a 30 s clip budgets the encoder's activations
const MAX_CLIP_SECONDS: usize = 30;
const MEL_HOP: usize = 160;

/// `MultimodalLoader` for Qwen3-ASR.
pub struct Qwen3AsrLoader;

pub struct Qwen3AsrPrefixer;

impl MultimodalPromptPrefixer for Qwen3AsrPrefixer {
    // the chat template renders text only; each clip opens the user turn, as the processor's template places it
    fn prefix_audio(&self, audio_indexes: Vec<usize>, prompt: &str) -> String {
        let clips = format!("{AUDIO_START}{AUDIO_TOKEN}{AUDIO_END}").repeat(audio_indexes.len());
        format!("{clips}{prompt}")
    }
}

impl Qwen3AsrLoader {
    fn text_config(cfg: &Qwen3AsrConfig) -> Result<String> {
        Ok(serde_json::to_string(
            &cfg.thinker_config.text_config.qwen3(),
        )?)
    }

    fn max_clip_tokens(cfg: &Qwen3AsrConfig) -> usize {
        let frames = MAX_CLIP_SECONDS * SAMPLE_RATE as usize / MEL_HOP;
        audio_token_count(frames, cfg.thinker_config.audio_config.n_window)
    }
}

impl MultimodalModelLoader for Qwen3AsrLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg = Qwen3AsrConfig::from_json(config)?;
        Ok(Box::new(Qwen3AsrModel::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(Qwen3AsrConfig::from_json(config)?))
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
        Arc::new(Qwen3AsrPrefixer)
    }
    fn audio_transcriber(&self, _config: &str) -> Option<AudioTranscriber> {
        Some(crate::qwen3_asr::transcription::TRANSCRIBER)
    }
    fn get_device_for_tensor(
        &self,
        config: &str,
        _mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        let cfg = Qwen3AsrConfig::from_json(config)?;
        layer_indexed_device(
            TEXT_LAYER_INDEX_PATTERN,
            cfg.thinker_config.text_config.num_hidden_layers,
            loading_isq,
        )
    }
}

impl IsqModelLoader for Qwen3AsrLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^(thinker\.model|model\.language_model)\.embed_tokens\.weight$",
            r"^(thinker\.)?lm_head\.(weight|bias)$",
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^(thinker\.)?lm_head\.(weight|bias)$",
            r"(thinker\.model|model\.language_model)\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$",
            r"(thinker\.model|model\.language_model)\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Qwen3AsrLoader {
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
        let cfg = Qwen3AsrConfig::from_json(config)?;
        let seq_len = max_seq_len.min(&ATTENTION_CHUNK_SIZE)
            + Self::max_clip_tokens(&cfg) * max_num_images.max(&1);
        let heads = cfg.thinker_config.text_config.num_attention_heads;
        Ok(max_batch_size * heads * seq_len * seq_len)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Multimodal { max_batch_size, .. } = params else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };
        let cfg = Qwen3AsrConfig::from_json(config)?;
        let ac = &cfg.thinker_config.audio_config;
        // the first conv's channels over a chunk's halved mel grid, or a window's attention, whichever is larger
        let chunks = MAX_CLIP_SECONDS * SAMPLE_RATE as usize / MEL_HOP / (2 * ac.n_window);
        let conv = chunks * ac.downsample_hidden_size * ac.num_mel_bins.div_ceil(2) * ac.n_window;
        let window = audio_token_count(ac.n_window_infer, ac.n_window);
        let attention = Self::max_clip_tokens(&cfg) * ac.encoder_attention_heads * window;
        Ok(max_batch_size * conv.max(attention))
    }
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = Qwen3AsrConfig::from_json(config)?;
        let tc = &cfg.thinker_config.text_config;
        let (embed, head) = (
            format!("{}.embed_tokens.weight", cfg.layout.text_model()),
            format!("{}.weight", cfg.layout.lm_head()),
        );
        let (embed_pack, head_pack) = super::language_model_pack_factors(
            quantization,
            &embed,
            &head,
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
        let ac = &cfg.thinker_config.audio_config;
        let (d, c) = (ac.d_model, ac.downsample_hidden_size);
        let freq = ac.num_mel_bins.div_ceil(2).div_ceil(2).div_ceil(2);
        let convs = (c * 9 + c) + 2 * (c * c * 9 + c);
        let layer = 4 * (d * d + d) + 2 * (d * ac.encoder_ffn_dim) + ac.encoder_ffn_dim + d + 4 * d;
        let audio = convs
            + c * freq * d
            + ac.encoder_layers * layer
            + 2 * d
            + (d * d + d)
            + (d * ac.output_dim + ac.output_dim);
        Ok((text + audio) * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = Qwen3AsrConfig::from_json(config)?;
        Qwen3Loader.layer_sizes_in_bytes(
            &Self::text_config(&cfg)?,
            dtype,
            weight_pack_factor,
            matformer_config,
        )
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Ok(Qwen3AsrConfig::from_json(config)?
            .thinker_config
            .text_config
            .num_hidden_layers)
    }
    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Audio])
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = Qwen3AsrConfig::from_json(config)?;
        Qwen3Loader.model_config(&Self::text_config(&cfg)?)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use inference_nn::testing::load_synthesized;
    use inference_tensor::Device;

    use super::*;
    use crate::qwen3_asr::audio::AudioEncoder;

    const TINY: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../inference/tests/fixtures/qwen3_asr/tiny/config.json"
    );

    // the device map sizes the audio tower and the embeddings by non_mapped_size_in_bytes
    #[test]
    fn non_mapped_size_counts_the_loaded_audio_tower() -> anyhow::Result<()> {
        let config = std::fs::read_to_string(TINY)?;
        let cfg = Qwen3AsrConfig::from_json(&config)?;
        let (_, names) = load_synthesized(&[], HashMap::new(), DType::F32, |vb| {
            Ok(AudioEncoder::new(
                &cfg.thinker_config.audio_config,
                vb,
                cfg.layout,
                &Device::Cpu,
            )?)
        })?;
        let tc = &cfg.thinker_config.text_config;
        let text = tc.vocab_size * tc.hidden_size + tc.hidden_size;
        let audio = names.values().sum::<usize>();
        let bytes = Qwen3AsrLoader.non_mapped_size_in_bytes(&config, DType::F32, 1, None, None)?;
        assert_eq!(bytes, (text + audio) * DType::F32.size_in_bytes());
        Ok(())
    }
}
