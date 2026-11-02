use super::*;

use inference_nn::utils::varbuilder_utils::DeviceForLoadTensor;

use crate::nemotron_transcribe::{Config, NemotronTranscribeModel};

/// `MultimodalLoader` for Nemotron 3.5 Transcribe (`Nemotron3_5TranscribeForConditionalGeneration`), from NVIDIA's
/// NeMo checkpoint.
pub struct NemotronTranscribeLoader;

pub struct NemotronTranscribePrefixer;

// the checkpoint's template writes one `<|audio|>` per audio part, behind the text, as NeMo trains it
impl MultimodalPromptPrefixer for NemotronTranscribePrefixer {}

fn text_config(config: &str) -> Result<String> {
    Ok(serde_json::to_string(
        &Config::from_json(config)?.text_config,
    )?)
}

impl MultimodalModelLoader for NemotronTranscribeLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg = Config::from_json(config)?;
        Ok(Box::new(NemotronTranscribeModel::new(
            &cfg,
            config,
            vb,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(Config::from_json(config)?))
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
        Arc::new(NemotronTranscribePrefixer)
    }
    fn audio_transcriber(&self, _config: &str) -> Option<AudioTranscriber> {
        Some(crate::nemotron_transcribe::transcription::TRANSCRIBER)
    }
    fn get_device_for_tensor(
        &self,
        config: &str,
        mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        let decoder =
            NemotronHLoader.get_device_for_tensor(&text_config(config)?, mapper, loading_isq)?;
        // the audio tower's layers are not the decoder's
        Ok(Arc::new(move |name: String| {
            if name.starts_with(crate::nemotron_transcribe::LLM) {
                decoder(name)
            } else {
                DeviceForLoadTensor::Base
            }
        }))
    }
}

impl IsqModelLoader for NemotronTranscribeLoader {
    fn promoted_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        NemotronHLoader.promoted_isq_predicates(&text_config(config)?)
    }
    // anchored to the decoder: the audio encoders' layers are left as they are
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^llm\.lm_head\.(weight|bias)$",
            r"^llm\.model\.layers\.(\d+)\.mixer\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$",
            r"^llm\.model\.layers\.(\d+)\.mixer\.(in_proj|out_proj)\.(weight|bias)$",
            r"^llm\.model\.layers\.(\d+)\.mixer\.(up_proj|down_proj)\.(weight|bias)$",
            r"^llm\.model\.layers\.(\d+)\.mixer\.shared_experts\.(up_proj|down_proj)\.(weight|bias)$",
            r"^llm\.model\.layers\.(\d+)\.mixer\.experts\.(gate_and_up_projs|down_projs)\.weight$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for NemotronTranscribeLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Multimodal {
            max_seq_len,
            max_batch_size,
            ..
        } = params
        else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };
        let cfg = Config::from_json(config)?;
        let seq_len = max_seq_len.min(&ATTENTION_CHUNK_SIZE);
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
        let asr = Config::from_json(config)?.audio_config.asr_config;
        // one chunk's attention scores, its context included
        let positions = crate::nemotron_transcribe::ENCODER_WINDOW;
        Ok(max_batch_size * asr.num_attention_heads * positions * positions)
    }
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let text = NemotronHLoader.non_mapped_size_in_bytes(
            &text_config(config)?,
            dtype,
            weight_pack_factor,
            quantization,
            matformer_config,
        )?;
        let asr = Config::from_json(config)?.audio_config.asr_config;
        let (h, ffn) = (asr.hidden_size, asr.intermediate_size);
        // the speech encoder; the speaker branch (about 100 M) and the projector beside it, all in f32
        let encoder = asr.num_hidden_layers * (4 * h * h + 2 * h * ffn)
            + asr.subsampling_factor * asr.num_mel_bins * h;
        Ok(text
            + (encoder + crate::nemotron_transcribe::SPEAKER_BRANCH_PARAMS)
                * DType::F32.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        NemotronHLoader.layer_sizes_in_bytes(
            &text_config(config)?,
            dtype,
            weight_pack_factor,
            matformer_config,
        )
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Ok(NemotronHLoader
            .model_config(&text_config(config)?)?
            .num_layers())
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        NemotronHLoader.model_config(&text_config(config)?)
    }
    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Audio])
    }
}
