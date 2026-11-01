use super::*;

use crate::qwen2_audio::{
    AudioFlamingo3Config, PooledAudioSpec, Projector, Qwen2AudioConfig, Qwen2AudioModel,
    inputs_processor::{AUDIO_BOS, AUDIO_EOS, AUDIO_TOKEN, clip_tokens},
};

const TEXT_LAYER_INDEX_PATTERN: &str = r"language_model\.model\.layers\.(\d+)\.";
const CONV_KERNEL: usize = 3;
// one 30 s window, what a clip's attention budget is sized by
const WINDOW_SAMPLES: usize = 480_000;

/// `MultimodalLoader` for Qwen2-Audio (`Qwen2AudioForConditionalGeneration`).
pub struct Qwen2AudioLoader;

/// `MultimodalLoader` for Audio Flamingo 3 (`AudioFlamingo3ForConditionalGeneration`).
pub struct AudioFlamingo3Loader;

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

pub struct AudioFlamingo3Prefixer;

// the repo's template writes `<sound>` for each audio part; the prompt takes nothing more
impl MultimodalPromptPrefixer for AudioFlamingo3Prefixer {}

fn non_mapped_bytes(
    spec: &PooledAudioSpec,
    dtype: DType,
    weight_pack_factor: usize,
    quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
) -> Result<usize> {
    let tc = &spec.text;
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
    let ac = &spec.audio;
    let (d, ffn, h) = (ac.d_model, ac.encoder_ffn_dim, tc.hidden_size);
    let convs = (ac.num_mel_bins * d * CONV_KERNEL + d) + (d * d * CONV_KERNEL + d);
    // q, v and out carry biases, k does not; two layer norms
    let layer = 4 * d * d + 3 * d + 2 * d * ffn + ffn + d + 4 * d;
    let audio = convs + ac.max_source_positions * d + ac.encoder_layers * layer + 2 * d;
    let projector = match spec.projector {
        Projector::Linear => d * h + h,
        Projector::Mlp { bias, .. } => d * h + h * h + if bias { 2 * h } else { 0 },
    };
    Ok((text + audio + projector) * dtype.size_in_bytes())
}

// the two loaders differ only in their config and in who writes each clip's placeholder
macro_rules! pooled_audio_loader {
    ($loader:ident, $config:ident, $prefixer:ident, $transcriber:expr) => {
        impl $loader {
            fn spec(config: &str) -> Result<PooledAudioSpec> {
                Ok($config::from_json(config)?.spec())
            }
        }

        impl MultimodalModelLoader for $loader {
            fn load(
                &self,
                config: &str,
                vb: ShardedVarBuilder,
                normal_loading_metadata: NormalLoadingMetadata,
                attention_mechanism: AttentionImplementation,
            ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
                Ok(Box::new(Qwen2AudioModel::new(
                    &Self::spec(config)?,
                    vb,
                    self.is_gptx_for(config, &normal_loading_metadata)?,
                    normal_loading_metadata,
                    attention_mechanism,
                )?))
            }
            fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
                Ok(Box::new($config::from_json(config)?))
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
                Arc::new($prefixer)
            }
            fn audio_transcriber(&self, _config: &str) -> Option<AudioTranscriber> {
                $transcriber
            }
            fn get_device_for_tensor(
                &self,
                config: &str,
                _mapper: &dyn DeviceMapper,
                loading_isq: bool,
            ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
                layer_indexed_device(
                    TEXT_LAYER_INDEX_PATTERN,
                    Self::spec(config)?.text.num_hidden_layers,
                    loading_isq,
                )
            }
        }

        impl IsqModelLoader for $loader {
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

        impl DeviceMappedModelLoader for $loader {
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
                let spec = Self::spec(config)?;
                let clip = clip_tokens(WINDOW_SAMPLES, spec.max_windows);
                let seq_len = max_seq_len.min(&ATTENTION_CHUNK_SIZE) + clip * max_num_images.max(&1);
                Ok(max_batch_size * spec.text.num_attention_heads * seq_len * seq_len)
            }
            fn non_mapped_max_act_size_elems(
                &self,
                config: &str,
                params: &AutoDeviceMapParams,
            ) -> Result<usize> {
                let AutoDeviceMapParams::Multimodal { max_batch_size, .. } = params else {
                    anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
                };
                let ac = Self::spec(config)?.audio;
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
                non_mapped_bytes(&Self::spec(config)?, dtype, weight_pack_factor, quantization)
            }
            fn layer_sizes_in_bytes(
                &self,
                config: &str,
                dtype: DType,
                weight_pack_factor: usize,
                matformer_config: Option<&MatformerSliceConfig>,
            ) -> Result<Vec<usize>> {
                Qwen2Loader.layer_sizes_in_bytes(
                    &serde_json::to_string(&Self::spec(config)?.text)?,
                    dtype,
                    weight_pack_factor,
                    matformer_config,
                )
            }
            fn num_layers(&self, config: &str) -> Result<usize> {
                Ok(Self::spec(config)?.text.num_hidden_layers)
            }
            fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
                Some(vec![NonMappedSubModel::Audio])
            }
            fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
                Qwen2Loader.model_config(&serde_json::to_string(&Self::spec(config)?.text)?)
            }
        }
    };
}

// Qwen2-Audio's processor has no transcription request
pooled_audio_loader!(Qwen2AudioLoader, Qwen2AudioConfig, Qwen2AudioPrefixer, None);
pooled_audio_loader!(
    AudioFlamingo3Loader,
    AudioFlamingo3Config,
    AudioFlamingo3Prefixer,
    Some(crate::qwen2_audio::transcription::AUDIO_FLAMINGO3)
);

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

    fn loaded(spec: &PooledAudioSpec) -> anyhow::Result<usize> {
        let (d, h) = (spec.audio.d_model, spec.text.hidden_size);
        let (_, names) = load_synthesized(&[], HashMap::new(), DType::F32, |vb| {
            WhisperEncoder::new(&spec.encoder(), vb.pp("audio_tower"))?;
            let projector = vb.pp("multi_modal_projector");
            match spec.projector {
                Projector::Linear => {
                    linear_b(d, h, true, projector.pp("linear"))?;
                }
                Projector::Mlp { bias, .. } => {
                    linear_b(d, h, bias, projector.pp("linear_1"))?;
                    linear_b(h, h, bias, projector.pp("linear_2"))?;
                }
            }
            Ok(())
        })?;
        let text = 2 * spec.text.vocab_size * h + h;
        Ok(text + names.values().sum::<usize>())
    }

    // the device map sizes the audio tower, the projector and the embeddings by non_mapped_size_in_bytes
    #[test]
    fn non_mapped_size_counts_the_loaded_audio_tower_and_projector() -> anyhow::Result<()> {
        let config = std::fs::read_to_string(TINY)?;
        let bytes =
            Qwen2AudioLoader.non_mapped_size_in_bytes(&config, DType::F32, 1, None, None)?;
        let spec = Qwen2AudioLoader::spec(&config)?;
        assert_eq!(bytes, loaded(&spec)? * DType::F32.size_in_bytes());
        // Audio Flamingo 3's two-layer projector over the same tower
        let mut af3: serde_json::Value = serde_json::from_str(&config)?;
        af3["audio_token_id"] = af3["audio_token_index"].clone();
        af3["projector_hidden_act"] = "gelu".into();
        af3["projector_bias"] = true.into();
        let af3 = af3.to_string();
        let bytes =
            AudioFlamingo3Loader.non_mapped_size_in_bytes(&af3, DType::F32, 1, None, None)?;
        let spec = AudioFlamingo3Loader::spec(&af3)?;
        assert_eq!(bytes, loaded(&spec)? * DType::F32.size_in_bytes());
        Ok(())
    }
}
