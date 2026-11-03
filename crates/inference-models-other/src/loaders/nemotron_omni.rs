use super::*;

use crate::nemotron_omni::{Config, LANGUAGE_MODEL, NemotronOmniModel};

/// `MultimodalLoader` for Nemotron-H Omni (`NemotronH_Omni_Reasoning_V3`): images through C-RADIO into Nemotron-H.
pub struct NemotronOmniLoader;

pub struct NemotronOmniPrefixer;

// the checkpoint's template writes `<image>` per image part
impl MultimodalPromptPrefixer for NemotronOmniPrefixer {}

fn text_config(config: &str) -> Result<String> {
    Ok(serde_json::to_string(
        &Config::from_json(config)?.llm_config,
    )?)
}

impl MultimodalModelLoader for NemotronOmniLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        Ok(Box::new(NemotronOmniModel::new(
            &Config::from_json(config)?,
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
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(NemotronOmniPrefixer)
    }
    fn get_device_for_tensor(
        &self,
        config: &str,
        mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        let decoder =
            NemotronHLoader.get_device_for_tensor(&text_config(config)?, mapper, loading_isq)?;
        let prefix = format!("{LANGUAGE_MODEL}.");
        // the vision tower's blocks are not the decoder's layers
        Ok(Arc::new(move |name: String| {
            match name.strip_prefix(&prefix) {
                Some(rest) => decoder(rest.to_string()),
                None => DeviceForLoadTensor::Base,
            }
        }))
    }
}

impl IsqModelLoader for NemotronOmniLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^language_model\.(backbone|model)\.embeddings\.weight$",
            r"^language_model\.lm_head\.(weight|bias)$",
        ])
    }
    // the decoder's: RADIO's blocks are `blocks.N.attn` / `blocks.N.mlp`, never `layers.N.mixer`
    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        NemotronHLoader.isq_layer_regexes(&text_config(config)?)
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for NemotronOmniLoader {
    fn checkpoint_layer_groups(&self, config: &str) -> Result<Option<Vec<usize>>> {
        NemotronHLoader.checkpoint_layer_groups(&text_config(config)?)
    }

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
        Ok(max_batch_size * cfg.llm_config.num_attention_heads * seq_len * seq_len)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Multimodal { max_batch_size, .. } = params else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };
        let cfg = Config::from_json(config)?;
        // the largest image's MLP activations (attention runs fused); the projector's are no wider
        let v = &cfg.vision_config;
        Ok(max_batch_size * v.max_num_patches * v.intermediate_size())
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
        let cfg = Config::from_json(config)?;
        let v = &cfg.vision_config;
        let (h, ffn) = (v.hidden_size, v.intermediate_size());
        let merged = h * cfg.merge() * cfg.merge();
        let side = v.max_img_size / cfg.patch_size;
        let tower = v.num_hidden_layers * (4 * h * h + 2 * h * ffn)
            + 3 * cfg.patch_size * cfg.patch_size * h;
        let projector = merged * cfg.projector_hidden_size
            + cfg.projector_hidden_size * cfg.llm_config.hidden_size;
        // the position grid stays in f32, which it is resized in
        let positions = side * side * h * DType::F32.size_in_bytes();
        Ok(text + (tower + projector) * dtype.size_in_bytes() + positions)
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
        Some(vec![NonMappedSubModel::Vision])
    }
}
