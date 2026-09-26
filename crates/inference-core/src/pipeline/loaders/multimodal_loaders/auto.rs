use super::*;

/// Automatically selects a MultimodalModelLoader implementation based on the JSON `architectures` field.
pub struct AutoMultimodalLoader;

impl AutoMultimodalLoader {
    fn get_loader(config: &str) -> Result<Box<dyn MultimodalModelLoader>> {
        let auto_cfg: AutoMultimodalLoaderConfig = serde_json::from_str(config)?;

        // Voxtral: params.json has `multimodal` but no `architectures`
        if auto_cfg.multimodal.is_some() && auto_cfg.architectures.is_empty() {
            once_log_debug("Automatic loader type determined to be `voxtral`");
            return Ok(Box::new(VoxtralLoader));
        }

        if auto_cfg.architectures.len() != 1 {
            anyhow::bail!("Expected exactly one architecture in config");
        }

        let name = &auto_cfg.architectures[0];
        let tp = MultimodalLoaderType::from_causal_lm_name(name)?;

        once_log_debug(format!("Automatic loader type determined to be `{tp}`"));

        // Delegate to the concrete loader
        Ok(tp.loader())
    }
}

impl MultimodalModelLoader for AutoMultimodalLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        Self::get_loader(config)?.load(config, vb, normal_loading_metadata, attention_mechanism)
    }
    fn runtime_config<'a>(
        &self,
        config: &'a str,
        max_model_len: Option<usize>,
    ) -> Result<Cow<'a, str>> {
        Self::get_loader(config)?.runtime_config(config, max_model_len)
    }

    fn is_gptx(&self, config: &str) -> bool {
        Self::get_loader(config)
            .expect("AutoMultimodalLoader get_loader")
            .is_gptx(config)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Self::get_loader(config)?.get_config_repr(config)
    }
    fn get_processor(
        &self,
        model_config: &str,
        proc_cfg: Option<ProcessorConfig>,
        preproc_cfg: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Self::get_loader(model_config)
            .expect("AutoMultimodalLoader get_loader")
            .get_processor(model_config, proc_cfg, preproc_cfg, max_edge)
    }
    fn supports_paged_attention(&self, config: &str) -> bool {
        Self::get_loader(config)
            .expect("AutoMultimodalLoader")
            .supports_paged_attention(config)
    }
    fn supports_encoder_cache(&self, config: &str) -> bool {
        Self::get_loader(config)
            .expect("AutoMultimodalLoader")
            .supports_encoder_cache(config)
    }
    fn modalities(&self, config: &str) -> Result<Modalities> {
        Self::get_loader(config)?.modalities(config)
    }
    fn supports_prefix_cacher(&self, config: &str) -> bool {
        Self::get_loader(config)
            .expect("AutoMultimodalLoader")
            .supports_prefix_cacher(config)
    }
    fn auto_device_map_params(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<AutoDeviceMapParams> {
        Self::get_loader(config)?.auto_device_map_params(config, params)
    }

    fn prefixer(&self, config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Self::get_loader(config)
            .expect("AutoMultimodalLoader")
            .prefixer(config)
    }

    fn video_frame_sampling(&self, config: &str) -> crate::VideoFrameSampling {
        Self::get_loader(config)
            .expect("AutoMultimodalLoader")
            .video_frame_sampling(config)
    }

    fn default_chat_template(&self, config: &str) -> Option<String> {
        Self::get_loader(config).ok()?.default_chat_template(config)
    }

    fn default_bos_eos(&self, config: &str) -> Option<(String, String)> {
        Self::get_loader(config).ok()?.default_bos_eos(config)
    }
    fn get_device_for_tensor(
        &self,
        config: &str,
        mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        Self::get_loader(config)?.get_device_for_tensor(config, mapper, loading_isq)
    }
}

impl IsqModelLoader for AutoMultimodalLoader {
    fn promoted_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.promoted_isq_predicates(config)
    }

    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.isq_layer_regexes(config)
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.immediate_isq_predicates(config)
    }
    fn isq_layer_regexes_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.isq_layer_regexes_moqe(config)
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.immediate_isq_predicates_moqe(config)
    }
}

impl DeviceMappedModelLoader for AutoMultimodalLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Self::get_loader(config)?.mapped_max_act_size_elems(config, params)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Self::get_loader(config)?.non_mapped_max_act_size_elems(config, params)
    }
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        Self::get_loader(config)?.non_mapped_size_in_bytes(
            config,
            dtype,
            weight_pack_factor,
            quantization,
            _matformer_config,
        )
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        Self::get_loader(config)?.layer_sizes_in_bytes(
            config,
            dtype,
            weight_pack_factor,
            _matformer_config,
        )
    }
    fn non_mapped_sub_models_for_config(
        &self,
        config: &str,
    ) -> Result<Option<Vec<NonMappedSubModel>>> {
        Self::get_loader(config)?.non_mapped_sub_models_for_config(config)
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Self::get_loader(config)?.num_layers(config)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        Self::get_loader(config)?.model_config(config)
    }
}

#[derive(Deserialize)]
struct AutoMultimodalLoaderConfig {
    #[serde(default)]
    architectures: Vec<String>,
    /// Voxtral params.json uses a `multimodal` key instead of `architectures`.
    #[serde(default)]
    multimodal: Option<serde_json::Value>,
}
