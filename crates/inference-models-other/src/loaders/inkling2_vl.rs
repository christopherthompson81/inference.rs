use super::*;

use crate::inkling2_vl::{Config as Inkling2VlConfig, Inkling2VlModel};

// pictures past this many cells are scaled down to fit (llama.cpp's cap); it bounds one encode's activations
pub const MAX_IMAGE_CELLS: usize = 4096;

/// `MultimodalLoader` for Inkling 2 with its vision tower; everything the text model sizes or lays out is
/// [`Inkling2Loader`]'s.
pub struct Inkling2VlLoader;

pub struct Inkling2VlPrefixer;

// the chat template writes the pad token for each picture
impl MultimodalPromptPrefixer for Inkling2VlPrefixer {
    fn prefix_image(&self, _image_indexes: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
}

fn parse(config: &str) -> Result<Inkling2VlConfig> {
    Inkling2VlConfig::parse(config)
}

impl MultimodalModelLoader for Inkling2VlLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        Ok(Box::new(Inkling2VlModel::new(
            &parse(config)?,
            vb,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(parse(config)?))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        cfg!(feature = "cuda")
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        true
    }
    fn supports_encoder_cache(&self, _config: &str) -> bool {
        true
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(Inkling2VlPrefixer)
    }
}

impl IsqModelLoader for Inkling2VlLoader {
    fn promoted_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        Inkling2Loader.promoted_isq_predicates(config)
    }
    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        Inkling2Loader.isq_layer_regexes(config)
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        Inkling2Loader.immediate_isq_predicates(config)
    }
}

impl DeviceMappedModelLoader for Inkling2VlLoader {
    // one picture's encode at the cap: the MLP's widened rows and the residual's copies, f32
    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let vision = parse(config)?.vision;
        let (h, w) = match params {
            AutoDeviceMapParams::Multimodal {
                max_image_shape, ..
            } => *max_image_shape,
            AutoDeviceMapParams::Text { .. } => (0, 0),
        };
        let (cols, rows) = vision.cells(w, h);
        let merge = vision.adapter_merge_size * vision.adapter_merge_size;
        let patches = (cols * rows).min(MAX_IMAGE_CELLS) * merge + 1;
        Ok(patches * (vision.intermediate_size() + 4 * vision.hidden_size))
    }
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let text = Inkling2Loader.non_mapped_size_in_bytes(
            config,
            dtype,
            weight_pack_factor,
            quantization,
            matformer_config,
        )?;
        Ok(text + parse(config)?.vision.num_weights() * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        Inkling2Loader.layer_sizes_in_bytes(config, dtype, weight_pack_factor, matformer_config)
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Inkling2Loader.num_layers(config)
    }
    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
    fn speculative_row_bytes(&self, config: &str) -> Result<Option<usize>> {
        Inkling2Loader.speculative_row_bytes(config)
    }
    fn prompt_activation_bytes_per_token(&self, config: &str) -> Result<Option<usize>> {
        Inkling2Loader.prompt_activation_bytes_per_token(config)
    }
    fn checkpoint_tensor_bytes(
        &self,
        config: &str,
        tensor_name: &str,
        dtype: &str,
        stored_bytes: usize,
    ) -> Option<usize> {
        Inkling2Loader.checkpoint_tensor_bytes(config, tensor_name, dtype, stored_bytes)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        Inkling2Loader.model_config(config)
    }
    fn model_config_for(
        &self,
        config: &str,
        device: &inference_tensor::Device,
        paged: Option<crate::paged_attention::PagedKvLayout>,
    ) -> Result<Box<dyn ModelConfigLike>> {
        Inkling2Loader.model_config_for(config, device, paged)
    }
}
