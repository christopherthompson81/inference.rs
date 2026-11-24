//! Inkling 2 with its vision tower: the text model reads each picture's rows in place of its pad tokens' embeddings.

pub mod inputs_processor;

use std::any::Any;
use std::sync::{Arc, Mutex};

use inference_quant::ShardedVarBuilder;
use inference_tensor::{Device, Result, Tensor};

use crate::inkling2_vision::{VisionConfig, VisionTower};
use crate::{
    amoe::AnyMoeBaseModelMixin,
    kv_cache::EitherCache,
    model::{IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata, NormalModel},
    paged_attention::{
        AttentionImplementation, ModelConfigLike, ModelConfigMetadata,
        encoder_cache::{CacheModality, EncoderCacheManager},
    },
};

/// `<|unused_200054|>`: one per cell, its embedding replaced by the cell's row.
pub const IMAGE_PAD_TOKEN: &str = "<|unused_200054|>";
pub const DEFAULT_IMAGE_TOKEN_ID: u32 = 200054;
// pictures whose rows stay cached, as the other vision models keep theirs
const ENCODER_CACHE_PICTURES: usize = 32;

/// The checkpoint's config: the text model's (`text_config`) beside the tower's.
#[derive(Debug, Clone)]
pub struct Config {
    pub text: crate::inkling2::Config,
    pub vision: VisionConfig,
    pub image_token_id: u32,
}

impl Config {
    pub fn parse(config: &str) -> anyhow::Result<Self> {
        use inference_json::FromJson;
        let value: serde_json::Value = serde_json::from_str(config)?;
        let vision = value
            .get("vision_config")
            .ok_or_else(|| anyhow::anyhow!("Inkling 2 vision needs a `vision_config`"))?;
        let image_token_id = value
            .get("image_token_id")
            .and_then(serde_json::Value::as_u64)
            .map_or(Ok(DEFAULT_IMAGE_TOKEN_ID), u32::try_from)?;
        Ok(Self {
            text: crate::loaders::inkling2_text_config(config)?,
            vision: VisionConfig::from_json(vision)?,
            image_token_id,
        })
    }
}

/// Each picture's native patch grid, in the order of `pixel_values`' rows, and its content hash when known (the
/// encoder cache's key).
#[derive(Default)]
pub struct Inkling2VlArgs {
    pub grids: Vec<(usize, usize)>,
    pub hashes: Vec<Option<u64>>,
}

pub struct Inkling2VlModel {
    text: crate::inkling2::Model,
    tower: VisionTower,
    image_token_id: u32,
    encoder_cache: Mutex<EncoderCacheManager>,
}

impl Inkling2VlModel {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let device = normal_loading_metadata.real_device.clone();
        // before the text model rebinds `vb` to its own names
        let tower = VisionTower::new(
            &cfg.vision,
            vb.clone()
                .without_lora_registry()
                .pp("model")
                .pp("visual")
                .set_device(device),
        )?;
        let text = crate::inkling2::Model::new(
            &cfg.text,
            vb,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        Ok(Self {
            text,
            tower,
            image_token_id: cfg.image_token_id,
            encoder_cache: Mutex::new(EncoderCacheManager::new(ENCODER_CACHE_PICTURES)),
        })
    }

    // each picture's rows from the encoder cache, or the tower on a miss (then kept)
    fn image_rows(&self, pixel_values: &Tensor, args: &Inkling2VlArgs) -> Result<Tensor> {
        let mut start = 0;
        let rows = args
            .grids
            .iter()
            .enumerate()
            .map(|(i, &(gh, gw))| {
                let patches = pixel_values.narrow(0, start, gh * gw)?;
                start += gh * gw;
                let hash = args.hashes.get(i).copied().flatten();
                if let Some(hash) = hash {
                    let cached = self
                        .encoder_cache
                        .lock()
                        .expect("encoder cache lock poisoned")
                        .get(CacheModality::Image, hash);
                    if let Some(cached) = cached {
                        return Ok(cached[0].clone());
                    }
                }
                let rows = self.tower.forward(&patches, gh, gw)?;
                if let Some(hash) = hash {
                    self.encoder_cache
                        .lock()
                        .expect("encoder cache lock poisoned")
                        .insert(CacheModality::Image, hash, vec![rows.clone()]);
                }
                Ok(rows)
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&rows, 0)
    }
}

impl IsqModel for Inkling2VlModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        self.text.residual_tensors()
    }
}

impl crate::model::BlockDiffusionMixin for Inkling2VlModel {}
impl AnyMoeBaseModelMixin for Inkling2VlModel {}
inference_nn::delegate_speculative_target!(Inkling2VlModel, text: crate::inkling2::Model);

impl MultimodalModel for Inkling2VlModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut x = self.text.embed(input_ids)?;
        if let Some(pixel_values) = pixel_values {
            let args = model_specific_args
                .downcast::<Inkling2VlArgs>()
                .map_err(|_| {
                    inference_tensor::Error::msg("Inkling 2 vision takes Inkling2VlArgs")
                })?;
            let rows = self
                .image_rows(&pixel_values.to_device(x.device())?, &args)?
                .to_dtype(x.dtype())?;
            x = crate::media_inputs::media::place_token_features(
                &x,
                input_ids,
                self.image_token_id,
                &rows,
                "Inkling 2",
            )?;
        }
        self.text.forward_embeds(x, ctx)
    }

    fn set_control_vectors(
        &self,
        set: inference_nn::control_vectors::ControlVectorSet,
    ) -> Result<bool> {
        NormalModel::set_control_vectors(&self.text, set)
    }

    fn control_vectors(&self) -> inference_nn::control_vectors::ControlVectorSet {
        NormalModel::control_vectors(&self.text)
    }

    fn begin_layer_capture(&self) -> bool {
        NormalModel::begin_layer_capture(&self.text)
    }

    fn take_layer_capture(&self) -> Result<Option<Vec<Tensor>>> {
        NormalModel::take_layer_capture(&self.text)
    }

    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs(&self) -> bool {
        NormalModel::supports_cuda_decode_graphs(&self.text)
    }

    // a decode step carries no pictures
    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs_for_args(&self, args: &dyn Any) -> bool {
        args.downcast_ref::<Inkling2VlArgs>()
            .is_some_and(|args| args.grids.is_empty())
            && self.supports_cuda_decode_graphs()
    }

    fn device(&self) -> &Device {
        NormalModel::device(&self.text)
    }

    fn encoder_cache(&self) -> Option<&Mutex<EncoderCacheManager>> {
        Some(&self.encoder_cache)
    }

    fn cache(&self) -> &EitherCache {
        NormalModel::cache(&self.text)
    }

    fn max_seq_len(&self) -> usize {
        NormalModel::max_seq_len(&self.text)
    }

    fn config(&self) -> &ModelConfigMetadata {
        NormalModel::config(&self.text)
    }

    fn model_config(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        NormalModel::model_config(&self.text)
    }

    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn Any> {
        Box::new(Inkling2VlArgs::default())
    }
}
