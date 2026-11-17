//! DeepSeek-OCR (`DeepseekOCRForCausalLM`): the DeepEncoder (SAM ViT-B into CLIP-L, projected) over a global view
//! and tiles, its tokens in place of the prompt's `<image>` run, read by a DeepSeek-V2 MoE without MLA.
//! MIT deepseek-ai/DeepSeek-OCR; spec: the checkpoint's `modeling_deepseekocr.py` and `deepencoder.py`.

pub mod config;
pub mod inputs_processor;
mod preprocess;
pub mod vision;

use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use inference_quant::ShardedVarBuilder;
use inference_tensor::{Device, Result, Tensor};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    decoder::CausalLm,
    kv_cache::EitherCache,
    media_inputs::media::place_token_features,
    model::{IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata, NormalModel},
    paged_attention::encoder_cache::{CacheModality, EncoderCacheManager},
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    utils::unvarbuilder::UnVarBuilder,
};

pub use config::Config;
use vision::DeepEncoder;

// a page's few hundred tokens of features each
const ENCODER_CACHE_ENTRIES: usize = 32;

/// The images whose tokens this pass fills, in prompt order; their global views are the pass's `pixel_values`.
#[derive(Default)]
pub struct DeepseekOcrSpecificArgs {
    pub tiles: Vec<Option<Tensor>>,
    /// Each image's tile grid, `(columns, rows)`.
    pub grids: Vec<(usize, usize)>,
    pub image_hashes: Vec<u64>,
}

pub struct DeepseekOcrModel {
    encoder: DeepEncoder,
    text: CausalLm,
    image_token_id: u32,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
    encoder_cache_hits: Arc<AtomicUsize>,
    encoder_cache_misses: Arc<AtomicUsize>,
}

impl DeepseekOcrModel {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        // the tower stays off ISQ's staging device
        let real_device = normal_loading_metadata.real_device.clone();
        let encoder = DeepEncoder::load(vb.pp("model").set_device(real_device), cfg)?;
        let text = CausalLm::new(
            &cfg.language_config.decoder_spec()?,
            vb,
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        Ok(Self {
            encoder,
            text,
            image_token_id: cfg.image_token_id,
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(ENCODER_CACHE_ENTRIES))),
            encoder_cache_hits: Arc::new(AtomicUsize::new(0)),
            encoder_cache_misses: Arc::new(AtomicUsize::new(0)),
        })
    }

    fn image_features(&self, globals: &Tensor, args: &DeepseekOcrSpecificArgs) -> Result<Tensor> {
        if globals.dim(0)? != args.grids.len() || args.tiles.len() != args.grids.len() {
            inference_tensor::bail!(
                "DeepSeek-OCR received {} global views for {} tile grids",
                globals.dim(0)?,
                args.grids.len()
            );
        }
        let mut features = Vec::with_capacity(args.grids.len());
        for (i, (&grid, tiles)) in args.grids.iter().zip(&args.tiles).enumerate() {
            let key = args.image_hashes.get(i).copied();
            let hit = key.and_then(|key| {
                let mut cache = self.encoder_cache.lock().expect("encoder cache poisoned");
                cache
                    .get(CacheModality::Image, key)
                    .map(|out| out[0].clone())
            });
            if let Some(hit) = hit {
                self.encoder_cache_hits.fetch_add(1, Ordering::Relaxed);
                features.push(hit);
                continue;
            }
            self.encoder_cache_misses.fetch_add(1, Ordering::Relaxed);
            let image =
                self.encoder
                    .image_features(&globals.narrow(0, i, 1)?, tiles.as_ref(), grid)?;
            if let Some(key) = key {
                self.encoder_cache
                    .lock()
                    .expect("encoder cache poisoned")
                    .insert(CacheModality::Image, key, vec![image.clone()]);
            }
            features.push(image);
        }
        Tensor::cat(&features, 0)
    }
}

impl IsqModel for DeepseekOcrModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.extend(self.text.residual_tensors());
        uvb.pp("model").extend(self.encoder.residual_tensors());
        uvb.to_safetensors()
    }
}

impl crate::speculative::SpeculativeTargetMixin for DeepseekOcrModel {}
impl crate::model::BlockDiffusionMixin for DeepseekOcrModel {}
impl AnyMoeBaseModelMixin for DeepseekOcrModel {}

impl MultimodalModel for DeepseekOcrModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let args = *model_specific_args
            .downcast::<DeepseekOcrSpecificArgs>()
            .expect("Cannot downcast into `DeepseekOcrSpecificArgs`");
        let mut embeds = self.text.get_input_embeddings(input_ids)?;
        if let Some(globals) = pixel_values {
            let features = self.image_features(&globals, &args)?;
            embeds = place_token_features(
                &embeds,
                input_ids,
                self.image_token_id,
                &features,
                "DeepSeek-OCR",
            )?;
        }
        self.text.forward_embeds(input_ids, embeds, ctx)
    }

    fn device(&self) -> &Device {
        NormalModel::device(&self.text)
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
    fn encoder_cache_counters(&self) -> Option<(Arc<AtomicUsize>, Arc<AtomicUsize>)> {
        Some((
            self.encoder_cache_hits.clone(),
            self.encoder_cache_misses.clone(),
        ))
    }
    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs(&self) -> bool {
        NormalModel::supports_cuda_decode_graphs(&self.text)
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn Any> {
        Box::new(DeepseekOcrSpecificArgs::default())
    }
    fn supports_packed_prefill(&self) -> bool {
        false
    }
    fn supports_mixed_media_batches(&self) -> bool {
        true
    }
}
