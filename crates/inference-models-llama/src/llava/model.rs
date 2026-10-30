#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_arguments
)]
//! LLaVA 1.5 and LLaVA-Next: a CLIP tower and projector in front of a decoder. Next also tiles each image over an
//! anyres grid, encoding a base view and its tiles and closing each row of tile features with `image_newline`.

use std::any::Any;
use std::sync::{Arc, Mutex};

use crate::amoe::{AnyMoeBaseModelMixin, AnyMoeConfig, AnyMoeExpertType, MlpLayer};
use crate::decoder::CausalLm;
use crate::layers;
use crate::llava::anyres::get_anyres_image_grid_shape;
use crate::llava::config::Config;
use crate::model::{
    IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata, NormalModel,
};
use crate::paged_attention::block_hash::MultimodalKind;
use crate::paged_attention::encoder_cache::{CacheModality, EncoderCacheManager};
use crate::paged_attention::{AttentionImplementation, ModelConfigMetadata};
use crate::utils::unvarbuilder::UnVarBuilder;
use crate::vision::clip::{ClipConfig, ClipVisionTransformer};
use crate::vision::multimodal_layout::{
    MultimodalEncoderKey, MultimodalEncoderOutputs, PackedMultimodalLayout,
};
use inference_quant::{NonZeroOp, ShardedVarBuilder};
use inference_tensor::nn::{Activation, Linear};
use inference_tensor::{DType, Device, IndexOp, Result, Tensor, bail};

const ENCODER_CACHE_ITEMS: usize = 32;

/// Per-image metadata for a prefill with images, each list in image order.
#[derive(Default)]
pub struct LLaVAVisionSpecificArgs {
    pub image_sizes: Option<Vec<(usize, usize)>>, // width, height
    pub num_image_tokens: Option<Vec<usize>>,
    // encoder samples per image: 1 for LLaVA 1.5, the base view and its tiles for Next
    pub num_image_samples: Option<Vec<usize>>,
    pub image_hashes: Vec<u64>,
    pub packed_layout: Option<PackedMultimodalLayout>,
}

// each image's features over the placeholder its negative marker starts, in marker order across batch rows
fn splice_llava_image_embeddings(
    mut result: Tensor,
    input_ids: &Tensor,
    image_features: &[Tensor],
    num_image_tokens: &[usize],
) -> Result<Tensor> {
    let image_indexes = input_ids.lt(0i64)?.nonzero()?.to_vec2::<u32>()?;
    let (batch_size, seq_len) = input_ids.dims2()?;
    if image_indexes.len() != image_features.len() || num_image_tokens.len() != image_features.len()
    {
        bail!(
            "LLaVA input has {} image markers but {} encoder outputs",
            image_indexes.len(),
            image_features.len()
        );
    }
    for ((coordinates, features), &tokens) in image_indexes
        .iter()
        .zip(image_features)
        .zip(num_image_tokens)
    {
        if features.dims3()? != (1, tokens, result.dim(2)?) {
            bail!(
                "LLaVA encoder returned shape {:?} for a {tokens}-token placeholder",
                features.dims()
            );
        }
        let batch_index = coordinates[0] as usize;
        let token_index = coordinates[1] as usize;
        if batch_index >= batch_size || token_index + tokens > seq_len {
            bail!("LLaVA image placeholder is outside the input batch");
        }
        result = result.slice_assign(
            &[
                batch_index..batch_index + 1,
                token_index..token_index + tokens,
                0..result.dim(2)?,
            ],
            features,
        )?;
    }
    Ok(result)
}

pub struct MMProjector {
    linear_1: Linear,
    activation: Activation,
    linear_2: Linear,
}

impl MMProjector {
    pub fn new(vb: &ShardedVarBuilder, config: &Config, device: &Device) -> Result<Self> {
        let linear_1 = layers::linear(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            vb.pp("multi_modal_projector.linear_1")
                .set_device(device.clone()),
        )?;
        let activation = match config.projector_hidden_act.as_str() {
            "gelu" => Activation::Gelu,
            act => bail!("Unsupported projector hidden act: {act}"),
        };
        let linear_2 = layers::linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            vb.pp("multi_modal_projector.linear_2")
                .set_device(device.clone()),
        )?;
        Ok(Self {
            linear_1,
            activation,
            linear_2,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.linear_1)?
            .apply(&self.activation)?
            .apply(&self.linear_2)
    }
}

pub struct ClipVisionTower {
    model: ClipVisionTransformer,
    select_layer: isize,
    select_feature_method: String,
    config: ClipConfig,
}

impl ClipVisionTower {
    pub fn new(
        vb: ShardedVarBuilder,
        select_layer: isize,
        select_feature_method: &str,
        config: &ClipConfig,
    ) -> Result<Self> {
        let model = ClipVisionTransformer::new(vb, config)?;
        Ok(Self {
            model,
            select_layer,
            select_feature_method: select_feature_method.to_string(),
            config: config.clone(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let result = self.model.forward_get_hidden_states(x)?;
        let index = result.len() as isize + self.select_layer;
        let result = result[index as usize].clone();
        if self.select_feature_method == "cls_patch" || self.select_feature_method == "full" {
            Ok(result)
        } else {
            result.i((.., 1..))
        }
    }

    pub fn num_patches_per_side(&self) -> usize {
        self.config.image_size / self.config.patch_size
    }
}

pub struct Model {
    clip_vision_tower: ClipVisionTower,
    // LLaVA-Next only: closes each row of tile features
    image_newline: Option<Tensor>,
    mm_projector: MMProjector,
    llm: CausalLm,
    config: Config,
    device: Device,
    dtype: DType,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
}

impl Model {
    /// `anyres` loads LLaVA-Next, which tiles images over `image_grid_pinpoints`.
    pub fn new(
        config: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        anyres: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let device = normal_loading_metadata.real_device.clone();
        let dtype = vb.dtype();
        let clip_config = config.to_clip_config();
        let mm_projector = MMProjector::new(&vb, config, &device)?;
        let clip_vision_tower = ClipVisionTower::new(
            vb.pp("vision_tower.vision_model")
                .set_device(device.clone()),
            config.vision_feature_layer,
            &config.vision_feature_select_strategy,
            &clip_config,
        )?;
        let image_newline = anyres
            .then(|| {
                vb.get(&[config.text_config.hidden_size], "image_newline")?
                    .to_device(&device)
            })
            .transpose()?;

        let llm = CausalLm::new(
            &config.decoder_spec()?,
            vb.pp("language_model"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        Ok(Self {
            clip_vision_tower,
            image_newline,
            mm_projector,
            llm,
            config: config.clone(),
            device,
            dtype,
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(ENCODER_CACHE_ITEMS))),
        })
    }

    pub fn encode_images(&self, x: &Tensor) -> Result<Tensor> {
        self.mm_projector
            .forward(&self.clip_vision_tower.forward(x)?)
    }

    fn unpad_image(tensor: &Tensor, original_size: (u32, u32)) -> Result<Tensor> {
        let (original_width, original_height) = original_size;
        let (_, current_height, current_width) = tensor.dims3()?;
        let original_aspect_ratio = (original_width as f32) / (original_height as f32);
        let current_aspect_ratio = (current_width as f32) / (current_height as f32);
        if original_aspect_ratio > current_aspect_ratio {
            let scale_factor = (current_width as f32) / (original_width as f32);
            let new_height = (original_height as f32 * scale_factor).floor() as usize;
            let padding = (current_height - new_height) / 2;
            tensor.i((.., padding..current_height - padding, ..))
        } else {
            let scale_factor = (current_height as f32) / (original_height as f32);
            let new_width = (original_width as f32 * scale_factor).floor() as usize;
            let padding = (current_width - new_width) / 2;
            tensor.i((.., .., padding..current_width - padding))
        }
    }

    // each image's encoded samples, from the encoder cache where its hash is there
    fn encode_per_image(
        &self,
        images: &Tensor,
        num_image_samples: &[usize],
        image_hashes: &[u64],
    ) -> Result<Vec<Tensor>> {
        let n_images = num_image_samples.len();
        let mut sample_offsets = vec![0usize; n_images + 1];
        for (i, &ns) in num_image_samples.iter().enumerate() {
            sample_offsets[i + 1] = sample_offsets[i] + ns;
        }
        if image_hashes.len() != n_images {
            let image_features = self.encode_images(images)?;
            if image_features.dim(0)? != sample_offsets[n_images] {
                bail!(
                    "LLaVA encoder returned {} samples for {} inputs",
                    image_features.dim(0)?,
                    sample_offsets[n_images]
                );
            }
            return (0..n_images)
                .map(|i| image_features.i(sample_offsets[i]..sample_offsets[i + 1]))
                .collect();
        }

        let mut per_image: Vec<Option<Tensor>> = vec![None; n_images];
        let mut miss_indices = Vec::new();
        {
            let mut guard = self
                .encoder_cache
                .lock()
                .expect("encoder cache lock poisoned");
            for (i, &hash) in image_hashes.iter().enumerate() {
                match guard.get(CacheModality::Image, hash) {
                    Some(cached) => {
                        let cached = cached.first().ok_or_else(|| {
                            inference_tensor::Error::Msg(
                                "cached LLaVA image has no encoder output".into(),
                            )
                        })?;
                        if cached.dim(0)? != num_image_samples[i] {
                            bail!(
                                "cached LLaVA image has {} samples but metadata describes {}",
                                cached.dim(0)?,
                                num_image_samples[i]
                            );
                        }
                        per_image[i] = Some(cached.clone());
                    }
                    None => miss_indices.push(i),
                }
            }
        }

        if !miss_indices.is_empty() {
            let mut miss_samples = Vec::new();
            for &idx in &miss_indices {
                for sample in sample_offsets[idx]..sample_offsets[idx + 1] {
                    miss_samples.push(images.get(sample)?);
                }
            }
            let miss_encoded = self.encode_images(&Tensor::stack(&miss_samples, 0)?)?;
            if miss_encoded.dim(0)? != miss_samples.len() {
                bail!(
                    "LLaVA encoder returned {} samples for {} inputs",
                    miss_encoded.dim(0)?,
                    miss_samples.len()
                );
            }
            let mut offset = 0;
            let mut guard = self
                .encoder_cache
                .lock()
                .expect("encoder cache lock poisoned");
            for &idx in &miss_indices {
                let ns = num_image_samples[idx];
                let encoded = miss_encoded.i(offset..offset + ns)?;
                guard.insert(
                    CacheModality::Image,
                    image_hashes[idx],
                    vec![encoded.clone()],
                );
                per_image[idx] = Some(encoded);
                offset += ns;
            }
        }

        per_image
            .into_iter()
            .enumerate()
            .map(|(index, output)| {
                output.ok_or_else(|| {
                    inference_tensor::Error::Msg(format!(
                        "LLaVA image {index} has no encoder output"
                    ))
                })
            })
            .collect()
    }

    // `(1, tokens, hidden)`: the base view, then for Next the unpadded tile grid with a newline after each row
    fn image_tokens(
        &self,
        image_feature: &Tensor,
        image_size: Option<(u32, u32)>,
    ) -> Result<Tensor> {
        let base_image_feature = image_feature.get(0)?;
        let Some(image_newline) = &self.image_newline else {
            return base_image_feature.unsqueeze(0);
        };
        let image_size = image_size
            .ok_or_else(|| inference_tensor::Error::Msg("LLaVA-Next image has no size".into()))?;
        let patch_image_feature = image_feature.i(1..)?;
        let height = self.clip_vision_tower.num_patches_per_side();
        let width = height;
        if height * width != base_image_feature.dim(0)? {
            bail!(
                "LLaVA-Next base image has {} patches but the vision tower expects {}",
                base_image_feature.dim(0)?,
                height * width
            );
        }
        let image_grid_pinpoints = self.config.image_grid_pinpoints.as_ref().ok_or_else(|| {
            inference_tensor::Error::Msg("LLaVA-Next needs image_grid_pinpoints".into())
        })?;
        let (num_patch_width, num_patch_height) = get_anyres_image_grid_shape(
            image_size,
            image_grid_pinpoints,
            self.clip_vision_tower.config.image_size as u32,
        );
        let new_image_feature = patch_image_feature
            .reshape((
                num_patch_height as usize,
                num_patch_width as usize,
                height,
                width,
                (),
            ))?
            .permute((4, 0, 2, 1, 3))?
            .flatten(1, 2)?
            .flatten(2, 3)?;
        let new_image_feature = Self::unpad_image(&new_image_feature, image_size)?;
        let (rows, cols) = (new_image_feature.dim(0)?, new_image_feature.dim(1)?);
        let image_new_line = image_newline
            .reshape((self.config.text_config.hidden_size, 1, 1))?
            .broadcast_as((rows, cols, 1))?;
        let new_image_feature = Tensor::cat(&[new_image_feature, image_new_line], 2)?
            .flatten(1, 2)?
            .transpose(0, 1)?;
        Tensor::cat(&[base_image_feature, new_image_feature], 0)?.unsqueeze(0)
    }

    pub fn prepare_inputs_labels_for_multimodal(
        &self,
        input_ids: &Tensor,
        images: &Tensor, // every image's samples, in image order
        num_image_tokens: &[usize],
        num_image_samples: &[usize],
        image_sizes: Option<&[(u32, u32)]>,
        image_hashes: &[u64],
        packed_layout: Option<&PackedMultimodalLayout>,
    ) -> Result<Tensor> {
        let mut result = input_ids.clamp(0i64, i64::MAX)?.to_dtype(DType::U32)?;
        result = self.llm.get_input_embeddings(&result)?;

        let images_typed = images.to_dtype(self.dtype)?;
        let n_images = num_image_samples.len();
        if n_images == 0 {
            bail!("LLaVA received image pixels without image metadata");
        }
        if num_image_tokens.len() != n_images
            || image_sizes.is_some_and(|sizes| sizes.len() != n_images)
        {
            bail!("LLaVA image metadata has inconsistent cardinality");
        }
        if !image_hashes.is_empty() && image_hashes.len() != n_images {
            bail!(
                "LLaVA received {} image hashes for {n_images} images",
                image_hashes.len()
            );
        }
        if num_image_samples.contains(&0) {
            bail!("LLaVA image has no preprocessed samples");
        }
        let total_samples = num_image_samples.iter().sum::<usize>();
        if images_typed.dim(0)? != total_samples {
            bail!(
                "LLaVA received {} image samples but metadata describes {total_samples}",
                images_typed.dim(0)?
            );
        }

        let image_features = self
            .encode_per_image(&images_typed, num_image_samples, image_hashes)?
            .iter()
            .enumerate()
            .map(|(i, feature)| self.image_tokens(feature, image_sizes.map(|sizes| sizes[i])))
            .collect::<Result<Vec<_>>>()?;
        if let Some(layout) = packed_layout {
            if image_hashes.len() != image_features.len() {
                bail!(
                    "packed LLaVA input has {} image hashes but {} encoder outputs",
                    image_hashes.len(),
                    image_features.len()
                );
            }
            let encoder_outputs = image_hashes
                .iter()
                .copied()
                .zip(image_features)
                .map(|(hash, output)| {
                    (
                        MultimodalEncoderKey {
                            kind: MultimodalKind::Image,
                            hash,
                        },
                        vec![output],
                    )
                })
                .collect::<MultimodalEncoderOutputs>();
            return layout.splice_embeddings(&result, &encoder_outputs);
        }
        result =
            splice_llava_image_embeddings(result, input_ids, &image_features, num_image_tokens)?;
        let (_, seq_len) = input_ids.dims2()?;
        if seq_len > self.config.text_config.max_length {
            result = result.i((.., ..self.config.text_config.max_length, ..))?
        }
        Ok(result)
    }
}

impl IsqModel for Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("language_model").extend(self.llm.residual_tensors());
        uvb.pp("multi_modal_projector.linear_1")
            .add(&self.mm_projector.linear_1);
        uvb.pp("multi_modal_projector.linear_2")
            .add(&self.mm_projector.linear_2);
        uvb.pp("vision_tower.vision_model")
            .extend(self.clip_vision_tower.model.residual_tensors());
        if let Some(image_newline) = &self.image_newline {
            uvb.add_tensor("image_newline", image_newline.clone());
        }
        uvb.to_safetensors()
    }
}

impl crate::speculative::SpeculativeTargetMixin for Model {}

impl crate::model::BlockDiffusionMixin for Model {}

impl MultimodalModel for Model {
    fn supports_packed_prefill(&self) -> bool {
        true
    }

    fn supports_mixed_media_batches(&self) -> bool {
        true
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let LLaVAVisionSpecificArgs {
            image_sizes,
            num_image_tokens,
            num_image_samples,
            image_hashes,
            packed_layout,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `LLaVAVisionSpecificArgs`");
        let Some(pixel_values) = pixel_values else {
            return self.llm.forward(input_ids, ctx);
        };
        let missing =
            |what: &str| inference_tensor::Error::Msg(format!("LLaVA input is missing {what}"));
        let num_image_tokens = num_image_tokens.ok_or_else(|| missing("image token counts"))?;
        let num_image_samples = num_image_samples.ok_or_else(|| missing("image sample counts"))?;
        let image_sizes = image_sizes.map(|sizes| {
            sizes
                .iter()
                .map(|&(w, h)| (w as u32, h as u32))
                .collect::<Vec<_>>()
        });
        let input_embeds = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            &pixel_values,
            &num_image_tokens,
            &num_image_samples,
            image_sizes.as_deref(),
            &image_hashes,
            packed_layout.as_ref(),
        )?;
        self.llm.forward_embeds(input_ids, input_embeds, ctx)
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn cache(&self) -> &crate::kv_cache::EitherCache {
        self.llm.cache()
    }
    fn max_seq_len(&self) -> usize {
        self.config.text_config.max_length
    }

    fn config(&self) -> &ModelConfigMetadata {
        self.llm.config()
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn Any> {
        Box::new(LLaVAVisionSpecificArgs::default())
    }
    fn encoder_cache(&self) -> Option<&Mutex<EncoderCacheManager>> {
        Some(&self.encoder_cache)
    }
    fn encoder_cache_counters(
        &self,
    ) -> Option<(
        Arc<std::sync::atomic::AtomicUsize>,
        Arc<std::sync::atomic::AtomicUsize>,
    )> {
        Some(
            self.encoder_cache
                .lock()
                .expect("encoder cache poisoned")
                .counters(),
        )
    }
}

impl AnyMoeBaseModelMixin for Model {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        self.llm.get_mlps()
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        self.llm.get_mlps_mut()
    }
    fn create_anymoe_layers(
        &mut self,
        additional_vbs: Vec<ShardedVarBuilder>,
        config: AnyMoeConfig,
        (prefix, mlp): (String, String),
        layers: Vec<usize>,
        expert_type: AnyMoeExpertType,
        gate_vb: Option<ShardedVarBuilder>,
    ) -> Result<()> {
        self.llm.create_anymoe_layers(
            additional_vbs,
            config,
            (prefix, mlp),
            layers,
            expert_type,
            gate_vb,
        )
    }
    fn amoe_supported(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_embeddings_follow_markers_across_batch_rows() {
        let device = Device::Cpu;
        let input_ids = Tensor::new(&[[1i64, -1, 0, 2], [3, 4, -1, 0]], &device).unwrap();
        let result = Tensor::zeros((2, 4, 2), DType::F32, &device).unwrap();
        let image_features = [
            Tensor::new(&[[[11f32, 12.0], [13.0, 14.0]]], &device).unwrap(),
            Tensor::new(&[[[21f32, 22.0], [23.0, 24.0]]], &device).unwrap(),
        ];
        let result =
            splice_llava_image_embeddings(result, &input_ids, &image_features, &[2, 2]).unwrap();
        assert_eq!(
            result.to_vec3::<f32>().unwrap(),
            vec![
                vec![
                    vec![0.0, 0.0],
                    vec![11.0, 12.0],
                    vec![13.0, 14.0],
                    vec![0.0, 0.0],
                ],
                vec![
                    vec![0.0, 0.0],
                    vec![0.0, 0.0],
                    vec![21.0, 22.0],
                    vec![23.0, 24.0],
                ],
            ]
        );
    }
}
