#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub mod inputs_processor;

use either::Either;
use inference_quant::{BitWiseOp, NonZeroOp, QuantMethod, QuantizedConfig, ShardedVarBuilder};
use inference_tensor::{
    D, DType, Device, IndexOp, Module, Result, Shape, Tensor, shape::ShapeWithOneHole,
};
use std::{
    any::Any,
    fmt::Debug,
    sync::{Arc, Mutex},
};

use crate::{
    amoe::{AnyMoeBaseModelMixin, AnyMoeLoraTarget, MlpLayer},
    decoder::{CausalLm, DecoderSpec},
    kv_cache::EitherCache,
    layers::{Activation, PhiRopeScalingConfig},
    model::{IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata, NormalModel},
    paged_attention::{
        AttentionImplementation, ModelConfigMetadata,
        encoder_cache::{CacheModality, EncoderCacheManager},
    },
    serde_default_fn,
    utils::unvarbuilder::UnVarBuilder,
    vision::clip::{ClipConfig, ClipVisionTransformer},
    vision::multimodal_layout::{
        MultimodalEncoderKey, MultimodalEncoderOutputs, PackedMultimodalLayout,
    },
};

use crate::vision::clip;

const ENCODER_CACHE_ENTRIES: usize = 32;

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct EmbedLayerConfig {
    pub hd_transform_order: Option<String>,
    pub projection_cls: Option<String>,
    pub use_hd_transform: Option<bool>,
    pub with_learnable_separator: Option<bool>,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct ImageProcessorConfig {
    pub image_dim_out: usize,
    pub name: String,
    pub num_img_tokens: usize,
    pub layer_idx: Option<isize>,
    pub type_feature: Option<String>,
}

serde_default_fn!(bool, word_emb_default, false);

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub rope_scaling: Option<PhiRopeScalingConfig>,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub original_max_position_embeddings: usize,
    pub embd_layer: EmbedLayerConfig,
    pub img_processor: ImageProcessorConfig,
    #[serde(alias = "quantization")]
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// The Phi-3 text model config these layers are built from.
    pub fn text_config(&self) -> crate::phi3::Config {
        crate::phi3::Config {
            vocab_size: self.vocab_size,
            hidden_act: self.hidden_act,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            rope_scaling: self.rope_scaling.clone(),
            rope_scaling_attn_factor: None,
            max_position_embeddings: self.max_position_embeddings,
            sliding_window: self.sliding_window,
            original_max_position_embeddings: self.original_max_position_embeddings,
            quantization_config: self.quantization_config.clone(),
            tie_word_embeddings: self.tie_word_embeddings,
            partial_rotary_factor: None,
        }
    }
}

trait ModuleWithMetadata: Module + Debug + Send + Sync {
    fn device(&self) -> Device;
    fn dtype(&self) -> DType;
}

#[derive(Debug)]
struct QuantMethodWrapper(Arc<dyn QuantMethod>);

impl Module for QuantMethodWrapper {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(xs)
    }
}

impl ModuleWithMetadata for QuantMethodWrapper {
    fn device(&self) -> Device {
        self.0.unquant_weight_bias().unwrap().0.device().clone()
    }
    fn dtype(&self) -> DType {
        self.0.unquant_weight_bias().unwrap().0.dtype()
    }
}

impl ModuleWithMetadata for inference_tensor::nn::Activation {
    fn device(&self) -> Device {
        unreachable!()
    }
    fn dtype(&self) -> DType {
        unreachable!()
    }
}

#[derive(Debug)]
struct BigShapeWithOneHole((usize, usize, usize, usize, usize, ()));

fn hole_size(el_count: usize, prod_d: usize, s: &dyn std::fmt::Debug) -> Result<usize> {
    if prod_d == 0 {
        inference_tensor::bail!("cannot reshape tensor of {el_count} elements to {s:?}")
    }
    if !el_count.is_multiple_of(prod_d) {
        inference_tensor::bail!("cannot reshape tensor with {el_count} elements to {s:?}")
    }
    Ok(el_count / prod_d)
}

impl ShapeWithOneHole for BigShapeWithOneHole {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, d2, d3, d4, d5, ()) = self.0;
        let d = hole_size(el_count, d1 * d2 * d3 * d4 * d5, &self)?;
        Ok((d1, d2, d3, d4, d5, d).into())
    }
}

const MAX_INPUT_ID: f64 = 1e9;

#[derive(Debug)]
struct EmbeddingLayers(Vec<Box<dyn ModuleWithMetadata>>);

impl Module for EmbeddingLayers {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in &self.0 {
            xs = layer.forward(&xs)?;
        }
        Ok(xs)
    }
}

#[derive(Debug)]
pub struct ImageEmbedding {
    wte: Arc<dyn QuantMethod>,
    dtype: DType,
    image_dim_out: usize,
    num_img_tokens: usize,
    glb_gn: Option<Tensor>,
    sub_gn: Option<Tensor>,
    layers: EmbeddingLayers,
    type_feature: String,
    layer_idx: isize,
    image_processor: ClipVisionTransformer,
    hd_transform_order: String,
    use_hd_transform: bool,
    vocab_size: usize,
    tensors: Vec<(String, Tensor)>,
}

pub const PHI3V_CLIP_CONFIG: ClipConfig = ClipConfig {
    hidden_act: clip::Activation::QuickGelu,
    hidden_size: 1024,
    image_size: 336,
    intermediate_size: 4096,
    num_attention_heads: 16,
    num_channels: 3,
    num_hidden_layers: 24,
    patch_size: 14,
};

impl ImageEmbedding {
    fn new(
        config: &Config,
        wte: Arc<dyn QuantMethod>,
        dtype: DType,
        embed_config: &EmbedLayerConfig,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        if config.img_processor.name != "clip_vision_model" {
            inference_tensor::bail!(
                "img_processor=`{}` nor supported.",
                config.img_processor.name
            );
        }
        let image_dim_out = config.img_processor.image_dim_out;
        let num_img_tokens = config.img_processor.num_img_tokens;

        // CLIP image processor here...
        let image_processor =
            ClipVisionTransformer::new(vb.pp("img_processor.vision_model"), &PHI3V_CLIP_CONFIG)?;

        // High dim transform
        let use_hd_transform = embed_config.use_hd_transform.unwrap_or(false);
        let with_learnable_separator = embed_config.with_learnable_separator.unwrap_or(false);
        let hd_transform_order = embed_config
            .hd_transform_order
            .clone()
            .unwrap_or("glb_sub".to_string());
        assert_eq!(use_hd_transform, with_learnable_separator);
        let (glb_gn, sub_gn) = if with_learnable_separator {
            let glb_gn = vb.get((1, 1, image_dim_out * 4), "glb_GN")?;
            let sub_gn = vb.get((1, 1, 1, image_dim_out * 4), "sub_GN")?;
            (Some(glb_gn), Some(sub_gn))
        } else {
            (None, None)
        };

        // Inner projection
        let projection_cls = embed_config
            .projection_cls
            .clone()
            .unwrap_or("linear".to_string());

        let mut tensors = Vec::new();
        let layers: Vec<Box<dyn ModuleWithMetadata>> =
            match (projection_cls.as_str(), use_hd_transform) {
                ("linear", _) => {
                    let a = inference_quant::linear_b(
                        image_dim_out,
                        hidden_size,
                        true,
                        &None,
                        vb.pp("img_projection"),
                    )?;
                    let (a_w, a_b) = a.unquant_weight_bias().unwrap();
                    tensors.push(("img_projection.weight".to_string(), a_w));
                    if let Some(b) = a_b {
                        tensors.push(("img_projection.bias".to_string(), b));
                    }
                    vec![Box::new(QuantMethodWrapper(a))]
                }
                ("mlp", true) => {
                    let dim_proj = hidden_size;
                    let a = inference_quant::linear_b(
                        image_dim_out * 4,
                        dim_proj,
                        true,
                        &None,
                        vb.pp("img_projection.0"),
                    )?;
                    let (a_w, a_b) = a.unquant_weight_bias().unwrap();
                    tensors.push(("img_projection.0.weight".to_string(), a_w));
                    if let Some(b) = a_b {
                        tensors.push(("img_projection.0.bias".to_string(), b));
                    }
                    let b = inference_quant::linear_b(
                        dim_proj,
                        dim_proj,
                        true,
                        &None,
                        vb.pp("img_projection.2"),
                    )?;
                    let (b_w, b_b) = b.unquant_weight_bias().unwrap();
                    tensors.push(("img_projection.2.weight".to_string(), b_w));
                    if let Some(b) = b_b {
                        tensors.push(("img_projection.2.bias".to_string(), b));
                    }
                    vec![
                        Box::new(QuantMethodWrapper(a)),
                        Box::new(inference_tensor::nn::Activation::Gelu),
                        Box::new(QuantMethodWrapper(b)),
                    ]
                }
                ("mlp", false) => {
                    let dim_proj = hidden_size;
                    let a = inference_quant::linear_b(
                        image_dim_out,
                        dim_proj,
                        true,
                        &None,
                        vb.pp("img_projection.0"),
                    )?;
                    let (a_w, a_b) = a.unquant_weight_bias().unwrap();
                    tensors.push(("img_projection.0.weight".to_string(), a_w));
                    if let Some(b) = a_b {
                        tensors.push(("img_projection.0.bias".to_string(), b));
                    }
                    let b = inference_quant::linear_b(
                        dim_proj,
                        dim_proj,
                        true,
                        &None,
                        vb.pp("img_projection.2"),
                    )?;
                    let (b_w, b_b) = b.unquant_weight_bias().unwrap();
                    tensors.push(("img_projection.2.weight".to_string(), b_w));
                    if let Some(b) = b_b {
                        tensors.push(("img_projection.2.bias".to_string(), b));
                    }
                    vec![
                        Box::new(QuantMethodWrapper(a)),
                        Box::new(inference_tensor::nn::Activation::Gelu),
                        Box::new(QuantMethodWrapper(b)),
                    ]
                }
                _ => {
                    inference_tensor::bail!("projection_cls=`{projection_cls}` not implemented.");
                }
            };

        let layer_idx = config.img_processor.layer_idx.unwrap_or(-2);
        let type_feature = config
            .img_processor
            .type_feature
            .clone()
            .unwrap_or("patch".to_string());

        Ok(Self {
            wte,
            dtype,
            image_dim_out,
            num_img_tokens,
            glb_gn,
            sub_gn,
            layer_idx,
            type_feature,
            image_processor,
            layers: EmbeddingLayers(layers),
            hd_transform_order,
            use_hd_transform,
            vocab_size: config.vocab_size,
            tensors,
        })
    }

    fn get_image_features(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let hidden_states = self
            .image_processor
            .forward_get_hidden_states(&pixel_values.to_dtype(self.dtype)?)?;
        let img_feature =
            hidden_states[(hidden_states.len() as isize + self.layer_idx) as usize].clone();
        if self.type_feature == "patch" {
            img_feature.i((.., 1..))
        } else if self.type_feature == "cls_patch" {
            Ok(img_feature)
        } else {
            inference_tensor::bail!("Unsupported image feature type {}", self.type_feature)
        }
    }

    #[allow(non_snake_case)]
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        image_sizes: Option<Vec<(usize, usize)>>,
        image_hashes: &[u64],
        packed_layout: Option<&PackedMultimodalLayout>,
        encoder_cache: &Mutex<EncoderCacheManager>,
    ) -> Result<Tensor> {
        let input_ids = input_ids.reshape(((), input_ids.dim(D::Minus1)?))?;

        let input_ids_lt = input_ids.lt(0.0f64)?;
        let input_ids_gt = input_ids.gt(-MAX_INPUT_ID)?;
        // positions = torch.nonzero((input_ids < 0) & (input_ids > -MAX_INPUT_ID), as_tuple=False)
        let positions = input_ids_lt.bitwise_and(&input_ids_gt)?.nonzero()?;
        let target_dev = self.layers.0[0].device();
        let target_dtype = self.layers.0[0].dtype();

        let hd_transform;
        let image_set_tensor;
        let n_hashes = image_hashes.len();
        if positions.dim(0)? > 0 {
            // input_ids[positions[:, 0], positions[:, 1]]
            if self.use_hd_transform {
                let image_sizes_ref = image_sizes.as_ref().ok_or_else(|| {
                    inference_tensor::Error::Msg("Phi3 HD input is missing image sizes".into())
                })?;
                if pixel_values.dims().len() != 5 {
                    inference_tensor::bail!(
                        "Phi3 HD input must have rank 5, got rank {}",
                        pixel_values.dims().len()
                    );
                }
                let bs = pixel_values.dim(0)?;
                if bs == 0 {
                    inference_tensor::bail!("Phi3 received an empty image batch");
                }
                if image_sizes_ref.len() != bs {
                    inference_tensor::bail!(
                        "Phi3 received {} image sizes for {bs} images",
                        image_sizes_ref.len()
                    );
                }
                if n_hashes != 0 && n_hashes != bs {
                    inference_tensor::bail!(
                        "Phi3 received {n_hashes} image hashes for {bs} images"
                    );
                }

                // Check cache for each image
                let mut per_image_cached: Vec<Option<Tensor>> = vec![None; bs];
                let mut miss_indices = Vec::new();
                if n_hashes == bs {
                    let mut guard = encoder_cache.lock().expect("encoder cache lock poisoned");
                    for (i, &hash) in image_hashes.iter().enumerate() {
                        match guard.get(CacheModality::Image, hash) {
                            Some(cached) => {
                                let cached = cached.first().ok_or_else(|| {
                                    inference_tensor::Error::Msg(
                                        "cached Phi3 image has no encoder output".into(),
                                    )
                                })?;
                                per_image_cached[i] = Some(cached.clone());
                            }
                            _ => {
                                miss_indices.push(i);
                            }
                        }
                    }
                } else {
                    miss_indices = (0..bs).collect();
                }

                // Only run CLIP on miss images
                let mut img_features_per_image: Vec<Option<Tensor>> = vec![None; bs];
                if !miss_indices.is_empty() {
                    // We need CLIP features for all miss images
                    let miss_pv: Vec<Tensor> = miss_indices
                        .iter()
                        .map(|&i| pixel_values.get(i))
                        .collect::<Result<Vec<_>>>()?;
                    let miss_pv = Tensor::stack(&miss_pv, 0)?;
                    let miss_features = self.get_image_features(&miss_pv.flatten(0, 1)?)?;
                    let patch_count = miss_features.dim(1)?;
                    let base_feat_dim = (patch_count as f32).sqrt() as usize;
                    if base_feat_dim != 24 || base_feat_dim * base_feat_dim != patch_count {
                        inference_tensor::bail!(
                            "Phi3 vision tower returned {patch_count} patches per crop"
                        );
                    }
                    let miss_bs = miss_indices.len();
                    let miss_features = miss_features.reshape((
                        miss_bs,
                        (),
                        base_feat_dim.pow(2),
                        self.image_dim_out,
                    ))?;
                    for (batch_idx, &orig_idx) in miss_indices.iter().enumerate() {
                        img_features_per_image[orig_idx] = Some(miss_features.get(batch_idx)?);
                    }
                }

                let C = self.image_dim_out;
                let H = 24usize; // base_feat_dim

                let mut image_set_tensor_inner = Vec::new();
                let mut output_len = Vec::new();
                for (bs_, &(h, w)) in image_sizes_ref.iter().enumerate() {
                    if h == 0 || w == 0 || h % 336 != 0 || w % 336 != 0 {
                        inference_tensor::bail!("Phi3 image size {h}x{w} is not a valid HD grid");
                    }
                    let h = h / 336;
                    let w = w / 336;
                    let B_ = h * w;
                    let temp_len = (B_ + 1) * 144 + 1 + (h + 1) * 12;

                    // Check if we have a cache hit for this image
                    if let Some(ref cached_tensor) = per_image_cached[bs_] {
                        let (output_batch, cnt, _) = cached_tensor.dims3()?;
                        if output_batch != 1 || cnt != temp_len {
                            inference_tensor::bail!(
                                "cached Phi3 image has shape {:?} but metadata requires one batch and {temp_len} rows",
                                cached_tensor.dims()
                            );
                        }
                        output_len.push(cnt);
                        image_set_tensor_inner.push(cached_tensor.clone());
                        continue;
                    }

                    let img_feats = img_features_per_image[bs_].as_ref().ok_or_else(|| {
                        inference_tensor::Error::Msg(format!(
                            "Phi3 image {bs_} has no vision features"
                        ))
                    })?;

                    // 1 x (24x24) x 1024
                    let global_img_feature = img_feats.i(..1)?;

                    // 1 x 12 x 12 x 4096
                    let glb_img = global_img_feature
                        .reshape((1, H, H, C))?
                        .reshape((1, H / 2, 2, H / 2, 2, C))?
                        .contiguous()?
                        .permute((0, 1, 3, 2, 4, 5))?
                        .reshape((1, H / 2, H / 2, 4 * C))?
                        .contiguous()?;
                    let temp_glbl_gn = self
                        .sub_gn
                        .as_ref()
                        .expect("Need `sub_gn` if `use_hd_transform`")
                        .repeat((1, H / 2, 1, 1))?;

                    // 1 x 156 x 4096
                    let glb_img =
                        Tensor::cat(&[glb_img, temp_glbl_gn], 2)?.reshape((1, (), 4 * C))?;

                    // (max_num_crops-1) x (12x12) x C
                    let sub_img = img_feats.i(1..)?;

                    // Get rid of padding sub_img
                    let sub_img = sub_img.i(..B_)?;

                    // (num_crops, 12, 2, 12, 2, 1024) -> (num_crops, 12, 12, 2, 2, 1024) -> (num_crops, 12*12, 4*1024)
                    let sub_img = sub_img
                        .reshape((B_, H, H, C))?
                        .reshape((B_, H / 2, 2, H / 2, 2, C))?
                        .contiguous()?
                        .permute((0, 1, 3, 2, 4, 5))?
                        .reshape((B_, (), 4 * C))?
                        .contiguous()?;
                    let sub_img = sub_img
                        .reshape(BigShapeWithOneHole((1usize, h, w, 12usize, 12usize, ())))?
                        .permute((0, 1, 3, 2, 4, 5))?
                        .reshape((1, h * 12, w * 12, 4 * C))?;
                    let temp_sub_gn = self
                        .sub_gn
                        .as_ref()
                        .expect("Need `sub_gn` if `use_hd_transform`")
                        .repeat((1, h * 12, 1, 1))?;

                    let sub_img =
                        Tensor::cat(&[sub_img, temp_sub_gn], 2)?.reshape((1, (), 4 * C))?;

                    // (1, num_img_tokens, 1024*4)

                    let img = match self.hd_transform_order.as_str() {
                        "glb_sub" => Tensor::cat(
                            &[
                                glb_img,
                                self.glb_gn
                                    .as_ref()
                                    .expect("Need `glb_gn` if `use_hd_transform`")
                                    .clone(),
                                sub_img,
                            ],
                            1,
                        )?,
                        "sub_glb" => Tensor::cat(
                            &[
                                sub_img,
                                self.glb_gn
                                    .as_ref()
                                    .expect("Need `glb_gn` if `use_hd_transform`")
                                    .clone(),
                                glb_img,
                            ],
                            1,
                        )?,
                        other => {
                            inference_tensor::bail!("Invalid hd_transform_order=`{other}`");
                        }
                    };

                    if temp_len != img.dim(1)? {
                        inference_tensor::bail!(
                            "Phi3 HD transform produced {} rows, expected {temp_len}",
                            img.dim(1)?
                        );
                    }
                    output_len.push(temp_len);

                    let layerout = self
                        .layers
                        .forward(&img.to_device(&target_dev)?.to_dtype(target_dtype)?)?;

                    // Cache the projected features for this image
                    if n_hashes == bs {
                        let mut guard = encoder_cache.lock().expect("encoder cache lock poisoned");
                        guard.insert(
                            CacheModality::Image,
                            image_hashes[bs_],
                            vec![layerout.clone()],
                        );
                    }

                    image_set_tensor_inner.push(layerout);
                }

                hd_transform = Some(output_len);
                image_set_tensor = Some(Either::Left(image_set_tensor_inner));
            } else if pixel_values.dims().len() == 4 {
                hd_transform = None;
                let n_imgs = pixel_values.dim(0)?;
                if n_imgs == 0 {
                    inference_tensor::bail!("Phi3 received an empty image batch");
                }
                if n_hashes != 0 && n_hashes != n_imgs {
                    inference_tensor::bail!(
                        "Phi3 received {n_hashes} image hashes for {n_imgs} images"
                    );
                }
                if n_hashes == n_imgs {
                    // Per-image caching for non-HD path
                    let mut per_image_features: Vec<Option<Tensor>> = vec![None; n_imgs];
                    let mut miss_indices = Vec::new();
                    {
                        let mut guard = encoder_cache.lock().expect("encoder cache lock poisoned");
                        for (i, &hash) in image_hashes.iter().enumerate() {
                            match guard.get(CacheModality::Image, hash) {
                                Some(cached) => {
                                    let cached = cached.first().ok_or_else(|| {
                                        inference_tensor::Error::Msg(
                                            "cached Phi3 image has no encoder output".into(),
                                        )
                                    })?;
                                    let (rows, _) = cached.dims2()?;
                                    if rows != self.num_img_tokens {
                                        inference_tensor::bail!(
                                            "cached Phi3 image has {} rows but metadata requires {}",
                                            rows,
                                            self.num_img_tokens
                                        );
                                    }
                                    per_image_features[i] = Some(cached.clone());
                                }
                                _ => {
                                    miss_indices.push(i);
                                }
                            }
                        }
                    }
                    if !miss_indices.is_empty() {
                        for &idx in &miss_indices {
                            let single_pv = pixel_values.get(idx)?.unsqueeze(0)?;
                            let tt = self
                                .get_image_features(&single_pv)?
                                .to_device(&target_dev)?
                                .to_dtype(target_dtype)?
                                .reshape(((), self.image_dim_out))?;
                            let feats = self.layers.forward(&tt)?;
                            {
                                let mut guard =
                                    encoder_cache.lock().expect("encoder cache lock poisoned");
                                guard.insert(
                                    CacheModality::Image,
                                    image_hashes[idx],
                                    vec![feats.clone()],
                                );
                            }
                            per_image_features[idx] = Some(feats);
                        }
                    }
                    let all_feats: Vec<Tensor> = per_image_features
                        .into_iter()
                        .enumerate()
                        .map(|(index, output)| {
                            output.ok_or_else(|| {
                                inference_tensor::Error::Msg(format!(
                                    "Phi3 image {index} has no encoder output"
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let image_set_tensor_inner = Tensor::cat(&all_feats, 0)?;
                    image_set_tensor = Some(Either::Right(image_set_tensor_inner));
                } else {
                    let tt = self
                        .get_image_features(pixel_values)?
                        .to_device(&target_dev)?
                        .to_dtype(target_dtype)?
                        .reshape(((), self.image_dim_out))?;
                    let image_set_tensor_inner = self.layers.forward(&tt)?;
                    image_set_tensor = Some(Either::Right(image_set_tensor_inner));
                }
            } else if pixel_values.dims().len() == 3 {
                hd_transform = None;
                let tt = pixel_values
                    .to_device(&target_dev)?
                    .to_dtype(target_dtype)?
                    .reshape(((), self.image_dim_out))?;
                let image_set_tensor_inner = self.layers.forward(&tt)?;
                image_set_tensor = Some(Either::Right(image_set_tensor_inner));
            } else {
                inference_tensor::bail!(
                    "Phi3 image input must have rank 3, 4, or 5, got rank {}",
                    pixel_values.dims().len()
                );
            }
        } else {
            inference_tensor::bail!("Phi3 received image pixels without image placeholders");
        }

        let input_ids = input_ids.clamp(0.0, self.vocab_size as f64)?;
        let mut hidden_states = self.wte.embedding_forward(&input_ids, self.dtype)?;
        let expected_placeholder_rows = match (&hd_transform, &image_set_tensor) {
            (Some(output_lens), Some(Either::Left(outputs))) => {
                if output_lens.len() != outputs.len() {
                    inference_tensor::bail!("Phi3 HD encoder output metadata is inconsistent");
                }
                let mut total = 0usize;
                for (&output_len, output) in output_lens.iter().zip(outputs) {
                    let (output_batch, output_rows, _) = output.dims3()?;
                    if output_batch != 1 || output_rows != output_len {
                        inference_tensor::bail!(
                            "Phi3 HD encoder returned shape {:?}, expected one batch and {output_len} rows",
                            output.dims()
                        );
                    }
                    total = total.checked_add(output_len).ok_or_else(|| {
                        inference_tensor::Error::Msg("Phi3 image token count overflow".into())
                    })?;
                }
                total
            }
            (None, Some(Either::Right(outputs))) => {
                let (output_rows, _) = outputs.dims2()?;
                let image_count = pixel_values.dim(0)?;
                let expected = image_count
                    .checked_mul(self.num_img_tokens)
                    .ok_or_else(|| {
                        inference_tensor::Error::Msg("Phi3 image token count overflow".into())
                    })?;
                if output_rows != expected {
                    inference_tensor::bail!(
                        "Phi3 encoder returned {} rows for {image_count} images",
                        output_rows
                    );
                }
                expected
            }
            _ => inference_tensor::bail!("Phi3 media has no encoder outputs"),
        };
        if positions.dim(0)? != expected_placeholder_rows {
            inference_tensor::bail!(
                "Phi3 received {} image placeholder tokens but encoder metadata requires {expected_placeholder_rows}",
                positions.dim(0)?
            );
        }
        if let Some(layout) = packed_layout {
            let image_outputs = match (&hd_transform, &image_set_tensor) {
                (Some(output_lens), Some(Either::Left(outputs))) => {
                    if output_lens.len() != outputs.len() {
                        inference_tensor::bail!("Phi3 HD encoder output metadata is inconsistent");
                    }
                    outputs.clone()
                }
                (None, Some(Either::Right(outputs))) => {
                    let image_count = pixel_values.dim(0)?;
                    if outputs.dim(0)? != image_count * self.num_img_tokens {
                        inference_tensor::bail!(
                            "Phi3 encoder returned {} rows for {image_count} images",
                            outputs.dim(0)?
                        );
                    }
                    (0..image_count)
                        .map(|index| {
                            outputs
                                .i(index * self.num_img_tokens..(index + 1) * self.num_img_tokens)
                        })
                        .collect::<Result<Vec<_>>>()?
                }
                _ => inference_tensor::bail!("Phi3 packed media has no encoder outputs"),
            };
            if image_hashes.len() != image_outputs.len() {
                inference_tensor::bail!(
                    "packed Phi3 input has {} image hashes but {} encoder outputs",
                    image_hashes.len(),
                    image_outputs.len()
                );
            }
            let encoder_outputs = image_hashes
                .iter()
                .copied()
                .zip(image_outputs)
                .map(|(hash, output)| {
                    (
                        MultimodalEncoderKey {
                            kind: crate::paged_attention::block_hash::MultimodalKind::Image,
                            hash,
                        },
                        vec![output],
                    )
                })
                .collect::<MultimodalEncoderOutputs>();
            return layout.splice_embeddings(&hidden_states, &encoder_outputs);
        }
        match (hd_transform, image_set_tensor) {
            (Some(output_lens), Some(Either::Left(image_set_tensors))) => {
                let mut idx = 0;
                for (cnt, img_set_tensor) in output_lens.into_iter().zip(image_set_tensors) {
                    let img_set_tensor = img_set_tensor
                        .to_device(&target_dev)?
                        .to_dtype(target_dtype)?;
                    // hidden_states[positions[idx, 0], positions[idx, 1] : positions[idx, 1] + cnt] = ...
                    let p_0 = positions.i((idx, 0))?.to_scalar::<u32>()? as usize;
                    let p_1 = positions.i((idx, 1))?.to_scalar::<u32>()? as usize;
                    hidden_states = hidden_states.slice_assign(
                        &[p_0..p_0 + 1, p_1..p_1 + cnt, 0..img_set_tensor.dim(2)?],
                        &img_set_tensor,
                    )?;
                    idx += cnt;
                }
            }
            (None, Some(Either::Right(image_set_tensor))) => {
                let mut idx = 0;
                // Know len(img_embeds) == pixel_values.dim(0) == len(selected_g_values)
                // https://huggingface.co/microsoft/Phi-3.5-vision-instruct/blob/dbcdaaacf52c8e40cf8de6d6ffa6ff6860e5f256/image_embedding_phi3_v.py#L259
                for i in 0..pixel_values.dim(0)? {
                    let cnt = self.num_img_tokens;
                    let img_set_tensor = image_set_tensor
                        .i(i * cnt..(i + 1) * cnt)?
                        .to_device(&target_dev)?
                        .to_dtype(target_dtype)?;
                    let hidden_size = img_set_tensor.dim(1)?;
                    let img_set_tensor = img_set_tensor.unsqueeze(0)?;
                    let p_0 = positions.i((idx, 0))?.to_scalar::<u32>()? as usize;
                    let p_1 = positions.i((idx, 1))?.to_scalar::<u32>()? as usize;
                    // hidden_states[positions[idx, 0], positions[idx, 1] : positions[idx, 1] + cnt] = ...
                    hidden_states = hidden_states.slice_assign(
                        &[p_0..p_0 + 1, p_1..p_1 + cnt, 0..hidden_size],
                        &img_set_tensor,
                    )?;
                    idx += cnt;
                }
            }
            _ => inference_tensor::bail!("Phi3 media has no encoder outputs"),
        }

        Ok(hidden_states)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        if let Some(glb_gn) = self.glb_gn.clone() {
            uvb.add_tensor("glb_GN", glb_gn);
        }
        if let Some(sub_gn) = self.sub_gn.clone() {
            uvb.add_tensor("sub_GN", sub_gn);
        }
        uvb.extend(self.tensors.clone());
        uvb.pp("img_processor.vision_model")
            .extend(self.image_processor.residual_tensors());

        uvb.to_safetensors()
    }
}

pub struct Model {
    vision_embed_tokens: ImageEmbedding,
    lm: CausalLm,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let spec = DecoderSpec {
            // Phi-3V reads its lm_head under the checkpoint's quantization config, Phi-3 never does
            unquantized_lm_head: false,
            ..cfg.text_config().decoder_spec()
        };
        let lm = CausalLm::new(
            &spec,
            vb,
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        let vision_embed_tokens = ImageEmbedding::new(
            cfg,
            lm.embed_tokens().clone(),
            vb_m.dtype(),
            &cfg.embd_layer,
            lm.stack_mapper()
                .set_nm_device(vb_m.pp("vision_embed_tokens"), false),
        )?;
        Ok(Self {
            vision_embed_tokens,
            lm,
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(ENCODER_CACHE_ENTRIES))),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        ctx: &mut ModelForwardContext<'_>,
        image_sizes: Option<Vec<(usize, usize)>>,
        image_hashes: &[u64],
        packed_layout: Option<&PackedMultimodalLayout>,
    ) -> Result<Tensor> {
        let Some(pixel_values) = pixel_values else {
            return self.lm.forward(input_ids, ctx);
        };
        let xs = self.vision_embed_tokens.forward(
            input_ids,
            &pixel_values,
            image_sizes,
            image_hashes,
            packed_layout,
            &self.encoder_cache,
        )?;
        self.lm.forward_embeds(input_ids, xs, ctx)
    }
}

impl IsqModel for Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_m = uvb.pp("model");
        uvb_m
            .pp("vision_embed_tokens")
            .extend(self.vision_embed_tokens.residual_tensors());
        self.lm.residual_tensors_m(uvb_m)
    }
}

#[derive(Default)]
pub struct Phi3VisionSpecificArgs {
    pub image_sizes: Option<Vec<(usize, usize)>>,
    pub image_hashes: Vec<u64>,
    pub packed_layout: Option<PackedMultimodalLayout>,
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
        let Phi3VisionSpecificArgs {
            image_sizes,
            image_hashes,
            packed_layout,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `Phi3VisionSpecificArgs`");
        self.forward(
            input_ids,
            pixel_values,
            ctx,
            image_sizes,
            &image_hashes,
            packed_layout.as_ref(),
        )
    }
    fn cache(&self) -> &EitherCache {
        NormalModel::cache(&self.lm)
    }
    fn device(&self) -> &Device {
        NormalModel::device(&self.lm)
    }
    fn max_seq_len(&self) -> usize {
        NormalModel::max_seq_len(&self.lm)
    }
    fn config(&self) -> &ModelConfigMetadata {
        NormalModel::config(&self.lm)
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn Any> {
        Box::new(Phi3VisionSpecificArgs::default())
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
        self.lm.get_mlps()
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        self.lm.get_mlps_mut()
    }
    fn amoe_lora_targets(&self) -> &'static [AnyMoeLoraTarget] {
        self.lm.amoe_lora_targets()
    }
    fn amoe_fine_tuned_expert(
        &self,
        layer: usize,
        base: &dyn MlpLayer,
        vb: ShardedVarBuilder,
    ) -> Result<Box<dyn MlpLayer>> {
        self.lm.amoe_fine_tuned_expert(layer, base, vb)
    }
    fn amoe_supported(&self) -> bool {
        self.lm.amoe_supported()
    }
}
