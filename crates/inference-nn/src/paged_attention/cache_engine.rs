use std::{
    str::FromStr,
    sync::{Arc, Mutex, MutexGuard},
};

use inference_tensor::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

use super::config::{KvCacheLayout, ModelConfigLike};
#[cfg(all(feature = "cuda", target_family = "unix"))]
use crate::flashinfer::{Fa3PrefillWorkspaceRegistration, register_fa3_prefill_caches};

#[cfg(all(feature = "cuda", target_family = "unix"))]
const FP8_CACHE_MIN_COMPUTE_MAJOR: usize = 8;

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn cuda_supports_fp8(device: &Device) -> bool {
    if !inference_paged_attn::USE_FP8 {
        return false;
    }
    let Device::Cuda(cuda) = device else {
        return false;
    };
    cuda.compute_major() >= FP8_CACHE_MIN_COMPUTE_MAJOR
}

/// The paged KV cache's storage: the activation dtype, fp8 e4m3, or K and V quantized to `k`/`v` bits per value
/// (`inference_paged_attn::kv_quant`). Written as `auto`, `none`, `f8e4m3`, `q4` (both 4 bits) or `q8/q4` (K/V).
/// `auto` lets the loader quantize a cache that buys GPU room for offloaded MoE experts; `none` keeps the activation
/// dtype regardless.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PagedCacheType {
    #[default]
    Auto,
    Unquantized,
    F8E4M3,
    Quant {
        k: u32,
        v: u32,
    },
}

const QUANT_PREFIX: char = 'q';
const KV_SEPARATOR: char = '/';

impl PagedCacheType {
    /// The element type of the K and V cache tensors; quantized caches are rows of bytes.
    pub fn to_dtype(&self, act_dtype: DType) -> DType {
        match self {
            PagedCacheType::F8E4M3 => DType::F8E4M3,
            PagedCacheType::Quant { .. } => DType::U8,
            PagedCacheType::Auto | PagedCacheType::Unquantized => act_dtype,
        }
    }

    /// Bytes a cached value takes, as `numerator / denominator`, over K and V together: quantized K and V hold the
    /// same number of values (an HND cache), each with its share of the group scales.
    pub fn bytes_per_value(&self, act_dtype: DType) -> (usize, usize) {
        match self {
            PagedCacheType::Quant { k, v } => {
                // kv_quant::bytes_per_value(b) is (2b + 1) / 16; averaged over K and V
                ((2 * *k + 1 + 2 * *v + 1) as usize, 32)
            }
            other => (other.to_dtype(act_dtype).size_in_bytes(), 1),
        }
    }

    pub fn validate(
        &self,
        act_dtype: DType,
        model_config: &dyn ModelConfigLike,
        device: &Device,
        layer_devices: &[Option<Device>],
    ) -> std::result::Result<(), String> {
        match self {
            Self::Auto | Self::Unquantized => Ok(()),
            Self::F8E4M3 => validate_fp8(act_dtype, model_config, device, layer_devices),
            Self::Quant { k, v } => {
                validate_quant(*k, *v, act_dtype, model_config, device, layer_devices)
            }
        }
    }
}

fn validate_fp8(
    act_dtype: DType,
    model_config: &dyn ModelConfigLike,
    device: &Device,
    layer_devices: &[Option<Device>],
) -> std::result::Result<(), String> {
    if !matches!(act_dtype, DType::F16 | DType::BF16 | DType::F32) {
        return Err(format!(
            "FP8 KV cache requires f16, bf16, or f32 activations, got {act_dtype:?}"
        ));
    }

    for layer_idx in 0..model_config.num_layers() {
        if !model_config.layer_has_paged_kv_cache(layer_idx) {
            continue;
        }
        if matches!(
            model_config.kv_cache_layout_for_layer(layer_idx),
            KvCacheLayout::Mla { .. }
        ) {
            return Err(format!(
                "FP8 KV cache is not supported for MLA layer {layer_idx}"
            ));
        }
        let layer_device = layer_devices
            .get(layer_idx)
            .and_then(Option::as_ref)
            .unwrap_or(device);
        if layer_device.is_cuda() {
            #[cfg(all(feature = "cuda", target_family = "unix"))]
            if !cuda_supports_fp8(layer_device) {
                return Err(
                    "FP8 KV cache requires CUDA compute capability 8.0 or newer and a matching CUDA build"
                        .to_string(),
                );
            }
            #[cfg(not(all(feature = "cuda", target_family = "unix")))]
            return Err("FP8 KV cache requires the CUDA paged-attention backend".to_string());
        } else if layer_device.is_metal() {
            #[cfg(not(feature = "metal"))]
            return Err("FP8 KV cache requires the Metal paged-attention backend".to_string());
        } else {
            return Err(format!(
                "FP8 KV cache is only supported on CUDA or Metal, got {layer_device:?} for layer {layer_idx}"
            ));
        }
    }
    Ok(())
}

// Quantized caches are read by fattn's paged mma kernel or dequantized by a gather: CUDA, the HND layout, and head
// dims the row format takes.
fn validate_quant(
    k_bits: u32,
    v_bits: u32,
    act_dtype: DType,
    model_config: &dyn ModelConfigLike,
    device: &Device,
    layer_devices: &[Option<Device>],
) -> std::result::Result<(), String> {
    use inference_paged_attn::kv_quant;
    let bits = kv_quant::MIN_BITS..=kv_quant::MAX_BITS;
    if !bits.contains(&k_bits) || !bits.contains(&v_bits) {
        return Err(format!(
            "a quantized KV cache takes {}..{} bits for K and V, got q{k_bits}/q{v_bits}",
            kv_quant::MIN_BITS,
            kv_quant::MAX_BITS
        ));
    }
    if !matches!(act_dtype, DType::F16 | DType::BF16 | DType::F32) {
        return Err(format!(
            "a quantized KV cache requires f16, bf16, or f32 activations, got {act_dtype:?}"
        ));
    }
    #[cfg(not(all(feature = "cuda", target_family = "unix")))]
    {
        let _ = (model_config, device, layer_devices);
        Err("a quantized KV cache requires the CUDA paged-attention backend".to_string())
    }
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    {
        for layer_idx in 0..model_config.num_layers() {
            if !model_config.layer_has_paged_kv_cache(layer_idx) {
                continue;
            }
            let layer_device = layer_devices
                .get(layer_idx)
                .and_then(Option::as_ref)
                .unwrap_or(device);
            if !layer_device.is_cuda() {
                return Err(format!(
                    "a quantized KV cache is only supported on CUDA, got {layer_device:?} for layer {layer_idx}"
                ));
            }
            if !inference_fattn::mma_available() {
                return Err(
                    "a quantized KV cache needs fattn's mma kernel (Turing or newer on every visible GPU)".to_string(),
                );
            }
            let head_dim = model_config.k_head_dim_for_layer(layer_idx);
            let layout = model_config.kv_cache_layout_for_layer(layer_idx);
            if !matches!(layout, KvCacheLayout::FlashInferHnd)
                || head_dim != model_config.v_head_dim_for_layer(layer_idx)
            {
                return Err(format!(
                    "a quantized KV cache needs equal K and V head dims in the HND cache layout; layer {layer_idx} uses {layout:?}"
                ));
            }
            if !head_dim.is_multiple_of(kv_quant::HEAD_DIM_MULTIPLE) {
                return Err(format!(
                    "a quantized KV cache needs head dims that are a multiple of {}, layer {layer_idx} has {head_dim}",
                    kv_quant::HEAD_DIM_MULTIPLE
                ));
            }
        }
        Ok(())
    }
}

impl FromStr for PagedCacheType {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let quant_bits = |part: &str| {
            part.strip_prefix(QUANT_PREFIX)
                .and_then(|bits| bits.parse::<u32>().ok())
                .filter(|bits| {
                    (inference_paged_attn::kv_quant::MIN_BITS
                        ..=inference_paged_attn::kv_quant::MAX_BITS)
                        .contains(bits)
                })
        };
        match s {
            "auto" => Ok(Self::Auto),
            "none" => Ok(Self::Unquantized),
            "f8e4m3" => Ok(Self::F8E4M3),
            other => {
                let (k, v) = other.split_once(KV_SEPARATOR).unwrap_or((other, other));
                match (quant_bits(k), quant_bits(v)) {
                    (Some(k), Some(v)) => Ok(Self::Quant { k, v }),
                    _ => Err(format!(
                        "unexpected KV cache type `{other}`: expected `auto`, `none`, `f8e4m3`, `q2`..`q8`, or `qK/qV` (K/V bits)"
                    )),
                }
            }
        }
    }
}

impl std::fmt::Display for PagedCacheType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Unquantized => f.write_str("none"),
            Self::F8E4M3 => f.write_str("f8e4m3"),
            Self::Quant { k, v } if k == v => write!(f, "{QUANT_PREFIX}{k}"),
            Self::Quant { k, v } => write!(f, "{QUANT_PREFIX}{k}{KV_SEPARATOR}{QUANT_PREFIX}{v}"),
        }
    }
}

impl Serialize for PagedCacheType {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

inference_json::via_serde!(PagedCacheType);

impl<'de> Deserialize<'de> for PagedCacheType {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(feature = "utoipa")]
impl utoipa::PartialSchema for PagedCacheType {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        use utoipa::openapi::{ObjectBuilder, RefOr, Schema, Type, schema::SchemaType};

        RefOr::T(Schema::Object(
            ObjectBuilder::new()
                .schema_type(SchemaType::Type(Type::String))
                .description(Some(
                    "KV cache storage: `auto` (the activation dtype, or a quantized cache where that buys GPU room for offloaded MoE experts), `none` (the activation dtype), `f8e4m3`, `q2`..`q8` (K and V bits), or `qK/qV`.",
                ))
                .build(),
        ))
    }
}

#[cfg(feature = "utoipa")]
impl utoipa::ToSchema for PagedCacheType {}

#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub block_size: usize,
    pub num_gpu_blocks: usize,
    pub cache_type: PagedCacheType,
    /// Each layer's own storage (`cache_layers::plan`); a layer past its end stores `cache_type`.
    pub layer_types: Vec<PagedCacheType>,
    pub kv_cache_group_ids: Vec<u32>,
}

impl CacheConfig {
    /// What layer `layer_idx` stores.
    pub fn layer_type(&self, layer_idx: usize) -> PagedCacheType {
        self.layer_types
            .get(layer_idx)
            .copied()
            .unwrap_or(self.cache_type)
    }
}

pub type KVCache = (Tensor, Tensor);

pub struct CacheEngine {
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    _fa3_prefill_workspaces: Fa3PrefillWorkspaceRegistration,
    fa3_prefill_num_sm_by_layer: Vec<Option<usize>>,
    gpu_cache: Arc<Mutex<Vec<KVCache>>>,
}

impl CacheEngine {
    pub fn new(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        layer_devices: Vec<Option<Device>>,
    ) -> Result<Self> {
        cache_config
            .cache_type
            .validate(dtype, model_config, device, &layer_devices)
            .map_err(inference_tensor::Error::msg)?;
        let gpu_cache =
            Self::allocate_gpu_cache(model_config, cache_config, dtype, device, layer_devices)?;
        #[cfg(all(feature = "cuda", target_family = "unix"))]
        let fa3_prefill_workspaces = register_fa3_prefill_caches(&gpu_cache)?;
        #[cfg(all(feature = "cuda", target_family = "unix"))]
        let fa3_prefill_num_sm_by_layer =
            Self::fa3_prefill_cache_coverage(model_config, cache_config, &gpu_cache)?;
        #[cfg(not(all(feature = "cuda", target_family = "unix")))]
        let fa3_prefill_num_sm_by_layer = vec![None; model_config.num_layers()];
        Ok(Self {
            #[cfg(all(feature = "cuda", target_family = "unix"))]
            _fa3_prefill_workspaces: fa3_prefill_workspaces,
            fa3_prefill_num_sm_by_layer,
            gpu_cache: Arc::new(Mutex::new(gpu_cache)),
        })
    }

    pub fn fa3_prefill_num_sm_by_layer(&self) -> &[Option<usize>] {
        &self.fa3_prefill_num_sm_by_layer
    }

    pub fn get_kv_cache(&self) -> MutexGuard<'_, Vec<KVCache>> {
        // Use blocking lock instead of busy-wait spin loop to avoid CPU waste
        // and potential thread starvation issues
        self.gpu_cache.lock().expect("KV cache mutex was poisoned")
    }

    fn allocate_gpu_cache(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        model_dtype: DType,
        device: &Device,
        layer_devices: Vec<Option<Device>>,
    ) -> Result<Vec<KVCache>> {
        let mut gpu_cache = Vec::new();

        for (layer_idx, device) in layer_devices
            .iter()
            .take(model_config.num_layers())
            .map(|x| x.as_ref().unwrap_or(device))
            .enumerate()
        {
            // a drafter's auxiliary layers keep the model dtype beside a quantized cache
            let dtype = cache_config.layer_type(layer_idx).to_dtype(model_dtype);
            // Hybrid models keep no paged cache on linear/recurrent layers, but the vec stays indexed
            // by absolute layer index, so those get an empty tensor of the right rank instead.
            let num_gpu_blocks = if model_config.layer_has_paged_kv_cache(layer_idx) {
                cache_config.num_gpu_blocks
            } else {
                0
            };
            let requested_kv_cache_layout = model_config.kv_cache_layout_for_layer(layer_idx);
            let kv_cache_layout =
                if matches!(requested_kv_cache_layout, KvCacheLayout::FlashInferHnd)
                    && !device.is_cuda()
                {
                    KvCacheLayout::Standard
                } else {
                    requested_kv_cache_layout
                };
            let (key_blocks, value_blocks) = match kv_cache_layout {
                KvCacheLayout::Standard => {
                    let key_block_shape = Self::calculate_key_block_shape(
                        model_config,
                        dtype,
                        cache_config.block_size,
                        layer_idx,
                    );
                    let value_block_shape = Self::calculate_value_block_shape(
                        model_config,
                        cache_config.block_size,
                        layer_idx,
                    );
                    #[cfg_attr(not(feature = "metal"), allow(unused_variables))]
                    let key_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use inference_tensor::{MetalStorage, Shape, Storage};

                            let elem_count = num_gpu_blocks
                                * key_block_shape.0
                                * key_block_shape.1
                                * key_block_shape.2
                                * key_block_shape.3;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "k_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                    key_block_shape.3,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                    key_block_shape.3,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    #[cfg_attr(not(feature = "metal"), allow(unused_variables))]
                    let value_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use inference_tensor::{MetalStorage, Shape, Storage};

                            let elem_count = num_gpu_blocks
                                * value_block_shape.0
                                * value_block_shape.1
                                * value_block_shape.2;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "v_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    num_gpu_blocks,
                                    value_block_shape.0,
                                    value_block_shape.1,
                                    value_block_shape.2,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    num_gpu_blocks,
                                    value_block_shape.0,
                                    value_block_shape.1,
                                    value_block_shape.2,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    (key_blocks, value_blocks)
                }
                KvCacheLayout::FlashInferHnd
                    if let PagedCacheType::Quant { k, v } = cache_config.layer_type(layer_idx) =>
                {
                    // rows of bytes: each head's values at K's or V's bits, then their group scales
                    let (heads, block_size, head_dim) = Self::calculate_flashinfer_block_shape(
                        model_config,
                        cache_config.block_size,
                        layer_idx,
                    );
                    let rows = |bits| {
                        let row = inference_paged_attn::kv_quant::row_bytes(head_dim, bits);
                        // SAFETY: a row is read only after the write that fills it, and masked rows read written ones
                        unsafe {
                            Tensor::empty(
                                (num_gpu_blocks, heads, block_size, row),
                                DType::U8,
                                device,
                            )
                        }
                    };
                    (rows(k)?, rows(v)?)
                }
                KvCacheLayout::FlashInferHnd => {
                    let key_block_shape = Self::calculate_flashinfer_block_shape(
                        model_config,
                        cache_config.block_size,
                        layer_idx,
                    );
                    #[cfg_attr(not(feature = "metal"), allow(unused_variables))]
                    let key_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use inference_tensor::{MetalStorage, Shape, Storage};

                            let elem_count = num_gpu_blocks
                                * key_block_shape.0
                                * key_block_shape.1
                                * key_block_shape.2;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "k_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    let value_blocks = unsafe {
                        Tensor::empty(
                            (
                                num_gpu_blocks,
                                key_block_shape.0,
                                key_block_shape.1,
                                key_block_shape.2,
                            ),
                            dtype,
                            device,
                        )?
                    };
                    (key_blocks, value_blocks)
                }
                KvCacheLayout::Mla {
                    kv_lora_rank,
                    kpe_head_dim,
                } => {
                    #[cfg_attr(not(feature = "metal"), allow(unused_variables))]
                    let key_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use inference_tensor::{MetalStorage, Shape, Storage};

                            let elem_count =
                                num_gpu_blocks * cache_config.block_size * kv_lora_rank;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "k_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    num_gpu_blocks,
                                    cache_config.block_size,
                                    kv_lora_rank,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (num_gpu_blocks, cache_config.block_size, kv_lora_rank),
                                dtype,
                                device,
                            )?
                        }
                    };
                    #[cfg_attr(not(feature = "metal"), allow(unused_variables))]
                    let value_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use inference_tensor::{MetalStorage, Shape, Storage};

                            let elem_count =
                                num_gpu_blocks * cache_config.block_size * kpe_head_dim;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "v_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    num_gpu_blocks,
                                    cache_config.block_size,
                                    kpe_head_dim,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (num_gpu_blocks, cache_config.block_size, kpe_head_dim),
                                dtype,
                                device,
                            )?
                        }
                    };
                    (key_blocks, value_blocks)
                }
            };
            gpu_cache.push((key_blocks, value_blocks));
        }
        Ok(gpu_cache)
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    fn fa3_prefill_cache_coverage(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        gpu_cache: &[KVCache],
    ) -> Result<Vec<Option<usize>>> {
        if !inference_paged_attn::USE_FA3_FP8_PAGED || cache_config.num_gpu_blocks == 0 {
            return Ok(vec![None; model_config.num_layers()]);
        }
        (0..model_config.num_layers())
            .map(|layer_idx| -> Result<Option<usize>> {
                if !model_config.layer_has_paged_kv_cache(layer_idx) {
                    return Ok(None);
                }
                let Some((key_cache, value_cache)) = gpu_cache.get(layer_idx) else {
                    return Ok(None);
                };
                let kv_heads = model_config.num_kv_heads_for_layer(layer_idx);
                let q_heads = model_config.num_attn_heads_for_layer(layer_idx);
                let k_head_dim = model_config.k_head_dim_for_layer(layer_idx);
                let v_head_dim = model_config.v_head_dim_for_layer(layer_idx);
                let expected_shape = (
                    cache_config.num_gpu_blocks,
                    kv_heads,
                    cache_config.block_size,
                    k_head_dim,
                );
                if model_config.kv_cache_layout_for_layer(layer_idx) != KvCacheLayout::FlashInferHnd
                    || kv_heads == 0
                    || q_heads == 0
                    || !q_heads.is_multiple_of(kv_heads)
                    || k_head_dim != 256
                    || v_head_dim != k_head_dim
                    || key_cache.dtype() != DType::F8E4M3
                    || value_cache.dtype() != DType::F8E4M3
                    || key_cache.dims4().ok() != Some(expected_shape)
                    || value_cache.dims4().ok() != Some(expected_shape)
                {
                    return Ok(None);
                }
                crate::flashinfer::fa3_prefill_cache_num_sm(
                    key_cache,
                    value_cache,
                    q_heads,
                    kv_heads,
                    k_head_dim,
                    cache_config.block_size,
                )
            })
            .collect()
    }

    fn calculate_key_block_shape(
        model_config: &dyn ModelConfigLike,
        dtype: DType,
        block_size: usize,
        layer_idx: usize,
    ) -> (usize, usize, usize, usize) {
        let element_size = dtype.size_in_bytes();
        let x = 16 / element_size;
        (
            model_config.num_kv_heads_for_layer(layer_idx),
            model_config.k_head_dim_for_layer(layer_idx) / x,
            block_size,
            x,
        )
    }

    fn calculate_value_block_shape(
        model_config: &dyn ModelConfigLike,
        block_size: usize,
        layer_idx: usize,
    ) -> (usize, usize, usize) {
        (
            model_config.num_kv_heads_for_layer(layer_idx),
            model_config.v_head_dim_for_layer(layer_idx),
            block_size,
        )
    }

    fn calculate_flashinfer_block_shape(
        model_config: &dyn ModelConfigLike,
        block_size: usize,
        layer_idx: usize,
    ) -> (usize, usize, usize) {
        (
            model_config.num_kv_heads_for_layer(layer_idx),
            block_size,
            model_config.k_head_dim_for_layer(layer_idx),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::config::ModelConfigMetadata;

    fn model_config(layout: KvCacheLayout) -> ModelConfigMetadata {
        ModelConfigMetadata {
            max_seq_len: 4096,
            num_layers: 2,
            hidden_size: 1024,
            num_kv_heads: 4,
            num_attn_heads: 16,
            sliding_window: None,
            k_head_dim: 128,
            v_head_dim: 128,
            kv_cache_layout: layout,
        }
    }

    #[test]
    fn fp8_cache_rejects_cpu_before_allocation() {
        let err = PagedCacheType::F8E4M3
            .validate(
                DType::BF16,
                &model_config(KvCacheLayout::Standard),
                &Device::Cpu,
                &[],
            )
            .unwrap_err();
        assert!(err.contains("only supported on CUDA or Metal"));
    }

    #[test]
    fn fp8_cache_rejects_mla_before_allocation() {
        let err = PagedCacheType::F8E4M3
            .validate(
                DType::BF16,
                &model_config(KvCacheLayout::Mla {
                    kv_lora_rank: 512,
                    kpe_head_dim: 64,
                }),
                &Device::Cpu,
                &[],
            )
            .unwrap_err();
        assert!(err.contains("not supported for MLA layer 0"));
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn fa3_prefill_coverage_requires_compatible_registered_caches() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let mut model = model_config(KvCacheLayout::FlashInferHnd);
        model.k_head_dim = 256;
        model.v_head_dim = 256;
        let cache = CacheConfig {
            block_size: 32,
            num_gpu_blocks: 2,
            cache_type: PagedCacheType::F8E4M3,
            layer_types: Vec::new(),
            kv_cache_group_ids: vec![0],
        };
        let engine = CacheEngine::new(
            &model,
            &cache,
            DType::BF16,
            &device,
            vec![None; model.num_layers],
        )?;
        let expected_num_sm = if inference_paged_attn::USE_FA3_FP8_PAGED {
            crate::flashinfer::fa3_device_num_sm(&device)
        } else {
            None
        };
        assert_eq!(
            engine.fa3_prefill_num_sm_by_layer(),
            &[expected_num_sm, expected_num_sm]
        );

        model.v_head_dim = 128;
        let incompatible = CacheEngine::new(
            &model,
            &cache,
            DType::BF16,
            &device,
            vec![None; model.num_layers],
        )?;
        assert_eq!(incompatible.fa3_prefill_num_sm_by_layer(), &[None, None]);
        Ok(())
    }
}
