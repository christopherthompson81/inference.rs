//! What each layer's paged KV cache stores: the base type, raised for the layers a model's profile (or the general
//! rule, for models without one) names as sensitive, then any layers the request sets itself.

use std::{str::FromStr, sync::OnceLock};

use serde::{Deserialize, Serialize};

use super::{cache_engine::PagedCacheType, config::ModelConfigLike};

// Measured profiles (docs/investigations/kv_cache_quant_investigation.md); `scripts/kv_layer_sweep.py` makes more.
const PROFILES_JSON: &str = include_str!("kv_profiles.json");
// Without a profile, a base K below this many bits keeps the first cached layer's K and V at `PROTECTED_BITS`: in
// dense Qwen2/Qwen3 models layer 0 alone carries most of the cost of low K widths, and its V at 8 bits gains more.
const RULE_BELOW_BITS: u32 = 6;
const PROTECTED_BITS: u32 = 8;
const UNIFORM: &str = "uniform";
// overrides live in a fixed array so the paged attention config stays `Copy`
const MAX_OVERRIDES: usize = 16;
const RANGE_SEPARATOR: char = '-';

/// Per-layer cache storage on top of a base type: `0=q8,5=q8,20-27=q6` sets those layers (and still takes the
/// model's profile or the general rule for the rest); `uniform` gives every layer the base type and nothing else.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PagedCacheLayers {
    uniform: bool,
    // (first layer, last layer, type), in the order given
    overrides: [Option<(usize, usize, PagedCacheType)>; MAX_OVERRIDES],
}

impl PagedCacheLayers {
    fn items(&self) -> impl Iterator<Item = (usize, usize, PagedCacheType)> + '_ {
        self.overrides.iter().flatten().copied()
    }

    fn has_overrides(&self) -> bool {
        self.items().next().is_some()
    }
}

impl FromStr for PagedCacheLayers {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let s = s.trim();
        if s.is_empty() {
            return Ok(Self::default());
        }
        if s == UNIFORM {
            return Ok(Self {
                uniform: true,
                ..Self::default()
            });
        }
        let mut layers = Self::default();
        for (slot, item) in s.split(',').enumerate() {
            if slot >= MAX_OVERRIDES {
                return Err(format!("at most {MAX_OVERRIDES} layer items, got `{s}`"));
            }
            let (range, cache) = item.split_once('=').ok_or_else(|| {
                format!("expected `layers=type` such as `0=q8` or `20-27=q6`, got `{item}`")
            })?;
            let layer = |n: &str| {
                n.trim()
                    .parse::<usize>()
                    .map_err(|_| format!("`{n}` in `{item}` is not a layer index"))
            };
            let (first, last) = match range.split_once(RANGE_SEPARATOR) {
                Some((first, last)) => (layer(first)?, layer(last)?),
                None => (layer(range)?, layer(range)?),
            };
            if first > last {
                return Err(format!("the layer range in `{item}` is empty"));
            }
            layers.overrides[slot] = Some((first, last, cache.trim().parse()?));
        }
        Ok(layers)
    }
}

impl std::fmt::Display for PagedCacheLayers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.uniform {
            return f.write_str(UNIFORM);
        }
        let items: Vec<String> = self
            .items()
            .map(|(first, last, cache)| match first == last {
                true => format!("{first}={cache}"),
                false => format!("{first}{RANGE_SEPARATOR}{last}={cache}"),
            })
            .collect();
        f.write_str(&items.join(","))
    }
}

impl Serialize for PagedCacheLayers {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for PagedCacheLayers {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(feature = "utoipa")]
impl utoipa::PartialSchema for PagedCacheLayers {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        use utoipa::openapi::{ObjectBuilder, RefOr, Schema, Type, schema::SchemaType};

        RefOr::T(Schema::Object(
            ObjectBuilder::new()
                .schema_type(SchemaType::Type(Type::String))
                .description(Some(
                    "Per-layer KV cache storage: `layers=type` items such as `0=q8,20-27=q6`, or `uniform` for the \
                     base type everywhere, without the model's profile or the general rule.",
                ))
                .build(),
        ))
    }
}

#[cfg(feature = "utoipa")]
impl utoipa::ToSchema for PagedCacheLayers {}

/// The shape a profile recognizes a model by, the same for every quantization of its weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
pub struct KvFingerprint {
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl KvFingerprint {
    fn of(model: &dyn ModelConfigLike) -> Self {
        Self {
            layers: model.num_layers() - model.auxiliary_paged_layers(),
            hidden: model.hidden_size(),
            heads: model.num_attn_heads(),
            kv_heads: model.num_kv_heads(),
            head_dim: model.k_head_dim(),
        }
    }
}

/// Lowest bits a layer's K and V take when the cache is quantized; unset keeps the base type's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct Floor {
    #[serde(default)]
    k: Option<u32>,
    #[serde(default)]
    v: Option<u32>,
}

/// A measured model's per-layer floors; an empty map means no layer needs one (and the general rule is not applied).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    name: String,
    fingerprint: KvFingerprint,
    floors: std::collections::BTreeMap<usize, Floor>,
}

fn profiles() -> &'static [Profile] {
    static PROFILES: OnceLock<Vec<Profile>> = OnceLock::new();
    PROFILES.get_or_init(|| serde_json::from_str(PROFILES_JSON).expect("kv_profiles.json is valid"))
}

/// The measured profile a model of this shape gets when its cache is quantized; `None` leaves it to the general rule.
pub fn profile_name(fingerprint: &KvFingerprint) -> Option<&'static str> {
    profiles()
        .iter()
        .find(|profile| profile.fingerprint == *fingerprint)
        .map(|profile| profile.name.as_str())
}

/// Each layer's cache type (by layer index; only layers with a paged cache matter) and what set it, for the log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CachePlan {
    pub layers: Vec<PagedCacheType>,
    pub source: String,
}

impl CachePlan {
    /// The types the plan's paged layers use, once each, for validation.
    pub fn distinct(&self, model: &dyn ModelConfigLike) -> Vec<PagedCacheType> {
        let mut seen = Vec::new();
        for (idx, cache) in self.layers.iter().enumerate() {
            if model.layer_has_paged_kv_cache(idx) && !seen.contains(cache) {
                seen.push(*cache);
            }
        }
        seen
    }

    /// Whether every paged layer stores the base type.
    pub fn is_uniform(&self, base: PagedCacheType, model: &dyn ModelConfigLike) -> bool {
        self.distinct(model).iter().all(|cache| *cache == base)
    }
}

// `cache` with its K and V raised to at least `floor`'s bits; other types are left alone.
fn raised(cache: PagedCacheType, floor: Floor) -> PagedCacheType {
    match cache {
        PagedCacheType::Quant { k, v } => PagedCacheType::Quant {
            k: floor.k.map_or(k, |f| k.max(f)),
            v: floor.v.map_or(v, |f| v.max(f)),
        },
        other => other,
    }
}

/// Each layer's type for `base` on `model`: its profile's floors (or the general rule), then `layers`' overrides.
pub fn plan(
    base: PagedCacheType,
    layers: &PagedCacheLayers,
    model: &dyn ModelConfigLike,
) -> CachePlan {
    let mut types = vec![base; model.num_layers()];
    let mut source = format!("{base}");
    if !layers.uniform && matches!(base, PagedCacheType::Quant { .. }) {
        let fingerprint = KvFingerprint::of(model);
        match profiles()
            .iter()
            .find(|profile| profile.fingerprint == fingerprint)
        {
            Some(profile) => {
                for (&layer, &floor) in &profile.floors {
                    if let Some(cache) = types.get_mut(layer) {
                        *cache = raised(*cache, floor);
                    }
                }
                source = format!("{source}, with the {} profile", profile.name);
            }
            None => {
                let first =
                    (0..model.num_layers()).find(|&idx| model.layer_has_paged_kv_cache(idx));
                if let (Some(first), PagedCacheType::Quant { k, .. }) = (first, base)
                    && k < RULE_BELOW_BITS
                {
                    let protected = Floor {
                        k: Some(PROTECTED_BITS),
                        v: Some(PROTECTED_BITS),
                    };
                    types[first] = raised(base, protected);
                    source = format!(
                        "{source}, layer {first} at q{PROTECTED_BITS} (the first cached layer)"
                    );
                }
            }
        }
    }
    for (first, last, cache) in layers.items() {
        for layer in first..=last {
            if let Some(slot) = types.get_mut(layer) {
                *slot = cache;
            }
        }
    }
    if layers.has_overrides() {
        source = format!("{source}, then {layers}");
    }
    // a drafter's auxiliary layers stay at the model dtype: its attention reads them unquantized
    let own = model.num_layers() - model.auxiliary_paged_layers();
    for slot in &mut types[own..] {
        *slot = PagedCacheType::Auto;
    }
    CachePlan {
        layers: types,
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::config::{KvCacheLayout, ModelConfigMetadata};

    fn model(
        layers: usize,
        hidden: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> ModelConfigMetadata {
        ModelConfigMetadata {
            max_seq_len: 4096,
            num_layers: layers,
            hidden_size: hidden,
            num_kv_heads: kv_heads,
            num_attn_heads: heads,
            sliding_window: None,
            k_head_dim: head_dim,
            v_head_dim: head_dim,
            kv_cache_layout: KvCacheLayout::FlashInferHnd,
        }
    }

    const Q4: PagedCacheType = PagedCacheType::Quant { k: 4, v: 4 };
    const Q8: PagedCacheType = PagedCacheType::Quant { k: 8, v: 8 };

    #[test]
    fn auxiliary_layers_keep_the_model_dtype_and_the_base_fingerprint() {
        let base = model(8, 1024, 16, 4, 128);
        let fingerprint = KvFingerprint::of(&base);
        let with_drafter = crate::paged_attention::AuxiliaryPagedKvCacheConfig::new(
            std::sync::Arc::new(base),
            crate::paged_attention::AuxiliaryPagedLayers {
                count: 3,
                num_attn_heads: 32,
                num_kv_heads: 8,
                head_dim: 128,
            },
        );
        assert_eq!(KvFingerprint::of(&with_drafter), fingerprint);
        let layers: PagedCacheLayers = "0-10=q6".parse().unwrap();
        let plan = plan(Q4, &layers, &with_drafter);
        assert_eq!(plan.layers.len(), 11);
        assert!(
            plan.layers[..8]
                .iter()
                .all(|cache| *cache != PagedCacheType::Auto)
        );
        assert!(
            plan.layers[8..]
                .iter()
                .all(|cache| *cache == PagedCacheType::Auto)
        );
    }

    #[test]
    fn layer_overrides_parse_and_print_back() {
        let layers: PagedCacheLayers = "0=q8, 20-27=q6/q4".parse().unwrap();
        assert_eq!(layers.to_string(), "0=q8,20-27=q6/q4");
        assert_eq!(
            "uniform".parse::<PagedCacheLayers>().unwrap().to_string(),
            "uniform"
        );
        assert!("5".parse::<PagedCacheLayers>().is_err());
        assert!("3-1=q8".parse::<PagedCacheLayers>().is_err());
        assert!("x=q8".parse::<PagedCacheLayers>().is_err());
    }

    #[test]
    fn a_model_without_a_profile_keeps_its_first_layer_at_8_bits_below_6() {
        let unprofiled = model(12, 777, 8, 4, 64);
        let plan = plan(Q4, &PagedCacheLayers::default(), &unprofiled);
        assert_eq!(plan.layers[0], Q8);
        assert!(plan.layers[1..].iter().all(|cache| *cache == Q4));
        // 6 bits and up need nothing, and neither do unquantized caches
        let q6 = PagedCacheType::Quant { k: 6, v: 4 };
        assert!(
            super::plan(q6, &PagedCacheLayers::default(), &unprofiled).is_uniform(q6, &unprofiled)
        );
        let fp8 = PagedCacheType::F8E4M3;
        assert!(
            super::plan(fp8, &PagedCacheLayers::default(), &unprofiled)
                .is_uniform(fp8, &unprofiled)
        );
    }

    #[test]
    fn a_profile_replaces_the_general_rule() {
        // Qwen3 0.6B: layers 0 and 5 stay at 8 bits
        let qwen3 = model(28, 1024, 16, 8, 128);
        let plan = plan(Q4, &PagedCacheLayers::default(), &qwen3);
        let raised: Vec<usize> = (0..28).filter(|&i| plan.layers[i] != Q4).collect();
        assert_eq!(raised, [0, 5]);
        assert_eq!(plan.layers[5], Q8);
        assert!(plan.source.contains("Qwen3 0.6B"), "{}", plan.source);
    }

    #[test]
    fn overrides_come_last_and_uniform_drops_the_rest() {
        let unprofiled = model(12, 777, 8, 4, 64);
        let layers: PagedCacheLayers = "0=q3,10-11=q8".parse().unwrap();
        let plan = plan(Q4, &layers, &unprofiled);
        assert_eq!(plan.layers[0], PagedCacheType::Quant { k: 3, v: 3 });
        assert_eq!(plan.layers[10], PagedCacheType::Quant { k: 8, v: 8 });
        let uniform: PagedCacheLayers = "uniform".parse().unwrap();
        assert!(super::plan(Q4, &uniform, &unprofiled).is_uniform(Q4, &unprofiled));
    }

    #[test]
    fn the_profiles_parse_and_name_distinct_models() {
        let fingerprints: Vec<_> = profiles().iter().map(|p| p.fingerprint).collect();
        for (i, f) in fingerprints.iter().enumerate() {
            assert!(
                !fingerprints[i + 1..].contains(f),
                "two profiles share {f:?}"
            );
        }
    }
}
