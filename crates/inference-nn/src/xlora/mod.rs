mod classifier;
mod config;

use std::sync::MutexGuard;

use inference_tensor::{DType, Device, Result, Tensor};

pub use crate::model::NonGranularState;
use crate::{
    attention::FlashParams,
    get_mut_arcmutex,
    kv_cache::{EitherCache, LayerCaches},
    lora::Ordering,
    model::extract_logits,
};
pub use classifier::XLoraClassifier;
pub use config::XLoraConfig;

/// One pass through an X-LoRA model's base layers; `position_ids` matter only to models whose rotary reads them.
pub struct XLoraPass<'a> {
    pub input_ids: &'a Tensor,
    pub seqlen_offsets: &'a [usize],
    pub position_ids: &'a [usize],
    pub scalings: Option<Tensor>,
    pub is_full_pass: bool,
    pub no_kv_cache: bool,
    pub is_scaling_pass: Option<f64>,
    pub flash_params: &'a FlashParams,
}

/// What each X-LoRA model provides; [`xlora_forward`] builds the scaling pass and the logits on it.
pub trait ScalingsMaker {
    /// `None` when the adapters run without a classifier, at fixed scalings.
    fn classifier(&self) -> Option<&XLoraClassifier>;
    /// For dummy scalings
    fn dtype(&self) -> DType;
    fn get_cache(&self) -> &EitherCache;
    /// The hidden states after the final norm.
    fn inner_forward(&self, pass: XLoraPass<'_>) -> Result<Tensor>;
    fn lm_head(&self, hidden: &Tensor) -> Result<Tensor>;
}

/// The caches a pass writes: the X-LoRA ones for a full pass (emptied first without a KV cache), else the main ones.
pub fn pass_cache(
    cache: &EitherCache,
    is_full_pass: bool,
    no_kv_cache: bool,
) -> MutexGuard<'_, LayerCaches> {
    if !is_full_pass {
        return cache.full().lock();
    }
    let mut layers = cache.full().xlora_lock();
    if no_kv_cache {
        let len = layers.len();
        layers.clone_from(&vec![None; len]);
    }
    layers
}

/// The arguments of [`xlora_forward`], as `NormalModel::xlora_forward` receives them.
pub struct XLoraForward<'a> {
    pub input_ids: &'a Tensor,
    pub input_ids_full: &'a Tensor,
    pub seqlen_offsets: &'a [usize],
    pub seqlen_offsets_full: &'a [usize],
    pub no_kv_cache: bool,
    pub non_granular_state: &'a Option<NonGranularState>,
    pub context_lens: Vec<(usize, usize)>,
    pub position_ids: &'a [usize],
    pub flash_params: &'a FlashParams,
    pub flash_params_full: &'a FlashParams,
}

/// The logits of an X-LoRA model: a scaling pass picks the adapter scalings when there is a classifier, then the
/// scaled pass runs over the full sequence without a KV cache, or over the new tokens with one.
pub fn xlora_forward(model: &dyn ScalingsMaker, args: XLoraForward<'_>) -> Result<Tensor> {
    let full = model.classifier().is_some() && args.no_kv_cache;
    let scalings = match model.classifier() {
        Some(classifier) => Some(get_scalings(model, classifier, &args)?),
        None => None,
    };
    let (input_ids, seqlen_offsets, flash_params) = if full {
        (
            args.input_ids_full,
            args.seqlen_offsets_full,
            args.flash_params_full,
        )
    } else {
        (args.input_ids, args.seqlen_offsets, args.flash_params)
    };
    let hidden = model
        .inner_forward(XLoraPass {
            input_ids,
            seqlen_offsets,
            position_ids: args.position_ids,
            // a scaled pass runs on the X-LoRA cache even with a KV cache
            is_full_pass: scalings.is_some(),
            scalings,
            no_kv_cache: args.no_kv_cache,
            is_scaling_pass: None,
            flash_params,
        })?
        .contiguous()?;
    model.lm_head(&extract_logits(&hidden, args.context_lens)?)
}

fn get_scalings(
    model: &dyn ScalingsMaker,
    classifier: &XLoraClassifier,
    args: &XLoraForward<'_>,
) -> Result<Tensor> {
    let (b_size, _) = args.input_ids_full.dims2()?;
    let (_, seq_len) = args.input_ids.dims2()?;

    if let Some(non_granular_state) = args.non_granular_state {
        if let Some(scalings_cache) = &*model.get_cache().full().get_scalings_cache() {
            return Ok(scalings_cache.clone());
        }
        if seq_len == 1 {
            *get_mut_arcmutex!(non_granular_state.non_granular_index) += 1;
        }
    }

    let dummy_scalings =
        classifier.get_dummy_scalings(b_size, seq_len, args.input_ids.device(), model.dtype())?;
    let scaling_pass = Some(classifier.config.scaling_pass_value);
    // Using X-LoRA cache here
    let hidden_states = if args.no_kv_cache {
        let res = model.inner_forward(XLoraPass {
            input_ids: args.input_ids_full,
            seqlen_offsets: args.seqlen_offsets_full,
            position_ids: args.position_ids,
            scalings: Some(dummy_scalings),
            is_full_pass: true,
            no_kv_cache: args.no_kv_cache,
            is_scaling_pass: scaling_pass,
            flash_params: args.flash_params_full,
        })?;

        let mut new_cache = Vec::new();
        for _ in 0..model.get_cache().full().xlora_lock().len() {
            new_cache.push(Some((
                Tensor::zeros((1,), DType::U8, &Device::Cpu)?,
                Tensor::zeros((1,), DType::U8, &Device::Cpu)?,
            )));
        }
        model.get_cache().full().lock().clone_from(&new_cache);

        res
    } else {
        model.inner_forward(XLoraPass {
            input_ids: args.input_ids,
            seqlen_offsets: args.seqlen_offsets,
            position_ids: args.position_ids,
            scalings: Some(dummy_scalings),
            is_full_pass: false,
            no_kv_cache: args.no_kv_cache,
            is_scaling_pass: scaling_pass,
            flash_params: args.flash_params,
        })?
    };

    let scalings = classifier.forward(hidden_states)?;
    if let Some(non_granular_state) = args.non_granular_state
        && *get_mut_arcmutex!(non_granular_state.non_granular_index)
            == non_granular_state.tgt_non_granular_index
    {
        *model.get_cache().full().get_scalings_cache() = Some(scalings.clone());
    }
    Ok(scalings)
}

pub fn verify_sanity_adapters(ordering: &Ordering, supported_layers: &[&str]) -> Result<()> {
    if ordering.layers.is_none() {
        return Ok(());
    }
    for path in ordering.layers.as_ref().unwrap().keys() {
        if !supported_layers.iter().any(|layer| path.ends_with(layer)) {
            inference_tensor::bail!(
                "Got a layer name `{path}` in the ordering, expected it to end with one of {supported_layers:?}"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Mutex};

    use inference_quant::ShardedSafeTensors;

    use super::*;
    use crate::kv_cache::Cache;

    const HIDDEN: usize = 4;
    const LAYERS: usize = 2;
    const ADAPTERS: usize = 2;
    const SCALING_PASS_VALUE: f64 = 0.5;

    #[derive(Debug, PartialEq)]
    struct Pass {
        tokens: usize,
        offset: usize,
        full: bool,
        scaled: bool,
        scaling_pass: Option<f64>,
        position_ids: Vec<usize>,
        no_kv_cache: bool,
        causal: bool,
    }

    struct Recorder {
        classifier: Option<XLoraClassifier>,
        cache: EitherCache,
        passes: Mutex<Vec<Pass>>,
    }

    impl Recorder {
        fn new(with_classifier: bool) -> Self {
            let config: XLoraConfig = serde_json::from_value(serde_json::json!({
                "hidden_size": HIDDEN,
                "base_model_id": "org/base",
                "adapters": ["a", "b"],
                "layerwise_scalings": false,
                "enable_relu_and_dropout": false,
                "xlora_depth": 1,
                "xlora_size": HIDDEN,
                "enable_softmax": true,
                "scaling_pass_value": SCALING_PASS_VALUE,
                "use_bias": false,
                "enable_softmax_topk": false,
            }))
            .unwrap();
            let weights = HashMap::from([(
                "last.weight".to_string(),
                Tensor::zeros((ADAPTERS, HIDDEN), DType::F32, &Device::Cpu).unwrap(),
            )]);
            let vb = ShardedSafeTensors::wrap(weights, DType::F32, Device::Cpu);
            Self {
                classifier: with_classifier
                    .then(|| XLoraClassifier::new(config, LAYERS, ADAPTERS, vb, false).unwrap()),
                cache: EitherCache::Full(Cache::new(LAYERS, true)),
                passes: Mutex::new(Vec::new()),
            }
        }
    }

    impl ScalingsMaker for Recorder {
        fn classifier(&self) -> Option<&XLoraClassifier> {
            self.classifier.as_ref()
        }
        fn dtype(&self) -> DType {
            DType::F32
        }
        fn get_cache(&self) -> &EitherCache {
            &self.cache
        }
        fn inner_forward(&self, pass: XLoraPass<'_>) -> Result<Tensor> {
            let (batch, tokens) = pass.input_ids.dims2()?;
            self.passes.lock().unwrap().push(Pass {
                tokens,
                offset: pass.seqlen_offsets[0],
                full: pass.is_full_pass,
                scaled: pass.scalings.is_some(),
                scaling_pass: pass.is_scaling_pass,
                position_ids: pass.position_ids.to_vec(),
                no_kv_cache: pass.no_kv_cache,
                causal: pass.flash_params.causal,
            });
            Tensor::zeros((batch, tokens, HIDDEN), DType::F32, &Device::Cpu)
        }
        fn lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
            Ok(hidden.clone())
        }
    }

    // One new token after a three-token prompt.
    fn run(model: &Recorder, no_kv_cache: bool) -> Vec<Pass> {
        let input_ids = Tensor::zeros((1, 1), DType::U32, &Device::Cpu).unwrap();
        let input_ids_full = Tensor::zeros((1, 4), DType::U32, &Device::Cpu).unwrap();
        // Told apart in the record: the full sequence's flash params are the non-causal ones.
        let (flash, flash_full) = (FlashParams::empty(true), FlashParams::empty(false));
        let tokens = if no_kv_cache { 4 } else { 1 };
        let logits = xlora_forward(
            model,
            XLoraForward {
                input_ids: &input_ids,
                input_ids_full: &input_ids_full,
                seqlen_offsets: &[3],
                seqlen_offsets_full: &[0],
                no_kv_cache,
                non_granular_state: &None,
                context_lens: vec![(tokens - 1, 1)],
                position_ids: &[3],
                flash_params: &flash,
                flash_params_full: &flash_full,
            },
        )
        .unwrap();
        assert_eq!(logits.dims(), [1, 1, HIDDEN]);
        std::mem::take(&mut *model.passes.lock().unwrap())
    }

    // Each pass sees the step's position ids and no_kv_cache; the full sequence starts at 0, the new token at 3.
    fn pass(
        tokens: usize,
        full: bool,
        scaled: bool,
        scaling_pass: Option<f64>,
        no_kv_cache: bool,
    ) -> Pass {
        let whole = tokens == 4;
        Pass {
            tokens,
            offset: if whole { 0 } else { 3 },
            full,
            scaled,
            scaling_pass,
            position_ids: vec![3],
            no_kv_cache,
            causal: !whole,
        }
    }

    #[test]
    fn without_a_classifier_one_unscaled_pass_runs_on_the_new_tokens() {
        let model = Recorder::new(false);
        assert_eq!(run(&model, false), [pass(1, false, false, None, false)]);
    }

    #[test]
    fn a_classifier_scores_a_scaling_pass_then_scales_the_new_tokens() {
        let model = Recorder::new(true);
        assert_eq!(
            run(&model, false),
            [
                pass(1, false, true, Some(SCALING_PASS_VALUE), false),
                pass(1, true, true, None, false)
            ]
        );
    }

    #[test]
    fn without_a_kv_cache_both_passes_run_over_the_whole_sequence() {
        let model = Recorder::new(true);
        assert_eq!(
            run(&model, true),
            [
                pass(4, true, true, Some(SCALING_PASS_VALUE), true),
                pass(4, true, true, None, true)
            ]
        );
    }

    #[test]
    fn a_full_pass_takes_the_xlora_caches_and_empties_them_only_without_a_kv_cache() {
        let cache = EitherCache::Full(Cache::new(LAYERS, true));
        let layer = || {
            let entry = Tensor::zeros((1,), DType::F32, &Device::Cpu).unwrap();
            Some((entry.clone(), entry))
        };
        cache.full().xlora_lock()[0] = layer();
        cache.full().lock()[1] = layer();
        assert!(pass_cache(&cache, true, false)[0].is_some());
        assert!(pass_cache(&cache, true, true)[0].is_none());
        assert!(pass_cache(&cache, false, true)[1].is_some());
    }
}
