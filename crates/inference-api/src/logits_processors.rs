//! Named logits processors a client registers once and selects per request by name.

use std::sync::Arc;

use candle_core::{DType, Tensor};
pub use inference_core::CustomLogitsProcessor;

use crate::registry::{Registered, Registry};

/// A processor that edits one step's f32 logits in place, given every token so far (the prompt's included).
pub fn in_place(
    edit: impl Fn(&mut [f32], &[u32]) -> Result<(), String> + Send + Sync + 'static,
) -> Arc<dyn CustomLogitsProcessor> {
    Arc::new(move |logits: &Tensor, context: &[u32]| {
        let mut values = logits
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        edit(&mut values, context).map_err(candle_core::Error::msg)?;
        Tensor::from_vec(values, logits.shape(), logits.device())?.to_dtype(logits.dtype())
    })
}

impl Registered for Arc<dyn CustomLogitsProcessor> {
    const WHAT: &'static str = "logits processor";
    const CONFLICT_CODE: &'static str = "logits_processor_conflict";
    const PARAM: &'static str = "logits_processors";
}

pub type LogitsProcessors = Registry<Arc<dyn CustomLogitsProcessor>>;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    #[test]
    fn an_in_place_edit_reaches_the_logits_and_its_failure_the_caller() -> candle_core::Result<()> {
        let logits = Tensor::new(&[1.0_f32, 2.0, 3.0], &candle_core::Device::Cpu)?;
        let banned = in_place(|logits, context| {
            logits[context[0] as usize] = f32::NEG_INFINITY;
            Ok(())
        });
        let edited = banned.apply(&logits, &[1])?.to_vec1::<f32>()?;
        assert_eq!(edited, [1.0, f32::NEG_INFINITY, 3.0]);
        let failing = in_place(|_, _| Err("host refused".to_string()));
        let error = failing
            .apply(&logits, &[])
            .err()
            .map(|error| error.to_string());
        assert!(error.is_some_and(|error| error.contains("host refused")));
        Ok(())
    }
}
