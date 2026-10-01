//! Named logits processors a client registers once and selects per request by name.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use candle_core::{DType, Tensor};
pub use inference_core::CustomLogitsProcessor;

use crate::api_error::{ApiError, ApiErrorKind};

const PARAM: &str = "logits_processors";

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

/// Shared by every clone of an engine, so a processor registered once serves all of its owners.
#[derive(Clone, Default)]
pub struct LogitsProcessors(Arc<RwLock<HashMap<String, Arc<dyn CustomLogitsProcessor>>>>);

impl LogitsProcessors {
    pub fn register(
        &self,
        name: String,
        processor: Arc<dyn CustomLogitsProcessor>,
    ) -> Result<(), ApiError> {
        if name.is_empty() {
            return Err(ApiError::invalid_request(
                "a logits processor needs a non-empty name",
            ));
        }
        let mut processors = self.0.write().expect("logits processor lock poisoned");
        if processors.contains_key(&name) {
            return Err(ApiError::new(
                ApiErrorKind::Conflict,
                format!("a logits processor named `{name}` is already registered"),
                Some("logits_processor_conflict"),
                Some("name"),
            ));
        }
        processors.insert(name, processor);
        Ok(())
    }

    pub fn unregister(&self, name: &str) -> Result<(), ApiError> {
        self.0
            .write()
            .expect("logits processor lock poisoned")
            .remove(name)
            .map(|_| ())
            .ok_or_else(|| {
                ApiError::new(
                    ApiErrorKind::NotFound,
                    format!("no logits processor named `{name}` is registered"),
                    None,
                    Some("name"),
                )
            })
    }

    /// The processors a request names, in its order; `None` when it names none.
    pub(crate) fn resolve(
        &self,
        names: Option<&[String]>,
    ) -> Result<Option<Vec<Arc<dyn CustomLogitsProcessor>>>, ApiError> {
        let Some(names) = names.filter(|names| !names.is_empty()) else {
            return Ok(None);
        };
        let processors = self.0.read().expect("logits processor lock poisoned");
        names
            .iter()
            .map(|name| {
                processors.get(name).cloned().ok_or_else(|| {
                    ApiError::new(
                        ApiErrorKind::InvalidRequest,
                        format!("no logits processor named `{name}` is registered"),
                        None,
                        Some(PARAM),
                    )
                })
            })
            .collect::<Result<_, _>>()
            .map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    fn identity() -> Arc<dyn CustomLogitsProcessor> {
        in_place(|_, _| Ok(()))
    }

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

    #[test]
    fn a_name_registers_once_and_resolves_until_unregistered() {
        let processors = LogitsProcessors::default();
        processors.register("plain".into(), identity()).unwrap();
        let again = processors.register("plain".into(), identity()).unwrap_err();
        assert_eq!(again.kind, ApiErrorKind::Conflict);
        let names = ["plain".to_string()];
        assert_eq!(processors.resolve(Some(&names)).unwrap().unwrap().len(), 1);
        processors.unregister("plain").unwrap();
        let unknown = processors
            .resolve(Some(&names))
            .err()
            .map(|error| error.kind);
        assert_eq!(unknown, Some(ApiErrorKind::InvalidRequest));
        assert_eq!(
            processors.unregister("plain").unwrap_err().kind,
            ApiErrorKind::NotFound
        );
    }

    #[test]
    fn a_request_naming_no_processors_gets_none() {
        let processors = LogitsProcessors::default();
        assert!(processors.resolve(None).unwrap().is_none());
        assert!(processors.resolve(Some(&[])).unwrap().is_none());
    }
}
