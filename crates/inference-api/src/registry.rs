//! Named entries a client registers on the engine once and selects per request by name.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use crate::api_error::{ApiError, ApiErrorKind};

/// What a registry holds, as its errors name it.
pub trait Registered: Clone {
    /// The entry in prose, e.g. "logits processor".
    const WHAT: &'static str;
    const CONFLICT_CODE: &'static str;
    /// The request field that selects entries.
    const PARAM: &'static str;
}

impl Registered for inference_core::ToolCallbackWithTool {
    const WHAT: &'static str = "host tool";
    const CONFLICT_CODE: &'static str = "host_tool_conflict";
    const PARAM: &'static str = "host_tools";
}

/// Tools a client registers after load, which a chat request offers the model by naming them.
pub type HostTools = Registry<inference_core::ToolCallbackWithTool>;

/// Shared by every clone of an engine, so an entry registered once serves all of its owners.
#[derive(Clone)]
pub struct Registry<T>(Arc<RwLock<HashMap<String, T>>>);

impl<T> Default for Registry<T> {
    fn default() -> Self {
        Self(Arc::default())
    }
}

impl<T: Registered> Registry<T> {
    pub fn register(&self, name: String, entry: T) -> Result<(), ApiError> {
        if name.is_empty() {
            return Err(ApiError::invalid_request(format!(
                "a {} needs a non-empty name",
                T::WHAT
            )));
        }
        let mut entries = self.0.write().expect("registry lock poisoned");
        if entries.contains_key(&name) {
            return Err(ApiError::new(
                ApiErrorKind::Conflict,
                format!("a {} named `{name}` is already registered", T::WHAT),
                Some(T::CONFLICT_CODE),
                Some("name"),
            ));
        }
        entries.insert(name, entry);
        Ok(())
    }

    pub fn unregister(&self, name: &str) -> Result<(), ApiError> {
        self.0
            .write()
            .expect("registry lock poisoned")
            .remove(name)
            .map(|_| ())
            .ok_or_else(|| {
                ApiError::new(
                    ApiErrorKind::NotFound,
                    format!("no {} named `{name}` is registered", T::WHAT),
                    None,
                    Some("name"),
                )
            })
    }

    /// The entries a request names, in its order; `None` when it names none.
    pub(crate) fn resolve(&self, names: Option<&[String]>) -> Result<Option<Vec<T>>, ApiError> {
        let Some(names) = names.filter(|names| !names.is_empty()) else {
            return Ok(None);
        };
        let entries = self.0.read().expect("registry lock poisoned");
        names
            .iter()
            .map(|name| {
                entries.get(name).cloned().ok_or_else(|| {
                    ApiError::new(
                        ApiErrorKind::InvalidRequest,
                        format!("no {} named `{name}` is registered", T::WHAT),
                        None,
                        Some(T::PARAM),
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

    #[derive(Clone)]
    struct Probe;

    impl Registered for Probe {
        const WHAT: &'static str = "probe";
        const CONFLICT_CODE: &'static str = "probe_conflict";
        const PARAM: &'static str = "probes";
    }

    #[test]
    fn a_name_registers_once_and_resolves_until_unregistered() {
        let registry = Registry::<Probe>::default();
        registry.register("plain".into(), Probe).unwrap();
        let again = registry.register("plain".into(), Probe).unwrap_err();
        assert_eq!(
            (again.kind, again.code.as_deref()),
            (ApiErrorKind::Conflict, Some("probe_conflict"))
        );
        let names = ["plain".to_string()];
        assert_eq!(registry.resolve(Some(&names)).unwrap().unwrap().len(), 1);
        registry.unregister("plain").unwrap();
        let unknown = registry.resolve(Some(&names)).err();
        assert_eq!(
            unknown.map(|error| (error.kind, error.param)),
            Some((ApiErrorKind::InvalidRequest, Some("probes".to_string())))
        );
        assert_eq!(
            registry.unregister("plain").unwrap_err().kind,
            ApiErrorKind::NotFound
        );
    }

    #[test]
    fn a_request_naming_nothing_gets_none() {
        let registry = Registry::<Probe>::default();
        assert!(registry.resolve(None).unwrap().is_none());
        assert!(registry.resolve(Some(&[])).unwrap().is_none());
    }
}
