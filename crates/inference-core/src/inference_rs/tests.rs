use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn empty_state() -> InferenceRs {
    InferenceRs {
        engines: RwLock::new(HashMap::new()),
        unloaded_models: RwLock::new(HashMap::new()),
        reloading_models: RwLock::new(HashSet::new()),
        default_engine_id: RwLock::new(None),
        model_aliases: RwLock::new(HashMap::new()),
        log: None,
        id: "test".to_string(),
        creation_time: 0,
        next_request_id: Mutex::new(RefCell::new(1)),
    }
}

#[test]
fn missing_default_sender_is_model_not_found() {
    assert!(matches!(
        empty_state().get_sender(None),
        Err(InferenceRsError::ModelNotFound(model)) if model == "default"
    ));
    assert!(matches!(
        empty_state().get_sender(Some("wrong-model")),
        Err(InferenceRsError::ModelNotFound(model)) if model == "wrong-model"
    ));
}

#[tokio::test(flavor = "multi_thread")]
async fn failed_reloads_leave_no_reloading_mark() {
    let state = empty_state();
    assert!(matches!(
        state.reload_model("model").await,
        Err(InferenceRsError::ModelNotFound(model)) if model == "model"
    ));
    assert!(state.reloading_models.read().unwrap().is_empty());

    state
        .reloading_models
        .write()
        .unwrap()
        .insert("model".to_string());
    assert!(matches!(
        state.reload_model("model").await,
        Err(InferenceRsError::ModelReloading(_))
    ));
    assert!(
        state.reloading_models.read().unwrap().contains("model"),
        "a refused reload leaves the running one's mark alone"
    );
}

#[test]
fn reloading_sender_preserves_model_state_error() {
    let state = empty_state();
    state
        .reloading_models
        .write()
        .unwrap()
        .insert("model".to_string());
    assert!(matches!(
        state.get_sender(Some("model")),
        Err(InferenceRsError::ModelReloading(model)) if model == "model"
    ));
}

#[test]
fn fallible_file_helpers_preserve_poisoned_engine_error() {
    let state = empty_state();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let _guard = state.engines.write().unwrap();
        panic!("poison engines lock");
    }));
    assert!(result.is_err());

    assert!(matches!(
        state.try_find_file("file-id"),
        Err(InferenceRsError::EnginePoisoned)
    ));
    assert!(matches!(
        state.try_list_files(),
        Err(InferenceRsError::EnginePoisoned)
    ));
    assert!(matches!(
        state.try_remove_file("file-id"),
        Err(InferenceRsError::EnginePoisoned)
    ));
}
