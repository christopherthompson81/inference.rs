//! ## Background task management for the Responses API.
//!
//! This module handles background processing of responses when `background: true` is set.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{SystemTime, UNIX_EPOCH},
};

use inference_core::RequestCancellation;

use crate::responses_types::{
    ItemStatus, OutputItem, ResponseError, ResponseResource, ResponseStatus,
};

/// State of a background task
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum BackgroundTaskState {
    /// Task is queued
    Queued,
    /// Task is in progress
    InProgress,
    /// Task completed successfully
    Completed(ResponseResource),
    /// Task failed
    Failed(ResponseError),
    /// Task was cancelled; holds what it had generated once the stopped request comes back.
    Cancelled(Option<ResponseResource>),
}

/// A background task for processing responses
#[derive(Debug, Clone)]
pub struct BackgroundTask {
    /// Task ID (same as response ID)
    pub id: String,
    /// Current state
    pub state: BackgroundTaskState,
    /// Created timestamp
    pub created_at: u64,
    /// Model name
    pub model: String,
    /// Stops the task's request.
    pub cancellation: RequestCancellation,
}

impl BackgroundTask {
    /// Create a new background task
    pub fn new(id: String, model: String, cancellation: RequestCancellation) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id,
            state: BackgroundTaskState::Queued,
            created_at,
            model,
            cancellation,
        }
    }

    /// Convert the current task state to a ResponseResource
    pub fn to_response_resource(&self) -> ResponseResource {
        let mut resource =
            ResponseResource::new(self.id.clone(), self.model.clone(), self.created_at);

        match &self.state {
            BackgroundTaskState::Queued => {
                resource.status = ResponseStatus::Queued;
            }
            BackgroundTaskState::InProgress => {
                resource.status = ResponseStatus::InProgress;
            }
            BackgroundTaskState::Completed(resp) => {
                return resp.clone();
            }
            BackgroundTaskState::Failed(error) => {
                resource.status = ResponseStatus::Failed;
                resource.error = Some(error.clone());
            }
            BackgroundTaskState::Cancelled(Some(partial)) => {
                let mut partial = partial.clone();
                partial.status = ResponseStatus::Cancelled;
                // the reply was cut short; tool calls it finished stay completed
                for item in &mut partial.output {
                    if matches!(item, OutputItem::Message { .. }) {
                        item.set_status(ItemStatus::Incomplete);
                    }
                }
                return partial;
            }
            BackgroundTaskState::Cancelled(None) => {
                resource.status = ResponseStatus::Cancelled;
            }
        }

        resource
    }
}

/// Manager for background tasks
#[derive(Debug, Default)]
pub struct BackgroundTaskManager {
    /// Map of task ID to task
    tasks: Arc<RwLock<HashMap<String, BackgroundTask>>>,
}

impl BackgroundTaskManager {
    /// Create a new background task manager
    pub fn new() -> Self {
        Self {
            tasks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Queue a task under the id of the response it produces
    pub fn create_task(&self, id: String, model: String, cancellation: RequestCancellation) {
        let task = BackgroundTask::new(id.clone(), model, cancellation);
        self.tasks.write().unwrap().insert(id, task);
    }

    /// Get the current state of a task
    pub fn get_task(&self, id: &str) -> Option<BackgroundTask> {
        let tasks = self.tasks.read().unwrap();
        tasks.get(id).cloned()
    }

    /// Get the response resource for a task
    pub fn get_response(&self, id: &str) -> Option<ResponseResource> {
        let tasks = self.tasks.read().unwrap();
        tasks.get(id).map(|t| t.to_response_resource())
    }

    // A cancelled task stays cancelled when its stopped request comes back; it keeps what was generated.
    fn transition(&self, id: &str, state: BackgroundTaskState) -> bool {
        let mut tasks = self.tasks.write().unwrap();
        match tasks.get_mut(id) {
            Some(BackgroundTask {
                state: BackgroundTaskState::Cancelled(partial),
                ..
            }) => {
                if partial.is_none()
                    && let BackgroundTaskState::Completed(response) = state
                {
                    *partial = Some(response);
                }
                false
            }
            Some(task) => {
                task.state = state;
                true
            }
            None => false,
        }
    }

    /// Update task to in_progress state
    pub fn mark_in_progress(&self, id: &str) -> bool {
        self.transition(id, BackgroundTaskState::InProgress)
    }

    /// Update task to completed state
    pub fn mark_completed(&self, id: &str, response: ResponseResource) -> bool {
        self.transition(id, BackgroundTaskState::Completed(response))
    }

    /// Update task to failed state
    pub fn mark_failed(&self, id: &str, error: ResponseError) -> bool {
        self.transition(id, BackgroundTaskState::Failed(error))
    }

    /// Cancel a queued or running task; a finished one is left as it is
    pub fn cancel(&self, id: &str) -> bool {
        let mut tasks = self.tasks.write().unwrap();
        match tasks.get_mut(id) {
            Some(task)
                if matches!(
                    task.state,
                    BackgroundTaskState::Queued | BackgroundTaskState::InProgress
                ) =>
            {
                task.state = BackgroundTaskState::Cancelled(None);
                task.cancellation.cancel();
                true
            }
            _ => false,
        }
    }

    /// Delete a task, stopping its request if it is still running
    pub fn delete_task(&self, id: &str) -> bool {
        let mut tasks = self.tasks.write().unwrap();
        tasks
            .remove(id)
            .inspect(|task| task.cancellation.cancel())
            .is_some()
    }
}

/// Global background task manager
static BACKGROUND_TASK_MANAGER: std::sync::LazyLock<BackgroundTaskManager> =
    std::sync::LazyLock::new(BackgroundTaskManager::new);

/// Get the global background task manager
pub fn get_background_task_manager() -> &'static BackgroundTaskManager {
    &BACKGROUND_TASK_MANAGER
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_get_task() {
        let manager = BackgroundTaskManager::new();
        let id = "resp_test".to_string();
        manager.create_task(
            id.clone(),
            "test-model".to_string(),
            RequestCancellation::default(),
        );

        let task = manager.get_task(&id).unwrap();
        assert_eq!(task.id, id);
        assert!(matches!(task.state, BackgroundTaskState::Queued));
    }

    #[test]
    fn test_task_state_transitions() {
        let manager = BackgroundTaskManager::new();
        let id = "resp_test".to_string();
        manager.create_task(
            id.clone(),
            "test-model".to_string(),
            RequestCancellation::default(),
        );

        // Move to in_progress
        assert!(manager.mark_in_progress(&id));
        let task = manager.get_task(&id).unwrap();
        assert!(matches!(task.state, BackgroundTaskState::InProgress));

        // Mark completed
        let response = ResponseResource::new(id.clone(), "test-model".to_string(), 0);
        assert!(manager.mark_completed(&id, response));
        let task = manager.get_task(&id).unwrap();
        assert!(matches!(task.state, BackgroundTaskState::Completed(_)));
        assert!(!manager.cancel(&id), "a finished task cannot be cancelled");
    }

    #[test]
    fn test_cancel_task() {
        let manager = BackgroundTaskManager::new();
        let id = "resp_test".to_string();
        manager.create_task(
            id.clone(),
            "test-model".to_string(),
            RequestCancellation::default(),
        );

        assert!(manager.cancel(&id));
        let task = manager.get_task(&id).unwrap();
        assert!(matches!(task.state, BackgroundTaskState::Cancelled(None)));
        assert!(
            task.cancellation.is_canceled(),
            "cancelling stops the request"
        );

        // The stopped request coming back keeps the cancellation, with what it generated
        let response = ResponseResource::new(id.clone(), "test-model".to_string(), 0);
        assert!(!manager.mark_completed(&id, response));
        let task = manager.get_task(&id).unwrap();
        assert!(matches!(
            task.state,
            BackgroundTaskState::Cancelled(Some(_))
        ));
        let resource = manager.get_response(&id).unwrap();
        assert_eq!(resource.status, ResponseStatus::Cancelled);
    }
}
