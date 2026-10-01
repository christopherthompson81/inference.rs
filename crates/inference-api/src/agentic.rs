//! Agentic tool-call policy and the broker that holds pending tool approvals until a client answers them.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
    time::Duration,
};

use inference_core::{
    AgentToolApproval, AgentToolApprovalAsyncCallback, AgentToolApprovalDecision,
    AgentToolApprovalNotifier, AgentToolApprovalRequest, Response,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc::Sender, oneshot};
use utoipa::ToSchema;

use crate::api_error::{ApiError, ApiErrorKind};

/// Server-level agentic defaults applied to requests that do not set their own.
#[derive(Clone, Default)]
pub struct AgenticDefaults {
    pub max_tool_rounds: Option<usize>,
    pub tool_dispatch_url: Option<String>,
    pub agent_permission: Option<inference_core::AgentPermission>,
    pub approval_broker: ApprovalBroker,
}

const APPROVAL_TIMEOUT: Duration = Duration::from_secs(300);
const APPROVAL_RESOLVED: &str = "resolved";
const APPROVAL_QUEUED: &str = "queued";

#[derive(Clone, Default)]
pub struct ApprovalBroker {
    inner: Arc<Mutex<ApprovalState>>,
}

#[derive(Default)]
struct ApprovalState {
    pending: HashMap<String, PendingApproval>,
    early_decisions: HashMap<String, ApprovalDecisionState>,
    /// Keyed by `sandbox_key(owner, session_id)`, as the agent loop keys its own.
    approved_sessions: HashSet<String>,
    /// Approvals announced to their requester, by who that requester acts for.
    notified: HashMap<String, Option<String>>,
}

struct PendingApproval {
    session_id: String,
    owner: Option<String>,
    tx: oneshot::Sender<AgentToolApprovalDecision>,
}

#[derive(Clone)]
struct ApprovalDecisionState {
    approve: bool,
    remember_for_session: bool,
    message: Option<String>,
}

impl ApprovalBroker {
    /// Waits on decisions for a request acting for `owner`; only the same owner can answer them.
    pub fn callback(&self, owner: Option<String>) -> AgentToolApprovalAsyncCallback {
        let broker = self.clone();
        Arc::new(move |approval| {
            let broker = broker.clone();
            let owner = owner.clone();
            Box::pin(async move { broker.wait_for_decision(approval, owner).await })
        })
    }

    pub fn notifier(
        &self,
        response: Sender<Response>,
        owner: Option<String>,
    ) -> Arc<AgentToolApprovalNotifier> {
        let broker = self.clone();
        Arc::new(move |approval| {
            broker.notify_approval_required(approval, response.clone(), owner.clone())
        })
    }

    fn notify_approval_required(
        &self,
        approval: AgentToolApprovalRequest,
        response: Sender<Response>,
        owner: Option<String>,
    ) {
        if self.is_session_approved(owner.as_deref(), &approval.session_id) {
            return;
        }

        let approval_id = approval.approval_id;
        self.inner
            .lock()
            .unwrap()
            .notified
            .insert(approval_id.clone(), owner.clone());
        let send_result = response.try_send(Response::AgenticToolApprovalRequired {
            approval_id: approval_id.clone(),
            session_id: approval.session_id,
            round: approval.round,
            tool: approval.tool,
            arguments: approval.arguments,
        });
        if send_result.is_err() {
            let _ = self.resolve(&approval_id, owner.as_deref(), false, false, None);
        }
    }

    async fn wait_for_decision(
        &self,
        approval: AgentToolApproval,
        owner: Option<String>,
    ) -> AgentToolApprovalDecision {
        if self.is_session_approved(owner.as_deref(), &approval.session_id) {
            return AgentToolApprovalDecision::approve();
        }

        let (tx, rx) = oneshot::channel();
        {
            let mut state = self.inner.lock().unwrap();
            if let Some(decision) = state.early_decisions.remove(&approval.approval_id) {
                if decision.approve && decision.remember_for_session {
                    let key = inference_core::sandbox_key(owner.as_deref(), &approval.session_id);
                    state.approved_sessions.insert(key);
                }
                return AgentToolApprovalDecision {
                    approve: decision.approve,
                    remember_for_session: decision.remember_for_session,
                    message: decision.message,
                };
            }
            state.pending.insert(
                approval.approval_id.clone(),
                PendingApproval {
                    session_id: approval.session_id.clone(),
                    owner,
                    tx,
                },
            );
        }

        let decision = tokio::time::timeout(APPROVAL_TIMEOUT, rx)
            .await
            .ok()
            .and_then(Result::ok)
            .unwrap_or_else(|| AgentToolApprovalDecision {
                approve: false,
                remember_for_session: false,
                message: Some("Approval timed out.".to_string()),
            });
        let mut state = self.inner.lock().unwrap();
        state.pending.remove(&approval.approval_id);
        state.notified.remove(&approval.approval_id);
        decision
    }

    /// Whether `approval_id` is waiting for a decision.
    #[doc(hidden)]
    pub fn is_pending(&self, approval_id: &str) -> bool {
        self.inner.lock().unwrap().pending.contains_key(approval_id)
    }

    /// Answers `owner`'s pending approval; a decision that arrives before the request is queued for it.
    pub fn resolve(
        &self,
        approval_id: &str,
        owner: Option<&str>,
        approve: bool,
        remember_for_session: bool,
        message: Option<String>,
    ) -> ApprovalResolveStatus {
        let mut state = self.inner.lock().unwrap();
        let theirs = |requester: &Option<String>| requester.as_deref() == owner;
        if state
            .pending
            .get(approval_id)
            .is_some_and(|pending| !theirs(&pending.owner))
        {
            return ApprovalResolveStatus::NotFound;
        }
        let Some(pending) = state.pending.remove(approval_id) else {
            if !state.notified.get(approval_id).is_some_and(theirs) {
                return ApprovalResolveStatus::NotFound;
            }
            state.notified.remove(approval_id);
            state.early_decisions.insert(
                approval_id.to_string(),
                ApprovalDecisionState {
                    approve,
                    remember_for_session,
                    message,
                },
            );
            return ApprovalResolveStatus::Queued;
        };

        state.notified.remove(approval_id);
        if approve && remember_for_session {
            let key = inference_core::sandbox_key(pending.owner.as_deref(), &pending.session_id);
            state.approved_sessions.insert(key);
        }
        let _ = pending.tx.send(AgentToolApprovalDecision {
            approve,
            remember_for_session,
            message,
        });
        ApprovalResolveStatus::Resolved
    }

    fn is_session_approved(&self, owner: Option<&str>, session_id: &str) -> bool {
        let key = inference_core::sandbox_key(owner, session_id);
        self.inner.lock().unwrap().approved_sessions.contains(&key)
    }
}

/// Answers the approval an `agentic_tool_approval_required` event named.
pub fn resolve_approval(
    broker: &ApprovalBroker,
    approval_id: &str,
    request: ApprovalDecisionRequest,
    owner: Option<&str>,
) -> Result<ApprovalDecisionResponse, ApiError> {
    let approve = matches!(request.decision, ApprovalDecision::Approve);
    let status = match broker.resolve(
        approval_id,
        owner,
        approve,
        request.remember_for_session,
        request.message,
    ) {
        ApprovalResolveStatus::Resolved => APPROVAL_RESOLVED,
        ApprovalResolveStatus::Queued => APPROVAL_QUEUED,
        ApprovalResolveStatus::NotFound => {
            return Err(ApiError::new(
                ApiErrorKind::NotFound,
                format!("Approval `{approval_id}` was not found."),
                Some("approval_not_found"),
                Some("approval_id"),
            ));
        }
    };
    Ok(ApprovalDecisionResponse { status })
}

/// Whether a decision reached its approval.
pub enum ApprovalResolveStatus {
    Resolved,
    Queued,
    NotFound,
}

/// Decision payload for a pending agentic tool approval.
#[derive(Deserialize, ToSchema)]
pub struct ApprovalDecisionRequest {
    pub decision: ApprovalDecision,
    /// Auto-approve all later tool calls in the same session.
    #[serde(default)]
    pub remember_for_session: bool,
    /// Optional note passed back to the model on denial.
    pub message: Option<String>,
}

#[derive(Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalDecision {
    Approve,
    Deny,
}

#[derive(Serialize, ToSchema)]
pub struct ApprovalDecisionResponse {
    /// "resolved" or "queued".
    pub status: &'static str,
}
