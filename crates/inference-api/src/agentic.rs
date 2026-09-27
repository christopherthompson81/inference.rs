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

/// Server-level agentic defaults applied to requests that do not set their own.
#[derive(Clone, Default)]
pub struct AgenticDefaults {
    pub max_tool_rounds: Option<usize>,
    pub tool_dispatch_url: Option<String>,
    pub agent_permission: Option<inference_core::AgentPermission>,
    pub approval_broker: ApprovalBroker,
}

const APPROVAL_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Clone, Default)]
pub struct ApprovalBroker {
    inner: Arc<Mutex<ApprovalState>>,
}

#[derive(Default)]
struct ApprovalState {
    pending: HashMap<String, PendingApproval>,
    early_decisions: HashMap<String, ApprovalDecisionState>,
    approved_sessions: HashSet<String>,
    notified: HashSet<String>,
}

struct PendingApproval {
    session_id: String,
    tx: oneshot::Sender<AgentToolApprovalDecision>,
}

#[derive(Clone)]
struct ApprovalDecisionState {
    approve: bool,
    remember_for_session: bool,
    message: Option<String>,
}

impl ApprovalBroker {
    pub fn callback(&self) -> AgentToolApprovalAsyncCallback {
        let broker = self.clone();
        Arc::new(move |approval| {
            let broker = broker.clone();
            Box::pin(async move { broker.wait_for_decision(approval).await })
        })
    }

    pub fn notifier(&self, response: Sender<Response>) -> Arc<AgentToolApprovalNotifier> {
        let broker = self.clone();
        Arc::new(move |approval| broker.notify_approval_required(approval, response.clone()))
    }

    fn notify_approval_required(
        &self,
        approval: AgentToolApprovalRequest,
        response: Sender<Response>,
    ) {
        if self.is_session_approved(&approval.session_id) {
            return;
        }

        let approval_id = approval.approval_id;
        self.inner
            .lock()
            .unwrap()
            .notified
            .insert(approval_id.clone());
        let send_result = response.try_send(Response::AgenticToolApprovalRequired {
            approval_id: approval_id.clone(),
            session_id: approval.session_id,
            round: approval.round,
            tool: approval.tool,
            arguments: approval.arguments,
        });
        if send_result.is_err() {
            let _ = self.resolve(&approval_id, false, false, None);
        }
    }

    async fn wait_for_decision(&self, approval: AgentToolApproval) -> AgentToolApprovalDecision {
        if self.is_session_approved(&approval.session_id) {
            return AgentToolApprovalDecision::approve();
        }

        let (tx, rx) = oneshot::channel();
        {
            let mut state = self.inner.lock().unwrap();
            if let Some(decision) = state.early_decisions.remove(&approval.approval_id) {
                if decision.approve && decision.remember_for_session {
                    state.approved_sessions.insert(approval.session_id.clone());
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

    /// Answers a pending approval; a decision that arrives before the request is queued for it.
    pub fn resolve(
        &self,
        approval_id: &str,
        approve: bool,
        remember_for_session: bool,
        message: Option<String>,
    ) -> ApprovalResolveStatus {
        let mut state = self.inner.lock().unwrap();
        let Some(pending) = state.pending.remove(approval_id) else {
            if !state.notified.remove(approval_id) {
                return ApprovalResolveStatus::NotFound;
            }
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
            state.approved_sessions.insert(pending.session_id);
        }
        let _ = pending.tx.send(AgentToolApprovalDecision {
            approve,
            remember_for_session,
            message,
        });
        ApprovalResolveStatus::Resolved
    }

    fn is_session_approved(&self, session_id: &str) -> bool {
        self.inner
            .lock()
            .unwrap()
            .approved_sessions
            .contains(session_id)
    }
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
