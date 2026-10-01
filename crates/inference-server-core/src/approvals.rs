use axum::extract::Path;

pub use crate::agentic::{ApprovalDecision, ApprovalDecisionRequest, ApprovalDecisionResponse};
use crate::handler_core::{ApiJson, ApiJsonRejection};
use crate::{
    handler_core::{ApiError, json_response, openai_error_response},
    types::OwnedEngine,
};

#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/agent/approvals/{approval_id}",
    params(("approval_id" = String, Path, description = "Approval ID from the approval-required SSE event")),
    request_body = ApprovalDecisionRequest,
    responses(
        (status = 200, description = "Decision applied or queued", body = ApprovalDecisionResponse),
        (status = 400, description = "Invalid decision payload"),
        (status = 404, description = "Unknown approval ID"),
        (status = 413, description = "Decision payload is too large"),
        (status = 415, description = "Unsupported content type"),
    )
))]
pub async fn resolve_agent_approval(
    OwnedEngine(engine): OwnedEngine,
    Path(approval_id): Path<String>,
    payload: Result<ApiJson<ApprovalDecisionRequest>, ApiJsonRejection>,
) -> axum::response::Response {
    decide(payload, |request| {
        engine.resolve_approval(&approval_id, request)
    })
}

fn decide(
    payload: Result<ApiJson<ApprovalDecisionRequest>, ApiJsonRejection>,
    resolve: impl FnOnce(ApprovalDecisionRequest) -> Result<ApprovalDecisionResponse, ApiError>,
) -> axum::response::Response {
    match payload {
        Ok(ApiJson(request)) => json_response(resolve(request)),
        Err(ApiJsonRejection(error)) => openai_error_response(error),
    }
}

#[cfg(test)]
mod tests {
    use axum::{
        body::{Body, to_bytes},
        extract::FromRequest,
        http::{Request as HttpRequest, StatusCode, header::CONTENT_TYPE},
    };
    use std::time::Duration;

    use inference_core::{
        AgentToolApproval, AgentToolApprovalRequest, AgentToolKind, AgentToolMetadata,
        AgentToolSource, Response,
    };

    use super::*;
    use crate::agentic::{ApprovalBroker, ApprovalResolveStatus, resolve_approval};

    const TEST_PENDING_WAIT_TIMEOUT: Duration = Duration::from_secs(1);
    const TEST_PENDING_WAIT_RETRY: Duration = Duration::from_millis(1);

    async fn error_body(response: axum::response::Response) -> serde_json::Value {
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    async fn approval_json_rejection(
        body: &'static str,
        content_type: Option<&'static str>,
    ) -> ApiJsonRejection {
        let mut builder = HttpRequest::builder();
        if let Some(content_type) = content_type {
            builder = builder.header(CONTENT_TYPE, content_type);
        }
        let request = builder.body(Body::from(body)).unwrap();
        match ApiJson::<ApprovalDecisionRequest>::from_request(request, &()).await {
            Ok(_) => panic!("expected JSON rejection"),
            Err(error) => error,
        }
    }

    async fn wait_for_pending(broker: &ApprovalBroker, approval_id: &str) {
        tokio::time::timeout(TEST_PENDING_WAIT_TIMEOUT, async {
            loop {
                if broker.is_pending(approval_id) {
                    return;
                }
                tokio::time::sleep(TEST_PENDING_WAIT_RETRY).await;
            }
        })
        .await
        .unwrap();
    }

    #[test]
    fn unknown_approval_id_is_not_found() {
        let broker = ApprovalBroker::default();

        assert!(matches!(
            broker.resolve("missing", None, true, false, None),
            ApprovalResolveStatus::NotFound
        ));
    }

    #[tokio::test]
    async fn approval_json_rejections_use_openai_errors() {
        let cases = [
            (
                approval_json_rejection("{", Some("application/json")).await,
                StatusCode::BAD_REQUEST,
                "malformed_json",
            ),
            (
                approval_json_rejection("{}", Some("application/json")).await,
                StatusCode::BAD_REQUEST,
                "invalid_request_body",
            ),
            (
                approval_json_rejection(r#"{"decision":"approve"}"#, None).await,
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "invalid_content_type",
            ),
        ];

        for (rejection, status, code) in cases {
            let response = decide(Err(rejection), |_| unreachable!("the body was rejected"));
            assert_eq!(response.status(), status);
            let body = error_body(response).await;
            assert_eq!(body["error"]["type"], "invalid_request_error");
            assert_eq!(body["error"]["code"], code);
        }
    }

    #[tokio::test]
    async fn unknown_approval_response_uses_openai_error() {
        let request = ApprovalDecisionRequest {
            decision: ApprovalDecision::Approve,
            remember_for_session: false,
            message: None,
        };
        let broker = ApprovalBroker::default();
        let response = decide(Ok(ApiJson(request)), |request| {
            resolve_approval(&broker, "missing", request, None)
        });

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = error_body(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "approval_not_found");
        assert_eq!(body["error"]["param"], "approval_id");
    }

    #[tokio::test]
    async fn only_the_requesting_owner_answers_an_approval() {
        let broker = ApprovalBroker::default();
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let notifier = broker.notifier(tx, Some("team-a".to_string()));
        notifier(AgentToolApprovalRequest {
            approval_id: "appr_owned".to_string(),
            session_id: "session".to_string(),
            round: 0,
            tool: AgentToolMetadata {
                source: AgentToolSource::BuiltIn,
                kind: AgentToolKind::CodeExecution,
                label: "Python code".to_string(),
            },
            arguments: serde_json::json!({"code": "print('hello')"}),
        });
        for other in [Some("team-b"), None] {
            assert!(matches!(
                broker.resolve("appr_owned", other, true, false, None),
                ApprovalResolveStatus::NotFound
            ));
        }
        assert!(matches!(
            broker.resolve("appr_owned", Some("team-a"), true, false, None),
            ApprovalResolveStatus::Queued
        ));
    }

    #[tokio::test]
    async fn early_http_decision_unblocks_callback() {
        let broker = ApprovalBroker::default();
        let approval_id = "appr_test".to_string();
        let session_id = "session".to_string();
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let notifier = broker.notifier(tx, None);

        notifier(AgentToolApprovalRequest {
            approval_id: approval_id.clone(),
            session_id: session_id.clone(),
            round: 0,
            tool: AgentToolMetadata {
                source: AgentToolSource::BuiltIn,
                kind: AgentToolKind::CodeExecution,
                label: "Python code".to_string(),
            },
            arguments: serde_json::json!({"code": "print('hello')"}),
        });

        assert!(matches!(
            rx.try_recv().unwrap(),
            Response::AgenticToolApprovalRequired { .. }
        ));
        assert!(matches!(
            broker.resolve(&approval_id, None, true, false, None),
            ApprovalResolveStatus::Queued
        ));

        let callback = broker.callback(None);
        assert!(
            callback(AgentToolApproval {
                approval_id,
                session_id,
                round: 0,
                tool: AgentToolMetadata {
                    source: AgentToolSource::BuiltIn,
                    kind: AgentToolKind::CodeExecution,
                    label: "Python code".to_string(),
                },
                arguments: serde_json::json!({"code": "print('hello')"}),
            })
            .await
            .approve
        );
    }

    #[tokio::test]
    async fn http_decision_resolves_waiting_callback() {
        let broker = ApprovalBroker::default();
        let approval_id = "appr_waiting".to_string();
        let session_id = "session".to_string();
        let callback = broker.callback(None);

        let decision_task = tokio::spawn({
            let approval_id = approval_id.clone();
            let session_id = session_id.clone();
            async move {
                callback(AgentToolApproval {
                    approval_id,
                    session_id,
                    round: 0,
                    tool: AgentToolMetadata {
                        source: AgentToolSource::BuiltIn,
                        kind: AgentToolKind::CodeExecution,
                        label: "Python code".to_string(),
                    },
                    arguments: serde_json::json!({"code": "print('hello')"}),
                })
                .await
            }
        });

        wait_for_pending(&broker, &approval_id).await;

        assert!(matches!(
            broker.resolve(&approval_id, None, true, true, None),
            ApprovalResolveStatus::Resolved
        ));
        let decision = decision_task.await.unwrap();
        assert!(decision.approve);
        assert!(decision.remember_for_session);
    }
}
