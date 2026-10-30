//! The jobs routes: only admins start a job; owners list, follow and fetch theirs.

use std::net::SocketAddr;
use std::time::Duration;

use axum::{
    Router,
    body::Body,
    extract::connect_info::MockConnectInfo,
    http::{Request, StatusCode, header},
};
use inference_api::Engine;
use inference_jobs::{JobEvent, JobOutput, JobQueue, JobResult, JobsConfig, Progress};
use inference_server_core::auth::Peer;
use inference_server_core::inference_server_router_builder::InferenceRsServerRouterBuilder;
use serde_json::{Value, json};
use tower::ServiceExt;

use crate::chat_route::body_text;

const LOOPBACK: [u8; 4] = [127, 0, 0, 1];
const LAN: [u8; 4] = [192, 168, 1, 20];
const PORT: u16 = 40000;
const BOUNDARY: &str = "job-boundary";
const JOB_WAIT: Duration = Duration::from_secs(30);

async fn router(engine: &Engine, jobs: &JobQueue, peer: [u8; 4]) -> anyhow::Result<Router> {
    Ok(InferenceRsServerRouterBuilder::new()
        .with_engine(engine)
        .with_jobs(jobs.clone())
        .build()
        .await?
        .layer(MockConnectInfo(Peer::from(SocketAddr::from((peer, PORT))))))
}

fn submission(request: Value) -> Request<Body> {
    let body = format!(
        "--{BOUNDARY}\r\ncontent-disposition: form-data; name=\"request\"\r\n\r\n{request}\r\n\
         --{BOUNDARY}\r\ncontent-disposition: form-data; name=\"file\"; filename=\"photo.png\"\r\n\
         content-type: image/png\r\n\r\nnot really a png\r\n--{BOUNDARY}--\r\n"
    );
    Request::post("/v1/jobs")
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(Body::from(body))
        .unwrap()
}

fn get(path: &str) -> Request<Body> {
    Request::get(path).body(Body::empty()).unwrap()
}

async fn json(app: &Router, request: Request<Body>) -> anyhow::Result<(StatusCode, Value)> {
    let response = app.clone().oneshot(request).await?;
    let status = response.status();
    let text = body_text(response).await?;
    Ok((status, serde_json::from_str(&text).unwrap_or(Value::Null)))
}

#[tokio::test(flavor = "multi_thread")]
async fn admins_start_jobs_and_owners_follow_and_fetch_them() -> anyhow::Result<()> {
    let engine = Engine::load(serde_json::from_value(
        json!({"runtime": {"device": "cpu"}}),
    )?)
    .await?;
    let jobs_dir = tempfile::tempdir()?;
    let jobs = JobQueue::default();
    jobs.configure(JobsConfig {
        dir: jobs_dir.path().to_path_buf(),
        worldmirror_repo: None,
        ..Default::default()
    });
    let local = router(&engine, &jobs, LOOPBACK).await?;
    let lan = router(&engine, &jobs, LAN).await?;

    let request = json!({"kind": "reconstruct"});
    let (status, _) = json(&lan, submission(request.clone())).await?;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, body) = json(&local, submission(request)).await?;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "job_not_configured", "{body}");

    let queued = jobs.submit_task(&engine, "probe", None, |ctx| {
        ctx.stage("drawing", 0);
        let look = ctx.dir().join("look.png");
        image::RgbImage::new(2, 2).save(&look)?;
        ctx.preview("a look", &look);
        std::fs::write(ctx.dir().join("notes.txt"), "not served")?;
        Ok(JobResult {
            outputs: vec![JobOutput {
                name: "look".into(),
                file: "look.png".into(),
            }],
            report: Value::Null,
        })
    })?;
    let (_, mut changes) = jobs.events(&engine, &queued.id)?;
    tokio::time::timeout(JOB_WAIT, async {
        while !matches!(changes.recv().await, Ok(JobEvent::Finished { .. }) | Err(_)) {}
    })
    .await?;

    let (status, listed) = json(&local, get("/v1/jobs")).await?;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(listed["data"][0]["id"], queued.id.as_str());
    let (status, record) = json(&local, get(&format!("/v1/jobs/{}", queued.id))).await?;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(record["status"], "succeeded");
    assert_eq!(record["previews"][0]["file"], "look.png");

    let image = local
        .clone()
        .oneshot(get(&format!("/v1/jobs/{}/files/look.png", queued.id)))
        .await?;
    assert_eq!(image.status(), StatusCode::OK);
    assert_eq!(image.headers()[header::CONTENT_TYPE], "image/png");
    for refused in ["notes.txt", "../job.json", "missing.png"] {
        let response = local
            .clone()
            .oneshot(get(&format!("/v1/jobs/{}/files/{refused}", queued.id)))
            .await?;
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{refused}");
    }

    // a finished job's stream is its record, then it ends
    let response = local
        .clone()
        .oneshot(get(&format!("/v1/jobs/{}/events", queued.id)))
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    let text = body_text(response).await?;
    assert!(text.starts_with("event: record\n"), "{text}");
    assert_eq!(text.matches("data:").count(), 1, "{text}");

    let response = local
        .clone()
        .oneshot(Request::delete(format!("/v1/jobs/{}", queued.id)).body(Body::empty())?)
        .await?;
    assert_eq!(response.status(), StatusCode::NO_CONTENT);
    let (status, _) = json(&local, get(&format!("/v1/jobs/{}", queued.id))).await?;
    assert_eq!(status, StatusCode::NOT_FOUND);
    Ok(())
}
