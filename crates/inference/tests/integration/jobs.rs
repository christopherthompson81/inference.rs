//! Jobs on the tiny Llama's engine: stages, previews and outputs recorded, cancellation, and a device held to itself.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use inference::{
    JobEvent, JobNeed, JobOutput, JobRecord, JobResult, JobStatus, JobsConfig, Model, ModelDType,
    ModelOperationRequest, ModelStatus, Placement, Progress, RequestBuilder, TextMessageRole,
    TextMessages, TextModelBuilder,
};

#[path = "../support/llama_tiny.rs"]
mod support;
#[path = "../support/traces.rs"]
mod traces;
use support::tiny_llama_checkpoint;
use traces::greedy;

const JOB_WAIT: Duration = Duration::from_secs(60);
const MAX_LEN: usize = 4;
// what a job that waits to be cancelled sleeps between checks
const CANCEL_POLL: Duration = Duration::from_millis(10);
// a model that took a request this recently is not idle yet
const IDLE: Duration = Duration::from_millis(1500);
// far below what the tiny model leaves free on any card
const SIZED_NEED_BYTES: u64 = 1 << 30;

async fn cpu_model() -> anyhow::Result<(Model, tempfile::TempDir)> {
    let checkpoint = tiny_llama_checkpoint()?;
    let model = TextModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    Ok((model, checkpoint))
}

fn use_dir(model: &Model) -> anyhow::Result<tempfile::TempDir> {
    let jobs = tempfile::tempdir()?;
    model.configure_jobs(JobsConfig {
        dir: jobs.path().to_path_buf(),
        ..Default::default()
    });
    Ok(jobs)
}

async fn finished(model: &Model, id: &str) -> anyhow::Result<JobRecord> {
    let (record, mut events) = model.job_events(id)?;
    if !record.status.is_finished() {
        tokio::time::timeout(JOB_WAIT, async {
            loop {
                match events.recv().await {
                    Ok(JobEvent::Finished { .. })
                    | Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    _ => {}
                }
            }
        })
        .await?;
    }
    Ok(model.job(id)?)
}

async fn started(model: &Model, id: &str) -> anyhow::Result<()> {
    let (record, mut events) = model.job_events(id)?;
    if record.stages.is_empty() {
        tokio::time::timeout(JOB_WAIT, async {
            while !matches!(events.recv().await, Ok(JobEvent::Stage { .. })) {}
        })
        .await?;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_job_records_its_stages_previews_and_outputs() -> anyhow::Result<()> {
    let (model, _checkpoint) = cpu_model().await?;
    let jobs = use_dir(&model)?;
    let queued = model.submit_task("probe", None, |ctx| {
        ctx.stage("first", 2);
        ctx.advance(1);
        let image = ctx.dir().join("look.png");
        image::RgbImage::new(2, 2).save(&image)?;
        ctx.preview("a look", &image);
        ctx.stage("second", 0);
        std::fs::write(ctx.dir().join("made.txt"), "made")?;
        Ok(JobResult {
            outputs: vec![JobOutput {
                name: "made".into(),
                file: "made.txt".into(),
            }],
            report: serde_json::json!({ "answer": 42 }),
        })
    })?;
    assert_eq!(queued.status, JobStatus::Queued);

    let done = finished(&model, &queued.id).await?;
    assert_eq!(done.status, JobStatus::Succeeded, "{:?}", done.error);
    let stages = done
        .stages
        .iter()
        .map(|s| (s.name.as_str(), s.done, s.total))
        .collect::<Vec<_>>();
    assert_eq!(stages, [("first", 2, 2), ("second", 0, 0)]);
    assert!(done.stages.iter().all(|s| s.finished_at.is_some()));
    assert_eq!(done.previews.len(), 1);
    assert_eq!(
        (
            done.previews[0].stage.as_str(),
            done.previews[0].file.as_str()
        ),
        ("first", "look.png")
    );
    assert_eq!(done.outputs[0].file, "made.txt");
    assert_eq!(done.report["answer"], 42);

    let saved: JobRecord =
        serde_json::from_slice(&std::fs::read(jobs.path().join(&done.id).join("job.json"))?)?;
    assert_eq!(saved.status, JobStatus::Succeeded);
    assert!(model.job_file(&done.id, "made.txt")?.is_file());
    assert!(model.job_file(&done.id, "../job.json").is_err());
    assert!(model.for_owner("someone else").job(&done.id).is_err());
    assert_eq!(model.jobs().data.len(), 1);

    model.delete_job(&done.id)?;
    assert!(!jobs.path().join(&done.id).exists());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_cancelled_job_stops_and_a_cancelled_queued_job_never_starts() -> anyhow::Result<()> {
    let (model, _checkpoint) = cpu_model().await?;
    let _jobs = use_dir(&model)?;
    let running = model.submit_task("waits", None, |ctx| {
        ctx.stage("waiting", 0);
        while !ctx.cancelled() {
            std::thread::sleep(CANCEL_POLL);
        }
        anyhow::bail!("{}", inference::api::reconstruction::CANCELLED)
    })?;
    let ran = Arc::new(AtomicBool::new(false));
    let queued = model.submit_task("queued", None, {
        let ran = ran.clone();
        move |_| {
            ran.store(true, Ordering::Relaxed);
            Ok(JobResult {
                outputs: Vec::new(),
                report: serde_json::Value::Null,
            })
        }
    })?;
    started(&model, &running.id).await?;
    model.cancel_job(&queued.id)?;
    model.cancel_job(&running.id)?;

    assert_eq!(
        finished(&model, &running.id).await?.status,
        JobStatus::Cancelled
    );
    assert_eq!(
        finished(&model, &queued.id).await?.status,
        JobStatus::Cancelled
    );
    assert!(!ran.load(Ordering::Relaxed), "the cancelled queued job ran");
    Ok(())
}

// A job on the device unloads the idle model and keeps it from loading until the job ends; a pinned model there makes
// a job fail instead.
#[tokio::test(flavor = "multi_thread")]
async fn a_device_job_evicts_the_idle_model_and_holds_the_device() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") || inference_tensor::Device::new_cuda(0).is_err() {
        return Ok(());
    }
    let checkpoint = tiny_llama_checkpoint()?;
    let model = TextModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::BF16)
        .build()
        .await?;
    let _jobs = use_dir(&model)?;
    let id = model
        .models()?
        .data
        .into_iter()
        .find(|m| m.default == Some(true))
        .map(|m| m.id)
        .ok_or_else(|| anyhow::anyhow!("no default model"))?;
    let request =
        || RequestBuilder::from(TextMessages::new().add_message(TextMessageRole::User, "hello"));
    greedy(&model, request(), MAX_LEN).await?;
    tokio::time::sleep(IDLE).await;

    let engine = (*model).clone();
    let model_id = id.clone();
    let held = model.submit_task("hold", Some(JobNeed::Device), move |ctx| {
        let op = || ModelOperationRequest {
            model_id: model_id.clone(),
        };
        let status = engine.model_status(op())?.status;
        let reload = ctx.block_on(engine.reload_model(op()));
        Ok(JobResult {
            outputs: Vec::new(),
            report: serde_json::json!({
                "unloaded": status == ModelStatus::Unloaded,
                "reload": reload.err().map(|e| e.to_string()),
            }),
        })
    })?;
    let done = finished(&model, &held.id).await?;
    assert_eq!(done.status, JobStatus::Succeeded, "{:?}", done.error);
    assert_eq!(done.report["unloaded"], true);
    let refused = done.report["reload"].as_str().unwrap_or_default();
    assert!(refused.contains("device is busy"), "{refused}");

    greedy(&model, request(), MAX_LEN).await?;
    let op = ModelOperationRequest {
        model_id: id.clone(),
    };
    assert_eq!(model.model_status(op)?.status, ModelStatus::Loaded);

    let spec = serde_json::json!({
        "model": {"Run": {"model_id": checkpoint.path(), "dtype": "bf16"}},
        "model_id": "pinned-copy",
        "pinned": true,
    });
    model.add_model_json(spec.to_string().as_bytes()).await?;
    let blocked = model.submit_task("blocked", Some(JobNeed::Device), |_| {
        anyhow::bail!("ran beside a pinned model")
    })?;
    let blocked = finished(&model, &blocked.id).await?;
    assert_eq!(blocked.status, JobStatus::Failed);
    let error = blocked.error.unwrap_or_default();
    assert!(error.contains("pinned"), "{error}");
    Ok(())
}

async fn cuda_model() -> anyhow::Result<Option<(Model, tempfile::TempDir, String)>> {
    if !cfg!(feature = "cuda") || inference_tensor::Device::new_cuda(0).is_err() {
        return Ok(None);
    }
    let checkpoint = tiny_llama_checkpoint()?;
    let model = TextModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::BF16)
        .build()
        .await?;
    let id = model
        .models()?
        .data
        .into_iter()
        .find(|m| m.default == Some(true))
        .map(|m| m.id)
        .ok_or_else(|| anyhow::anyhow!("no default model"))?;
    Ok(Some((model, checkpoint, id)))
}

// A job that needs only some memory runs beside the resident model, which keeps serving.
#[tokio::test(flavor = "multi_thread")]
async fn a_sized_job_runs_beside_the_resident_model() -> anyhow::Result<()> {
    let Some((model, _checkpoint, id)) = cuda_model().await? else {
        return Ok(());
    };
    let _jobs = use_dir(&model)?;
    let engine = (*model).clone();
    let model_id = id.clone();
    let sized = model.submit_task("sized", Some(JobNeed::Bytes(SIZED_NEED_BYTES)), move |_| {
        let status = engine
            .model_status(ModelOperationRequest { model_id })?
            .status;
        Ok(JobResult {
            outputs: Vec::new(),
            report: serde_json::json!({ "loaded": status == ModelStatus::Loaded }),
        })
    })?;
    let done = finished(&model, &sized.id).await?;
    assert_eq!(done.status, JobStatus::Succeeded, "{:?}", done.error);
    assert_eq!(
        done.report["loaded"], true,
        "the resident model was unloaded"
    );
    assert_eq!(done.device, Some(0));
    assert_eq!(
        model
            .model_status(ModelOperationRequest { model_id: id })?
            .status,
        ModelStatus::Loaded
    );
    Ok(())
}

// A whole-device job that may displace nothing waits while a model is loaded there, saying why, and runs once it
// unloads.
#[tokio::test(flavor = "multi_thread")]
async fn a_job_that_may_not_displace_waits_for_the_device_to_empty() -> anyhow::Result<()> {
    let Some((model, _checkpoint, id)) = cuda_model().await? else {
        return Ok(());
    };
    let jobs = tempfile::tempdir()?;
    model.configure_jobs(JobsConfig {
        dir: jobs.path().to_path_buf(),
        placement: Placement::NoEviction,
        ..Default::default()
    });
    let queued = model.submit_task("polite", Some(JobNeed::Device), |_| {
        Ok(JobResult {
            outputs: Vec::new(),
            report: serde_json::Value::Null,
        })
    })?;
    let (_, mut events) = model.job_events(&queued.id)?;
    let reason = tokio::time::timeout(JOB_WAIT, async {
        loop {
            if let Ok(JobEvent::Waiting { reason }) = events.recv().await {
                break reason;
            }
        }
    })
    .await?;
    assert!(reason.contains(&format!("`{id}`")), "{reason}");
    let record = model.job(&queued.id)?;
    assert_eq!(record.status, JobStatus::Queued);
    assert_eq!(record.waiting.as_deref(), Some(reason.as_str()));

    model.unload_model(ModelOperationRequest { model_id: id })?;
    let done = finished(&model, &queued.id).await?;
    assert_eq!(done.status, JobStatus::Succeeded, "{:?}", done.error);
    assert_eq!(done.waiting, None);
    Ok(())
}
