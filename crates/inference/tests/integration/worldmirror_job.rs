//! A reconstruction job on real WorldMirror 2.0 weights; skips unless `INFERENCE_TEST_WORLDMIRROR_MODEL` is the
//! checkpoint folder and `INFERENCE_TEST_WORLDMIRROR_IMAGES` a folder of photos of one scene.

use std::time::Duration;

use inference::{
    JobEvent, JobInput, JobRequest, JobStatus, JobsConfig, ModelDType, ReconstructJob,
    TextModelBuilder,
};

#[path = "../support/llama_tiny.rs"]
mod support;
use support::tiny_llama_checkpoint;

const MODEL_ENV: &str = "INFERENCE_TEST_WORLDMIRROR_MODEL";
const IMAGES_ENV: &str = "INFERENCE_TEST_WORLDMIRROR_IMAGES";
const JOB_WAIT: Duration = Duration::from_secs(600);

#[tokio::test(flavor = "multi_thread")]
async fn a_reconstruct_job_writes_splats_points_and_a_depth_preview() -> anyhow::Result<()> {
    let (Some(checkpoint), Some(images)) =
        (std::env::var_os(MODEL_ENV), std::env::var_os(IMAGES_ENV))
    else {
        return Ok(());
    };
    let mut photos = std::fs::read_dir(&images)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "jpg" || e == "png"))
        .collect::<Vec<_>>();
    photos.sort();
    let tiny = tiny_llama_checkpoint()?;
    let model = TextModelBuilder::new(tiny.path().to_string_lossy())
        .with_dtype(ModelDType::BF16)
        .build()
        .await?;
    let jobs = tempfile::tempdir()?;
    model.configure_jobs(JobsConfig {
        dir: jobs.path().to_path_buf(),
        worldmirror: Some(checkpoint.into()),
        ..Default::default()
    });
    let inputs = photos.iter().cloned().map(JobInput::Path).collect();
    let queued = model.submit_job(JobRequest::Reconstruct(ReconstructJob::default()), inputs)?;
    let (_, mut events) = model.job_events(&queued.id)?;
    tokio::time::timeout(JOB_WAIT, async {
        while !matches!(events.recv().await, Ok(JobEvent::Finished { .. }) | Err(_)) {}
    })
    .await?;

    let done = model.job(&queued.id)?;
    assert_eq!(done.status, JobStatus::Succeeded, "{:?}", done.error);
    assert_eq!(done.report["frames"], photos.len());
    for name in ["splats", "points", "cameras"] {
        let output = done.outputs.iter().find(|o| o.name == name);
        let output = output.ok_or_else(|| anyhow::anyhow!("no {name} output"))?;
        assert!(std::fs::metadata(model.job_file(&done.id, &output.file)?)?.len() > 0);
    }
    assert_eq!(done.previews.len(), 1);
    Ok(())
}
