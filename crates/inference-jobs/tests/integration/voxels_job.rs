//! Voxel jobs: what a submission must hold, and a run on real weights that skips unless `INFERENCE_TEST_JOVOX_MODEL`,
//! `INFERENCE_TEST_DINOV3_MODEL` and `INFERENCE_TEST_BIREFNET_MODEL` are the checkpoint folders and
//! `INFERENCE_TEST_JOVOX_IMAGE` a photo of an object on an opaque background.

use std::time::Duration;

use inference::{ModelDType, TextModelBuilder};
use inference_jobs::{JobEvent, JobInput, JobQueue, JobRequest, JobStatus, JobsConfig, VoxelsJob};

#[path = "../../../inference/tests/support/llama_tiny.rs"]
mod support;
use support::tiny_llama_checkpoint;

const JOB_WAIT: Duration = Duration::from_secs(600);
// a sampler step count outside the trained 36..=108
const UNTRAINED_STEPS: usize = 10;

fn image() -> JobInput {
    JobInput::Bytes {
        name: "object.png".into(),
        data: Vec::new(),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_voxel_job_takes_one_image_and_trained_settings() -> anyhow::Result<()> {
    let tiny = tiny_llama_checkpoint()?;
    let model = TextModelBuilder::new(tiny.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let jobs = tempfile::tempdir()?;
    let queue = JobQueue::default();
    queue.configure(JobsConfig {
        dir: jobs.path().to_path_buf(),
        ..Default::default()
    });
    let job = |job: VoxelsJob| JobRequest::Voxels(job);
    let no_image = queue.submit(&model, job(VoxelsJob::default()), Vec::new());
    assert!(no_image.is_err_and(|e| e.to_string().contains("one image")));
    let two = queue.submit(&model, job(VoxelsJob::default()), vec![image(), image()]);
    assert!(two.is_err_and(|e| e.to_string().contains("one image")));
    let steps = VoxelsJob {
        steps: Some(UNTRAINED_STEPS),
        ..Default::default()
    };
    let rejected = queue.submit(&model, job(steps), vec![image()]);
    assert!(rejected.is_err_and(|e| e.to_string().contains("steps")));
    let guidance = VoxelsJob {
        cfg_strength: Some(0.5),
        ..Default::default()
    };
    assert!(queue.submit(&model, job(guidance), vec![image()]).is_err());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_voxel_job_cuts_out_the_object_and_writes_its_structure() -> anyhow::Result<()> {
    let vars = [
        "INFERENCE_TEST_JOVOX_MODEL",
        "INFERENCE_TEST_DINOV3_MODEL",
        "INFERENCE_TEST_BIREFNET_MODEL",
        "INFERENCE_TEST_JOVOX_IMAGE",
    ]
    .map(std::env::var_os);
    let [Some(jovox), Some(dinov3), Some(birefnet), Some(photo)] = vars else {
        return Ok(());
    };
    let tiny = tiny_llama_checkpoint()?;
    let model = TextModelBuilder::new(tiny.path().to_string_lossy())
        .with_dtype(ModelDType::BF16)
        .build()
        .await?;
    let jobs = tempfile::tempdir()?;
    let queue = JobQueue::default();
    queue.configure(JobsConfig {
        dir: jobs.path().to_path_buf(),
        jovox: Some(jovox.into()),
        dinov3: Some(dinov3.into()),
        birefnet: Some(birefnet.into()),
        ..Default::default()
    });
    let queued = queue.submit(
        &model,
        JobRequest::Voxels(VoxelsJob::default()),
        vec![JobInput::Path(photo.into())],
    )?;
    let (_, mut events) = queue.events(&model, &queued.id)?;
    tokio::time::timeout(JOB_WAIT, async {
        while !matches!(events.recv().await, Ok(JobEvent::Finished { .. }) | Err(_)) {}
    })
    .await?;

    let done = queue.get(&model, &queued.id)?;
    assert_eq!(done.status, JobStatus::Succeeded, "{:?}", done.error);
    assert_eq!(done.report["background_removed"], true);
    assert!(done.report["voxels"].as_u64().is_some_and(|n| n > 0));
    for name in ["voxels", "prepared image", "meta"] {
        let output = done.outputs.iter().find(|o| o.name == name);
        let output = output.ok_or_else(|| anyhow::anyhow!("no {name} output"))?;
        assert!(std::fs::metadata(queue.file(&model, &done.id, &output.file)?)?.len() > 0);
    }
    assert_eq!(done.previews.len(), 1);
    Ok(())
}
