//! A world job on real weights, panorama to trained splats; skips unless `INFERENCE_TEST_WORLDGEN_MODELS` is a folder
//! holding the checkpoints under their repository names and `INFERENCE_TEST_WORLDGEN_PANORAMA` a 2:1 panorama.

use std::time::Duration;

use inference::{ModelDType, TextModelBuilder};
use inference_jobs::{JobEvent, JobInput, JobQueue, JobRequest, JobStatus, JobsConfig, WorldJob};

#[path = "../../../inference/tests/support/llama_tiny.rs"]
mod support;
use support::tiny_llama_checkpoint;

const MODELS_ENV: &str = "INFERENCE_TEST_WORLDGEN_MODELS";
const PANORAMA_ENV: &str = "INFERENCE_TEST_WORLDGEN_PANORAMA";
const JOB_WAIT: Duration = Duration::from_secs(3600);
// enough steps for one preview render
const STEPS: usize = 1000;

#[tokio::test(flavor = "multi_thread")]
async fn a_world_job_runs_from_a_panorama_to_splats() -> anyhow::Result<()> {
    let (Some(models), Some(panorama)) =
        (std::env::var_os(MODELS_ENV), std::env::var_os(PANORAMA_ENV))
    else {
        return Ok(());
    };
    let models = std::path::PathBuf::from(models);
    let tiny = tiny_llama_checkpoint()?;
    let model = TextModelBuilder::new(tiny.path().to_string_lossy())
        .with_dtype(ModelDType::BF16)
        .build()
        .await?;
    let jobs = tempfile::tempdir()?;
    let queue = JobQueue::default();
    queue.configure(
        JobsConfig {
            dir: jobs.path().to_path_buf(),
            ..Default::default()
        }
        .with_models_root(&models),
    );
    let request = JobRequest::World(WorldJob {
        prompt: "A furnished room with carved wooden panels, a round window and a table.".into(),
        steps: Some(STEPS),
        ..Default::default()
    });
    let queued = queue.submit(&model, request, vec![JobInput::Path(panorama.into())])?;
    let (_, mut events) = queue.events(&model, &queued.id)?;
    tokio::time::timeout(JOB_WAIT, async {
        while !matches!(
            events.recv().await,
            Ok(JobEvent::Finished { .. }) | Err(tokio::sync::broadcast::error::RecvError::Closed)
        ) {}
    })
    .await?;

    let done = queue.get(&model, &queued.id)?;
    assert_eq!(done.status, JobStatus::Succeeded, "{:?}", done.error);
    let stages = done
        .stages
        .iter()
        .map(|s| s.name.as_str())
        .collect::<Vec<_>>();
    for stage in [
        "scene",
        "clips",
        "alignment",
        "splat data",
        "reconstruction",
        "training",
        "mesh",
    ] {
        assert!(stages.contains(&stage), "no {stage} stage in {stages:?}");
    }
    let clips = done
        .stages
        .iter()
        .find(|s| s.name == "clips")
        .expect("a clips stage");
    assert!(clips.total > 0 && clips.done == clips.total, "{clips:?}");
    for stage in ["scene", "clips", "training"] {
        assert!(
            done.previews.iter().any(|p| p.stage == stage),
            "no {stage} preview"
        );
    }
    for name in ["splats", "mesh"] {
        let output = done.outputs.iter().find(|o| o.name == name);
        let output = output.ok_or_else(|| anyhow::anyhow!("no {name} output"))?;
        assert!(std::fs::metadata(queue.file(&model, &done.id, &output.file)?)?.len() > 0);
    }
    Ok(())
}
