//! Refine-mesh jobs: what a submission must hold, and a Hi3D run on real weights (TRELLIS.2 making the coarse mesh from
//! an opaque photo, BiRefNet cutting it out) that skips unless `INFERENCE_TEST_HI3D_MODEL`,
//! `INFERENCE_TEST_DINOV2_LARGE_MODEL`, `INFERENCE_TEST_CLIP_LARGE_MODEL`, `INFERENCE_TEST_TRELLIS2_MODEL`,
//! `INFERENCE_TEST_TRELLIS_LARGE_MODEL`, `INFERENCE_TEST_DINOV3_MODEL` and `INFERENCE_TEST_BIREFNET_MODEL` are the
//! checkpoint folders and `INFERENCE_TEST_JOVOX_IMAGE` a photo of an object on an opaque background.

use std::time::Duration;

use inference::{ModelDType, TextModelBuilder};
use inference_jobs::{
    JobEvent, JobInput, JobQueue, JobRequest, JobStatus, JobsConfig, RefineMeshJob,
};

#[path = "../../../inference/tests/support/llama_tiny.rs"]
mod support;
use support::tiny_llama_checkpoint;

const JOB_WAIT: Duration = Duration::from_secs(900);
// a 512 refinement of a whole object decodes hundreds of thousands of cells; its surface has at least this many faces
const MIN_FACES: f64 = 50_000.0;

#[tokio::test(flavor = "multi_thread")]
async fn a_refine_mesh_job_takes_an_image_and_at_most_one_mesh() -> anyhow::Result<()> {
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
    let job = |job: RefineMeshJob| JobRequest::RefineMesh(job);
    let file = |name: &str| JobInput::Bytes {
        name: name.into(),
        data: Vec::new(),
    };
    let none = queue.submit(&model, job(RefineMeshJob::default()), Vec::new());
    assert!(none.is_err_and(|e| e.to_string().contains("one image")));
    let meshes_only = queue.submit(
        &model,
        job(RefineMeshJob::default()),
        vec![file("coarse.glb")],
    );
    assert!(meshes_only.is_err_and(|e| e.to_string().contains("one image")));
    let two_meshes = queue.submit(
        &model,
        job(RefineMeshJob::default()),
        vec![file("a.glb"), file("b.ply"), file("photo.png")],
    );
    assert!(two_meshes.is_err_and(|e| e.to_string().contains("at most one mesh")));
    let odd = RefineMeshJob {
        resolution: Some(768),
        ..Default::default()
    };
    let refused = queue.submit(&model, job(odd), vec![file("photo.png")]);
    assert!(refused.is_err_and(|e| e.to_string().contains("resolution must be")));
    let stepless = RefineMeshJob {
        steps: Some(0),
        ..Default::default()
    };
    let refused = queue.submit(
        &model,
        job(stepless),
        vec![file("photo.png"), file("coarse.obj")],
    );
    assert!(refused.is_err_and(|e| e.to_string().contains("steps must be")));
    let parsed: JobRequest = serde_json::from_value(
        serde_json::json!({"kind": "refine_mesh", "resolution": 1536, "texture": true}),
    )?;
    assert!(matches!(
        parsed,
        JobRequest::RefineMesh(RefineMeshJob {
            resolution: Some(1536),
            texture: Some(true),
            ..
        })
    ));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_refine_mesh_job_writes_a_refined_mesh() -> anyhow::Result<()> {
    let vars = [
        "INFERENCE_TEST_HI3D_MODEL",
        "INFERENCE_TEST_DINOV2_LARGE_MODEL",
        "INFERENCE_TEST_CLIP_LARGE_MODEL",
        "INFERENCE_TEST_TRELLIS2_MODEL",
        "INFERENCE_TEST_TRELLIS_LARGE_MODEL",
        "INFERENCE_TEST_DINOV3_MODEL",
        "INFERENCE_TEST_BIREFNET_MODEL",
        "INFERENCE_TEST_JOVOX_IMAGE",
    ]
    .map(std::env::var_os);
    let [
        Some(hi3d),
        Some(dinov2),
        Some(clip),
        Some(trellis2),
        Some(large),
        Some(dinov3),
        Some(birefnet),
        Some(photo),
    ] = vars
    else {
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
        hi3d: Some(hi3d.into()),
        dinov2_large: Some(dinov2.into()),
        clip_large: Some(clip.into()),
        trellis2: Some(trellis2.into()),
        trellis_large: Some(large.into()),
        dinov3: Some(dinov3.into()),
        birefnet: Some(birefnet.into()),
        ..Default::default()
    });
    let request = JobRequest::RefineMesh(RefineMeshJob {
        resolution: Some(512),
        ..Default::default()
    });
    let queued = queue.submit(&model, request, vec![JobInput::Path(photo.into())])?;
    let (_, mut events) = queue.events(&model, &queued.id)?;
    tokio::time::timeout(JOB_WAIT, async {
        while !matches!(events.recv().await, Ok(JobEvent::Finished { .. }) | Err(_)) {}
    })
    .await?;

    let done = queue.get(&model, &queued.id)?;
    assert_eq!(done.status, JobStatus::Succeeded, "{:?}", done.error);
    assert_eq!(done.report["background_removed"], true);
    assert_eq!(done.report["coarse_generated"], true);
    assert_eq!(done.report["passes"].as_array().map(Vec::len), Some(1));
    let faces = done.report["faces"].as_f64().unwrap_or(0.0);
    assert!(faces > MIN_FACES, "{faces} faces");
    for name in [
        "model",
        "mesh (ply)",
        "coarse mesh",
        "conditioning image",
        "meta",
    ] {
        let output = done.outputs.iter().find(|o| o.name == name);
        let output = output.ok_or_else(|| anyhow::anyhow!("no {name} output"))?;
        assert!(std::fs::metadata(queue.file(&model, &done.id, &output.file)?)?.len() > 0);
    }
    Ok(())
}
