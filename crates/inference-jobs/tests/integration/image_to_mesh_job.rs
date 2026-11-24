//! Image-to-mesh jobs: what a submission must hold, and a LATO.2 run on real weights that skips unless
//! `INFERENCE_TEST_LATO2_MODEL`, `INFERENCE_TEST_TRELLIS_LARGE_MODEL`, `INFERENCE_TEST_DINOV2_REG_MODEL` and
//! `INFERENCE_TEST_BIREFNET_MODEL` are the checkpoint folders and `INFERENCE_TEST_JOVOX_IMAGE` a photo of an object on
//! an opaque background.

use std::time::Duration;

use inference::{ModelDType, TextModelBuilder};
use inference_jobs::{
    ImageToMeshJob, JobEvent, JobInput, JobQueue, JobRequest, JobStatus, JobsConfig,
};

#[path = "../../../inference/tests/support/llama_tiny.rs"]
mod support;
use support::tiny_llama_checkpoint;

const JOB_WAIT: Duration = Duration::from_secs(900);
const VERTICES: u32 = 1000;
// the count only follows the one asked: the reference makes ~2600 for 2000 on its own example, ~1750 for 1000 here
const VERTEX_BAND: (f64, f64) = (0.5, 2.5);

#[tokio::test(flavor = "multi_thread")]
async fn an_image_to_mesh_job_takes_one_image() -> anyhow::Result<()> {
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
    let job = |job: ImageToMeshJob| JobRequest::ImageToMesh(job);
    let image = || JobInput::Bytes {
        name: "object.png".into(),
        data: Vec::new(),
    };
    let none = queue.submit(&model, job(ImageToMeshJob::default()), Vec::new());
    assert!(none.is_err_and(|e| e.to_string().contains("one image")));
    let two = queue.submit(
        &model,
        job(ImageToMeshJob::default()),
        vec![image(), image()],
    );
    assert!(two.is_err_and(|e| e.to_string().contains("one image")));
    let few = ImageToMeshJob {
        vertices: Some(10),
        ..Default::default()
    };
    let refused = queue.submit(&model, job(few), vec![image()]);
    assert!(refused.is_err_and(|e| e.to_string().contains("vertices must be")));
    let stepless = ImageToMeshJob {
        topology_steps: Some(0),
        ..Default::default()
    };
    let refused = queue.submit(&model, job(stepless), vec![image()]);
    assert!(refused.is_err_and(|e| e.to_string().contains("topology_steps must be")));
    let parsed: JobRequest =
        serde_json::from_value(serde_json::json!({"kind": "image_to_mesh", "vertices": 1500}))?;
    assert!(matches!(
        parsed,
        JobRequest::ImageToMesh(ImageToMeshJob {
            vertices: Some(1500),
            ..
        })
    ));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn an_image_to_mesh_job_writes_a_mesh() -> anyhow::Result<()> {
    let vars = [
        "INFERENCE_TEST_LATO2_MODEL",
        "INFERENCE_TEST_TRELLIS_LARGE_MODEL",
        "INFERENCE_TEST_DINOV2_REG_MODEL",
        "INFERENCE_TEST_BIREFNET_MODEL",
        "INFERENCE_TEST_JOVOX_IMAGE",
    ]
    .map(std::env::var_os);
    let [
        Some(lato2),
        Some(large),
        Some(dinov2),
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
        lato2: Some(lato2.into()),
        trellis_large: Some(large.into()),
        dinov2_reg: Some(dinov2.into()),
        birefnet: Some(birefnet.into()),
        ..Default::default()
    });
    let request = JobRequest::ImageToMesh(ImageToMeshJob {
        vertices: Some(VERTICES),
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
    let vertices = done.report["vertices"].as_f64().unwrap_or(0.0);
    let asked = f64::from(VERTICES);
    assert!(
        (VERTEX_BAND.0 * asked..=VERTEX_BAND.1 * asked).contains(&vertices),
        "{vertices} vertices for {asked} asked"
    );
    // a closed triangle mesh has about twice as many faces as vertices
    let faces = done.report["faces"].as_f64().unwrap_or(0.0);
    assert!(faces > vertices, "{faces} faces on {vertices} vertices");
    for name in ["model", "mesh (obj)", "conditioning image", "meta"] {
        let output = done.outputs.iter().find(|o| o.name == name);
        let output = output.ok_or_else(|| anyhow::anyhow!("no {name} output"))?;
        assert!(std::fs::metadata(queue.file(&model, &done.id, &output.file)?)?.len() > 0);
    }
    Ok(())
}
