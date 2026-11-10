//! Image-to-3D jobs: what a submission must hold, and a TRELLIS.2 run on real weights that skips unless
//! `INFERENCE_TEST_TRELLIS2_MODEL`, `INFERENCE_TEST_TRELLIS_LARGE_MODEL`, `INFERENCE_TEST_DINOV3_MODEL` and
//! `INFERENCE_TEST_BIREFNET_MODEL` are the checkpoint folders and `INFERENCE_TEST_JOVOX_IMAGE` a photo of an object on
//! an opaque background.

use std::time::Duration;

use inference::{ModelDType, TextModelBuilder};
use inference_jobs::{
    ImageTo3dJob, JobEvent, JobInput, JobQueue, JobRequest, JobStatus, JobsConfig, Trellis2Variant,
};

#[path = "../../../inference/tests/support/llama_tiny.rs"]
mod support;
use support::tiny_llama_checkpoint;

const JOB_WAIT: Duration = Duration::from_secs(900);

#[tokio::test(flavor = "multi_thread")]
async fn an_image_to_3d_job_takes_one_image() -> anyhow::Result<()> {
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
    let job = || JobRequest::ImageTo3d(ImageTo3dJob::default());
    let none = queue.submit(&model, job(), Vec::new());
    assert!(none.is_err_and(|e| e.to_string().contains("one image")));
    let image = || JobInput::Bytes {
        name: "object.png".into(),
        data: Vec::new(),
    };
    let two = queue.submit(&model, job(), vec![image(), image()]);
    assert!(two.is_err_and(|e| e.to_string().contains("one image")));
    let parsed: JobRequest =
        serde_json::from_value(serde_json::json!({"kind": "image_to_3d", "variant": "512"}))?;
    assert!(matches!(
        parsed,
        JobRequest::ImageTo3d(ImageTo3dJob {
            variant: Some(Trellis2Variant::Res512),
            ..
        })
    ));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn an_image_to_3d_job_writes_a_textured_model() -> anyhow::Result<()> {
    let vars = [
        "INFERENCE_TEST_TRELLIS2_MODEL",
        "INFERENCE_TEST_TRELLIS_LARGE_MODEL",
        "INFERENCE_TEST_DINOV3_MODEL",
        "INFERENCE_TEST_BIREFNET_MODEL",
        "INFERENCE_TEST_JOVOX_IMAGE",
    ]
    .map(std::env::var_os);
    let [
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
        trellis2: Some(trellis2.into()),
        trellis_large: Some(large.into()),
        dinov3: Some(dinov3.into()),
        birefnet: Some(birefnet.into()),
        ..Default::default()
    });
    let request = JobRequest::ImageTo3d(ImageTo3dJob {
        variant: Some(Trellis2Variant::Res512),
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
    assert_eq!(done.report["resolution"], 512);
    let faces = done.report["faces"].as_u64().unwrap_or(0);
    assert!(faces > 0 && faces <= 1_000_000, "{faces} faces");
    for name in ["model", "mesh", "prepared image", "meta"] {
        let output = done.outputs.iter().find(|o| o.name == name);
        let output = output.ok_or_else(|| anyhow::anyhow!("no {name} output"))?;
        assert!(std::fs::metadata(queue.file(&model, &done.id, &output.file)?)?.len() > 0);
    }
    assert_eq!(done.report["texture_size"], 2048);
    let model_file = done
        .outputs
        .iter()
        .find(|o| o.name == "model")
        .expect("a model output");
    let glb = std::fs::read(queue.file(&model, &done.id, &model_file.file)?)?;
    let json_len = u32::from_le_bytes(glb[12..16].try_into()?) as usize;
    let gltf: serde_json::Value = serde_json::from_slice(&glb[20..20 + json_len])?;
    assert_eq!(
        gltf["images"].as_array().map(Vec::len),
        Some(2),
        "base color and metallic-roughness textures"
    );
    Ok(())
}
