//! Image-to-splat jobs: what a submission must hold, and an Idol run on real weights that skips unless
//! `INFERENCE_TEST_IDOL_MODEL` is Idol's folder, `INFERENCE_TEST_DINOV3_HPLUS_MODEL` and `INFERENCE_TEST_FLUX2_VAE` the
//! DINOv3 ViT-H+/16 and FLUX.2 VAE safetensors, and `INFERENCE_TEST_JOVOX_IMAGE` a photo of an object on an opaque
//! background.

use std::time::Duration;

use inference::{ModelDType, TextModelBuilder};
use inference_jobs::{
    ImageToSplatJob, JobEvent, JobInput, JobQueue, JobRequest, JobStatus, JobsConfig,
    SplatImageMode,
};

#[path = "../../../inference/tests/support/llama_tiny.rs"]
mod support;
use support::tiny_llama_checkpoint;

const JOB_WAIT: Duration = Duration::from_secs(900);
const GAUSSIANS: u32 = 65_536;
const STEPS: u32 = 12;
// a 3DGS PLY row: position, normal, 3 dc, opacity, 3 scales, 4 rotations
const PLY_ROW_BYTES: u64 = 17 * 4;

#[tokio::test(flavor = "multi_thread")]
async fn an_image_to_splat_job_takes_one_image() -> anyhow::Result<()> {
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
    let job = |job: ImageToSplatJob| JobRequest::ImageToSplat(job);
    let image = || JobInput::Bytes {
        name: "object.png".into(),
        data: Vec::new(),
    };
    let none = queue.submit(&model, job(ImageToSplatJob::default()), Vec::new());
    assert!(none.is_err_and(|e| e.to_string().contains("one image")));
    let two = queue.submit(
        &model,
        job(ImageToSplatJob::default()),
        vec![image(), image()],
    );
    assert!(two.is_err_and(|e| e.to_string().contains("one image")));
    let few = ImageToSplatJob {
        gaussians: Some(1000),
        ..Default::default()
    };
    let refused = queue.submit(&model, job(few), vec![image()]);
    assert!(refused.is_err_and(|e| e.to_string().contains("gaussians must be")));
    let stepless = ImageToSplatJob {
        steps: Some(0),
        ..Default::default()
    };
    let refused = queue.submit(&model, job(stepless), vec![image()]);
    assert!(refused.is_err_and(|e| e.to_string().contains("steps must be")));
    let unshifted = ImageToSplatJob {
        rescale_t: Some(0.0),
        ..Default::default()
    };
    let refused = queue.submit(&model, job(unshifted), vec![image()]);
    assert!(refused.is_err_and(|e| e.to_string().contains("rescale_t")));
    let huge = ImageToSplatJob {
        print_size_mm: Some(5000.0),
        ..Default::default()
    };
    let refused = queue.submit(&model, job(huge), vec![image()]);
    assert!(refused.is_err_and(|e| e.to_string().contains("print_size_mm must be")));
    let odd_texture = ImageToSplatJob {
        texture_size: Some(1000),
        ..Default::default()
    };
    let refused = queue.submit(&model, job(odd_texture), vec![image()]);
    assert!(refused.is_err_and(|e| e.to_string().contains("texture_size must be")));
    let parsed: JobRequest =
        serde_json::from_value(serde_json::json!({"kind": "image_to_splat", "mode": "real"}))?;
    assert!(matches!(
        parsed,
        JobRequest::ImageToSplat(ImageToSplatJob {
            mode: Some(SplatImageMode::Real),
            ..
        })
    ));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn an_image_to_splat_job_writes_splats() -> anyhow::Result<()> {
    let vars = [
        "INFERENCE_TEST_IDOL_MODEL",
        "INFERENCE_TEST_DINOV3_HPLUS_MODEL",
        "INFERENCE_TEST_FLUX2_VAE",
        "INFERENCE_TEST_JOVOX_IMAGE",
    ]
    .map(std::env::var_os);
    let [Some(idol), Some(dinov3), Some(vae), Some(photo)] = vars else {
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
        idol: Some(idol.into()),
        dinov3_hplus: Some(dinov3.into()),
        flux2_vae: Some(vae.into()),
        ..Default::default()
    });
    let request = JobRequest::ImageToSplat(ImageToSplatJob {
        mode: Some(SplatImageMode::Real),
        gaussians: Some(GAUSSIANS),
        steps: Some(STEPS),
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
    assert_eq!(done.report["gaussians"], GAUSSIANS);
    let fov = done.report["camera"]["fov"].as_f64().unwrap_or(0.0);
    assert!(fov > 0.0 && fov < 180.0, "fov {fov}");
    let volume = done.report["print"]["volume_cm3"].as_f64().unwrap_or(0.0);
    assert!(volume > 0.0, "printable volume {volume}");
    let faces = done.report["textured"]["faces"].as_u64().unwrap_or(0);
    assert!(faces > 0, "textured faces {faces}");
    for name in [
        "splats",
        "splats (ply)",
        "printable (stl)",
        "textured model",
        "prepared image",
        "meta",
    ] {
        let output = done.outputs.iter().find(|o| o.name == name);
        let output = output.ok_or_else(|| anyhow::anyhow!("no {name} output"))?;
        assert!(std::fs::metadata(queue.file(&model, &done.id, &output.file)?)?.len() > 0);
    }
    let ply = done
        .outputs
        .iter()
        .find(|o| o.name == "splats (ply)")
        .expect("a ply output");
    let size = std::fs::metadata(queue.file(&model, &done.id, &ply.file)?)?.len();
    assert!(size > GAUSSIANS as u64 * PLY_ROW_BYTES, "{size} bytes");
    Ok(())
}
