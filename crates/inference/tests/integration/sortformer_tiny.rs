//! A Streaming Sortformer `.nemo` through the engine on a tiny random-weight checkpoint whose clip fills and
//! compresses the speaker cache: 80 ms frames, segments, and loading from the file or its directory.

use inference::{DiarizationModelBuilder, DiarizationRequest, DiarizationResponse, ModelDType};

#[path = "../support/sortformer_tiny.rs"]
mod support;
use support::{checkpoint, tiny_sortformer};

const RATE: u32 = 16_000;
const SECONDS: f64 = 6.0;
// the encoder's 8x subsampling over 10 ms mel frames
const FRAME_SECONDS: f64 = 0.08;
const SPEAKERS: usize = 4;
const CHIRP_START_HZ: f64 = 150.0;
const CHIRP_RISE_HZ: f64 = 300.0;

fn chirp_wav() -> Vec<u8> {
    let pcm: Vec<f32> = (0..(SECONDS * f64::from(RATE)) as usize)
        .map(|i| {
            let t = i as f64 / f64::from(RATE);
            (0.3 * (2.0
                * std::f64::consts::PI
                * (CHIRP_START_HZ * t + CHIRP_RISE_HZ * t * t / 2.0))
                .sin()) as f32
        })
        .collect();
    let mut wav = Vec::new();
    inference_models_speech::utils::write_pcm_as_wav(&mut wav, &pcm, RATE, 1).unwrap();
    wav
}

#[tokio::test]
async fn a_nemo_checkpoint_scores_every_encoder_frame() -> anyhow::Result<()> {
    let dir = tiny_sortformer()?;
    let wav = chirp_wav();
    let mut request = DiarizationRequest::new();
    request.return_probabilities = true;
    let mut seen = Vec::new();
    for model_id in [checkpoint(dir.path()), dir.path().to_path_buf()] {
        let model = DiarizationModelBuilder::new(model_id.to_string_lossy())
            .with_dtype(ModelDType::F32)
            .with_force_cpu()
            .build()
            .await?;
        let out = model.diarization(request.clone(), &wav).await?;
        let diarization: DiarizationResponse = serde_json::from_str(&out.body)?;
        let probabilities = diarization.probabilities.clone().unwrap_or_default();
        assert_eq!(diarization.num_speakers, SPEAKERS);
        assert!((diarization.frame_seconds - FRAME_SECONDS).abs() < 1e-9);
        // 600 mel frames, subsampled 8x with each stage rounding up
        assert_eq!(probabilities.len(), 75);
        assert!(probabilities.iter().all(|row| row.len() == SPEAKERS));
        assert!(
            probabilities
                .iter()
                .flatten()
                .all(|p| (0.0..=1.0).contains(p))
        );
        assert!(
            diarization
                .segments
                .iter()
                .all(|s| s.start < s.end && s.end <= diarization.duration + FRAME_SECONDS)
        );
        seen.push(probabilities);
    }
    assert_eq!(
        seen[0], seen[1],
        "the file and its directory load the same model"
    );
    Ok(())
}
