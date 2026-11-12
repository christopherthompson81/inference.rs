//! Nemotron-3 Diarization through the engine on a tiny random-weight checkpoint whose 6 s clips fill and compress
//! the speaker cache: frame counts, thresholds, RTTM, and repeats on the default device.

use inference::{
    ApiErrorKind, DiarizationModelBuilder, DiarizationRequest, DiarizationResponse,
    DiarizationResponseFormat, ModelDType,
};

#[path = "../support/audiocpp_package.rs"]
mod package;
#[path = "../support/nemotron3_diarization_tiny.rs"]
mod support;
use support::tiny_nemotron3_diarization;

const RATE: u32 = 16_000;
const SECONDS: f64 = 6.0;
const HOP: usize = 160;
const SPEAKERS: usize = 4;
const REPEATS: usize = 2;
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

fn json(request: &DiarizationRequest, body: &str) -> DiarizationResponse {
    assert_eq!(request.response_format, DiarizationResponseFormat::Json);
    serde_json::from_str(body).unwrap()
}

#[tokio::test]
async fn the_cache_backed_model_scores_every_frame_and_thresholds_cut_it() -> anyhow::Result<()> {
    let dir = tiny_nemotron3_diarization()?;
    let model = DiarizationModelBuilder::new(dir.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let wav = chirp_wav();
    let mut request = DiarizationRequest::new();
    request.return_probabilities = true;
    let out = model.diarization(request.clone(), &wav).await?;
    assert_eq!(out.content_type, "application/json");
    let diarization = json(&request, &out.body);
    let probabilities = diarization.probabilities.clone().unwrap_or_default();
    assert_eq!(diarization.num_speakers, SPEAKERS);
    assert_eq!(
        probabilities.len(),
        (SECONDS * f64::from(RATE)) as usize / HOP
    );
    assert!(
        probabilities
            .iter()
            .flatten()
            .all(|p| (0.0..=1.0).contains(p))
    );
    assert!(
        diarization
            .segments
            .windows(2)
            .all(|w| (w[0].start, w[0].speaker) <= (w[1].start, w[1].speaker))
    );

    // a zero threshold makes every speaker speak throughout, one segment each; 1 silences them all
    request.threshold = Some(0.0);
    request.return_probabilities = false;
    let all = json(
        &request,
        &model.diarization(request.clone(), &wav).await?.body,
    );
    assert_eq!(all.segments.len(), SPEAKERS, "{:?}", all.segments);
    assert!(
        all.segments
            .iter()
            .all(|s| s.start == 0.0 && (s.end - diarization.duration).abs() < 0.011)
    );
    request.threshold = Some(1.0);
    assert!(
        json(
            &request,
            &model.diarization(request.clone(), &wav).await?.body
        )
        .segments
        .is_empty()
    );
    request.threshold = Some(1.5);
    assert!(model.diarization(request.clone(), &wav).await.is_err());

    request.threshold = Some(0.0);
    request.response_format = DiarizationResponseFormat::Rttm;
    let rttm = model.diarization(request, &wav).await?;
    assert_eq!(
        rttm.body
            .lines()
            .filter(|l| l.starts_with("SPEAKER audio 1 "))
            .count(),
        SPEAKERS
    );

    let err = model
        .diarization(DiarizationRequest::new(), b"not audio")
        .await
        .expect_err("garbage decoded");
    assert_eq!(err.kind, ApiErrorKind::InvalidRequest, "{err}");
    Ok(())
}

// On the default device (CUDA under `--features cuda`) the cache's choices repeat exactly
#[tokio::test]
async fn diarization_repeats_exactly_on_the_default_device() -> anyhow::Result<()> {
    let dir = tiny_nemotron3_diarization()?;
    let model = DiarizationModelBuilder::new(dir.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .build()
        .await?;
    let wav = chirp_wav();
    let mut request = DiarizationRequest::new();
    request.return_probabilities = true;
    let first = model.diarization(request.clone(), &wav).await?.body;
    for _ in 0..REPEATS {
        assert_eq!(model.diarization(request.clone(), &wav).await?.body, first);
    }
    Ok(())
}

// a recording diarized as a stream: `started` with its length, a `progress` per chunk rising to it, and last `done`
// with the one-shot response's segments; canceled at its first chunk it stops before the last
#[tokio::test]
async fn a_recording_diarizes_as_a_stream_of_chunk_progress() -> anyhow::Result<()> {
    use futures::StreamExt;
    let dir = tiny_nemotron3_diarization()?;
    let model = DiarizationModelBuilder::new(dir.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let wav = chirp_wav();
    let whole = model.diarization(DiarizationRequest::new(), &wav).await?;
    let whole: serde_json::Value = serde_json::from_str(&whole.body)?;
    let read = |stream: inference_api::speech_streams::SpeechStream, cancel: bool| async move {
        let mut events = stream.events;
        let mut out: Vec<serde_json::Value> = Vec::new();
        while let Some(event) = events.next().await {
            let event: serde_json::Value = serde_json::from_str(&event).unwrap();
            if cancel && event["event"] == "progress" {
                stream.cancellation.cancel();
            }
            out.push(event);
        }
        out
    };
    let events = read(
        model
            .diarization_stream_json(b"{}", Some(wav.clone()))
            .await?,
        false,
    )
    .await;
    assert_eq!(events[0]["event"], "started");
    assert!((events[0]["data"]["duration"].as_f64().unwrap() - SECONDS).abs() < 1e-6);
    let progress: Vec<f64> = events
        .iter()
        .filter(|e| e["event"] == "progress")
        .map(|e| e["data"]["seconds_done"].as_f64().unwrap())
        .collect();
    assert!(progress.len() > 1, "{} chunks", progress.len());
    assert!(progress.windows(2).all(|p| p[0] < p[1]));
    assert!((progress.last().unwrap() - SECONDS).abs() < 1e-6);
    let done = events.last().unwrap();
    assert_eq!(done["event"], "done");
    assert_eq!(done["data"]["segments"], whole["segments"]);
    let canceled = read(model.diarization_stream_json(b"{}", Some(wav)).await?, true).await;
    assert_eq!(canceled.last().unwrap()["event"], "canceled");
    // the chunk running when the cancel lands still finishes and reports
    assert!(canceled.iter().filter(|e| e["event"] == "progress").count() < progress.len());
    Ok(())
}

// the release's chunking around the speaker cache, which a package of the `.nemo` runs whatever its config says
const RELEASE_CHUNKING: [(&str, u64); 4] = [
    ("chunk_length", 340),
    ("chunk_right_context", 40),
    ("fifo_length", 40),
    ("speaker_cache_update_period", 300),
];
// the release's names for the `.nemo`'s, as audio.cpp's package keeps them
const NEMO_NAMES: [(&str, &str); 13] = [
    (
        "model.audio_tower.embedder.projection.",
        "encoder.pre_encode.proj.",
    ),
    ("model.audio_tower.input_layer_norm.", "encoder.embed_norm."),
    ("model.audio_tower.layer_norm.", "encoder.final_norm."),
    ("model.audio_tower.layers.", "encoder.layers."),
    (".layer_norm1.", ".norm1."),
    (".layer_norm2.", ".norm2."),
    (".mlp.fc1.", ".ffn.net.0."),
    (".mlp.fc2.", ".ffn.net.3."),
    (".self_attn.o_proj.", ".attn.out_proj."),
    (
        "classifier.dense.",
        "sortformer_modules.first_hidden_to_hidden.",
    ),
    (
        "classifier.out_proj.",
        "sortformer_modules.single_hidden_to_spks.",
    ),
    ("model.proj.", "sortformer_modules.encoder_proj."),
    (
        "model.upsampler.conv.",
        "sortformer_modules.subpixel_upsample.",
    ),
];

fn nemo_name(name: &str) -> String {
    if name == "silence_embeds" {
        return "sortformer_modules.learnable_sil_emb".to_string();
    }
    NEMO_NAMES
        .iter()
        .fold(name.to_string(), |name, (from, to)| name.replace(from, to))
}

// audio.cpp's own config for its package of the `.nemo`, from the release's
fn audiocpp_config(release: &serde_json::Value) -> serde_json::Value {
    let (a, h, s) = (
        &release["audio_config"],
        &release["head_config"],
        &release["streaming_config"],
    );
    serde_json::json!({
        "model_type": "nemotron_3_diar",
        "architectures": ["Nemotron3Diarization"],
        "num_speakers": h["num_speakers"],
        "encoder": {
            "feat_in": a["num_mel_bins"],
            "hidden_size": a["hidden_size"],
            "intermediate_size": a["intermediate_size"],
            "num_attention_heads": a["num_attention_heads"],
            "num_hidden_layers": a["num_hidden_layers"],
            "subsampling_factor": a["subsampling_factor"],
            "rope_theta": a["rope_parameters"]["rope_theta"],
        },
        "head": {"hidden_size": h["hidden_size"]},
        "streaming": {
            "spkcache_len": s["speaker_cache_length"],
            "spkcache_sil_frames_per_spk": s["speaker_cache_silence_frames_per_speaker"],
            "pred_score_threshold": s["prediction_score_threshold"],
            "scores_boost_latest": s["latest_frames_score_boost"],
            "min_pos_scores_rate": s["min_positive_scores_rate"],
            "strong_boost_rate": s["strong_boost_rate"],
            "weak_boost_rate": s["weak_boost_rate"],
            "chunk_len": 2,
            "fifo_len": 0,
        },
    })
}

// audio.cpp's package of the `.nemo` (its names, query, key and value in one matrix, its own config) diarizes as the
// release does with the release's chunking
#[tokio::test]
async fn an_audiocpp_package_of_the_nemo_diarizes_as_the_release() -> anyhow::Result<()> {
    let release = tiny_nemotron3_diarization()?;
    let config_path = release.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
    for (key, value) in RELEASE_CHUNKING {
        config[key] = value.into();
    }
    std::fs::write(&config_path, config.to_string())?;

    let nemo = tempfile::tempdir()?;
    let mut tensors = std::collections::HashMap::new();
    let mut qkv =
        std::collections::BTreeMap::<String, [Option<inference_tensor::Tensor>; 3]>::new();
    for (name, t) in inference_tensor::safetensors::load(
        release.path().join("model.safetensors"),
        &inference_tensor::Device::Cpu,
    )? {
        let part = ["q_proj", "k_proj", "v_proj"]
            .iter()
            .position(|p| name.ends_with(&format!(".self_attn.{p}.weight")));
        match part {
            Some(part) => {
                let layer = name.split(".self_attn.").next().unwrap_or_default();
                qkv.entry(nemo_name(&format!("{layer}."))).or_default()[part] = Some(t);
            }
            None => {
                tensors.insert(nemo_name(&name), t);
            }
        }
    }
    for (layer, [q, k, v]) in qkv {
        let parts = [q, k, v].map(|t| t.expect("each of query, key and value"));
        tensors.insert(
            format!("{layer}attn.w_qkv.weight"),
            inference_tensor::Tensor::cat(&parts, 0)?,
        );
    }
    inference_tensor::safetensors::save(&tensors, nemo.path().join("model.safetensors"))?;
    std::fs::write(
        nemo.path().join("config.json"),
        audiocpp_config(&config).to_string(),
    )?;
    std::fs::copy(
        release.path().join("processor_config.json"),
        nemo.path().join("processor_config.json"),
    )?;
    let out = tempfile::tempdir()?;
    let gguf = package::repack_all(
        nemo.path(),
        "nemotron_3_diar",
        &[("", "model.safetensors")],
        &out.path().join("tiny-f32.gguf"),
    )?;

    let mut request = DiarizationRequest::new();
    request.return_probabilities = true;
    let wav = chirp_wav();
    let mut bodies = Vec::new();
    for path in [release.path(), gguf.as_path()] {
        let model = DiarizationModelBuilder::new(path.to_string_lossy())
            .with_dtype(ModelDType::F32)
            .with_force_cpu()
            .build()
            .await?;
        bodies.push(model.diarization(request.clone(), &wav).await?.body);
    }
    let [want, got] = [&bodies[0], &bodies[1]].map(|body| json(&request, body));
    assert_eq!(got.segments, want.segments);
    assert_eq!(got.probabilities, want.probabilities);
    Ok(())
}
