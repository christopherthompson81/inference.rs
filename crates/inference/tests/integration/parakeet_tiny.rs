//! Parakeet and the streaming Nemotron ASR through the engine on tiny random-weight checkpoints: each head, each
//! response format, the language prompt, request errors.

use inference::{
    ApiErrorKind, ModelDType, TimestampGranularity, TranscriptionModelBuilder,
    TranscriptionRequest, TranscriptionResponse, TranscriptionResponseFormat,
    VerboseTranscriptionResponse,
};

#[path = "../support/parakeet_tiny.rs"]
mod support;
use support::{
    HEADS, LANGUAGES, PROMPTED, tiny_parakeet_checkpoint, tiny_parakeet_gguf, tiny_parakeet_nemo,
};

// off the model's 16 kHz, so the request is resampled
const RATE: u32 = 22_050;
const SECONDS: f64 = 1.5;
const CHIRP_START_HZ: f64 = 200.0;
const CHIRP_RISE_HZ: f64 = 600.0;
const AMPLITUDE: f64 = 0.4;
// past the encoder's 512-frame attention block, 41 s at 80 ms a frame
const LONG_SECONDS: f64 = 45.0;
const REPEATS: usize = 1;
// under the two 10 ms feature frames the model needs
const SHORT_SECONDS: f64 = 0.01;
// the resampler's filter shortens the audio by a few samples
const DURATION_TOLERANCE: f64 = 0.01;

fn chirp_wav() -> Vec<u8> {
    chirp_wav_of(SECONDS)
}

fn chirp_wav_of(seconds: f64) -> Vec<u8> {
    let n = (seconds * f64::from(RATE)) as usize;
    let pcm: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 / f64::from(RATE);
            let phase =
                2.0 * std::f64::consts::PI * (CHIRP_START_HZ * t + CHIRP_RISE_HZ * t * t / 2.0);
            (AMPLITUDE * phase.sin()) as f32
        })
        .collect();
    let mut wav = Vec::new();
    inference_models_speech::utils::write_pcm_as_wav(&mut wav, &pcm, RATE, 1).unwrap();
    wav
}

fn request(format: TranscriptionResponseFormat) -> TranscriptionRequest {
    let mut request = TranscriptionRequest::new(format);
    request.timestamp_granularities =
        vec![TimestampGranularity::Word, TimestampGranularity::Segment];
    request
}

#[tokio::test]
async fn each_head_transcribes_into_each_format() -> anyhow::Result<()> {
    let wav = chirp_wav();
    for head in HEADS {
        let checkpoint = tiny_parakeet_checkpoint(head)?;
        let model = TranscriptionModelBuilder::new(checkpoint.path().to_string_lossy())
            .with_dtype(ModelDType::F32)
            .with_force_cpu()
            .build()
            .await?;

        let verbose = model
            .transcription(request(TranscriptionResponseFormat::VerboseJson), &wav)
            .await?;
        assert_eq!(verbose.content_type, "application/json", "{head}");
        let parsed: VerboseTranscriptionResponse = serde_json::from_str(&verbose.body)?;
        assert!(
            (parsed.duration - SECONDS).abs() < DURATION_TOLERANCE,
            "{head}: {}",
            parsed.duration
        );
        let words = parsed.words.as_deref().unwrap_or_default();
        // the seeded weights emit words under every head, so the timing checks below have something to check
        assert!(!words.is_empty(), "{head}");
        assert!(words.windows(2).all(|w| w[0].start <= w[1].start), "{head}");
        assert!(
            words
                .iter()
                .all(|w| w.start <= w.end && w.end <= parsed.duration + DURATION_TOLERANCE),
            "{head}"
        );
        let joined = words
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(joined, parsed.text, "{head}");
        let segmented = parsed
            .segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(segmented, parsed.text, "{head}");

        let again = model
            .transcription(request(TranscriptionResponseFormat::VerboseJson), &wav)
            .await?;
        assert_eq!(
            again.body, verbose.body,
            "{head}: a repeat transcribes differently"
        );

        let text = model
            .transcription(request(TranscriptionResponseFormat::Text), &wav)
            .await?;
        assert_eq!(text.body, parsed.text, "{head}");
        let json = model
            .transcription(request(TranscriptionResponseFormat::Json), &wav)
            .await?;
        let json: TranscriptionResponse = serde_json::from_str(&json.body)?;
        assert_eq!(json.text, parsed.text, "{head}");
        let vtt = model
            .transcription(request(TranscriptionResponseFormat::Vtt), &wav)
            .await?;
        assert!(vtt.body.starts_with("WEBVTT"), "{head}");
        let srt = model
            .transcription(request(TranscriptionResponseFormat::Srt), &wav)
            .await?;
        assert_eq!(
            srt.content_type, "application/x-subrip; charset=utf-8",
            "{head}"
        );
        assert_eq!(
            srt.body.matches(" --> ").count(),
            parsed.segments.len(),
            "{head}"
        );
    }
    Ok(())
}

// On the default device (CUDA under `--features cuda`), past one attention block, every head repeats exactly
#[tokio::test]
async fn each_head_repeats_exactly_on_the_default_device() -> anyhow::Result<()> {
    let wav = chirp_wav_of(LONG_SECONDS);
    for head in HEADS {
        let checkpoint = tiny_parakeet_checkpoint(head)?;
        let model = TranscriptionModelBuilder::new(checkpoint.path().to_string_lossy())
            .with_dtype(ModelDType::F32)
            .build()
            .await?;
        let first = model
            .transcription(request(TranscriptionResponseFormat::VerboseJson), &wav)
            .await?
            .body;
        for _ in 0..REPEATS {
            let again = model
                .transcription(request(TranscriptionResponseFormat::VerboseJson), &wav)
                .await?;
            assert_eq!(again.body, first, "{head}");
        }
    }
    Ok(())
}

#[tokio::test]
async fn undecodable_audio_is_an_invalid_request() -> anyhow::Result<()> {
    let checkpoint = tiny_parakeet_checkpoint(HEADS[0])?;
    let model = TranscriptionModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let err = model
        .transcription(
            request(TranscriptionResponseFormat::Json),
            b"not audio at all",
        )
        .await
        .expect_err("garbage bytes decoded");
    assert_eq!(err.kind, ApiErrorKind::InvalidRequest, "{err}");
    Ok(())
}

// requests that land in one engine step answer each on its own: a clip the model refuses fails only itself
#[tokio::test]
async fn too_short_audio_fails_only_its_own_request() -> anyhow::Result<()> {
    let checkpoint = tiny_parakeet_checkpoint(HEADS[0])?;
    let model = TranscriptionModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let good = chirp_wav();
    let short = chirp_wav_of(SHORT_SECONDS);
    let (a, b, c) = tokio::join!(
        model.transcription(request(TranscriptionResponseFormat::Text), &good),
        model.transcription(request(TranscriptionResponseFormat::Text), &short),
        model.transcription(request(TranscriptionResponseFormat::Text), &good),
    );
    assert_eq!(a?.body, c?.body);
    let err = b.expect_err("a clip shorter than two feature frames transcribed");
    assert_eq!(err.kind, ApiErrorKind::InvalidRequest, "{err}");
    assert!(err.message.contains("too short"), "{err}");
    Ok(())
}

// Nemotron-3.5 takes the request's language as its prompt: it is reported back, it conditions the encoder, and a
// language outside the checkpoint's dictionary is refused for its own request alone
#[tokio::test]
async fn a_prompted_model_takes_the_requested_language() -> anyhow::Result<()> {
    let checkpoint = tiny_parakeet_checkpoint(PROMPTED)?;
    let model = TranscriptionModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let wav = chirp_wav();
    let mut transcribed = Vec::new();
    for (language, _) in &LANGUAGES[..2] {
        let mut asked = request(TranscriptionResponseFormat::VerboseJson);
        asked.language = Some(language.to_string());
        let out = model.transcription(asked, &wav).await?;
        let parsed: VerboseTranscriptionResponse = serde_json::from_str(&out.body)?;
        assert_eq!(parsed.language.as_deref(), Some(*language));
        transcribed.push(parsed.text);
    }
    assert_ne!(
        transcribed[0], transcribed[1],
        "the prompt does not reach the encoder"
    );

    let mut unknown = request(TranscriptionResponseFormat::Text);
    unknown.language = Some("tlh".to_string());
    let err = model
        .transcription(unknown, &wav)
        .await
        .expect_err("a language outside the dictionary transcribed");
    assert_eq!(err.kind, ApiErrorKind::InvalidRequest, "{err}");
    assert!(err.message.contains("de-DE"), "{err}");
    Ok(())
}

// a `.nemo` of the same weights transcribes as the transformers layout does, for every head it carries
#[tokio::test]
async fn a_nemo_checkpoint_transcribes_as_its_transformers_layout() -> anyhow::Result<()> {
    let wav = chirp_wav();
    for head in HEADS.iter().filter(|h| **h != PROMPTED) {
        let (dir, nemo) = tiny_parakeet_nemo(head)?;
        let mut bodies = Vec::new();
        for model_id in [dir.path().to_path_buf(), nemo] {
            let model = TranscriptionModelBuilder::new(model_id.to_string_lossy())
                .with_dtype(ModelDType::F32)
                .with_force_cpu()
                .build()
                .await?;
            let out = model
                .transcription(request(TranscriptionResponseFormat::VerboseJson), &wav)
                .await?;
            bodies.push(out.body);
        }
        assert_eq!(bodies[0], bodies[1], "{head}");
    }
    Ok(())
}

// NVIDIA's `asr` GGUF of the same weights (here in F32) transcribes as the transformers layout does
#[tokio::test]
async fn an_asr_gguf_transcribes_as_its_transformers_layout() -> anyhow::Result<()> {
    let wav = chirp_wav();
    for head in ["parakeet_tdt", "parakeet_rnnt", "nemotron_asr_streaming"] {
        let (dir, gguf) = tiny_parakeet_gguf(head)?;
        let mut bodies = Vec::new();
        for model_id in [dir.path().to_path_buf(), gguf] {
            let model = TranscriptionModelBuilder::new(model_id.to_string_lossy())
                .with_dtype(ModelDType::F32)
                .with_force_cpu()
                .build()
                .await?;
            let out = model
                .transcription(request(TranscriptionResponseFormat::VerboseJson), &wav)
                .await?;
            bodies.push(out.body);
        }
        assert_eq!(bodies[0], bodies[1], "{head}");
    }
    Ok(())
}

// a `.nemo` written as an `asr` GGUF with full-precision matrices loads and transcribes as the `.nemo` does
#[tokio::test]
async fn a_written_asr_gguf_transcribes_as_its_nemo() -> anyhow::Result<()> {
    use inference_models_speech::parakeet::{QuantPolicy, write_asr_gguf};
    use inference_tensor::quantized::GgmlDType;
    let wav = chirp_wav();
    for head in ["parakeet_tdt", "parakeet_rnnt", "nemotron_asr_streaming"] {
        let (dir, nemo) = tiny_parakeet_nemo(head)?;
        let gguf = dir.path().join("written.gguf");
        write_asr_gguf(&nemo, &gguf, &QuantPolicy::uniform(GgmlDType::F32))?;
        let mut bodies = Vec::new();
        for model_id in [nemo.clone(), gguf] {
            let model = TranscriptionModelBuilder::new(model_id.to_string_lossy())
                .with_dtype(ModelDType::F32)
                .with_force_cpu()
                .build()
                .await?;
            let out = model
                .transcription(request(TranscriptionResponseFormat::VerboseJson), &wav)
                .await?;
            bodies.push(out.body);
        }
        assert_eq!(bodies[0], bodies[1], "{head}");
        // a mixed policy's file loads too: its K-quant rows fall back, its decoder overrides apply
        let mixed = dir.path().join("mixed.gguf");
        write_asr_gguf(&nemo, &mixed, &QuantPolicy::q6_k_mix())?;
        TranscriptionModelBuilder::new(mixed.to_string_lossy())
            .with_dtype(ModelDType::F32)
            .with_force_cpu()
            .build()
            .await?
            .transcription(request(TranscriptionResponseFormat::Text), &wav)
            .await?;
    }
    Ok(())
}

// a beam decodes each transducer head, repeating exactly; a CTC head and a beam out of range are refused
#[tokio::test]
async fn beam_search_decodes_transducers_and_refuses_the_rest() -> anyhow::Result<()> {
    let wav = chirp_wav();
    let beamed = |beam: usize| {
        let mut r = request(TranscriptionResponseFormat::VerboseJson);
        r.beam_size = Some(beam);
        r
    };
    for head in ["parakeet_tdt", "parakeet_rnnt", "nemotron_asr_streaming"] {
        let checkpoint = tiny_parakeet_checkpoint(head)?;
        let model = TranscriptionModelBuilder::new(checkpoint.path().to_string_lossy())
            .with_dtype(ModelDType::F32)
            .with_force_cpu()
            .build()
            .await?;
        let first = model.transcription(beamed(4), &wav).await?.body;
        let parsed: VerboseTranscriptionResponse = serde_json::from_str(&first)?;
        assert!(
            parsed
                .words
                .as_deref()
                .unwrap_or_default()
                .windows(2)
                .all(|w| w[0].start <= w[1].start),
            "{head}"
        );
        assert_eq!(
            model.transcription(beamed(4), &wav).await?.body,
            first,
            "{head}"
        );
        let err = model
            .transcription(beamed(17), &wav)
            .await
            .expect_err("a beam past the limit decoded");
        assert_eq!(err.kind, ApiErrorKind::InvalidRequest, "{err}");
    }
    let ctc = tiny_parakeet_checkpoint("parakeet_ctc")?;
    let model = TranscriptionModelBuilder::new(ctc.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let err = model
        .transcription(beamed(4), &wav)
        .await
        .expect_err("a CTC head beam-searched");
    assert_eq!(err.kind, ApiErrorKind::InvalidRequest, "{err}");
    assert!(err.message.contains("CTC"), "{err}");
    Ok(())
}

// pieces off and on the 160-sample hop and the 2-frame chunk, up to the whole clip at once
const STREAM_PIECES: [usize; 5] = [13, 997, 1280, 16_000, usize::MAX];
const STREAM_RATE: u32 = 16_000;
// 50 encoder frames: 25 of the tiny model's chunks
const STREAM_SECONDS: f64 = 4.0;

fn tiny_parakeet(
    dir: &std::path::Path,
) -> anyhow::Result<inference_models_speech::parakeet::Parakeet> {
    let files = inference_models_speech::parakeet::ParakeetFiles {
        config: dir.join("config.json"),
        processor_config: dir.join("processor_config.json"),
        tokenizer: dir.join("tokenizer.json"),
        weights: vec![dir.join("model.safetensors")],
    };
    Ok(inference_models_speech::parakeet::Parakeet::load(
        &files,
        &inference_tensor::Device::Cpu,
        inference_tensor::DType::F32,
    )?)
}

// a streamed transcription, fed in any pieces, emits the offline pass's tokens at its frames, its deltas add up to its
// text, and too little audio fails as offline
#[test]
fn a_streamed_transcription_equals_the_offline_one() -> anyhow::Result<()> {
    let n = (STREAM_SECONDS * f64::from(STREAM_RATE)) as usize;
    let pcm: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 / f64::from(STREAM_RATE);
            let phase =
                2.0 * std::f64::consts::PI * (CHIRP_START_HZ * t + CHIRP_RISE_HZ * t * t / 2.0);
            (AMPLITUDE * phase.sin() * (1.0 + (3.0 * t).sin()) / 2.0) as f32
        })
        .collect();
    // the 3.5 checkpoint also streams with no lookahead: one-frame chunks, shorter than the subsampling's drop
    for (head, lookahead) in [
        ("nemotron_asr_streaming", None),
        (PROMPTED, None),
        (PROMPTED, Some(0)),
    ] {
        let dir = tiny_parakeet_checkpoint(head)?;
        if let Some(lookahead) = lookahead {
            let path = dir.path().join("config.json");
            let mut config: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&path)?)?;
            config["encoder_config"]["default_num_lookahead_tokens"] = lookahead.into();
            std::fs::write(&path, config.to_string())?;
        }
        let model = tiny_parakeet(dir.path())?;
        assert!(model.can_stream());
        let options = inference_models_speech::TranscriptionOptions::default();
        let offline = model.transcribe(&pcm, STREAM_RATE, &options)?;
        assert!(
            !offline.tokens.is_empty(),
            "{head}: the tiny model emits tokens"
        );
        for piece in STREAM_PIECES {
            let mut stream = model.stream(&options)?;
            let mut deltas = String::new();
            for chunk in pcm.chunks(piece.min(pcm.len())) {
                deltas.push_str(&model.stream_push(&mut stream, chunk)?);
            }
            let (last, streamed) = model.stream_finish(stream)?;
            deltas.push_str(&last);
            assert_eq!(streamed, offline, "{head}, pieces of {piece}");
            assert_eq!(deltas.trim_end(), offline.text, "{head}, pieces of {piece}");
        }
    }
    let dir = tiny_parakeet_checkpoint("nemotron_asr_streaming")?;
    let model = tiny_parakeet(dir.path())?;
    let options = inference_models_speech::TranscriptionOptions::default();
    let mut short = model.stream(&options)?;
    model.stream_push(&mut short, &pcm[..100])?;
    assert!(
        model
            .transcribe(&pcm[..100], STREAM_RATE, &options)
            .is_err()
    );
    assert!(model.stream_finish(short).is_err());
    let dir = tiny_parakeet_checkpoint("parakeet_tdt")?;
    let offline_only = tiny_parakeet(dir.path())?;
    assert!(!offline_only.can_stream());
    assert!(
        offline_only
            .stream(&inference_models_speech::TranscriptionOptions::default())
            .is_err()
    );
    Ok(())
}
