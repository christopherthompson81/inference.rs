//! Qwen3-TTS through the engine on a tiny random-weight checkpoint: preset voices, languages, instructions, seeds,
//! detection from the config, and request errors.

use inference::{
    AudioResponseFormat, ModelBuilder, ModelDType, SpeechGenerationRequest, SpeechGenerationSpec,
    SpeechLoaderType, SpeechModelBuilder,
};

#[path = "../support/audiocpp_package.rs"]
mod package;
#[path = "../support/qwen3_tts_tiny.rs"]
mod support;
use support::{
    LANGUAGES, SAMPLES_PER_FRAME, SPEAKERS, tiny_qwen3_tts_checkpoint, tiny_qwen3_tts_gguf,
};

const TEXT: &str = "Hello world, this is a test.";
const SEED: u64 = 7;
const MAX_FRAMES: usize = 12;
const PCM_BYTES: usize = 2;
const CUSTOM_VOICE: &str = "custom_voice";
const VOICE_DESIGN: &str = "voice_design";
const SMALL: &str = "0b6";
const LARGE: &str = "1b7";
const INSTRUCTIONS: &str = "A calm, low voice, speaking slowly.";

fn request(voice: Option<&str>) -> SpeechGenerationRequest {
    SpeechGenerationRequest {
        voice: voice.map(str::to_string),
        language: Some(LANGUAGES[0].to_string()),
        seed: Some(SEED),
        ..SpeechGenerationRequest::new(TEXT, AudioResponseFormat::Pcm)
    }
}

async fn load(path: &std::path::Path) -> anyhow::Result<inference::Model> {
    Ok(
        SpeechModelBuilder::new(path.to_string_lossy(), SpeechLoaderType::Qwen3Tts)
            .with_dtype(ModelDType::F32)
            .with_force_cpu()
            .with_generation(SpeechGenerationSpec {
                max_tokens: Some(MAX_FRAMES),
                ..SpeechGenerationSpec::default()
            })
            .build()
            .await?,
    )
}

fn error_text<T, E: std::fmt::Display>(result: Result<T, E>) -> String {
    match result {
        Ok(_) => panic!("the request should have failed"),
        Err(err) => err.to_string(),
    }
}

// whole frames of audio, at most the configured cap
fn assert_frames(bytes: &[u8]) {
    let samples = bytes.len() / PCM_BYTES;
    assert!(samples > 0, "no audio");
    assert_eq!(samples % SAMPLES_PER_FRAME, 0, "{samples} samples");
    assert!(
        samples <= MAX_FRAMES * SAMPLES_PER_FRAME,
        "{samples} samples"
    );
}

#[tokio::test]
async fn qwen3_tts_speaks_preset_voices_with_seeds() -> anyhow::Result<()> {
    let checkpoint = tiny_qwen3_tts_checkpoint(CUSTOM_VOICE, SMALL)?;
    let model = load(checkpoint.path()).await?;
    let base = model.generate_speech(request(Some(SPEAKERS[0]))).await?;
    assert!(
        base.content_type.contains("rate=24000"),
        "{}",
        base.content_type
    );
    assert_frames(&base.bytes);
    assert_eq!(
        model
            .generate_speech(request(Some(SPEAKERS[0])))
            .await?
            .bytes,
        base.bytes
    );

    let reseeded = SpeechGenerationRequest {
        seed: Some(SEED + 1),
        ..request(Some(SPEAKERS[0]))
    };
    assert_ne!(model.generate_speech(reseeded).await?.bytes, base.bytes);
    // voice names are case-insensitive, as the reference's are
    let upper = request(Some(&SPEAKERS[0].to_uppercase()));
    assert_eq!(model.generate_speech(upper).await?.bytes, base.bytes);
    let other = model.generate_speech(request(Some(SPEAKERS[1]))).await?;
    assert_frames(&other.bytes);
    assert_ne!(other.bytes, base.bytes);
    // no language lets the model pick
    let auto = SpeechGenerationRequest {
        language: None,
        ..request(Some(SPEAKERS[0]))
    };
    assert_ne!(model.generate_speech(auto).await?.bytes, base.bytes);
    Ok(())
}

#[tokio::test]
async fn qwen3_tts_refuses_what_the_checkpoint_cannot_do() -> anyhow::Result<()> {
    let checkpoint = tiny_qwen3_tts_checkpoint(CUSTOM_VOICE, SMALL)?;
    let model = load(checkpoint.path()).await?;
    let unknown = error_text(model.generate_speech(request(Some("nobody"))).await);
    assert!(
        unknown.contains("unknown voice `nobody`") && unknown.contains(SPEAKERS[1]),
        "{unknown}"
    );
    let missing = error_text(model.generate_speech(request(None)).await);
    assert!(missing.contains("needs a `voice`"), "{missing}");
    let language = SpeechGenerationRequest {
        language: Some("klingon".to_string()),
        ..request(Some(SPEAKERS[0]))
    };
    let language = error_text(model.generate_speech(language).await);
    assert!(
        language.contains("unknown language `klingon`"),
        "{language}"
    );
    let instructed = SpeechGenerationRequest {
        instructions: Some(INSTRUCTIONS.to_string()),
        ..request(Some(SPEAKERS[0]))
    };
    let instructed = error_text(model.generate_speech(instructed).await);
    assert!(
        instructed.contains("take no `instructions`"),
        "{instructed}"
    );
    let phonemes = SpeechGenerationRequest {
        phonemes: Some("hɛloʊ".to_string()),
        ..request(Some(SPEAKERS[0]))
    };
    let phonemes = error_text(model.generate_speech(phonemes).await);
    assert!(phonemes.contains("does not take `phonemes`"), "{phonemes}");
    Ok(())
}

// A 1.7B-style checkpoint takes instructions; a voice design one needs them and ignores the voice
#[tokio::test]
async fn qwen3_tts_follows_instructions() -> anyhow::Result<()> {
    let checkpoint = tiny_qwen3_tts_checkpoint(CUSTOM_VOICE, LARGE)?;
    let model = load(checkpoint.path()).await?;
    let plain = model.generate_speech(request(Some(SPEAKERS[0]))).await?;
    let instructed = SpeechGenerationRequest {
        instructions: Some(INSTRUCTIONS.to_string()),
        ..request(Some(SPEAKERS[0]))
    };
    let instructed = model.generate_speech(instructed).await?;
    assert_frames(&instructed.bytes);
    assert_ne!(instructed.bytes, plain.bytes);

    let checkpoint = tiny_qwen3_tts_checkpoint(VOICE_DESIGN, LARGE)?;
    let model = load(checkpoint.path()).await?;
    let missing = error_text(model.generate_speech(request(Some(SPEAKERS[0]))).await);
    assert!(missing.contains("needs `instructions`"), "{missing}");
    let designed = |voice: Option<&str>| SpeechGenerationRequest {
        instructions: Some(INSTRUCTIONS.to_string()),
        ..request(voice)
    };
    let designed_speech = model.generate_speech(designed(Some("alloy"))).await?;
    assert_frames(&designed_speech.bytes);
    assert_eq!(
        model.generate_speech(designed(None)).await?.bytes,
        designed_speech.bytes
    );
    Ok(())
}

// No architecture named: the auto loader reads config.json and picks Qwen3-TTS
#[tokio::test]
async fn qwen3_tts_is_detected_from_its_config() -> anyhow::Result<()> {
    let checkpoint = tiny_qwen3_tts_checkpoint(CUSTOM_VOICE, SMALL)?;
    let detected = ModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let speech = detected.generate_speech(request(Some(SPEAKERS[0]))).await?;
    assert!(!speech.bytes.is_empty());
    Ok(())
}

const BASE: &str = "base";
const CLIP_RATE: u32 = 16_000;
const CLIP_SAMPLES: usize = 9_600;
const CLIP_TEXT: &str = "Hello there.";
const WAV_HEADER_BYTES: u32 = 36;
const PCM16_BITS: u16 = 16;
const TONE_AMPLITUDE: f32 = 0.3;

// a 0.6 s 16-bit tone at 16 kHz, base64-encoded as a WAV file; the model resamples it to 24 kHz
fn tone_base64(hertz: f32) -> String {
    use base64::Engine as _;
    let samples: Vec<i16> = (0..CLIP_SAMPLES)
        .map(|n| {
            let t = n as f32 / CLIP_RATE as f32;
            (TONE_AMPLITUDE * (std::f32::consts::TAU * hertz * t).sin() * f32::from(i16::MAX))
                as i16
        })
        .collect();
    let data = u32::try_from(samples.len() * 2).unwrap();
    let mut wav = Vec::new();
    wav.extend(b"RIFF");
    wav.extend((WAV_HEADER_BYTES + data).to_le_bytes());
    wav.extend(b"WAVEfmt ");
    wav.extend(16u32.to_le_bytes());
    wav.extend(1u16.to_le_bytes());
    wav.extend(1u16.to_le_bytes());
    wav.extend(CLIP_RATE.to_le_bytes());
    wav.extend((CLIP_RATE * 2).to_le_bytes());
    wav.extend(2u16.to_le_bytes());
    wav.extend(PCM16_BITS.to_le_bytes());
    wav.extend(b"data");
    wav.extend(data.to_le_bytes());
    for s in samples {
        wav.extend(s.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(wav)
}

fn clone_request(audio: Option<String>, text: Option<&str>) -> SpeechGenerationRequest {
    SpeechGenerationRequest {
        reference_audio: audio,
        reference_text: text.map(str::to_string),
        ..request(None)
    }
}

// x-vector cloning from the clip alone, in-context cloning with its transcript; another clip is another voice
#[tokio::test]
async fn qwen3_tts_base_clones_a_voice() -> anyhow::Result<()> {
    let checkpoint = tiny_qwen3_tts_checkpoint(BASE, SMALL)?;
    let model = load(checkpoint.path()).await?;
    let clip = tone_base64(220.0);
    let voiced = model
        .generate_speech(clone_request(Some(clip.clone()), None))
        .await?;
    assert_frames(&voiced.bytes);
    assert_eq!(
        model
            .generate_speech(clone_request(Some(clip.clone()), None))
            .await?
            .bytes,
        voiced.bytes
    );
    let continued = model
        .generate_speech(clone_request(Some(clip.clone()), Some(CLIP_TEXT)))
        .await?;
    assert_frames(&continued.bytes);
    assert_ne!(continued.bytes, voiced.bytes);
    let other = model
        .generate_speech(clone_request(Some(tone_base64(440.0)), None))
        .await?;
    assert_ne!(other.bytes, voiced.bytes);
    // the voice comes from the clip, so a named voice changes nothing
    let named = SpeechGenerationRequest {
        voice: Some("alloy".to_string()),
        ..clone_request(Some(clip), None)
    };
    assert_eq!(model.generate_speech(named).await?.bytes, voiced.bytes);
    Ok(())
}

#[tokio::test]
async fn qwen3_tts_cloning_refuses_bad_requests() -> anyhow::Result<()> {
    let checkpoint = tiny_qwen3_tts_checkpoint(BASE, SMALL)?;
    let model = load(checkpoint.path()).await?;
    let missing = error_text(model.generate_speech(clone_request(None, None)).await);
    assert!(missing.contains("clones a voice"), "{missing}");
    let text_only = error_text(
        model
            .generate_speech(clone_request(None, Some(CLIP_TEXT)))
            .await,
    );
    assert!(
        text_only.contains("needs the `reference_audio`"),
        "{text_only}"
    );
    let not_base64 = error_text(
        model
            .generate_speech(clone_request(Some("not base64!".to_string()), None))
            .await,
    );
    assert!(not_base64.contains("not base64"), "{not_base64}");
    let not_audio = error_text(
        model
            .generate_speech(clone_request(Some("bm90IGF1ZGlv".to_string()), None))
            .await,
    );
    assert!(not_audio.contains("could not be decoded"), "{not_audio}");
    let instructed = SpeechGenerationRequest {
        instructions: Some(INSTRUCTIONS.to_string()),
        ..clone_request(Some(tone_base64(220.0)), None)
    };
    let instructed = error_text(model.generate_speech(instructed).await);
    assert!(
        instructed.contains("takes no `instructions`"),
        "{instructed}"
    );

    let checkpoint = tiny_qwen3_tts_checkpoint(CUSTOM_VOICE, SMALL)?;
    let model = load(checkpoint.path()).await?;
    let preset = SpeechGenerationRequest {
        reference_audio: Some(tone_base64(220.0)),
        ..request(Some(SPEAKERS[0]))
    };
    let preset = error_text(model.generate_speech(preset).await);
    assert!(preset.contains("only the Base"), "{preset}");
    Ok(())
}

// the GGUF's folds round apart from the checkpoint's arithmetic only in the codec's SnakeBeta (exp after log)
const GGUF_PCM_TOLERANCE: i32 = 2;

fn pcm16(bytes: &[u8]) -> Vec<i32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| i32::from(i16::from_le_bytes(*b)))
        .collect()
}

// audio.cpp's package of it (the talker and the codec as two sources, the configs and tokenizer embedded) speaks as the
// checkpoint, cloning from the clip and its transcript
#[tokio::test]
async fn an_audiocpp_package_speaks_as_its_checkpoint() -> anyhow::Result<()> {
    let checkpoint = tiny_qwen3_tts_checkpoint(BASE, SMALL)?;
    let dir = tempfile::tempdir()?;
    let gguf = package::repack_all(
        checkpoint.path(),
        "qwen3_tts",
        &[
            ("model_weights", "model.safetensors"),
            (
                "speech_tokenizer_weights",
                "speech_tokenizer/model.safetensors",
            ),
        ],
        &dir.path().join("tiny-f32.gguf"),
    )?;
    let clip = tone_base64(220.0);
    let speak = |path: std::path::PathBuf, clip: String| async move {
        anyhow::Ok(
            load(&path)
                .await?
                .generate_speech(clone_request(Some(clip), Some(CLIP_TEXT)))
                .await?
                .bytes,
        )
    };
    let want = speak(checkpoint.path().to_path_buf(), clip.clone()).await?;
    assert_eq!(speak(gguf, clip).await?, want);
    Ok(())
}

// llama.cpp's conversion of the tiny checkpoint speaks as the checkpoint, named or detected, and cannot continue a clip
#[tokio::test]
async fn a_qwen3_tts_gguf_speaks_as_its_checkpoint() -> anyhow::Result<()> {
    let checkpoint = tiny_qwen3_tts_checkpoint(BASE, SMALL)?;
    let talker = tiny_qwen3_tts_gguf(checkpoint.path())?;
    let clip = tone_base64(220.0);
    let want = load(checkpoint.path())
        .await?
        .generate_speech(clone_request(Some(clip.clone()), None))
        .await?;
    let gguf = load(&talker).await?;
    let got = gguf
        .generate_speech(clone_request(Some(clip.clone()), None))
        .await?;
    let (want, got) = (pcm16(&want.bytes), pcm16(&got.bytes));
    assert_eq!(want.len(), got.len());
    let worst = want
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .max()
        .unwrap_or(0);
    assert!(
        worst <= GGUF_PCM_TOLERANCE,
        "samples differ by up to {worst}"
    );

    let dir = tempfile::tempdir()?;
    for file in [
        &talker,
        &talker.with_file_name("mmproj-Qwen3-TTS-tiny-F32.gguf"),
    ] {
        std::fs::copy(file, dir.path().join(file.file_name().unwrap()))?;
    }
    let detected = ModelBuilder::new(dir.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let detected = detected
        .generate_speech(clone_request(Some(clip.clone()), None))
        .await?;
    assert!(!detected.bytes.is_empty());

    let continued = error_text(
        gguf.generate_speech(clone_request(Some(clip), Some(CLIP_TEXT)))
            .await,
    );
    assert!(continued.contains("no codec encoder"), "{continued}");
    Ok(())
}
