//! Kokoro through the engine on a tiny random-weight checkpoint: voices, speed, seeds, chunking and request errors.

use inference::{
    AudioResponseFormat, ModelBuilder, ModelDType, SpeechGenerationRequest, SpeechLoaderType,
    SpeechModelBuilder,
};

#[path = "../support/kokoro_tiny.rs"]
mod support;
use support::{VOICES, tiny_kokoro_checkpoint, tiny_kokoro_gguf};

// invented phoneme strings over Kokoro's vocabulary
const PHONEMES: &str = "həlˈoʊ wˈɜːld.";
const TEXT: &str = "Hello world, this is Kokoro reading text.";
const JAPANESE_TEXT: &str = "科学者たちが発表しました。";
const SEED: u64 = 7;
const WAV_HEADER: usize = 44;
const SAMPLE_RATE: u32 = 24_000;
// the tiny config's context is 64 tokens, so this splits into several chunks
const LONG_WORDS: usize = 30;
// enough runs that ops raced across threads on one device would show
const REPEATS: usize = 4;
// enough copies of TEXT to pass the 510-phoneme context
const LONG_REPEATS: usize = 20;

fn request(phonemes: Option<&str>) -> SpeechGenerationRequest {
    SpeechGenerationRequest {
        phonemes: phonemes.map(str::to_string),
        seed: Some(SEED),
        ..SpeechGenerationRequest::new("hello world", AudioResponseFormat::Pcm)
    }
}

fn error_text<T, E: std::fmt::Display>(result: Result<T, E>) -> String {
    match result {
        Ok(_) => panic!("the request should have failed"),
        Err(err) => err.to_string(),
    }
}

fn samples(bytes: &[u8]) -> usize {
    bytes.len() / 2
}

#[tokio::test]
async fn kokoro_speaks_phonemes_with_voices_speeds_and_seeds() -> anyhow::Result<()> {
    let checkpoint = tiny_kokoro_checkpoint()?;
    let model = SpeechModelBuilder::new(
        checkpoint.path().to_string_lossy(),
        SpeechLoaderType::Kokoro,
    )
    .with_dtype(ModelDType::F32)
    .with_force_cpu()
    .build()
    .await?;

    let base = model.generate_speech(request(Some(PHONEMES))).await?;
    assert!(
        base.content_type.contains(&format!("rate={SAMPLE_RATE}")),
        "{}",
        base.content_type
    );
    assert!(samples(&base.bytes) > 0);
    assert_eq!(
        model.generate_speech(request(Some(PHONEMES))).await?.bytes,
        base.bytes
    );

    let reseeded = SpeechGenerationRequest {
        seed: Some(SEED + 1),
        ..request(Some(PHONEMES))
    };
    assert_ne!(model.generate_speech(reseeded).await?.bytes, base.bytes);

    // the default is the first voice, so naming it changes nothing and naming the other does
    let first = SpeechGenerationRequest {
        voice: Some(VOICES[0].into()),
        ..request(Some(PHONEMES))
    };
    assert_eq!(model.generate_speech(first).await?.bytes, base.bytes);
    let second = SpeechGenerationRequest {
        voice: Some(VOICES[1].into()),
        ..request(Some(PHONEMES))
    };
    assert_ne!(model.generate_speech(second).await?.bytes, base.bytes);
    let blend = SpeechGenerationRequest {
        voice: Some(VOICES[..2].join(",")),
        ..request(Some(PHONEMES))
    };
    assert!(samples(&model.generate_speech(blend).await?.bytes) > 0);

    let slow = SpeechGenerationRequest {
        speed: Some(0.5),
        ..request(Some(PHONEMES))
    };
    assert!(samples(&model.generate_speech(slow).await?.bytes) >= samples(&base.bytes));

    let long = vec![PHONEMES; LONG_WORDS].join(" ");
    assert!(
        samples(&model.generate_speech(request(Some(&long))).await?.bytes) > samples(&base.bytes)
    );

    let wav = SpeechGenerationRequest {
        response_format: AudioResponseFormat::Wav,
        ..request(Some(PHONEMES))
    };
    assert_eq!(
        model.generate_speech(wav).await?.bytes.len(),
        base.bytes.len() + WAV_HEADER
    );

    let unknown = SpeechGenerationRequest {
        voice: Some("no_such_voice".into()),
        ..request(Some(PHONEMES))
    };
    let err = error_text(model.generate_speech(unknown).await);
    assert!(err.contains("no_such_voice"), "{err}");
    let no_language = SpeechGenerationRequest {
        voice: Some(VOICES[3].into()),
        ..request(None)
    };
    let err = error_text(model.generate_speech(no_language).await);
    assert!(err.contains("phonemes"), "{err}");
    let bad_speed = SpeechGenerationRequest {
        speed: Some(0.),
        ..request(Some(PHONEMES))
    };
    assert!(model.generate_speech(bad_speed).await.is_err());
    Ok(())
}

// No architecture named: the auto loader reads config.json and picks Kokoro, giving the same audio
#[tokio::test]
async fn kokoro_is_detected_from_its_config() -> anyhow::Result<()> {
    let checkpoint = tiny_kokoro_checkpoint()?;
    let path = checkpoint.path().to_string_lossy();
    let detected = ModelBuilder::new(&path)
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let named = SpeechModelBuilder::new(&path, SpeechLoaderType::Kokoro)
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    assert_eq!(
        detected
            .generate_speech(request(Some(PHONEMES)))
            .await?
            .bytes,
        named.generate_speech(request(Some(PHONEMES))).await?.bytes
    );
    Ok(())
}

// An F32 GGUF speaks as its checkpoint, named or auto-detected, alone in a directory or beside the release files
#[tokio::test]
async fn a_kokoro_gguf_speaks_as_its_checkpoint() -> anyhow::Result<()> {
    let checkpoint = tiny_kokoro_checkpoint()?;
    let speak = |model: inference::Model| async move {
        anyhow::Ok(model.generate_speech(request(Some(PHONEMES))).await?.bytes)
    };
    let named = |path: String| {
        SpeechModelBuilder::new(path, SpeechLoaderType::Kokoro)
            .with_dtype(ModelDType::F32)
            .with_force_cpu()
            .build()
    };
    let dir = checkpoint.path().to_string_lossy().to_string();
    let want = speak(named(dir.clone()).await?).await?;
    let gguf = tiny_kokoro_gguf(checkpoint.path(), checkpoint.path())?;
    let alone = tempfile::tempdir()?;
    std::fs::copy(&gguf, alone.path().join(gguf.file_name().unwrap()))?;
    let gguf = gguf.to_string_lossy().to_string();
    assert_eq!(speak(named(gguf.clone()).await?).await?, want);
    for path in [gguf.clone(), alone.path().to_string_lossy().to_string()] {
        let detected = ModelBuilder::new(&path)
            .with_dtype(ModelDType::F32)
            .with_force_cpu()
            .build()
            .await?;
        assert_eq!(speak(detected).await?, want, "{path}");
    }
    assert_eq!(speak(named(dir).await?).await?, want);
    Ok(())
}

// On the build's default device (the GPU under `cuda`), one seed gives one waveform, request after request
#[tokio::test]
async fn kokoro_repeats_exactly_on_the_default_device() -> anyhow::Result<()> {
    let checkpoint = tiny_kokoro_checkpoint()?;
    let model = SpeechModelBuilder::new(
        checkpoint.path().to_string_lossy(),
        SpeechLoaderType::Kokoro,
    )
    .with_dtype(ModelDType::F32)
    .build()
    .await?;
    let long = vec![PHONEMES; LONG_WORDS].join(" ");
    let first = model.generate_speech(request(Some(&long))).await?.bytes;
    for _ in 0..REPEATS {
        assert_eq!(
            model.generate_speech(request(Some(&long))).await?.bytes,
            first
        );
    }
    Ok(())
}

// Without `phonemes` the input text is phonemized for the voice's language, and speaks as those phonemes would
#[tokio::test]
async fn kokoro_reads_english_text() -> anyhow::Result<()> {
    use inference_models_speech::kokoro::g2p::{phonemes, voice_language};

    let checkpoint = tiny_kokoro_checkpoint()?;
    let model = SpeechModelBuilder::new(
        checkpoint.path().to_string_lossy(),
        SpeechLoaderType::Kokoro,
    )
    .with_dtype(ModelDType::F32)
    .with_force_cpu()
    .build()
    .await?;
    for voice in &VOICES[..2] {
        let text = SpeechGenerationRequest {
            input: TEXT.into(),
            voice: Some((*voice).into()),
            ..request(None)
        };
        let spoken = model.generate_speech(text).await?.bytes;
        assert!(samples(&spoken) > 0);
        // English renders without consulting the vocabulary
        let ps = phonemes(TEXT, voice_language(voice)?, |_| true)?;
        let explicit = SpeechGenerationRequest {
            voice: Some((*voice).into()),
            ..request(Some(&ps))
        };
        assert_eq!(
            model.generate_speech(explicit).await?.bytes,
            spoken,
            "{voice}"
        );
    }
    let japanese = SpeechGenerationRequest {
        input: JAPANESE_TEXT.into(),
        voice: Some(VOICES[2].into()),
        ..request(None)
    };
    assert!(samples(&model.generate_speech(japanese).await?.bytes) > 0);
    // text past the model's context is split into pieces that each fit
    let long = SpeechGenerationRequest {
        input: TEXT.repeat(LONG_REPEATS),
        voice: Some(VOICES[0].into()),
        ..request(None)
    };
    assert!(model.generate_speech(long).await.is_ok());
    let blank = SpeechGenerationRequest {
        input: "  \n ".into(),
        ..request(None)
    };
    assert!(model.generate_speech(blank).await.is_err());
    Ok(())
}
