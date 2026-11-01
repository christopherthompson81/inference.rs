//! Audio Flamingo 3's transcription prompt, as its processor's `apply_transcription_request` writes it; the answer is
//! the transcript. The reference takes no language, so a requested one is only a hint.

use inference_nn::loaders::{AudioTranscriber, TranscriptText};

use super::inputs_processor::SOUND_TOKEN;

const INSTRUCTION: &str = "Transcribe the input speech.";

fn prompt(_language: Option<&str>) -> anyhow::Result<String> {
    Ok(format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{SOUND_TOKEN}{INSTRUCTION}<|im_end|>\n<|im_start|>assistant\n"
    ))
}

fn parse(answer: &str) -> TranscriptText {
    TranscriptText {
        language: None,
        text: answer.trim().to_string(),
    }
}

pub const AUDIO_FLAMINGO3: AudioTranscriber = AudioTranscriber { prompt, parse };
