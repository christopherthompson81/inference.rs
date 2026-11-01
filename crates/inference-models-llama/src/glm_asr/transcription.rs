//! GLM-ASR's transcription prompt, as its processor's `apply_transcription_request` writes it; the answer is the
//! transcript. The reference takes no language, so a requested one is only a hint.

use inference_nn::loaders::{AudioTranscriber, TranscriptText};

use super::inputs_processor::{AUDIO_BEGIN, AUDIO_END, AUDIO_TOKEN};

const INSTRUCTION: &str = "Please transcribe this audio into text";

fn prompt(_language: Option<&str>) -> anyhow::Result<String> {
    Ok(format!(
        "<|user|>\n{AUDIO_BEGIN}{AUDIO_TOKEN}{AUDIO_END}<|user|>\n{INSTRUCTION}<|assistant|>\n"
    ))
}

fn parse(answer: &str) -> TranscriptText {
    TranscriptText {
        language: None,
        text: answer.trim().to_string(),
    }
}

pub const TRANSCRIBER: AudioTranscriber = AudioTranscriber { prompt, parse };

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prompt_is_the_processors() -> anyhow::Result<()> {
        assert_eq!(
            prompt(Some("en"))?,
            "<|user|>\n<|begin_of_audio|><|pad|><|end_of_audio|><|user|>\nPlease transcribe this audio into text<|assistant|>\n"
        );
        Ok(())
    }
}
