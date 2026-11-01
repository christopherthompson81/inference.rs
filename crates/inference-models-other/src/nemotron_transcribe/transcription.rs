//! Nemotron 3.5 Transcribe's transcription prompt: the model card's verbatim instruction (its processor's default)
//! with the audio after it and an empty system turn, thinking off, as the checkpoint's template renders it. A
//! requested language is not written in: the instruction asks for the language as spoken.

use inference_nn::loaders::{AudioTranscriber, TranscriptText};

use super::inputs_processor::AUDIO_TOKEN;

const TRANSCRIBE: &str = "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and \
                          repetitions. Formatting: punctuation and capitalization. Output: transcript only.";

fn prompt(_language: Option<&str>) -> anyhow::Result<String> {
    Ok(format!(
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{TRANSCRIBE} {AUDIO_TOKEN}<|im_end|>\n\
         <|im_start|>assistant\n<think></think>"
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

    // as transformers' `apply_chat_template` renders the processor's default transcription request
    #[test]
    fn the_prompt_is_the_templates_rendering_of_the_default_request() -> anyhow::Result<()> {
        assert_eq!(
            prompt(Some("en"))?,
            "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nAudio transcription. Language: as spoken. Wording: \
             verbatim, including fillers and repetitions. Formatting: punctuation and capitalization. Output: \
             transcript only. <|audio|><|im_end|>\n<|im_start|>assistant\n<think></think>"
        );
        Ok(())
    }
}
