//! MERaLiON-2's transcription prompt: its model card's instruction in the template it was trained on, inside the Gemma
//! turn the repo's chat template writes. The model takes no language, so a requested one is only a hint.

use inference_nn::loaders::{AudioTranscriber, TranscriptText};

use super::inputs_processor::instruction;

const TRANSCRIBE: &str = "Please transcribe this speech.";

fn prompt(_language: Option<&str>) -> anyhow::Result<String> {
    Ok(format!(
        "<bos><start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
        instruction(TRANSCRIBE, 1)
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
    fn the_prompt_is_the_model_cards() -> anyhow::Result<()> {
        assert_eq!(
            prompt(Some("en"))?,
            "<bos><start_of_turn>user\nInstruction: Please transcribe this speech. \nFollow the text instruction \
             based on the following audio: <SpeechHere><end_of_turn>\n<start_of_turn>model\n"
        );
        Ok(())
    }
}
