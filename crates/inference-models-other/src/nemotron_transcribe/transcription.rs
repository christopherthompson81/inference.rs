//! Nemotron 3.5 Transcribe's transcription prompt: the model card's verbatim instruction (its processor's default)
//! with the audio after it and an empty system turn, thinking off, as the checkpoint's template renders it. A
//! requested language is not written in: the instruction asks for the language as spoken.

use inference_nn::loaders::{AudioTranscriber, SpeakerPrompt, TranscriptText};

use super::inputs_processor::AUDIO_TOKEN;
use super::word_timestamps::strip_speaker_tags;

const TRANSCRIBE: &str = "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and \
                          repetitions. Formatting: punctuation and capitalization. Output: transcript only.";
// the model card's prompt for speaker attribution and word timestamps
const TRANSCRIBE_SPEAKERS: &str = "Transcribe this audio with punctuation and capitalization. Include all spoken \
                                   words and verbal fillers. Write numbers in their standard written form. Put speaker \
                                   tags <spk:0> <spk:1> <spk:2> <spk:3> on the transcription.";

fn render(instruction: &str) -> String {
    format!(
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{instruction} {AUDIO_TOKEN}<|im_end|>\n\
         <|im_start|>assistant\n<think></think>"
    )
}

fn prompt(_language: Option<&str>) -> anyhow::Result<String> {
    Ok(render(TRANSCRIBE))
}

fn speaker_prompt(_language: Option<&str>) -> anyhow::Result<String> {
    Ok(render(TRANSCRIBE_SPEAKERS))
}

fn parse_speakers(answer: &str) -> TranscriptText {
    parse(&strip_speaker_tags(answer))
}

fn parse(answer: &str) -> TranscriptText {
    TranscriptText {
        language: None,
        text: answer.trim().to_string(),
    }
}

pub const TRANSCRIBER: AudioTranscriber = AudioTranscriber {
    prompt,
    parse,
    speakers: Some(SpeakerPrompt {
        prompt: speaker_prompt,
        parse: parse_speakers,
    }),
};

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

    #[test]
    fn a_speaker_attributed_answer_parses_to_its_plain_words() {
        let answer = "<spk:0> Hello there. <spk:1> Hi,  how are you? <SPK:0> Fine.";
        assert_eq!(
            parse_speakers(answer).text,
            "Hello there. Hi, how are you? Fine."
        );
    }
}
