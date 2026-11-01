//! Voxtral's transcription mode, as mistral_common encodes a transcription request: the clip in an instruction turn
//! with no text, then `lang:<code>` when the language is given, then `[TRANSCRIBE]`; the answer is the transcript.

use inference_nn::loaders::{AudioTranscriber, TranscriptText};

use super::inputs_processor::{AUDIO_TOKEN, BEGIN_AUDIO};

// the languages the model card lists, by ISO-639-1 code
const LANGUAGES: [(&str, &str); 8] = [
    ("en", "English"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("pt", "Portuguese"),
    ("hi", "Hindi"),
    ("de", "German"),
    ("nl", "Dutch"),
    ("it", "Italian"),
];

fn prompt(language: Option<&str>) -> anyhow::Result<String> {
    let language = match language {
        None => String::new(),
        Some(code) => {
            let Some((code, _)) = LANGUAGES.iter().find(|(known, name)| {
                known.eq_ignore_ascii_case(code) || name.eq_ignore_ascii_case(code)
            }) else {
                anyhow::bail!("Voxtral does not transcribe language `{code}`");
            };
            format!("lang:{code}")
        }
    };
    Ok(format!(
        "<s>[INST]{BEGIN_AUDIO}{AUDIO_TOKEN}[/INST]{language}[TRANSCRIBE]"
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
    fn a_language_precedes_the_transcribe_token() -> anyhow::Result<()> {
        assert_eq!(
            prompt(None)?,
            "<s>[INST][BEGIN_AUDIO][AUDIO][/INST][TRANSCRIBE]"
        );
        assert!(prompt(Some("en"))?.ends_with("[/INST]lang:en[TRANSCRIBE]"));
        assert!(prompt(Some("German"))?.ends_with("[/INST]lang:de[TRANSCRIBE]"));
        assert!(prompt(Some("ja")).is_err());
        Ok(())
    }
}
