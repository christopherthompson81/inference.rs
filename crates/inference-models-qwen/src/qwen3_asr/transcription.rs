//! Qwen3-ASR's transcription prompt and answer: `language <name><asr_text><transcript>`, the language forced by
//! starting the answer with its prefix, as the reference processor does.

use inference_nn::loaders::{AudioTranscriber, TranscriptText};

use super::inputs_processor::{AUDIO_END, AUDIO_START, AUDIO_TOKEN};

const ASR_TEXT: &str = "<asr_text>";
const LANGUAGE_PREFIX: &str = "language ";
// the reference's empty-audio answer
const NO_LANGUAGE: &str = "none";

// the names the model was trained with, by ISO-639-1 code (yue and fil have no two-letter code)
const LANGUAGES: [(&str, &str); 30] = [
    ("ar", "Arabic"),
    ("yue", "Cantonese"),
    ("zh", "Chinese"),
    ("cs", "Czech"),
    ("da", "Danish"),
    ("nl", "Dutch"),
    ("en", "English"),
    ("fil", "Filipino"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("de", "German"),
    ("el", "Greek"),
    ("hi", "Hindi"),
    ("hu", "Hungarian"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("mk", "Macedonian"),
    ("ms", "Malay"),
    ("fa", "Persian"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("es", "Spanish"),
    ("sv", "Swedish"),
    ("th", "Thai"),
    ("tr", "Turkish"),
    ("vi", "Vietnamese"),
];

fn prompt(language: Option<&str>) -> anyhow::Result<String> {
    let answer = match language {
        None => String::new(),
        Some(code) => {
            let Some((_, name)) = LANGUAGES.iter().find(|(known, name)| {
                known.eq_ignore_ascii_case(code) || name.eq_ignore_ascii_case(code)
            }) else {
                anyhow::bail!("Qwen3-ASR does not transcribe language `{code}`");
            };
            format!("{LANGUAGE_PREFIX}{name}{ASR_TEXT}")
        }
    };
    Ok(format!(
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{AUDIO_START}{AUDIO_TOKEN}{AUDIO_END}<|im_end|>\n\
         <|im_start|>assistant\n{answer}"
    ))
}

// the answer's language is its first line before the marker; without the marker, all of it is transcript
fn parse(answer: &str) -> TranscriptText {
    let Some((prefix, text)) = answer.split_once(ASR_TEXT) else {
        return TranscriptText {
            language: None,
            text: answer.trim().to_string(),
        };
    };
    let language = prefix
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(|line| line.strip_prefix(LANGUAGE_PREFIX).unwrap_or(line).trim())
        .filter(|name| !name.is_empty() && !name.eq_ignore_ascii_case(NO_LANGUAGE))
        .map(str::to_lowercase);
    TranscriptText {
        language,
        text: text.trim().to_string(),
    }
}

pub const TRANSCRIBER: AudioTranscriber = AudioTranscriber { prompt, parse };

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_language_hint_starts_the_answer() -> anyhow::Result<()> {
        assert!(prompt(None)?.ends_with("<|im_start|>assistant\n"));
        assert!(prompt(Some("en"))?.ends_with("<|im_start|>assistant\nlanguage English<asr_text>"));
        assert!(prompt(Some("yue"))?.ends_with("language Cantonese<asr_text>"));
        assert!(prompt(Some("xx")).is_err());
        Ok(())
    }

    #[test]
    fn answers_split_into_language_and_text() {
        let parsed = parse("language English<asr_text>The quick brown fox.");
        assert_eq!(parsed.language.as_deref(), Some("english"));
        assert_eq!(parsed.text, "The quick brown fox.");
        assert_eq!(parse("language None<asr_text>").language, None);
        assert_eq!(parse("no marker here").text, "no marker here");
    }
}
