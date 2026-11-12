//! VibeVoice-ASR's transcription prompt, as its chat template writes one clip, and the parse of its answer: a JSON
//! array of segments, each with its start and end in seconds, a speaker index and the content. The model takes no
//! language; it identifies it.

use inference_nn::loaders::{AudioTranscriber, SpeakerPrompt, TimedSegment, TranscriptText};

use super::inputs_processor::{AUDIO_BOS, AUDIO_DURATION, AUDIO_EOS, AUDIO_TOKEN};

const SYSTEM: &str =
    "You are a helpful assistant that transcribes audio input into text output in JSON format.";
const INSTRUCTION: &str =
    "please transcribe it with these keys: Start time, End time, Speaker ID, Content";
const START: &str = "Start";
const END: &str = "End";
const SPEAKER: &str = "Speaker";
const CONTENT: &str = "Content";

// the template ends with the user turn; the model writes its own assistant header before the JSON
fn prompt(_language: Option<&str>) -> anyhow::Result<String> {
    Ok(format!(
        "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{AUDIO_BOS}{AUDIO_TOKEN}{AUDIO_EOS}\n\
         This is a {AUDIO_DURATION} seconds audio, {INSTRUCTION}<|im_end|>\n"
    ))
}

fn seconds(value: &serde_json::Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|s| s.trim().parse().ok()))
}

/// The segments of an answer, its text their contents joined; an answer that is not the JSON array stands as text.
pub fn parse(answer: &str) -> TranscriptText {
    let json = answer
        .find('[')
        .zip(answer.rfind(']'))
        .and_then(|(from, to)| {
            serde_json::from_str::<Vec<serde_json::Value>>(&answer[from..=to]).ok()
        });
    let Some(items) = json else {
        return TranscriptText {
            text: answer.trim().to_string(),
            ..TranscriptText::default()
        };
    };
    let segments: Vec<TimedSegment> = items
        .iter()
        .filter_map(|item| {
            let text = item.get(CONTENT)?.as_str()?.trim().to_string();
            Some(TimedSegment {
                start: item.get(START).and_then(seconds).unwrap_or(0.0),
                end: item.get(END).and_then(seconds).unwrap_or(0.0),
                speaker: item
                    .get(SPEAKER)
                    .and_then(|s| s.as_u64().or_else(|| s.as_str()?.trim().parse().ok()))
                    .and_then(|s| u32::try_from(s).ok()),
                text,
            })
        })
        .collect();
    let text = segments
        .iter()
        .map(|s| s.text.as_str())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    TranscriptText {
        language: None,
        text,
        segments,
    }
}

/// The same answer serves both: it always names its speakers.
pub const VIBEVOICE_ASR: AudioTranscriber = AudioTranscriber {
    prompt,
    parse,
    speakers: Some(SpeakerPrompt { prompt, parse }),
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_answer_parses_into_timed_speaker_segments() {
        let answer = "assistant\n[{\"Start\":0.0,\"End\":2.5,\"Speaker\":0,\"Content\":\"Hello there.\"},\
                      {\"Start\":\"2.50\",\"End\":4.1,\"Speaker\":1,\"Content\":\"Hi.\"}]\n";
        let parsed = parse(answer);
        assert_eq!(parsed.text, "Hello there. Hi.");
        assert_eq!(parsed.segments.len(), 2);
        assert_eq!(parsed.segments[1].speaker, Some(1));
        assert!((parsed.segments[1].start - 2.5).abs() < 1e-9);
        let raw = parse("not json");
        assert_eq!(raw.text, "not json");
        assert!(raw.segments.is_empty());
    }
}
