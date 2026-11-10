//! A second pass over speech the first one drops where speakers overlap: each long diarized turn another speaker
//! cuts into, holding a stretch with no transcribed word, is transcribed alone, and the words the first pass lacks are
//! added as that turn's speaker's.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use inference_core::{
    AudioInput, RequestCancellation, RequestMessage, Response, SpeakerSegment, TimedText,
    Transcription, TranscriptionOptions, TranscriptionProgress, TranscriptionProgressHook,
};

use crate::{
    api_error::ApiError,
    generation::{run, unexpected, word_speakers},
    types::SharedInferenceRsState,
};

// shorter turns are mostly backchannels, which a window of their own can read in another language
const MIN_TURN_SECONDS: f64 = 1.5;
// a turn the first pass covers without a gap this long gains little from a second look
const MIN_GAP_SECONDS: f64 = 1.0;
const TURN_PAD_SECONDS: f64 = 0.2;
// a second-pass word within this of the same word already there is that word
const SAME_WORD_SECONDS: f64 = 1.0;

/// A progress hook passing on to `inner` that also notes whether the model planned windows (and so takes spans).
pub(crate) fn noting_windows(
    inner: Option<TranscriptionProgressHook>,
) -> (TranscriptionProgressHook, Arc<AtomicBool>) {
    let windowed = Arc::new(AtomicBool::new(false));
    let noted = windowed.clone();
    let hook = TranscriptionProgressHook(Arc::new(move |progress: TranscriptionProgress<'_>| {
        if matches!(progress, TranscriptionProgress::Started { .. }) {
            noted.store(true, Ordering::Relaxed);
        }
        inner.as_ref().is_none_or(|hook| (hook.0)(progress))
    }));
    (hook, windowed)
}

/// The turns worth a second pass, as padded `(start, end, speaker)`, starts in order.
fn picked_turns(
    words: &[TimedText],
    turns: &[SpeakerSegment],
    duration: f64,
) -> Vec<(f64, f64, usize)> {
    let mut picked: Vec<(f64, f64, usize)> = turns
        .iter()
        .filter(|t| t.end - t.start >= MIN_TURN_SECONDS)
        .filter(|t| {
            turns
                .iter()
                .any(|o| o.speaker != t.speaker && o.start < t.end && o.end > t.start)
        })
        .map(|t| {
            (
                (t.start - TURN_PAD_SECONDS).max(0.0),
                (t.end + TURN_PAD_SECONDS).min(duration),
                t.speaker,
            )
        })
        .filter(|&(start, end, _)| longest_gap(words, start, end) >= MIN_GAP_SECONDS)
        .collect();
    picked.sort_by(|a, b| a.0.total_cmp(&b.0));
    picked
}

// the longest stretch of `start..end` with no word starting in it
fn longest_gap(words: &[TimedText], start: f64, end: f64) -> f64 {
    let mut last = start;
    let mut longest: f64 = 0.0;
    for word in words.iter().filter(|w| w.start >= start && w.start < end) {
        longest = longest.max(word.start - last);
        last = word.start;
    }
    longest.max(end - last)
}

// lowercase, punctuation gone (apostrophes kept), so "Yeah," and "yeah" are the same word
fn normalized(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric() || *c == '\'')
        .flat_map(char::to_lowercase)
        .collect()
}

/// The transcription a second pass repeats, over the same model with the same options.
pub(crate) struct PassContext<'a> {
    pub state: &'a SharedInferenceRsState,
    pub model: &'a str,
    pub repr: String,
    pub options: &'a TranscriptionOptions,
    /// The first pass planned windows, so the model takes spans.
    pub windowed: bool,
    pub enabled: bool,
    /// Stops the pass between spans, as it stops the first.
    pub canceled: Option<&'a RequestCancellation>,
}

/// `transcript`'s words with their speakers over `turns`, after a second pass when the context allows one.
pub(crate) async fn overlap_pass(
    ctx: PassContext<'_>,
    audio: AudioInput,
    mut transcript: Transcription,
    turns: &[SpeakerSegment],
) -> Result<(Transcription, Vec<usize>), ApiError> {
    let PassContext {
        state,
        model,
        repr,
        options,
        windowed,
        enabled,
        canceled,
    } = ctx;
    let canceled = canceled.cloned();
    let speakers = word_speakers(&transcript.words, turns);
    // a resumed transcript holds words from there on only
    let resumed = options.resume_at.unwrap_or(0.0);
    let picked: Vec<(f64, f64, usize)> =
        picked_turns(&transcript.words, turns, transcript.duration)
            .into_iter()
            .filter(|&(start, _, _)| start >= resumed)
            .collect();
    if !windowed || !enabled || picked.is_empty() {
        return Ok((transcript, speakers));
    }
    let heard: Arc<Mutex<Vec<Vec<TimedText>>>> =
        Arc::new(Mutex::new(vec![Vec::new(); picked.len()]));
    let into = heard.clone();
    let hook = TranscriptionProgressHook(Arc::new(move |progress: TranscriptionProgress<'_>| {
        if let TranscriptionProgress::Window {
            index, transcript, ..
        } = progress
            && let Some(slot) = into.lock().expect("unpoisoned").get_mut(index)
        {
            *slot = transcript.words.clone();
        }
        canceled.as_ref().is_none_or(|c| !c.is_canceled())
    }));
    let message = RequestMessage::Transcription {
        audio,
        options: TranscriptionOptions {
            resume_at: None,
            max_window_seconds: None,
            spans: Some(picked.iter().map(|&(start, end, _)| (start, end)).collect()),
            ..options.clone()
        },
        progress: Some(hook),
    };
    let Response::Transcription(_) = run(state, model, repr, message).await? else {
        return Err(unexpected(state));
    };
    let heard = std::mem::take(&mut *heard.lock().expect("unpoisoned"));
    let base: Vec<(TimedText, usize)> = transcript.words.drain(..).zip(speakers).collect();
    let before = base.len();
    let words = merged(base, heard, &picked);
    if words.len() > before {
        transcript.text = words
            .iter()
            .map(|(w, _)| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
    }
    let (words, speakers): (Vec<TimedText>, Vec<usize>) = words.into_iter().unzip();
    transcript.words = words;
    Ok((transcript, speakers))
}

// `base` with each span's words it lacks, as that span's speaker's, in time order
fn merged(
    mut words: Vec<(TimedText, usize)>,
    heard: Vec<Vec<TimedText>>,
    picked: &[(f64, f64, usize)],
) -> Vec<(TimedText, usize)> {
    let mut keys: Vec<(String, f64)> = words
        .iter()
        .map(|(w, _)| (normalized(&w.text), w.start))
        .collect();
    for (span_words, &(_, _, speaker)) in heard.into_iter().zip(picked) {
        for word in span_words {
            let key = normalized(&word.text);
            let known = keys
                .iter()
                .any(|(k, at)| *k == key && (at - word.start).abs() <= SAME_WORD_SECONDS);
            if key.is_empty() || known {
                continue;
            }
            keys.push((key, word.start));
            words.push((word, speaker));
        }
    }
    words.sort_by(|a, b| a.0.start.total_cmp(&b.0.start));
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, start: f64) -> TimedText {
        TimedText {
            text: text.into(),
            start,
            end: start + 0.2,
        }
    }

    fn turn(speaker: usize, start: f64, end: f64) -> SpeakerSegment {
        SpeakerSegment {
            speaker,
            start,
            end,
        }
    }

    #[test]
    fn long_overlapped_turns_with_a_word_gap_are_picked() {
        // speaker 0 talks 0-10 s; speaker 1 cuts in 4-7 s (picked: no words 4-7) and 8.0-8.9 s (too short)
        let turns = [
            turn(0, 0.0, 10.0),
            turn(1, 4.0, 7.0),
            turn(1, 8.0, 8.9),
            turn(1, 20.0, 25.0),
        ];
        let words: Vec<TimedText> = (0..20)
            .map(|i| f64::from(i) * 0.5)
            .filter(|t| !(3.5..7.5).contains(t))
            .map(|t| word("w", t))
            .collect();
        let picked = picked_turns(&words, &turns, 30.0);
        // speaker 0's turn holds the 3-7.5 s gap too; speaker 1's lone turn at 20 s overlaps no one
        assert_eq!(picked, [(0.0, 10.2, 0), (3.8, 7.2, 1)]);
    }

    #[test]
    fn a_turn_the_first_pass_covers_is_not_picked() {
        let turns = [turn(0, 0.0, 6.0), turn(1, 2.0, 5.0)];
        let words: Vec<TimedText> = (0..12).map(|i| word("w", f64::from(i) * 0.5)).collect();
        assert!(picked_turns(&words, &turns, 6.0).is_empty());
    }

    #[test]
    fn new_words_join_in_time_order_and_known_ones_do_not() {
        let base = vec![(word("Well,", 1.0), 0), (word("yes", 3.0), 0)];
        let heard = vec![vec![
            word("well", 1.4),
            word("we're", 2.0),
            word("going", 2.3),
            word("yes", 5.0),
        ]];
        let words = merged(base, heard, &[(0.0, 6.0, 1)]);
        let got: Vec<(&str, usize)> = words.iter().map(|(w, s)| (w.text.as_str(), *s)).collect();
        assert_eq!(
            got,
            [
                ("Well,", 0),
                ("we're", 1),
                ("going", 1),
                ("yes", 0),
                ("yes", 1)
            ]
        );
    }
}
