//! Whisper's greedy decoding of one 30 s window, as transformers' and OpenAI's references do it: the prompt, the
//! suppressed tokens, and the timestamp rules (timestamps in pairs, never backwards, forced when their probability
//! mass beats any text token).

use super::config::WhisperGenerationConfig;

// the time one timestamp token step stands for
pub(super) const SECONDS_PER_TIMESTAMP: f64 = 0.02;

/// The special ids a checkpoint's generation config gives.
#[derive(Debug, Clone)]
pub(super) struct SpecialTokens {
    pub start: u32,
    pub eot: u32,
    pub no_timestamps: u32,
    pub timestamp_begin: u32,
    pub transcribe: Option<u32>,
    pub begin_suppress: Vec<u32>,
    pub suppress: Vec<u32>,
    pub max_initial_timestamp_index: Option<usize>,
}

impl SpecialTokens {
    pub fn new(generation: &WhisperGenerationConfig) -> Self {
        Self {
            start: generation.decoder_start_token_id,
            eot: generation.eos_token_id,
            no_timestamps: generation.no_timestamps_token_id,
            timestamp_begin: generation.no_timestamps_token_id + 1,
            transcribe: generation.task_to_id.get("transcribe").copied(),
            begin_suppress: generation.begin_suppress_tokens.clone(),
            suppress: generation.suppress_tokens.clone(),
            max_initial_timestamp_index: generation.max_initial_timestamp_index,
        }
    }

    pub fn is_timestamp(&self, token: u32) -> bool {
        token >= self.timestamp_begin
    }
}

fn mask(logits: &mut [f32], range: std::ops::Range<usize>) {
    let end = range.end.min(logits.len());
    for value in &mut logits[range.start.min(end)..end] {
        *value = f32::NEG_INFINITY;
    }
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|value| (value - max).exp()).sum();
    let log_sum = max + sum.ln();
    logits.iter().map(|value| value - log_sum).collect()
}

fn log_sum_exp(values: &[f32]) -> f32 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return max;
    }
    max + values
        .iter()
        .map(|value| (value - max).exp())
        .sum::<f32>()
        .ln()
}

/// transformers' `SuppressTokens`, `SuppressTokensAtBegin` and `WhisperTimeStampLogitsProcessor`, applied to one
/// step's logits; `generated` is what this window decoded after its prompt.
pub(super) fn apply_rules(
    special: &SpecialTokens,
    logits: &mut [f32],
    generated: &[u32],
    timestamps: bool,
) {
    for &token in &special.suppress {
        mask(logits, token as usize..token as usize + 1);
    }
    // the control tokens between EOT and the timestamps (start, languages, tasks) are prompt, never text
    mask(
        logits,
        special.eot as usize + 1..special.timestamp_begin as usize,
    );
    if generated.is_empty() {
        for &token in &special.begin_suppress {
            mask(logits, token as usize..token as usize + 1);
        }
    }
    let begin = special.timestamp_begin as usize;
    if !timestamps {
        mask(logits, begin..logits.len());
        return;
    }
    mask(
        logits,
        special.no_timestamps as usize..special.no_timestamps as usize + 1,
    );
    let last_was_timestamp = generated.last().is_some_and(|&t| special.is_timestamp(t));
    let penultimate_was_timestamp =
        generated.len() < 2 || special.is_timestamp(generated[generated.len() - 2]);
    if last_was_timestamp {
        if penultimate_was_timestamp {
            // a pair is closed: text comes next
            mask(logits, begin..logits.len());
        } else {
            // an open timestamp is closed by another or ends the window
            mask(logits, 0..special.eot as usize);
        }
    }
    if let Some(&last) = generated.iter().rev().find(|&&t| special.is_timestamp(t)) {
        // timestamps never go backwards; a closing one may repeat its opener
        let floor = if last_was_timestamp && !penultimate_was_timestamp {
            last
        } else {
            last + 1
        };
        mask(logits, begin..floor as usize);
    }
    if generated.is_empty() {
        mask(logits, 0..begin);
        if let Some(index) = special.max_initial_timestamp_index {
            mask(logits, begin + index + 1..logits.len());
        }
    }
    let logprobs = log_softmax(logits);
    let timestamp_mass = log_sum_exp(&logprobs[begin..]);
    let best_text = logprobs[..begin]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    if timestamp_mass > best_text {
        mask(logits, 0..begin);
    }
}

pub(super) fn argmax(logits: &[f32]) -> u32 {
    let (index, _) = logits
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 { (i, v) } else { best }
        });
    u32::try_from(index).expect("a vocabulary index fits u32")
}

/// One timestamped span of a window's tokens: its text tokens and its times in seconds from the window's start.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct WindowSegment {
    pub tokens: Vec<u32>,
    pub start: f64,
    pub end: f64,
}

/// A decoded window cut at its timestamp pairs, as OpenAI's `transcribe` does, and how many timestamp steps of the
/// window it consumed (`None`: the whole window).
pub(super) fn split_segments(
    special: &SpecialTokens,
    tokens: &[u32],
    window_seconds: f64,
) -> (Vec<WindowSegment>, Option<u32>) {
    let time = |token: u32| f64::from(token - special.timestamp_begin) * SECONDS_PER_TIMESTAMP;
    let is_ts = |i: usize| special.is_timestamp(tokens[i]);
    let n = tokens.len();
    let single_ending = n >= 2 && !is_ts(n - 2) && is_ts(n - 1);
    let cuts: Vec<usize> = (1..n).filter(|&i| is_ts(i - 1) && is_ts(i)).collect();
    let text = |span: &[u32]| {
        span.iter()
            .copied()
            .filter(|&t| !special.is_timestamp(t))
            .collect::<Vec<_>>()
    };
    if cuts.is_empty() {
        let last_ts = tokens
            .iter()
            .rev()
            .find(|&&t| special.is_timestamp(t))
            .copied();
        let end = last_ts
            .filter(|&t| t > special.timestamp_begin)
            .map_or(window_seconds, time);
        let tokens = text(tokens);
        let segments = if tokens.is_empty() {
            Vec::new()
        } else {
            vec![WindowSegment {
                tokens,
                start: 0.0,
                end,
            }]
        };
        return (segments, None);
    }
    let mut bounds = cuts.clone();
    if single_ending {
        bounds.push(n);
    }
    let mut segments = Vec::new();
    let mut last = 0;
    for &bound in &bounds {
        let span = &tokens[last..bound];
        if let (Some(&first), Some(&end)) = (span.first(), span.last())
            && special.is_timestamp(first)
            && special.is_timestamp(end)
        {
            let tokens = text(span);
            if !tokens.is_empty() {
                segments.push(WindowSegment {
                    tokens,
                    start: time(first),
                    end: time(end),
                });
            }
        }
        last = bound;
    }
    let consumed = if single_ending {
        None
    } else {
        // the last closed pair's end: the next window starts there
        Some(tokens[last - 1] - special.timestamp_begin)
    };
    (segments, consumed)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EOT: u32 = 10;
    const NO_TS: u32 = 15;
    const TS: u32 = NO_TS + 1;
    const VOCAB: usize = 40;

    fn special() -> SpecialTokens {
        SpecialTokens {
            start: 11,
            eot: EOT,
            no_timestamps: NO_TS,
            timestamp_begin: TS,
            transcribe: Some(13),
            begin_suppress: vec![2, EOT],
            suppress: vec![1],
            max_initial_timestamp_index: Some(3),
        }
    }

    fn allowed(logits: &[f32]) -> Vec<u32> {
        (0..logits.len())
            .filter(|&i| logits[i].is_finite())
            .map(|i| u32::try_from(i).unwrap())
            .collect()
    }

    // the first token is a timestamp no later than max_initial_timestamp_index
    #[test]
    fn a_window_opens_on_an_early_timestamp() {
        let mut logits = vec![0.0; VOCAB];
        apply_rules(&special(), &mut logits, &[], true);
        assert_eq!(allowed(&logits), [TS, TS + 1, TS + 2, TS + 3]);
    }

    // after a pair, text; after an open timestamp, a closing one at or past it, or the end
    #[test]
    fn timestamps_come_in_pairs_and_never_go_back() {
        let mut logits = vec![0.0; VOCAB];
        apply_rules(&special(), &mut logits, &[TS + 2, 5, TS + 4, TS + 4], true);
        assert!(allowed(&logits).iter().all(|&t| t < TS));
        let mut logits = vec![0.0; VOCAB];
        logits[5] = 10.0;
        apply_rules(&special(), &mut logits, &[TS + 2, 5, 6, TS + 4], true);
        assert!(allowed(&logits).iter().all(|&t| t == EOT || t >= TS + 4));
    }

    #[test]
    fn pairs_cut_segments_and_a_single_ending_consumes_the_window() {
        let tokens = [TS, 3, 4, TS + 50, TS + 50, 5, TS + 100];
        let (segments, consumed) = split_segments(&special(), &tokens, 30.0);
        assert_eq!(consumed, None);
        assert_eq!(segments.len(), 2);
        assert_eq!((segments[0].start, segments[0].end), (0.0, 1.0));
        assert_eq!(segments[1].tokens, [5]);
        let (segments, consumed) = split_segments(&special(), &[TS, 3, TS + 50, TS + 50, 4], 30.0);
        assert_eq!(consumed, Some(50));
        assert_eq!(segments.len(), 1);
    }
}
