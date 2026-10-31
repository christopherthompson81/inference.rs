//! Word times from the decoder's cross-attention, as OpenAI's `timing.find_alignment` takes them: the alignment
//! heads' scores over the window's audio, softmaxed, standardized over tokens, median-filtered over time and
//! averaged, then a DTW path from tokens to audio frames.

use super::decode::WindowSegment;
use crate::TimedText;

// the encoder's positions per second of audio
pub(super) const POSITIONS_PER_SECOND: f64 = 50.0;
const MEDIAN_FILTER_WIDTH: usize = 7;
// OpenAI's caps on a word's length: the median word's, at most this, and twice it
const MAX_MEDIAN_SECONDS: f64 = 0.7;
const SENTENCE_END_MARKS: &str = ".\u{3002}!\u{ff01}?\u{ff1f}";
// punctuation that joins the word after it, or the word before it, as OpenAI's `merge_punctuations` sets them
const PREPEND_PUNCTUATION: &str = "\"'\u{201c}\u{bf}([{-";
const APPEND_PUNCTUATION: &str =
    "\"'.\u{3002},\u{ff0c}!\u{ff01}?\u{ff1f}:\u{ff1a}\u{201d})]}\u{3001}";
// how far a first or last word may run past its segment's timestamp before the timestamp wins
const SEGMENT_SLACK_SECONDS: f64 = 0.5;
// after a pause longer than this many median words, the first words are capped
const PAUSE_MEDIANS: f64 = 4.0;

/// One alignment word: its text (leading space kept), its token count, and its times from the window's start.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct WordTiming {
    pub word: String,
    pub tokens: usize,
    pub start: f64,
    pub end: f64,
}

fn round2(seconds: f64) -> f64 {
    (seconds * 100.0).round() / 100.0
}

// OpenAI's `merge_punctuations`: an opening mark joins the next word, a closing one the previous, keeping that
// word's times; the emptied entry stays for its tokens' place
fn merge_punctuation(alignment: &mut [WordTiming]) {
    if alignment.len() >= 2 {
        let mut j = alignment.len() - 1;
        for i in (0..alignment.len() - 1).rev() {
            let previous = alignment[i].word.clone();
            if previous.starts_with(' ') && PREPEND_PUNCTUATION.contains(previous.trim()) {
                alignment[j].word = format!("{previous}{}", alignment[j].word);
                alignment[j].tokens += alignment[i].tokens;
                alignment[i].word.clear();
                alignment[i].tokens = 0;
            } else {
                j = i;
            }
        }
    }
    let mut i = 0;
    for j in 1..alignment.len() {
        if !alignment[i].word.ends_with(' ')
            && APPEND_PUNCTUATION.contains(alignment[j].word.as_str())
        {
            let following = std::mem::take(&mut alignment[j].word);
            alignment[i].word.push_str(&following);
            alignment[i].tokens += alignment[j].tokens;
            alignment[j].tokens = 0;
        } else {
            i = j;
        }
    }
}

/// OpenAI's `add_word_timestamps` past the alignment: words at sentence ends and after pauses capped, punctuation
/// merged, each segment's words moved to absolute time and held to its timestamps. `last_speech` is the previous
/// word's end in seconds.
pub(super) fn place_words(
    mut alignment: Vec<WordTiming>,
    segments: &[WindowSegment],
    offset: f64,
    mut last_speech: f64,
) -> Vec<TimedText> {
    let mut durations: Vec<f64> = alignment
        .iter()
        .map(|w| w.end - w.start)
        .filter(|d| *d != 0.0)
        .collect();
    durations.sort_by(f64::total_cmp);
    let median = if durations.is_empty() {
        0.0
    } else if durations.len() % 2 == 1 {
        durations[durations.len() / 2]
    } else {
        (durations[durations.len() / 2 - 1] + durations[durations.len() / 2]) / 2.0
    }
    .min(MAX_MEDIAN_SECONDS);
    let max_duration = median * 2.0;
    if !durations.is_empty() {
        for i in 1..alignment.len() {
            if alignment[i].end - alignment[i].start > max_duration {
                if SENTENCE_END_MARKS.contains(alignment[i].word.as_str()) {
                    alignment[i].end = alignment[i].start + max_duration;
                } else if SENTENCE_END_MARKS.contains(alignment[i - 1].word.as_str()) {
                    alignment[i].start = alignment[i].end - max_duration;
                }
            }
        }
    }
    merge_punctuation(&mut alignment);
    let mut placed = Vec::new();
    let mut index = 0;
    for segment in segments {
        let mut saved = 0;
        let mut words: Vec<TimedText> = Vec::new();
        while index < alignment.len() && saved < segment.tokens.len() {
            let timing = &alignment[index];
            if !timing.word.is_empty() {
                words.push(TimedText {
                    text: timing.word.clone(),
                    start: round2(offset + timing.start),
                    end: round2(offset + timing.end),
                });
            }
            saved += timing.tokens;
            index += 1;
        }
        let (segment_start, segment_end) = (offset + segment.start, offset + segment.end);
        if let Some(first) = words.first() {
            let long_first = first.end - first.start > max_duration
                || (words.len() > 1 && words[1].end - first.start > max_duration * 2.0);
            if first.end - last_speech > median * PAUSE_MEDIANS && long_first {
                if words.len() > 1 && words[1].end - words[1].start > max_duration {
                    let boundary = (words[1].end / 2.0).max(words[1].end - max_duration);
                    words[0].end = boundary;
                    words[1].start = boundary;
                }
                words[0].start = (words[0].end - max_duration).max(0.0);
            }
            if segment_start < words[0].end
                && segment_start - SEGMENT_SLACK_SECONDS > words[0].start
            {
                words[0].start = (words[0].end - median).min(segment_start).max(0.0);
            }
            let last = words.len() - 1;
            // the pause reference is the segment's end: its timestamp when that wins, else its last word's
            last_speech = if segment_end > words[last].start
                && segment_end + SEGMENT_SLACK_SECONDS < words[last].end
            {
                words[last].end = (words[last].start + median).max(segment_end);
                segment_end
            } else {
                words[last].end
            };
        }
        placed.extend(words);
    }
    placed
}

/// `heads` gives each alignment head's `(tokens, frames)` scores (row-major) over the window's real audio.
pub(super) fn alignment_matrix(heads: &[Vec<f32>], tokens: usize, frames: usize) -> Vec<f32> {
    let mut mean = vec![0.0f32; tokens * frames];
    for head in heads {
        let mut weights = head.clone();
        for row in weights.chunks_mut(frames) {
            softmax(row);
        }
        // standardize each frame's column over the tokens
        for f in 0..frames {
            let column: Vec<f32> = (0..tokens).map(|t| weights[t * frames + f]).collect();
            let mu = column.iter().sum::<f32>() / tokens as f32;
            let var = column.iter().map(|v| (v - mu).powi(2)).sum::<f32>() / tokens as f32;
            let sd = var.sqrt();
            for t in 0..tokens {
                weights[t * frames + f] = (weights[t * frames + f] - mu) / sd;
            }
        }
        for (t, row) in weights.chunks(frames).enumerate() {
            let filtered = median_filter(row, MEDIAN_FILTER_WIDTH);
            for (f, value) in filtered.into_iter().enumerate() {
                mean[t * frames + f] += value / heads.len() as f32;
            }
        }
    }
    mean
}

fn softmax(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = row
        .iter_mut()
        .map(|v| {
            *v = (*v - max).exp();
            *v
        })
        .sum();
    for v in row.iter_mut() {
        *v /= sum;
    }
}

// OpenAI's `median_filter`: torch's reflect padding (the edge not repeated), then each window's median
fn median_filter(row: &[f32], width: usize) -> Vec<f32> {
    let half = width / 2;
    let n = row.len();
    if n <= half {
        return row.to_vec();
    }
    let at = |i: isize| -> f32 {
        let n = n as isize;
        let i = if i < 0 {
            -i
        } else if i >= n {
            2 * (n - 1) - i
        } else {
            i
        };
        row[i.clamp(0, n - 1) as usize]
    };
    (0..n)
        .map(|i| {
            let mut window: Vec<f32> = (0..width)
                .map(|k| at(i as isize + k as isize - half as isize))
                .collect();
            window.sort_by(f32::total_cmp);
            window[half]
        })
        .collect()
}

/// The DTW path through `cost` (`rows` x `cols`, row-major), as `(row, col)` pairs from the start; ties prefer the
/// diagonal, then the row step.
pub(super) fn dtw(cost: &[f32], rows: usize, cols: usize) -> Vec<(usize, usize)> {
    let width = cols + 1;
    let mut acc = vec![f32::INFINITY; (rows + 1) * width];
    let mut trace = vec![0u8; (rows + 1) * width];
    acc[0] = 0.0;
    for j in 1..=cols {
        for i in 1..=rows {
            let c0 = acc[(i - 1) * width + j - 1];
            let c1 = acc[(i - 1) * width + j];
            let c2 = acc[i * width + j - 1];
            let (c, t) = if c0 < c1 && c0 < c2 {
                (c0, 0)
            } else if c1 < c0 && c1 < c2 {
                (c1, 1)
            } else {
                (c2, 2)
            };
            acc[i * width + j] = cost[(i - 1) * cols + j - 1] + c;
            trace[i * width + j] = t;
        }
    }
    trace[..=cols].fill(2);
    for i in 0..=rows {
        trace[i * width] = 1;
    }
    let (mut i, mut j) = (rows, cols);
    let mut path = Vec::new();
    while i > 0 || j > 0 {
        path.push((i - 1, j.saturating_sub(1)));
        match trace[i * width + j] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            _ => j -= 1,
        }
    }
    path.reverse();
    path
}

/// Each word's start and end in seconds, from the alignment of `[no_timestamps, text tokens...]` rows: a word starts
/// where the path first reaches the row before its first token, as OpenAI's `find_alignment` takes it.
pub(super) fn word_times(
    matrix: &[f32],
    rows: usize,
    frames: usize,
    word_token_counts: &[usize],
) -> Vec<(f64, f64)> {
    let cost: Vec<f32> = matrix.iter().map(|v| -v).collect();
    let path = dtw(&cost, rows, frames);
    // the time each row is first entered
    let mut jump_times = Vec::with_capacity(rows);
    let mut last_row = None;
    for &(row, col) in &path {
        if last_row != Some(row) {
            jump_times.push(col as f64 / POSITIONS_PER_SECOND);
            last_row = Some(row);
        }
    }
    let mut boundaries = vec![0usize];
    for count in word_token_counts {
        boundaries.push(boundaries[boundaries.len() - 1] + count);
    }
    let at = |i: usize| {
        jump_times
            .get(i)
            .copied()
            .unwrap_or_else(|| jump_times.last().copied().unwrap_or(0.0))
    };
    boundaries
        .windows(2)
        .map(|pair| (at(pair[0]), at(pair[1])))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // a diagonal cost walks the diagonal
    #[test]
    fn dtw_follows_the_cheapest_path() {
        let cost = [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        assert_eq!(dtw(&cost, 3, 3), [(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn the_median_filter_reflects_at_the_edges() {
        assert_eq!(
            median_filter(&[5.0, 1.0, 9.0, 2.0], 3),
            [1.0, 5.0, 2.0, 9.0]
        );
    }
}
