//! Nemotron 3.5 Transcribe's word timestamps from its CTC head, ported from transformers'
//! `Nemotron3_5TranscribeProcessor.align_word_timestamps` (NeMo's `MultiSpeakerSOTWordTimestampAligner`): the words of
//! each `<spk:N>` tag force-aligned on the head's frames where the diarization speaker matched to the tag is active,
//! then snapped to that speaker's diarization segments. A "frame" is one of the head's (an encoder step over its
//! upsampling), a "tag" a `<spk:N>`, a "speaker" a diarization speaker.

use std::collections::{BTreeSet, HashMap};

use anyhow::{Context, Result, bail};
use regex::Regex;
use tokenizers::Tokenizer;
use unicode_normalization::UnicodeNormalization;

/// Probability above which a diarization speaker is active.
pub const SPEAKER_ACTIVITY_THRESHOLD: f32 = 0.5;
/// The longest an aligned word lasts, in seconds.
pub const MAX_WORD_DURATION: f64 = 1.0;
/// The most a word snapped to a diarization segment shifts or lasts, in encoder frames of at most
/// `MAX_SNAP_FRAME_SECONDS`.
pub const MAX_WORD_FRAMES: usize = 12;
const MAX_SNAP_FRAME_SECONDS: f64 = 0.08;
// a tag's words align on the frames where its speaker is at least this likely
const MIN_SPEAKER_PROBABILITY: f32 = 0.01;
const PROBABILITY_FLOOR: f32 = 1e-6;
// how much a tag and a speaker with different amounts of speech are kept apart
const AMOUNT_PENALTY: f64 = 0.1;
const NORM_FLOOR: f64 = 1e-8;
const UNK: &str = "<unk>";
const WORD_BOUNDARY: &str = "\u{2581}";
type InScript = fn(char) -> bool;
// the scripts that set a transcript's language, by priority (Unicode name prefixes in the original)
const SCRIPT_LANGUAGES: [(&str, InScript); 6] = [
    ("ko", is_hangul),
    ("ja", is_hiragana),
    ("ja", is_katakana),
    ("zh", is_cjk_unified),
    ("zh", is_cjk_compatibility),
    ("th", is_thai),
];

/// A `<spk:N>` tag's number: `None` for the words before a transcript's first tag, 0 in a transcript without tags.
pub type Tag = Option<u32>;

/// The words of one tag, each with its alignment-tokenizer ids.
#[derive(Debug, Clone, PartialEq)]
pub struct Stream {
    pub tag: Tag,
    pub words: Vec<(String, Vec<u32>)>,
}

/// Log-probabilities of the head's frames over some of its classes: `columns[c]` is the class of column `c`, the blank
/// first.
#[derive(Debug, Clone)]
pub struct LogProbs {
    pub frames: usize,
    pub columns: Vec<u32>,
    /// `frames x columns.len()`, row-major.
    pub values: Vec<f32>,
}

impl LogProbs {
    fn at(&self, frame: usize, column: usize) -> f32 {
        self.values[frame * self.columns.len() + column]
    }
}

/// How the head's frames map onto the recording.
#[derive(Debug, Clone, Copy)]
pub struct Timing {
    /// The recording's spectrogram frames, before padding.
    pub spectrogram_frames: usize,
    /// Spectrogram frames per encoder frame.
    pub subsampling: usize,
    /// Head frames per encoder frame.
    pub upsampling: usize,
    /// Seconds per spectrogram frame.
    pub hop_seconds: f64,
}

impl Timing {
    pub fn encoder_frames(&self) -> usize {
        self.spectrogram_frames.div_ceil(self.subsampling)
    }

    pub fn frames(&self) -> usize {
        self.encoder_frames() * self.upsampling
    }
}

/// The diarization model's logits, one row of `speakers` per spectrogram frame (at least the padded recording's).
#[derive(Debug, Clone)]
pub struct SpeakerLogits {
    pub speakers: usize,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TimedWord {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub tag: Tag,
}

/// A run of one diarization speaker's activity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeakerSegment {
    pub speaker: usize,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WordTimestamps {
    /// Sorted by time.
    pub words: Vec<TimedWord>,
    pub diarization: Vec<SpeakerSegment>,
    /// Each tag's diarization speaker, `None` when unmatched.
    pub tag_to_speaker: Vec<(Tag, Option<usize>)>,
    /// The tags whose words did not fit on their speaker's frames, missing from `words`.
    pub failed: Vec<Tag>,
}

fn speaker_tag() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)<spk:(\d+)>").expect("valid regex"))
}

// speaker, event and style tags, which the head has pieces for
fn kept_tag() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^(?:(?i:<spk:\d+>)|<(?:ev|sty):[^<>]*>)$").expect("valid regex"))
}

// a special token of the decoder's vocabulary, which the head has no piece for
fn special_token() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"<[^<>\s]+>").expect("valid regex"))
}

fn is_hangul(c: char) -> bool {
    matches!(c, '\u{1100}'..='\u{11FF}' | '\u{3131}'..='\u{318E}' | '\u{A960}'..='\u{A97C}' | '\u{AC00}'..='\u{D7A3}'
        | '\u{D7B0}'..='\u{D7FB}')
}

fn is_hiragana(c: char) -> bool {
    matches!(c, '\u{3041}'..='\u{3096}' | '\u{3099}'..='\u{309A}' | '\u{309D}'..='\u{309F}' | '\u{1B001}'..='\u{1B11F}')
}

// with the marks the two kana share, which are named after katakana first
fn is_katakana(c: char) -> bool {
    matches!(c, '\u{309B}'..='\u{309C}' | '\u{30A0}'..='\u{30FF}' | '\u{31F0}'..='\u{31FF}' | '\u{32D0}'..='\u{32FE}'
        | '\u{1B000}')
}

fn is_cjk_unified(c: char) -> bool {
    matches!(c, '\u{3400}'..='\u{4DBF}' | '\u{4E00}'..='\u{9FFF}' | '\u{20000}'..='\u{2A6DF}'
        | '\u{2A700}'..='\u{2EBEF}' | '\u{30000}'..='\u{323AF}')
}

fn is_cjk_compatibility(c: char) -> bool {
    matches!(c, '\u{F900}'..='\u{FAFF}' | '\u{2F800}'..='\u{2FA1F}')
}

fn is_thai(c: char) -> bool {
    matches!(c, '\u{0E01}'..='\u{0E3A}' | '\u{0E3F}'..='\u{0E5B}')
}

/// Letters, marks and numbers of `text`, NFC-normalized.
fn lexical_characters(text: &str) -> Vec<char> {
    use unicode_categories::UnicodeCategories;
    text.nfc()
        .filter(|c| c.is_letter() || c.is_mark() || c.is_number())
        .collect()
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tokenizer
        .encode(text, false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec())
}

// a character is its own piece when the vocabulary has one, else its pieces without the bare word boundary
fn character_token_ids(tokenizer: &Tokenizer, character: char) -> Result<Vec<u32>> {
    let unk = tokenizer.token_to_id(UNK);
    let own = tokenizer.token_to_id(&character.to_string());
    let ids = match own {
        Some(id) if Some(id) != unk => vec![id],
        _ => {
            let boundary = tokenizer.token_to_id(WORD_BOUNDARY);
            encode(tokenizer, &format!(" {character}"))?
                .into_iter()
                .filter(|&id| Some(id) != boundary)
                .collect()
        }
    };
    if unk.is_some_and(|unk| ids.contains(&unk)) {
        bail!("Unrepresentable character: {character:?}");
    }
    Ok(ids)
}

/// Each tag's words with their ids: the decoder's special tokens dropped, a word's ids those of the word with a leading
/// space (the SentencePiece prefix), Chinese and Japanese (`language`, detected from the script when `None`) character
/// by character. Streams in the order of their tags' first words.
pub fn tokenize_transcript(
    tokenizer: &Tokenizer,
    text: &str,
    language: Option<&str>,
    blank: u32,
) -> Result<Vec<Stream>> {
    let text = special_token().replace_all(text, |caps: &regex::Captures<'_>| {
        if kept_tag().is_match(&caps[0]) {
            caps[0].to_string()
        } else {
            " ".to_string()
        }
    });
    let text = text.replace('\u{2014}', " ");
    // [text, tag, text, tag, ...]
    let mut parts: Vec<(Tag, &str)> = Vec::new();
    let mut at = 0;
    let mut tag = None;
    for caps in speaker_tag().captures_iter(&text) {
        let whole = caps.get(0).expect("a match");
        parts.push((tag, &text[at..whole.start()]));
        tag = Some(caps[1].parse::<u32>().context("speaker tag")?);
        at = whole.end();
    }
    let tagged = !parts.is_empty();
    parts.push((tag, &text[at..]));
    let words: Vec<(Tag, &str)> = parts
        .into_iter()
        .map(|(tag, part)| (if tagged { tag } else { Some(0) }, part))
        .flat_map(|(tag, part)| part.split_whitespace().map(move |word| (tag, word)))
        .collect();
    let language = match language {
        Some(language) => Some(language),
        None => SCRIPT_LANGUAGES
            .iter()
            .find(|(_, in_script)| {
                words
                    .iter()
                    .filter(|(_, word)| !kept_tag().is_match(word))
                    .any(|(_, word)| word.chars().any(in_script))
            })
            .map(|(language, _)| *language),
    };
    if language == Some("th") {
        bail!("Word timestamps are not supported for Thai, which needs a word segmenter.");
    }
    let mut streams: Vec<Stream> = Vec::new();
    for (tag, word) in words {
        let units = if matches!(language, Some("zh" | "ja")) && !kept_tag().is_match(word) {
            lexical_characters(word)
                .into_iter()
                .map(|c| Ok((c.to_string(), character_token_ids(tokenizer, c)?)))
                .collect::<Result<Vec<_>>>()?
        } else {
            vec![(word.to_string(), encode(tokenizer, &format!(" {word}"))?)]
        };
        if units
            .iter()
            .any(|(_, ids)| ids.is_empty() || ids.iter().any(|&id| id >= blank))
        {
            bail!("Tokenizer produced IDs outside the non-blank CTC vocabulary.");
        }
        match streams.iter_mut().find(|stream| stream.tag == tag) {
            Some(stream) => stream.words.extend(units),
            None => streams.push(Stream { tag, words: units }),
        }
    }
    Ok(streams)
}

/// `text` without its `<spk:N>` tags, its words single-spaced.
pub fn strip_speaker_tags(text: &str) -> String {
    speaker_tag()
        .replace_all(text, " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Whether a transcript attributes its words with `<spk:N>` tags; one without aligns on every frame, as one speaker.
pub fn has_speaker_tags(text: &str) -> bool {
    speaker_tag().is_match(text)
}

/// The classes the alignment reads: the blank, then the transcript's pieces, sorted.
pub fn classes(streams: &[Stream], blank: u32) -> Vec<u32> {
    let pieces: BTreeSet<u32> = streams
        .iter()
        .flat_map(|stream| stream.words.iter().flat_map(|(_, ids)| ids.iter().copied()))
        .collect();
    std::iter::once(blank).chain(pieces).collect()
}

/// Linear interpolation of `(frames, speakers)` rows onto `frames_out` rows, first and last aligned, as torch's float32
/// `linspace` places them.
fn resample_frames(values: &[f32], speakers: usize, frames_out: usize) -> Vec<f32> {
    let frames = values.len() / speakers;
    if frames == frames_out {
        return values.to_vec();
    }
    let end = (frames - 1) as f32;
    let step = if frames_out > 1 {
        end / (frames_out - 1) as f32
    } else {
        0.0
    };
    let halfway = frames_out / 2;
    let mut out = Vec::with_capacity(frames_out * speakers);
    for i in 0..frames_out {
        let position = if i < halfway {
            step * i as f32
        } else {
            end - step * (frames_out - i - 1) as f32
        };
        let (lower, upper) = (position.floor() as usize, position.ceil() as usize);
        let fraction = position - lower as f32;
        for s in 0..speakers {
            let (a, b) = (values[lower * speakers + s], values[upper * speakers + s]);
            out.push(a * (1.0 - fraction) + b * fraction);
        }
    }
    out
}

/// Each speaker's runs of activity over `is_active` (`frames x speakers`), sorted by time.
fn diarization_segments(
    is_active: &[bool],
    speakers: usize,
    frame_seconds: f64,
    duration: f64,
) -> Vec<SpeakerSegment> {
    let frames = is_active.len() / speakers;
    let mut segments = Vec::new();
    for speaker in 0..speakers {
        let mut start = None;
        for frame in 0..=frames {
            let active = frame < frames && is_active[frame * speakers + speaker];
            match (start, active) {
                (None, true) => start = Some(frame),
                (Some(first), false) => {
                    let segment = SpeakerSegment {
                        speaker,
                        start: first as f64 * frame_seconds,
                        end: (frame as f64 * frame_seconds).min(duration),
                    };
                    if segment.end > segment.start {
                        segments.push(segment);
                    }
                    start = None;
                }
                _ => {}
            }
        }
    }
    segments.sort_by(|a, b| {
        a.start
            .total_cmp(&b.start)
            .then(a.end.total_cmp(&b.end))
            .then(a.speaker.cmp(&b.speaker))
    });
    segments
}

/// Rows assigned to columns maximizing the summed `scores` (`rows x cols`), `min(rows, cols)` of them, as scipy's
/// `linear_sum_assignment`: the Hungarian method on the matrix padded square with zeros.
fn assign(scores: &[Vec<f64>], cols: usize) -> Vec<Option<usize>> {
    let rows = scores.len();
    let n = rows.max(cols);
    if n == 0 {
        return Vec::new();
    }
    let cost = |i: usize, j: usize| {
        if i < rows && j < cols {
            -scores[i][j]
        } else {
            0.0
        }
    };
    // potentials and matching over 1-based indices, column 0 the virtual start
    let (mut u, mut v) = (vec![0f64; n + 1], vec![0f64; n + 1]);
    let (mut row_of, mut way) = (vec![0usize; n + 1], vec![0usize; n + 1]);
    for i in 1..=n {
        row_of[0] = i;
        let mut j0 = 0;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = row_of[j0];
            let (mut delta, mut j1) = (f64::INFINITY, 0);
            for j in 1..=n {
                if !used[j] {
                    let reduced = cost(i0 - 1, j - 1) - u[i0] - v[j];
                    if reduced < minv[j] {
                        minv[j] = reduced;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[row_of[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if row_of[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            row_of[j0] = row_of[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut assigned = vec![None; rows];
    for (j, &i) in row_of.iter().enumerate().skip(1) {
        if i >= 1 && i <= rows && j <= cols {
            assigned[i - 1] = Some(j - 1);
        }
    }
    assigned
}

/// Each stream's diarization speaker, matched one to one on the pieces they share: a speaker collects the greedy
/// pieces decoded while it is active, a tag the pieces of its words, and the IDF-weighted cosine similarity of their
/// counts, penalized when their amounts of speech differ, is maximized.
fn match_tags_to_speakers(
    log_probs: &LogProbs,
    streams: &[Stream],
    speaker_probabilities: &[f32],
    speakers: usize,
) -> Vec<Option<usize>> {
    let classes = log_probs.columns.len();
    let column: HashMap<u32, usize> = log_probs
        .columns
        .iter()
        .enumerate()
        .map(|(c, &class)| (class, c))
        .collect();
    // the blank (column 0) first on ties
    let greedy: Vec<usize> = (0..log_probs.frames)
        .map(|t| {
            let mut best = 0;
            for c in 1..classes {
                if log_probs.at(t, c) > log_probs.at(t, best) {
                    best = c;
                }
            }
            best
        })
        .collect();
    let mut tag_counts = vec![vec![0f64; classes]; streams.len()];
    for (counts, stream) in tag_counts.iter_mut().zip(streams) {
        for id in stream.words.iter().flat_map(|(_, ids)| ids) {
            counts[column[id]] += 1.0;
        }
    }
    let mut speaker_counts = vec![vec![0f64; classes]; speakers];
    for (t, &piece) in greedy.iter().enumerate() {
        let onset = piece != 0 && (t == 0 || greedy[t - 1] != piece);
        if !onset {
            continue;
        }
        for (s, counts) in speaker_counts.iter_mut().enumerate() {
            if speaker_probabilities[t * speakers + s] >= SPEAKER_ACTIVITY_THRESHOLD {
                counts[piece] += 1.0;
            }
        }
    }
    let idf: Vec<f64> = (0..classes)
        .map(|c| {
            let present = speaker_counts
                .iter()
                .filter(|counts| counts[c] > 0.0)
                .count();
            (speakers as f64 / (1 + present) as f64).ln_1p()
        })
        .collect();
    let vectors = |counts: &[Vec<f64>]| -> Vec<Vec<f64>> {
        counts
            .iter()
            .map(|row| row.iter().zip(&idf).map(|(n, w)| (n * w).sqrt()).collect())
            .collect()
    };
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let (tag_vectors, speaker_vectors) = (vectors(&tag_counts), vectors(&speaker_counts));
    let scores: Vec<Vec<f64>> = tag_vectors
        .iter()
        .zip(&tag_counts)
        .map(|(tv, tc)| {
            speaker_vectors
                .iter()
                .zip(&speaker_counts)
                .map(|(sv, sc)| {
                    let dot: f64 = tv.iter().zip(sv).map(|(a, b)| a * b).sum();
                    let cosine = dot / (norm(tv) * norm(sv)).max(NORM_FLOOR);
                    let amounts = (tc.iter().sum::<f64>() + 1.0) / (sc.iter().sum::<f64>() + 1.0);
                    cosine - AMOUNT_PENALTY * amounts.ln().abs()
                })
                .collect()
        })
        .collect();
    assign(&scores, speakers)
}

/// torchaudio's `forced_align`: the best CTC path of `target` (column indices) over `emissions` (`frames x columns`,
/// column 0 the blank), as each frame's column, ties broken as torchaudio breaks them.
fn forced_align(emissions: &[f32], columns: usize, target: &[usize]) -> Vec<usize> {
    let frames = emissions.len() / columns;
    let labels = target.len();
    let states = 2 * labels + 1;
    let emission = |t: usize, c: usize| emissions[t * columns + c];
    let label = |s: usize| {
        if s.is_multiple_of(2) {
            0
        } else {
            target[s / 2]
        }
    };
    let repeats = target.windows(2).filter(|w| w[0] == w[1]).count();
    let mut alphas = [
        vec![f32::NEG_INFINITY; states],
        vec![f32::NEG_INFINITY; states],
    ];
    let mut back = vec![0u8; frames * states];
    let mut start = usize::from(frames <= labels + repeats);
    let mut end = if states == 1 { 1 } else { 2 };
    for (s, alpha) in alphas[0].iter_mut().enumerate().take(end).skip(start) {
        *alpha = emission(0, label(s));
    }
    for t in 1..frames {
        if frames - t <= labels + repeats {
            if start % 2 == 1 && target[start / 2] != target[start / 2 + 1] {
                start += 1;
            }
            start += 1;
        }
        if t <= labels + repeats {
            if end % 2 == 0 && end < 2 * labels && target[end / 2 - 1] != target[end / 2] {
                end += 1;
            }
            end += 1;
        }
        let (cur, prev) = (t % 2, (t - 1) % 2);
        let mut row = vec![f32::NEG_INFINITY; states];
        let mut first = start;
        if start == 0 {
            row[0] = alphas[prev][0] + emission(t, 0);
            back[t * states] = 0;
            first += 1;
        }
        for s in first..end {
            let x0 = alphas[prev][s];
            let x1 = alphas[prev][s - 1];
            let x2 = if s % 2 == 1 && s != 1 && target[s / 2] != target[s / 2 - 1] {
                alphas[prev][s - 2]
            } else {
                f32::NEG_INFINITY
            };
            let (best, step) = if x2 > x1 && x2 > x0 {
                (x2, 2)
            } else if x1 > x0 && x1 > x2 {
                (x1, 1)
            } else {
                (x0, 0)
            };
            back[t * states + s] = step;
            row[s] = best + emission(t, label(s));
        }
        alphas[cur] = row;
    }
    let last = &alphas[(frames - 1) % 2];
    let mut s = if last[states - 1] > last[states - 2] {
        states - 1
    } else {
        states - 2
    };
    let mut path = vec![0; frames];
    for t in (0..frames).rev() {
        path[t] = label(s);
        s -= usize::from(back[t * states + s]);
    }
    path
}

/// torchaudio's `merge_tokens`: each run of one non-blank column, `(start, end)` frames.
fn merge_tokens(path: &[usize]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut at = 0;
    while at < path.len() {
        let mut end = at + 1;
        while end < path.len() && path[end] == path[at] {
            end += 1;
        }
        if path[at] != 0 {
            spans.push((at, end));
        }
        at = end;
    }
    spans
}

/// The start and end of each word, force-aligned on the frames where its speaker's probability is at least
/// `MIN_SPEAKER_PROBABILITY`, that probability's log added to the pieces' log-probabilities. `None` when the words do not
/// fit on those frames.
fn align_words(
    log_probs: &LogProbs,
    words: &[(String, Vec<u32>)],
    probabilities: &[f32],
    frame_seconds: f64,
) -> Option<Vec<TimedWord>> {
    let column: HashMap<u32, usize> = log_probs
        .columns
        .iter()
        .enumerate()
        .map(|(c, &class)| (class, c))
        .collect();
    let target: Vec<usize> = words
        .iter()
        .flat_map(|(_, ids)| ids.iter().map(|id| column[id]))
        .collect();
    let frames: Vec<usize> = (0..log_probs.frames)
        .filter(|&t| probabilities[t] >= MIN_SPEAKER_PROBABILITY)
        .collect();
    let repeats = target.windows(2).filter(|w| w[0] == w[1]).count();
    // a blank must separate repeated pieces
    if frames.len() < target.len() + repeats {
        return None;
    }
    let columns = log_probs.columns.len();
    let mut emissions = Vec::with_capacity(frames.len() * columns);
    for &t in &frames {
        let weight = probabilities[t].max(PROBABILITY_FLOOR).ln();
        emissions.push(log_probs.at(t, 0));
        emissions.extend((1..columns).map(|c| log_probs.at(t, c) + weight));
    }
    let path = forced_align(&emissions, columns, &target);
    let mut spans = merge_tokens(&path).into_iter();
    words
        .iter()
        .map(|(word, ids)| {
            let word_spans: Vec<(usize, usize)> = spans.by_ref().take(ids.len()).collect();
            let (first, last) = (word_spans.first()?, word_spans.last()?);
            let start = frames[first.0] as f64 * frame_seconds;
            let end =
                ((frames[last.1 - 1] + 1) as f64 * frame_seconds).min(start + MAX_WORD_DURATION);
            Some(TimedWord {
                text: word.clone(),
                start,
                end,
                tag: None,
            })
        })
        .collect()
}

/// Snaps one speaker's words, in time order, to its diarization `segments`: the first word of a segment starts with it;
/// a word ends where the next word of its segment starts, else with its segment; a word outside segments ends where the
/// next word starts. Shifts and durations are capped at `max_word_frames` frames of at most `MAX_SNAP_FRAME_SECONDS`.
fn refine_word_timestamps(
    words: &mut [TimedWord],
    segments: &[(f64, f64)],
    frame_seconds: f64,
    max_word_frames: usize,
) {
    let limit = max_word_frames as f64 * frame_seconds.min(MAX_SNAP_FRAME_SECONDS);
    let mut segments = segments.to_vec();
    segments.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let segment_of: Vec<Option<usize>> = words
        .iter()
        .map(|word| {
            let index = segments
                .partition_point(|s| s.0 <= word.start)
                .checked_sub(1)?;
            (word.start < segments[index].1).then_some(index)
        })
        .collect();
    for i in 0..words.len() {
        let Some(segment) = segment_of[i] else {
            continue;
        };
        let first = i == 0 || segment_of[i - 1] != Some(segment);
        let shift = words[i].start - segments[segment].0;
        if first && shift > 0.0 && shift <= limit {
            words[i].start = segments[segment].0;
        }
    }
    for i in 0..words.len() {
        let last_of_segment =
            segment_of[i].is_some() && segment_of.get(i + 1) != Some(&segment_of[i]);
        let end = match segment_of[i] {
            Some(segment) if last_of_segment => segments[segment].1,
            _ => words.get(i + 1).map_or(words[i].end, |next| next.start),
        };
        words[i].end = words[i].start.max(end.min(words[i].start + limit));
    }
}

/// Times each word of `streams` and attributes it to a speaker, from the head's `log_probs` (its `timing.frames()`
/// frames, over `classes(streams)`) and, when given, the diarization model's `speaker_logits`; without them the
/// transcript aligns as a single speaker and no diarization is returned.
pub fn align(
    streams: &[Stream],
    log_probs: &LogProbs,
    speaker_logits: Option<&SpeakerLogits>,
    timing: Timing,
) -> Result<WordTimestamps> {
    if streams.is_empty() {
        bail!("The transcription is empty once its special tokens are removed.");
    }
    let frames = timing.frames();
    if log_probs.frames != frames {
        bail!(
            "{} head frames for a recording of {frames}",
            log_probs.frames
        );
    }
    let duration = timing.spectrogram_frames as f64 * timing.hop_seconds;
    let frame_seconds = duration / frames as f64;
    let mut tag_to_speaker = vec![None; streams.len()];
    let mut speaker_probabilities = None;
    let mut diarization = Vec::new();
    if let Some(logits) = speaker_logits {
        let speakers = logits.speakers;
        let encoder_frames = timing.encoder_frames();
        if logits.values.len() < encoder_frames * timing.subsampling * speakers {
            bail!("speaker logits cover fewer frames than the recording's");
        }
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        // the mean of the probabilities over each encoder frame's spectrogram frames
        let mut per_encoder_frame = vec![0f32; encoder_frames * speakers];
        for e in 0..encoder_frames {
            for s in 0..speakers {
                let sum: f32 = (0..timing.subsampling)
                    .map(|k| sigmoid(logits.values[(e * timing.subsampling + k) * speakers + s]))
                    .sum();
                per_encoder_frame[e * speakers + s] = sum / timing.subsampling as f32;
            }
        }
        let probabilities = resample_frames(&per_encoder_frame, speakers, frames);
        tag_to_speaker = match_tags_to_speakers(log_probs, streams, &probabilities, speakers);
        let is_active: Vec<bool> = logits.values[..timing.spectrogram_frames * speakers]
            .iter()
            .map(|&x| sigmoid(x) > SPEAKER_ACTIVITY_THRESHOLD)
            .collect();
        diarization = diarization_segments(&is_active, speakers, timing.hop_seconds, duration);
        speaker_probabilities = Some((probabilities, speakers));
    }
    let mut words = Vec::new();
    let mut failed = Vec::new();
    for (stream, speaker) in streams.iter().zip(&tag_to_speaker) {
        let probabilities: Vec<f32> = match (&speaker_probabilities, speaker) {
            (None, _) => vec![1.0; frames],
            // an unmatched tag has no frame to align on
            (Some(_), None) => vec![0.0; frames],
            (Some((all, speakers)), Some(s)) => {
                (0..frames).map(|t| all[t * speakers + s]).collect()
            }
        };
        let Some(mut timed) = align_words(log_probs, &stream.words, &probabilities, frame_seconds)
        else {
            failed.push(stream.tag);
            continue;
        };
        let segments: Vec<(f64, f64)> = diarization
            .iter()
            .filter(|segment| Some(segment.speaker) == *speaker)
            .map(|segment| (segment.start, segment.end))
            .collect();
        refine_word_timestamps(
            &mut timed,
            &segments,
            frame_seconds,
            MAX_WORD_FRAMES * timing.upsampling,
        );
        words.extend(timed.into_iter().map(|word| TimedWord {
            tag: stream.tag,
            ..word
        }));
    }
    words.sort_by(|a, b| a.start.total_cmp(&b.start).then(a.end.total_cmp(&b.end)));
    Ok(WordTimestamps {
        words,
        diarization,
        tag_to_speaker: streams.iter().map(|s| s.tag).zip(tag_to_speaker).collect(),
        failed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // three pieces over eight frames: the path follows the peaks, a blank between the repeated pieces
    #[test]
    fn forced_alignment_takes_the_best_path_with_a_blank_between_repeats() {
        let columns = 3;
        // frames' log-probabilities over [blank, a, b]
        let peaks = [1, 1, 0, 1, 0, 2, 2, 0];
        let emissions: Vec<f32> = peaks
            .iter()
            .flat_map(|&p| (0..columns).map(move |c| if c == p { -0.1 } else { -3.0 }))
            .collect();
        let path = forced_align(&emissions, columns, &[1, 1, 2]);
        assert_eq!(path, vec![1, 1, 0, 1, 0, 2, 2, 0]);
        assert_eq!(merge_tokens(&path), vec![(0, 2), (3, 4), (5, 7)]);
    }

    #[test]
    fn the_assignment_maximizes_the_summed_scores_rectangular_either_way() {
        let scores = vec![vec![0.1, 0.9, 0.3], vec![0.8, 0.85, 0.2]];
        assert_eq!(assign(&scores, 3), vec![Some(1), Some(0)]);
        let tall = vec![vec![0.1], vec![0.9], vec![0.5]];
        assert_eq!(assign(&tall, 1), vec![None, Some(0), None]);
    }

    // a segment's first word starts with it, words end at the next word or their segment's end, capped
    #[test]
    fn words_snap_to_their_speakers_segments() {
        let word = |start: f64, end: f64| TimedWord {
            text: String::new(),
            start,
            end,
            tag: None,
        };
        let mut words = vec![word(1.1, 1.3), word(1.5, 1.7), word(3.0, 3.2)];
        refine_word_timestamps(&mut words, &[(1.0, 2.0)], 0.04, 24);
        let times: Vec<(f64, f64)> = words.iter().map(|w| (w.start, w.end)).collect();
        assert_eq!(times, vec![(1.0, 1.5), (1.5, 2.0), (3.0, 3.2)]);
    }

    #[test]
    fn segments_are_each_speakers_runs_of_activity() {
        // frames x 2 speakers
        let active = [true, false, true, true, false, true, false, false];
        let segments = diarization_segments(&active, 2, 0.5, 1.8);
        let runs: Vec<(usize, f64, f64)> = segments
            .iter()
            .map(|s| (s.speaker, s.start, s.end))
            .collect();
        assert_eq!(runs, vec![(0, 0.0, 1.0), (1, 0.5, 1.5)]);
    }

    #[test]
    fn resampling_aligns_the_first_and_last_frames() {
        let out = resample_frames(&[0.0, 1.0], 1, 4);
        let expected = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
        for (got, want) in out.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "{out:?}");
        }
    }

    #[test]
    fn the_script_sets_the_language() {
        assert!(is_hiragana('\u{3042}') && is_katakana('\u{30A2}') && is_cjk_unified('\u{4E2D}'));
        assert!(is_hangul('\u{D55C}') && is_thai('\u{0E01}') && !is_cjk_unified('a'));
    }
}
