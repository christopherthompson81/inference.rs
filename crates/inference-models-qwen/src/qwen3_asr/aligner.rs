//! Qwen3-ForcedAligner: the Qwen3-ASR model with a classifier over time steps instead of a vocabulary. The prompt is
//! the clip's placeholders, then each word of the transcript followed by two `<timestamp>` slots; the class read at
//! each slot is its word's start or end, in steps of 80 ms. The words are the transcript's letters, numbers and
//! apostrophes, a CJK character each, as the reference splits them (Qwen's `qwen3_forced_aligner.py`).

use inference_nn::loaders::AlignedWord;
use inference_tensor::{Device, Tensor};
use tokenizers::Tokenizer;

const TIMESTAMP: &str = "<timestamp>";
// the slots after each word: its start, then its end
const SLOTS_PER_WORD: usize = 2;
const MS_PER_SECOND: f64 = 1000.0;
// a run of out-of-order times this short snaps to a neighbour; a longer one is interpolated
const SNAP_RUN: usize = 2;
/// The languages the aligner was trained on, by ISO-639-1 code and name (Cantonese has no two-letter code)
pub const LANGUAGES: [(&str, &str); 11] = [
    ("zh", "chinese"),
    ("yue", "cantonese"),
    ("en", "english"),
    ("de", "german"),
    ("es", "spanish"),
    ("fr", "french"),
    ("it", "italian"),
    ("pt", "portuguese"),
    ("ru", "russian"),
    ("ko", "korean"),
    ("ja", "japanese"),
];

fn is_cjk(c: char) -> bool {
    matches!(u32::from(c),
        0x4E00..=0x9FFF
        | 0x3400..=0x4DBF
        | 0x20000..=0x2A6DF
        | 0x2A700..=0x2B73F
        | 0x2B740..=0x2B81F
        | 0x2B820..=0x2CEAF
        | 0xF900..=0xFAFF
        | 0x2F800..=0x2FA1F)
}

fn is_kept(c: char) -> bool {
    c == '\'' || c.is_alphabetic() || c.is_numeric() || is_cjk(c)
}

/// One word the aligner times: as it reads it (punctuation left out) and as the transcript writes it, with the
/// punctuation up to the next word.
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub read: String,
    pub written: String,
}

/// The words the aligner times: a CJK character each, else the runs between spaces, punctuation left out.
pub fn split_words(text: &str) -> Vec<Word> {
    let text = text.trim();
    // each word's text and the byte where its first kept character starts
    let mut words: Vec<(String, usize)> = Vec::new();
    let mut word: Option<(String, usize)> = None;
    for (at, c) in text.char_indices() {
        if is_cjk(c) {
            words.extend(word.take());
            words.push((c.to_string(), at));
        } else if c.is_whitespace() {
            words.extend(word.take());
        } else if is_kept(c) {
            word.get_or_insert_with(|| (String::new(), at)).0.push(c);
        }
    }
    words.extend(word);
    let starts: Vec<usize> = words.iter().map(|(_, at)| *at).collect();
    words
        .into_iter()
        .enumerate()
        .map(|(i, (read, at))| {
            let from = if i == 0 { 0 } else { at };
            let to = starts.get(i + 1).copied().unwrap_or(text.len());
            Word {
                read,
                written: text[from..to].trim().to_string(),
            }
        })
        .collect()
}

// the nearest in-order value before or after `at`
fn neighbour(values: &[f64], in_order: &[bool], at: usize, backwards: bool) -> Option<f64> {
    if backwards {
        (0..at).rev().find(|&i| in_order[i]).map(|i| values[i])
    } else {
        (at..values.len()).find(|&i| in_order[i]).map(|i| values[i])
    }
}

/// The reference's `fix_timestamp`: the longest non-decreasing run of times stands; a run of outliers of up to two
/// snaps to its nearer in-order neighbour, a longer one is interpolated between its neighbours.
// runs are a transcript's words long, far inside f64's exact integers
#[allow(clippy::cast_precision_loss)]
pub fn fix_timestamps(raw: &[f64]) -> Vec<f64> {
    let n = raw.len();
    if n == 0 {
        return Vec::new();
    }
    let mut length = vec![1usize; n];
    let mut parent = vec![usize::MAX; n];
    for current in 1..n {
        for previous in 0..current {
            if raw[previous] <= raw[current] && length[previous] + 1 > length[current] {
                length[current] = length[previous] + 1;
                parent[current] = previous;
            }
        }
    }
    let longest = *length.iter().max().expect("a time");
    let mut at = length.iter().position(|&l| l == longest).expect("a time");
    let mut in_order = vec![false; n];
    loop {
        in_order[at] = true;
        if parent[at] == usize::MAX {
            break;
        }
        at = parent[at];
    }
    let mut out = raw.to_vec();
    let mut start = 0;
    while start < n {
        if in_order[start] {
            start += 1;
            continue;
        }
        let mut end = start;
        while end < n && !in_order[end] {
            end += 1;
        }
        let left = neighbour(&out, &in_order, start, true);
        let right = neighbour(&out, &in_order, end, false);
        let count = end - start;
        for (pos, value) in out.iter_mut().enumerate().take(end).skip(start) {
            *value = match (left, right) {
                (Some(l), Some(r)) if count <= SNAP_RUN => {
                    if pos + 1 - start <= end - pos {
                        l
                    } else {
                        r
                    }
                }
                (Some(l), Some(r)) => l + (r - l) / (count + 1) as f64 * (pos - start + 1) as f64,
                (Some(l), None) => l,
                (None, Some(r)) => r,
                (None, None) => *value,
            };
        }
        start = end;
    }
    out.iter().map(|v| v.trunc()).collect()
}

/// The aligner's head and how it reads a prompt.
pub struct Aligner {
    pub head: Tensor,
    pub tokenizer: Tokenizer,
    pub timestamp_token: u32,
    pub audio_start: u32,
    pub audio_end: u32,
    pub segment_ms: f64,
    pub classes: usize,
}

impl Aligner {
    pub fn new(
        head: Tensor,
        tokenizer: Tokenizer,
        cfg: &super::config::AlignerConfig,
    ) -> anyhow::Result<Self> {
        let id = |token: &str| {
            tokenizer
                .token_to_id(token)
                .ok_or_else(|| anyhow::anyhow!("the aligner's tokenizer has no `{token}`"))
        };
        Ok(Self {
            audio_start: id(super::inputs_processor::AUDIO_START)?,
            audio_end: id(super::inputs_processor::AUDIO_END)?,
            timestamp_token: cfg.timestamp_token,
            segment_ms: cfg.segment_ms,
            classes: cfg.classes,
            head,
            tokenizer,
        })
    }

    /// The longest clip the head's classes cover, in seconds
    #[allow(clippy::cast_precision_loss)]
    pub fn max_seconds(&self) -> f64 {
        self.classes as f64 * self.segment_ms / MS_PER_SECOND
    }

    /// The prompt's ids after the clip's `audio_tokens` placeholders (`audio_token`): each word, then its two slots.
    pub fn prompt(
        &self,
        words: &[Word],
        audio_token: u32,
        audio_tokens: usize,
    ) -> anyhow::Result<Vec<u32>> {
        let slots = TIMESTAMP.repeat(SLOTS_PER_WORD);
        let read: Vec<&str> = words.iter().map(|w| w.read.as_str()).collect();
        let text = format!("{}{slots}", read.join(&slots));
        let encoded = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("aligner tokenizer: {e}"))?;
        let mut ids = Vec::with_capacity(audio_tokens + 2 + encoded.len());
        ids.push(self.audio_start);
        ids.extend(std::iter::repeat_n(audio_token, audio_tokens));
        ids.push(self.audio_end);
        ids.extend(encoded.get_ids());
        Ok(ids)
    }

    /// `words` timed from the prompt's normed hidden states `(1, tokens, hidden)`.
    pub fn read(
        &self,
        hidden: &Tensor,
        ids: &[u32],
        words: Vec<Word>,
    ) -> anyhow::Result<Vec<AlignedWord>> {
        let slots: Vec<u32> = ids
            .iter()
            .enumerate()
            .filter(|&(_, &id)| id == self.timestamp_token)
            .map(|(i, _)| u32::try_from(i))
            .collect::<Result<_, _>>()?;
        anyhow::ensure!(
            slots.len() == SLOTS_PER_WORD * words.len(),
            "{} timestamp slots for {} words",
            slots.len(),
            words.len()
        );
        let device: &Device = hidden.device();
        let rows = hidden
            .squeeze(0)?
            .index_select(&Tensor::from_vec(slots.clone(), slots.len(), device)?, 0)?;
        let classes = rows.matmul(&self.head.t()?)?.argmax(1)?.to_vec1::<u32>()?;
        let raw: Vec<f64> = classes
            .iter()
            .map(|&c| f64::from(c) * self.segment_ms)
            .collect();
        let times = fix_timestamps(&raw);
        Ok(words
            .into_iter()
            .enumerate()
            .map(|(i, word)| AlignedWord {
                text: word.written,
                start: times[SLOTS_PER_WORD * i] / MS_PER_SECOND,
                end: times[SLOTS_PER_WORD * i + 1] / MS_PER_SECOND,
                speaker: None,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // read as the reference splits; written with the punctuation up to the next word
    #[test]
    fn words_split_as_the_reference_does() {
        let words = split_words("Hello, world! It's 2024; 你好吗?");
        let read: Vec<&str> = words.iter().map(|w| w.read.as_str()).collect();
        let written: Vec<&str> = words.iter().map(|w| w.written.as_str()).collect();
        assert_eq!(read, ["Hello", "world", "It's", "2024", "你", "好", "吗"]);
        assert_eq!(
            written,
            ["Hello,", "world!", "It's", "2024;", "你", "好", "吗?"]
        );
        assert!(split_words(" ... ").is_empty());
    }

    // the reference's own outputs: an outlier snaps to its nearer in-order neighbour, a run with none after it takes
    // the one before, a longer run between two interpolates
    #[test]
    fn out_of_order_times_are_fixed() {
        assert_eq!(
            fix_timestamps(&[0.0, 80.0, 40.0, 160.0]),
            [0.0, 80.0, 80.0, 160.0]
        );
        assert_eq!(
            fix_timestamps(&[0.0, 900.0, 950.0, 990.0, 400.0, 480.0]),
            [0.0, 900.0, 950.0, 990.0, 990.0, 990.0]
        );
        assert_eq!(
            fix_timestamps(&[0.0, 900.0, 950.0, 990.0, 120.0, 160.0, 200.0, 240.0, 280.0]),
            [0.0, 30.0, 60.0, 90.0, 120.0, 160.0, 200.0, 240.0, 280.0]
        );
    }
}
