/// How probabilities become speech segments: the reference `get_speech_timestamps`'s parameters and defaults.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SegmentOptions {
    /// Probability at which speech starts.
    pub threshold: f32,
    /// Probability under which speech may end; unset is `threshold - 0.15`, at least 0.01.
    pub neg_threshold: Option<f32>,
    /// Shorter speech is dropped.
    pub min_speech_duration_ms: f64,
    /// Longer speech is split, at its longest silence when it has one; unset is unbounded.
    pub max_speech_duration_s: Option<f64>,
    /// Silence shorter than this does not end speech.
    pub min_silence_duration_ms: f64,
    /// Padding added on each side of a segment.
    pub speech_pad_ms: f64,
    /// Silences at least this long are candidates for a max-duration split.
    pub min_silence_at_max_speech_ms: f64,
    /// Split at the longest candidate silence rather than the last one.
    pub use_max_possible_silence_at_max_speech: bool,
}

// the reference's defaults
const THRESHOLD: f32 = 0.5;
const NEG_THRESHOLD_GAP: f32 = 0.15;
const MIN_NEG_THRESHOLD: f32 = 0.01;
const MIN_SPEECH_MS: f64 = 250.0;
const MIN_SILENCE_MS: f64 = 100.0;
const SPEECH_PAD_MS: f64 = 30.0;
const MIN_SILENCE_AT_MAX_SPEECH_MS: f64 = 98.0;
const MS_PER_S: f64 = 1000.0;

impl Default for SegmentOptions {
    fn default() -> Self {
        Self {
            threshold: THRESHOLD,
            neg_threshold: None,
            min_speech_duration_ms: MIN_SPEECH_MS,
            max_speech_duration_s: None,
            min_silence_duration_ms: MIN_SILENCE_MS,
            speech_pad_ms: SPEECH_PAD_MS,
            min_silence_at_max_speech_ms: MIN_SILENCE_AT_MAX_SPEECH_MS,
            use_max_possible_silence_at_max_speech: true,
        }
    }
}

struct Cut {
    speeches: Vec<(f64, f64)>,
    start: Option<f64>,
    triggered: bool,
    temp_end: f64,
    prev_end: f64,
    next_start: f64,
    possible_ends: Vec<(f64, f64)>,
}

impl Cut {
    fn reset(&mut self) {
        self.prev_end = 0.;
        self.next_start = 0.;
        self.temp_end = 0.;
        self.possible_ends.clear();
    }
}

/// `(start, end)` speech sample ranges from one probability per `window` samples (`get_speech_timestamps_from_probs`).
pub fn speech_segments(
    probs: &[f32],
    audio_len: usize,
    rate: u32,
    window: usize,
    o: &SegmentOptions,
) -> Vec<(usize, usize)> {
    let sr = f64::from(rate);
    let w = window as f64;
    let min_speech = sr * o.min_speech_duration_ms / MS_PER_S;
    let pad = sr * o.speech_pad_ms / MS_PER_S;
    let max_speech = sr * o.max_speech_duration_s.unwrap_or(f64::INFINITY) - w - 2. * pad;
    let min_silence = sr * o.min_silence_duration_ms / MS_PER_S;
    let min_silence_at_max = sr * o.min_silence_at_max_speech_ms / MS_PER_S;
    // compared in f64 against f64 thresholds, as the reference does, so a probability at a boundary lands alike
    let threshold = f64::from(o.threshold);
    let neg = o
        .neg_threshold
        .map(f64::from)
        .unwrap_or((threshold - f64::from(NEG_THRESHOLD_GAP)).max(f64::from(MIN_NEG_THRESHOLD)));
    let mut s = Cut {
        speeches: Vec::new(),
        start: None,
        triggered: false,
        temp_end: 0.,
        prev_end: 0.,
        next_start: 0.,
        possible_ends: Vec::new(),
    };
    for (i, &p) in probs.iter().enumerate() {
        let p = f64::from(p);
        let cur = w * i as f64;
        if p >= threshold && s.temp_end != 0. {
            let silence = cur - s.temp_end;
            if silence > min_silence_at_max {
                s.possible_ends.push((s.temp_end, silence));
            }
            s.temp_end = 0.;
            if s.next_start < s.prev_end {
                s.next_start = cur;
            }
        }
        if p >= threshold && !s.triggered {
            s.triggered = true;
            s.start = Some(cur);
            continue;
        }
        if let Some(begun) = s.start.filter(|_| s.triggered)
            && cur - begun > max_speech
        {
            if o.use_max_possible_silence_at_max_speech && !s.possible_ends.is_empty() {
                // the first of the longest silences, as Python's max picks
                let (end, dur) = s.possible_ends.iter().fold(s.possible_ends[0], |best, &e| {
                    if e.1 > best.1 { e } else { best }
                });
                s.speeches.push((begun, end));
                s.start = None;
                s.next_start = end + dur;
                if s.next_start < end + cur {
                    s.start = Some(s.next_start);
                } else {
                    s.triggered = false;
                }
                s.reset();
            } else if s.prev_end != 0. {
                s.speeches.push((begun, s.prev_end));
                s.start = None;
                if s.next_start < s.prev_end {
                    s.triggered = false;
                } else {
                    s.start = Some(s.next_start);
                }
                s.reset();
            } else {
                s.speeches.push((begun, cur));
                s.start = None;
                s.triggered = false;
                s.reset();
                continue;
            }
        }
        if p < neg && s.triggered {
            if s.temp_end == 0. {
                s.temp_end = cur;
            }
            let silence = cur - s.temp_end;
            if !o.use_max_possible_silence_at_max_speech && silence > min_silence_at_max {
                s.prev_end = s.temp_end;
            }
            if silence < min_silence {
                continue;
            }
            if let Some(begun) = s.start
                && s.temp_end - begun > min_speech
            {
                s.speeches.push((begun, s.temp_end));
            }
            s.start = None;
            s.triggered = false;
            s.reset();
        }
    }
    let len = audio_len as f64;
    if let Some(start) = s.start
        && len - start > min_speech
    {
        s.speeches.push((start, len));
    }
    let mut speeches = s.speeches;
    let n = speeches.len();
    for i in 0..n {
        if i == 0 {
            speeches[i].0 = (speeches[i].0 - pad).max(0.).trunc();
        }
        if i + 1 < n {
            let silence = speeches[i + 1].0 - speeches[i].1;
            if silence < 2. * pad {
                speeches[i].1 += (silence / 2.).floor();
                speeches[i + 1].0 = (speeches[i + 1].0 - (silence / 2.).floor()).max(0.).trunc();
            } else {
                speeches[i].1 = (speeches[i].1 + pad).min(len).trunc();
                speeches[i + 1].0 = (speeches[i + 1].0 - pad).max(0.).trunc();
            }
        } else {
            speeches[i].1 = (speeches[i].1 + pad).min(len).trunc();
        }
    }
    speeches
        .into_iter()
        .map(|(a, b)| (a as usize, b as usize))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 16_000;
    const WINDOW: usize = 512;

    fn cut(probs: &[f32], options: &SegmentOptions) -> Vec<(usize, usize)> {
        speech_segments(probs, probs.len() * WINDOW, RATE, WINDOW, options)
    }

    #[test]
    fn silence_has_no_segments_and_constant_speech_is_one() {
        let options = SegmentOptions::default();
        assert!(cut(&[0.1; 50], &options).is_empty());
        assert_eq!(cut(&[0.9; 50], &options), [(0, 50 * WINDOW)]);
    }

    // expected values are the reference get_speech_timestamps_from_probs's on the same probabilities
    #[test]
    fn short_blips_drop_and_short_gaps_bridge() {
        let options = SegmentOptions::default();
        let mut probs = vec![0.1f32; 60];
        probs[5..7].fill(0.9);
        assert!(cut(&probs, &options).is_empty());
        // 96 ms after the blip, under the 100 ms minimum silence, so it joins the speech that follows
        probs[10..30].fill(0.9);
        probs[18..20].fill(0.1);
        assert_eq!(cut(&probs, &options), [(2080, 15840)]);
        let mut gap = vec![0.1f32; 60];
        gap[20..40].fill(0.9);
        gap[28..30].fill(0.1);
        assert_eq!(cut(&gap, &options), [(9760, 20960)]);
    }

    #[test]
    fn long_speech_splits_at_its_longest_silences() {
        let mut probs = vec![0.9f32; 400];
        probs[100..105].fill(0.1);
        probs[200..210].fill(0.1);
        let options = SegmentOptions {
            max_speech_duration_s: Some(8.0),
            ..SegmentOptions::default()
        };
        assert_eq!(
            cut(&probs, &options),
            [(0, 51680), (53280, 102880), (107040, 204800)]
        );
    }
}
