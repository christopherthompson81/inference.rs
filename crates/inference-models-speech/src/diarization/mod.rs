//! Speaker diarization: who speaks when, as a speech probability per speaker per 10 ms frame and the segments
//! cut from them. Nemotron-3 Diarization, a streaming Sortformer with an arrival-order speaker cache.

mod cache;
mod nemotron3;

pub use nemotron3::{Nemotron3Config, Nemotron3Diarizer, Nemotron3Files};

// a speaker counts as speaking in a frame once its probability passes this, as transformers' processor cuts them
pub const DEFAULT_THRESHOLD: f32 = 0.5;

/// What a diarization request asks for beyond its audio.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DiarizationOptions {
    /// Probability at which a speaker counts as speaking; unset is `DEFAULT_THRESHOLD`.
    pub threshold: Option<f32>,
}

/// One speaker's turn, in seconds from the start of the audio.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct SpeakerSegment {
    pub speaker: usize,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Diarization {
    /// Seconds of audio.
    pub duration: f64,
    /// Seconds each frame of probabilities covers.
    pub frame_seconds: f64,
    pub num_speakers: usize,
    /// `frames * num_speakers` row-major speech probabilities.
    pub probabilities: Vec<f32>,
    /// Ordered by start, then speaker.
    pub segments: Vec<SpeakerSegment>,
}

/// Runs of frames each speaker's probability passes `threshold` in, as segments ordered by start then speaker.
pub fn speaker_segments(
    probabilities: &[f32],
    num_speakers: usize,
    frame_seconds: f64,
    threshold: f32,
) -> Vec<SpeakerSegment> {
    let frames = probabilities.len() / num_speakers;
    let mut segments = Vec::new();
    for speaker in 0..num_speakers {
        let mut start = None;
        for t in 0..=frames {
            let active = t < frames && probabilities[t * num_speakers + speaker] > threshold;
            match (active, start) {
                (true, None) => start = Some(t),
                (false, Some(s)) => {
                    segments.push(SpeakerSegment {
                        speaker,
                        start: s as f64 * frame_seconds,
                        end: t as f64 * frame_seconds,
                    });
                    start = None;
                }
                _ => {}
            }
        }
    }
    segments.sort_by(|a, b| a.start.total_cmp(&b.start).then(a.speaker.cmp(&b.speaker)));
    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_over_the_threshold_become_segments_in_start_order() {
        // two speakers over 6 frames of 10 ms
        let probs = [0.9, 0.1, 0.9, 0.1, 0.2, 0.8, 0.2, 0.8, 0.9, 0.9, 0.1, 0.9];
        let segments = speaker_segments(&probs, 2, 0.01, 0.5);
        let spans: Vec<(usize, f64, f64)> = segments
            .iter()
            .map(|s| (s.speaker, (s.start * 100.).round(), (s.end * 100.).round()))
            .collect();
        assert_eq!(spans, [(0, 0., 2.), (1, 2., 6.), (0, 4., 5.)]);
    }
}
