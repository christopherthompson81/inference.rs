/// A span of the transcript with its time in seconds.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct TimedText {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Transcription {
    pub text: String,
    /// The tokenizer's pieces, each at the frames that emitted it.
    pub tokens: Vec<TimedText>,
    /// Whitespace-separated words, punctuation attached.
    pub words: Vec<TimedText>,
    /// Seconds of audio transcribed.
    pub duration: f64,
    /// The language the model was told, or identified (Nemotron-3.5's tag, as `de-DE`).
    pub language: Option<String>,
}

/// What a windowed transcription reports as it goes.
pub enum TranscriptionProgress<'a> {
    /// The audio's length and the windows it is cut into, `(start, end)` seconds, before the first runs.
    Started {
        duration: f64,
        windows: &'a [(f64, f64)],
    },
    /// Window `index`'s transcript, its times where it sits in the whole audio.
    Window {
        index: usize,
        start: f64,
        end: f64,
        transcript: &'a Transcription,
    },
}

/// Told of a transcription's progress; `false` stops it before its next window, with [`TRANSCRIPTION_STOPPED`].
pub type ProgressFn = dyn Fn(TranscriptionProgress<'_>) -> bool + Send + Sync;

/// The error a transcription its progress stopped ends with.
pub const TRANSCRIPTION_STOPPED: &str = "the transcription was stopped";

/// What one transcription request asks for beyond its audio.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, inference_json::FromJson)]
pub struct TranscriptionOptions {
    /// The audio's language, for a model that takes one; others identify it themselves.
    pub language: Option<String>,
    /// Beam search over this many hypotheses for a transducer (RNN-T, TDT); unset or 1 decodes greedily.
    pub beam_size: Option<usize>,
    /// Write an English translation instead of a transcript, for a model that can.
    #[serde(default)]
    pub translate: bool,
    /// Resume at this second: windows planned over the whole audio, those ending at or before it skipped, so the
    /// rest come out as in an uninterrupted run.
    #[serde(default)]
    pub resume_at: Option<f64>,
    /// Windows of at most this many seconds at any length of audio (cut at the VAD's silences, else the quietest
    /// moments); unset keeps the model's own plan.
    #[serde(default)]
    pub max_window_seconds: Option<f64>,
    /// These `(start, end)` seconds as the windows, each transcribed alone in place of the model's plan (overlaps
    /// allowed, starts in order).
    #[serde(default)]
    pub spans: Option<Vec<(f64, f64)>>,
}
