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
}
