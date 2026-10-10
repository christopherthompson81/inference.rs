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

/// What one transcription request asks for beyond its audio.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TranscriptionOptions {
    /// The audio's language, for a model that takes one; others identify it themselves.
    pub language: Option<String>,
}
