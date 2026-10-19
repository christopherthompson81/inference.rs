use std::path::PathBuf;

/// The coding agent's own options; the model options are `run`'s.
#[derive(clap::Args, Clone, Default)]
pub struct CoderOptions {
    /// Use the model on an OpenAI-compatible server at this base URL instead of loading one; `-m` names the server's model
    #[arg(long)]
    pub url: Option<String>,

    /// File holding the server's API key (default: the INFERENCE_API_KEY environment variable)
    #[arg(long, requires = "url")]
    pub api_key_file: Option<PathBuf>,

    /// Run this prompt without the REPL, print the answer and exit
    #[arg(short = 'p', long)]
    pub prompt: Option<String>,

    /// Continue the newest session for this project
    #[arg(long = "continue", conflicts_with = "resume")]
    pub continue_session: bool,

    /// Resume a session by id (or an id prefix)
    #[arg(long)]
    pub resume: Option<String>,

    /// List this project's sessions and exit
    #[arg(long)]
    pub sessions: bool,

    /// Which tool calls ask first: ask (edits and commands ask), auto (nothing asks) or read-only (edits and commands
    /// are refused). With `-p`, calls that would ask are refused
    #[arg(long, default_value = "ask", value_parser = clap::value_parser!(inference_coder::PermissionMode))]
    pub permission_mode: inference_coder::PermissionMode,

    /// The project directory the tools work in (default: the current directory)
    #[arg(long)]
    pub cwd: Option<PathBuf>,

    /// Model replies (each followed by its tool calls) one prompt may take before the agent stops
    #[arg(long, default_value_t = inference_coder::DEFAULT_MAX_ROUNDS)]
    pub max_rounds: usize,
}
