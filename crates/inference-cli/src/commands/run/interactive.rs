//! Interactive and one-shot modes: the terminal loop over the engine API for each kind of model.

use std::{fs, path::PathBuf, time::Instant};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use inference_api::{
    Engine,
    engine::AgentPermission,
    engine_chat::ReasoningEffort,
    lora_adapters::ListLoraAdaptersQuery,
    openai::{
        ImageGenerationRequest, ModelCategory, SpeechGenerationRequest, TranscriptionRequest,
        TranscriptionResponseFormat, VoiceActivityRequest,
    },
};
use regex::Regex;
use rustyline::{DefaultEditor, Editor, Helper, error::ReadlineError, history::History};
use serde_json::{Value, json};
use tracing::{error, info};

use super::chat::{self, AUDIO_PART, ChatOptions, IMAGE_PART, Sampling, SessionMedia, VIDEO_PART};

const COMMAND_COMMANDS: &str = r#"
Commands:
- `/help`: Display this message.
- `/exit`: Quit interactive mode.
- `/system <system message here>`:
    Add a system message to the chat without running the model.
    Ex: `/system Always respond as a pirate.`
- `/clear`: Clear the chat history.
- `/adapter <alias>`: Use a loaded LoRA adapter for subsequent requests.
- `/adapter use <alias>`: Select an alias named `none` or `list`.
- `/adapter none`: Use the base model for subsequent requests.
- `/adapter list`: List loaded LoRA adapters.
- `/temperature <float>`: Set sampling temperature (0.0 to 2.0).
- `/topk <int>`: Set top-k sampling value (>0).
- `/topp <float>`: Set top-p sampling value in (0.0 to 1.0).
"#;

const TEXT_INTERACTIVE_HELP: &str = r#"
Welcome to interactive mode! Because this model is a text model, you can enter prompts and chat with the model.
"#;

const VISION_INTERACTIVE_HELP: &str = r#"
Welcome to interactive mode! Because this model is a multimodal model, you can enter prompts and chat with the model.

To specify a message with one or more images, audios, or videos, simply include the image/audio/video URL or path:

- `Describe these images: path/to/image1.jpg path/to/image2.png`
- `Describe the image and transcribe the audio: path/to/image1.jpg path/to/sound.mp3`
- `Describe this video: path/to/video.mp4`
"#;

const DIFFUSION_INTERACTIVE_HELP: &str = r#"
Welcome to interactive mode! Because this model is a diffusion model, you can enter prompts and the model will generate an image.

Commands:
- `/help`: Display this message.
- `/exit`: Quit interactive mode.
"#;

const SPEECH_INTERACTIVE_HELP: &str = r#"
Welcome to interactive mode! Because this model is a speech generation model, you can enter prompts and the model will generate audio.

Commands:
- `/help`: Display this message.
- `/exit`: Quit interactive mode.
"#;

const TRANSCRIPTION_INTERACTIVE_HELP: &str = r#"
Welcome to interactive mode! Because this model is a speech recognition model, you can enter the path of an audio
file (WAV, MP3, FLAC, ...) and the model will transcribe it.

Commands:
- `/help`: Display this message.
- `/exit`: Quit interactive mode.
"#;

const VOICE_ACTIVITY_INTERACTIVE_HELP: &str = r#"
Welcome to interactive mode! Because this model is a voice activity model, you can enter the path of an audio
file (WAV, MP3, FLAC, ...) and the model will list where it speaks.

Commands:
- `/help`: Display this message.
- `/exit`: Quit interactive mode.
"#;

const HELP_CMD: &str = "/help";
const EXIT_CMD: &str = "/exit";
const SYSTEM_CMD: &str = "/system";
const CLEAR_CMD: &str = "/clear";
const ADAPTER_CMD: &str = "/adapter";
const BANNER: &str = "====================";
const IMAGE_FILE_PREFIX: &str = "image-generation-";
const SPEECH_FILE_PREFIX: &str = "speech-";

/// Regex string used to extract image URLs from prompts.
const IMAGE_REGEX: &str = r#"((?:https?://|file://)?\S+?\.(?:png|jpe?g|bmp|gif|webp)(?:\?\S+?)?)"#;
const AUDIO_REGEX: &str = r#"((?:https?://|file://)?\S+?\.(?:wav|mp3|flac|ogg)(?:\?\S+?)?)"#;
const VIDEO_REGEX: &str =
    r#"((?:https?://|file://)?\S+?\.(?:mp4|avi|mov|mkv|webm|gif|m4v)(?:\?\S+?)?)"#;

pub struct OneshotInput {
    pub text: String,
    pub images: Vec<String>,
    pub videos: Vec<String>,
    pub audios: Vec<String>,
}

pub struct InteractiveConfig {
    pub do_search: bool,
    pub do_code_exec: bool,
    pub do_shell: bool,
    pub agent_permission: AgentPermission,
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub adapter: Option<String>,
}

impl InteractiveConfig {
    fn chat_options(self) -> ChatOptions {
        let session_id =
            (self.do_code_exec || self.do_shell).then(|| uuid::Uuid::new_v4().to_string());
        ChatOptions {
            do_search: self.do_search,
            do_code_exec: self.do_code_exec,
            do_shell: self.do_shell,
            agent_permission: self.agent_permission,
            enable_thinking: self.enable_thinking,
            reasoning_effort: self.reasoning_effort,
            adapter: self.adapter,
            session_id,
        }
    }
}

fn history_file_path() -> PathBuf {
    let config_dir = dirs::config_dir()
        .expect("Could not determine the config directory")
        .join("inference.rs");
    fs::create_dir_all(&config_dir).expect("Failed to create config directory");

    // e.g. ~/.config/inference.rs/history.txt
    config_dir.join("history.txt")
}

fn build_prompt(options: &ChatOptions) -> String {
    let mut tags = Vec::new();
    if options.do_code_exec {
        tags.push("code".to_string());
    }
    if options.do_shell {
        tags.push("shell".to_string());
    }
    if options.do_search {
        tags.push("search".to_string());
    }
    if let Some(adapter) = &options.adapter {
        tags.push(format!("lora:{adapter}"));
    }
    if tags.is_empty() {
        "> ".to_string()
    } else {
        format!("[{}] > ", tags.join(","))
    }
}

fn read_line<H: Helper, I: History>(editor: &mut Editor<H, I>, prompt: &str) -> String {
    match editor.readline(prompt) {
        Err(ReadlineError::Interrupted | ReadlineError::Eof) => {
            editor.save_history(&history_file_path()).unwrap();
            std::process::exit(0);
        }
        Err(e) => {
            editor.save_history(&history_file_path()).unwrap();
            eprintln!("Error reading input: {e:?}");
            std::process::exit(1);
        }
        Ok(prompt) => {
            editor.add_history_entry(prompt.clone()).unwrap();
            prompt
        }
    }
}

// Ctrl-C ends the turn in flight (it still prints its stats) and quits between turns.
fn install_ctrlc_handler() {
    ctrlc::set_handler(|| {
        if !chat::cancel_in_flight() {
            std::process::exit(0);
        }
    })
    .expect("Failed to set CTRL-C handler for interactive mode");
}

fn open_editor() -> DefaultEditor {
    let mut editor = DefaultEditor::new().expect("Failed to open input");
    let _ = editor.load_history(&history_file_path());
    editor
}

pub async fn oneshot_mode(engine: &Engine, input: OneshotInput, config: InteractiveConfig) {
    if let Err(e) = oneshot(engine, input, config).await {
        error!("{e}");
    }
    println!();
}

async fn oneshot(
    engine: &Engine,
    input: OneshotInput,
    config: InteractiveConfig,
) -> anyhow::Result<()> {
    let model = chat::default_model(engine)?;
    let options = config.chat_options();
    let sampling = Sampling::for_model(model.generation_defaults.as_ref());
    let mut media = SessionMedia::default();
    let has_media =
        !input.images.is_empty() || !input.videos.is_empty() || !input.audios.is_empty();
    let message = if has_media {
        anyhow::ensure!(
            model.category == Some(ModelCategory::Multimodal),
            "--image/--video/--audio require a multimodal model, but the loaded model is not multimodal."
        );
        let mut parts = Vec::new();
        for (kind, references) in [
            (IMAGE_PART, &input.images),
            (AUDIO_PART, &input.audios),
            (VIDEO_PART, &input.videos),
        ] {
            for reference in references {
                parts.push((kind, media.source_for(reference, kind).await?));
            }
        }
        chat::media_message(parts, &input.text)
    } else {
        chat::text_message("user", &input.text)
    };
    install_ctrlc_handler();
    let turn = chat::stream_turn(engine, &[message], &media, &options, &sampling).await?;
    chat::print_stats(&turn, &sampling);
    Ok(())
}

pub async fn interactive_mode(engine: &Engine, config: InteractiveConfig) {
    let model = match chat::default_model(engine) {
        Ok(model) => model,
        Err(e) => return error!("{e}"),
    };
    install_ctrlc_handler();
    match model.category {
        Some(ModelCategory::Text | ModelCategory::Multimodal) => {
            chat_interactive_mode(engine, config, &model).await
        }
        Some(ModelCategory::Diffusion) => diffusion_interactive_mode(engine).await,
        Some(ModelCategory::Speech) => speech_interactive_mode(engine).await,
        Some(ModelCategory::Transcription) => transcription_interactive_mode(engine).await,
        Some(ModelCategory::VoiceActivity) => voice_activity_interactive_mode(engine).await,
        Some(ModelCategory::Embedding) => error!(
            "Embedding models do not support interactive mode. Use the server or Python/Rust APIs."
        ),
        None => error!("The engine did not report the loaded model's category."),
    }
}

async fn chat_interactive_mode(
    engine: &Engine,
    config: InteractiveConfig,
    model: &inference_api::openai::ModelObject,
) {
    let multimodal = model.category == Some(ModelCategory::Multimodal);
    let help = if multimodal {
        VISION_INTERACTIVE_HELP
    } else {
        TEXT_INTERACTIVE_HELP
    };
    let media_regexes = media_regexes();
    let mut options = config.chat_options();
    let mut sampling = Sampling::for_model(model.generation_defaults.as_ref());
    let mut messages: Vec<Value> = Vec::new();
    let mut media = SessionMedia::default();

    info!(
        "Starting interactive loop with sampling: {}",
        sampling.describe()
    );
    println!(
        "{BANNER}{help}{COMMAND_COMMANDS}\nSampling: {}\n{BANNER}",
        sampling.describe()
    );

    let mut editor = open_editor();
    loop {
        let prompt = read_line(&mut editor, &build_prompt(&options));
        let prompt = prompt.trim();
        if prompt.is_empty() || sampling.apply_command(prompt) {
            continue;
        }
        if handle_adapter_command(engine, prompt, &mut options.adapter).await {
            continue;
        }
        match prompt {
            HELP_CMD => {
                println!("{BANNER}{help}{COMMAND_COMMANDS}{BANNER}");
                continue;
            }
            EXIT_CMD => break,
            CLEAR_CMD => {
                messages.clear();
                media.clear();
                info!("Cleared chat history.");
                continue;
            }
            _ if prompt.starts_with(SYSTEM_CMD) => {
                let parsed = match &prompt.split(SYSTEM_CMD).collect::<Vec<_>>()[..] {
                    &["", system] => system.trim(),
                    _ => {
                        println!(
                            "Error: Setting the system command should be done with this format: `{SYSTEM_CMD} This is a system message.`"
                        );
                        continue;
                    }
                };
                info!("Set system message to `{parsed}`.");
                messages.push(chat::text_message("system", parsed));
                continue;
            }
            _ => {}
        }

        // A turn that fails leaves the conversation as it was, so the next prompt can go on.
        let media_before = media.len();
        let message = if multimodal {
            match media_user_message(prompt, &media_regexes, &mut media).await {
                Ok(message) => message,
                Err(e) => {
                    media.truncate(media_before);
                    error!("{e}");
                    continue;
                }
            }
        } else {
            chat::text_message("user", prompt)
        };
        messages.push(message);

        match chat::stream_turn(engine, &messages, &media, &options, &sampling).await {
            Ok(turn) => {
                chat::print_stats(&turn, &sampling);
                messages.push(turn.message);
            }
            Err(e) => {
                messages.pop();
                media.truncate(media_before);
                error!("{e}");
            }
        }
        println!();
    }

    editor.save_history(&history_file_path()).unwrap();
}

fn media_regexes() -> [(&'static str, Regex); 3] {
    [
        (IMAGE_PART, Regex::new(IMAGE_REGEX).unwrap()),
        (AUDIO_PART, Regex::new(AUDIO_REGEX).unwrap()),
        (VIDEO_PART, Regex::new(VIDEO_REGEX).unwrap()),
    ]
}

// Media named in the prompt become content parts, and the text keeps what's left.
async fn media_user_message(
    prompt: &str,
    regexes: &[(&'static str, Regex)],
    media: &mut SessionMedia,
) -> anyhow::Result<Value> {
    let mut text = prompt.to_string();
    let mut parts = Vec::new();
    for (kind, regex) in regexes {
        let (references, rest) = parse_files_and_message(&text, regex);
        for reference in references {
            let source = media.source_for(&reference, kind).await?;
            info!("Added `{reference}`");
            parts.push((*kind, source));
        }
        text = rest;
    }
    Ok(if parts.is_empty() {
        chat::text_message("user", prompt)
    } else {
        chat::media_message(parts, &text)
    })
}

fn parse_files_and_message(input: &str, regex: &Regex) -> (Vec<String>, String) {
    // Trailing punctuation after a URL is sentence punctuation, not part of it.
    let urls = regex
        .captures_iter(input)
        .filter_map(|cap| {
            cap.get(1).map(|m| {
                m.as_str()
                    .trim_end_matches(|c: char| {
                        matches!(
                            c,
                            '.' | ',' | ';' | ':' | '!' | '?' | ')' | ']' | '}' | '"' | '\''
                        )
                    })
                    .to_string()
            })
        })
        .collect::<Vec<_>>();
    let text = regex.replace_all(input, "").trim().to_string();
    (urls, text)
}

async fn handle_adapter_command(
    engine: &Engine,
    prompt: &str,
    adapter: &mut Option<String>,
) -> bool {
    let Some(command) = parse_adapter_command(prompt) else {
        return false;
    };
    let loaded = || async {
        engine
            .lora_adapters(ListLoraAdaptersQuery::default())
            .await
            .map(|adapters| adapters.data.into_iter().map(|a| a.id).collect::<Vec<_>>())
    };
    match command {
        AdapterCommand::Invalid => println!("Error: format is `{ADAPTER_CMD} <alias|none|list>`"),
        AdapterCommand::Base => {
            *adapter = None;
            info!("Using the base model.");
        }
        AdapterCommand::List => match loaded().await {
            Ok(aliases) if aliases.is_empty() => println!("No LoRA adapters are loaded."),
            Ok(aliases) => {
                println!("Loaded LoRA adapters:");
                for alias in aliases {
                    let marker = if adapter.as_deref() == Some(alias.as_str()) {
                        " (selected)"
                    } else {
                        ""
                    };
                    println!("- {alias}{marker}");
                }
            }
            Err(error) => println!("Error: {error}"),
        },
        AdapterCommand::Select(alias) => match loaded().await {
            Ok(aliases) if aliases.iter().any(|loaded| loaded == alias) => {
                *adapter = Some(alias.to_string());
                info!("Using LoRA adapter `{alias}`.");
            }
            Ok(_) => println!("Error: LoRA adapter alias `{alias}` is not loaded"),
            Err(error) => println!("Error: {error}"),
        },
    }
    true
}

#[derive(Debug, Eq, PartialEq)]
enum AdapterCommand<'a> {
    Invalid,
    Base,
    List,
    Select(&'a str),
}

fn parse_adapter_command(prompt: &str) -> Option<AdapterCommand<'_>> {
    let argument = if prompt == ADAPTER_CMD {
        ""
    } else {
        prompt.strip_prefix("/adapter ")?.trim()
    };
    Some(match argument {
        "" => AdapterCommand::Invalid,
        "none" => AdapterCommand::Base,
        "list" => AdapterCommand::List,
        argument => match argument.strip_prefix("use ").map(str::trim) {
            Some("") => AdapterCommand::Invalid,
            Some(alias) => AdapterCommand::Select(alias),
            None => AdapterCommand::Select(argument),
        },
    })
}

async fn diffusion_interactive_mode(engine: &Engine) {
    println!("{BANNER}{DIFFUSION_INTERACTIVE_HELP}{BANNER}");
    let mut editor = open_editor();
    loop {
        let prompt = match read_line(&mut editor, "> ").trim() {
            "" => continue,
            HELP_CMD => {
                println!("{BANNER}{DIFFUSION_INTERACTIVE_HELP}{BANNER}");
                continue;
            }
            EXIT_CMD => break,
            prompt => prompt.to_string(),
        };
        match generate_image(engine, &prompt).await {
            Ok(message) => println!("{message}"),
            Err(e) => error!("{e}"),
        }
        println!();
    }
    editor.save_history(&history_file_path()).unwrap();
}

// The image comes back inline and is written to the working directory.
async fn generate_image(engine: &Engine, prompt: &str) -> anyhow::Result<String> {
    let request: ImageGenerationRequest =
        serde_json::from_value(json!({"prompt": prompt, "response_format": "b64_json"}))?;
    let pixels = (request.height * request.width) as f32;
    let start = Instant::now();
    let response = engine
        .image_generation(request)
        .await
        .map_err(anyhow::Error::msg)?;
    let duration = start.elapsed().as_secs_f32();
    let png = response
        .data
        .first()
        .and_then(|choice| choice.b64_json.as_deref())
        .ok_or_else(|| anyhow::anyhow!("the engine returned no image"))?;
    let path = format!("{IMAGE_FILE_PREFIX}{}.png", uuid::Uuid::new_v4());
    fs::write(&path, STANDARD.decode(png)?)?;
    Ok(format!(
        "Image generated can be found at: image is at `{path}`. Took {duration:.2}s ({:.2} pixels/s).",
        pixels / duration
    ))
}

async fn speech_interactive_mode(engine: &Engine) {
    println!("{BANNER}{SPEECH_INTERACTIVE_HELP}{BANNER}");
    let mut editor = open_editor();
    let mut n = 0;
    loop {
        let prompt = match read_line(&mut editor, "> ").trim() {
            "" => continue,
            HELP_CMD => {
                println!("{BANNER}{SPEECH_INTERACTIVE_HELP}{BANNER}");
                continue;
            }
            EXIT_CMD => break,
            prompt => prompt.to_string(),
        };
        let out_file = format!("{SPEECH_FILE_PREFIX}{n}.wav");
        match generate_speech(engine, &prompt, &out_file).await {
            Ok(duration) => {
                println!("Speech generated can be found at `{out_file}`. Took {duration:.2}s.");
                n += 1;
            }
            Err(e) => error!("{e}"),
        }
        println!();
    }
    editor.save_history(&history_file_path()).unwrap();
}

async fn generate_speech(engine: &Engine, prompt: &str, out_file: &str) -> anyhow::Result<f32> {
    let request: SpeechGenerationRequest =
        serde_json::from_value(json!({"input": prompt, "response_format": "wav"}))?;
    let start = Instant::now();
    let audio = engine
        .speech_generation(request)
        .await
        .map_err(anyhow::Error::msg)?;
    let duration = start.elapsed().as_secs_f32();
    fs::write(out_file, audio.bytes)?;
    Ok(duration)
}

async fn transcription_interactive_mode(engine: &Engine) {
    println!("{BANNER}{TRANSCRIPTION_INTERACTIVE_HELP}{BANNER}");
    let mut editor = open_editor();
    loop {
        let path = match read_line(&mut editor, "audio file> ").trim() {
            "" => continue,
            HELP_CMD => {
                println!("{BANNER}{TRANSCRIPTION_INTERACTIVE_HELP}{BANNER}");
                continue;
            }
            EXIT_CMD => break,
            path => path.to_string(),
        };
        match transcribe(engine, &path).await {
            Ok((text, duration)) => println!("{text}\n(took {duration:.2}s)"),
            Err(e) => error!("{e}"),
        }
        println!();
    }
    editor.save_history(&history_file_path()).unwrap();
}

async fn voice_activity_interactive_mode(engine: &Engine) {
    println!("{BANNER}{VOICE_ACTIVITY_INTERACTIVE_HELP}{BANNER}");
    let mut editor = open_editor();
    loop {
        let path = match read_line(&mut editor, "audio file> ").trim() {
            "" => continue,
            HELP_CMD => {
                println!("{BANNER}{VOICE_ACTIVITY_INTERACTIVE_HELP}{BANNER}");
                continue;
            }
            EXIT_CMD => break,
            path => path.to_string(),
        };
        let detected = async {
            let audio = fs::read(&path)?;
            engine
                .voice_activity(VoiceActivityRequest::new(), &audio)
                .await
                .map_err(anyhow::Error::msg)
        };
        match detected.await {
            Ok(activity) => {
                for s in &activity.segments {
                    println!("{:>9.2}s - {:>9.2}s", s.start, s.end);
                }
                println!(
                    "({} speech segments in {:.2}s)",
                    activity.segments.len(),
                    activity.duration
                );
            }
            Err(e) => error!("{e}"),
        }
        println!();
    }
    editor.save_history(&history_file_path()).unwrap();
}

async fn transcribe(engine: &Engine, path: &str) -> anyhow::Result<(String, f32)> {
    let audio = fs::read(path)?;
    let request = TranscriptionRequest::new(TranscriptionResponseFormat::Text);
    let start = Instant::now();
    let output = engine
        .transcription(request, &audio)
        .await
        .map_err(anyhow::Error::msg)?;
    Ok((output.body, start.elapsed().as_secs_f32()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_files_and_message_trims_trailing_punctuation() {
        let regex = Regex::new(IMAGE_REGEX).unwrap();
        let input = "Look at this https://example.com/test.png.";
        let (urls, text) = parse_files_and_message(input, &regex);
        assert_eq!(urls, vec!["https://example.com/test.png"]);
        assert_eq!(text, "Look at this .");
    }

    #[test]
    fn adapter_command_can_select_reserved_aliases() {
        assert_eq!(
            parse_adapter_command("/adapter use none"),
            Some(AdapterCommand::Select("none"))
        );
        assert_eq!(
            parse_adapter_command("/adapter use list"),
            Some(AdapterCommand::Select("list"))
        );
        assert_eq!(
            parse_adapter_command("/adapter none"),
            Some(AdapterCommand::Base)
        );
    }

    #[tokio::test]
    async fn a_prompt_naming_media_becomes_media_parts_and_text() {
        let dir = tempfile::tempdir().unwrap();
        let image = dir.path().join("cat.png");
        fs::write(&image, b"png").unwrap();
        let regexes = media_regexes();
        let mut media = SessionMedia::default();
        let prompt = format!("Describe {}", image.display());
        let message = media_user_message(&prompt, &regexes, &mut media)
            .await
            .unwrap();
        assert_eq!(message["content"][0]["type"], "image_url");
        assert_eq!(message["content"][0]["image_url"]["url"], "media://0");
        assert_eq!(message["content"][1]["text"], "Describe");
        let plain = media_user_message("hello", &regexes, &mut media)
            .await
            .unwrap();
        assert_eq!(plain["content"], "hello");
    }
}
