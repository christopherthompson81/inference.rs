use axum::{
    Json,
    extract::{Extension, Multipart},
    http::StatusCode,
    response::IntoResponse,
};
use chrono::Utc;
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;
use tracing::error;
use uuid::Uuid;

use inference_api::media_source::{
    AUDIO_UPLOAD_EXTENSIONS, IMAGE_UPLOAD_EXTENSIONS, VIDEO_UPLOAD_EXTENSIONS,
};
use inference_api::openai::{AudioResponseFormat, SpeechGenerationRequest};

use crate::chat::append_chat_message;
use crate::types::{
    AppState, ChatFile, DeleteChatRequest, LoadChatRequest, NewChatRequest, RenameChatRequest,
    SelectRequest,
};
use crate::utils::get_cache_dir;

const INVALID_CHAT_ID: &str = "Invalid chat id";

fn media_upload_extension(
    filename: Option<&str>,
    allowed: &[&str],
    unsupported: &'static str,
) -> Result<String, &'static str> {
    let ext = filename
        .ok_or("No filename provided")?
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_lowercase();
    if ext.is_empty() {
        return Err("No file extension");
    }
    if allowed.contains(&ext.as_str()) {
        Ok(ext)
    } else {
        Err(unsupported)
    }
}

fn validate_image_upload(
    filename: Option<&str>,
    content_type: Option<&str>,
) -> Result<String, &'static str> {
    if content_type.is_some_and(|mime| !mime.starts_with("image/")) {
        return Err("File must be an image");
    }
    media_upload_extension(
        filename,
        IMAGE_UPLOAD_EXTENSIONS,
        "Unsupported image format",
    )
}

fn validate_video_upload(
    filename: Option<&str>,
    content_type: Option<&str>,
) -> Result<String, &'static str> {
    if content_type.is_some_and(|mime| !mime.starts_with("video/") && mime != "image/gif") {
        return Err("File must be a video");
    }
    media_upload_extension(
        filename,
        VIDEO_UPLOAD_EXTENSIONS,
        "Unsupported video format",
    )
}

fn validate_audio_upload(
    filename: Option<&str>,
    content_type: Option<&str>,
) -> Result<String, &'static str> {
    if content_type.is_some_and(|mime| !mime.starts_with("audio/")) {
        return Err("File must be an audio file");
    }
    media_upload_extension(
        filename,
        AUDIO_UPLOAD_EXTENSIONS,
        "Unsupported audio format",
    )
}

fn validate_text_upload(
    filename: Option<&str>,
    content_type: Option<&str>,
) -> Result<String, &'static str> {
    if let Some(mime) = content_type
        && !mime.starts_with("text/")
        && mime != "application/json"
        && mime != "application/javascript"
        && !matches!(
            mime,
            "application/octet-stream"
                | "application/x-python"
                | "application/x-rust"
                | "application/x-sh"
        )
    {
        return Err("File must be a text file");
    }

    let ext = if let Some(name) = filename {
        name.rsplit('.').next().unwrap_or("").to_lowercase()
    } else {
        return Err("No filename provided");
    };

    match ext.as_str() {
        "txt" | "md" | "markdown" | "log" | "csv" | "tsv" | "json" | "xml" | "yaml" | "yml"
        | "toml" | "ini" | "cfg" | "conf" => Ok(ext),
        "rs" | "py" | "js" | "ts" | "jsx" | "tsx" | "html" | "htm" | "css" | "scss" | "sass"
        | "less" => Ok(ext),
        "c" | "cpp" | "cc" | "cxx" | "h" | "hpp" | "hxx" | "java" | "kt" | "swift" | "go"
        | "rb" | "php" => Ok(ext),
        "cu" | "cuh" | "cl" | "ptx" | "glsl" | "vert" | "frag" | "geom" | "comp" | "tesc"
        | "tese" | "hlsl" | "metal" | "wgsl" => Ok(ext),
        "sh" | "bash" | "zsh" | "fish" | "ps1" | "bat" | "cmd" | "sql" | "dockerfile"
        | "makefile" => Ok(ext),
        "r" | "scala" | "clj" | "cljs" | "hs" | "elm" | "ex" | "exs" | "erl" | "fs" | "fsx"
        | "ml" | "mli" => Ok(ext),
        "vue" | "svelte" | "astro" | "lua" | "nim" | "zig" | "d" | "dart" | "jl" | "pl" | "pm"
        | "tcl" => Ok(ext),
        "gitignore" | "dockerignore" | "editorconfig" | "env" | "htaccess" => Ok(ext),
        "" => {
            if let Some(name) = filename {
                let name_lower = name.to_lowercase();
                if matches!(
                    name_lower.as_str(),
                    "readme"
                        | "license"
                        | "changelog"
                        | "makefile"
                        | "dockerfile"
                        | "vagrantfile"
                        | "gemfile"
                        | "rakefile"
                ) {
                    return Ok("txt".to_string());
                }
            }
            Err("No file extension")
        }
        _ => Err("Unsupported text file format"),
    }
}

pub async fn upload_audio(
    Extension(_app): Extension<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    match multipart.next_field().await {
        Ok(Some(field)) => {
            let orig_filename = field.file_name().map(|s| s.to_string());
            let content_type_opt = field.content_type().map(|s| s.to_string());

            let ext = match validate_audio_upload(
                orig_filename.as_deref(),
                content_type_opt.as_deref(),
            ) {
                Ok(ext) => ext,
                Err(msg) => return (StatusCode::BAD_REQUEST, msg).into_response(),
            };

            let data = match field.bytes().await {
                Ok(b) => b,
                Err(e) => {
                    error!("multipart bytes error: {}", e);
                    let msg = if e.to_string().contains("exceeded") {
                        "audio too large (limit 50 MB)"
                    } else {
                        "failed to read upload"
                    };
                    return (StatusCode::BAD_REQUEST, msg).into_response();
                }
            };

            let uploads_dir = get_cache_dir().join("uploads");
            if let Err(e) = tokio::fs::create_dir_all(&uploads_dir).await {
                error!("create uploads dir error: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "failed to create uploads directory",
                )
                    .into_response();
            }

            let filename = format!("{}.{}", Uuid::new_v4(), ext);
            let filepath = uploads_dir.join(&filename);
            if let Err(e) = tokio::fs::write(&filepath, &data).await {
                error!("write upload error: {}", e);
                return (StatusCode::INTERNAL_SERVER_ERROR, "failed to save audio").into_response();
            }

            let path = filepath.to_string_lossy().to_string();
            let url = format!("uploads/{filename}");
            (StatusCode::OK, Json(json!({ "path": path, "url": url }))).into_response()
        }
        _ => (StatusCode::BAD_REQUEST, "missing audio part").into_response(),
    }
}

pub async fn upload_video(
    Extension(_app): Extension<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    match multipart.next_field().await {
        Ok(Some(field)) => {
            let orig_filename = field.file_name().map(|s| s.to_string());
            let content_type_opt = field.content_type().map(|s| s.to_string());

            let ext = match validate_video_upload(
                orig_filename.as_deref(),
                content_type_opt.as_deref(),
            ) {
                Ok(ext) => ext,
                Err(msg) => return (StatusCode::BAD_REQUEST, msg).into_response(),
            };

            let data = match field.bytes().await {
                Ok(b) => b,
                Err(e) => {
                    error!("multipart bytes error: {}", e);
                    let msg = if e.to_string().contains("exceeded") {
                        "video too large (limit 50 MB)"
                    } else {
                        "failed to read upload"
                    };
                    return (StatusCode::BAD_REQUEST, msg).into_response();
                }
            };

            let uploads_dir = get_cache_dir().join("uploads");
            if let Err(e) = tokio::fs::create_dir_all(&uploads_dir).await {
                error!("create uploads dir error: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "failed to create uploads directory",
                )
                    .into_response();
            }

            let filename = format!("{}.{}", Uuid::new_v4(), ext);
            let filepath = uploads_dir.join(&filename);
            if let Err(e) = tokio::fs::write(&filepath, &data).await {
                error!("write upload error: {}", e);
                return (StatusCode::INTERNAL_SERVER_ERROR, "failed to save video").into_response();
            }

            let path = filepath.to_string_lossy().to_string();
            let url = format!("uploads/{filename}");
            (StatusCode::OK, Json(json!({ "path": path, "url": url }))).into_response()
        }
        _ => (StatusCode::BAD_REQUEST, "missing video part").into_response(),
    }
}

pub async fn upload_image(
    Extension(_app): Extension<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    match multipart.next_field().await {
        Ok(Some(field)) => {
            let orig_filename = field.file_name().map(|s| s.to_string());
            let content_type_opt = field.content_type().map(|s| s.to_string());

            let ext = match validate_image_upload(
                orig_filename.as_deref(),
                content_type_opt.as_deref(),
            ) {
                Ok(extension) => extension,
                Err(msg) => return (StatusCode::BAD_REQUEST, msg).into_response(),
            };

            let data = match field.bytes().await {
                Ok(b) => b,
                Err(e) => {
                    error!("multipart bytes error: {}", e);
                    let msg = if e.to_string().contains("exceeded") {
                        "image too large (limit 50 MB)"
                    } else {
                        "failed to read upload"
                    };
                    return (StatusCode::BAD_REQUEST, msg).into_response();
                }
            };

            let uploads_dir = get_cache_dir().join("uploads");
            if let Err(e) = tokio::fs::create_dir_all(&uploads_dir).await {
                error!("create uploads dir error: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "failed to create uploads directory",
                )
                    .into_response();
            }

            let filename = format!("{}.{}", Uuid::new_v4(), ext);
            let filepath = uploads_dir.join(&filename);
            if let Err(e) = tokio::fs::write(&filepath, &data).await {
                error!("write upload error: {}", e);
                return (StatusCode::INTERNAL_SERVER_ERROR, "failed to save image").into_response();
            }

            let path = filepath.to_string_lossy().to_string();
            let url = format!("uploads/{filename}");
            (StatusCode::OK, Json(json!({ "path": path, "url": url }))).into_response()
        }
        _ => (StatusCode::BAD_REQUEST, "missing image part").into_response(),
    }
}

pub async fn upload_text(
    Extension(_app): Extension<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    match multipart.next_field().await {
        Ok(Some(field)) => {
            let orig_filename = field.file_name().map(|s| s.to_string());
            let content_type_opt = field.content_type().map(|s| s.to_string());

            let ext =
                match validate_text_upload(orig_filename.as_deref(), content_type_opt.as_deref()) {
                    Ok(ext) => ext,
                    Err(msg) => return (StatusCode::BAD_REQUEST, msg).into_response(),
                };

            let data = match field.bytes().await {
                Ok(b) => b,
                Err(e) => {
                    error!("multipart bytes error: {}", e);
                    let msg = if e.to_string().contains("exceeded") {
                        "file too large (limit 50 MB)"
                    } else {
                        "failed to read upload"
                    };
                    return (StatusCode::BAD_REQUEST, msg).into_response();
                }
            };

            let uploads_dir = get_cache_dir().join("uploads");
            if let Err(e) = tokio::fs::create_dir_all(&uploads_dir).await {
                error!("create uploads dir error: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "failed to create uploads directory",
                )
                    .into_response();
            }

            let filename = format!("{}.{}", Uuid::new_v4(), ext);
            let filepath = uploads_dir.join(&filename);
            if let Err(e) = tokio::fs::write(&filepath, &data).await {
                error!("write upload error: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "failed to save text file",
                )
                    .into_response();
            }

            let path = filepath.to_string_lossy().to_string();
            let url = format!("uploads/{filename}");
            (StatusCode::OK, Json(json!({ "path": path, "url": url }))).into_response()
        }
        _ => (StatusCode::BAD_REQUEST, "missing text part").into_response(),
    }
}

pub async fn list_models(Extension(app): Extension<Arc<AppState>>) -> impl IntoResponse {
    let models: Vec<_> = app.models.values().cloned().collect();
    Json(json!({ "models": models }))
}

pub async fn select_model(
    Extension(app): Extension<Arc<AppState>>,
    Json(req): Json<SelectRequest>,
) -> impl IntoResponse {
    if !app.models.contains_key(&req.name) {
        return (
            StatusCode::BAD_REQUEST,
            format!("Model '{}' not found", req.name),
        )
            .into_response();
    }
    let mut cur = app.current.write().await;
    *cur = Some(req.name);
    (StatusCode::OK, "Selected").into_response()
}

pub async fn list_chats(Extension(app): Extension<Arc<AppState>>) -> impl IntoResponse {
    let dir = &app.chats_dir;
    let mut chats = Vec::new();

    if let Ok(mut entries) = fs::read_dir(dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let filename = entry.file_name().to_string_lossy().to_string();
            let id = filename
                .strip_suffix(".json")
                .unwrap_or(&filename)
                .to_string();
            if let Ok(bytes) = fs::read(entry.path()).await
                && let Ok(chat) = serde_json::from_slice::<ChatFile>(&bytes)
            {
                let mut value = serde_json::to_value(&chat).unwrap();
                value["id"] = serde_json::Value::String(id);
                chats.push(value);
            }
        }
    }
    chats.sort_by(|a, b| {
        let a_date = a["created_at"].as_str().unwrap_or("");
        let b_date = b["created_at"].as_str().unwrap_or("");
        b_date.cmp(a_date)
    });
    Json(json!({ "chats": chats }))
}

pub async fn new_chat(
    Extension(app): Extension<Arc<AppState>>,
    Json(req): Json<NewChatRequest>,
) -> impl IntoResponse {
    if !app.models.contains_key(&req.model) {
        return (StatusCode::BAD_REQUEST, "Unknown model").into_response();
    }

    let chat_id = format!("chat_{}", Uuid::new_v4().simple());

    let now = Utc::now().to_rfc3339();
    let kind = app
        .models
        .get(&req.model)
        .map(|m| m.kind.clone())
        .unwrap_or_else(|| "text".to_string());
    let chat = ChatFile {
        title: None,
        model: req.model,
        kind,
        created_at: now,
        messages: Vec::new(),
        session_id: None,
        tail: None,
    };

    let path = app
        .chat_path(&chat_id)
        .expect("server-generated chat ids are valid");
    if let Err(e) = fs::write(&path, serde_json::to_vec_pretty(&chat).unwrap()).await {
        error!("write chat error: {}", e);
        return (StatusCode::INTERNAL_SERVER_ERROR, "write failed").into_response();
    }

    let mut cur = app.current_chat.write().await;
    *cur = Some(chat_id.clone());
    Json(json!({ "id": chat_id })).into_response()
}

pub async fn delete_chat(
    Extension(app): Extension<Arc<AppState>>,
    Json(req): Json<DeleteChatRequest>,
) -> impl IntoResponse {
    let (Some(path), Some(session_path)) = (app.chat_path(&req.id), app.chat_session_path(&req.id))
    else {
        return (StatusCode::BAD_REQUEST, INVALID_CHAT_ID).into_response();
    };
    // Best-effort delete the session sidecar; ignore errors (it may not exist)
    let _ = fs::remove_file(&session_path).await;
    match fs::remove_file(&path).await {
        Ok(_) => (StatusCode::OK, "Deleted").into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "Chat not found").into_response(),
    }
}

pub async fn load_chat(
    Extension(app): Extension<Arc<AppState>>,
    Json(req): Json<LoadChatRequest>,
) -> impl IntoResponse {
    let Some(path) = app.chat_path(&req.id) else {
        return (StatusCode::BAD_REQUEST, INVALID_CHAT_ID).into_response();
    };
    if let Ok(bytes) = fs::read(&path).await
        && let Ok(chat) = serde_json::from_slice::<ChatFile>(&bytes)
    {
        let mut cur = app.current_chat.write().await;
        *cur = Some(req.id.clone());
        return Json(chat).into_response();
    }
    (StatusCode::NOT_FOUND, "Chat not found").into_response()
}

pub async fn rename_chat(
    Extension(app): Extension<Arc<AppState>>,
    Json(req): Json<RenameChatRequest>,
) -> impl IntoResponse {
    let Some(path) = app.chat_path(&req.id) else {
        return (StatusCode::BAD_REQUEST, INVALID_CHAT_ID).into_response();
    };
    if let Ok(bytes) = fs::read(&path).await
        && let Ok(mut chat) = serde_json::from_slice::<ChatFile>(&bytes)
    {
        chat.title = Some(req.title);
        if fs::write(&path, serde_json::to_vec_pretty(&chat).unwrap())
            .await
            .is_ok()
        {
            return (StatusCode::OK, "Renamed").into_response();
        }
    }
    (StatusCode::INTERNAL_SERVER_ERROR, "rename failed").into_response()
}

#[derive(Deserialize)]
pub struct AppendMessageRequest {
    pub id: String,
    #[serde(default)]
    pub message_id: Option<String>,
    #[serde(default)]
    pub parent_id: Option<String>,
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub images: Option<Vec<String>>,
    #[serde(default)]
    pub videos: Option<Vec<String>>,
    #[serde(default)]
    pub blocks: Option<serde_json::Value>,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub elapsed_ms: Option<f64>,
    #[serde(default)]
    pub ttft_ms: Option<f64>,
    #[serde(default)]
    pub tokens: Option<u32>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
}

pub async fn append_message(
    Extension(app): Extension<Arc<AppState>>,
    Json(req): Json<AppendMessageRequest>,
) -> impl IntoResponse {
    if let Err(e) = append_chat_message(
        &app,
        &req.id,
        req.message_id,
        req.parent_id,
        &req.role,
        &req.content,
        req.images,
        req.videos,
        req.blocks,
        req.finish_reason,
        crate::handlers::api::MessageStats {
            elapsed_ms: req.elapsed_ms,
            ttft_ms: req.ttft_ms,
            tokens: req.tokens,
            model: req.model,
            session_id: req.session_id,
        },
    )
    .await
    {
        error!("append message error: {}", e);
        return (StatusCode::INTERNAL_SERVER_ERROR, "append failed").into_response();
    }
    (StatusCode::OK, "Appended").into_response()
}

#[derive(Deserialize)]
pub struct EditMessageRequest {
    pub id: String,
    pub message_id: String,
    pub content: String,
}

pub async fn edit_message(
    Extension(app): Extension<Arc<AppState>>,
    Json(req): Json<EditMessageRequest>,
) -> impl IntoResponse {
    if let Err(e) =
        crate::chat::edit_chat_message(&app, &req.id, &req.message_id, &req.content).await
    {
        error!("edit message error: {}", e);
        return (StatusCode::INTERNAL_SERVER_ERROR, "edit failed").into_response();
    }
    (StatusCode::OK, "Edited").into_response()
}

#[derive(Deserialize)]
pub struct SetTailRequest {
    pub id: String,
    #[serde(default)]
    pub tail: Option<String>,
}

pub async fn set_tail(
    Extension(app): Extension<Arc<AppState>>,
    Json(req): Json<SetTailRequest>,
) -> impl IntoResponse {
    if let Err(e) = crate::chat::set_chat_tail(&app, &req.id, req.tail).await {
        error!("set tail error: {}", e);
        return (StatusCode::INTERNAL_SERVER_ERROR, "set_tail failed").into_response();
    }
    (StatusCode::OK, "OK").into_response()
}

#[derive(Deserialize)]
pub struct ForkSessionRequest {
    pub src_session_id: String,
    pub num_turns: usize,
}

pub async fn fork_session(
    Extension(app): Extension<Arc<AppState>>,
    Json(req): Json<ForkSessionRequest>,
) -> impl IntoResponse {
    // server-named, so a fork can't land on a session that already exists
    let session_id = Uuid::new_v4().to_string();
    let result = app.inference.fork_session(
        None,
        &req.src_session_id,
        session_id.clone(),
        req.num_turns,
        app.owner.as_deref(),
    );
    if let Err(e) = result {
        error!("fork session error: {}", e);
        return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
    }
    Json(json!({ "session_id": session_id })).into_response()
}

#[derive(Default)]
pub struct MessageStats {
    pub elapsed_ms: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub tokens: Option<u32>,
    pub model: Option<String>,
    pub session_id: Option<String>,
}

#[derive(Deserialize)]
pub struct GenerateSpeechRequest {
    pub text: String,
}

pub async fn get_settings(Extension(app): Extension<Arc<AppState>>) -> impl IntoResponse {
    let current_model = app.current.read().await.clone();
    let defaults = current_model
        .as_ref()
        .and_then(|model_id| app.models.get(model_id))
        .map(|model| model.generation_defaults.clone())
        .unwrap_or_else(|| app.default_params.clone());

    Json(json!({
        "defaults": {
            "temperature": defaults.temperature,
            "top_p": defaults.top_p,
            "top_k": defaults.top_k,
            "max_tokens": defaults.max_tokens,
            "repetition_penalty": defaults.repetition_penalty,
            "system_prompt": defaults.system_prompt,
        },
        "model": current_model,
        "search_enabled": app.search_enabled,
        "search_embedding_model": app.search_embedding_model.map(|m| m.to_string()),
    }))
}

pub async fn generate_speech(
    Extension(app): Extension<Arc<AppState>>,
    Json(req): Json<GenerateSpeechRequest>,
) -> impl IntoResponse {
    let model_name = {
        let cur = app.current.read().await;
        if let Some(name) = &*cur {
            name.clone()
        } else {
            return (StatusCode::BAD_REQUEST, "No model selected").into_response();
        }
    };

    let kind = app
        .models
        .get(&model_name)
        .map(|m| m.kind.as_str())
        .unwrap_or("text");
    if kind != "speech" {
        return (
            StatusCode::BAD_REQUEST,
            "Selected model is not a speech model",
        )
            .into_response();
    }

    let request = SpeechGenerationRequest {
        model: model_name,
        input: req.text,
        response_format: AudioResponseFormat::Wav,
    };
    let audio = match inference_api::generation::generate_speech(&app.inference, request).await {
        Ok(audio) => audio,
        Err(e) => {
            error!("speech generation error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "speech generation failed",
            )
                .into_response();
        }
    };

    let filename = format!("{}.wav", Uuid::new_v4());
    let filepath = PathBuf::from(&app.speech_dir).join(&filename);
    if let Err(e) = fs::write(&filepath, &audio.bytes).await {
        error!("failed to write wav file: {}", e);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "failed to write wav file",
        )
            .into_response();
    }

    let url = format!("speech/{filename}");
    (StatusCode::OK, Json(json!({ "url": url }))).into_response()
}

pub async fn get_capabilities(Extension(app): Extension<Arc<AppState>>) -> impl IntoResponse {
    Json(json!({
        "search_enabled": app.search_enabled,
        "code_execution_enabled": app.code_execution_enabled,
        "shell_enabled": app.shell_enabled,
        "tool_dispatch_url": app.tool_dispatch_url,
        "signed_in": app.owner.is_some(),
    }))
}

#[derive(Deserialize)]
pub struct SaveChatSessionRequest {
    pub chat_id: String,
    pub session_id: String,
}

/// Export the agentic session to a sidecar file beside the chat. Stamps the session_id into the chat JSON.
pub async fn save_chat_session(
    Extension(app): Extension<Arc<AppState>>,
    Json(req): Json<SaveChatSessionRequest>,
) -> impl IntoResponse {
    // Export the session from the in-memory store
    let session = match app
        .inference
        .export_session(None, &req.session_id, app.owner.as_deref())
    {
        Ok(Some(s)) => s,
        Ok(None) => {
            return (StatusCode::NOT_FOUND, "Session not found in store").into_response();
        }
        Err(e) => {
            error!("export_session error: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "export failed").into_response();
        }
    };

    // Write session blob to sidecar file
    let (Some(session_path), Some(chat_path)) = (
        app.chat_session_path(&req.chat_id),
        app.chat_path(&req.chat_id),
    ) else {
        return (StatusCode::BAD_REQUEST, INVALID_CHAT_ID).into_response();
    };
    let session_bytes = match serde_json::to_vec(&session) {
        Ok(b) => b,
        Err(e) => {
            error!("serialize session error: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "serialize failed").into_response();
        }
    };
    if let Err(e) = fs::write(&session_path, &session_bytes).await {
        error!("write session sidecar error: {}", e);
        return (StatusCode::INTERNAL_SERVER_ERROR, "write sidecar failed").into_response();
    }

    // Stamp session_id into the chat JSON for fast lookup
    if let Ok(bytes) = fs::read(&chat_path).await
        && let Ok(mut chat) = serde_json::from_slice::<ChatFile>(&bytes)
    {
        chat.session_id = Some(req.session_id);
        let _ = fs::write(&chat_path, serde_json::to_vec_pretty(&chat).unwrap()).await;
    }

    (StatusCode::OK, "Saved").into_response()
}

#[derive(Deserialize)]
pub struct RestoreChatSessionRequest {
    pub chat_id: String,
}

/// Import the sidecar session into the in-memory store under the chat's saved session_id. Returns the id, or null if none.
pub async fn restore_chat_session(
    Extension(app): Extension<Arc<AppState>>,
    Json(req): Json<RestoreChatSessionRequest>,
) -> impl IntoResponse {
    let (Some(chat_path), Some(session_path)) = (
        app.chat_path(&req.chat_id),
        app.chat_session_path(&req.chat_id),
    ) else {
        return (StatusCode::BAD_REQUEST, INVALID_CHAT_ID).into_response();
    };
    let session_id = match fs::read(&chat_path).await {
        Ok(bytes) => match serde_json::from_slice::<ChatFile>(&bytes) {
            Ok(chat) => chat.session_id,
            Err(_) => None,
        },
        Err(_) => return (StatusCode::NOT_FOUND, "Chat not found").into_response(),
    };

    let Some(session_id) = session_id else {
        return Json(json!({ "session_id": serde_json::Value::Null })).into_response();
    };

    let bytes = match fs::read(&session_path).await {
        Ok(b) => b,
        Err(_) => {
            // Sidecar missing. No persisted session.
            return Json(json!({ "session_id": serde_json::Value::Null })).into_response();
        }
    };

    let serialized: inference_core::SerializedSession = match serde_json::from_slice(&bytes) {
        Ok(s) => s,
        Err(e) => {
            error!("parse session sidecar error: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "parse sidecar failed").into_response();
        }
    };

    if let Err(e) =
        app.inference
            .import_session(None, session_id.clone(), serialized, app.owner.as_deref())
    {
        error!("import_session error: {}", e);
        return (StatusCode::INTERNAL_SERVER_ERROR, "import failed").into_response();
    }

    Json(json!({ "session_id": session_id })).into_response()
}

/// Return the list of MCP-provided tools registered on the default model.
pub async fn list_mcp_tools(Extension(app): Extension<Arc<AppState>>) -> impl IntoResponse {
    match app.inference.list_mcp_tools(None) {
        Ok(tools) => {
            let payload: Vec<_> = tools
                .into_iter()
                .map(|(name, description)| json!({ "name": name, "description": description }))
                .collect();
            Json(json!({ "tools": payload })).into_response()
        }
        Err(e) => {
            error!("list_mcp_tools error: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, e).into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uploads_are_checked_against_the_lists_media_sources_accept() {
        assert_eq!(
            validate_image_upload(Some("Photo.PNG"), Some("image/png")),
            Ok("png".to_string())
        );
        assert_eq!(
            validate_video_upload(Some("clip.gif"), Some("image/gif")),
            Ok("gif".to_string())
        );
        assert_eq!(
            validate_audio_upload(Some("take.opus"), None),
            Ok("opus".to_string())
        );
        assert_eq!(
            validate_image_upload(Some("scan.tiff"), Some("image/tiff")),
            Err("Unsupported image format")
        );
        assert_eq!(
            validate_audio_upload(Some("take.wav"), Some("video/mp4")),
            Err("File must be an audio file")
        );
        assert_eq!(
            validate_video_upload(None, None),
            Err("No filename provided")
        );
        assert_eq!(
            validate_image_upload(Some("photo."), None),
            Err("No file extension")
        );
        for ext in IMAGE_UPLOAD_EXTENSIONS {
            assert!(validate_image_upload(Some(&format!("a.{ext}")), None).is_ok());
        }
    }

    #[test]
    fn text_uploads_accept_source_files_by_extension() {
        assert_eq!(
            validate_text_upload(Some("notes.R"), None),
            Ok("r".to_string())
        );
        assert_eq!(
            validate_text_upload(Some("main.rs"), Some("application/x-rust")),
            Ok("rs".to_string())
        );
        assert_eq!(
            validate_text_upload(Some("image.png"), Some("image/png")),
            Err("File must be a text file")
        );
    }
}
