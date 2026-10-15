//! The history of the UI's non-chat jobs (transcriptions, speech, images, documents): each job is a directory of a
//! `job.json` record and the media it took and made, kept per owner like the chats.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::Json;
use axum::body::Body;
use axum::extract::{Extension, Multipart, Path as UrlPath, Query, Request};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::fs;
use tower_http::services::ServeFile;
use tracing::error;
use uuid::Uuid;

use crate::types::AppState;

const JOB_RECORD: &str = "job.json";
const JOB_RECORD_PARTIAL: &str = "job.json.partial";
// held while a record is read and written back, or its job deleted; saves are rare, so one lock serves every job
static RECORDS: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
const JOB_ID_PREFIX: &str = "job_";
const JOB_FIELD: &str = "job";
pub(crate) const JOB_KINDS: &[&str] = &["transcription", "speech", "images", "documents"];
// what a job file may be, and the type it is served as; never markup or script, since it is served on the UI's origin
const JOB_FILE_TYPES: &[(&str, &str)] = &[
    ("png", "image/png"),
    ("jpg", "image/jpeg"),
    ("jpeg", "image/jpeg"),
    ("webp", "image/webp"),
    ("gif", "image/gif"),
    ("bmp", "image/bmp"),
    ("wav", "audio/wav"),
    ("mp3", "audio/mpeg"),
    ("ogg", "audio/ogg"),
    ("opus", "audio/ogg"),
    ("flac", "audio/flac"),
    ("m4a", "audio/mp4"),
    ("aac", "audio/aac"),
    ("webm", "audio/webm"),
    ("mka", "audio/x-matroska"),
    ("mkv", "video/x-matroska"),
    ("aiff", "audio/aiff"),
];

#[derive(Serialize, Deserialize)]
pub struct JobRecord {
    pub id: String,
    pub kind: String,
    pub created_at: String,
    #[serde(default)]
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The view's own fields: its inputs and the results that aren't files.
    #[serde(default)]
    pub data: serde_json::Value,
    /// The names of the job's files, served at `api/job_file/<id>/<name>`.
    #[serde(default)]
    pub files: Vec<String>,
}

// the `job` field of a save: a new job, or with `id` new data and more files for one already kept
#[derive(Deserialize)]
struct SaveJob {
    #[serde(default)]
    id: Option<String>,
    kind: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    data: serde_json::Value,
}

#[derive(Deserialize)]
pub struct ListJobsQuery {
    pub kind: String,
}

#[derive(Deserialize)]
pub struct DeleteJobRequest {
    pub id: String,
}

fn is_plain(part: &str) -> bool {
    !part.is_empty()
        && part
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

// ids are server-generated (`job_<uuid>`), so anything else is a client trying to leave the jobs dir
fn job_dir(jobs_dir: &str, id: &str) -> Option<PathBuf> {
    (id.starts_with(JOB_ID_PREFIX) && is_plain(id)).then(|| Path::new(jobs_dir).join(id))
}

// `<stem>.<ext>` with a plain stem and a listed extension, and the type it is served as
fn job_file_type(name: &str) -> Option<&'static str> {
    let (stem, ext) = name.rsplit_once('.')?;
    let ext = ext.to_ascii_lowercase();
    is_plain(stem)
        .then(|| JOB_FILE_TYPES.iter().find(|(known, _)| *known == ext))
        .flatten()
        .map(|(_, mime)| *mime)
}

async fn read_record(dir: &Path) -> Option<JobRecord> {
    let bytes = fs::read(dir.join(JOB_RECORD)).await.ok()?;
    serde_json::from_slice(&bytes).ok()
}

// through a temporary file, so a reader never sees half a record
async fn write_record(dir: &Path, record: &JobRecord) -> std::io::Result<()> {
    let partial = dir.join(JOB_RECORD_PARTIAL);
    fs::write(&partial, serde_json::to_vec_pretty(record)?).await?;
    fs::rename(&partial, dir.join(JOB_RECORD)).await
}

// a refused request, small enough to return as an error
type Refusal = (StatusCode, String);

fn bad_request(message: impl Into<String>) -> Refusal {
    (StatusCode::BAD_REQUEST, message.into())
}

pub async fn list_jobs(
    Extension(app): Extension<Arc<AppState>>,
    Query(query): Query<ListJobsQuery>,
) -> impl IntoResponse {
    let mut jobs = Vec::new();
    if let Ok(mut entries) = fs::read_dir(&app.jobs_dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let name = entry.file_name().to_string_lossy().to_string();
            if job_dir(&app.jobs_dir, &name).is_none() {
                continue;
            }
            if let Some(record) = read_record(&entry.path()).await
                && record.kind == query.kind
            {
                jobs.push(record);
            }
        }
    }
    jobs.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Json(json!({ "jobs": jobs }))
}

/// Keeps a job: a multipart body of the `job` field (JSON) first, then its files, each named by its filename.
pub async fn save_job(
    Extension(app): Extension<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response {
    let job = match multipart.next_field().await {
        Ok(Some(field)) if field.name() == Some(JOB_FIELD) => match field.text().await {
            Ok(text) => text,
            Err(_) => return bad_request("failed to read the job").into_response(),
        },
        _ => return bad_request(format!("the first part must be `{JOB_FIELD}`")).into_response(),
    };
    let job: SaveJob = match serde_json::from_str(&job) {
        Ok(job) => job,
        Err(e) => return bad_request(format!("invalid job: {e}")).into_response(),
    };
    if !JOB_KINDS.contains(&job.kind.as_str()) {
        return bad_request(format!("unknown job kind `{}`", job.kind)).into_response();
    }
    let new = job.id.is_none();
    let (dir, record) = match job.id {
        Some(id) => {
            let Some(dir) = job_dir(&app.jobs_dir, &id) else {
                return bad_request("invalid job id").into_response();
            };
            match read_record(&dir).await {
                Some(record) if record.kind == job.kind => (dir, record),
                _ => return (StatusCode::NOT_FOUND, "job not found").into_response(),
            }
        }
        None => {
            let id = format!("{JOB_ID_PREFIX}{}", Uuid::new_v4().simple());
            let dir = job_dir(&app.jobs_dir, &id).expect("server-generated job ids are valid");
            if let Err(e) = fs::create_dir_all(&dir).await {
                error!("create job dir error: {e}");
                return internal_error().into_response();
            }
            let record = JobRecord {
                id,
                kind: job.kind,
                created_at: Utc::now().to_rfc3339(),
                title: String::new(),
                model: None,
                data: serde_json::Value::Null,
                files: Vec::new(),
            };
            (dir, record)
        }
    };

    let kept = async {
        let names = keep_files(&mut multipart, &dir).await?;
        let _records = RECORDS.lock().await;
        // read again under the lock, so a concurrent update or delete of the same job isn't lost
        let mut record = if new {
            record
        } else {
            read_record(&dir)
                .await
                .ok_or_else(|| (StatusCode::NOT_FOUND, "job not found".to_string()))?
        };
        record.title = job.title;
        record.model = job.model;
        record.data = job.data;
        for name in names {
            if !record.files.contains(&name) {
                record.files.push(name);
            }
        }
        write_record(&dir, &record).await.map_err(|e| {
            error!("write job error: {e}");
            internal_error()
        })?;
        Ok::<_, Refusal>(record)
    };
    match kept.await {
        Ok(record) => Json(record).into_response(),
        Err(refusal) => {
            // a job refused part way leaves nothing behind; one kept before keeps its old record
            if new {
                let _ = fs::remove_dir_all(&dir).await;
            }
            refusal.into_response()
        }
    }
}

fn internal_error() -> Refusal {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        "failed to keep the job".to_string(),
    )
}

// the multipart's files, written into `dir`; their names
async fn keep_files(multipart: &mut Multipart, dir: &Path) -> Result<Vec<String>, Refusal> {
    let mut names = Vec::new();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| bad_request(format!("failed to read a file: {e}")))?
    {
        let Some(name) = field.file_name().map(str::to_string) else {
            return Err(bad_request("every file needs a filename"));
        };
        if job_file_type(&name).is_none() {
            return Err(bad_request(format!(
                "`{name}` is not a job file this UI keeps"
            )));
        }
        let bytes = field
            .bytes()
            .await
            .map_err(|e| bad_request(format!("failed to read `{name}`: {e}")))?;
        fs::write(dir.join(&name), &bytes).await.map_err(|e| {
            error!("write job file error: {e}");
            internal_error()
        })?;
        names.push(name);
    }
    Ok(names)
}

pub async fn delete_job(
    Extension(app): Extension<Arc<AppState>>,
    Json(req): Json<DeleteJobRequest>,
) -> Response {
    let Some(dir) = job_dir(&app.jobs_dir, &req.id) else {
        return bad_request("invalid job id").into_response();
    };
    let _records = RECORDS.lock().await;
    match fs::remove_dir_all(&dir).await {
        Ok(()) => (StatusCode::OK, "Deleted").into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "job not found").into_response(),
    }
}

/// A job's file, served with range support so a player can seek in its audio.
pub async fn job_file(
    Extension(app): Extension<Arc<AppState>>,
    UrlPath((id, name)): UrlPath<(String, String)>,
    request: Request,
) -> Response {
    let (Some(dir), Some(mime)) = (job_dir(&app.jobs_dir, &id), job_file_type(&name)) else {
        return bad_request("invalid job file").into_response();
    };
    let mime: mime_guess::mime::Mime = mime.parse().expect("job file types are valid media types");
    match ServeFile::new_with_mime(dir.join(&name), &mime)
        .try_call(request)
        .await
    {
        Ok(response) => {
            let mut response = response.map(Body::new);
            let headers = response.headers_mut();
            headers.insert(
                header::X_CONTENT_TYPE_OPTIONS,
                HeaderValue::from_static("nosniff"),
            );
            headers.insert(
                header::CONTENT_SECURITY_POLICY,
                HeaderValue::from_static("sandbox"),
            );
            response
        }
        Err(e) => {
            error!("read job file error: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to read the job file",
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{job_dir, job_file_type};

    #[test]
    fn job_paths_stay_inside_the_jobs_dir() {
        assert_eq!(
            job_dir("/j", "job_12ab"),
            Some(std::path::Path::new("/j").join("job_12ab"))
        );
        for id in ["", "..", "job_../x", "job_a/b", "owners", "chat_1"] {
            assert_eq!(job_dir("/j", id), None, "{id:?}");
        }
        assert_eq!(job_file_type("page.png"), Some("image/png"));
        assert_eq!(job_file_type("audio.MP3"), Some("audio/mpeg"));
        for name in [
            "page",
            ".png",
            "../page.png",
            "a/b.png",
            "page.svg",
            "page.html",
            "a.b.png",
        ] {
            assert_eq!(job_file_type(name), None, "{name:?}");
        }
    }
}
