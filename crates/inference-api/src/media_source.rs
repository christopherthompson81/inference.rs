use std::sync::Arc;
use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
    time::Duration,
};

use anyhow::{Context, Result};
use image::DynamicImage;
use inference_core::remote_fetch::{FetchOptions, NetworkPolicy, fetch_limited};
use inference_core::{AudioInput, VideoInput};
use tokio::{fs::File, io::AsyncReadExt};
use url::Url;

pub const MAX_MEDIA_BYTES: usize = 64 * 1024 * 1024;
pub const MEDIA_FETCH_TIMEOUT: Duration = Duration::from_secs(30);
pub const MEDIA_FETCH_REDIRECTS: usize = 3;
pub const SERVER_VIDEO_FRAME_LIMIT: usize = 32;

const DATA_URL_HEADER_ALLOWANCE: usize = 4096;
const UI_UPLOAD_SCHEME: &str = "inference-upload";
// The web UI accepts uploads by these lists, and an `inference-upload:` source is refused outside them.
pub const IMAGE_UPLOAD_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "gif", "webp", "bmp", "svg"];
pub const VIDEO_UPLOAD_EXTENSIONS: &[&str] = &["mp4", "avi", "mov", "mkv", "webm", "m4v", "gif"];
pub const AUDIO_UPLOAD_EXTENSIONS: &[&str] =
    &["wav", "mp3", "ogg", "flac", "m4a", "aac", "opus", "webm"];

static UI_UPLOAD_DIR: OnceLock<PathBuf> = OnceLock::new();

#[derive(Clone, Copy)]
pub enum MediaSourcePolicy {
    ServerRequest,
    Local,
}

/// A request names an attached buffer as `media://<index>`.
pub const ATTACHMENT_PREFIX: &str = "media://";

/// Media the caller passed alongside a request instead of by URL.
#[derive(Clone, Debug)]
pub struct MediaAttachment {
    pub bytes: Vec<u8>,
    pub mime_type: Option<String>,
}

/// One attachment: encoded bytes, or media a Rust caller already decoded, which skips an encode and decode.
#[derive(Clone)]
pub enum Media {
    Encoded(MediaAttachment),
    Image(DynamicImage),
    Audio(AudioInput),
    Video(VideoInput),
}

impl std::fmt::Debug for Media {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Encoded(attachment) => attachment.fmt(f),
            decoded => write!(f, "decoded {}", decoded.kind()),
        }
    }
}

impl Media {
    fn kind(&self) -> &'static str {
        match self {
            Self::Encoded(_) => "encoded",
            Self::Image(_) => "image",
            Self::Audio(_) => "audio",
            Self::Video(_) => "video",
        }
    }
}

/// The attachments of one request, indexed by position.
#[derive(Clone, Debug, Default)]
pub struct MediaAttachments(Arc<Vec<Media>>);

impl MediaAttachments {
    pub fn new(attachments: Vec<MediaAttachment>) -> Self {
        Self::from_media(attachments.into_iter().map(Media::Encoded).collect())
    }

    pub fn from_media(media: Vec<Media>) -> Self {
        Self(Arc::new(media))
    }

    /// The attachment `source` names, `None` when it names none (a URL or path).
    pub(crate) fn attached(&self, source: &str, kind: &str) -> Result<Option<&Media>> {
        let Some(index) = source.strip_prefix(ATTACHMENT_PREFIX) else {
            return Ok(None);
        };
        index
            .parse::<usize>()
            .ok()
            .and_then(|index| self.0.get(index))
            .map(Some)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{kind} source `{source}` names no attachment ({} attached)",
                    self.0.len()
                )
            })
    }

    /// Loads `source` from the attachments when it names one, otherwise as a URL or path under `policy`.
    pub async fn load(
        &self,
        source: &str,
        policy: MediaSourcePolicy,
        kind: &str,
    ) -> Result<LoadedMedia> {
        let attachment = match self.attached(source, kind)? {
            None => return load_media_source(source, policy, kind).await,
            Some(Media::Encoded(attachment)) => attachment,
            Some(decoded) => {
                anyhow::bail!("{kind} source `{source}` is a decoded {}", decoded.kind())
            }
        };
        Ok(LoadedMedia {
            bytes: attachment.bytes.clone(),
            mime_type: attachment.mime_type.clone(),
            final_url: None,
        })
    }
}

pub struct LoadedMedia {
    pub bytes: Vec<u8>,
    pub mime_type: Option<String>,
    pub final_url: Option<Url>,
}

pub async fn configure_ui_upload_dir(path: impl AsRef<Path>) -> Result<()> {
    let path = tokio::fs::canonicalize(path.as_ref())
        .await
        .context("Failed to resolve UI uploads directory")?;
    if !tokio::fs::metadata(&path).await?.is_dir() {
        anyhow::bail!("UI uploads path is not a directory.");
    }
    if let Some(configured) = UI_UPLOAD_DIR.get() {
        if configured != &path {
            anyhow::bail!("UI uploads directory is already configured.");
        }
        return Ok(());
    }
    match UI_UPLOAD_DIR.set(path) {
        Ok(()) => Ok(()),
        Err(path) if UI_UPLOAD_DIR.get() == Some(&path) => Ok(()),
        Err(_) => anyhow::bail!("UI uploads directory was configured concurrently"),
    }
}

pub async fn load_media_source(
    source: &str,
    policy: MediaSourcePolicy,
    kind: &str,
) -> Result<LoadedMedia> {
    match Url::parse(source) {
        Ok(url) => load_media_url(url, policy, kind).await,
        Err(err) => {
            if matches!(policy, MediaSourcePolicy::Local) {
                let path = PathBuf::from(source);
                let bytes = read_local_file_limited(&path, MAX_MEDIA_BYTES)
                    .await
                    .with_context(|| format!("Failed to read local {kind} path"))?;
                Ok(LoadedMedia {
                    bytes,
                    mime_type: None,
                    final_url: None,
                })
            } else {
                anyhow::bail!("Invalid {kind} source: expected http, https, or data URL: {err}");
            }
        }
    }
}

async fn load_media_url(url: Url, policy: MediaSourcePolicy, kind: &str) -> Result<LoadedMedia> {
    match url.scheme() {
        "http" | "https" => {
            let network = match policy {
                MediaSourcePolicy::ServerRequest => NetworkPolicy::PublicOnly,
                MediaSourcePolicy::Local => NetworkPolicy::Any,
            };
            fetch_media(url, MAX_MEDIA_BYTES, kind, network).await
        }
        "data" => {
            let bytes = decode_data_url_limited(url.as_str(), MAX_MEDIA_BYTES, kind)?;
            Ok(LoadedMedia {
                bytes,
                mime_type: data_url_mime(url.as_str()),
                final_url: Some(url),
            })
        }
        "file" if matches!(policy, MediaSourcePolicy::Local) => {
            let path = url
                .to_file_path()
                .map_err(|_| anyhow::anyhow!("Invalid local {kind} file URL."))?;
            let bytes = read_local_file_limited(&path, MAX_MEDIA_BYTES).await?;
            Ok(LoadedMedia {
                bytes,
                mime_type: None,
                final_url: Some(url),
            })
        }
        "file" => anyhow::bail!("Server request {kind} sources do not support file URLs."),
        UI_UPLOAD_SCHEME if matches!(policy, MediaSourcePolicy::ServerRequest) => {
            load_ui_upload(url, kind).await
        }
        scheme => anyhow::bail!("Unsupported {kind} URL scheme: {scheme}"),
    }
}

async fn load_ui_upload(url: Url, kind: &str) -> Result<LoadedMedia> {
    let root = UI_UPLOAD_DIR
        .get()
        .ok_or_else(|| anyhow::anyhow!("Unsupported {kind} URL scheme: {UI_UPLOAD_SCHEME}"))?;
    load_ui_upload_from_root(url, kind, root, MAX_MEDIA_BYTES).await
}

async fn load_ui_upload_from_root(
    url: Url,
    kind: &str,
    root: &Path,
    max_bytes: usize,
) -> Result<LoadedMedia> {
    let filename = ui_upload_filename(&url, kind)?;
    let path = resolve_ui_upload_path(root, filename).await?;
    let bytes = read_local_file_limited(&path, max_bytes).await?;
    Ok(LoadedMedia {
        bytes,
        mime_type: None,
        final_url: Some(url),
    })
}

fn ui_upload_filename<'a>(url: &'a Url, kind: &str) -> Result<&'a str> {
    if url.scheme() != UI_UPLOAD_SCHEME
        || url.host().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        anyhow::bail!("Invalid UI upload reference.");
    }

    let filename = url.path();
    if url.as_str() != format!("{UI_UPLOAD_SCHEME}:{filename}") {
        anyhow::bail!("Invalid UI upload reference.");
    }
    let (id, extension) = filename
        .rsplit_once('.')
        .ok_or_else(|| anyhow::anyhow!("Invalid UI upload filename."))?;
    let uuid = uuid::Uuid::parse_str(id).context("Invalid UI upload filename")?;
    if uuid.get_version_num() != 4 || uuid.hyphenated().to_string() != id {
        anyhow::bail!("Invalid UI upload filename.");
    }

    let extensions = match kind {
        "image" => IMAGE_UPLOAD_EXTENSIONS,
        "video" => VIDEO_UPLOAD_EXTENSIONS,
        "audio" => AUDIO_UPLOAD_EXTENSIONS,
        _ => anyhow::bail!("UI uploads do not support {kind} sources."),
    };
    if !extensions.contains(&extension) {
        anyhow::bail!("Unsupported UI upload format for {kind}.");
    }
    Ok(filename)
}

async fn resolve_ui_upload_path(root: &Path, filename: &str) -> Result<PathBuf> {
    let candidate = root.join(filename);
    let metadata = tokio::fs::symlink_metadata(&candidate)
        .await
        .with_context(|| format!("Could not read UI upload metadata: {}", candidate.display()))?;
    if metadata.file_type().is_symlink() {
        anyhow::bail!("UI upload references may not target symlinks.");
    }
    let path = tokio::fs::canonicalize(&candidate)
        .await
        .with_context(|| format!("Could not resolve UI upload: {}", candidate.display()))?;
    if path.parent() != Some(root) {
        anyhow::bail!("UI upload resolved outside the uploads directory.");
    }
    Ok(path)
}

pub async fn fetch_remote_limited(url: Url, max_bytes: usize, kind: &str) -> Result<LoadedMedia> {
    fetch_media(url, max_bytes, kind, NetworkPolicy::PublicOnly).await
}

async fn fetch_media(
    url: Url,
    max_bytes: usize,
    kind: &str,
    network: NetworkPolicy,
) -> Result<LoadedMedia> {
    let options = FetchOptions {
        max_bytes,
        timeout: MEDIA_FETCH_TIMEOUT,
        max_redirects: MEDIA_FETCH_REDIRECTS,
        user_agent: None,
    };
    let fetched = fetch_limited(url, options, network, kind).await?;
    Ok(LoadedMedia {
        bytes: fetched.bytes,
        mime_type: fetched.mime_type,
        final_url: Some(fetched.final_url),
    })
}

pub async fn read_local_file_limited(path: &Path, max_bytes: usize) -> Result<Vec<u8>> {
    let metadata = tokio::fs::metadata(path)
        .await
        .with_context(|| format!("Could not read local file metadata: {}", path.display()))?;
    if !metadata.is_file() {
        anyhow::bail!("Local media source is not a file.");
    }
    if metadata.len() > max_bytes as u64 {
        anyhow::bail!("Local file exceeds the {max_bytes} byte limit.");
    }
    let file = File::open(path)
        .await
        .with_context(|| format!("Could not open local file: {}", path.display()))?;
    let mut reader = file.take(max_bytes as u64 + 1);
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).await?;
    if bytes.len() > max_bytes {
        anyhow::bail!("Local file exceeds the {max_bytes} byte limit.");
    }
    Ok(bytes)
}

pub fn decode_data_url_limited(source: &str, max_bytes: usize, kind: &str) -> Result<Vec<u8>> {
    let encoded_limit = max_bytes
        .saturating_mul(4)
        .saturating_div(3)
        .saturating_add(DATA_URL_HEADER_ALLOWANCE);
    if source.len() > encoded_limit {
        anyhow::bail!("{kind} data URL exceeds the {max_bytes} byte limit.");
    }
    let data_url = data_url::DataUrl::process(source)?;
    let bytes = data_url.decode_to_vec()?.0;
    if bytes.len() > max_bytes {
        anyhow::bail!("{kind} data URL exceeds the {max_bytes} byte limit.");
    }
    Ok(bytes)
}

pub fn data_url_mime(source: &str) -> Option<String> {
    let header = source.strip_prefix("data:")?.split_once(',')?.0;
    let mime = header.split(';').next().unwrap_or_default();
    (!mime.is_empty()).then(|| mime.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_oversized_data_urls_before_decode() {
        let source = format!("data:text/plain;base64,{}", "a".repeat(128));
        assert!(decode_data_url_limited(&source, 4, "test").is_err());
    }

    #[test]
    fn validates_ui_upload_references() {
        let image =
            Url::parse("inference-upload:550e8400-e29b-41d4-a716-446655440000.png").unwrap();
        assert_eq!(
            ui_upload_filename(&image, "image").unwrap(),
            "550e8400-e29b-41d4-a716-446655440000.png"
        );

        let video =
            Url::parse("inference-upload:550e8400-e29b-41d4-a716-446655440000.webm").unwrap();
        assert!(ui_upload_filename(&video, "video").is_ok());
        assert!(ui_upload_filename(&video, "image").is_err());

        for source in [
            "inference-upload:../../etc/passwd.png",
            "inference-upload:%2e%2e%2fetc%2fpasswd.png",
            "inference-upload://host/550e8400-e29b-41d4-a716-446655440000.png",
            "inference-upload:6ba7b810-9dad-11d1-80b4-00c04fd430c8.png",
            "inference-upload:550e8400-e29b-41d4-a716-446655440000.PNG",
            "inference-upload:550e8400-e29b-41d4-a716-446655440000.png?x=1",
        ] {
            let url = Url::parse(source).unwrap();
            assert!(ui_upload_filename(&url, "image").is_err(), "{source}");
        }
    }

    #[tokio::test]
    async fn reads_ui_upload_with_size_limit() {
        let dir = tempfile::tempdir().unwrap();
        let root = tokio::fs::canonicalize(dir.path()).await.unwrap();
        let filename = "550e8400-e29b-41d4-a716-446655440000.png";
        tokio::fs::write(root.join(filename), b"image")
            .await
            .unwrap();
        let url = Url::parse(&format!("inference-upload:{filename}")).unwrap();

        let media = load_ui_upload_from_root(url.clone(), "image", &root, 5)
            .await
            .unwrap();
        assert_eq!(media.bytes, b"image");
        assert!(
            load_ui_upload_from_root(url, "image", &root, 4)
                .await
                .is_err()
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_ui_upload_symlinks() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let root = tokio::fs::canonicalize(dir.path()).await.unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();
        let filename = "550e8400-e29b-41d4-a716-446655440000.png";
        symlink(outside.path(), root.join(filename)).unwrap();

        assert!(resolve_ui_upload_path(&root, filename).await.is_err());
    }
}
