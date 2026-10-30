//! Where a request's media comes from: attachments passed beside it (`media://<index>`) and `data:` URLs, which the
//! engine reads itself, and anything else (http, files, a web UI's uploads), which a [`MediaFetcher`] the host gives
//! the engine resolves, under the host's own policy.

use std::sync::Arc;

use anyhow::Result;
use futures::future::BoxFuture;
use image::DynamicImage;
use inference_core::{AudioInput, VideoInput};
use url::Url;

pub const MAX_MEDIA_BYTES: usize = 64 * 1024 * 1024;
pub const SERVER_VIDEO_FRAME_LIMIT: usize = 32;

const DATA_URL_HEADER_ALLOWANCE: usize = 4096;

pub const IMAGE_UPLOAD_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "gif", "webp", "bmp", "svg"];
pub const VIDEO_UPLOAD_EXTENSIONS: &[&str] = &["mp4", "avi", "mov", "mkv", "webm", "m4v", "gif"];
pub const AUDIO_UPLOAD_EXTENSIONS: &[&str] =
    &["wav", "mp3", "ogg", "flac", "m4a", "aac", "opus", "webm"];

/// Resolves a media source the engine does not read itself (an http URL, a path, a host's own scheme), up to
/// `max_bytes`; a host gives one to an engine with [`crate::Engine::with_media_fetcher`].
pub trait MediaFetcher: Send + Sync {
    fn fetch<'a>(
        &'a self,
        source: &'a str,
        kind: &'a str,
        max_bytes: usize,
    ) -> BoxFuture<'a, Result<LoadedMedia>>;
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

/// The attachments of one request, indexed by position, and the fetcher for sources that are not attachments.
#[derive(Clone, Default)]
pub struct MediaAttachments {
    media: Arc<Vec<Media>>,
    fetcher: Option<Arc<dyn MediaFetcher>>,
}

impl std::fmt::Debug for MediaAttachments {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MediaAttachments")
            .field("media", &self.media)
            .field("fetcher", &self.fetcher.is_some())
            .finish()
    }
}

impl MediaAttachments {
    pub fn new(attachments: Vec<MediaAttachment>) -> Self {
        Self::from_media(attachments.into_iter().map(Media::Encoded).collect())
    }

    pub fn from_media(media: Vec<Media>) -> Self {
        Self {
            media: Arc::new(media),
            fetcher: None,
        }
    }

    /// These attachments, with `fetcher` for sources they do not hold, unless they already have one.
    pub fn with_fetcher(mut self, fetcher: Option<Arc<dyn MediaFetcher>>) -> Self {
        if self.fetcher.is_none() {
            self.fetcher = fetcher;
        }
        self
    }

    /// The attachment `source` names, `None` when it names none (a URL or path).
    pub(crate) fn attached(&self, source: &str, kind: &str) -> Result<Option<&Media>> {
        let Some(index) = source.strip_prefix(ATTACHMENT_PREFIX) else {
            return Ok(None);
        };
        index
            .parse::<usize>()
            .ok()
            .and_then(|index| self.media.get(index))
            .map(Some)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{kind} source `{source}` names no attachment ({} attached)",
                    self.media.len()
                )
            })
    }

    /// Loads `source`: an attachment, a `data:` URL, or anything else through the fetcher.
    pub async fn load(&self, source: &str, kind: &str) -> Result<LoadedMedia> {
        self.load_limited(source, kind, MAX_MEDIA_BYTES).await
    }

    pub async fn load_limited(
        &self,
        source: &str,
        kind: &str,
        max_bytes: usize,
    ) -> Result<LoadedMedia> {
        match self.attached(source, kind)? {
            Some(Media::Encoded(attachment)) => {
                return Ok(LoadedMedia {
                    bytes: attachment.bytes.clone(),
                    mime_type: attachment.mime_type.clone(),
                    final_url: None,
                });
            }
            Some(decoded) => {
                anyhow::bail!("{kind} source `{source}` is a decoded {}", decoded.kind())
            }
            None => {}
        }
        if source.starts_with("data:") {
            return Ok(LoadedMedia {
                bytes: decode_data_url_limited(source, max_bytes, kind)?,
                mime_type: data_url_mime(source),
                final_url: Url::parse(source).ok(),
            });
        }
        match &self.fetcher {
            Some(fetcher) => fetcher.fetch(source, kind, max_bytes).await,
            None => anyhow::bail!(
                "the engine reads {kind} from attachments (media://<index>) and data: URLs; \
                 fetch it and attach the bytes"
            ),
        }
    }
}

pub struct LoadedMedia {
    pub bytes: Vec<u8>,
    pub mime_type: Option<String>,
    pub final_url: Option<Url>,
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

    #[tokio::test]
    async fn a_url_without_a_fetcher_is_the_callers_to_fetch() {
        let media = MediaAttachments::default();
        let error = media
            .load("https://example.com/a.png", "image")
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains("fetch it"), "{error}");
        let data = media
            .load("data:text/plain;base64,aGk=", "image")
            .await
            .unwrap();
        assert_eq!(data.bytes, b"hi");
    }
}
