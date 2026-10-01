//! API keys: a keyed server maps each request's key to the owner it acts for; an open one acts for no owner.

use std::{collections::HashSet, path::Path, sync::Arc};

use axum::{
    Json,
    extract::{Request, State},
    http::{HeaderMap, Method, StatusCode, header::AUTHORIZATION},
    middleware::Next,
    response::{IntoResponse, Response},
};
use sha2::{Digest, Sha256};

use crate::{
    handler_core::{ApiError, ApiErrorKind, openai_error_response},
    route_registry::{ANTHROPIC_MESSAGES_ROUTE, HEALTH_ROUTE, ROOT_ROUTE},
};

/// The environment variable holding a single key, which acts for [`ENV_KEY_OWNER`].
pub const API_KEY_ENV: &str = "INFERENCE_RS_API_KEY";
/// The owner a key from [`API_KEY_ENV`] acts for.
pub const ENV_KEY_OWNER: &str = "default";
pub(crate) const API_KEY_HEADER: &str = "x-api-key";
const BEARER_SCHEME: &str = "bearer";
const KEY_SEPARATOR: char = '=';
const COMMENT_PREFIX: char = '#';

/// Who a request acts for: its key's owner on a keyed server, `None` on an open one.
#[derive(Clone, Debug, Default)]
pub struct Owner(pub Option<String>);

impl Owner {
    pub fn as_deref(&self) -> Option<&str> {
        self.0.as_deref()
    }
}

/// The keys a server accepts, each naming the owner it acts for. Keys are held as digests.
#[derive(Clone, Debug, Default)]
pub struct ApiKeys {
    keys: Vec<([u8; 32], String)>,
}

impl ApiKeys {
    /// Parses `name = key` lines; blank lines and `#` comments are skipped. Names and keys must be unique.
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        let mut keys = Self::default();
        for (index, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with(COMMENT_PREFIX) {
                continue;
            }
            let Some((name, key)) = line.split_once(KEY_SEPARATOR) else {
                anyhow::bail!("line {}: expected `name = key`", index + 1);
            };
            keys.add(name.trim(), key.trim())
                .map_err(|error| anyhow::anyhow!("line {}: {error}", index + 1))?;
        }
        Ok(keys)
    }

    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|error| {
            anyhow::anyhow!("reading API keys from {}: {error}", path.display())
        })?;
        Self::parse(&text).map_err(|error| anyhow::anyhow!("{}: {error}", path.display()))
    }

    /// Adds a key acting for `owner`.
    pub fn add(&mut self, owner: &str, key: &str) -> anyhow::Result<()> {
        anyhow::ensure!(!owner.is_empty(), "an owner name is empty");
        anyhow::ensure!(!key.is_empty(), "the key for `{owner}` is empty");
        let digest = digest(key);
        anyhow::ensure!(
            self.keys
                .iter()
                .all(|(other, name)| *other != digest && name != owner),
            "`{owner}` or its key is listed twice"
        );
        self.keys.push((digest, owner.to_string()));
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn owners(&self) -> HashSet<&str> {
        self.keys.iter().map(|(_, owner)| owner.as_str()).collect()
    }

    /// The owner `key` acts for. Compares every digest in full, so timing says nothing about near misses.
    pub fn owner_of(&self, key: &str) -> Option<&str> {
        let presented = digest(key);
        let mut found = None;
        for (digest, owner) in &self.keys {
            let differs = digest
                .iter()
                .zip(presented.iter())
                .fold(0u8, |acc, (a, b)| acc | (a ^ b));
            if differs == 0 {
                found = Some(owner.as_str());
            }
        }
        found
    }
}

fn digest(key: &str) -> [u8; 32] {
    Sha256::digest(key.as_bytes()).into()
}

fn presented_key(headers: &HeaderMap) -> Option<&str> {
    let bearer = headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split_once(' '))
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case(BEARER_SCHEME))
        .map(|(_, key)| key.trim());
    bearer.or_else(|| {
        headers
            .get(API_KEY_HEADER)
            .and_then(|value| value.to_str().ok())
    })
}

// Anthropic clients read their own error envelope; everyone else reads OpenAI's.
fn unauthorized(path: &str) -> Response {
    let error = ApiError::new(
        ApiErrorKind::Unauthorized,
        "Missing or unknown API key. Send it as `Authorization: Bearer <key>` or `x-api-key`.",
        Some("invalid_api_key"),
        None,
    );
    if path.starts_with(ANTHROPIC_MESSAGES_ROUTE.path) {
        let body = inference_api::anthropic::anthropic_error_body(&error);
        return (StatusCode::UNAUTHORIZED, Json(body)).into_response();
    }
    openai_error_response(error)
}

/// Tags each request with its [`Owner`], refusing a keyed server's requests that carry no known key.
pub async fn authenticate(
    State(keys): State<Option<Arc<ApiKeys>>>,
    mut request: Request,
    next: Next,
) -> Response {
    let path = request.uri().path();
    // load balancers probe health without credentials, and it reports nothing about anyone
    let probe =
        request.method() == Method::GET && [HEALTH_ROUTE.path, ROOT_ROUTE.path].contains(&path);
    // a CORS preflight never carries credentials; the CORS layer answers it without reaching a handler
    let preflight = request.method() == Method::OPTIONS;
    let owner = match keys.as_deref() {
        None => Owner(None),
        Some(_) if probe || preflight => Owner(None),
        Some(keys) => match presented_key(request.headers()).and_then(|key| keys.owner_of(key)) {
            Some(owner) => Owner(Some(owner.to_string())),
            None => return unauthorized(path),
        },
    };
    request.extensions_mut().insert(owner);
    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_keys_file_maps_each_key_to_its_owner() {
        let keys = ApiKeys::parse("# team keys\nteam-a = key-alpha\n\nteam-b=key-beta\n").unwrap();
        assert_eq!(keys.owner_of("key-alpha"), Some("team-a"));
        assert_eq!(keys.owner_of("key-beta"), Some("team-b"));
        assert_eq!(keys.owner_of("key-alpha0"), None);
        assert_eq!(keys.owner_of(""), None);
    }

    #[test]
    fn a_keys_file_with_a_repeat_or_a_bad_line_is_refused() {
        for text in [
            "team-a = key-alpha\nteam-a = key-beta",
            "team-a = key-alpha\nteam-b = key-alpha",
            "team-a key-alpha",
            "= key-alpha",
            "team-a =",
        ] {
            assert!(ApiKeys::parse(text).is_err(), "{text:?}");
        }
    }
}
