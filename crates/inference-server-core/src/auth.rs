//! API keys: a keyed server maps each request's key, or a browser's signed-in session, to the owner it acts for; an
//! open one acts for no owner.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use axum::{
    Extension, Json, Router,
    extract::{Request, State},
    http::{
        HeaderMap, HeaderValue, Method, StatusCode,
        header::{AUTHORIZATION, COOKIE, SET_COOKIE},
    },
    middleware::{self, Next},
    response::{IntoResponse, Response},
};
use sha2::{Digest, Sha256};

use crate::{
    handler_core::{ApiError, ApiErrorKind, openai_error_response},
    route_registry::{ANTHROPIC_MESSAGES_ROUTE, AUTH_SESSION_ROUTE, HEALTH_ROUTE, ROOT_ROUTE},
};

/// The environment variable holding a single key, which acts for [`ENV_KEY_OWNER`].
pub const API_KEY_ENV: &str = "INFERENCE_RS_API_KEY";
/// The owner a key from [`API_KEY_ENV`] acts for.
pub const ENV_KEY_OWNER: &str = "default";
pub(crate) const API_KEY_HEADER: &str = "x-api-key";
const BEARER_SCHEME: &str = "bearer";
const KEY_SEPARATOR: char = '=';
const COMMENT_PREFIX: char = '#';
/// The cookie a signed-in browser carries; it names a server-side session, never the key.
pub const SESSION_COOKIE: &str = "inference_session";
const SESSION_TTL: Duration = Duration::from_secs(12 * 60 * 60);
/// An owner's newest sessions kept; signing in again past this drops the oldest.
const MAX_SESSIONS_PER_OWNER: usize = 32;
const FETCH_SITE_HEADER: &str = "sec-fetch-site";
const SAME_ORIGIN: &str = "same-origin";
const FORWARDED_PROTO_HEADER: &str = "x-forwarded-proto";
/// Which requests pass without a key: `(method, path as the guarded router sees it)`.
pub type PublicRequests = fn(&Method, &str) -> bool;

/// How a router is guarded: which requests pass keyless, and whether a signed-in browser's cookie counts.
#[derive(Clone, Copy)]
pub struct Guard {
    pub public: PublicRequests,
    pub cookies: bool,
}

/// The API's guard: health probes and signing in pass, and browsers use their cookie.
pub const API_GUARD: Guard = Guard {
    public: api_public_requests,
    cookies: true,
};

/// For a router no browser uses, such as MCP: a key on every request.
pub const KEY_ONLY_GUARD: Guard = Guard {
    public: |_, _| false,
    cookies: false,
};

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

/// The keys a keyed server accepts and the browser sessions signed in with them.
pub struct Auth {
    keys: ApiKeys,
    sessions: Mutex<HashMap<String, (String, Instant)>>,
}

impl std::fmt::Debug for Auth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Auth")
            .field("owners", &self.keys.owners())
            .finish_non_exhaustive()
    }
}

impl Auth {
    /// `None` for no keys: the server is open.
    pub fn new(keys: ApiKeys) -> Option<Arc<Self>> {
        (!keys.is_empty()).then(|| {
            Arc::new(Self {
                keys,
                sessions: Mutex::new(HashMap::new()),
            })
        })
    }

    /// A session token acting for `key`'s owner, or `None` for an unknown key.
    pub fn sign_in(&self, key: &str) -> Option<String> {
        let owner = self.keys.owner_of(key)?.to_string();
        let token = format!(
            "{}{}",
            uuid::Uuid::new_v4().simple(),
            uuid::Uuid::new_v4().simple()
        );
        let now = Instant::now();
        let mut sessions = self.sessions.lock().unwrap();
        sessions.retain(|_, (_, expires)| *expires > now);
        let mut theirs: Vec<(String, Instant)> = sessions
            .iter()
            .filter(|(_, (other, _))| *other == owner)
            .map(|(token, (_, expires))| (token.clone(), *expires))
            .collect();
        if theirs.len() >= MAX_SESSIONS_PER_OWNER {
            theirs.sort_by_key(|(_, expires)| *expires);
            for (stale, _) in &theirs[..=theirs.len() - MAX_SESSIONS_PER_OWNER] {
                sessions.remove(stale);
            }
        }
        sessions.insert(token.clone(), (owner, now + SESSION_TTL));
        Some(token)
    }

    pub fn sign_out(&self, token: &str) {
        self.sessions.lock().unwrap().remove(token);
    }

    fn session_owner(&self, token: &str) -> Option<String> {
        let sessions = self.sessions.lock().unwrap();
        sessions
            .get(token)
            .filter(|(_, expires)| *expires > Instant::now())
            .map(|(owner, _)| owner.clone())
    }

    fn owner(&self, method: &Method, headers: &HeaderMap, cookies: bool) -> Option<String> {
        if let Some(owner) = presented_key(headers).and_then(|key| self.keys.owner_of(key)) {
            return Some(owner.to_string());
        }
        // SameSite stops other sites, not other ports or subdomains of this one: a change must come from this origin
        let reads = matches!(*method, Method::GET | Method::HEAD);
        if !cookies || !(reads || same_origin(headers)) {
            return None;
        }
        session_token(headers).and_then(|token| self.session_owner(token))
    }
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

// Two session cookies means one was planted beside ours (say with a narrower Path), so neither is trusted.
fn session_token(headers: &HeaderMap) -> Option<&str> {
    let mut tokens = headers
        .get_all(COOKIE)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(';'))
        .filter_map(|pair| pair.trim().split_once('='))
        .filter(|(name, _)| *name == SESSION_COOKIE)
        .map(|(_, token)| token);
    let token = tokens.next()?;
    tokens.next().is_none().then_some(token)
}

// A browser names where a request came from; a client that sends neither header is not a browser being steered.
fn same_origin(headers: &HeaderMap) -> bool {
    let header = |name| headers.get(name).and_then(|value| value.to_str().ok());
    if let Some(site) = header(FETCH_SITE_HEADER) {
        return site == SAME_ORIGIN;
    }
    match (
        header(axum::http::header::ORIGIN.as_str()),
        header(axum::http::header::HOST.as_str()),
    ) {
        (Some(origin), Some(host)) => origin
            .split_once("://")
            .is_some_and(|(_, rest)| rest == host),
        (Some(_), None) => false,
        (None, _) => true,
    }
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

/// The API's keyless requests: health probes, which report nothing about anyone, and signing in.
pub fn api_public_requests(method: &Method, path: &str) -> bool {
    (method == Method::GET && [HEALTH_ROUTE.path, ROOT_ROUTE.path].contains(&path))
        || ([Method::POST, Method::DELETE].contains(method) && path == AUTH_SESSION_ROUTE.path)
}

/// Tags every request `router` serves with its [`Owner`]; with `auth`, refuses those that carry no known key or
/// session, other than `public` ones. Outermost, so a refused request's body is never read.
pub fn require(router: Router, auth: Option<Arc<Auth>>, guard: Guard) -> Router {
    router.layer(middleware::from_fn_with_state((auth, guard), authenticate))
}

async fn authenticate(
    State((auth, guard)): State<(Option<Arc<Auth>>, Guard)>,
    mut request: Request,
    next: Next,
) -> Response {
    // a CORS preflight never carries credentials; the CORS layer answers it without reaching a handler
    let passes = request.method() == Method::OPTIONS
        || (guard.public)(request.method(), request.uri().path());
    let owner = match auth.as_deref() {
        None => Owner(None),
        Some(auth) => match auth.owner(request.method(), request.headers(), guard.cookies) {
            Some(owner) => Owner(Some(owner)),
            None if passes => Owner(None),
            None => return unauthorized(request.uri().path()),
        },
    };
    request.extensions_mut().insert(owner);
    next.run(request).await
}

/// `{"key"}` to sign a browser in.
#[derive(serde::Deserialize, utoipa::ToSchema)]
pub struct SignInRequest {
    pub key: String,
}

impl inference_api::request_body::JsonRequest for SignInRequest {
    fn from_json(body: &[u8]) -> Result<Self, ApiError> {
        inference_api::request_body::parse_json(body)
    }
}

#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/auth/session",
    request_body = SignInRequest,
    responses(
        (status = 204, description = "Signed in; the response sets the session cookie"),
        (status = 401, description = "Unknown key"),
    )
))]
/// POST `/auth/session`: exchanges a key for an `HttpOnly`, `SameSite=Strict` session cookie. An open server needs none.
pub async fn sign_in(
    auth: Option<Extension<Arc<Auth>>>,
    headers: HeaderMap,
    payload: Result<
        crate::handler_core::ApiJson<SignInRequest>,
        crate::handler_core::ApiJsonRejection,
    >,
) -> Response {
    let request = match payload {
        Ok(crate::handler_core::ApiJson(request)) => request,
        Err(crate::handler_core::ApiJsonRejection(error)) => return openai_error_response(error),
    };
    let Some(Extension(auth)) = auth else {
        return StatusCode::NO_CONTENT.into_response();
    };
    let Some(token) = auth.sign_in(request.key.trim()) else {
        return unauthorized(AUTH_SESSION_ROUTE.path);
    };
    // behind a TLS proxy the browser must never send the token over plain HTTP
    let secure = headers
        .get(FORWARDED_PROTO_HEADER)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|proto| proto.eq_ignore_ascii_case("https"));
    let cookie = format!(
        "{SESSION_COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={}{}",
        SESSION_TTL.as_secs(),
        if secure { "; Secure" } else { "" }
    );
    session_cookie_response(&cookie)
}

#[cfg_attr(test, utoipa::path(
    delete,
    tag = "inference.rs",
    path = "/auth/session",
    responses((status = 204, description = "Signed out; the response clears the session cookie"))
))]
/// DELETE `/auth/session`: signs the browser out.
pub async fn sign_out(auth: Option<Extension<Arc<Auth>>>, headers: HeaderMap) -> Response {
    if let (Some(Extension(auth)), Some(token)) = (auth, session_token(&headers)) {
        auth.sign_out(token);
    }
    session_cookie_response(&format!(
        "{SESSION_COOKIE}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0"
    ))
}

fn session_cookie_response(cookie: &str) -> Response {
    let mut response = StatusCode::NO_CONTENT.into_response();
    if let Ok(value) = HeaderValue::from_str(cookie) {
        response.headers_mut().insert(SET_COOKIE, value);
    }
    response
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
    fn an_owner_keeps_only_its_newest_sessions() {
        let auth =
            Auth::new(ApiKeys::parse("team-a = key-alpha\nteam-b = key-beta").unwrap()).unwrap();
        let other = auth.sign_in("key-beta").unwrap();
        let tokens: Vec<String> = (0..=MAX_SESSIONS_PER_OWNER)
            .map(|_| auth.sign_in("key-alpha").unwrap())
            .collect();
        assert_eq!(auth.session_owner(&tokens[0]), None);
        assert_eq!(
            auth.session_owner(tokens.last().unwrap()).as_deref(),
            Some("team-a")
        );
        assert_eq!(auth.session_owner(&other).as_deref(), Some("team-b"));
        assert!(auth.sign_in("key-gamma").is_none());
        assert!(
            !format!("{auth:?}").contains(&other),
            "Debug shows no tokens"
        );
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
