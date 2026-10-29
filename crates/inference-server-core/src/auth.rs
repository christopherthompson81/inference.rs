//! API keys: a keyed server maps each request's key, or a browser's signed-in session, to the owner it acts for; an
//! open one acts for no owner.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use axum::{
    Extension, Json, RequestExt, Router,
    extract::{ConnectInfo, Request, State},
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
const WEBSOCKET_KEY_PREFIX: &str = "openai-insecure-api-key.";
const WEBSOCKET: &str = "websocket";
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
const LOCALHOST: &str = "localhost";
// headers a proxy adds for the client it passes a request on for
const FORWARDING_HEADERS: &[&str] = &[
    "forwarded",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-real-ip",
    "via",
];
// how long the machine's name resolves the same
const HOSTNAME_TTL: Duration = Duration::from_secs(30);
#[cfg(unix)]
const HOSTNAME_MAX: usize = 256;
/// The error code of a request [`require_admin`] refuses.
pub const ADMIN_REQUIRED_CODE: &str = "admin_required";
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

/// Whether a request may change what the server serves (load, unload, download and delete models, tune, re-quantize):
/// a keyed server's admin owners, or on an open server a client on this machine.
#[derive(Clone, Copy, Debug, Default)]
pub struct Admin(pub bool);

/// The keys a server accepts, each naming the owner it acts for. Keys are held as digests.
#[derive(Clone, Debug, Default)]
pub struct ApiKeys {
    keys: Vec<([u8; 32], String)>,
    admins: HashSet<String>,
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

    /// Makes `owners` the admins; each must own a key.
    pub fn set_admins(&mut self, owners: impl IntoIterator<Item = String>) -> anyhow::Result<()> {
        let owners: HashSet<String> = owners.into_iter().collect();
        let known = self.owners();
        if let Some(unknown) = owners.iter().find(|owner| !known.contains(owner.as_str())) {
            anyhow::bail!("admin `{unknown}` owns no key");
        }
        self.admins = owners;
        Ok(())
    }

    pub fn is_admin(&self, owner: &str) -> bool {
        self.admins.contains(owner)
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
        // SameSite stops other sites, not other ports or subdomains of this one: a change must come from this origin,
        // and so must a WebSocket, a GET whose replies CORS does not keep from the page that opened it
        let reads = matches!(*method, Method::GET | Method::HEAD) && !is_upgrade(headers);
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
    bearer
        .or_else(|| {
            headers
                .get(API_KEY_HEADER)
                .and_then(|value| value.to_str().ok())
        })
        .or_else(|| subprotocol_key(headers))
}

fn is_upgrade(headers: &HeaderMap) -> bool {
    headers
        .get(axum::http::header::UPGRADE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.eq_ignore_ascii_case(WEBSOCKET))
}

// a browser cannot set headers on a WebSocket, so Realtime clients offer the key as a subprotocol
fn subprotocol_key(headers: &HeaderMap) -> Option<&str> {
    if !is_upgrade(headers) {
        return None;
    }
    headers
        .get(axum::http::header::SEC_WEBSOCKET_PROTOCOL)?
        .to_str()
        .ok()?
        .split(',')
        .find_map(|protocol| protocol.trim().strip_prefix(WEBSOCKET_KEY_PREFIX))
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

/// A connection's two ends, which tell a client on the server's own machine from the rest. A keyless server's
/// local admin needs it as the router's connect info (`into_make_service_with_connect_info::<Peer>()`).
#[derive(Debug, Clone, Copy)]
pub struct Peer {
    pub remote: std::net::SocketAddr,
    /// The address the client reached, when the listener knows it; without it only loopback is this machine, by
    /// a loopback name.
    pub local: Option<std::net::SocketAddr>,
}

impl From<std::net::SocketAddr> for Peer {
    fn from(remote: std::net::SocketAddr) -> Self {
        Self {
            remote,
            local: None,
        }
    }
}

impl Peer {
    // a client on this machine: over loopback, or from the very address it reached, which only the machine itself
    // can complete a handshake from
    fn is_this_machine(&self) -> bool {
        let remote = self.remote.ip().to_canonical();
        remote.is_loopback() || self.local_ip() == Some(remote)
    }

    fn local_ip(&self) -> Option<std::net::IpAddr> {
        self.local.map(|local| local.ip().to_canonical())
    }
}

// a request a proxy passed on: its client may be anywhere, however local the proxy's own connection looks
fn forwarded(headers: &HeaderMap) -> bool {
    FORWARDING_HEADERS
        .iter()
        .any(|name| headers.contains_key(*name))
}

// This machine's name, which a browser on it may name the server by.
fn machine_hostname() -> Option<&'static str> {
    static NAME: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    NAME.get_or_init(|| {
        #[cfg(unix)]
        {
            let mut buffer = [0u8; HOSTNAME_MAX];
            // SAFETY: the buffer outlives the call, which writes at most its length
            let written = unsafe { libc::gethostname(buffer.as_mut_ptr().cast(), buffer.len()) };
            let end = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
            (written == 0 && end > 0).then(|| String::from_utf8_lossy(&buffer[..end]).into_owned())
        }
        #[cfg(not(unix))]
        {
            std::env::var("COMPUTERNAME").ok()
        }
    })
    .as_deref()
}

// What the machine's name resolves to here, kept briefly: a browser naming the server by it gets one of these.
async fn machine_hostname_addresses(machine: &str) -> Vec<std::net::IpAddr> {
    static RESOLVED: Mutex<Option<(Instant, Vec<std::net::IpAddr>)>> = Mutex::new(None);
    if let Some((at, addresses)) = RESOLVED.lock().unwrap().as_ref()
        && at.elapsed() < HOSTNAME_TTL
    {
        return addresses.clone();
    }
    let addresses: Vec<_> = tokio::net::lookup_host((machine, 0))
        .await
        .map(|found| found.map(|address| address.ip().to_canonical()).collect())
        .unwrap_or_default();
    *RESOLVED.lock().unwrap() = Some((Instant::now(), addresses.clone()));
    addresses
}

// The `Host` a request names is this machine: a loopback name, or the address the client reached, by its literal or
// by the machine's own single-label name resolving to it. A site's domain rebound here is none of these, and a proxy
// on this machine reaches a different address than the name its clients use. A client sending no `Host` is not a
// browser.
async fn this_machine_host(headers: &HeaderMap, peer: &Peer) -> bool {
    let Some(host) = headers
        .get(axum::http::header::HOST)
        .and_then(|value| value.to_str().ok())
    else {
        return true;
    };
    let name = match host.strip_prefix('[') {
        Some(bracketed) => bracketed.split(']').next().unwrap_or_default(),
        None => host.split(':').next().unwrap_or_default(),
    };
    if name.eq_ignore_ascii_case(LOCALHOST) {
        return true;
    }
    if let Ok(ip) = name.parse::<std::net::IpAddr>() {
        let ip = ip.to_canonical();
        return ip.is_loopback() || peer.local_ip() == Some(ip);
    }
    // a dotted name (an FQDN, `.local`) may be answered by the network, so it could be rebound
    let Some(machine) = machine_hostname().filter(|machine| !machine.contains('.')) else {
        return false;
    };
    let Some(local) = peer.local_ip() else {
        return false;
    };
    name.eq_ignore_ascii_case(machine) && machine_hostname_addresses(machine).await.contains(&local)
}

// Anthropic clients read their own error envelope; everyone else reads OpenAI's.
fn unauthorized(path: &str) -> Response {
    let error = ApiError::new(
        ApiErrorKind::Unauthorized,
        "Missing or unknown API key. Send it as `Authorization: Bearer <key>` or `x-api-key` (a WebSocket may offer it as the `openai-insecure-api-key.<key>` subprotocol).",
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
    let admin = match (auth.as_deref(), owner.as_deref()) {
        (Some(auth), Some(owner)) => auth.keys.is_admin(owner),
        (Some(_), None) => false,
        // no peer address (a router served without connect info) is never taken for this machine
        (None, _) => {
            let peer = request
                .extract_parts::<ConnectInfo<Peer>>()
                .await
                .ok()
                .map(|ConnectInfo(peer)| peer);
            // any page the server's user opens is a local client too: only the server's own pages count, at a name
            // for this machine, so neither another site nor a name rebound to this machine acts as admin
            match peer {
                Some(peer)
                    if peer.is_this_machine()
                        && !forwarded(request.headers())
                        && same_origin(request.headers()) =>
                {
                    this_machine_host(request.headers(), &peer).await
                }
                _ => false,
            }
        }
    };
    request.extensions_mut().insert(owner);
    request.extensions_mut().insert(Admin(admin));
    next.run(request).await
}

/// Refuses a request that is not an [`Admin`]'s; layered on the routes that change what the server serves.
pub async fn require_admin(request: Request, next: Next) -> Response {
    if request
        .extensions()
        .get::<Admin>()
        .is_some_and(|admin| admin.0)
    {
        return next.run(request).await;
    }
    openai_error_response(ApiError::new(
        ApiErrorKind::Forbidden,
        "Changing the served models needs an admin: a key the server lists with `--admin`, or on a server without \
         keys a client on the server's own machine.",
        Some(ADMIN_REQUIRED_CODE),
        None,
    ))
}

/// `{"key"}` to sign a browser in.
#[derive(inference_json::FromJson, utoipa::ToSchema)]
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

    #[tokio::test]
    async fn a_browser_on_this_machine_may_name_it_by_its_hostname() {
        let Some(machine) = machine_hostname().filter(|machine| !machine.contains('.')) else {
            return;
        };
        let Some(resolved) = machine_hostname_addresses(machine).await.first().copied() else {
            return;
        };
        let at = |local: std::net::IpAddr| Peer {
            remote: (local, 40000).into(),
            local: Some((local, 1234).into()),
        };
        let host = async |name: String, peer: Peer| {
            let mut headers = HeaderMap::new();
            headers.insert(axum::http::header::HOST, name.parse().unwrap());
            this_machine_host(&headers, &peer).await
        };
        assert!(host(format!("{machine}:1234"), at(resolved)).await);
        assert!(host(format!("{}:1234", machine.to_uppercase()), at(resolved)).await);
        // a proxy on this machine reaches loopback, not what the name its clients use resolves to
        let proxy = std::net::Ipv4Addr::new(127, 0, 0, 3).into();
        if resolved != proxy {
            assert!(!host(format!("{machine}:1234"), at(proxy)).await);
        }
        assert!(!host(format!("{machine}.local:1234"), at(resolved)).await);
        assert!(!host(format!("{machine}.attacker.example:1234"), at(resolved)).await);
        assert!(host("localhost:1234".into(), at(proxy)).await);
        assert!(!host("192.0.2.1:1234".into(), at(proxy)).await);
    }

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
