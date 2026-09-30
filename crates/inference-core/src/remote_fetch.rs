use std::{
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    time::Duration,
};

use anyhow::{Context, Result};
use futures::future::BoxFuture;
use reqwest::{Url, header, redirect::Policy};

const URL_FETCH_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NetworkPolicy {
    PublicOnly,
    Any,
}

/// Fetches a URL the caller chose, honouring proxy settings; request-driven fetches go through `fetch_limited`.
pub fn fetch_url<'a>(url: &'a str) -> BoxFuture<'a, Result<Vec<u8>>> {
    Box::pin(fetch_url_inner(url))
}

async fn fetch_url_inner(url: &str) -> Result<Vec<u8>> {
    let response = http_client_builder()
        .connect_timeout(URL_FETCH_CONNECT_TIMEOUT)
        .build()?
        .get(url)
        .send()
        .await
        .with_context(|| format!("Failed to fetch {url}"))?
        .error_for_status()?;
    Ok(response.bytes().await?.to_vec())
}

/// A client builder with the TLS provider installed, which every reqwest client needs first.
pub fn http_client_builder() -> reqwest::ClientBuilder {
    inference_mcp::tls::install_provider();
    reqwest::Client::builder()
}

/// The blocking counterpart of [`http_client_builder`].
pub fn blocking_http_client_builder() -> reqwest::blocking::ClientBuilder {
    inference_mcp::tls::install_provider();
    reqwest::blocking::Client::builder()
}

pub struct FetchOptions<'a> {
    pub max_bytes: usize,
    pub timeout: Duration,
    pub max_redirects: usize,
    pub user_agent: Option<&'a str>,
}

pub struct FetchedResponse {
    pub bytes: Vec<u8>,
    pub mime_type: Option<String>,
    pub final_url: Url,
}

/// Fetch an http(s) URL with a streamed byte cap, re-validating every redirect hop under `network`.
pub fn fetch_limited<'a>(
    url: Url,
    options: FetchOptions<'a>,
    network: NetworkPolicy,
    kind: &'a str,
) -> BoxFuture<'a, Result<FetchedResponse>> {
    Box::pin(fetch_limited_inner(url, options, network, kind))
}

async fn fetch_limited_inner(
    mut url: Url,
    options: FetchOptions<'_>,
    network: NetworkPolicy,
    kind: &str,
) -> Result<FetchedResponse> {
    let max_bytes = options.max_bytes;
    for redirect_idx in 0..=options.max_redirects {
        let mut client = http_client_builder()
            .timeout(options.timeout)
            .redirect(Policy::none())
            .no_proxy();
        if let Some(user_agent) = options.user_agent {
            client = client.user_agent(user_agent);
        }
        match network {
            NetworkPolicy::PublicOnly => {
                let addrs = validate_remote_url(&url).await?;
                client = client
                    .resolve_to_addrs(url.host_str().expect("validated URL has host"), &addrs);
            }
            NetworkPolicy::Any if url.scheme() != "http" && url.scheme() != "https" => {
                anyhow::bail!("Remote URLs must use http or https.");
            }
            NetworkPolicy::Any => {}
        }
        let client = client.build()?;
        let response = client
            .get(url.clone())
            .send()
            .await
            .with_context(|| format!("Failed to fetch {kind}: {url}"))?;

        if response.status().is_redirection() {
            if redirect_idx == options.max_redirects {
                anyhow::bail!("{kind} URL exceeded the redirect limit.");
            }
            url = redirect_target(&url, &response, kind)?;
            continue;
        }

        response
            .error_for_status_ref()
            .map_err(|err| anyhow::anyhow!("Failed to fetch {kind}: {url}: {err}"))?;
        if response
            .content_length()
            .is_some_and(|len| len > max_bytes as u64)
        {
            anyhow::bail!("{kind} response exceeds the {max_bytes} byte limit.");
        }
        let mime_type = response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(|value| value.split(';').next().unwrap_or(value).trim().to_string())
            .filter(|value| !value.is_empty());
        let bytes = read_response_limited(response, max_bytes, kind).await?;
        return Ok(FetchedResponse {
            bytes,
            mime_type,
            final_url: url,
        });
    }

    unreachable!("redirect loop returns or bails")
}

fn redirect_target(current: &Url, response: &reqwest::Response, kind: &str) -> Result<Url> {
    let Some(location) = response.headers().get(header::LOCATION) else {
        anyhow::bail!("{kind} redirect response is missing Location.");
    };
    let location = location
        .to_str()
        .context("redirect Location header is not valid UTF-8")?;
    let target = current
        .join(location)
        .with_context(|| format!("Invalid {kind} redirect target."))?;
    if target.scheme() != "http" && target.scheme() != "https" {
        anyhow::bail!("{kind} redirects may only target http or https URLs.");
    }
    Ok(target)
}

async fn read_response_limited(
    mut response: reqwest::Response,
    max_bytes: usize,
    kind: &str,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if bytes.len().saturating_add(chunk.len()) > max_bytes {
            anyhow::bail!("{kind} response exceeds the {max_bytes} byte limit.");
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

async fn validate_remote_url(url: &Url) -> Result<Vec<SocketAddr>> {
    if url.scheme() != "http" && url.scheme() != "https" {
        anyhow::bail!("Remote URLs must use http or https.");
    }
    let Some(host) = url.host_str() else {
        anyhow::bail!("Remote URL must include a host.");
    };
    reject_private_hostname(host)?;
    let port = url
        .port_or_known_default()
        .ok_or_else(|| anyhow::anyhow!("Remote URL must include a valid port."))?;
    let addrs = tokio::net::lookup_host((host, port))
        .await
        .with_context(|| format!("Failed to resolve remote host `{host}`"))?
        .collect::<Vec<_>>();
    if addrs.is_empty() {
        anyhow::bail!("Remote host `{host}` did not resolve to any addresses.");
    }
    for addr in &addrs {
        reject_private_ip(addr.ip())?;
    }
    Ok(addrs)
}

fn reject_private_hostname(host: &str) -> Result<()> {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    if host == "localhost" || host.ends_with(".localhost") || host.ends_with(".local") {
        anyhow::bail!("Remote URLs must not target local hosts.");
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        reject_private_ip(ip)?;
    }
    Ok(())
}

fn reject_private_ip(ip: IpAddr) -> Result<()> {
    if !is_global_ip(ip) {
        anyhow::bail!("Remote URLs must not target private or local IP addresses.");
    }
    Ok(())
}

fn is_global_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => is_global_ipv4(ip),
        IpAddr::V6(ip) => match embedded_ipv4(ip) {
            Some(ip) => is_global_ipv4(ip),
            None => is_global_ipv6(ip),
        },
    }
}

// mapped, IPv4-compatible, NAT64 and 6to4 addresses reach the embedded IPv4, so judge them by it
fn embedded_ipv4(ip: Ipv6Addr) -> Option<Ipv4Addr> {
    if let Some(ip) = ip.to_ipv4_mapped() {
        return Some(ip);
    }
    let join = |hi: u16, lo: u16| Ipv4Addr::from((u32::from(hi) << 16) | u32::from(lo));
    match ip.segments() {
        [0x2002, hi, lo, ..] => Some(join(hi, lo)),
        [0x0064, 0xff9b, 0, 0, 0, 0, hi, lo] | [0, 0, 0, 0, 0, 0, hi, lo] => Some(join(hi, lo)),
        _ => None,
    }
}

fn is_global_ipv4(ip: Ipv4Addr) -> bool {
    let octets = ip.octets();
    !(ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.is_unspecified()
        || ip.is_multicast()
        || octets[0] == 0
        || matches!(octets, [100, 64..=127, _, _])
        || matches!(octets, [192, 0, 0, _])
        || matches!(octets, [192, 0, 2, _])
        || matches!(octets, [198, 18 | 19, _, _])
        || matches!(octets, [198, 51, 100, _])
        || matches!(octets, [203, 0, 113, _])
        || octets[0] >= 240)
}

fn is_global_ipv6(ip: Ipv6Addr) -> bool {
    let segments = ip.segments();
    !(ip.is_loopback()
        || ip.is_unspecified()
        || ip.is_unique_local()
        || ip.is_unicast_link_local()
        || ip.is_multicast()
        || segments[0] == 0x2001 && segments[1] == 0x0db8
        // local-use NAT64 (RFC 8215) can embed the IPv4 at several offsets, so don't try to decode it
        || segments[..3] == [0x0064, 0xff9b, 0x0001])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn rejects_private_remote_hosts() {
        for source in [
            "http://127.0.0.1/image.png",
            "http://169.254.169.254/latest/meta-data",
            "http://[::1]/image.png",
            "http://localhost/image.png",
            "http://service.local/image.png",
        ] {
            let url = Url::parse(source).unwrap();
            assert!(validate_remote_url(&url).await.is_err(), "{source}");
        }
    }

    #[tokio::test]
    async fn fetch_url_downloads_the_response_body() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        const BODY: &[u8] = b"not really a png";
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/clip.png", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0u8; 1024];
            let _ = stream.read(&mut request).await.unwrap();
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                BODY.len()
            );
            stream.write_all(head.as_bytes()).await.unwrap();
            stream.write_all(BODY).await.unwrap();
        });
        assert_eq!(fetch_url(&url).await.unwrap(), BODY);
        server.await.unwrap();
    }

    #[test]
    fn blocking_clients_build_with_the_installed_provider() {
        blocking_http_client_builder().build().unwrap();
    }

    #[test]
    fn rejects_ipv6_transition_addresses_for_internal_ipv4() {
        for source in [
            "64:ff9b::7f00:1",
            "64:ff9b::a9fe:a9fe",
            "64:ff9b:1::a9fe:a9fe",
            "64:ff9b:1:a9fe:a9:fe00::",
            "2002:7f00:1::",
            "2002:a9fe:a9fe::",
            "::7f00:1",
            "::",
            "::1",
        ] {
            let ip: IpAddr = source.parse().unwrap();
            assert!(reject_private_ip(ip).is_err(), "{source}");
        }
        for source in ["2002:808:808::", "64:ff9b::101:101", "::ffff:8.8.8.8"] {
            let ip: IpAddr = source.parse().unwrap();
            assert!(is_global_ip(ip), "{source}");
        }
    }
}
