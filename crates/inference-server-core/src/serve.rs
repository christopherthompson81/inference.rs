//! Running a built router: its listener, the MCP listener beside it, and the startup log.

use anyhow::{Context, Result};
use axum::{
    Router,
    extract::connect_info::Connected,
    serve::{IncomingStream, Listener},
};
use inference_api::Engine;
use tracing::{debug, info};

use crate::{
    auth::Peer,
    lora_adapters::runtime_lora_updates_enabled,
    mcp_server::{MCP_PROTOCOL_VERSION, MCP_ROUTE, create_mcp_router},
    route_registry::{INFERENCE_RS_API_ROUTES, RUNTIME_LORA_API_ROUTES, RouteInfo, RouteKind},
};

/// Where `serve` listens.
pub struct ServeOptions<'a> {
    pub host: &'a str,
    pub port: u16,
    /// Serves MCP on its own port too; it must differ from `port`.
    pub mcp_port: Option<u16>,
    /// The keys MCP requires, as the HTTP router does; `None` for an open server.
    pub auth: Option<std::sync::Arc<crate::auth::Auth>>,
}

/// Serves `app` until the listener fails, with an MCP server on `mcp_port` when one is given.
pub async fn serve(app: Router, engine: &Engine, options: ServeOptions<'_>) -> Result<()> {
    let ServeOptions {
        host,
        port,
        mcp_port,
        auth,
    } = options;
    if let Some(mcp_port) = mcp_port {
        spawn_mcp_server(engine, auth, host, mcp_port, port).await?;
    }
    let listener = tokio::net::TcpListener::bind(format!("{host}:{port}"))
        .await
        .with_context(|| format!("Failed to bind the HTTP server to {host}:{port}"))?;
    info!("Server listening on http://{host}:{port}");
    log_api_surfaces(host, port);
    // the peer address tells an open server's admin requests (from this machine) from the rest
    let app = app.into_make_service_with_connect_info::<Peer>();
    axum::serve(NodelayListener(listener), app).await?;
    Ok(())
}

async fn spawn_mcp_server(
    engine: &Engine,
    auth: Option<std::sync::Arc<crate::auth::Auth>>,
    host: &str,
    mcp_port: u16,
    http_port: u16,
) -> Result<()> {
    if mcp_port == http_port {
        anyhow::bail!("the MCP port must differ from the HTTP port ({http_port})");
    }
    let listener = tokio::net::TcpListener::bind(format!("{host}:{mcp_port}"))
        .await
        .with_context(|| format!("Failed to bind MCP server to {host}:{mcp_port}"))?;
    let router = create_mcp_router(engine, auth);
    info!("MCP server listening on http://{host}:{mcp_port}{MCP_ROUTE}");
    info!("MCP protocol version is {MCP_PROTOCOL_VERSION}");
    tokio::spawn(async move {
        if let Err(e) = axum::serve(NodelayListener(listener), router).await {
            tracing::error!("MCP server error: {e}");
        }
    });
    Ok(())
}

// Streamed tokens are small writes; Nagle would hold each one back waiting for an ACK.
struct NodelayListener(tokio::net::TcpListener);

impl Listener for NodelayListener {
    type Io = tokio::net::TcpStream;
    type Addr = std::net::SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        let (stream, addr) = Listener::accept(&mut self.0).await;
        if let Err(error) = stream.set_nodelay(true) {
            tracing::warn!("failed to set TCP_NODELAY on incoming connection: {error}");
        }
        (stream, addr)
    }

    fn local_addr(&self) -> std::io::Result<Self::Addr> {
        Listener::local_addr(&self.0)
    }
}

impl Connected<IncomingStream<'_, NodelayListener>> for Peer {
    fn connect_info(stream: IncomingStream<'_, NodelayListener>) -> Self {
        Self {
            remote: *stream.remote_addr(),
            local: stream.io().local_addr().ok(),
        }
    }
}

fn log_api_surfaces(host: &str, port: u16) {
    let client_host = match host {
        "0.0.0.0" => "localhost",
        "::" => "[::1]",
        host => host,
    };
    let root = format!("http://{client_host}:{port}");

    info!("OpenAI-compatible API: {root}/v1");
    info!("Anthropic-compatible API: {root}");
    info!("Swagger UI docs: {root}/docs");

    debug!("Available OpenAI-compatible routes:");
    log_routes(INFERENCE_RS_API_ROUTES, RouteKind::OpenAi);
    debug!("Available Anthropic-compatible routes:");
    log_routes(INFERENCE_RS_API_ROUTES, RouteKind::Anthropic);
    debug!("Available additional inference.rs routes:");
    log_routes(INFERENCE_RS_API_ROUTES, RouteKind::InferenceRs);
    if runtime_lora_updates_enabled() {
        log_routes(RUNTIME_LORA_API_ROUTES, RouteKind::InferenceRs);
    }
}

fn log_routes(routes: &[RouteInfo], kind: RouteKind) {
    for route in routes.iter().filter(|route| route.kind == kind) {
        debug!("  Route: {}, Methods: {}", route.path, route.methods);
    }
}

#[cfg(test)]
mod tests {
    use axum::serve::Listener;

    use super::*;

    #[tokio::test]
    async fn accepted_connections_enable_tcp_nodelay() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let mut listener = NodelayListener(listener);
        let connect = tokio::net::TcpStream::connect(address);

        let ((stream, _), client) = tokio::join!(listener.accept(), connect);

        client.unwrap();
        assert!(stream.nodelay().unwrap());
    }
}
