//! The TCP listener the servers accept on, with the connect info [`Peer`] reads.

use axum::extract::connect_info::Connected;
use axum::serve::{IncomingStream, Listener};

use crate::auth::Peer;

// Streamed tokens are small writes; Nagle would hold each one back waiting for an ACK.
pub struct NodelayListener(pub tokio::net::TcpListener);

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

#[cfg(test)]
mod tests {
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
