//! The process-wide rustls provider the workspace's reqwest clients and `wss://` connections are built on.

use std::sync::Once;

/// Installs ring as rustls's default provider unless one is already installed; call before building a client.
pub fn install_provider() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        // An embedder may have installed its own provider first; keep it.
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

#[cfg(test)]
mod tests {
    #[test]
    fn clients_build_once_the_provider_is_installed() {
        super::install_provider();
        super::install_provider();
        reqwest::Client::builder().build().unwrap();
        reqwest::blocking::Client::builder().build().unwrap();
    }
}
