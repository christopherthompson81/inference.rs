//! Where the web UI stores uploads, which a request may name by the upload scheme.

use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
};

use anyhow::{Context, Result};

static UI_UPLOAD_DIR: OnceLock<PathBuf> = OnceLock::new();

pub fn ui_upload_dir() -> Option<&'static Path> {
    UI_UPLOAD_DIR.get().map(PathBuf::as_path)
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
