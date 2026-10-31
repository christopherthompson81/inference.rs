//! HTTP building blocks below the API server and the web UI: error envelopes and JSON extraction, API keys and
//! sessions, the observability middleware, SSE stream outcomes and the route names.

pub mod auth;
pub mod handler_core;
pub mod listener;
pub mod metrics;
pub mod route_registry;
pub mod streaming;
mod ui_uploads;

pub use ui_uploads::{configure_ui_upload_dir, ui_upload_dir};

pub const MB_TO_B: usize = 1024 * 1024;
// NOTE(EricLBuehler): Accept up to 50mb input
const N_INPUT_SIZE: usize = 50;
/// This is the axum default request body limit for the router. Accept up to 50mb input.
pub const DEFAULT_MAX_BODY_LIMIT: usize = N_INPUT_SIZE * MB_TO_B;
