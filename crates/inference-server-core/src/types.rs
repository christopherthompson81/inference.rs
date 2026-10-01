//! ## Custom types used in inference.rs server core.

use axum::{Extension, extract::FromRequestParts, http::request::Parts};
use inference_api::Engine;
pub use inference_api::types::*;

use crate::auth::Owner;

/// The served engine acting for the request's owner, as an axum extractor.
pub struct OwnedEngine(pub Engine);

impl FromRequestParts<Engine> for OwnedEngine {
    type Rejection = <Extension<Owner> as FromRequestParts<Engine>>::Rejection;

    async fn from_request_parts(
        parts: &mut Parts,
        engine: &Engine,
    ) -> Result<Self, Self::Rejection> {
        let Extension(owner) = Extension::<Owner>::from_request_parts(parts, engine).await?;
        Ok(Self(match owner.0 {
            Some(owner) => engine.for_owner(owner),
            None => engine.clone(),
        }))
    }
}
