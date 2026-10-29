//! `QueryParams<T>`: axum's `Query<T>` decoded through `FromJson`, rejecting as `Query` does (400, the same message).

use axum::{
    extract::FromRequestParts,
    http::{StatusCode, Uri, request::Parts},
    response::{IntoResponse, Response},
};

use crate::{FromJson, JsonError};

pub struct QueryParams<T>(pub T);

pub struct QueryParamsRejection(JsonError);

impl<T: FromJson> QueryParams<T> {
    pub fn try_from_uri(uri: &Uri) -> Result<Self, QueryParamsRejection> {
        crate::from_query(uri.query().unwrap_or_default())
            .map(Self)
            .map_err(QueryParamsRejection)
    }
}

impl QueryParamsRejection {
    pub fn body_text(&self) -> String {
        format!("Failed to deserialize query string: {}", self.0)
    }
}

impl IntoResponse for QueryParamsRejection {
    fn into_response(self) -> Response {
        (StatusCode::BAD_REQUEST, self.body_text()).into_response()
    }
}

impl<S: Send + Sync, T: FromJson> FromRequestParts<S> for QueryParams<T> {
    type Rejection = QueryParamsRejection;

    async fn from_request_parts(
        parts: &mut Parts,
        _state: &S,
    ) -> Result<Self, QueryParamsRejection> {
        crate::from_query(parts.uri.query().unwrap_or_default())
            .map(Self)
            .map_err(QueryParamsRejection)
    }
}
