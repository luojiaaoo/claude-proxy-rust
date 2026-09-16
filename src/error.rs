use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProxyError {
    #[error("请求格式错误: {0}")]
    InvalidRequest(String),
    #[error("上游请求失败: {0}")]
    Upstream(String),
    #[error("协议转换失败: {0}")]
    Transform(String),
}

impl ProxyError {
    pub fn response(self) -> Response {
        let (status, error_type) = match self {
            Self::InvalidRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
            Self::Upstream(_) => (StatusCode::BAD_GATEWAY, "api_error"),
            Self::Transform(_) => (StatusCode::BAD_GATEWAY, "api_error"),
        };
        anthropic_error(status, error_type, self.to_string())
    }
}

pub fn anthropic_error(
    status: StatusCode,
    error_type: &str,
    message: impl Into<String>,
) -> Response {
    (
        status,
        Json(json!({
            "type": "error",
            "error": {
                "type": error_type,
                "message": message.into()
            }
        })),
    )
        .into_response()
}

pub fn upstream_error(status: StatusCode, body: &[u8]) -> Response {
    let value: Option<Value> = serde_json::from_slice(body).ok();
    let message = value
        .as_ref()
        .and_then(|v| v.pointer("/error/message").or_else(|| v.get("message")))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| String::from_utf8_lossy(body).trim().to_string());
    let message = if message.is_empty() {
        format!("上游返回 HTTP {status}")
    } else {
        message
    };
    let error_type = match status.as_u16() {
        400 | 422 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        429 => "rate_limit_error",
        _ => "api_error",
    };
    anthropic_error(status, error_type, message)
}
