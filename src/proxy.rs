use std::{
    convert::Infallible,
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use async_stream::stream;
use axum::{
    Json,
    body::{Body, Bytes},
    extract::{ConnectInfo, Extension, Request, State},
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use futures_util::{Stream, StreamExt};
use http_body::Body as HttpBody;
use serde_json::{Value, json};
use tracing::{info, warn};
use url::Url;

use crate::{
    OpenAiType,
    error::{ProxyError, anthropic_error, upstream_error},
    sse::{convert_chat_stream, convert_responses_stream},
    transform::{anthropic_to_chat, anthropic_to_responses, fix_system_message_order},
};

pub struct AppState {
    pub client: reqwest::Client,
    pub openai_type: OpenAiType,
    pub upstream_url: Url,
    pub fix_system_message: bool,
}

#[derive(Clone)]
pub struct RequestId(String);

static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

pub async fn access_log(mut request: Request, next: Next) -> Response {
    let request_id = RequestId(format!(
        "req-{:016x}",
        NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let method = request.method().clone();
    let uri = request.uri().to_string();
    let version = format!("{:?}", request.version());
    let client = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|address| address.0.to_string())
        .unwrap_or_else(|| "unknown".into());
    let user_agent = request
        .headers()
        .get(header::USER_AGENT)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("-")
        .to_string();
    let request_bytes = request
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("unknown")
        .to_string();
    request.extensions_mut().insert(request_id.clone());

    info!(
        request_id = %request_id.0,
        client = %client,
        method = %method,
        uri = %uri,
        http_version = %version,
        request_bytes = %request_bytes,
        user_agent = %user_agent,
        "access request"
    );
    let started = Instant::now();
    let mut response = next.run(request).await;
    let status = response.status();
    let response_bytes = response
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned)
        .or_else(|| {
            response
                .body()
                .size_hint()
                .exact()
                .map(|size| size.to_string())
        })
        .unwrap_or_else(|| "streaming".into());
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    if let Ok(value) = HeaderValue::from_str(&request_id.0) {
        response.headers_mut().insert("x-proxy-request-id", value);
    }
    info!(
        request_id = %request_id.0,
        client = %client,
        method = %method,
        uri = %uri,
        status = status.as_u16(),
        response_bytes = %response_bytes,
        handler_elapsed_ms = format_args!("{elapsed_ms:.3}"),
        "access response"
    );
    response
}

pub fn upstream_url(base: &Url, api_type: OpenAiType) -> Result<Url, String> {
    let mut url = base.clone();
    let endpoint = match api_type {
        OpenAiType::Responses => "responses",
        OpenAiType::Chat => "chat/completions",
    };
    let path = url.path().trim_end_matches('/');
    let complete_suffix = format!("/{endpoint}");
    let new_path = if path.ends_with(&complete_suffix) {
        path.to_string()
    } else if path.is_empty() || path == "/" {
        format!("/v1/{endpoint}")
    } else {
        format!("{path}/{endpoint}")
    };
    url.set_path(&new_path);
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err("base_url 必须是有效的 http(s) URL".into());
    }
    Ok(url)
}

pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(json!({
        "status":"ok",
        "openai_type":match state.openai_type {
            OpenAiType::Responses => "Responses",
            OpenAiType::Chat => "Chat",
        }
    }))
}

pub async fn not_found() -> Response {
    anthropic_error(
        StatusCode::NOT_FOUND,
        "not_found_error",
        "仅支持 POST /v1/messages（兼容路径：POST /messages）",
    )
}

pub async fn messages(
    State(state): State<Arc<AppState>>,
    Extension(request_id): Extension<RequestId>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    match handle_messages(state, &request_id, headers, body).await {
        Ok(response) => response,
        Err(error) => {
            warn!(request_id = %request_id.0, error = %error, "request failed");
            error.response()
        }
    }
}

async fn handle_messages(
    state: Arc<AppState>,
    request_id: &RequestId,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ProxyError> {
    let mut anthropic: Value = serde_json::from_slice(&body)
        .map_err(|error| ProxyError::InvalidRequest(format!("JSON 无效: {error}")))?;
    if anthropic.get("stream").and_then(Value::as_bool) != Some(true) {
        return Err(ProxyError::InvalidRequest(
            "本代理只支持流式请求，请设置 stream: true".into(),
        ));
    }
    if state.fix_system_message {
        fix_system_message_order(&mut anthropic);
    }
    let upstream_body = match state.openai_type {
        OpenAiType::Responses => anthropic_to_responses(&anthropic)?,
        OpenAiType::Chat => anthropic_to_chat(&anthropic)?,
    };

    let auth_source = if headers.contains_key(header::AUTHORIZATION) {
        "authorization"
    } else if headers.contains_key("x-api-key") {
        "x-api-key"
    } else {
        "none"
    };
    let model = anthropic
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let message_count = anthropic
        .get("messages")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let tool_count = anthropic
        .get("tools")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    info!(
        request_id = %request_id.0,
        openai_type = ?state.openai_type,
        model,
        messages = message_count,
        tools = tool_count,
        request_bytes = body.len(),
        auth_source,
        fix_system_message = state.fix_system_message,
        upstream_host = state.upstream_url.host_str().unwrap_or("unknown"),
        upstream_path = state.upstream_url.path(),
        "forwarding stream request"
    );

    let upstream_started = Instant::now();
    let mut request = state
        .client
        .post(state.upstream_url.clone())
        .header(header::ACCEPT, "text/event-stream")
        .json(&upstream_body);
    request = copy_request_headers(request, &headers)?;
    let upstream = request
        .send()
        .await
        .map_err(|error| ProxyError::Upstream(error.to_string()))?;
    let status = upstream.status();
    let response_headers = upstream.headers().clone();
    let content_type = response_headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("unknown")
        .to_string();
    info!(
        request_id = %request_id.0,
        upstream_status = status.as_u16(),
        content_type = %content_type,
        upstream_elapsed_ms = format_args!("{:.3}", upstream_started.elapsed().as_secs_f64() * 1000.0),
        "upstream response headers"
    );
    if !status.is_success() {
        let bytes = upstream
            .bytes()
            .await
            .map_err(|error| ProxyError::Upstream(error.to_string()))?;
        return Ok(upstream_error(status, &bytes));
    }

    let is_sse = content_type
        .to_ascii_lowercase()
        .contains("text/event-stream");
    if !is_sse {
        return Err(ProxyError::Transform(
            "上游在 stream 模式下未返回 text/event-stream".into(),
        ));
    }
    let converted: Body = match state.openai_type {
        OpenAiType::Responses => Body::from_stream(observe_stream(
            convert_responses_stream(upstream.bytes_stream()),
            request_id.0.clone(),
        )),
        OpenAiType::Chat => Body::from_stream(observe_stream(
            convert_chat_stream(upstream.bytes_stream()),
            request_id.0.clone(),
        )),
    };
    let mut response = Response::new(converted);
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    response
        .headers_mut()
        .insert(header::CONNECTION, HeaderValue::from_static("keep-alive"));
    copy_response_headers(&response_headers, response.headers_mut());
    Ok(response)
}

fn observe_stream<S>(
    upstream: S,
    request_id: String,
) -> impl Stream<Item = Result<Bytes, Infallible>>
where
    S: Stream<Item = Result<Bytes, Infallible>> + Send + 'static,
{
    stream! {
        futures_util::pin_mut!(upstream);
        let mut observation = StreamObservation::new(request_id);
        while let Some(item) = upstream.next().await {
            if let Ok(bytes) = &item {
                observation.chunks += 1;
                observation.bytes += bytes.len();
            }
            yield item;
        }
        observation.completed = true;
    }
}

struct StreamObservation {
    request_id: String,
    started: Instant,
    chunks: u64,
    bytes: usize,
    completed: bool,
}

impl StreamObservation {
    fn new(request_id: String) -> Self {
        Self {
            request_id,
            started: Instant::now(),
            chunks: 0,
            bytes: 0,
            completed: false,
        }
    }
}

impl Drop for StreamObservation {
    fn drop(&mut self) {
        let outcome = if self.completed {
            "completed"
        } else {
            "client_disconnected"
        };
        info!(
            request_id = %self.request_id,
            outcome,
            chunks = self.chunks,
            response_bytes = self.bytes,
            stream_elapsed_ms = format_args!("{:.3}", duration_ms(self.started.elapsed())),
            "SSE stream closed"
        );
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn copy_request_headers(
    mut request: reqwest::RequestBuilder,
    incoming: &HeaderMap,
) -> Result<reqwest::RequestBuilder, ProxyError> {
    if let Some(authorization) = incoming.get(header::AUTHORIZATION) {
        request = request.header(header::AUTHORIZATION, authorization.clone());
    } else if let Some(key) = incoming.get("x-api-key") {
        let key = key
            .to_str()
            .map_err(|_| ProxyError::InvalidRequest("x-api-key 不是有效字符串".into()))?;
        let value = HeaderValue::from_str(&format!("Bearer {key}"))
            .map_err(|_| ProxyError::InvalidRequest("x-api-key 无法转换为 Authorization".into()))?;
        request = request.header(header::AUTHORIZATION, value);
    }

    // Preserve OpenAI routing/account headers and vendor-specific headers used by
    // compatible gateways, while excluding Anthropic protocol and hop-by-hop fields.
    for (name, value) in incoming {
        let key = name.as_str();
        let lower = key.to_ascii_lowercase();
        let allowed = matches!(lower.as_str(), "openai-organization" | "openai-project")
            || (lower.starts_with("x-")
                && lower != "x-api-key"
                && !lower.starts_with("x-anthropic-"));
        if allowed {
            request = request.header(name, value);
        }
    }
    Ok(request)
}

fn copy_response_headers(source: &reqwest::header::HeaderMap, target: &mut HeaderMap) {
    for key in [
        "x-request-id",
        "request-id",
        "retry-after",
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-tokens",
    ] {
        if let (Ok(name), Some(value)) = (HeaderName::from_bytes(key.as_bytes()), source.get(key))
            && let Ok(value) = HeaderValue::from_bytes(value.as_bytes())
        {
            target.insert(name, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_endpoint_from_root_and_v1_base_urls() {
        let root = Url::parse("https://example.com").unwrap();
        assert_eq!(
            upstream_url(&root, OpenAiType::Responses).unwrap().as_str(),
            "https://example.com/v1/responses"
        );
        let v1 = Url::parse("https://example.com/openai/v1?api-version=1").unwrap();
        assert_eq!(
            upstream_url(&v1, OpenAiType::Chat).unwrap().as_str(),
            "https://example.com/openai/v1/chat/completions?api-version=1"
        );
    }

    #[test]
    fn does_not_duplicate_complete_endpoint() {
        let full = Url::parse("https://example.com/v1/responses").unwrap();
        assert_eq!(upstream_url(&full, OpenAiType::Responses).unwrap(), full);
    }
}
