use std::{collections::BTreeMap, convert::Infallible};

use async_stream::stream;
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use serde_json::{Value, json};

use crate::transform::{chat_usage, map_chat_stop, responses_usage};

pub fn convert_chat_stream<S, E>(upstream: S) -> impl Stream<Item = Result<Bytes, Infallible>>
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: std::fmt::Display + Send + 'static,
{
    stream! {
        futures_util::pin_mut!(upstream);
        let mut decoder = SseDecoder::default();
        let mut state = ChatState::default();
        let mut failed = false;
        while let Some(chunk) = upstream.next().await {
            match chunk {
                Ok(bytes) => {
                    for frame in decoder.push(&bytes) {
                        for output in state.process(&frame.data) {
                            yield Ok(output);
                        }
                    }
                }
                Err(error) => {
                    failed = true;
                    yield Ok(error_event(format!("上游流读取失败: {error}")));
                    break;
                }
            }
        }
        if !failed {
            if let Some(frame) = decoder.finish() {
                for output in state.process(&frame.data) {
                    yield Ok(output);
                }
            }
            for output in state.finish() {
                yield Ok(output);
            }
        }
    }
}

pub fn convert_responses_stream<S, E>(upstream: S) -> impl Stream<Item = Result<Bytes, Infallible>>
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: std::fmt::Display + Send + 'static,
{
    stream! {
        futures_util::pin_mut!(upstream);
        let mut decoder = SseDecoder::default();
        let mut state = ResponsesState::default();
        let mut failed = false;
        while let Some(chunk) = upstream.next().await {
            match chunk {
                Ok(bytes) => {
                    for frame in decoder.push(&bytes) {
                        for output in state.process(frame.event.as_deref(), &frame.data) {
                            yield Ok(output);
                        }
                    }
                }
                Err(error) => {
                    failed = true;
                    yield Ok(error_event(format!("上游流读取失败: {error}")));
                    break;
                }
            }
        }
        if !failed {
            if let Some(frame) = decoder.finish() {
                for output in state.process(frame.event.as_deref(), &frame.data) {
                    yield Ok(output);
                }
            }
            for output in state.finish() {
                yield Ok(output);
            }
        }
    }
}

#[derive(Default)]
struct SseDecoder {
    buffer: Vec<u8>,
}

struct SseFrame {
    event: Option<String>,
    data: String,
}

impl SseDecoder {
    fn push(&mut self, chunk: &[u8]) -> Vec<SseFrame> {
        self.buffer.extend_from_slice(chunk);
        let mut frames = Vec::new();
        while let Some((at, delimiter_len)) = find_sse_boundary(&self.buffer) {
            let block = self.buffer.drain(..at).collect::<Vec<_>>();
            self.buffer.drain(..delimiter_len);
            if let Some(frame) = parse_sse_block(&block) {
                frames.push(frame);
            }
        }
        frames
    }

    fn finish(&mut self) -> Option<SseFrame> {
        if self.buffer.is_empty() {
            None
        } else {
            let remaining = std::mem::take(&mut self.buffer);
            parse_sse_block(&remaining)
        }
    }
}

fn find_sse_boundary(bytes: &[u8]) -> Option<(usize, usize)> {
    for i in 0..bytes.len() {
        if bytes.get(i..i + 2) == Some(b"\n\n") {
            return Some((i, 2));
        }
        if bytes.get(i..i + 4) == Some(b"\r\n\r\n") {
            return Some((i, 4));
        }
    }
    None
}

fn parse_sse_block(bytes: &[u8]) -> Option<SseFrame> {
    let text = String::from_utf8_lossy(bytes);
    let mut event = None;
    let mut data = Vec::new();
    for line in text.lines() {
        let line = line.trim_end_matches('\r');
        if let Some(value) = line.strip_prefix("event:") {
            event = Some(value.trim_start().to_string());
        } else if let Some(value) = line.strip_prefix("data:") {
            data.push(value.strip_prefix(' ').unwrap_or(value));
        }
    }
    (!data.is_empty()).then(|| SseFrame {
        event,
        data: data.join("\n"),
    })
}

fn event(name: &str, value: Value) -> Bytes {
    let data = serde_json::to_string(&value).unwrap_or_else(|_| "{}".into());
    Bytes::from(format!("event: {name}\ndata: {data}\n\n"))
}

fn error_event(message: String) -> Bytes {
    event(
        "error",
        json!({
            "type":"error",
            "error":{"type":"api_error", "message":message}
        }),
    )
}

#[derive(Default)]
struct ChatState {
    started: bool,
    finished: bool,
    id: String,
    model: String,
    next_index: u64,
    text_index: Option<u64>,
    thinking_index: Option<u64>,
    tools: BTreeMap<u64, ChatTool>,
    finish_reason: Option<String>,
    usage: Option<Value>,
}

#[derive(Default)]
struct ChatTool {
    anthropic_index: u64,
    id: String,
    name: String,
    started: bool,
    closed: bool,
    pending_arguments: String,
}

impl ChatState {
    fn process(&mut self, data: &str) -> Vec<Bytes> {
        if self.finished || data.trim().is_empty() {
            return Vec::new();
        }
        if data.trim() == "[DONE]" {
            return self.finish();
        }
        let value: Value = match serde_json::from_str(data) {
            Ok(value) => value,
            Err(error) => return vec![error_event(format!("无法解析上游 Chat SSE: {error}"))],
        };
        if let Some(message) = value.pointer("/error/message").and_then(Value::as_str) {
            self.finished = true;
            return vec![error_event(message.to_string())];
        }
        if self.id.is_empty() {
            self.id = value
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
        }
        if self.model.is_empty() {
            self.model = value
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
        }
        let mut out = self.ensure_started();
        if let Some(usage) = value.get("usage").filter(|v| !v.is_null()) {
            self.usage = Some(chat_usage(Some(usage)));
        }
        let Some(choice) = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|v| v.first())
        else {
            return out;
        };
        let delta = choice.get("delta").unwrap_or(&Value::Null);
        if let Some(reasoning) = delta
            .get("reasoning_content")
            .or_else(|| delta.get("reasoning"))
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
        {
            self.close_text(&mut out);
            let index = if let Some(index) = self.thinking_index {
                index
            } else {
                let index = self.allocate_index();
                self.thinking_index = Some(index);
                out.push(event(
                    "content_block_start",
                    json!({
                        "type":"content_block_start", "index":index,
                        "content_block":{"type":"thinking", "thinking":"", "signature":""}
                    }),
                ));
                index
            };
            out.push(event(
                "content_block_delta",
                json!({
                    "type":"content_block_delta", "index":index,
                    "delta":{"type":"thinking_delta", "thinking":reasoning}
                }),
            ));
        }
        if let Some(text) = delta
            .get("content")
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
        {
            self.close_thinking(&mut out);
            let index = if let Some(index) = self.text_index {
                index
            } else {
                let index = self.allocate_index();
                self.text_index = Some(index);
                out.push(event(
                    "content_block_start",
                    json!({
                        "type":"content_block_start", "index":index,
                        "content_block":{"type":"text", "text":""}
                    }),
                ));
                index
            };
            out.push(event(
                "content_block_delta",
                json!({
                    "type":"content_block_delta", "index":index,
                    "delta":{"type":"text_delta", "text":text}
                }),
            ));
        }
        if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
            self.close_text(&mut out);
            self.close_thinking(&mut out);
            for call in calls {
                let key = call.get("index").and_then(Value::as_u64).unwrap_or(0);
                if !self.tools.contains_key(&key) {
                    let anthropic_index = self.allocate_index();
                    self.tools.insert(
                        key,
                        ChatTool {
                            anthropic_index,
                            ..Default::default()
                        },
                    );
                }
                let tool = self.tools.get_mut(&key).expect("inserted tool");
                if let Some(id) = call.get("id").and_then(Value::as_str) {
                    tool.id.push_str(id);
                }
                if let Some(name) = call.pointer("/function/name").and_then(Value::as_str) {
                    tool.name.push_str(name);
                }
                if let Some(args) = call.pointer("/function/arguments").and_then(Value::as_str) {
                    tool.pending_arguments.push_str(args);
                }
                if !tool.started && (!tool.id.is_empty() || !tool.name.is_empty()) {
                    tool.started = true;
                    out.push(event("content_block_start", json!({
                        "type":"content_block_start", "index":tool.anthropic_index,
                        "content_block":{"type":"tool_use", "id":tool.id, "name":tool.name, "input":{}}
                    })));
                }
                if tool.started && !tool.pending_arguments.is_empty() {
                    let args = std::mem::take(&mut tool.pending_arguments);
                    out.push(event(
                        "content_block_delta",
                        json!({
                            "type":"content_block_delta", "index":tool.anthropic_index,
                            "delta":{"type":"input_json_delta", "partial_json":args}
                        }),
                    ));
                }
            }
        }
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish_reason = Some(reason.to_string());
        }
        out
    }

    fn ensure_started(&mut self) -> Vec<Bytes> {
        if self.started {
            return Vec::new();
        }
        self.started = true;
        vec![event(
            "message_start",
            json!({
                "type":"message_start",
                "message":{
                    "id":self.id, "type":"message", "role":"assistant", "content":[],
                    "model":self.model, "stop_reason":null, "stop_sequence":null,
                    "usage":{"input_tokens":0,"output_tokens":0}
                }
            }),
        )]
    }

    fn allocate_index(&mut self) -> u64 {
        let index = self.next_index;
        self.next_index += 1;
        index
    }

    fn close_text(&mut self, out: &mut Vec<Bytes>) {
        if let Some(index) = self.text_index.take() {
            out.push(event(
                "content_block_stop",
                json!({"type":"content_block_stop", "index":index}),
            ));
        }
    }

    fn close_thinking(&mut self, out: &mut Vec<Bytes>) {
        if let Some(index) = self.thinking_index.take() {
            out.push(event(
                "content_block_stop",
                json!({"type":"content_block_stop", "index":index}),
            ));
        }
    }

    fn finish(&mut self) -> Vec<Bytes> {
        if self.finished {
            return Vec::new();
        }
        let mut out = self.ensure_started();
        self.close_text(&mut out);
        self.close_thinking(&mut out);
        for tool in self.tools.values_mut() {
            if !tool.started {
                tool.started = true;
                if tool.id.is_empty() {
                    tool.id = format!("tool_call_{}", tool.anthropic_index);
                }
                if tool.name.is_empty() {
                    tool.name = "unknown_tool".into();
                }
                out.push(event("content_block_start", json!({
                    "type":"content_block_start", "index":tool.anthropic_index,
                    "content_block":{"type":"tool_use", "id":tool.id, "name":tool.name, "input":{}}
                })));
            }
            if !tool.pending_arguments.is_empty() {
                let args = std::mem::take(&mut tool.pending_arguments);
                out.push(event(
                    "content_block_delta",
                    json!({
                        "type":"content_block_delta", "index":tool.anthropic_index,
                        "delta":{"type":"input_json_delta", "partial_json":args}
                    }),
                ));
            }
            if !tool.closed {
                tool.closed = true;
                out.push(event(
                    "content_block_stop",
                    json!({"type":"content_block_stop", "index":tool.anthropic_index}),
                ));
            }
        }
        let has_tools = !self.tools.is_empty();
        let stop = map_chat_stop(self.finish_reason.as_deref(), has_tools);
        out.push(event("message_delta", json!({
            "type":"message_delta",
            "delta":{"stop_reason":stop, "stop_sequence":null},
            "usage":self.usage.clone().unwrap_or_else(|| json!({"input_tokens":0,"output_tokens":0}))
        })));
        out.push(event("message_stop", json!({"type":"message_stop"})));
        self.finished = true;
        out
    }
}

#[derive(Default)]
struct ResponsesState {
    started: bool,
    finished: bool,
    id: String,
    model: String,
    next_index: u64,
    text_index: Option<u64>,
    thinking_index: Option<u64>,
    tools: BTreeMap<String, ResponsesTool>,
    usage: Option<Value>,
    incomplete_reason: Option<String>,
}

#[derive(Default)]
struct ResponsesTool {
    anthropic_index: u64,
    id: String,
    name: String,
    started: bool,
    closed: bool,
    emitted_arguments: bool,
}

impl ResponsesState {
    fn process(&mut self, event_name: Option<&str>, data: &str) -> Vec<Bytes> {
        if self.finished || data.trim().is_empty() || data.trim() == "[DONE]" {
            return if data.trim() == "[DONE]" {
                self.finish()
            } else {
                Vec::new()
            };
        }
        let value: Value = match serde_json::from_str(data) {
            Ok(value) => value,
            Err(error) => return vec![error_event(format!("无法解析上游 Responses SSE: {error}"))],
        };
        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .or(event_name)
            .unwrap_or("");
        if let Some(response) = value.get("response") {
            self.capture_response(response);
        }
        if let Some(message) = value.pointer("/error/message").and_then(Value::as_str) {
            self.finished = true;
            return vec![error_event(message.to_string())];
        }
        if matches!(kind, "response.failed" | "response.cancelled" | "error") {
            self.finished = true;
            let message = value
                .pointer("/response/error/message")
                .or_else(|| value.pointer("/error/message"))
                .and_then(Value::as_str)
                .unwrap_or("OpenAI Responses 流失败");
            return vec![error_event(message.to_string())];
        }
        let mut out = self.ensure_started();
        match kind {
            "response.output_item.added" => {
                if let Some(item) = value.get("item")
                    && item.get("type").and_then(Value::as_str) == Some("function_call")
                {
                    self.ensure_tool(item_key(&value, item), item, &mut out);
                }
            }
            "response.content_part.added" => {
                if value.pointer("/part/type").and_then(Value::as_str) == Some("output_text") {
                    self.ensure_text(&mut out);
                }
            }
            "response.output_text.delta" => {
                let index = self.ensure_text(&mut out);
                if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                    out.push(event(
                        "content_block_delta",
                        json!({
                            "type":"content_block_delta", "index":index,
                            "delta":{"type":"text_delta", "text":delta}
                        }),
                    ));
                }
            }
            "response.output_text.done" | "response.content_part.done" => self.close_text(&mut out),
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                let index = self.ensure_thinking(&mut out);
                if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                    out.push(event(
                        "content_block_delta",
                        json!({
                            "type":"content_block_delta", "index":index,
                            "delta":{"type":"thinking_delta", "thinking":delta}
                        }),
                    ));
                }
            }
            "response.reasoning_summary_text.done" | "response.reasoning_text.done" => {
                self.close_thinking(&mut out)
            }
            "response.function_call_arguments.delta" => {
                let key = item_key(&value, &Value::Null);
                let placeholder = json!({
                    "id":value.get("item_id"),
                    "call_id":value.get("call_id"),
                    "name":value.get("name")
                });
                self.ensure_tool(key.clone(), &placeholder, &mut out);
                if let Some(tool) = self.tools.get_mut(&key)
                    && let Some(delta) = value.get("delta").and_then(Value::as_str)
                {
                    tool.emitted_arguments = true;
                    out.push(event(
                        "content_block_delta",
                        json!({
                            "type":"content_block_delta", "index":tool.anthropic_index,
                            "delta":{"type":"input_json_delta", "partial_json":delta}
                        }),
                    ));
                }
            }
            "response.output_item.done" => {
                if let Some(item) = value.get("item")
                    && item.get("type").and_then(Value::as_str) == Some("function_call")
                {
                    let key = item_key(&value, item);
                    self.ensure_tool(key.clone(), item, &mut out);
                    if let Some(tool) = self.tools.get_mut(&key) {
                        if !tool.emitted_arguments
                            && let Some(arguments) = item
                                .get("arguments")
                                .and_then(Value::as_str)
                                .filter(|s| !s.is_empty())
                        {
                            out.push(event(
                                "content_block_delta",
                                json!({
                                    "type":"content_block_delta", "index":tool.anthropic_index,
                                    "delta":{"type":"input_json_delta", "partial_json":arguments}
                                }),
                            ));
                        }
                        if !tool.closed {
                            tool.closed = true;
                            out.push(event(
                                "content_block_stop",
                                json!({"type":"content_block_stop", "index":tool.anthropic_index}),
                            ));
                        }
                    }
                }
            }
            "response.completed" | "response.incomplete" => {
                out.extend(self.finish());
            }
            _ => {}
        }
        out
    }

    fn capture_response(&mut self, response: &Value) {
        if self.id.is_empty() {
            self.id = response
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
        }
        if self.model.is_empty() {
            self.model = response
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
        }
        if let Some(usage) = response.get("usage").filter(|value| !value.is_null()) {
            self.usage = Some(responses_usage(Some(usage)));
        }
        if let Some(reason) = response
            .pointer("/incomplete_details/reason")
            .and_then(Value::as_str)
        {
            self.incomplete_reason = Some(reason.to_string());
        }
    }

    fn ensure_started(&mut self) -> Vec<Bytes> {
        if self.started {
            return Vec::new();
        }
        self.started = true;
        vec![event(
            "message_start",
            json!({
                "type":"message_start",
                "message":{
                    "id":self.id, "type":"message", "role":"assistant", "content":[],
                    "model":self.model, "stop_reason":null, "stop_sequence":null,
                    "usage":{"input_tokens":0,"output_tokens":0}
                }
            }),
        )]
    }

    fn allocate_index(&mut self) -> u64 {
        let index = self.next_index;
        self.next_index += 1;
        index
    }

    fn ensure_text(&mut self, out: &mut Vec<Bytes>) -> u64 {
        self.close_thinking(out);
        if let Some(index) = self.text_index {
            return index;
        }
        let index = self.allocate_index();
        self.text_index = Some(index);
        out.push(event(
            "content_block_start",
            json!({
                "type":"content_block_start", "index":index,
                "content_block":{"type":"text", "text":""}
            }),
        ));
        index
    }

    fn ensure_thinking(&mut self, out: &mut Vec<Bytes>) -> u64 {
        self.close_text(out);
        if let Some(index) = self.thinking_index {
            return index;
        }
        let index = self.allocate_index();
        self.thinking_index = Some(index);
        out.push(event(
            "content_block_start",
            json!({
                "type":"content_block_start", "index":index,
                "content_block":{"type":"thinking", "thinking":"", "signature":""}
            }),
        ));
        index
    }

    fn ensure_tool(&mut self, key: String, item: &Value, out: &mut Vec<Bytes>) {
        self.close_text(out);
        self.close_thinking(out);
        if !self.tools.contains_key(&key) {
            let anthropic_index = self.allocate_index();
            self.tools.insert(
                key.clone(),
                ResponsesTool {
                    anthropic_index,
                    ..Default::default()
                },
            );
        }
        let tool = self.tools.get_mut(&key).expect("inserted tool");
        if tool.id.is_empty() {
            tool.id = item
                .get("call_id")
                .and_then(Value::as_str)
                .or_else(|| item.get("id").and_then(Value::as_str))
                .unwrap_or("")
                .to_string();
        }
        if tool.name.is_empty() {
            tool.name = item
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
        }
        if !tool.started {
            tool.started = true;
            out.push(event(
                "content_block_start",
                json!({
                    "type":"content_block_start", "index":tool.anthropic_index,
                    "content_block":{"type":"tool_use", "id":tool.id, "name":tool.name, "input":{}}
                }),
            ));
        }
    }

    fn close_text(&mut self, out: &mut Vec<Bytes>) {
        if let Some(index) = self.text_index.take() {
            out.push(event(
                "content_block_stop",
                json!({"type":"content_block_stop", "index":index}),
            ));
        }
    }

    fn close_thinking(&mut self, out: &mut Vec<Bytes>) {
        if let Some(index) = self.thinking_index.take() {
            out.push(event(
                "content_block_stop",
                json!({"type":"content_block_stop", "index":index}),
            ));
        }
    }

    fn finish(&mut self) -> Vec<Bytes> {
        if self.finished {
            return Vec::new();
        }
        let mut out = self.ensure_started();
        self.close_text(&mut out);
        self.close_thinking(&mut out);
        for tool in self.tools.values_mut() {
            if !tool.closed {
                tool.closed = true;
                out.push(event(
                    "content_block_stop",
                    json!({"type":"content_block_stop", "index":tool.anthropic_index}),
                ));
            }
        }
        let stop = if !self.tools.is_empty() {
            "tool_use"
        } else if self.incomplete_reason.is_some() {
            "max_tokens"
        } else {
            "end_turn"
        };
        out.push(event("message_delta", json!({
            "type":"message_delta",
            "delta":{"stop_reason":stop, "stop_sequence":null},
            "usage":self.usage.clone().unwrap_or_else(|| json!({"input_tokens":0,"output_tokens":0}))
        })));
        out.push(event("message_stop", json!({"type":"message_stop"})));
        self.finished = true;
        out
    }
}

fn item_key(event: &Value, item: &Value) -> String {
    event
        .get("item_id")
        .or_else(|| item.get("id"))
        .or_else(|| item.get("call_id"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| {
            format!(
                "output_{}",
                event
                    .get("output_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
            )
        })
}

#[cfg(test)]
mod tests {
    use futures_util::{StreamExt, stream};

    use super::*;

    #[tokio::test]
    async fn chat_stream_is_anthropic_sse() {
        let source = concat!(
            "data: {\"id\":\"chat_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1}}\n\n",
            "data: [DONE]\n\n"
        );
        let chunks = stream::iter(vec![Ok::<_, std::io::Error>(Bytes::from(source))]);
        let output = convert_chat_stream(chunks)
            .map(|chunk| String::from_utf8_lossy(&chunk.unwrap()).into_owned())
            .collect::<Vec<_>>()
            .await
            .join("");
        assert!(output.contains("event: message_start"));
        assert!(output.contains("\"text\":\"Hi\""));
        assert!(output.contains("event: message_stop"));
    }

    #[tokio::test]
    async fn decoder_accepts_split_crlf_frames() {
        let mut decoder = SseDecoder::default();
        assert!(decoder.push(b"event: x\r\ndata: {\"").is_empty());
        let frames = decoder.push(b"a\":1}\r\n\r\n");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].event.as_deref(), Some("x"));
        assert_eq!(frames[0].data, "{\"a\":1}");
    }

    #[tokio::test]
    async fn responses_stream_is_anthropic_sse() {
        let source = concat!(
            "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5\"}}\n\n",
            "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n",
            "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5\",\"status\":\"completed\",\"usage\":{\"input_tokens\":8,\"output_tokens\":2}}}\n\n"
        );
        let chunks = stream::iter(vec![Ok::<_, std::io::Error>(Bytes::from(source))]);
        let output = convert_responses_stream(chunks)
            .map(|chunk| String::from_utf8_lossy(&chunk.unwrap()).into_owned())
            .collect::<Vec<_>>()
            .await
            .join("");
        assert!(output.contains("\"id\":\"resp_1\""));
        assert!(output.contains("\"text\":\"Hello\""));
        assert!(output.contains("\"output_tokens\":2"));
        assert!(output.contains("event: message_stop"));
    }
}
