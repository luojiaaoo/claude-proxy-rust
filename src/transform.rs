use serde_json::{Map, Value, json};

use crate::error::ProxyError;

/// Move every `role: "system"` entry out of `messages` and merge its text into
/// the top-level Anthropic `system` field. This guarantees that Chat backends
/// receive exactly one leading system message and Responses backends receive
/// the same text through `instructions`.
pub fn fix_system_message_order(body: &mut Value) {
    let mut system_parts = body
        .get("system")
        .map(text_from_content)
        .filter(|text| !text.is_empty())
        .into_iter()
        .collect::<Vec<_>>();

    if let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) {
        messages.retain(|message| {
            if message.get("role").and_then(Value::as_str) != Some("system") {
                return true;
            }
            let text = message
                .get("content")
                .map(text_from_content)
                .unwrap_or_default();
            if !text.is_empty() {
                system_parts.push(text);
            }
            false
        });
    }

    if !system_parts.is_empty() {
        body["system"] = json!(system_parts.join("\n\n"));
    }
}

pub fn anthropic_to_chat(body: &Value) -> Result<Value, ProxyError> {
    validate_anthropic_request(body)?;
    let mut out = Map::new();
    copy(&mut out, body, "model", "model");

    let mut messages = Vec::new();
    if let Some(system) = body.get("system") {
        let text = text_from_content(system);
        if !text.is_empty() {
            messages.push(json!({"role":"system", "content":text}));
        }
    }

    for message in body["messages"].as_array().expect("validated messages") {
        convert_anthropic_message_to_chat(message, &mut messages)?;
    }
    out.insert("messages".into(), Value::Array(messages));

    let model = body["model"].as_str().unwrap_or_default();
    if let Some(value) = body.get("max_tokens") {
        let key = if is_o_series(model) {
            "max_completion_tokens"
        } else {
            "max_tokens"
        };
        out.insert(key.into(), value.clone());
    }
    for key in ["temperature", "top_p", "stream"] {
        copy(&mut out, body, key, key);
    }
    if let Some(value) = body.get("stop_sequences") {
        out.insert("stop".into(), value.clone());
    }
    if body.get("stream").and_then(Value::as_bool) == Some(true) {
        out.insert("stream_options".into(), json!({"include_usage":true}));
    }
    if supports_reasoning(model)
        && let Some(effort) = reasoning_effort(body)
    {
        out.insert("reasoning_effort".into(), json!(effort));
    }
    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        let tools = tools
            .iter()
            .filter(|tool| tool.get("name").and_then(Value::as_str).is_some())
            .map(|tool| {
                let mut function = json!({
                    "name": tool["name"],
                    "parameters": tool.get("input_schema").cloned().unwrap_or_else(|| json!({"type":"object"}))
                });
                if let Some(description) = tool.get("description") {
                    function["description"] = description.clone();
                }
                json!({"type":"function", "function":function})
            })
            .collect::<Vec<_>>();
        if !tools.is_empty() {
            out.insert("tools".into(), Value::Array(tools));
        }
    }
    if let Some(choice) = body.get("tool_choice") {
        out.insert("tool_choice".into(), chat_tool_choice(choice));
        if let Some(disable) = choice
            .get("disable_parallel_tool_use")
            .and_then(Value::as_bool)
        {
            out.insert("parallel_tool_calls".into(), json!(!disable));
        }
    }
    if let Some(user_id) = body.pointer("/metadata/user_id") {
        out.insert("user".into(), user_id.clone());
    }
    Ok(Value::Object(out))
}

fn convert_anthropic_message_to_chat(
    message: &Value,
    output: &mut Vec<Value>,
) -> Result<(), ProxyError> {
    let role = message
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or("user");
    let content = message.get("content").unwrap_or(&Value::Null);
    if let Some(text) = content.as_str() {
        output.push(json!({"role":role, "content":text}));
        return Ok(());
    }
    let Some(blocks) = content.as_array() else {
        output.push(json!({"role":role, "content":null}));
        return Ok(());
    };

    let mut content_parts = Vec::new();
    let mut tool_calls = Vec::new();
    let mut tool_results = Vec::new();
    let mut reasoning = Vec::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str).unwrap_or("") {
            "text" => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    content_parts.push(json!({"type":"text", "text":text}));
                }
            }
            "image" => {
                if let Some(url) = anthropic_image_url(block) {
                    content_parts.push(json!({"type":"image_url", "image_url":{"url":url}}));
                }
            }
            "tool_use" => {
                let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                let arguments = serde_json::to_string(&input)
                    .map_err(|e| ProxyError::InvalidRequest(e.to_string()))?;
                tool_calls.push(json!({
                    "id": block.get("id").and_then(Value::as_str).unwrap_or(""),
                    "type":"function",
                    "function":{
                        "name":block.get("name").and_then(Value::as_str).unwrap_or(""),
                        "arguments":arguments
                    }
                }));
            }
            "tool_result" => {
                tool_results.push(json!({
                    "role":"tool",
                    "tool_call_id":block.get("tool_use_id").and_then(Value::as_str).unwrap_or(""),
                    "content":tool_result_text(block.get("content"))
                }));
            }
            "thinking" => {
                if let Some(text) = block.get("thinking").and_then(Value::as_str) {
                    reasoning.push(text);
                }
            }
            _ => {}
        }
    }

    // OpenAI requires tool results to be standalone messages following an assistant call.
    if !tool_results.is_empty() {
        output.extend(tool_results);
    }
    if !content_parts.is_empty() || !tool_calls.is_empty() || role == "assistant" {
        let content = if content_parts.is_empty() {
            Value::Null
        } else if content_parts.iter().all(|part| part["type"] == "text") {
            Value::String(
                content_parts
                    .iter()
                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join(""),
            )
        } else {
            Value::Array(content_parts)
        };
        let mut converted = json!({"role":role, "content":content});
        if !tool_calls.is_empty() {
            converted["tool_calls"] = Value::Array(tool_calls);
        }
        if !reasoning.is_empty() {
            converted["reasoning_content"] = json!(reasoning.join("\n"));
        }
        output.push(converted);
    }
    Ok(())
}

pub fn anthropic_to_responses(body: &Value) -> Result<Value, ProxyError> {
    validate_anthropic_request(body)?;
    let mut out = Map::new();
    copy(&mut out, body, "model", "model");
    if let Some(system) = body.get("system") {
        let instructions = text_from_content(system);
        if !instructions.is_empty() {
            out.insert("instructions".into(), json!(instructions));
        }
    }
    let mut input = Vec::new();
    for message in body["messages"].as_array().expect("validated messages") {
        convert_anthropic_message_to_responses(message, &mut input)?;
    }
    out.insert("input".into(), Value::Array(input));
    if let Some(value) = body.get("max_tokens") {
        let value = match value.as_u64() {
            Some(1..=15) => json!(16),
            _ => value.clone(),
        };
        out.insert("max_output_tokens".into(), value);
    }
    for key in ["temperature", "top_p", "stream"] {
        copy(&mut out, body, key, key);
    }
    let model = body["model"].as_str().unwrap_or_default();
    if supports_reasoning(model)
        && let Some(effort) = reasoning_effort(body)
    {
        out.insert("reasoning".into(), json!({"effort":effort}));
    }
    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        let converted = tools
            .iter()
            .filter_map(|tool| {
                let name = tool.get("name").and_then(Value::as_str)?;
                let mut item = json!({
                    "type":"function",
                    "name":name,
                    "parameters":tool.get("input_schema").cloned().unwrap_or_else(|| json!({"type":"object"}))
                });
                if let Some(description) = tool.get("description") {
                    item["description"] = description.clone();
                }
                Some(item)
            })
            .collect::<Vec<_>>();
        if !converted.is_empty() {
            out.insert("tools".into(), Value::Array(converted));
        }
    }
    if let Some(choice) = body.get("tool_choice") {
        out.insert("tool_choice".into(), responses_tool_choice(choice));
        if let Some(disable) = choice
            .get("disable_parallel_tool_use")
            .and_then(Value::as_bool)
        {
            out.insert("parallel_tool_calls".into(), json!(!disable));
        }
    }
    if let Some(user_id) = body.pointer("/metadata/user_id") {
        out.insert("user".into(), user_id.clone());
    }
    Ok(Value::Object(out))
}

fn convert_anthropic_message_to_responses(
    message: &Value,
    output: &mut Vec<Value>,
) -> Result<(), ProxyError> {
    let role = message
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or("user");
    let content = message.get("content").unwrap_or(&Value::Null);
    if let Some(text) = content.as_str() {
        let kind = if role == "assistant" {
            "output_text"
        } else {
            "input_text"
        };
        output.push(json!({"role":role, "content":[{"type":kind, "text":text}]}));
        return Ok(());
    }
    let Some(blocks) = content.as_array() else {
        output.push(json!({"role":role}));
        return Ok(());
    };
    let mut parts = Vec::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str).unwrap_or("") {
            "text" => {
                let kind = if role == "assistant" {
                    "output_text"
                } else {
                    "input_text"
                };
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    parts.push(json!({"type":kind, "text":text}));
                }
            }
            "image" => {
                if let Some(url) = anthropic_image_url(block) {
                    parts.push(json!({"type":"input_image", "image_url":url}));
                }
            }
            "document" => {
                if let Some(file) = anthropic_document(block) {
                    parts.push(file);
                }
            }
            "tool_use" => {
                flush_response_message(role, &mut parts, output);
                let arguments =
                    serde_json::to_string(block.get("input").unwrap_or(&Value::Object(Map::new())))
                        .map_err(|e| ProxyError::InvalidRequest(e.to_string()))?;
                output.push(json!({
                    "type":"function_call",
                    "call_id":block.get("id").and_then(Value::as_str).unwrap_or(""),
                    "name":block.get("name").and_then(Value::as_str).unwrap_or(""),
                    "arguments":arguments
                }));
            }
            "tool_result" => {
                flush_response_message(role, &mut parts, output);
                output.push(json!({
                    "type":"function_call_output",
                    "call_id":block.get("tool_use_id").and_then(Value::as_str).unwrap_or(""),
                    "output":tool_result_text(block.get("content"))
                }));
            }
            _ => {}
        }
    }
    flush_response_message(role, &mut parts, output);
    Ok(())
}

fn flush_response_message(role: &str, parts: &mut Vec<Value>, output: &mut Vec<Value>) {
    if !parts.is_empty() {
        output.push(json!({"role":role, "content":std::mem::take(parts)}));
    }
}

pub fn chat_usage(value: Option<&Value>) -> Value {
    let usage = value.unwrap_or(&Value::Null);
    let cached = number_at(
        usage,
        &[
            "/cache_read_input_tokens",
            "/prompt_tokens_details/cached_tokens",
        ],
    );
    let created = number_at(
        usage,
        &[
            "/cache_creation_input_tokens",
            "/prompt_tokens_details/cache_write_tokens",
        ],
    );
    let total = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    usage_json(
        total,
        usage
            .get("completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        cached,
        created,
    )
}

pub fn responses_usage(value: Option<&Value>) -> Value {
    let usage = value.unwrap_or(&Value::Null);
    let cached = number_at(
        usage,
        &[
            "/cache_read_input_tokens",
            "/input_tokens_details/cached_tokens",
        ],
    );
    let created = number_at(
        usage,
        &[
            "/cache_creation_input_tokens",
            "/input_tokens_details/cache_write_tokens",
        ],
    );
    let total = usage
        .get("input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    usage_json(
        total,
        usage
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        cached,
        created,
    )
}

fn usage_json(total: u64, output: u64, cached: u64, created: u64) -> Value {
    let mut usage = json!({
        "input_tokens":total.saturating_sub(cached).saturating_sub(created),
        "output_tokens":output
    });
    if cached > 0 {
        usage["cache_read_input_tokens"] = json!(cached);
    }
    if created > 0 {
        usage["cache_creation_input_tokens"] = json!(created);
    }
    usage
}

fn number_at(value: &Value, pointers: &[&str]) -> u64 {
    pointers
        .iter()
        .find_map(|pointer| value.pointer(pointer).and_then(Value::as_u64))
        .unwrap_or(0)
}

pub fn map_chat_stop(finish: Option<&str>, has_tools: bool) -> &'static str {
    match finish {
        Some("length") => "max_tokens",
        Some("tool_calls" | "function_call") => "tool_use",
        _ if has_tools => "tool_use",
        _ => "end_turn",
    }
}

fn validate_anthropic_request(body: &Value) -> Result<(), ProxyError> {
    if body.get("model").and_then(Value::as_str).is_none() {
        return Err(ProxyError::InvalidRequest("缺少字符串字段 model".into()));
    }
    if body.get("messages").and_then(Value::as_array).is_none() {
        return Err(ProxyError::InvalidRequest("缺少数组字段 messages".into()));
    }
    Ok(())
}

fn anthropic_image_url(block: &Value) -> Option<String> {
    let source = block.get("source")?;
    match source.get("type").and_then(Value::as_str)? {
        "url" => source
            .get("url")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        "base64" => {
            let media_type = source
                .get("media_type")
                .and_then(Value::as_str)
                .unwrap_or("image/png");
            let data = source.get("data").and_then(Value::as_str)?;
            Some(format!("data:{media_type};base64,{data}"))
        }
        _ => None,
    }
}

fn anthropic_document(block: &Value) -> Option<Value> {
    let source = block.get("source")?;
    let filename = block
        .get("title")
        .and_then(Value::as_str)
        .unwrap_or("document");
    match source.get("type").and_then(Value::as_str)? {
        "url" => Some(json!({
            "type":"input_file",
            "file_url":source.get("url")?,
            "filename":filename
        })),
        "base64" => {
            let media = source
                .get("media_type")
                .and_then(Value::as_str)
                .unwrap_or("application/pdf");
            let data = source.get("data").and_then(Value::as_str)?;
            Some(
                json!({"type":"input_file", "file_data":format!("data:{media};base64,{data}"), "filename":filename}),
            )
        }
        _ => None,
    }
}

fn tool_result_text(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .map(|part| {
                part.get("text")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| serde_json::to_string(part).unwrap_or_default())
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Some(other) => serde_json::to_string(other).unwrap_or_default(),
    }
}

fn text_from_content(value: &Value) -> String {
    if let Some(text) = value.as_str() {
        return strip_billing_header(text).to_string();
    }
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .map(strip_billing_header)
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn strip_billing_header(text: &str) -> &str {
    const PREFIX: &str = "x-anthropic-billing-header:";
    if !text.starts_with(PREFIX) {
        return text;
    }
    text.split_once('\n')
        .map(|(_, rest)| rest.trim_start_matches(['\r', '\n']))
        .unwrap_or("")
}

fn copy(out: &mut Map<String, Value>, input: &Value, from: &str, to: &str) {
    if let Some(value) = input.get(from) {
        out.insert(to.into(), value.clone());
    }
}

fn chat_tool_choice(choice: &Value) -> Value {
    match choice
        .as_str()
        .or_else(|| choice.get("type").and_then(Value::as_str))
    {
        Some("any") => json!("required"),
        Some("tool") => {
            json!({"type":"function", "function":{"name":choice.get("name").and_then(Value::as_str).unwrap_or("")}})
        }
        Some(value) => json!(value),
        None => choice.clone(),
    }
}

fn responses_tool_choice(choice: &Value) -> Value {
    match choice
        .as_str()
        .or_else(|| choice.get("type").and_then(Value::as_str))
    {
        Some("any") => json!("required"),
        Some("tool") => {
            json!({"type":"function", "name":choice.get("name").and_then(Value::as_str).unwrap_or("")})
        }
        Some(value) => json!(value),
        None => choice.clone(),
    }
}

fn is_o_series(model: &str) -> bool {
    model.starts_with('o') && model.as_bytes().get(1).is_some_and(u8::is_ascii_digit)
}

fn supports_reasoning(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    is_o_series(&lower)
        || lower
            .strip_prefix("gpt-")
            .and_then(|s| s.chars().next())
            .is_some_and(|c| c >= '5' && c.is_ascii_digit())
}

fn reasoning_effort(body: &Value) -> Option<&'static str> {
    if let Some(effort) = body
        .pointer("/output_config/effort")
        .and_then(Value::as_str)
    {
        return match effort {
            "low" => Some("low"),
            "medium" => Some("medium"),
            "high" => Some("high"),
            "xhigh" | "max" => Some("xhigh"),
            _ => None,
        };
    }
    match body.pointer("/thinking/type").and_then(Value::as_str) {
        Some("adaptive") => Some("xhigh"),
        Some("enabled") => match body
            .pointer("/thinking/budget_tokens")
            .and_then(Value::as_u64)
        {
            Some(0..=3_999) => Some("low"),
            Some(4_000..=15_999) => Some("medium"),
            _ => Some("high"),
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_anthropic_tools_to_chat() {
        let source = json!({
            "model":"gpt-4o", "max_tokens":100, "stream":true,
            "system":"be useful",
            "messages":[
                {"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"weather","input":{"city":"上海"}}]},
                {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"晴"}]}
            ],
            "tools":[{"name":"weather","description":"天气","input_schema":{"type":"object"}}]
        });
        let result = anthropic_to_chat(&source).unwrap();
        assert_eq!(
            result["messages"][1]["tool_calls"][0]["function"]["name"],
            "weather"
        );
        assert_eq!(result["messages"][2]["role"], "tool");
        assert_eq!(result["stream_options"]["include_usage"], true);
    }

    #[test]
    fn converts_anthropic_tools_to_responses() {
        let source = json!({
            "model":"gpt-5", "max_tokens":8, "stream":true,
            "messages":[
                {"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"lookup","input":{"q":"rust"}}]},
                {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"result"}]}
            ]
        });
        let result = anthropic_to_responses(&source).unwrap();
        assert_eq!(result["input"][0]["type"], "function_call");
        assert_eq!(result["input"][1]["type"], "function_call_output");
        assert_eq!(result["max_output_tokens"], 16);
        assert_eq!(result["stream"], true);
    }

    #[test]
    fn moves_all_system_messages_to_the_beginning() {
        let mut source = json!({
            "model":"gpt-4o", "stream":true,
            "system":"top-level",
            "messages":[
                {"role":"user","content":"first"},
                {"role":"system","content":"middle"},
                {"role":"assistant","content":"answer"},
                {"role":"system","content":[{"type":"text","text":"last"}]}
            ]
        });

        fix_system_message_order(&mut source);
        assert_eq!(source["system"], "top-level\n\nmiddle\n\nlast");
        assert_eq!(source["messages"].as_array().unwrap().len(), 2);

        let chat = anthropic_to_chat(&source).unwrap();
        assert_eq!(chat["messages"][0]["role"], "system");
        assert_eq!(chat["messages"][1]["role"], "user");
        assert_eq!(chat["messages"][2]["role"], "assistant");
    }
}
