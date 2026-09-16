# claude-proxy-rust

把本机 Anthropic Messages API 流式请求转换为 OpenAI Responses API 或 Chat Completions API。

## 启动

程序使用以下命令行参数：

```text
claude-proxy-rust --port <PORT> --openai-type <OPENAI_TYPE> --base-url <BASE_URL> [OPTIONS]
```

- `-p, --port <PORT>`：本机代理端口。
- `-t, --openai-type <OPENAI_TYPE>`：`Responses` 或 `Chat`，大小写不敏感。
- `-u, --base-url <BASE_URL>`：OpenAI 兼容服务地址。可以传 API 根地址、`.../v1` 地址或对应的完整接口地址。
- `-l, --log-path <LOG_PATH>`：可选日志文件路径。控制台日志会同步追加到该文件；父目录不存在时会自动创建。
- `-s, --fix-system-message <true|false>`：归并所有 system 内容并放到最前面，避免上游返回 `System message must be at the beginning`。默认 `true`。
- `-v, --version`：显示版本。
- `-h, --help`：显示帮助。

例如：

```powershell
cargo run --release -- -p 8080 -t Responses -u https://api.openai.com/v1
cargo run --release -- -p 8080 -t Chat -u https://api.openai.com/v1
cargo run --release -- -p 8080 -t Chat -u https://api.openai.com/v1 -l .\logs\proxy.log
cargo run --release -- -p 8080 -t Chat -u https://api.openai.com/v1 -s false
```

启动后只监听回环地址：

```text
http://127.0.0.1:8080/v1/messages
```

可把 Anthropic 客户端配置为：

```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8080"
$env:ANTHROPIC_API_KEY = "sk-..."
```

客户端传入的 `Authorization` 会原样转发；如果只传 `x-api-key`，代理会把它转换成 OpenAI 的 `Authorization: Bearer ...`。

## 支持范围

- Anthropic `POST /v1/messages`，以及兼容路径 `POST /messages`
- 只支持 `stream: true` 的 SSE 流式请求；非流式请求会返回 `400`
- 文本、图片、Responses 文档输入
- 自定义工具、工具选择、工具调用及工具结果
- token usage、缓存 token 和常见停止原因映射
- 上游错误转换为 Anthropic 错误结构
- `GET /health` 健康检查

## 访问日志

每次访问会记录请求 ID、客户端地址、HTTP 方法、路径、User-Agent、请求大小、状态码、响应大小和处理耗时。转发日志还会记录模型、消息数、工具数、上游状态和上游响应耗时；SSE 结束时记录总字节数、数据块数、总时长及客户端是否提前断开。

日志不会输出 `Authorization` 或 API Key 的具体内容。可通过 `RUST_LOG` 调整日志级别，例如：

```powershell
$env:RUST_LOG = "debug"
```

## 构建与测试

```powershell
cargo test
cargo build --release
```
