mod error;
mod proxy;
mod sse;
mod transform;

use std::{
    fs::{self, OpenOptions},
    io::{self, Write},
    net::{Ipv4Addr, SocketAddr},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use axum::{
    Router,
    extract::DefaultBodyLimit,
    middleware,
    routing::{get, post},
};
use clap::{Parser, ValueEnum};
use proxy::{AppState, health, messages};
use tracing::{error, info};
use tracing_subscriber::{
    EnvFilter, fmt, fmt::MakeWriter, layer::SubscriberExt, util::SubscriberInitExt,
};
use url::Url;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum OpenAiType {
    Responses,
    Chat,
}

#[derive(Debug, Parser)]
#[command(
    name = "claude-proxy-rust",
    version,
    disable_version_flag = true,
    about = "将 Anthropic Messages API 转发到 OpenAI Responses/Chat API",
    arg_required_else_help = true
)]
struct Cli {
    /// 本地 Anthropic 服务监听端口
    #[arg(short = 'p', long)]
    port: u16,

    /// 上游 OpenAI API 类型：Responses 或 Chat（大小写不敏感）
    #[arg(short = 't', long, value_enum, ignore_case = true)]
    openai_type: OpenAiType,

    /// OpenAI 兼容服务的基础 URL，例如 https://api.openai.com/v1
    #[arg(short = 'u', long, value_parser = parse_base_url)]
    base_url: Url,

    /// 可选日志文件路径；指定后日志会同时输出到控制台和该文件
    #[arg(short = 'l', long)]
    log_path: Option<PathBuf>,

    /// 将 messages 中的 system 消息归并到开头，默认开启
    #[arg(
        short = 's',
        long,
        default_value_t = true,
        action = clap::ArgAction::Set
    )]
    fix_system_message: bool,

    /// 显示版本信息
    #[arg(
        short = 'v',
        long,
        action = clap::ArgAction::Version,
        required = false
    )]
    version: Option<bool>,
}

fn parse_base_url(raw: &str) -> Result<Url, String> {
    let url = Url::parse(raw).map_err(|error| format!("base_url 无效: {error}"))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err("base_url 只支持 http 或 https".to_string());
    }
    if url.host_str().is_none() {
        return Err("base_url 必须包含主机名".to_string());
    }
    Ok(url)
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    if let Err(error) = init_logging(cli.log_path.as_deref()) {
        eprintln!("无法初始化日志: {error}");
        std::process::exit(2);
    }
    let upstream_url = match proxy::upstream_url(&cli.base_url, cli.openai_type) {
        Ok(url) => url,
        Err(error) => {
            error!(%error, "invalid configuration");
            std::process::exit(2);
        }
    };

    let state = Arc::new(AppState {
        client: reqwest::Client::builder()
            .tcp_nodelay(true)
            .build()
            .expect("failed to build HTTP client"),
        openai_type: cli.openai_type,
        upstream_url,
        fix_system_message: cli.fix_system_message,
    });

    let app = Router::new()
        .route("/v1/messages", post(messages))
        .route("/messages", post(messages))
        .route("/health", get(health))
        .fallback(proxy::not_found)
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .layer(middleware::from_fn(proxy::access_log))
        .with_state(state.clone());

    // Deliberately bind only to loopback. The proxy is never exposed on a LAN interface.
    let address = (Ipv4Addr::LOCALHOST, cli.port);
    let listener = match tokio::net::TcpListener::bind(address).await {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("无法监听 127.0.0.1:{}: {error}", cli.port);
            std::process::exit(1);
        }
    };

    let mut logged_upstream = state.upstream_url.clone();
    logged_upstream.set_query(None);
    logged_upstream.set_fragment(None);
    info!(
        listen = %format_args!("http://127.0.0.1:{}", cli.port),
        upstream = %logged_upstream,
        openai_type = ?state.openai_type,
        fix_system_message = state.fix_system_message,
        log_path = ?cli.log_path,
        "Anthropic proxy started"
    );

    if let Err(error) = axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    {
        error!(%error, "server exited unexpectedly");
        std::process::exit(1);
    }
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    info!("shutdown signal received");
}

#[derive(Clone)]
struct SharedLogFile(Arc<Mutex<std::fs::File>>);

struct SharedLogFileWriter(Arc<Mutex<std::fs::File>>);

impl<'a> MakeWriter<'a> for SharedLogFile {
    type Writer = SharedLogFileWriter;

    fn make_writer(&'a self) -> Self::Writer {
        SharedLogFileWriter(self.0.clone())
    }
}

impl Write for SharedLogFileWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .map_err(|_| io::Error::other("日志文件锁已损坏"))?
            .write_all(bytes)?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0
            .lock()
            .map_err(|_| io::Error::other("日志文件锁已损坏"))?
            .flush()
    }
}

fn init_logging(path: Option<&Path>) -> Result<(), String> {
    let file = path
        .map(|path| {
            let parent = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."));
            fs::create_dir_all(parent)
                .map_err(|error| format!("无法创建日志目录 {}: {error}", parent.display()))?;
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map(|file| Arc::new(Mutex::new(file)))
                .map_err(|error| format!("无法打开日志文件 {}: {error}", path.display()))
        })
        .transpose()?;

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into());
    let result = if let Some(file) = file {
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt::layer().with_target(false).with_ansi(true).compact())
            .with(
                fmt::layer()
                    .with_target(false)
                    .with_ansi(false)
                    .with_writer(SharedLogFile(file))
                    .compact(),
            )
            .try_init()
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt::layer().with_target(false).with_ansi(true).compact())
            .try_init()
    };
    result.map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_options_enable_system_fix_by_default() {
        let cli = Cli::try_parse_from([
            "claude-proxy-rust",
            "-p",
            "8080",
            "-t",
            "Chat",
            "-u",
            "https://api.openai.com/v1",
        ])
        .unwrap();
        assert_eq!(cli.port, 8080);
        assert_eq!(cli.openai_type, OpenAiType::Chat);
        assert!(cli.fix_system_message);
    }

    #[test]
    fn system_fix_can_be_disabled_explicitly() {
        let cli = Cli::try_parse_from([
            "claude-proxy-rust",
            "--port",
            "8080",
            "--openai-type",
            "Responses",
            "--base-url",
            "https://api.openai.com/v1",
            "--fix-system-message",
            "false",
        ])
        .unwrap();
        assert_eq!(cli.openai_type, OpenAiType::Responses);
        assert!(!cli.fix_system_message);
    }
}
