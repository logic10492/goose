use crate::agents::extension::PlatformExtensionContext;
use crate::agents::mcp_client::{Error, McpClientTrait};
use crate::agents::tool_execution::ToolCallContext;
use crate::config::Config;
use crate::providers::chatgpt_codex::{get_stored_codex_token, CHATGPT_CODEX_DEFAULT_MODEL};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use indoc::indoc;
use rmcp::model::{
    CallToolResult, ContentBlock, Implementation, InitializeResult, JsonObject, ListToolsResult,
    ServerCapabilities, Tool, ToolAnnotations,
};
use schemars::{schema_for, JsonSchema};
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

pub static EXTENSION_NAME: &str = "websearch";

const DEFAULT_NUM_RESULTS: usize = 8;
const DEFAULT_MAX_FETCH_CHARS: usize = 10_000;
const MAX_FETCH_BODY_BYTES: u64 = 10 * 1024 * 1024;
const FETCH_USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) goose-websearch";

const DEEPSEEK_DEFAULT_BASE_URL: &str = "https://api.deepseek.com";
const KIMI_DEFAULT_BASE_URL: &str = "https://api.kimi.com";
const OPENAI_DEFAULT_BASE_URL: &str = "https://api.openai.com";
const OPENAI_DEFAULT_MODEL: &str = "gpt-5-mini";
const CODEX_RESPONSES_URL: &str = "https://chatgpt.com/backend-api/codex/responses";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    DeepSeek,
    Kimi,
    Codex,
    OpenAI,
}

impl Backend {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "deepseek" => Some(Self::DeepSeek),
            "kimi" => Some(Self::Kimi),
            "codex" => Some(Self::Codex),
            "openai" => Some(Self::OpenAI),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::DeepSeek => "deepseek",
            Self::Kimi => "kimi",
            Self::Codex => "codex",
            Self::OpenAI => "openai",
        }
    }

    /// Maps goose's active provider name to the matching search backend, so a
    /// session running on a given model family naturally searches through it.
    fn from_provider_name(provider: &str) -> Option<Self> {
        let name = provider.to_ascii_lowercase();
        if name.contains("codex") || name.contains("chatgpt") {
            Some(Self::Codex)
        } else if name.contains("openai") {
            Some(Self::OpenAI)
        } else if name.contains("deepseek") {
            Some(Self::DeepSeek)
        } else if name.contains("kimi") || name.contains("moonshot") {
            Some(Self::Kimi)
        } else {
            None
        }
    }
}

fn parse_backend_order(value: &str) -> Vec<Backend> {
    let mut backends = Vec::new();
    for part in value.split(',') {
        if let Some(backend) = Backend::parse(part) {
            if !backends.contains(&backend) {
                backends.push(backend);
            }
        }
    }
    backends
}

fn backend_order() -> Vec<Backend> {
    let mut order = Config::global()
        .get_param::<String>("WEBSEARCH_BACKEND_ORDER")
        .ok()
        .map(|order| parse_backend_order(&order))
        .unwrap_or_default();
    if order.is_empty() {
        order = vec![
            Backend::DeepSeek,
            Backend::Kimi,
            Backend::Codex,
            Backend::OpenAI,
        ];
    }
    if let Ok(provider) = Config::global().get_goose_provider() {
        if let Some(hint) = Backend::from_provider_name(&provider) {
            if let Some(position) = order.iter().position(|b| *b == hint) {
                order.remove(position);
            }
            order.insert(0, hint);
        }
    }
    order
}

fn config_secret(key: &str) -> Option<String> {
    Config::global()
        .get(key, true)
        .ok()?
        .as_str()
        .map(str::to_string)
}

fn config_param(key: &str) -> Option<String> {
    Config::global().get_param::<String>(key).ok()
}

#[derive(Debug, Deserialize, JsonSchema)]
struct WebSearchToolParams {
    /// The search query.
    query: String,
    /// Maximum number of source results to return (default 8).
    num_results: Option<usize>,
    /// Search depth: "auto" (default), "fast", or "deep". Deep search requests a
    /// more thorough strategy on backends that support it.
    #[serde(rename = "type")]
    search_type: Option<String>,
    /// "preferred" enables full page crawling on backends that support it;
    /// "fallback" (default) keeps to snippets.
    livecrawl: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct WebFetchToolParams {
    /// The http(s) URL to fetch.
    url: String,
    /// Maximum characters of extracted content to return (default 10000).
    max_chars: Option<usize>,
}

#[derive(Debug)]
struct SearchRequest {
    query: String,
    num_results: usize,
    deep: bool,
    livecrawl_preferred: bool,
}

impl SearchRequest {
    fn prompt(&self) -> String {
        format!(
            "Use web search to answer this query.\nReturn a concise answer with up to {} source URLs.\nQuery: {}",
            self.num_results, self.query
        )
    }
}

#[derive(Debug, Default)]
struct SearchOutput {
    answer: String,
    sources: Vec<(Option<String>, String)>,
}

impl SearchOutput {
    fn render(&self) -> String {
        let mut out = self.answer.clone();
        if !self.sources.is_empty() {
            out.push_str("\n\nSources:\n");
            for (index, (title, url)) in self.sources.iter().enumerate() {
                out.push_str(&format!(
                    "{}. {}\n   {}\n",
                    index + 1,
                    title.as_deref().unwrap_or("Untitled"),
                    url
                ));
            }
        }
        out.trim_end().to_string()
    }
}

pub struct WebSearchClient {
    info: InitializeResult,
    http: reqwest::Client,
}

impl WebSearchClient {
    pub fn new(_context: PlatformExtensionContext) -> Result<Self> {
        let info = InitializeResult::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(
                Implementation::new(EXTENSION_NAME.to_string(), "1.0.0".to_string())
                    .with_title("Web Search"),
            )
            .with_instructions(
                indoc! {r#"
                Aggregated web search across multiple backends (deepseek, kimi, codex, openai)
                with automatic fallback, plus a web_fetch tool to read full page content.

                Use web_search for current information and facts not in training data. Use
                web_fetch to read the full content of a specific URL, for example a promising
                search result.
            "#}
                .to_string(),
            );

        Ok(Self {
            info,
            http: reqwest::Client::new(),
        })
    }

    async fn handle_web_search(
        &self,
        arguments: Option<JsonObject>,
    ) -> Result<Vec<ContentBlock>, String> {
        let args = arguments.ok_or("Missing arguments")?;
        let params: WebSearchToolParams = serde_json::from_value(Value::Object(args))
            .map_err(|e| format!("Invalid arguments: {}", e))?;
        let request = SearchRequest {
            query: params.query,
            num_results: params.num_results.unwrap_or(DEFAULT_NUM_RESULTS),
            deep: params.search_type.as_deref() == Some("deep"),
            livecrawl_preferred: params.livecrawl.as_deref() == Some("preferred"),
        };

        let mut failures: Vec<String> = Vec::new();
        for backend in backend_order() {
            match self.try_backend(backend, &request).await {
                Ok(output) => {
                    let rendered = output.render();
                    if failures.is_empty() {
                        return Ok(vec![ContentBlock::text(rendered)]);
                    }
                    return Ok(vec![ContentBlock::text(format!(
                        "{}\n\n(fell back to {}; earlier failures: {})",
                        rendered,
                        backend.name(),
                        failures.join("; ")
                    ))]);
                }
                Err(error) => failures.push(format!("{}: {}", backend.name(), error)),
            }
        }

        Err(format!(
            "Web search unavailable.\n- {}",
            failures.join("\n- ")
        ))
    }

    async fn try_backend(&self, backend: Backend, request: &SearchRequest) -> Result<SearchOutput> {
        match backend {
            Backend::DeepSeek => {
                let api_key = config_secret("DEEPSEEK_API_KEY")
                    .ok_or_else(|| anyhow!("not configured (missing DEEPSEEK_API_KEY)"))?;
                let base_url = config_param("DEEPSEEK_BASE_URL")
                    .unwrap_or_else(|| DEEPSEEK_DEFAULT_BASE_URL.to_string());
                search_deepseek(&self.http, &base_url, &api_key, request).await
            }
            Backend::Kimi => {
                let api_key = config_secret("KIMI_CODE_TOKEN")
                    .or_else(|| config_secret("KIMI_API_KEY"))
                    .ok_or_else(|| {
                        anyhow!("not configured (missing KIMI_CODE_TOKEN or KIMI_API_KEY)")
                    })?;
                let base_url = config_param("KIMI_BASE_URL")
                    .unwrap_or_else(|| KIMI_DEFAULT_BASE_URL.to_string());
                search_kimi(&self.http, &base_url, &api_key, request).await
            }
            Backend::Codex => {
                let token = get_stored_codex_token().await.ok_or_else(|| {
                    anyhow!("not configured (log in to the chatgpt_codex provider first)")
                })?;
                let model = config_param("WEBSEARCH_CODEX_MODEL")
                    .unwrap_or_else(|| CHATGPT_CODEX_DEFAULT_MODEL.to_string());
                search_codex(
                    &self.http,
                    CODEX_RESPONSES_URL,
                    &token.access_token,
                    token.account_id.as_deref(),
                    &model,
                    request,
                )
                .await
            }
            Backend::OpenAI => {
                let api_key = config_secret("OPENAI_API_KEY")
                    .ok_or_else(|| anyhow!("not configured (missing OPENAI_API_KEY)"))?;
                let base_url = config_param("OPENAI_BASE_URL")
                    .unwrap_or_else(|| OPENAI_DEFAULT_BASE_URL.to_string());
                let model = config_param("WEBSEARCH_OPENAI_MODEL")
                    .unwrap_or_else(|| OPENAI_DEFAULT_MODEL.to_string());
                search_openai(&self.http, &base_url, &api_key, &model, request).await
            }
        }
    }

    async fn handle_web_fetch(
        &self,
        arguments: Option<JsonObject>,
    ) -> Result<Vec<ContentBlock>, String> {
        let args = arguments.ok_or("Missing arguments")?;
        let params: WebFetchToolParams = serde_json::from_value(Value::Object(args))
            .map_err(|e| format!("Invalid arguments: {}", e))?;
        let max_chars = params.max_chars.unwrap_or(DEFAULT_MAX_FETCH_CHARS);

        let url = params.url;
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err("Only http:// and https:// URLs are supported".to_string());
        }

        let response = self
            .http
            .get(&url)
            .header(reqwest::header::USER_AGENT, FETCH_USER_AGENT)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| format!("Fetch failed: {}", e))?
            .error_for_status()
            .map_err(|e| format!("Fetch failed: {}", e))?;

        if let Some(length) = response.content_length() {
            if length > MAX_FETCH_BODY_BYTES {
                return Err(format!("Response too large ({} bytes)", length));
            }
        }

        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let body = response
            .text()
            .await
            .map_err(|e| format!("Failed to read response body: {}", e))?;

        let is_html =
            content_type.contains("text/html") || content_type.contains("application/xhtml");
        let text = if is_html { html_to_text(&body) } else { body };

        if text.chars().count() > max_chars {
            let truncated: String = text.chars().take(max_chars).collect();
            Ok(vec![ContentBlock::text(format!(
                "{}\n\n(truncated to {} characters)",
                truncated, max_chars
            ))])
        } else {
            Ok(vec![ContentBlock::text(text)])
        }
    }

    fn get_tools() -> Vec<Tool> {
        let search_schema = schema_for!(WebSearchToolParams);
        let search_schema_value =
            serde_json::to_value(search_schema).expect("Failed to serialize WebSearchToolParams");
        let fetch_schema = schema_for!(WebFetchToolParams);
        let fetch_schema_value =
            serde_json::to_value(fetch_schema).expect("Failed to serialize WebFetchToolParams");

        vec![
            Tool::new(
                "web_search".to_string(),
                indoc! {r#"
                    Search the web for current information using an aggregated set of search
                    backends (deepseek, kimi, codex, openai) with automatic fallback.
                    Returns a concise answer with source URLs.
                "#}
                .to_string(),
                search_schema_value.as_object().unwrap().clone(),
            )
            .annotate(ToolAnnotations::from_raw(
                Some("Web Search".to_string()),
                Some(true),
                Some(false),
                Some(false),
                Some(true),
            )),
            Tool::new(
                "web_fetch".to_string(),
                indoc! {r#"
                    Fetch a web page and return its content as plain text. Use this to read
                    the full content of a URL, for example a promising web_search result.
                "#}
                .to_string(),
                fetch_schema_value.as_object().unwrap().clone(),
            )
            .annotate(ToolAnnotations::from_raw(
                Some("Web Fetch".to_string()),
                Some(true),
                Some(false),
                Some(true),
                Some(true),
            )),
        ]
    }
}

#[async_trait]
impl McpClientTrait for WebSearchClient {
    async fn list_tools(
        &self,
        _session_id: &str,
        _next_cursor: Option<String>,
        _cancellation_token: CancellationToken,
    ) -> Result<ListToolsResult, Error> {
        Ok(ListToolsResult {
            tools: Self::get_tools(),
            next_cursor: None,
            meta: None,
            ..Default::default()
        })
    }

    async fn call_tool(
        &self,
        _ctx: &ToolCallContext,
        name: &str,
        arguments: Option<JsonObject>,
        _cancellation_token: CancellationToken,
    ) -> Result<CallToolResult, Error> {
        let content = match name {
            "web_search" => self.handle_web_search(arguments).await,
            "web_fetch" => self.handle_web_fetch(arguments).await,
            _ => Err(format!("Unknown tool: {}", name)),
        };

        match content {
            Ok(content) => Ok(CallToolResult::success(content)),
            Err(error) => Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "Error: {}",
                error
            ))])),
        }
    }

    fn get_info(&self) -> Option<&InitializeResult> {
        Some(&self.info)
    }
}

async fn search_deepseek(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    request: &SearchRequest,
) -> Result<SearchOutput> {
    let body = json!({
        "model": "deepseek-chat",
        "max_tokens": 1024,
        "tools": [{ "type": "web_search_20250305", "name": "web_search", "max_uses": 1 }],
        "messages": [{ "role": "user", "content": request.prompt() }],
    });

    let response: Value = client
        .post(format!(
            "{}/anthropic/v1/messages",
            base_url.trim_end_matches('/')
        ))
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .timeout(Duration::from_secs(30))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    let blocks = response
        .get("content")
        .and_then(|c| c.as_array())
        .ok_or_else(|| anyhow!("response has no content blocks"))?;

    let mut answer_parts: Vec<String> = Vec::new();
    let mut sources: Vec<(Option<String>, String)> = Vec::new();
    for block in blocks {
        match block.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    answer_parts.push(text.to_string());
                }
            }
            Some("web_search_tool_result") => {
                if let Some(results) = block.get("content").and_then(|c| c.as_array()) {
                    for result in results {
                        let url = result.get("url").and_then(|u| u.as_str());
                        let title = result
                            .get("title")
                            .and_then(|t| t.as_str())
                            .map(str::to_string);
                        if let Some(url) = url {
                            sources.push((title, url.to_string()));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    sources.truncate(request.num_results);

    if answer_parts.is_empty() {
        return Err(anyhow!("returned no text answer"));
    }
    Ok(SearchOutput {
        answer: answer_parts.join("\n\n"),
        sources,
    })
}

async fn search_kimi(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    request: &SearchRequest,
) -> Result<SearchOutput> {
    let body = json!({
        "text_query": request.query,
        "limit": request.num_results,
        "enable_page_crawling": request.livecrawl_preferred,
        "timeout_seconds": 30,
    });

    let response: Value = client
        .post(format!(
            "{}/coding/v1/search",
            base_url.trim_end_matches('/')
        ))
        .bearer_auth(api_key)
        .json(&body)
        .timeout(Duration::from_secs(30))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    let results = response
        .get("search_results")
        .and_then(|r| r.as_array())
        .ok_or_else(|| anyhow!("response has no search_results"))?;
    if results.is_empty() {
        return Ok(SearchOutput {
            answer: "No results found.".to_string(),
            sources: Vec::new(),
        });
    }

    let mut rendered: Vec<String> = Vec::new();
    for result in results {
        let mut block = String::new();
        if let Some(title) = result.get("title").and_then(|t| t.as_str()) {
            block.push_str(&format!("Title: {}\n", title));
        }
        if let Some(date) = result.get("date").and_then(|d| d.as_str()) {
            block.push_str(&format!("Date: {}\n", date));
        }
        if let Some(url) = result.get("url").and_then(|u| u.as_str()) {
            block.push_str(&format!("URL: {}\n", url));
        }
        if let Some(snippet) = result.get("snippet").and_then(|s| s.as_str()) {
            block.push_str(&format!("{}\n", snippet));
        }
        if let Some(content) = result.get("content").and_then(|c| c.as_str()) {
            block.push_str(&format!("\n{}", content));
        }
        rendered.push(block.trim_end().to_string());
    }

    Ok(SearchOutput {
        answer: rendered.join("\n\n---\n\n"),
        sources: Vec::new(),
    })
}

async fn search_codex(
    client: &reqwest::Client,
    url: &str,
    access_token: &str,
    account_id: Option<&str>,
    model: &str,
    request: &SearchRequest,
) -> Result<SearchOutput> {
    let body = json!({
        "model": model,
        "instructions": "You are a web search assistant. Use the web_search tool to answer the user's query. Return a concise answer citing source URLs.",
        "input": [{
            "role": "user",
            "content": [{ "type": "input_text", "text": request.prompt() }],
        }],
        "tools": [{
            "type": "web_search",
            "external_web_access": true,
            "search_context_size": if request.deep { "high" } else { "medium" },
        }],
        "include": ["web_search_call.action.sources"],
        "store": false,
        "stream": true,
    });

    let mut http_request = client
        .post(url)
        .bearer_auth(access_token)
        .json(&body)
        .timeout(Duration::from_secs(60));
    if let Some(account_id) = account_id {
        http_request = http_request.header("chatgpt-account-id", account_id);
    }

    let body = http_request
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    parse_codex_sse(&body, request.num_results)
}

async fn search_openai(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    request: &SearchRequest,
) -> Result<SearchOutput> {
    let body = json!({
        "model": model,
        "instructions": "You are a web search assistant. Use the web_search tool to answer the user's query. Return a concise answer citing source URLs.",
        "input": [{
            "role": "user",
            "content": [{ "type": "input_text", "text": request.prompt() }],
        }],
        "tools": [{
            "type": "web_search",
            "search_context_size": if request.deep { "high" } else { "medium" },
        }],
        "include": ["web_search_call.action.sources"],
        "store": false,
        "stream": true,
    });

    let body = client
        .post(format!("{}/v1/responses", base_url.trim_end_matches('/')))
        .bearer_auth(api_key)
        .json(&body)
        .timeout(Duration::from_secs(60))
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    parse_codex_sse(&body, request.num_results)
}

fn collect_codex_output_item(
    item: &Value,
    answer_parts: &mut Vec<String>,
    sources: &mut Vec<(Option<String>, String)>,
) {
    match item.get("type").and_then(|t| t.as_str()) {
        Some("message") => {
            if let Some(content) = item.get("content").and_then(|c| c.as_array()) {
                for part in content {
                    if part.get("type").and_then(|t| t.as_str()) == Some("output_text") {
                        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                            answer_parts.push(text.to_string());
                        }
                    }
                }
            }
        }
        Some("web_search_call") => {
            if let Some(items) = item
                .get("action")
                .and_then(|a| a.get("sources"))
                .and_then(|s| s.as_array())
            {
                for source in items {
                    if let Some(url) = source.get("url").and_then(|u| u.as_str()) {
                        let title = source
                            .get("title")
                            .and_then(|t| t.as_str())
                            .map(str::to_string);
                        sources.push((title, url.to_string()));
                    }
                }
            }
        }
        _ => {}
    }
}

// The ChatGPT Codex endpoint streams real content through response.output_item.done
// events and leaves response.completed's output empty, so items are collected from
// both; output_item.done takes precedence.
fn parse_codex_sse(body: &str, num_results: usize) -> Result<SearchOutput> {
    let mut answer_parts: Vec<String> = Vec::new();
    let mut sources: Vec<(Option<String>, String)> = Vec::new();
    let mut completed = false;

    for line in body.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        let Ok(event) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        match event.get("type").and_then(|t| t.as_str()) {
            Some("response.output_item.done") => {
                if let Some(item) = event.get("item") {
                    collect_codex_output_item(item, &mut answer_parts, &mut sources);
                }
            }
            Some("response.completed") => {
                completed = true;
                if answer_parts.is_empty() {
                    if let Some(output) = event
                        .get("response")
                        .and_then(|r| r.get("output"))
                        .and_then(|o| o.as_array())
                    {
                        for item in output {
                            collect_codex_output_item(item, &mut answer_parts, &mut sources);
                        }
                    }
                }
            }
            Some("response.failed") => {
                let message = event
                    .pointer("/response/error/message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown error");
                return Err(anyhow!("response failed: {}", message));
            }
            _ => {}
        }
    }

    if !completed {
        return Err(anyhow!("stream ended without response.completed"));
    }
    sources.truncate(num_results);
    if answer_parts.is_empty() {
        return Err(anyhow!("returned no text answer"));
    }
    Ok(SearchOutput {
        answer: answer_parts.join("\n\n"),
        sources,
    })
}

fn decode_html_entities(text: &str) -> String {
    let named = [
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", "\""),
        ("&#39;", "'"),
        ("&#x27;", "'"),
        ("&apos;", "'"),
        ("&nbsp;", " "),
    ];
    let mut out = text.to_string();
    for (entity, replacement) in named {
        out = out.replace(entity, replacement);
    }
    out
}

fn html_to_text(html: &str) -> String {
    let script_or_style = regex::Regex::new(
        "(?is)<script[^>]*>.*?</script>|<style[^>]*>.*?</style>|<noscript[^>]*>.*?</noscript>",
    )
    .unwrap();

    let without_blocks = script_or_style.replace_all(html, " ");

    let anchors =
        regex::Regex::new("(?is)<a[^>]*href\\s*=\\s*[\"']([^\"']+)[\"'][^>]*>(.*?)</a>").unwrap();
    let with_links = anchors.replace_all(&without_blocks, "$2 ($1)");

    let block_breaks =
        regex::Regex::new("(?i)<\\s*(br|/p|/div|/li|/h[1-6]|/tr|/section|/article)\\s*/?\\s*>")
            .unwrap();
    let with_breaks = block_breaks.replace_all(&with_links, "\n");

    let tags = regex::Regex::new("<[^>]+>").unwrap();
    let stripped = tags.replace_all(&with_breaks, "");

    let decoded = decode_html_entities(&stripped);
    decoded
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{body_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn test_request() -> SearchRequest {
        SearchRequest {
            query: "rust async runtime".to_string(),
            num_results: 2,
            deep: false,
            livecrawl_preferred: false,
        }
    }

    #[test]
    fn parses_backend_order() {
        assert_eq!(
            parse_backend_order("kimi, deepseek,unknown,kimi"),
            vec![Backend::Kimi, Backend::DeepSeek]
        );
        assert!(parse_backend_order("").is_empty());
    }

    #[test]
    fn maps_provider_names_to_backends() {
        assert_eq!(
            Backend::from_provider_name("chatgpt_codex"),
            Some(Backend::Codex)
        );
        assert_eq!(Backend::from_provider_name("codex"), Some(Backend::Codex));
        assert_eq!(Backend::from_provider_name("openai"), Some(Backend::OpenAI));
        assert_eq!(
            Backend::from_provider_name("custom_deepseek"),
            Some(Backend::DeepSeek)
        );
        assert_eq!(Backend::from_provider_name("kimicode"), Some(Backend::Kimi));
        assert_eq!(Backend::from_provider_name("anthropic"), None);
    }

    #[test]
    fn renders_answer_with_sources() {
        let output = SearchOutput {
            answer: "Answer.".to_string(),
            sources: vec![(Some("Doc".to_string()), "https://example.com".to_string())],
        };
        let rendered = output.render();
        assert!(rendered.contains("Answer."));
        assert!(rendered.contains("1. Doc\n   https://example.com"));
    }

    #[test]
    fn converts_html_to_text() {
        let html = r#"<html><head><style>body{color:red}</style></head>
            <body><h1>Title</h1><p>Hello <a href="https://example.com">link</a> &amp; more</p>
            <script>alert(1)</script></body></html>"#;
        let text = html_to_text(html);
        assert!(text.contains("Title"));
        assert!(text.contains("Hello link (https://example.com) & more"));
        assert!(!text.contains("alert"));
        assert!(!text.contains("color:red"));
    }

    #[test]
    fn parses_codex_sse_completed() {
        let body = concat!(
            "data: {\"type\": \"response.output_text.delta\", \"delta\": \"partial\"}\n",
            "data: {\"type\": \"response.completed\", \"response\": {\"output\": [",
            "{\"type\": \"web_search_call\", \"action\": {\"sources\": [{\"url\": \"https://a.com\", \"title\": \"A\"}, {\"url\": \"https://b.com\"}]}},",
            "{\"type\": \"message\", \"content\": [{\"type\": \"output_text\", \"text\": \"final answer\"}]}",
            "]}}\n"
        );
        let output = parse_codex_sse(body, 1).unwrap();
        assert_eq!(output.answer, "final answer");
        assert_eq!(output.sources.len(), 1);
        assert_eq!(output.sources[0].1, "https://a.com");
    }

    #[test]
    fn parses_codex_sse_streamed_items_with_empty_completed_output() {
        // The real ChatGPT Codex endpoint delivers content via response.output_item.done
        // events and leaves response.completed's output empty.
        let body = concat!(
            "data: {\"type\": \"response.output_item.done\", \"item\": {\"type\": \"web_search_call\", \"action\": {\"sources\": [{\"url\": \"https://a.com\", \"title\": \"A\"}]}}}\n",
            "data: {\"type\": \"response.output_item.done\", \"item\": {\"type\": \"message\", \"content\": [{\"type\": \"output_text\", \"text\": \"streamed answer\"}]}}\n",
            "data: {\"type\": \"response.completed\", \"response\": {\"output\": []}}\n"
        );
        let output = parse_codex_sse(body, 8).unwrap();
        assert_eq!(output.answer, "streamed answer");
        assert_eq!(output.sources.len(), 1);
        assert_eq!(output.sources[0].1, "https://a.com");
    }

    #[test]
    fn parses_codex_sse_failed() {
        let body = "data: {\"type\": \"response.failed\", \"response\": {\"error\": {\"message\": \"boom\"}}}\n";
        assert!(parse_codex_sse(body, 8)
            .unwrap_err()
            .to_string()
            .contains("boom"));
    }

    #[tokio::test]
    async fn deepseek_request_shape_and_sources() {
        let server = MockServer::start().await;
        let expected = json!({
            "model": "deepseek-chat",
            "max_tokens": 1024,
            "tools": [{ "type": "web_search_20250305", "name": "web_search", "max_uses": 1 }],
            "messages": [{ "role": "user", "content": test_request().prompt() }],
        });

        Mock::given(method("POST"))
            .and(path("/anthropic/v1/messages"))
            .and(header("x-api-key", "test-key"))
            .and(header("anthropic-version", "2023-06-01"))
            .and(body_json(&expected))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "content": [
                    { "type": "web_search_tool_result", "content": [
                        { "title": "Tokio", "url": "https://tokio.rs" },
                        { "title": "async-std", "url": "https://async.rs" },
                        { "title": "smol", "url": "https://smol.rs" }
                    ]},
                    { "type": "text", "text": "Use tokio." }
                ]
            })))
            .mount(&server)
            .await;

        let output = search_deepseek(
            &reqwest::Client::new(),
            &server.uri(),
            "test-key",
            &test_request(),
        )
        .await
        .unwrap();
        assert_eq!(output.answer, "Use tokio.");
        assert_eq!(output.sources.len(), 2);
        let rendered = output.render();
        assert!(rendered.contains("Sources:"));
        assert!(rendered.contains("https://tokio.rs"));
    }

    #[tokio::test]
    async fn kimi_request_shape() {
        let server = MockServer::start().await;
        let expected = json!({
            "text_query": "rust async runtime",
            "limit": 2,
            "enable_page_crawling": false,
            "timeout_seconds": 30,
        });

        Mock::given(method("POST"))
            .and(path("/coding/v1/search"))
            .and(header("authorization", "Bearer test-key"))
            .and(body_json(&expected))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "search_results": [
                    { "title": "Tokio", "url": "https://tokio.rs", "snippet": "A runtime." }
                ]
            })))
            .mount(&server)
            .await;

        let output = search_kimi(
            &reqwest::Client::new(),
            &server.uri(),
            "test-key",
            &test_request(),
        )
        .await
        .unwrap();
        assert!(output.answer.contains("Title: Tokio"));
        assert!(output.answer.contains("URL: https://tokio.rs"));
        assert!(output.answer.contains("A runtime."));
    }

    #[tokio::test]
    async fn codex_request_shape() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .and(header("authorization", "Bearer codex-token"))
            .and(header("chatgpt-account-id", "acc-1"))
            .respond_with(ResponseTemplate::new(200).set_body_string(concat!(
                "data: {\"type\": \"response.completed\", \"response\": {\"output\": [",
                "{\"type\": \"message\", \"content\": [{\"type\": \"output_text\", \"text\": \"codex answer\"}]}",
                "]}}\n"
            )))
            .mount(&server)
            .await;

        let url = format!("{}/responses", server.uri());
        let output = search_codex(
            &reqwest::Client::new(),
            &url,
            "codex-token",
            Some("acc-1"),
            "gpt-5.5",
            &test_request(),
        )
        .await
        .unwrap();
        assert_eq!(output.answer, "codex answer");
    }

    #[tokio::test]
    async fn openai_request_shape() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_string(concat!(
                "data: {\"type\": \"response.completed\", \"response\": {\"output\": [",
                "{\"type\": \"message\", \"content\": [{\"type\": \"output_text\", \"text\": \"openai answer\"}]}",
                "]}}\n"
            )))
            .mount(&server)
            .await;

        let output = search_openai(
            &reqwest::Client::new(),
            &server.uri(),
            "test-key",
            "gpt-5-mini",
            &test_request(),
        )
        .await
        .unwrap();
        assert_eq!(output.answer, "openai answer");
    }

    #[tokio::test]
    async fn backend_http_error_is_returned() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let result = search_kimi(
            &reqwest::Client::new(),
            &server.uri(),
            "test-key",
            &test_request(),
        )
        .await;
        assert!(result.is_err());
    }
}
