use crate::conversation::token_usage::{CostSource, ProviderUsage};
use crate::http_status::read_json_response;
use crate::images::ImageFormat;
use anyhow::Error;
use async_stream::try_stream;
use futures::TryStreamExt;
use reqwest::{Response, StatusCode};
use serde_json::Value;
use tokio::pin;
use tokio_stream::StreamExt;
use tokio_util::codec::{FramedRead, LinesCodec};
use tokio_util::io::StreamReader;

use super::api_client::ApiClient;
use super::base::{
    model_info_for_provider_model, stream_from_single_message, MessageStream, ModelInfo, Provider,
};
use super::retry::ProviderRetry;
use crate::conversation::message::Message;
use crate::errors::ProviderError;
use crate::formats::openai::{
    create_request, create_request_for_model_with_options, get_cost, get_usage,
    record_response_metadata, response_to_message, response_to_streaming_message,
    OpenAiFormatOptions,
};
use crate::formats::openai_responses::responses_api_to_streaming_message;
use crate::model::ModelConfig;
use crate::request_log::{start_log, LoggerHandleExt, RequestLogHandle};
use rmcp::model::Tool;

pub struct OpenAiCompatibleProvider {
    name: String,
    /// Client targeted at the base URL (e.g. `https://api.x.ai/v1`)
    api_client: ApiClient,
    /// Path prefix prepended to `chat/completions` (e.g. `"deployments/{name}/"` for Azure).
    completions_prefix: String,
    supports_streaming: bool,
    metadata_provider: String,
    model_info_cache:
        std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, ModelInfo>>>,
}

impl OpenAiCompatibleProvider {
    pub fn new(name: String, api_client: ApiClient, completions_prefix: String) -> Self {
        Self {
            metadata_provider: name.clone(),
            name,
            api_client,
            completions_prefix,
            supports_streaming: true,
            model_info_cache: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
        }
    }

    pub fn with_supports_streaming(mut self, supports_streaming: bool) -> Self {
        self.supports_streaming = supports_streaming;
        self
    }

    pub fn with_metadata_provider(mut self, provider: impl Into<String>) -> Self {
        self.metadata_provider = provider.into();
        self
    }

    #[allow(clippy::too_many_arguments)]
    fn build_request_for_model(
        &self,
        model_config: &ModelConfig,
        wire_model: &str,
        capability_model: &str,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
        for_streaming: bool,
    ) -> Result<Value, ProviderError> {
        create_request_for_model_with_options(
            model_config,
            wire_model,
            capability_model,
            system,
            messages,
            tools,
            &ImageFormat::OpenAi,
            for_streaming,
            OpenAiFormatOptions {
                preserve_thinking_context: true,
                supports_vision: model_config.supports_vision.unwrap_or_default(),
                ..Default::default()
            },
        )
        .map_err(|e| ProviderError::RequestFailed(format!("Failed to create request: {}", e)))
    }

    pub async fn stream_for_model(
        &self,
        model_config: &ModelConfig,
        wire_model: &str,
        capability_model: &str,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        let payload = self.build_request_for_model(
            model_config,
            wire_model,
            capability_model,
            system,
            messages,
            tools,
            self.supports_streaming,
        )?;
        self.stream_payload(model_config, payload).await
    }

    async fn stream_payload(
        &self,
        model_config: &ModelConfig,
        payload: Value,
    ) -> Result<MessageStream, ProviderError> {
        let mut log = start_log(model_config, &payload)?;
        let path = format!("{}chat/completions", self.completions_prefix);
        let response = self
            .with_retry(|| async {
                handle_status(
                    self.api_client
                        .request(&path)
                        .model_headers(model_config)?
                        .streaming(self.supports_streaming)
                        .response_post(&payload)
                        .await?,
                )
                .await
            })
            .await
            .inspect_err(|e| {
                let _ = log.error(e);
            })?;
        if self.supports_streaming {
            stream_openai_compat(response, log)
        } else {
            let json = read_json_response(response).await?;
            let message = response_to_message(&json).map_err(|e| {
                ProviderError::RequestFailed(format!("Failed to parse message: {}", e))
            })?;
            let usage_json = json.get("usage").unwrap_or(&Value::Null);
            let usage_data = get_usage(usage_json);
            let mut usage = ProviderUsage::new(model_config.model_name.clone(), usage_data);
            record_response_metadata(&mut usage, &json);
            if let Some(cost) = get_cost(usage_json) {
                usage = usage.with_cost(cost, CostSource::ProviderReported);
            }
            log.write(
                &serde_json::to_value(&message).unwrap_or_default(),
                Some(&usage.usage),
            )?;
            Ok(stream_from_single_message(message, usage))
        }
    }

    fn build_request(
        &self,
        model_config: &ModelConfig,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
        for_streaming: bool,
    ) -> Result<Value, ProviderError> {
        create_request(
            model_config,
            system,
            messages,
            tools,
            &ImageFormat::OpenAi,
            for_streaming,
        )
        .map_err(|e| ProviderError::RequestFailed(format!("Failed to create request: {}", e)))
    }
}

#[async_trait::async_trait]
impl Provider for OpenAiCompatibleProvider {
    fn get_name(&self) -> &str {
        &self.name
    }

    fn canonical_provider_name(&self) -> &str {
        &self.metadata_provider
    }

    async fn refresh_credentials(&self) -> Result<(), ProviderError> {
        self.api_client
            .refresh_credentials()
            .await
            .map_err(|error| ProviderError::Authentication(error.to_string()))
    }

    async fn get_context_limit(&self, model: &str, override_limit: Option<usize>) -> usize {
        let resolver = crate::context_limit::ContextLimitResolver::new(&self.metadata_provider);
        resolver
            .resolve(model, override_limit, || async {
                if let Some(limit) = self
                    .cached_model_info(model)
                    .and_then(|info| info.context_limit)
                {
                    return Ok(Some(limit));
                }

                let paths = model_endpoint_paths(&self.api_client, "models");
                let Some(discovered) =
                    fetch_openai_compatible_model_info_for_model(&self.api_client, &paths, model)
                        .await?
                else {
                    return Ok(None);
                };
                let info = discovered.into_model_info(&self.metadata_provider);
                let context_limit = info.context_limit;
                self.cache_model_info(std::slice::from_ref(&info));
                if info.name != model {
                    let mut alias = info;
                    alias.name = model.to_string();
                    self.cache_model_info(std::slice::from_ref(&alias));
                }
                Ok(context_limit)
            })
            .await
    }

    async fn fetch_supported_models(&self) -> Result<Vec<String>, ProviderError> {
        Ok(self
            .fetch_supported_model_info()
            .await?
            .into_iter()
            .map(|model| model.name)
            .collect())
    }

    async fn fetch_supported_model_info(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        let paths = model_endpoint_paths(&self.api_client, "models");
        let models = fetch_openai_compatible_model_info(&self.api_client, &paths)
            .await?
            .into_iter()
            .map(|model| model.into_model_info(&self.metadata_provider))
            .collect::<Vec<_>>();
        self.cache_model_info(&models);
        Ok(models)
    }

    async fn fetch_model_info(&self, model_name: &str) -> Result<ModelInfo, ProviderError> {
        if let Some(info) = self.cached_model_info(model_name) {
            return Ok(info);
        }
        let info = model_info_for_provider_model(&self.metadata_provider, model_name);
        self.cache_model_info(std::slice::from_ref(&info));
        Ok(info)
    }

    async fn stream(
        &self,
        model_config: &ModelConfig,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        let payload = self.build_request(
            model_config,
            system,
            messages,
            tools,
            self.supports_streaming,
        )?;
        self.stream_payload(model_config, payload).await
    }
}

impl OpenAiCompatibleProvider {
    fn cache_model_info(&self, models: &[ModelInfo]) {
        if let Ok(mut cache) = self.model_info_cache.lock() {
            for model in models {
                cache.insert(model.name.clone(), model.clone());
            }
        }
    }

    fn cached_model_info(&self, model_name: &str) -> Option<ModelInfo> {
        self.model_info_cache.lock().ok().and_then(|cache| {
            cache.get(model_name).cloned().or_else(|| {
                cache
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case(model_name))
                    .map(|(_, model)| model.clone())
            })
        })
    }
}

pub(crate) fn model_endpoint_paths(api_client: &ApiClient, primary_path: &str) -> Vec<String> {
    let primary_path = normalize_model_endpoint_path(primary_path);
    let mut paths = vec![primary_path.clone()];
    let alternate = if primary_path == "models" {
        let host_path = url::Url::parse(api_client.host())
            .ok()
            .map(|url| url.path().trim_end_matches('/').to_string())
            .unwrap_or_default();
        if host_path.ends_with("/v1") {
            let parent = host_path.strip_suffix("/v1").unwrap_or_default();
            Some(if parent.is_empty() {
                "/models".to_string()
            } else {
                format!("{parent}/models")
            })
        } else {
            Some("v1/models".to_string())
        }
    } else {
        toggle_v1_model_endpoint_path(&primary_path)
    };
    if let Some(alternate) = alternate {
        let alternate = normalize_model_endpoint_path(&alternate);
        if !paths.contains(&alternate) {
            paths.push(alternate);
        }
    }
    paths
}

fn normalize_model_endpoint_path(path: &str) -> String {
    let path = path.trim();
    if path.is_empty() {
        return "models".to_string();
    }
    if path.starts_with('/') {
        format!("/{}", path.trim_matches('/'))
    } else {
        path.trim_matches('/').to_string()
    }
}

fn toggle_v1_model_endpoint_path(path: &str) -> Option<String> {
    let absolute = path.starts_with('/');
    let path = path.trim_matches('/');
    let toggled = if path == "v1/models" {
        "models".to_string()
    } else if let Some(prefix) = path.strip_suffix("/v1/models") {
        format!("{prefix}/models")
    } else if path == "models" {
        "v1/models".to_string()
    } else if let Some(prefix) = path.strip_suffix("/models") {
        format!("{prefix}/v1/models")
    } else {
        return None;
    };
    Some(if absolute {
        format!("/{toggled}")
    } else {
        toggled
    })
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DiscoveredModel {
    pub name: String,
    pub context_limit: Option<usize>,
    pub reasoning: Option<bool>,
}

impl DiscoveredModel {
    pub(crate) fn into_model_info(self, provider_name: &str) -> ModelInfo {
        let mut info = model_info_for_provider_model(provider_name, &self.name);
        if let Some(context_limit) = self.context_limit {
            info.context_limit = Some(context_limit);
        }
        if let Some(reasoning) = self.reasoning {
            info.reasoning = reasoning;
        }
        info
    }
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) async fn fetch_openai_compatible_models(
    api_client: &ApiClient,
    paths: &[String],
) -> Result<Vec<String>, ProviderError> {
    Ok(fetch_openai_compatible_model_info(api_client, paths)
        .await?
        .into_iter()
        .map(|model| model.name)
        .collect())
}
pub(crate) async fn fetch_openai_compatible_model_info(
    api_client: &ApiClient,
    paths: &[String],
) -> Result<Vec<DiscoveredModel>, ProviderError> {
    if paths.is_empty() {
        return Err(ProviderError::EndpointNotFound(
            "No model discovery endpoint configured".to_string(),
        ));
    }

    let mut invalid_json_error = None;
    let mut parse_error = None;

    for path in paths {
        let response = api_client
            .response_get(path)
            .await
            .map_err(ProviderError::from)?;

        if matches!(
            response.status(),
            StatusCode::NOT_FOUND | StatusCode::METHOD_NOT_ALLOWED
        ) {
            drop(response);
            continue;
        }

        let response = handle_status(response).await?;
        let json = match read_json_response::<Value>(response).await {
            Ok(json) => json,
            Err(error) if error.to_string().contains("not valid JSON") => {
                invalid_json_error = Some(error.to_string());
                continue;
            }
            Err(error) => return Err(error),
        };

        if let Some(err_obj) = json.get("error").filter(|error| !error.is_null()) {
            let message = err_obj
                .get("message")
                .and_then(Value::as_str)
                .or_else(|| err_obj.as_str())
                .unwrap_or("unknown error");
            return Err(ProviderError::Authentication(message.to_string()));
        }

        match parse_model_info(&json) {
            Ok(models) if !models.is_empty() => return Ok(models),
            Ok(_) => {}
            Err(error) => parse_error = Some(error),
        }
    }

    if let Some(error) = parse_error {
        return Err(error);
    }
    Err(ProviderError::EndpointNotFound(
        invalid_json_error.unwrap_or_else(|| "Models endpoint is not available".to_string()),
    ))
}

pub(crate) async fn fetch_openai_compatible_model_info_for_model(
    api_client: &ApiClient,
    paths: &[String],
    model_name: &str,
) -> Result<Option<DiscoveredModel>, ProviderError> {
    let models = fetch_openai_compatible_model_info(api_client, paths).await?;
    Ok(models
        .iter()
        .find(|model| model.name == model_name)
        .or_else(|| {
            models
                .iter()
                .find(|model| model.name.eq_ignore_ascii_case(model_name))
        })
        .or_else(|| (models.len() == 1).then_some(&models[0]))
        .cloned())
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn parse_model_ids(json: &Value) -> Result<Vec<String>, ProviderError> {
    Ok(parse_model_info(json)?
        .into_iter()
        .map(|model| model.name)
        .collect())
}

pub(crate) fn parse_model_info(json: &Value) -> Result<Vec<DiscoveredModel>, ProviderError> {
    let models = json
        .as_array()
        .or_else(|| json.get("data").and_then(Value::as_array))
        .or_else(|| json.get("models").and_then(Value::as_array))
        .ok_or_else(|| {
            ProviderError::RequestFailed("Missing models array in JSON response".into())
        })?;

    let mut discovered: Vec<DiscoveredModel> = Vec::new();
    for model in models {
        let (name, context_limit, reasoning) = if let Some(name) = model.as_str() {
            (name, None, None)
        } else {
            let Some(name) = ["id", "model", "name"]
                .iter()
                .find_map(|field| model.get(*field).and_then(Value::as_str))
            else {
                continue;
            };
            (name, model_context_limit(model), model_reasoning(model))
        };
        let Some(name) = non_empty_model_id(name) else {
            continue;
        };

        if let Some(existing) = discovered.iter_mut().find(|model| model.name == name) {
            if existing.context_limit.is_none() {
                existing.context_limit = context_limit;
            }
            if existing.reasoning.is_none() {
                existing.reasoning = reasoning;
            }
        } else {
            discovered.push(DiscoveredModel {
                name,
                context_limit,
                reasoning,
            });
        }
    }

    discovered.sort_by(|left, right| left.name.cmp(&right.name));
    if discovered.is_empty() {
        return Err(ProviderError::RequestFailed(
            "Models response did not contain any model ids".into(),
        ));
    }
    Ok(discovered)
}

fn model_context_limit(model: &Value) -> Option<usize> {
    [
        model.get("context_length"),
        model.get("context_window"),
        model.get("max_context_length"),
        model.get("max_model_len"),
        model.get("n_ctx"),
        model.get("context"),
        model
            .get("meta")
            .and_then(|meta| meta.get("context_length")),
        model
            .get("meta")
            .and_then(|meta| meta.get("context_window")),
        model.get("meta").and_then(|meta| meta.get("max_model_len")),
        model.get("meta").and_then(|meta| meta.get("n_ctx")),
    ]
    .into_iter()
    .flatten()
    .filter_map(Value::as_u64)
    .filter(|limit| *limit > 0)
    .find_map(|limit| usize::try_from(limit).ok())
}

fn model_reasoning(model: &Value) -> Option<bool> {
    [
        model.get("reasoning"),
        model.get("supports_reasoning"),
        model
            .get("capabilities")
            .and_then(|capabilities| capabilities.get("reasoning")),
        model.get("meta").and_then(|meta| meta.get("reasoning")),
    ]
    .into_iter()
    .flatten()
    .find_map(Value::as_bool)
}

fn non_empty_model_id(model_id: &str) -> Option<String> {
    let model_id = model_id.trim();
    (!model_id.is_empty()).then(|| model_id.to_string())
}

// Re-exported from the dedicated `http_status` module — these helpers are
// format-agnostic and used across all provider families.
pub use super::http_status::{
    handle_response, handle_status, map_http_error_to_provider_error, sanitize_url,
};

// Legacy alias kept for callers that haven't migrated their import path yet.
pub use super::http_status::handle_response as handle_response_openai_compat;

pub fn stream_openai_compat(
    response: Response,
    mut log: Option<Box<dyn RequestLogHandle>>,
) -> Result<MessageStream, ProviderError> {
    let stream = response.bytes_stream().map_err(std::io::Error::other);

    Ok(Box::pin(try_stream! {
        let stream_reader = StreamReader::new(stream);
        let framed = FramedRead::new(stream_reader, LinesCodec::new())
            .map_err(Error::from);

        let message_stream = response_to_streaming_message(framed);
        pin!(message_stream);
        while let Some(message) = message_stream.next().await {
            let (message, usage) = message.map_err(|e|
                e.downcast::<ProviderError>()
                    .unwrap_or_else(ProviderError::stream_decode_error)
            )?;
            log.write(&message, usage.as_ref().map(|f| f.usage).as_ref())?;
            yield (message, usage);
        }
    }))
}

pub fn stream_responses_compat(
    response: Response,
    mut log: Option<Box<dyn RequestLogHandle>>,
) -> Result<MessageStream, ProviderError> {
    let stream = response.bytes_stream().map_err(std::io::Error::other);

    Ok(Box::pin(try_stream! {
        let stream_reader = StreamReader::new(stream);
        let framed = FramedRead::new(stream_reader, LinesCodec::new())
            .map_err(Error::from);

        let message_stream = responses_api_to_streaming_message(framed);
        pin!(message_stream);
        while let Some(message) = message_stream.next().await {
            let (message, usage) = message.map_err(|e|
                e.downcast::<ProviderError>()
                    .unwrap_or_else(ProviderError::stream_decode_error)
            )?;
            log.write(&message, usage.as_ref().map(|f| f.usage).as_ref())?;
            yield (message, usage);
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelConfig;
    use serde_json::json;
    use test_case::test_case;

    #[test_case(
        StatusCode::PAYMENT_REQUIRED,
        Some(json!({"error": {"message": "Insufficient credits to complete this request"}})),
        "CreditsExhausted"
        ; "402 with payload"
    )]
    #[test_case(
        StatusCode::PAYMENT_REQUIRED,
        None,
        "CreditsExhausted"
        ; "402 without payload"
    )]
    #[test_case(
        StatusCode::TOO_MANY_REQUESTS,
        Some(json!({"error": {"message": "Rate limit exceeded"}})),
        "RateLimitExceeded"
        ; "429 rate limit"
    )]
    #[test_case(
        StatusCode::UNAUTHORIZED,
        None,
        "Authentication"
        ; "401 unauthorized"
    )]
    #[test_case(
        StatusCode::BAD_REQUEST,
        Some(json!({"error": {"message": "This request exceeds the maximum context length"}})),
        "ContextLengthExceeded"
        ; "400 context length"
    )]
    #[test_case(
        StatusCode::INTERNAL_SERVER_ERROR,
        None,
        "ServerError"
        ; "500 server error"
    )]
    #[test_case(
        StatusCode::NOT_FOUND,
        None,
        "RequestFailed"
        ; "404 not found"
    )]
    #[test_case(
        StatusCode::NOT_FOUND,
        Some(json!({"error": {"message": "model not available"}})),
        "RequestFailed"
        ; "404 with error payload"
    )]
    fn http_status_maps_to_expected_error(
        status: StatusCode,
        payload: Option<Value>,
        expected_variant: &str,
    ) {
        let err = map_http_error_to_provider_error(status, payload, "http://test/endpoint");
        let actual = err.telemetry_type();
        let expected_telemetry = match expected_variant {
            "CreditsExhausted" => "credits_exhausted",
            "RateLimitExceeded" => "rate_limit",
            "Authentication" => "auth",
            "ContextLengthExceeded" => "context_length",
            "ServerError" => "server",
            "RequestFailed" => "request",
            other => panic!("Unknown variant: {other}"),
        };
        assert_eq!(
            actual, expected_telemetry,
            "Expected {expected_variant}, got error: {err:?}"
        );
    }

    #[test]
    fn model_endpoint_paths_try_root_and_versioned_forms() {
        let client = ApiClient::new_with_tls(
            "http://localhost".to_string(),
            super::super::api_client::AuthMethod::NoAuth,
            None,
        )
        .unwrap();
        assert_eq!(
            model_endpoint_paths(&client, "models"),
            ["models", "v1/models"]
        );
    }

    #[test]
    fn model_endpoint_paths_try_root_for_v1_host() {
        let client = ApiClient::new_with_tls(
            "http://localhost/v1".to_string(),
            super::super::api_client::AuthMethod::NoAuth,
            None,
        )
        .unwrap();
        assert_eq!(
            model_endpoint_paths(&client, "models"),
            ["models", "/models"]
        );
    }

    #[test]
    fn parse_model_ids_accepts_common_openai_compatible_shapes() {
        let response = json!({
            "models": [
                " model-z ",
                {"model": "model-y"},
                {"name": "model-x"},
                {"id": "model-y"},
                {"id": ""},
                {"id": 42}
            ]
        });

        assert_eq!(
            parse_model_ids(&response).unwrap(),
            ["model-x", "model-y", "model-z"]
        );
    }

    #[tokio::test]
    async fn fetch_supported_models_falls_back_to_versioned_endpoint() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                {"name": "model-from-versioned-endpoint"}
            ])))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiCompatibleProvider::new(
            "test".to_string(),
            ApiClient::new_with_tls(server.uri(), crate::api_client::AuthMethod::NoAuth, None)
                .unwrap(),
            String::new(),
        );

        assert_eq!(
            provider.fetch_supported_models().await.unwrap(),
            ["model-from-versioned-endpoint"]
        );
    }

    #[test]
    fn build_request_respects_non_streaming_mode() {
        let provider = OpenAiCompatibleProvider::new(
            "test".to_string(),
            ApiClient::new_with_tls(
                "http://localhost".to_string(),
                super::super::api_client::AuthMethod::NoAuth,
                None,
            )
            .unwrap(),
            String::new(),
        )
        .with_supports_streaming(false);

        let model = ModelConfig::new("test-model");
        let payload = provider
            .build_request(&model, "", &[], &[], provider.supports_streaming)
            .unwrap();

        assert_eq!(payload.get("stream"), None);
        assert_eq!(payload.get("stream_options"), None);
    }

    #[tokio::test]
    async fn nonstreaming_completion_accepts_legitimate_response() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "hello"}
                }]
            })))
            .mount(&server)
            .await;

        let provider = OpenAiCompatibleProvider::new(
            "test".to_string(),
            ApiClient::new_with_tls(server.uri(), crate::api_client::AuthMethod::NoAuth, None)
                .unwrap(),
            String::new(),
        )
        .with_supports_streaming(false);

        let _stream = provider
            .stream(&ModelConfig::new("test-model"), "", &[], &[])
            .await
            .expect("legitimate non-streaming response should be accepted");
    }

    #[tokio::test]
    async fn nonstreaming_completion_rejects_oversized_response_body() {
        use crate::http_status::MAX_PROVIDER_JSON_RESPONSE_BYTES;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "a".repeat(MAX_PROVIDER_JSON_RESPONSE_BYTES + 1)
                    }
                }]
            })))
            .mount(&server)
            .await;

        let provider = OpenAiCompatibleProvider::new(
            "test".to_string(),
            ApiClient::new_with_tls(server.uri(), crate::api_client::AuthMethod::NoAuth, None)
                .unwrap(),
            String::new(),
        )
        .with_supports_streaming(false);

        let err = match provider
            .stream(&ModelConfig::new("test-model"), "", &[], &[])
            .await
        {
            Ok(_) => panic!("oversized response should be rejected"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("response body exceeds"),
            "got: {err}"
        );
    }
}
