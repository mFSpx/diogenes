use crate::error::ApiError;
use crate::providers::claw_provider::{self, AuthSource, ClawApiClient};
use crate::providers::openai_compat::{self, OpenAiCompatClient, OpenAiCompatConfig};
use crate::providers::{self, Provider, ProviderKind};
use crate::types::{MessageRequest, MessageResponse, StreamEvent};

async fn send_via_provider<P: Provider>(
    provider: &P,
    request: &MessageRequest,
) -> Result<MessageResponse, ApiError> {
    provider.send_message(request).await
}

async fn stream_via_provider<P: Provider>(
    provider: &P,
    request: &MessageRequest,
) -> Result<P::Stream, ApiError> {
    provider.stream_message(request).await
}

#[derive(Debug, Clone)]
pub enum ProviderClient {
    ClawApi(ClawApiClient),
    Xai(OpenAiCompatClient),
    OpenAi(OpenAiCompatClient),
    Groq(OpenAiCompatClient),
    Cohere(OpenAiCompatClient),
    Cerebras(OpenAiCompatClient),
    LucidotaLocal(OpenAiCompatClient),
}

impl ProviderClient {
    pub fn from_model(model: &str) -> Result<Self, ApiError> {
        Self::from_model_with_default_auth(model, None)
    }

    pub fn from_model_with_default_auth(
        model: &str,
        default_auth: Option<AuthSource>,
    ) -> Result<Self, ApiError> {
        let resolved_model = providers::resolve_model_alias(model);
        match providers::detect_provider_kind(&resolved_model) {
            ProviderKind::ClawApi => Ok(Self::ClawApi(match default_auth {
                Some(auth) => ClawApiClient::from_auth(auth),
                None => ClawApiClient::from_env()?,
            })),
            ProviderKind::Xai => Ok(Self::Xai(OpenAiCompatClient::from_env(
                OpenAiCompatConfig::xai(),
            )?)),
            ProviderKind::OpenAi => Ok(Self::OpenAi(OpenAiCompatClient::from_env(
                OpenAiCompatConfig::openai(),
            )?)),
            ProviderKind::Groq => Ok(Self::Groq(OpenAiCompatClient::from_env(
                OpenAiCompatConfig::groq(),
            )?)),
            ProviderKind::Cohere => Ok(Self::Cohere(OpenAiCompatClient::from_env(
                OpenAiCompatConfig::cohere(),
            )?)),
            ProviderKind::Cerebras => Ok(Self::Cerebras(OpenAiCompatClient::from_env(
                OpenAiCompatConfig::cerebras(),
            )?)),
            ProviderKind::LucidotaLocal => Ok(Self::LucidotaLocal(
                OpenAiCompatClient::new(
                    std::env::var("LUCIDOTA_LOCAL_API_KEY").unwrap_or_else(|_| "local".to_string()),
                    OpenAiCompatConfig::lucidota_local(),
                )
                .with_base_url(local_base_url_for_model(&resolved_model)),
            )),
        }
    }

    #[must_use]
    pub const fn provider_kind(&self) -> ProviderKind {
        match self {
            Self::ClawApi(_) => ProviderKind::ClawApi,
            Self::Xai(_) => ProviderKind::Xai,
            Self::OpenAi(_) => ProviderKind::OpenAi,
            Self::Groq(_) => ProviderKind::Groq,
            Self::Cohere(_) => ProviderKind::Cohere,
            Self::Cerebras(_) => ProviderKind::Cerebras,
            Self::LucidotaLocal(_) => ProviderKind::LucidotaLocal,
        }
    }

    pub async fn send_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageResponse, ApiError> {
        match self {
            Self::ClawApi(client) => send_via_provider(client, request).await,
            Self::Xai(client)
            | Self::OpenAi(client)
            | Self::Groq(client)
            | Self::Cohere(client)
            | Self::Cerebras(client)
            | Self::LucidotaLocal(client) => send_via_provider(client, request).await,
        }
    }

    pub async fn stream_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageStream, ApiError> {
        match self {
            Self::ClawApi(client) => stream_via_provider(client, request)
                .await
                .map(MessageStream::ClawApi),
            Self::Xai(client)
            | Self::OpenAi(client)
            | Self::Groq(client)
            | Self::Cohere(client)
            | Self::Cerebras(client)
            | Self::LucidotaLocal(client) => stream_via_provider(client, request)
                .await
                .map(MessageStream::OpenAiCompat),
        }
    }
}

fn local_base_url_for_model(model: &str) -> String {
    let key = model.to_ascii_lowercase();
    if key == "luci" || key == "mamba7b-ram" {
        std::env::var("LUCIDOTA_MAMBA_RAM_BASE_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8081/v1".to_string())
    } else if key == "mamba7b-gpu" {
        std::env::var("LUCIDOTA_MAMBA_GPU_BASE_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8083/v1".to_string())
    } else {
        std::env::var("LUCIDOTA_LOCAL_BASE_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8080/v1".to_string())
    }
}

#[derive(Debug)]
pub enum MessageStream {
    ClawApi(claw_provider::MessageStream),
    OpenAiCompat(openai_compat::MessageStream),
}

impl MessageStream {
    #[must_use]
    pub fn request_id(&self) -> Option<&str> {
        match self {
            Self::ClawApi(stream) => stream.request_id(),
            Self::OpenAiCompat(stream) => stream.request_id(),
        }
    }

    pub async fn next_event(&mut self) -> Result<Option<StreamEvent>, ApiError> {
        match self {
            Self::ClawApi(stream) => stream.next_event().await,
            Self::OpenAiCompat(stream) => stream.next_event().await,
        }
    }
}

pub use claw_provider::{
    oauth_token_is_expired, resolve_saved_oauth_token, resolve_startup_auth_source, OAuthTokenSet,
};
#[must_use]
pub fn read_base_url() -> String {
    claw_provider::read_base_url()
}

#[must_use]
pub fn read_xai_base_url() -> String {
    openai_compat::read_base_url(OpenAiCompatConfig::xai())
}

#[cfg(test)]
mod tests {
    use crate::providers::{detect_provider_kind, resolve_model_alias, ProviderKind};

    #[test]
    fn resolves_existing_and_grok_aliases() {
        assert_eq!(resolve_model_alias("opus"), "claude-opus-4-6");
        assert_eq!(resolve_model_alias("grok"), "grok-3");
        assert_eq!(resolve_model_alias("grok-mini"), "grok-3-mini");
    }

    #[test]
    fn provider_detection_prefers_model_family() {
        assert_eq!(detect_provider_kind("grok-3"), ProviderKind::Xai);
        assert_eq!(
            detect_provider_kind("claude-sonnet-4-6"),
            ProviderKind::ClawApi
        );
    }
}
