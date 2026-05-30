use std::future::Future;
use std::pin::Pin;

use crate::error::ApiError;
use crate::types::{MessageRequest, MessageResponse};

pub mod claw_provider;
pub mod openai_compat;

pub type ProviderFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, ApiError>> + Send + 'a>>;

pub trait Provider {
    type Stream;

    fn send_message<'a>(
        &'a self,
        request: &'a MessageRequest,
    ) -> ProviderFuture<'a, MessageResponse>;

    fn stream_message<'a>(
        &'a self,
        request: &'a MessageRequest,
    ) -> ProviderFuture<'a, Self::Stream>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    ClawApi,
    Xai,
    OpenAi,
    Groq,
    Cohere,
    Cerebras,
    LucidotaLocal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderMetadata {
    pub provider: ProviderKind,
    pub auth_env: &'static str,
    pub base_url_env: &'static str,
    pub default_base_url: &'static str,
}

const MODEL_REGISTRY: &[(&str, ProviderMetadata)] = &[
    (
        "luci",
        ProviderMetadata {
            provider: ProviderKind::LucidotaLocal,
            auth_env: "LUCIDOTA_LOCAL_API_KEY",
            base_url_env: "LUCIDOTA_LOCAL_BASE_URL",
            default_base_url: openai_compat::DEFAULT_LUCIDOTA_LOCAL_BASE_URL,
        },
    ),
    (
        "diogenes-go-local",
        ProviderMetadata {
            provider: ProviderKind::LucidotaLocal,
            auth_env: "LUCIDOTA_LOCAL_API_KEY",
            base_url_env: "LUCIDOTA_LOCAL_BASE_URL",
            default_base_url: openai_compat::DEFAULT_LUCIDOTA_LOCAL_BASE_URL,
        },
    ),
    (
        "deepseek-local",
        ProviderMetadata {
            provider: ProviderKind::LucidotaLocal,
            auth_env: "LUCIDOTA_LOCAL_API_KEY",
            base_url_env: "LUCIDOTA_LOCAL_BASE_URL",
            default_base_url: openai_compat::DEFAULT_LUCIDOTA_LOCAL_BASE_URL,
        },
    ),
    (
        "mamba7b-ram",
        ProviderMetadata {
            provider: ProviderKind::LucidotaLocal,
            auth_env: "LUCIDOTA_LOCAL_API_KEY",
            base_url_env: "LUCIDOTA_MAMBA_RAM_BASE_URL",
            default_base_url: "http://127.0.0.1:8081/v1",
        },
    ),
    (
        "mamba7b-gpu",
        ProviderMetadata {
            provider: ProviderKind::LucidotaLocal,
            auth_env: "LUCIDOTA_LOCAL_API_KEY",
            base_url_env: "LUCIDOTA_MAMBA_GPU_BASE_URL",
            default_base_url: "http://127.0.0.1:8083/v1",
        },
    ),
    (
        "groq",
        ProviderMetadata {
            provider: ProviderKind::Groq,
            auth_env: "GROQ_API_KEY",
            base_url_env: "GROQ_BASE_URL",
            default_base_url: openai_compat::DEFAULT_GROQ_BASE_URL,
        },
    ),
    (
        "cerebras",
        ProviderMetadata {
            provider: ProviderKind::Cerebras,
            auth_env: "CEREBRAS_API_KEY",
            base_url_env: "CEREBRAS_BASE_URL",
            default_base_url: openai_compat::DEFAULT_CEREBRAS_BASE_URL,
        },
    ),
    (
        "gpt-oss-120b",
        ProviderMetadata {
            provider: ProviderKind::Cerebras,
            auth_env: "CEREBRAS_API_KEY",
            base_url_env: "CEREBRAS_BASE_URL",
            default_base_url: openai_compat::DEFAULT_CEREBRAS_BASE_URL,
        },
    ),
    (
        "cohere",
        ProviderMetadata {
            provider: ProviderKind::Cohere,
            auth_env: "COHERE_API_KEY",
            base_url_env: "COHERE_BASE_URL",
            default_base_url: openai_compat::DEFAULT_COHERE_BASE_URL,
        },
    ),
    (
        "opus",
        ProviderMetadata {
            provider: ProviderKind::ClawApi,
            auth_env: "ANTHROPIC_API_KEY",
            base_url_env: "ANTHROPIC_BASE_URL",
            default_base_url: claw_provider::DEFAULT_BASE_URL,
        },
    ),
    (
        "sonnet",
        ProviderMetadata {
            provider: ProviderKind::ClawApi,
            auth_env: "ANTHROPIC_API_KEY",
            base_url_env: "ANTHROPIC_BASE_URL",
            default_base_url: claw_provider::DEFAULT_BASE_URL,
        },
    ),
    (
        "haiku",
        ProviderMetadata {
            provider: ProviderKind::ClawApi,
            auth_env: "ANTHROPIC_API_KEY",
            base_url_env: "ANTHROPIC_BASE_URL",
            default_base_url: claw_provider::DEFAULT_BASE_URL,
        },
    ),
    (
        "claude-opus-4-6",
        ProviderMetadata {
            provider: ProviderKind::ClawApi,
            auth_env: "ANTHROPIC_API_KEY",
            base_url_env: "ANTHROPIC_BASE_URL",
            default_base_url: claw_provider::DEFAULT_BASE_URL,
        },
    ),
    (
        "claude-sonnet-4-6",
        ProviderMetadata {
            provider: ProviderKind::ClawApi,
            auth_env: "ANTHROPIC_API_KEY",
            base_url_env: "ANTHROPIC_BASE_URL",
            default_base_url: claw_provider::DEFAULT_BASE_URL,
        },
    ),
    (
        "claude-haiku-4-5-20251213",
        ProviderMetadata {
            provider: ProviderKind::ClawApi,
            auth_env: "ANTHROPIC_API_KEY",
            base_url_env: "ANTHROPIC_BASE_URL",
            default_base_url: claw_provider::DEFAULT_BASE_URL,
        },
    ),
    (
        "grok",
        ProviderMetadata {
            provider: ProviderKind::Xai,
            auth_env: "XAI_API_KEY",
            base_url_env: "XAI_BASE_URL",
            default_base_url: openai_compat::DEFAULT_XAI_BASE_URL,
        },
    ),
    (
        "grok-3",
        ProviderMetadata {
            provider: ProviderKind::Xai,
            auth_env: "XAI_API_KEY",
            base_url_env: "XAI_BASE_URL",
            default_base_url: openai_compat::DEFAULT_XAI_BASE_URL,
        },
    ),
    (
        "grok-mini",
        ProviderMetadata {
            provider: ProviderKind::Xai,
            auth_env: "XAI_API_KEY",
            base_url_env: "XAI_BASE_URL",
            default_base_url: openai_compat::DEFAULT_XAI_BASE_URL,
        },
    ),
    (
        "grok-3-mini",
        ProviderMetadata {
            provider: ProviderKind::Xai,
            auth_env: "XAI_API_KEY",
            base_url_env: "XAI_BASE_URL",
            default_base_url: openai_compat::DEFAULT_XAI_BASE_URL,
        },
    ),
    (
        "grok-2",
        ProviderMetadata {
            provider: ProviderKind::Xai,
            auth_env: "XAI_API_KEY",
            base_url_env: "XAI_BASE_URL",
            default_base_url: openai_compat::DEFAULT_XAI_BASE_URL,
        },
    ),
];

#[must_use]
pub fn resolve_model_alias(model: &str) -> String {
    let trimmed = model.trim();
    let lower = trimmed.to_ascii_lowercase();
    MODEL_REGISTRY
        .iter()
        .find_map(|(alias, metadata)| {
            (*alias == lower).then_some(match metadata.provider {
                ProviderKind::ClawApi => match *alias {
                    "opus" => "claude-opus-4-6",
                    "sonnet" => "claude-sonnet-4-6",
                    "haiku" => "claude-haiku-4-5-20251213",
                    _ => trimmed,
                },
                ProviderKind::Xai => match *alias {
                    "grok" | "grok-3" => "grok-3",
                    "grok-mini" | "grok-3-mini" => "grok-3-mini",
                    "grok-2" => "grok-2",
                    _ => trimmed,
                },
                ProviderKind::OpenAi => trimmed,
                ProviderKind::Groq => match *alias {
                    "groq" => "llama-3.3-70b-versatile",
                    _ => trimmed,
                },
                ProviderKind::Cerebras => match *alias {
                    "cerebras" => "gpt-oss-120b",
                    _ => trimmed,
                },
                ProviderKind::Cohere => match *alias {
                    "cohere" => "command-a-03-2025",
                    _ => trimmed,
                },
                ProviderKind::LucidotaLocal => trimmed,
            })
        })
        .map_or_else(|| trimmed.to_string(), ToOwned::to_owned)
}

#[must_use]
pub fn metadata_for_model(model: &str) -> Option<ProviderMetadata> {
    let canonical = resolve_model_alias(model);
    let lower = canonical.to_ascii_lowercase();
    if let Some((_, metadata)) = MODEL_REGISTRY.iter().find(|(alias, _)| *alias == lower) {
        return Some(*metadata);
    }
    if lower.starts_with("grok") {
        return Some(ProviderMetadata {
            provider: ProviderKind::Xai,
            auth_env: "XAI_API_KEY",
            base_url_env: "XAI_BASE_URL",
            default_base_url: openai_compat::DEFAULT_XAI_BASE_URL,
        });
    }
    if lower.starts_with("llama-") || lower.starts_with("mixtral") || lower.contains("groq") {
        return Some(ProviderMetadata {
            provider: ProviderKind::Groq,
            auth_env: "GROQ_API_KEY",
            base_url_env: "GROQ_BASE_URL",
            default_base_url: openai_compat::DEFAULT_GROQ_BASE_URL,
        });
    }
    if lower.starts_with("command-") || lower.contains("cohere") {
        return Some(ProviderMetadata {
            provider: ProviderKind::Cohere,
            auth_env: "COHERE_API_KEY",
            base_url_env: "COHERE_BASE_URL",
            default_base_url: openai_compat::DEFAULT_COHERE_BASE_URL,
        });
    }
    if lower.contains("cerebras") || lower == "gpt-oss-120b" {
        return Some(ProviderMetadata {
            provider: ProviderKind::Cerebras,
            auth_env: "CEREBRAS_API_KEY",
            base_url_env: "CEREBRAS_BASE_URL",
            default_base_url: openai_compat::DEFAULT_CEREBRAS_BASE_URL,
        });
    }
    if lower.contains("local") || lower.starts_with("mamba7b-") || lower == "luci" {
        return Some(ProviderMetadata {
            provider: ProviderKind::LucidotaLocal,
            auth_env: "LUCIDOTA_LOCAL_API_KEY",
            base_url_env: "LUCIDOTA_LOCAL_BASE_URL",
            default_base_url: openai_compat::DEFAULT_LUCIDOTA_LOCAL_BASE_URL,
        });
    }
    None
}

#[must_use]
pub fn detect_provider_kind(model: &str) -> ProviderKind {
    if let Some(metadata) = metadata_for_model(model) {
        return metadata.provider;
    }
    if claw_provider::has_auth_from_env_or_saved().unwrap_or(false) {
        return ProviderKind::ClawApi;
    }
    if openai_compat::has_api_key("OPENAI_API_KEY") {
        return ProviderKind::OpenAi;
    }
    if openai_compat::has_api_key("GROQ_API_KEY") {
        return ProviderKind::Groq;
    }
    if openai_compat::has_api_key("COHERE_API_KEY") {
        return ProviderKind::Cohere;
    }
    if openai_compat::has_api_key("CEREBRAS_API_KEY") {
        return ProviderKind::Cerebras;
    }
    if openai_compat::has_api_key("XAI_API_KEY") {
        return ProviderKind::Xai;
    }
    ProviderKind::ClawApi
}

#[must_use]
pub fn max_tokens_for_model(model: &str) -> u32 {
    let canonical = resolve_model_alias(model);
    if canonical.contains("local") || canonical.starts_with("mamba7b-") || canonical == "luci" {
        2_048
    } else if canonical.starts_with("llama-") || canonical.starts_with("command-") {
        4_096
    } else if canonical.contains("opus") {
        32_000
    } else {
        8_192
    }
}

#[cfg(test)]
mod tests {
    use super::{detect_provider_kind, max_tokens_for_model, resolve_model_alias, ProviderKind};

    #[test]
    fn resolves_grok_aliases() {
        assert_eq!(resolve_model_alias("grok"), "grok-3");
        assert_eq!(resolve_model_alias("grok-mini"), "grok-3-mini");
        assert_eq!(resolve_model_alias("grok-2"), "grok-2");
    }

    #[test]
    fn detects_provider_from_model_name_first() {
        assert_eq!(detect_provider_kind("grok"), ProviderKind::Xai);
        assert_eq!(
            detect_provider_kind("claude-sonnet-4-6"),
            ProviderKind::ClawApi
        );
    }

    #[test]
    fn keeps_existing_max_token_heuristic() {
        assert_eq!(max_tokens_for_model("opus"), 32_000);
        assert_eq!(max_tokens_for_model("grok-3"), 8_192);
        assert_eq!(max_tokens_for_model("luci"), 2_048);
    }
}
