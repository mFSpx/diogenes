use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};
use std::time::Duration;

use hyper_util::rt::TokioIo;
use serde::{Deserialize, Serialize};
use tokio::net::UnixStream;
use tonic::transport::{Channel, Endpoint};
use tower::service_fn;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PermissionMode {
    ReadOnly,
    WorkspaceWrite,
    DangerFullAccess,
    Prompt,
    Allow,
}

impl PermissionMode {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::WorkspaceWrite => "workspace-write",
            Self::DangerFullAccess => "danger-full-access",
            Self::Prompt => "prompt",
            Self::Allow => "allow",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionRequest {
    pub tool_name: String,
    pub input: String,
    pub current_mode: PermissionMode,
    pub required_mode: PermissionMode,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelClientConfig {
    pub endpoint: String,
    pub timeout_ms: u64,
    pub unix_socket_path: Option<PathBuf>,
}

#[allow(dead_code)]
impl KernelClientConfig {
    #[must_use]
    pub fn from_env() -> Self {
        let unix_socket_path = env::var("CKDOG_KERNEL_SOCKET").ok().map(PathBuf::from);
        Self {
            endpoint: env::var("CKDOG_KERNEL_ENDPOINT").unwrap_or_else(|_| {
                if unix_socket_path.is_some() {
                    "http://[::]:50051".to_string()
                } else {
                    "http://127.0.0.1:50051".to_string()
                }
            }),
            timeout_ms: env::var("CKDOG_KERNEL_TIMEOUT_MS")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(750),
            unix_socket_path,
        }
    }

    #[must_use]
    pub fn for_unix_socket(path: impl Into<PathBuf>) -> Self {
        Self {
            endpoint: "http://[::]:50051".to_string(),
            timeout_ms: 750,
            unix_socket_path: Some(path.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelRouteDecision {
    pub trace_hash: String,
    pub trinary_flags: Vec<i8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KernelControlPacket {
    pub tool_name: String,
    pub input_sha256: String,
    pub input_len: usize,
    pub prompt_token_ids: Vec<u64>,
}

impl KernelControlPacket {
    #[must_use]
    pub fn from_prompt(tool_name: &str, input: &str) -> Self {
        Self {
            tool_name: tool_name.to_string(),
            input_sha256: stable_text_hash(input),
            input_len: input.len(),
            prompt_token_ids: prompt_text_token_ids(input),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionPromptDecision {
    Allow,
    Deny { reason: String },
}

pub trait PermissionPrompter {
    fn decide(&mut self, request: &PermissionRequest) -> PermissionPromptDecision;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionOutcome {
    Allow,
    Deny { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionPolicy {
    active_mode: PermissionMode,
    tool_requirements: BTreeMap<String, PermissionMode>,
}

impl PermissionPolicy {
    #[must_use]
    pub fn new(active_mode: PermissionMode) -> Self {
        Self {
            active_mode,
            tool_requirements: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn with_tool_requirement(
        mut self,
        tool_name: impl Into<String>,
        required_mode: PermissionMode,
    ) -> Self {
        self.tool_requirements
            .insert(tool_name.into(), required_mode);
        self
    }

    #[must_use]
    pub fn active_mode(&self) -> PermissionMode {
        self.active_mode
    }

    #[must_use]
    pub fn required_mode_for(&self, tool_name: &str) -> PermissionMode {
        self.tool_requirements
            .get(tool_name)
            .copied()
            .unwrap_or(PermissionMode::DangerFullAccess)
    }

    #[must_use]
    pub fn authorize(
        &self,
        tool_name: &str,
        input: &str,
        mut prompter: Option<&mut dyn PermissionPrompter>,
    ) -> PermissionOutcome {
        if let Some(route_decision) = extract_kernel_route_decision(input) {
            if let Err(reason) = evaluate_trinary_route_flags(&route_decision.trinary_flags) {
                return PermissionOutcome::Deny { reason };
            }
        }
        let required_mode = self.required_mode_for(tool_name);
        if self.active_mode == PermissionMode::Allow
            || self.active_mode == PermissionMode::DangerFullAccess
            || (self.active_mode != PermissionMode::Prompt && self.active_mode >= required_mode)
        {
            return PermissionOutcome::Allow;
        }

        let request = PermissionRequest {
            tool_name: tool_name.to_string(),
            input: input.to_string(),
            current_mode: self.active_mode,
            required_mode,
        };
        let Some(prompter) = prompter.as_deref_mut() else {
            return PermissionOutcome::Deny {
                reason: format!(
                    "tool {tool_name} requires {} but active mode is {}",
                    required_mode.as_str(),
                    self.active_mode.as_str()
                ),
            };
        };
        match prompter.decide(&request) {
            PermissionPromptDecision::Allow => PermissionOutcome::Allow,
            PermissionPromptDecision::Deny { reason } => PermissionOutcome::Deny { reason },
        }
    }
}

#[allow(dead_code)]
#[must_use]
pub fn build_kernel_control_payload(tool_name: &str, input: &str) -> serde_json::Value {
    serde_json::to_value(KernelControlPacket::from_prompt(tool_name, input))
        .expect("KernelControlPacket serializes")
}

#[allow(dead_code)]
pub async fn connect_ckdog_kernel_channel(
    config: &KernelClientConfig,
) -> Result<Channel, Box<dyn std::error::Error + Send + Sync>> {
    let endpoint = Endpoint::from_shared(config.endpoint.clone())?
        .connect_timeout(Duration::from_millis(config.timeout_ms));
    if let Some(path) = config.unix_socket_path.clone() {
        return connect_ckdog_kernel_unix(endpoint, path).await;
    }
    Ok(endpoint.connect().await?)
}

#[allow(dead_code)]
async fn connect_ckdog_kernel_unix(
    endpoint: Endpoint,
    path: PathBuf,
) -> Result<Channel, Box<dyn std::error::Error + Send + Sync>> {
    Ok(endpoint
        .connect_with_connector(service_fn(move |_uri: http::Uri| {
            let path = path.clone();
            async move {
                let stream = UnixStream::connect(Path::new(&path)).await?;
                Ok::<_, std::io::Error>(TokioIo::new(stream))
            }
        }))
        .await?)
}

pub fn evaluate_trinary_route_flags(flags: &[i8]) -> Result<(), String> {
    if let Some((index, _)) = flags.iter().enumerate().find(|(_, flag)| **flag == -1) {
        return Err(format!("ckdog kernel denied route flag at index {index}"));
    }
    if flags.iter().any(|flag| !matches!(*flag, -1 | 0 | 1)) {
        return Err("ckdog kernel route flags must be trinary -1/0/1".to_string());
    }
    Ok(())
}

#[must_use]
pub fn extract_kernel_route_decision(input: &str) -> Option<KernelRouteDecision> {
    let value = serde_json::from_str::<serde_json::Value>(input).ok()?;
    let flags = value
        .get("ckdog_route_flags")?
        .as_array()?
        .iter()
        .map(|item| item.as_i64().and_then(|value| i8::try_from(value).ok()))
        .collect::<Option<Vec<_>>>()?;
    Some(KernelRouteDecision {
        trace_hash: value
            .get("trace_hash")
            .and_then(|item| item.as_str())
            .unwrap_or_default()
            .to_string(),
        trinary_flags: flags,
    })
}

#[allow(dead_code)]
fn stable_text_hash(text: &str) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn prompt_text_token_ids(input: &str) -> Vec<u64> {
    input.split_whitespace().map(stable_text_hash_u64).collect()
}

fn stable_text_hash_u64(text: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::{
        build_kernel_control_payload, evaluate_trinary_route_flags, extract_kernel_route_decision,
        KernelClientConfig, KernelControlPacket, PermissionMode, PermissionOutcome,
        PermissionPolicy, PermissionPromptDecision, PermissionPrompter, PermissionRequest,
    };

    struct RecordingPrompter {
        seen: Vec<PermissionRequest>,
        allow: bool,
    }

    impl PermissionPrompter for RecordingPrompter {
        fn decide(&mut self, request: &PermissionRequest) -> PermissionPromptDecision {
            self.seen.push(request.clone());
            if self.allow {
                PermissionPromptDecision::Allow
            } else {
                PermissionPromptDecision::Deny {
                    reason: "not now".to_string(),
                }
            }
        }
    }

    #[test]
    fn allows_tools_when_active_mode_meets_requirement() {
        let policy = PermissionPolicy::new(PermissionMode::WorkspaceWrite)
            .with_tool_requirement("read_file", PermissionMode::ReadOnly)
            .with_tool_requirement("write_file", PermissionMode::WorkspaceWrite);

        assert_eq!(
            policy.authorize("read_file", "{}", None),
            PermissionOutcome::Allow
        );
        assert_eq!(
            policy.authorize("write_file", "{}", None),
            PermissionOutcome::Allow
        );
    }

    #[test]
    fn denies_escalations_without_prompter() {
        let policy = PermissionPolicy::new(PermissionMode::ReadOnly)
            .with_tool_requirement("write_file", PermissionMode::WorkspaceWrite)
            .with_tool_requirement("bash", PermissionMode::DangerFullAccess);

        assert!(matches!(
            policy.authorize("write_file", "{}", None),
            PermissionOutcome::Deny { reason } if reason.contains("requires workspace-write")
        ));
        assert!(matches!(
            policy.authorize("bash", "echo hi", None),
            PermissionOutcome::Deny { reason } if reason.contains("requires danger-full-access")
        ));
    }

    #[test]
    fn prompts_for_workspace_write_to_danger_full_access_escalation() {
        let policy = PermissionPolicy::new(PermissionMode::WorkspaceWrite)
            .with_tool_requirement("bash", PermissionMode::DangerFullAccess);
        let mut prompter = RecordingPrompter {
            seen: Vec::new(),
            allow: true,
        };

        let outcome = policy.authorize("bash", "echo hi", Some(&mut prompter));

        assert_eq!(outcome, PermissionOutcome::Allow);
        assert_eq!(prompter.seen.len(), 1);
    }

    #[test]
    fn prompt_rejection_reason_is_preserved() {
        let policy = PermissionPolicy::new(PermissionMode::WorkspaceWrite)
            .with_tool_requirement("bash", PermissionMode::DangerFullAccess);
        let mut prompter = RecordingPrompter {
            seen: Vec::new(),
            allow: false,
        };

        assert_eq!(
            policy.authorize("bash", "echo hi", Some(&mut prompter)),
            PermissionOutcome::Deny {
                reason: "not now".to_string()
            }
        );
        assert_eq!(prompter.seen.len(), 1);
    }

    #[test]
    fn kernel_client_config_uses_local_endpoint_default() {
        let config = KernelClientConfig::from_env();

        assert!(!config.endpoint.is_empty());
        assert!(config.timeout_ms > 0);
    }

    #[test]
    fn kernel_client_config_can_target_local_unix_socket() {
        let config = KernelClientConfig::for_unix_socket("/tmp/ckdog-kernel.sock");

        assert_eq!(config.endpoint, "http://[::]:50051");
        assert_eq!(
            config.unix_socket_path.as_deref(),
            Some(std::path::Path::new("/tmp/ckdog-kernel.sock"))
        );
    }

    #[test]
    fn trinary_route_flags_reject_explicit_negative_one() {
        let result = evaluate_trinary_route_flags(&[1, 0, -1, 1]);

        assert!(result.expect_err("negative one denies").contains("denied"));
    }

    #[test]
    fn trinary_route_flags_reject_non_trinary_values() {
        let result = evaluate_trinary_route_flags(&[1, 2, 0]);

        assert!(result
            .expect_err("non-trinary values fail")
            .contains("trinary"));
    }

    #[test]
    fn permission_policy_denies_embedded_kernel_negative_route() {
        let policy = PermissionPolicy::new(PermissionMode::DangerFullAccess);
        let input = r#"{"trace_hash":"trace","ckdog_route_flags":[1,0,-1]}"#;

        assert!(matches!(
            policy.authorize("bash", input, None),
            PermissionOutcome::Deny { reason } if reason.contains("ckdog kernel denied")
        ));
    }

    #[test]
    fn extracts_kernel_route_decision_from_json_payload() {
        let decision =
            extract_kernel_route_decision(r#"{"trace_hash":"abc","ckdog_route_flags":[1,0,1]}"#)
                .expect("route decision present");

        assert_eq!(decision.trace_hash, "abc");
        assert_eq!(decision.trinary_flags, vec![1, 0, 1]);
    }

    #[test]
    fn kernel_control_payload_is_language_invariant_hash_metadata() {
        let payload = build_kernel_control_payload("bash", "echo hello");

        assert_eq!(payload["tool_name"], "bash");
        assert_eq!(payload["input_len"], "echo hello".len());
        assert!(payload["input_sha256"].as_str().is_some());
        assert_eq!(payload["prompt_token_ids"].as_array().unwrap().len(), 2);
        assert!(payload.get("raw_text").is_none());
    }

    #[test]
    fn kernel_control_packet_maps_prompt_to_numeric_token_ids_only() {
        let packet = KernelControlPacket::from_prompt("bash", "ponyboy lucidota operator");

        assert_eq!(packet.tool_name, "bash");
        assert_eq!(packet.input_len, "ponyboy lucidota operator".len());
        assert_eq!(packet.prompt_token_ids.len(), 3);
        assert!(packet.prompt_token_ids.iter().all(|value| *value > 0));
    }
}
