# GOAL 3 Model Fabric Audit

Evidence: `05_OUTPUTS/goals/goal3_local_model_provider_audit_20260526T013651Z.txt`, `05_OUTPUTS/model_runtime/strict_model_stack_admission_20260526T014322843048Z.json`, and `05_OUTPUTS/goals/goal3_needle_swarm_health_20260526T014422Z.json`.

## Local lanes already in LUCIDOTA

- `needle_swarm_6x`: `scripts/lucidota_start_needle_swarm.sh`, six CPU Needle workers, ports `8090-8095`, model `03_VAULT/models/needle/needle.pkl`.
- `mamba7b_ram`: `scripts/lucidota_start_mamba_llama.sh`, llama.cpp server, port `8081`, model `03_VAULT/models/tensorblock/Falcon3-Mamba-7B-Instruct-GGUF/Falcon3-Mamba-7B-Instruct-Q2_K.gguf`.
- `bonsai4b_ram`: `scripts/lucidota_start_bonsai_ternary_llama.sh`, PrismML llama.cpp server, port `8082`, model `03_VAULT/models/prism-ml/Ternary-Bonsai-4B-gguf/Ternary-Bonsai-4B-Q2_0.gguf`.
- `deepseek_r1_qwen_1p5b_gpu`: `scripts/lucidota_start_deepseek_llama.sh`, llama.cpp server, port `8080`, model `03_VAULT/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf`.
- Optional `mamba7b_gpu_partial`: strict stack inline lane, port `8083`, preemptible VRAM.

## Cloud/API lanes already present

- Groq: `scripts/groq_chat_cli.py`, `scripts/model_runner_cli.py groq-chat`, env key name `GROQ_API_KEY`.
- Cohere: `scripts/cohere_chat_cli.py`, `scripts/model_runner_cli.py cohere-chat`, env key names `COHERE_API_KEY` or `CO_API_KEY`.
- Legacy LiteLLM inventory exists at `scripts/legacy/lucidota_litellm_bridge.py`; copy/adapt only if needed, do not promote the corpse directly.

These lanes are optional peripherals. Core LUCIDOTA stays runnable without them; they are acceleration or provider-fanout options, not a baseline requirement.

secret values are not printed. Current shell env audit found no exported Groq/Cohere/OpenAI/Anthropic key values in this session; repo references only name the expected env vars. Prior redacted receipts show `GROQ_API_KEY` and `COHERE_API_KEY` were used successfully on 2026-05-22, so the adapters are wired; the keys just are not exported in this recovered shell.

## GOALS integration decision

GOALS should not start heavy models automatically. It should point agentic build systems at the right dry-run/execute command and require receipt evidence before claiming a lane is live. Starting Needle/Mamba/Bonsai/DeepSeek remains an explicit operator/runtime action through existing launchers and strict admission receipts.

## 2026-05-26 runtime status

- Strict model stack admission passed after sourcing `scripts/lucidota_safe_ops_env.sh`.
- Groq and Cohere adapters passed dry-run receipt generation; no cloud execution was performed and no key values were printed.
- Six Needle workers are live on `8090-8095` with health PASS; receipt reports about `2070.3 MiB` total RSS, so they are CPU-light when idle but not RAM-free.
- Mamba/Bonsai/DeepSeek heavy lanes were admitted but not auto-started in this tranche because current memory margin after Needle startup was small enough that launching multiple GGUF servers would risk repeating the freeze. Use the existing launch scripts one at a time when the operator wants the runtime, not as GOALS background habit.

## 2026-05-26 live fabric continuation

Receipt `05_OUTPUTS/goals/goal3_model_fabric_live_20260526T015103Z.json` shows 10/10 local health endpoints live: DeepSeek GPU on `8080`, Mamba 7B CPU on `8081`, Bonsai 4B CPU on `8082`, Mamba 7B partial GPU on `8083`, and six Needle workers on `8090-8095`.

Runtime guard: do not treat this as a forever daemon requirement. This is a proved live fabric state for GOAL 3; future runs should still launch one lane at a time with memory/GPU health receipts.

## Runtime control

Use `python3 scripts/goal_model_fabric_control.py status` for a receipt-backed status snapshot and `python3 scripts/goal_model_fabric_control.py stop --target optional|heavy|needles|all` when the live proof/demo is over. This keeps GOAL 3 from becoming a hidden forever-load: startup remains explicit, stop remains explicit, and both write receipts.

## 2026-05-26 post-proof resource trim

After live proof and bounded generation receipts passed, heavy lanes were explicitly stopped to avoid background slop and keep the GTX 1650/8GB laptop safe. Receipt `05_OUTPUTS/goals/goal_model_fabric_control_20260526T020156Z.json` stopped DeepSeek, Mamba CPU, Bonsai, and Mamba partial GPU; `05_OUTPUTS/goals/goal_model_fabric_control_20260526T020159Z.json` shows six Needle workers remain healthy and GPU memory returned to about `3714 MiB` free.

Restart heavy lanes only through existing launch scripts or strict stack admission, one lane at a time, with fresh receipts.
