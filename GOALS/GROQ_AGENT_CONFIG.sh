#!/usr/bin/env bash
# Groq Agent Configuration — 4 agents for the Diogenes swarm
# Source this file to set up agent credentials
# Any agent can call Groq anytime.

set -a

# ─── Groq API Key ──────────────────────────────────────────────────
# Loaded from secrets
if [ -f ~/.config/lucidota/secrets.env ]; then
    source ~/.config/lucidota/secrets.env
fi
GROQ_API_KEY="${GROQ_API_KEY}"

# ─── Agent 1: Deep Research ────────────────────────────────────────
AGENT_RESEARCH_MODEL="openai/gpt-oss-120b"
AGENT_RESEARCH_TEMP=0.3
AGENT_RESEARCH_MAX_TOKENS=4096
AGENT_RESEARCH_DESC="Deep research agent — long-context, low temperature"

# ─── Agent 2: Code Generation ──────────────────────────────────────
AGENT_CODEGEN_MODEL="meta-llama/llama-4-scout-17b-16e-instruct"
AGENT_CODEGEN_TEMP=0.7
AGENT_CODEGEN_MAX_TOKENS=8192
AGENT_CODEGEN_DESC="Code generation agent — creative, higher temperature"

# ─── Agent 3: Vision/Image ─────────────────────────────────────────
AGENT_VISION_MODEL="meta-llama/llama-4-scout-17b-16e-instruct"
AGENT_VISION_TEMP=0.5
AGENT_VISION_MAX_TOKENS=2048
AGENT_VISION_DESC="Vision agent — multimodal, routes image analysis to BitVLA and Groq"

# ─── Agent 4: Orchestrator/Router ──────────────────────────────────
AGENT_ORCHESTRATOR_MODEL="openai/gpt-oss-120b"
AGENT_ORCHESTRATOR_TEMP=0.8
AGENT_ORCHESTRATOR_MAX_TOKENS=2048
AGENT_ORCHESTRATOR_DESC="Orchestrator — routes tasks between agents and models"

# ─── Diogenes Governor ─────────────────────────────────────────────
LUCIDOTA_WATCHDOG_POLL_SECS=10
LUCIDOTA_VRAM_BUDGET_MB=4096
LUCIDOTA_VRAM_RESERVE_MB=768

# ─── Local Model Endpoints ─────────────────────────────────────────
LUCIDOTA_LOCAL_BASE_URL="${LUCIDOTA_LOCAL_BASE_URL:-http://127.0.0.1:8080/v1}"
BITVLA_VISION_URL="${BITVLA_VISION_URL:-http://127.0.0.1:7845}"

# ─── Export all ────────────────────────────────────────────────────
export GROQ_API_KEY
export AGENT_RESEARCH_MODEL AGENT_RESEARCH_TEMP AGENT_RESEARCH_MAX_TOKENS AGENT_RESEARCH_DESC
export AGENT_CODEGEN_MODEL AGENT_CODEGEN_TEMP AGENT_CODEGEN_MAX_TOKENS AGENT_CODEGEN_DESC
export AGENT_VISION_MODEL AGENT_VISION_TEMP AGENT_VISION_MAX_TOKENS AGENT_VISION_DESC
export AGENT_ORCHESTRATOR_MODEL AGENT_ORCHESTRATOR_TEMP AGENT_ORCHESTRATOR_MAX_TOKENS AGENT_ORCHESTRATOR_DESC
export LUCIDOTA_WATCHDOG_POLL_SECS LUCIDOTA_VRAM_BUDGET_MB LUCIDOTA_VRAM_RESERVE_MB BITVLA_VISION_URL

echo "[agents] Diogenes 4-agent swarm configured"
echo "[agents] Research: ${AGENT_RESEARCH_MODEL}"
echo "[agents] Codegen: ${AGENT_CODEGEN_MODEL}"
echo "[agents] Vision:  ${AGENT_VISION_MODEL}"
echo "[agents] Router:  ${AGENT_ORCHESTRATOR_MODEL}"
echo "[agents] Any agent can call Groq anytime."
