#!/usr/bin/env bash
set -euo pipefail

# LUCIDOTA/LLxprt hard-binding: prefer Groq on login/session initialization.
LUCIDOTA_SECRET_ENV="${LUCIDOTA_SECRET_ENV:-$HOME/.config/lucidota/secrets.env}"
if [ -f "$LUCIDOTA_SECRET_ENV" ]; then
  # shellcheck disable=SC1090
  set -a
  . "$LUCIDOTA_SECRET_ENV"
  set +a
fi

if [ -f "$HOME/LUCIDOTA/.env" ]; then
  # shellcheck disable=SC1090
  set -a
  . "$HOME/LUCIDOTA/.env"
  set +a
fi

LUCIDOTA_GROQ_KEY_FILE="${LUCIDOTA_GROQ_KEY_FILE:-/tmp/lucidota_groq_key}"
if [ -r "$LUCIDOTA_GROQ_KEY_FILE" ]; then
  GROQ_API_KEY="$(tr -d '\r\n' < "$LUCIDOTA_GROQ_KEY_FILE")"
fi

: "${GROQ_API_KEY:?GROQ_API_KEY is required}"
: "${GROQ_BASE_URL:=https://api.groq.com/openai/v1}"

export GROQ_API_KEY
export GROQ_BASE_URL
export LUCIDOTA_STRATEGIC_PROVIDER="groq"
export LUCIDOTA_STRATEGIC_MODEL="${GROQ_MODEL:-llama-3.1-8b-instant}"
export OPENAI_API_KEY="$GROQ_API_KEY"
export OPENAI_BASE_URL="$GROQ_BASE_URL"
export LLXPRT_PROFILE="groq"
export LLXPRT_FORCE_PROFILE="groq"
export LLXPRT_DEFAULT_PROFILE="groq"
