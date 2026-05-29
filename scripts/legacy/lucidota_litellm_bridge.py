#!/usr/bin/env python3
"""LiteLLM-compatible model management bridge for LUCIDOTA.

This is intentionally a dry inventory/validation command. It does not import
litellm, open a model, call a provider API, or run inference. It reports the
model/provider configuration that can be translated into a LiteLLM `model_list`
and validates only cheap local facts: environment key presence and artifact path
existence.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DB = os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql://mfspx@/lucidota_state")


@dataclass(frozen=True)
class Provider:
    name: str
    key_env: str | None
    models_env: str
    default_models: tuple[str, ...] = ()


PROVIDERS: tuple[Provider, ...] = (
    Provider("openai", "OPENAI_API_KEY", "LUCIDOTA_OPENAI_MODELS", ("gpt-4o-mini",)),
    Provider("anthropic", "ANTHROPIC_API_KEY", "LUCIDOTA_ANTHROPIC_MODELS", ("claude-3-5-haiku-latest",)),
    Provider("gemini", "GEMINI_API_KEY", "LUCIDOTA_GEMINI_MODELS", ("gemini/gemini-1.5-flash",)),
    Provider("mistral", "MISTRAL_API_KEY", "LUCIDOTA_MISTRAL_MODELS", ("mistral/mistral-small-latest",)),
    Provider("groq", "GROQ_API_KEY", "LUCIDOTA_GROQ_MODELS", ()),
    Provider("cohere", "COHERE_API_KEY", "LUCIDOTA_COHERE_MODELS", ()),
    Provider("openrouter", "OPENROUTER_API_KEY", "LUCIDOTA_OPENROUTER_MODELS", ()),
    Provider("azure", "AZURE_API_KEY", "LUCIDOTA_AZURE_MODELS", ()),
)


def split_csv(value: str | None) -> list[str]:
    return [x.strip() for x in (value or "").split(",") if x.strip()]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_local_path(value: str | None) -> dict[str, Any]:
    if not value:
        return {"configured": False, "status": "not_configured"}
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    exists = path.exists()
    info: dict[str, Any] = {"configured": True, "path": rel(path), "status": "exists" if exists else "missing"}
    if exists:
        try:
            st = path.stat()
            info.update({"is_file": path.is_file(), "is_dir": path.is_dir(), "bytes": st.st_size if path.is_file() else None})
        except OSError as exc:
            info.update({"status": "error", "error": str(exc)[:200]})
    return info


def env_provider_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    raw_extra = os.environ.get("LUCIDOTA_LITELLM_MODELS_JSON")
    if raw_extra:
        try:
            parsed = json.loads(raw_extra)
            if not isinstance(parsed, list):
                raise ValueError("must be a JSON list")
            for item in parsed:
                if not isinstance(item, dict):
                    raise ValueError("each item must be an object")
                model_name = str(item.get("model_name") or item.get("model") or "").strip()
                litellm_model = str(item.get("litellm_model") or item.get("model") or model_name).strip()
                provider = str(item.get("provider") or litellm_model.split("/", 1)[0] or "custom")
                key_env = item.get("api_key_env")
                local_path = item.get("local_path")
                entries.append(
                    {
                        "source": "env:LUCIDOTA_LITELLM_MODELS_JSON",
                        "provider": provider,
                        "model_name": model_name,
                        "litellm_model": litellm_model,
                        "api_key_env": key_env,
                        "api_key_status": key_status(str(key_env)) if key_env else "not_required",
                        "local_path": resolve_local_path(str(local_path)) if local_path else resolve_local_path(None),
                    }
                )
        except Exception as exc:
            entries.append({"source": "env:LUCIDOTA_LITELLM_MODELS_JSON", "error": str(exc)[:300]})

    for provider in PROVIDERS:
        configured_models = split_csv(os.environ.get(provider.models_env))
        key_present = bool(provider.key_env and os.environ.get(provider.key_env))
        if not configured_models and not key_present:
            continue
        models = configured_models or list(provider.default_models)
        if not models:
            entries.append(
                {
                    "source": f"env:{provider.key_env}",
                    "provider": provider.name,
                    "model_name": None,
                    "litellm_model": None,
                    "api_key_env": provider.key_env,
                    "api_key_status": key_status(provider.key_env),
                    "local_path": resolve_local_path(None),
                    "note": f"Set {provider.models_env}=model-a,model-b to publish concrete LiteLLM routes.",
                }
            )
            continue
        for model in models:
            entries.append(
                {
                    "source": f"env:{provider.models_env if configured_models else provider.key_env}",
                    "provider": provider.name,
                    "model_name": model.split("/", 1)[-1] if "/" in model else model,
                    "litellm_model": model if "/" in model or provider.name == "azure" else f"{provider.name}/{model}",
                    "api_key_env": provider.key_env,
                    "api_key_status": key_status(provider.key_env),
                    "local_path": resolve_local_path(None),
                }
            )
    return entries


def key_status(key_env: str | None) -> str:
    if not key_env:
        return "not_required"
    return "present" if os.environ.get(key_env) else "missing"


def db_registry_entries() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import psycopg
    except Exception as exc:
        return [], {"status": "unavailable", "error": f"psycopg import failed: {exc}"[:300]}
    try:
        with psycopg.connect(DB, connect_timeout=3) as conn:
            rows = conn.execute(
                """
                SELECT model_id, role, source_url, local_path, license, parameter_count,
                       quantization, expected_vram_mb, benchmark_status, notes
                FROM lucidota_runtime.model_candidate
                ORDER BY model_id
                """
            ).fetchall()
    except Exception as exc:
        return [], {"status": "unavailable", "database_url": DB, "error": str(exc)[:300]}

    entries: list[dict[str, Any]] = []
    for row in rows:
        model_id, role, source_url, local_path, license_, params, quant, vram, status, notes = row
        path_check = resolve_local_path(local_path)
        provider = "local" if local_path else "registry-intent"
        entries.append(
            {
                "source": "db:lucidota_runtime.model_candidate",
                "provider": provider,
                "model_name": model_id,
                "litellm_model": model_id,
                "api_key_env": None,
                "api_key_status": "not_required" if local_path else "not_configured",
                "local_path": path_check,
                "model_info": {
                    "role": role,
                    "source_url": source_url,
                    "license": license_,
                    "parameter_count": params,
                    "quantization": quant,
                    "expected_vram_mb": vram,
                    "benchmark_status": status,
                    "notes": notes,
                },
            }
        )
    return entries, {"status": "ok", "database_url": DB, "model_count": len(entries)}


def model_list(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("error") or not entry.get("model_name") or not entry.get("litellm_model"):
            continue
        local = entry.get("local_path") or {}
        local_path = str(local.get("path") or "")
        if entry.get("provider") == "local" and local_path.endswith(".gguf"):
            # llama.cpp server exposes an OpenAI-compatible /v1 API; LiteLLM routes it via openai/*.
            model_key = str(entry.get("model_name") or "").lower() + " " + local_path.lower()
            if "mamba" in model_key:
                api_base = os.environ.get("LUCIDOTA_MAMBA_API_BASE", "http://127.0.0.1:8081/v1")
            elif "deepseek" in model_key:
                api_base = os.environ.get("LUCIDOTA_DEEPSEEK_API_BASE", "http://127.0.0.1:8080/v1")
            else:
                api_base = os.environ.get("LUCIDOTA_LLAMA_API_BASE", "http://127.0.0.1:8080/v1")
            params: dict[str, Any] = {
                "model": f"openai/{entry['model_name']}",
                "api_base": api_base,
                "api_key": os.environ.get("LUCIDOTA_LLAMA_API_KEY", "not-needed"),
            }
        else:
            params = {"model": entry["litellm_model"]}
        key_env = entry.get("api_key_env")
        if key_env:
            params["api_key"] = f"os.environ/{key_env}"
        if local.get("configured"):
            params["local_path"] = local.get("path")
        out.append(
            {
                "model_name": entry["model_name"],
                "litellm_params": params,
                "model_info": {"provider": entry.get("provider"), "source": entry.get("source"), **entry.get("model_info", {})},
            }
        )
    return out


def validate(entries: list[dict[str, Any]]) -> dict[str, Any]:
    missing_keys = [e for e in entries if e.get("api_key_status") == "missing"]
    missing_paths = [e for e in entries if (e.get("local_path") or {}).get("status") in {"missing", "error"}]
    config_errors = [e for e in entries if e.get("error")]
    # Missing API keys are reported, but not fatal unless --strict. This keeps check_diogenes offline.
    hard_errors = config_errors + missing_paths
    return {
        "ok": not hard_errors,
        "hard_error_count": len(hard_errors),
        "missing_api_key_count": len(missing_keys),
        "missing_local_path_count": len(missing_paths),
        "config_error_count": len(config_errors),
        "missing_api_keys": [compact(e) for e in missing_keys],
        "missing_local_paths": [compact(e) for e in missing_paths],
        "config_errors": config_errors,
    }


def compact(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": entry.get("source"),
        "provider": entry.get("provider"),
        "model_name": entry.get("model_name"),
        "api_key_env": entry.get("api_key_env"),
        "local_path": entry.get("local_path"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="lucidota-litellm-bridge")
    parser.add_argument("--json", action="store_true", help="emit machine-readable report")
    parser.add_argument("--strict", action="store_true", help="treat missing API keys as failures")
    parser.add_argument("--model-list-only", action="store_true", help="emit just the LiteLLM model_list JSON")
    args = parser.parse_args()

    db_entries, db_status = db_registry_entries()
    entries = db_entries + env_provider_entries()
    lite = model_list(entries)
    checks = validate(entries)
    ok = checks["ok"] and (not args.strict or checks["missing_api_key_count"] == 0)
    report = {
        "ok": ok,
        "live_inference": False,
        "db_registry": db_status,
        "providers": sorted({str(e.get("provider")) for e in entries if e.get("provider")}),
        "models_configured": len(lite),
        "checks": checks,
        "entries": entries,
        "litellm": {"model_list": lite},
    }

    payload: Any = lite if args.model_list_only else report
    if args.json or args.model_list_only:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
