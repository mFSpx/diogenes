#!/usr/bin/env python3
"""LOCAL BONSAI TERNARY runtime wiring.

This is intentionally a thin, local-only handler around the PrismML llama.cpp
fork because the Bonsai GGUF uses GGML type 42 / Q2_0 ternary weights that the
stock llama.cpp build in this checkout rejected during load probing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path("03_VAULT/models/prism-ml/Ternary-Bonsai-4B-gguf/Ternary-Bonsai-4B-Q2_0.gguf")
MODEL_SOURCE_PATH = MODEL_PATH.parent / "model_source.json"
MODEL_ID = MODEL_PATH.name
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8082
DEFAULT_CTX = 1024
DEFAULT_NGL = 0
HANDLER_ID = "llama.cpp-prismml-q2_0"
PRISMML_LLAMA_ROOT = ROOT / "01_REPOS" / "prismml_llama.cpp"
PRISMML_BUILD_CUDA = PRISMML_LLAMA_ROOT / "build-cuda"
PRISMML_BUILD = PRISMML_LLAMA_ROOT / "build"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _rooted(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def load_model_source() -> dict[str, Any]:
    path = ROOT / MODEL_SOURCE_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("selected_file") != MODEL_ID:
        raise SystemExit(f"model_source selected_file mismatch: {data.get('selected_file')!r} != {MODEL_ID!r}")
    return data


def llama_server_candidates() -> list[Path]:
    env = os.environ.get("LUCIDOTA_BONSAI_LLAMA_SERVER")
    candidates = []
    if env:
        candidates.append(_rooted(env))
    candidates.extend(
        [
            PRISMML_BUILD_CUDA / "bin" / "llama-server",
            PRISMML_BUILD / "bin" / "llama-server",
            PRISMML_BUILD / "tools" / "server" / "llama-server",
        ]
    )
    return candidates


def find_llama_server() -> Path:
    candidates = llama_server_candidates()
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    joined = ", ".join(_rel(c) for c in candidates)
    raise SystemExit(
        "LOCAL BONSAI TERNARY llama-server binary not found. "
        "Build it with: cmake --build 01_REPOS/prismml_llama.cpp/build --target llama-server -j$(nproc). "
        f"Checked: {joined}"
    )


def base_url(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    return f"http://{host}:{int(port)}/v1"


def default_runtime_config(*, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> dict[str, Any]:
    source = load_model_source()
    model_path = ROOT / MODEL_PATH
    return {
        "schema": "lucidota.local_bonsai_ternary.runtime.v1",
        "model_id": MODEL_ID,
        "model_path": _rel(model_path),
        "model_size_bytes": model_path.stat().st_size if model_path.exists() else None,
        "source_model_id": source.get("model_id"),
        "source_url": source.get("source_url"),
        "expected_sha256": source.get("expected_sha256"),
        "format": source.get("format"),
        "base_model": source.get("base_model"),
        "context_length": source.get("context_length"),
        "handler": HANDLER_ID,
        "llama_root": _rel(PRISMML_LLAMA_ROOT),
        "preferred_build": _rel(PRISMML_BUILD_CUDA),
        "base_url": base_url(host, port),
        "host": host,
        "port": int(port),
        "prompting_note": "zero-shot preferred; avoid static few-shot examples because Bonsai may regurgitate examples",
        "local_only": True,
        "offline": True,
    }


def build_server_command(
    *,
    binary: Path | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    ctx: int = DEFAULT_CTX,
    ngl: int = DEFAULT_NGL,
    model_path: Path | None = None,
) -> list[str]:
    model = model_path or (ROOT / MODEL_PATH)
    exe = binary or find_llama_server()
    return [
        str(exe),
        "-m",
        str(model),
        "--host",
        host,
        "--port",
        str(int(port)),
        "-ngl",
        str(int(ngl)),
        "-c",
        str(int(ctx)),
        "--parallel",
        os.environ.get("LUCIDOTA_BONSAI_PARALLEL", "1"),
        "--batch-size",
        os.environ.get("LUCIDOTA_BONSAI_BATCH", "128"),
        "--ubatch-size",
        os.environ.get("LUCIDOTA_BONSAI_UBATCH", "32"),
        "--cache-ram",
        os.environ.get("LUCIDOTA_BONSAI_CACHE_RAM", "0"),
        "--no-warmup",
    ]


def llxprt_provider_config(*, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> dict[str, Any]:
    return {
        "name": "lucidota-local",
        "modelsDevProviderId": "openai",
        "baseProvider": "openai",
        "base-url": base_url(host, port),
        "defaultModel": MODEL_ID,
        "description": "LUCIDOTA LOCAL BONSAI TERNARY via PrismML llama.cpp Q2_0 server; local/offline only.",
        "apiKeyEnv": "LLXPRT_LOCAL_API_KEY",
        "staticModels": [{"id": MODEL_ID, "name": "Ternary Bonsai 4B Q2_0 local"}],
    }


def llxprt_profile_config() -> dict[str, Any]:
    return {
        "version": 1,
        "provider": "lucidota-local",
        "model": MODEL_ID,
        "modelParams": {"temperature": 0.2, "max_tokens": 512},
        "ephemeralSettings": {
            "streaming": "enabled",
            "context-limit": 1024,
            "socket-timeout": 300000,
            "socket-keepalive": True,
            "socket-nodelay": True,
            "tool-output-max-items": 50,
            "tool-output-max-tokens": 30000,
            "shell-default-timeout-seconds": 120,
        },
        "auth": {"type": "none"},
    }


def write_llxprt_config(*, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, home: Path | None = None) -> dict[str, str]:
    root = home or Path.home()
    provider_path = root / ".llxprt" / "providers" / "lucidota-local.config"
    profile_path = root / ".llxprt" / "profiles" / "lucidota-local.json"
    provider_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    provider_path.write_text(json.dumps(llxprt_provider_config(host=host, port=port), indent=2, sort_keys=False) + "\n", encoding="utf-8")
    profile_path.write_text(json.dumps(llxprt_profile_config(), indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return {"provider": str(provider_path), "profile": str(profile_path)}


def health_url(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    return f"http://{host}:{int(port)}/health"


def health(*, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 2.0) -> dict[str, Any]:
    url = health_url(host, port)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            raw = response.read(4096).decode("utf-8", "replace")
        return {"ok": True, "url": url, "status": response.status, "raw": raw}
    except Exception as exc:
        return {"ok": False, "url": url, "error": f"{type(exc).__name__}:{exc}"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LOCAL BONSAI TERNARY llama.cpp wiring helper")
    parser.add_argument("--host", default=os.environ.get("LUCIDOTA_BONSAI_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LUCIDOTA_BONSAI_PORT", DEFAULT_PORT)))
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("config", help="Print runtime config JSON")
    sub.add_parser("command", help="Print llama-server command as JSON argv")
    sub.add_parser("health", help="Probe llama-server /health")
    write = sub.add_parser("write-llxprt-config", help="Write ~/.llxprt provider/profile for Bonsai")
    write.add_argument("--home", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "config":
        print(json.dumps(default_runtime_config(host=args.host, port=args.port), indent=2, sort_keys=True))
        return 0
    if args.cmd == "command":
        print(json.dumps(build_server_command(host=args.host, port=args.port), indent=2))
        return 0
    if args.cmd == "health":
        result = health(host=args.host, port=args.port)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("ok") else 4
    if args.cmd == "write-llxprt-config":
        result = write_llxprt_config(host=args.host, port=args.port, home=args.home)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
