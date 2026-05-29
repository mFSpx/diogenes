#!/usr/bin/env python3
"""CLI front door for local model-runner config validation and STUB receipts."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from model_runner_config import validate_model_config  # noqa: E402
from model_runner_stub import run_stub_model  # noqa: E402


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.config_json:
        data = json.loads(args.config_json)
    elif args.config_file:
        path = Path(args.config_file)
        if not path.is_absolute():
            path = ROOT / path
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {
            "model_id": args.model_id,
            "backend": args.backend,
            "requested_vram_mb": args.requested_vram_mb,
            "available_vram_mb": args.available_vram_mb,
        }
    if not isinstance(data, dict):
        raise SystemExit("model config must be a JSON object")
    return data


def add_config_args(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config-json")
    source.add_argument("--config-file")
    parser.add_argument("--model-id", default="local-stub.gguf")
    parser.add_argument("--backend", default="STUB")
    parser.add_argument("--requested-vram-mb", type=int, default=512)
    parser.add_argument("--available-vram-mb", type=int, default=4096)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate or run local model-runner plumbing without loading external services.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    validate = sub.add_parser("validate", help="Validate a model-runner config.")
    add_config_args(validate)
    stub = sub.add_parser("stub", help="Run deterministic STUB model and write invocation receipt.")
    add_config_args(stub)
    stub.add_argument("--prompt", required=True)
    stub.add_argument("--json", action="store_true")
    cohere = sub.add_parser("cohere-chat", help="Call Cohere Chat API through scripts/cohere_chat_cli.py.")
    cohere.add_argument("--prompt", required=True, help="Prompt text, or @path.")
    cohere.add_argument("--system", default="", help="Optional system message text, or @path.")
    cohere.add_argument("--model", default="command-a-03-2025")
    cohere.add_argument("--max-tokens", type=int)
    cohere.add_argument("--temperature", type=float)
    cohere.add_argument("--execute", action="store_true")
    cohere.add_argument("--no-log-prompts", action="store_true")
    cohere.add_argument("--json", action="store_true")
    groq = sub.add_parser("groq-chat", help="Call Groq Chat Completions through scripts/groq_chat_cli.py.")
    groq.add_argument("--prompt", required=True, help="Prompt text, or @path.")
    groq.add_argument("--system", default="", help="Optional system message text, or @path.")
    groq.add_argument("--model", default="llama-3.1-8b-instant")
    groq.add_argument("--max-tokens", type=int)
    groq.add_argument("--temperature", type=float)
    groq.add_argument("--execute", action="store_true")
    groq.add_argument("--no-log-prompts", action="store_true")
    groq.add_argument("--json", action="store_true")
    local = sub.add_parser("local-chat", help="Call a local LUCIDOTA llama.cpp or Needle lane through scripts/local_model_chat_cli.py.")
    local.add_argument("--lane", required=True)
    local.add_argument("--prompt", required=True, help="Prompt text, or @path.")
    local.add_argument("--system", default="", help="Optional system message text, or @path.")
    local.add_argument("--clear-history", action="store_true", help="Compatibility no-op for callers that request a hard context reset on local lanes.")
    local.add_argument("--max-tokens", type=int, default=16)
    local.add_argument("--temperature", type=float, default=0.0)
    local.add_argument("--timeout-sec", type=float, default=60.0)
    local.add_argument("--execute", action="store_true")
    local.add_argument("--no-log-prompts", action="store_true")
    local.add_argument("--json", action="store_true")
    validate.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "validate":
        config = load_config(args)
        result = validate_model_config(config)
        if args.json:
            print(json.dumps(result, sort_keys=True))
        print("MODEL_CONFIG=PASS" if result.get("ok") else "MODEL_CONFIG=FAIL")
        return 0 if result.get("ok") else 4
    if args.cmd == "cohere-chat":
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "cohere_chat_cli.py"),
            "--prompt",
            args.prompt,
            "--system",
            args.system,
            "--model",
            args.model,
        ]
        if args.max_tokens is not None:
            cmd += ["--max-tokens", str(args.max_tokens)]
        if args.temperature is not None:
            cmd += ["--temperature", str(args.temperature)]
        if args.execute:
            cmd.append("--execute")
        if args.no_log_prompts:
            cmd.append("--no-log-prompts")
        if args.json:
            cmd.append("--json")
        return subprocess.run(cmd, cwd=ROOT).returncode
    if args.cmd == "groq-chat":
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "groq_chat_cli.py"),
            "--prompt",
            args.prompt,
            "--system",
            args.system,
            "--model",
            args.model,
        ]
        if args.max_tokens is not None:
            cmd += ["--max-tokens", str(args.max_tokens)]
        if args.temperature is not None:
            cmd += ["--temperature", str(args.temperature)]
        if args.execute:
            cmd.append("--execute")
        if args.no_log_prompts:
            cmd.append("--no-log-prompts")
        if args.json:
            cmd.append("--json")
        return subprocess.run(cmd, cwd=ROOT).returncode
    if args.cmd == "local-chat":
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "local_model_chat_cli.py"),
            "--lane",
            args.lane,
            "--prompt",
            args.prompt,
            "--system",
            args.system,
            "--max-tokens",
            str(args.max_tokens),
            "--temperature",
            str(args.temperature),
            "--timeout-sec",
            str(args.timeout_sec),
        ]
        if args.execute:
            cmd.append("--execute")
        if args.clear_history:
            cmd += ["--clear-history"]
        if args.no_log_prompts:
            cmd.append("--no-log-prompts")
        if args.json:
            cmd.append("--json")
        return subprocess.run(cmd, cwd=ROOT).returncode
    config = load_config(args)
    result = run_stub_model(config, args.prompt)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    if result.get("receipt_path"):
        print("RECEIPT_PATH=" + str(result["receipt_path"]))
    print("MODEL_RUNNER_STUB=" + str(result.get("status", "UNKNOWN")))
    return 0 if result.get("status") == "PASSED" else 4


if __name__ == "__main__":
    raise SystemExit(main())
