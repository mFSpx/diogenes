#!/usr/bin/env python3
"""CLI front door for local model-runner config validation and STUB receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from model_runner_config import validate_model_config  # noqa: E402
from model_runner_stub import run_stub_model  # noqa: E402
from local_model_chat_cli import probe_lane as probe_local_lane  # noqa: E402


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
    groq.add_argument("--run-id")
    groq.add_argument("--max-tokens", type=int)
    groq.add_argument("--temperature", type=float)
    groq.add_argument("--execute", action="store_true")
    groq.add_argument("--no-log-prompts", action="store_true")
    groq.add_argument("--json", action="store_true")
    gemini = sub.add_parser("gemini-chat", help="Call Gemini generateContent through scripts/gemini_chat_cli.py.")
    gemini.add_argument("--prompt", required=True, help="Prompt text, or @path.")
    gemini.add_argument("--system", default="", help="Optional system message text, or @path.")
    gemini.add_argument("--model", default="gemini-2.5-flash")
    gemini.add_argument("--max-tokens", type=int)
    gemini.add_argument("--temperature", type=float)
    gemini.add_argument("--execute", action="store_true")
    gemini.add_argument("--json", action="store_true")
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
    bonsai_chain = sub.add_parser("bonsai-chain", help="Run the explicit Bonsai -> Needles -> Bonsai chain.")
    bonsai_chain.add_argument("--prompt", required=True, help="Prompt text, or @path.")
    bonsai_chain.add_argument("--system", default="", help="Optional system message text, or @path.")
    bonsai_chain.add_argument("--execute", action="store_true")
    bonsai_chain.add_argument("--json", action="store_true")
    validate.add_argument("--json", action="store_true")
    return parser


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _write_chain_receipt(payload: dict[str, Any]) -> Path:
    out = ROOT / "05_OUTPUTS" / "model_invocations"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"model_runner_bonsai_chain_{_stamp()}.json"
    payload["report_path"] = str(path.relative_to(ROOT))
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def run_bonsai_chain(
    *,
    prompt: str,
    system: str = "",
    execute: bool = False,
    lane_runner=probe_local_lane,
) -> dict[str, Any]:
    lane_sequence = ["bonsai_q1_0", "needle_0", "needle_1", "needle_2", "needle_3", "needle_4", "needle_5", "bonsai_q1_0"]
    stage_receipts: list[dict[str, Any]] = []

    first = lane_runner(lane="bonsai_q1_0", prompt=prompt, system=system, execute=execute)
    stage_receipts.append(first)

    needle_prompt = json.dumps(
        {
            "stage": "bonsai_1_to_needles",
            "seed_text": first.get("text", ""),
            "prompt": prompt,
            "system": system,
            "chain_digest": _sha({"prompt": prompt, "system": system, "first_output": first.get("text", "")}),
        },
        sort_keys=True,
    )

    needle_outputs: list[dict[str, Any]] = []
    for needle_lane in lane_sequence[1:7]:
        receipt = lane_runner(
            lane=needle_lane,
            prompt=needle_prompt,
            system="Return compact structured JSON only.",
            execute=execute,
            max_tokens=64,
        )
        stage_receipts.append(receipt)
        needle_outputs.append(receipt)

    needle_digest = _sha(
        {
            "lane_sequence": lane_sequence[1:7],
            "outputs": [r.get("text", "") for r in needle_outputs],
        }
    )

    second_prompt = json.dumps(
        {
            "stage": "needles_to_bonsai_2_mutate",
            "seed_text": first.get("text", ""),
            "needle_digest": needle_digest,
            "needle_outputs": [r.get("text", "") for r in needle_outputs],
            "prompt": prompt,
            "system": system,
        },
        sort_keys=True,
    )

    final = lane_runner(lane="bonsai_q1_0", prompt=second_prompt, system=system, execute=execute)
    stage_receipts.append(final)

    blockers = [b for receipt in stage_receipts for b in receipt.get("blockers", []) if isinstance(receipt, dict)]
    status = "PASS" if all(r.get("status") == "PASS" for r in stage_receipts) else "BLOCKED"
    payload: dict[str, Any] = {
        "schema": "lucidota.model_invocation.bonsai_chain.v1",
        "generated_at": _now(),
        "mode": "execute" if execute else "dry_run",
        "status": status,
        "prompt": prompt,
        "system": system,
        "lane_sequence": lane_sequence,
        "stage_receipts": stage_receipts,
        "needle_stage_count": 6,
        "needle_digest": needle_digest,
        "final_lane": "bonsai_q1_0",
        "merge_strategy": "deterministic_needle_digest",
        "execute_performed": bool(execute),
        "model_calls_performed": bool(execute),
        "canonical_graph_writes_performed": False,
        "blockers": blockers,
        "input_sha256": _sha({"prompt": prompt, "system": system}),
        "output_sha256": _sha({"final_output": final.get("text", ""), "needle_digest": needle_digest}),
        "text": final.get("text", ""),
    }
    _write_chain_receipt(payload)
    return payload


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "validate":
        config = load_config(args)
        result = validate_model_config(config)
        if args.json:
            print(json.dumps(result, sort_keys=True))
        else:
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
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if proc.stdout:
            sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        return proc.returncode
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
        if args.run_id:
            cmd += ["--run-id", args.run_id]
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
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if proc.stdout:
            sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        return proc.returncode
    if args.cmd == "gemini-chat":
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "gemini_chat_cli.py"),
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
        if args.json:
            cmd.append("--json")
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if proc.stdout:
            sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        return proc.returncode
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
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if proc.stdout:
            sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        return proc.returncode
    if args.cmd == "bonsai-chain":
        payload = run_bonsai_chain(prompt=args.prompt, system=args.system, execute=args.execute)
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            if payload.get("text"):
                print(payload["text"])
            print("RECEIPT_PATH=" + payload["report_path"])
            print("BONSAI_CHAIN=" + payload["status"])
        return 0 if payload["status"] == "PASS" else 4
    config = load_config(args)
    result = run_stub_model(config, args.prompt)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        if result.get("receipt_path"):
            print("RECEIPT_PATH=" + str(result["receipt_path"]))
        print("MODEL_RUNNER_STUB=" + str(result.get("status", "UNKNOWN")))
    return 0 if result.get("status") == "PASSED" else 4


if __name__ == "__main__":
    raise SystemExit(main())
