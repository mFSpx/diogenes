#!/usr/bin/env python3
"""Project 2501 admin-prompt compiler/enforcer for LUCIDOTA model calls."""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "00_PROJECT_BRAIN" / "PROJECT_2501_ADMIN_PROMPT.md"
OUT = ROOT / "05_OUTPUTS" / "model_admin_prompt"
PROMPT_ID = "project2501_major_admin_v1"
TARGET_SURFACES = [
    "model_runner_cli.groq-chat",
    "model_runner_cli.cohere-chat",
    "model_runner_cli.local-chat",
    "groq_chat_cli",
    "cohere_chat_cli",
    "local_model_chat_cli",
    "model_runner_stub",
    "goal_agent_packet",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compose_system_prompt(caller_system: str | None = "") -> tuple[str, dict[str, Any]]:
    admin = read_prompt().strip()
    caller = (caller_system or "").strip()
    effective = admin
    if caller:
        effective += "\n\n---\n## CALLER_SYSTEM_CONTEXT_NON_AUTHORITY\n\n" + caller
    policy = {
        "schema": "lucidota.project2501.admin_prompt_policy.v1",
        "prompt_id": PROMPT_ID,
        "prompt_path": rel(PROMPT_PATH),
        "prompt_sha256": sha256_text(admin),
        "enforced": True,
        "exclusive_admin_prompt": True,
        "caller_system_demoted_to_context": bool(caller),
        "caller_system_chars": len(caller),
        "effective_system_chars": len(effective),
    }
    return effective, policy


def build_distribution_packet() -> dict[str, Any]:
    admin = read_prompt().strip()
    return {
        "schema": "lucidota.project2501.admin_prompt_distribution.v1",
        "generated_at": now(),
        "prompt_id": PROMPT_ID,
        "prompt_path": rel(PROMPT_PATH),
        "prompt_sha256": sha256_text(admin),
        "target_surfaces": TARGET_SURFACES,
        "distribution_scope": "repo_owned_model_invocation_surfaces",
        "external_system_prompt_mutated": False,
        "policy": "Project 2501 is the sole LUCIDOTA admin layer; caller system text is demoted to non-authority context.",
    }


def write_receipt(packet: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"project2501_admin_prompt_{stamp()}.json"
    packet["report_path"] = rel(path)
    path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit/inspect the Project 2501 admin-prompt distribution packet.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    emit = sub.add_parser("emit")
    emit.add_argument("--json", action="store_true")
    show = sub.add_parser("show")
    show.add_argument("--caller-system", default="")
    args = ap.parse_args()
    if args.cmd == "show":
        system, policy = compose_system_prompt(args.caller_system)
        print(system)
        print(json.dumps(policy, sort_keys=True))
        return 0
    packet = build_distribution_packet()
    path = write_receipt(packet)
    if args.json:
        print(json.dumps(packet, sort_keys=True))
    print("REPORT_PATH=" + rel(path))
    print("PROJECT2501_ADMIN_PROMPT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
