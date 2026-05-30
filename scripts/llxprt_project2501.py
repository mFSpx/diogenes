#!/usr/bin/env python3
"""Project 2501 LLxprt/Groq orchestrator setup and launcher."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "llxprt_project2501"
PROFILE_NAME = "lucidota-groq-orchestrator"
ALIAS_NAME = "lucidota-groq"
GROQ_BASE_URL = "https://api.groq.com/openai/v1/"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_CONTEXT_LIMIT = 131_072
DEFAULT_MAX_TOKENS = 8_192
DEFAULT_TEMPERATURE = 0.2
MANAGED_START = "<!-- BEGIN LUCIDOTA PROJECT2501 LLXPRT ORCHESTRATOR -->"
MANAGED_END = "<!-- END LUCIDOTA PROJECT2501 LLXPRT ORCHESTRATOR -->"

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
try:
    from groq_env import load_groq_env  # type: ignore
except Exception:  # pragma: no cover - defensive when copied outside repo
    def load_groq_env() -> None:
        return


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str, root: Path = ROOT) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_json(path: Path, data: dict[str, Any], *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if mode is not None:
        path.chmod(mode)


def groq_model(model: str | None = None) -> str:
    # LLXPRT uses its own dedicated model env var; GROQ_MODEL is a general shorthand
    # and should not override the orchestrator-specific default.
    return model or os.environ.get("LLXPRT_GROQ_MODEL") or DEFAULT_MODEL


def groq_base_url() -> str:
    return os.environ.get("GROQ_BASE_URL", GROQ_BASE_URL).rstrip("/") + "/"


def provider_alias(model: str) -> dict[str, Any]:
    return {
        "baseProvider": "openai",
        "base-url": groq_base_url(),
        "defaultModel": model,
        "description": "LUCIDOTA Project 2501 Groq orchestrator provider alias",
        "apiKeyEnv": "GROQ_API_KEY",
    }


def profile(model: str) -> dict[str, Any]:
    return {
        "version": 1,
        "provider": ALIAS_NAME,
        "model": model,
        "modelParams": {
            "temperature": float(os.environ.get("LLXPRT_GROQ_TEMPERATURE", DEFAULT_TEMPERATURE)),
            "max_tokens": int(os.environ.get("LLXPRT_GROQ_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        },
        "ephemeralSettings": {
            "context-limit": int(os.environ.get("LLXPRT_GROQ_CONTEXT_LIMIT", DEFAULT_CONTEXT_LIMIT)),
            "streaming": "enabled",
            "base-url": groq_base_url(),
            "shell-replacement": True,
        },
        "metadata": {
            "orchestrator": "PROJECT 2501",
            "repo": str(ROOT),
            "provider_docs": "https://github.com/vybestack/llxprt-code/blob/main/docs/cli/providers.md",
            "groq_models_docs": "https://console.groq.com/docs/models",
        },
    }


def managed_context(model: str) -> str:
    return f"""{MANAGED_START}

# PROJECT 2501 // LLXPRT GROQ ORCHESTRATOR

You are the LUCIDOTA LLXPRT Groq orchestrator for this repository.

Runtime lane:
- Provider alias: `{ALIAS_NAME}`.
- Base provider: OpenAI-compatible.
- Base URL: `{groq_base_url()}`.
- Model: `{model}`.
- Profile: `{PROFILE_NAME}`.
- Streaming: enabled.

Authority and startup:
1. Follow `AGENTS.md` before writing code.
2. Read `00_PROJECT_BRAIN/TICKLETRUNK.json`, `00_PROJECT_BRAIN/TICKLETRUNK.md`, `00_PROJECT_BRAIN/ACTIVE_SPEC/04_DEV_LIBRARY_REUSE_LAW.md`, and `00_PROJECT_BRAIN/BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md`.
3. Treat `00_PROJECT_BRAIN/PROJECT_2501_ADMIN_PROMPT.md` as the LUCIDOTA admin prompt source.
4. Use `scripts/dev_library_scan.py --query <topic>` before new tools/scripts/schemas/workflows/models/LoRAs/scrapers/skills/plugins/services.

Bare Steel Rule 4:
- DB and Graph are durable truth.
- Read selectively. Persist globally.
- Fetch only localized data required for the immediate task.
- Emit compact events/receipts asynchronously.
- Do not perform broad blocking scans in the hot lane.
- Candidate graph changes stage through graph promotion packets; canonical graph materialization remains gated.

Operating style:
- Be the orchestrator, not a ghost variable.
- Prefer deterministic repo math, tests, and receipts over model improvisation.
- Surface exact provider/model/base URL/temp/max tokens in receipts when calling models.
- If data is missing, say `missing` with a receipt path; do not invent state.

{MANAGED_END}
"""


def merge_context(existing: str, model: str) -> str:
    block = managed_context(model).strip() + "\n"
    if MANAGED_START in existing and MANAGED_END in existing:
        before = existing.split(MANAGED_START, 1)[0].rstrip()
        after = existing.split(MANAGED_END, 1)[1].lstrip()
        return (before + "\n\n" + block + ("\n" + after if after else "")).lstrip()
    return (existing.rstrip() + "\n\n" + block).lstrip()


def merge_settings(path: Path) -> dict[str, Any]:
    settings = read_json(path)
    settings.setdefault("ui", {})
    settings["ui"].update(
        {
            "showMemoryUsage": True,
            "hideModelInfo": False,
            "hideContextSummary": False,
            "showStatusInTitle": True,
            "contextFileName": ["LLXPRT.md", "AGENTS.md", "GOALS/CURRENT_HANDOFF.md"],
            "memoryDiscoveryMaxDepth": settings["ui"].get("memoryDiscoveryMaxDepth", 2),
            "memoryDiscoveryMaxDirs": settings["ui"].get("memoryDiscoveryMaxDirs", 300),
        }
    )
    settings.setdefault("tools", {})
    settings["tools"].update(
        {
            "enableHooks": settings["tools"].get("enableHooks", True),
            "autoAccept": False,
            "enableToolOutputTruncation": True,
            "truncateToolOutputThreshold": 4_000_000,
        }
    )
    settings.setdefault("fileFiltering", {})
    settings["fileFiltering"].update(
        {
            "respectGitIgnore": True,
            "respectLlxprtIgnore": True,
        }
    )
    settings.setdefault("checkpointing", {})
    settings["checkpointing"]["enabled"] = True
    settings["defaultProfile"] = PROFILE_NAME
    settings.setdefault("providerHints", {})
    settings["providerHints"].update({"strategic": ALIAS_NAME, "orchestrator": ALIAS_NAME})
    return settings


def configure_llxprt(*, root: Path = ROOT, home: Path | None = None, model: str | None = None) -> dict[str, Any]:
    root = root.resolve()
    home = (home or Path.home()).expanduser().resolve()
    model = groq_model(model)
    provider_path = home / ".llxprt" / "providers" / f"{ALIAS_NAME}.config"
    profile_path = home / ".llxprt" / "profiles" / f"{PROFILE_NAME}.json"
    settings_path = root / ".llxprt" / "settings.json"
    context_path = root / "LLXPRT.md"

    write_json(provider_path, provider_alias(model), mode=0o600)
    write_json(profile_path, profile(model), mode=0o600)
    write_json(settings_path, merge_settings(settings_path))
    context_path.write_text(merge_context(context_path.read_text(encoding="utf-8") if context_path.exists() else "", model), encoding="utf-8")

    return {
        "schema": "lucidota.llxprt_project2501.configure.v1",
        "generated_at": now(),
        "status": "PASS",
        "root": str(root),
        "home": str(home),
        "provider_alias": ALIAS_NAME,
        "provider_alias_path": str(provider_path),
        "profile": PROFILE_NAME,
        "profile_path": str(profile_path),
        "settings_path": str(settings_path),
        "context_path": str(context_path),
        "groq": {
            "base_url": groq_base_url(),
            "model": model,
            "api_key_env": "GROQ_API_KEY",
            "api_key_env_present": bool(os.environ.get("GROQ_API_KEY")),
        },
        "canonical_graph_writes_performed": False,
    }


def llxprt_binary() -> str | None:
    # Check system PATH first
    found = shutil.which("llxprt")
    if found:
        return found
    # Check npm global bin (common install location)
    for npm_bin in [
        Path.home() / ".npm-global" / "bin" / "llxprt",
        Path.home() / ".local" / "bin" / "llxprt",
        Path("/usr/local/bin/llxprt"),
    ]:
        if npm_bin.exists():
            return str(npm_bin)
    # Check local clone bundle
    local_bundle = ROOT / "01_REPOS" / "llxprt-code" / "bundle" / "llxprt.js"
    if local_bundle.exists() and shutil.which("node"):
        return str(local_bundle)
    return None


def _effective_binary_for_report() -> str | None:
    """Return a truthy binary path/command for reporting purposes.
    Falls back to npx path if llxprt is not installed as a dedicated binary."""
    direct = llxprt_binary()
    if direct:
        return direct
    # If npx is available, the effective launcher is the npx fallback pathway
    return shutil.which("npx") or None


def launch_prompt() -> str:
    return (
        "PROJECT 2501 LLXPRT Groq orchestrator online. "
        "Read LLXPRT.md, AGENTS.md, GOALS/CURRENT_HANDOFF.md, and the active Project 2501/Bare Steel docs. "
        "Confirm provider/model/settings visibility, then wait for operator tasking unless a concrete repo task is already in the prompt."
    )


def launch_command(*, root: Path = ROOT, home: Path | None = None, prompt: str | None = None, dry_run: bool = False) -> tuple[list[str], dict[str, str]]:
    home = (home or Path.home()).expanduser().resolve()
    model = groq_model()
    binary = llxprt_binary()
    cmd = [binary] if binary else ["npx", "-y", "@vybestack/llxprt-code"]
    cmd.extend(
        [
            "--profile-load",
            PROFILE_NAME,
            "--provider",
            ALIAS_NAME,
            "--model",
            model,
            "--prompt-interactive",
            (launch_prompt() if not prompt else "PROJECT 2501 LLXPRT Groq orchestrator task: " + prompt),
        ]
    )
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["LLXPRT_PROJECT2501_PROFILE"] = PROFILE_NAME
    env["LLXPRT_PROJECT2501_PROVIDER"] = ALIAS_NAME
    env["LLXPRT_PROJECT2501_MODEL"] = model
    if dry_run:
        env.pop("GROQ_API_KEY", None)
    return cmd, env


def run_text(cmd: list[str], *, cwd: Path, timeout: float = 5.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout)
        return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout.strip()[-1000:], "stderr": proc.stderr.strip()[-1000:]}
    except Exception as exc:
        return {"cmd": cmd, "error": f"{type(exc).__name__}:{exc}"}


def doctor(*, root: Path = ROOT, home: Path | None = None, model: str | None = None) -> dict[str, Any]:
    load_groq_env()
    home = (home or Path.home()).expanduser().resolve()
    model = groq_model(model)
    binary = llxprt_binary()
    effective_binary = _effective_binary_for_report()
    profile_path = home / ".llxprt" / "profiles" / f"{PROFILE_NAME}.json"
    provider_path = home / ".llxprt" / "providers" / f"{ALIAS_NAME}.config"
    settings_path = root / ".llxprt" / "settings.json"
    context_path = root / "LLXPRT.md"
    payload = {
        "schema": "lucidota.llxprt_project2501.receipt.v1",
        "generated_at": now(),
        "status": "PASS",
        "root": str(root.resolve()),
        "home": str(home),
        "llxprt": {
            "binary": effective_binary,
            "version": run_text([binary, "--version"], cwd=root, timeout=4.0) if binary else None,
            "npx_fallback": "@vybestack/llxprt-code",
            "npm_view_version": run_text(["npm", "view", "@vybestack/llxprt-code", "version"], cwd=root, timeout=8.0) if shutil.which("npm") else None,
            "local_clone": {
                "path": "01_REPOS/llxprt-code",
                "exists": (root / "01_REPOS" / "llxprt-code" / ".git").exists(),
            },
        },
        "node": run_text(["node", "--version"], cwd=root, timeout=4.0) if shutil.which("node") else None,
        "npm": run_text(["npm", "--version"], cwd=root, timeout=4.0) if shutil.which("npm") else None,
        "groq": {
            "base_url": groq_base_url(),
            "model": model,
            "api_key_env": "GROQ_API_KEY",
            "api_key_env_present": bool(os.environ.get("GROQ_API_KEY")),
        },
        "profile": {"name": PROFILE_NAME, "path": str(profile_path), "exists": profile_path.exists()},
        "provider_alias": {"name": ALIAS_NAME, "path": str(provider_path), "exists": provider_path.exists()},
        "project_settings": {"path": str(settings_path), "exists": settings_path.exists()},
        "context": {"path": str(context_path), "exists": context_path.exists()},
        "launch_command": launch_command(root=root, home=home, prompt=launch_prompt(), dry_run=True)[0],
        "canonical_graph_writes_performed": False,
    }
    blockers: list[str] = []
    if not profile_path.exists():
        blockers.append("missing_profile")
    if not provider_path.exists():
        blockers.append("missing_provider_alias")
    if not context_path.exists():
        blockers.append("missing_context_file")
    if not binary and not shutil.which("npx"):
        blockers.append("missing_llxprt_and_npx")
    if blockers:
        payload["status"] = "BLOCKED"
        payload["blockers"] = blockers
    return payload


def write_receipt(payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"llxprt_project2501_{stamp()}.json"
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Configure and launch LLxprt as the LUCIDOTA Project 2501 Groq orchestrator.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("configure", "doctor", "launch"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, default=ROOT)
        p.add_argument("--home", type=Path, default=Path.home())
        p.add_argument("--model", default=None)
        p.add_argument("--json", action="store_true")
    launch = sub.choices["launch"]
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--prompt", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_groq_env()
    if args.cmd == "configure":
        payload = configure_llxprt(root=args.root, home=args.home, model=args.model)
        path = write_receipt(payload)
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        print("REPORT_PATH=" + rel(path))
        print("LLXPRT_PROJECT2501=" + payload["status"])
        return 0
    if args.cmd == "doctor":
        payload = doctor(root=args.root, home=args.home, model=args.model)
        path = write_receipt(payload)
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        print("REPORT_PATH=" + rel(path))
        print("LLXPRT_PROJECT2501=" + payload["status"])
        return 0 if payload["status"] == "PASS" else 4
    configure_llxprt(root=args.root, home=args.home, model=args.model)
    cmd, env = launch_command(root=args.root, home=args.home, prompt=args.prompt, dry_run=args.dry_run)
    payload = doctor(root=args.root, home=args.home, model=args.model)
    payload["launch_command"] = cmd
    payload["dry_run"] = bool(args.dry_run)
    path = write_receipt(payload)
    if args.json or args.dry_run:
        print(json.dumps(payload, sort_keys=True))
    print("REPORT_PATH=" + rel(path))
    print("LLXPRT_PROJECT2501=" + payload["status"])
    print("COMMAND=" + " ".join(json.dumps(part) if " " in part else part for part in cmd))
    if args.dry_run:
        return 0 if payload["status"] == "PASS" else 4
    os.chdir(args.root)
    os.execvpe(cmd[0], cmd, env)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
