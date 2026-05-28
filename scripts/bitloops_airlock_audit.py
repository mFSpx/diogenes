#!/usr/bin/env python3
"""Sovereign airlock audit for Bitloops before any LUCIDOTA install/enable."""
from __future__ import annotations

import argparse, json, os, re, shutil, tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "bitloops"
CURL_PIPE = re.compile(r"\b(?:curl|wget)\b[^\n|]{0,200}\|\s*(?:sh|bash)\b", re.I)
RELEASE_LATEST = re.compile(r"releases/latest|/latest\b|TAG=.*latest|API_URL=.*latest", re.I)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path | None) -> str | None:
    if not path:
        return None
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read(path: Path | None) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path and path.exists() else ""


def load_toml(path: Path | None) -> dict[str, Any]:
    text = read(path)
    if not text:
        return {}
    return tomllib.loads(text)


def walk(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(obj, dict):
        out: list[tuple[str, Any]] = []
        for key, value in obj.items():
            out.extend(walk(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return [(prefix, obj)]


def val(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def add_once(blockers: list[str], blocker: str) -> None:
    if blocker not in blockers:
        blockers.append(blocker)


def build_report(
    *,
    source_tag: str,
    source_commit: str,
    install_script: Path | None = None,
    daemon_config: Path | None = None,
    repo_policy: Path | None = None,
    telemetry_optout_env: bool | None = None,
    allow_remote: bool = False,
    allow_platform: bool = False,
    require_binary: bool = False,
) -> dict[str, Any]:
    blockers: list[str] = []
    binary_path = shutil.which("bitloops")
    if require_binary and not binary_path:
        add_once(blockers, "bitloops_binary_missing")
    if not source_tag or source_tag.lower() in {"latest", "main", "head"} or not re.fullmatch(r"v?\d+\.\d+\.\d+", source_tag):
        add_once(blockers, "source_not_pinned")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit or ""):
        add_once(blockers, "source_not_pinned")
    install_text = read(install_script)
    if CURL_PIPE.search(install_text):
        add_once(blockers, "curl_pipe_shell_install_detected")
    if RELEASE_LATEST.search(install_text):
        add_once(blockers, "release_latest_installer_detected")

    cfg, policy = load_toml(daemon_config), load_toml(repo_policy)
    optout = bool(telemetry_optout_env) or os.environ.get("BITLOOPS_TELEMETRY_OPTOUT") in {"1", "true", "TRUE", "yes"}
    if val(cfg, "telemetry", "enabled") is not False and not optout:
        add_once(blockers, "telemetry_not_opted_out")
    if not allow_remote:
        if val(cfg, "stores", "relational", "postgres_dsn"):
            add_once(blockers, "remote_relational_store_detected")
        if val(cfg, "stores", "events", "clickhouse_url"):
            add_once(blockers, "remote_event_store_detected")
        if val(cfg, "stores", "blob", "s3_bucket") or val(cfg, "stores", "blob", "gcs_bucket"):
            add_once(blockers, "remote_blob_store_detected")
    flat_text = json.dumps(walk(cfg), sort_keys=True, default=str)
    if not allow_platform and re.search(r"bitloops_platform|platform_chat|PLATFORM_GATEWAY|gateway-url|https://gateway", flat_text, re.I):
        add_once(blockers, "platform_inference_detected")
    mode = str(val(policy, "semantic_clones", "embedding_mode") or "off").lower()
    if mode not in {"off", "local", "local_only"}:
        add_once(blockers, "repo_embeddings_not_off_or_local")

    return {
        "schema": "lucidota.bitloops.airlock_audit.v1",
        "generated_at": now(),
        "status": "PASS" if not blockers else "FAIL",
        "sovereign_mode": not blockers,
        "source": {"tag": source_tag, "commit": source_commit},
        "inputs": {"install_script": rel(install_script), "daemon_config": rel(daemon_config), "repo_policy": rel(repo_policy)},
        "binary": {"required": require_binary, "path": binary_path},
        "blockers": blockers,
        "checks": {
            "bitloops_binary_present": bool(binary_path),
            "source_pinned": "source_not_pinned" not in blockers,
            "curl_pipe_shell_install_absent": "curl_pipe_shell_install_detected" not in blockers,
            "release_latest_installer_absent": "release_latest_installer_detected" not in blockers,
            "telemetry_opted_out": "telemetry_not_opted_out" not in blockers,
            "remote_stores_absent": not any(b.startswith("remote_") for b in blockers),
            "platform_inference_absent": "platform_inference_detected" not in blockers,
            "repo_embeddings_off_or_local": "repo_embeddings_not_off_or_local" not in blockers,
        },
        "model_calls_performed": False,
        "canonical_graph_writes_performed": False,
    }


def write_report(report: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / f"bitloops_airlock_audit_{stamp()}.json"
    path = base
    n = 1
    while path.exists():
        path = base.with_name(f"{base.stem}_{n}{base.suffix}")
        n += 1
    report["report_path"] = rel(path)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("REPORT_PATH=" + str(report["report_path"]))
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-tag", required=True)
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--install-script")
    ap.add_argument("--daemon-config")
    ap.add_argument("--repo-policy")
    ap.add_argument("--telemetry-optout-env", action="store_true")
    ap.add_argument("--allow-remote", action="store_true")
    ap.add_argument("--allow-platform", action="store_true")
    ap.add_argument("--require-binary", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    report = build_report(
        source_tag=a.source_tag,
        source_commit=a.source_commit,
        install_script=Path(a.install_script) if a.install_script else None,
        daemon_config=Path(a.daemon_config) if a.daemon_config else None,
        repo_policy=Path(a.repo_policy) if a.repo_policy else None,
        telemetry_optout_env=a.telemetry_optout_env,
        allow_remote=a.allow_remote,
        allow_platform=a.allow_platform,
        require_binary=a.require_binary,
    )
    write_report(report)
    if a.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print("BITLOOPS_AIRLOCK=" + report["status"])
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
