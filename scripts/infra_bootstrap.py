#!/usr/bin/env python3
"""Validate and bootstrap the local AI/DB stack without touching kernel paths."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime_dsns import resolve_state_dsn

OUT_DIR = ROOT / "05_OUTPUTS" / "goals"
DEFAULT_OLLAMA_MODELS = ("qwen2.5:1.5b-instruct",)
MODEL_CACHE_ROOTS = [ROOT / "03_VAULT" / "models", ROOT / "04_RUNTIME" / "models", Path.home() / ".cache" / "huggingface" / "hub"]
REQUIRED_LOCAL_MODELS = {
    "embeddings": {
        "repo_id": "BAAI/bge-m3",
        "env_path": "LUCIDOTA_BGE_MODEL_PATH",
        "env_id": "LUCIDOTA_BGE_MODEL_ID",
        "local_dir": ROOT / "04_RUNTIME" / "models" / "bge-m3",
        "suggested_layout": "04_RUNTIME/models/bge-m3/",
    },
    "encoder_ner": {
        "repo_id": "answerdotai/ModernBERT-base",
        "env_path": "LUCIDOTA_MODERNBERT_MODEL_PATH",
        "env_id": "LUCIDOTA_MODERNBERT_MODEL_ID",
        "local_dir": ROOT / "04_RUNTIME" / "models" / "modernbert-base",
        "suggested_layout": "04_RUNTIME/models/modernbert-base/",
    },
    "vision_parser": {
        "repo_id": "ds4sd/SmolDocling-256M-preview",
        "alt_repo_id": "ibm-granite/granite-docling-258M",
        "env_path": "LUCIDOTA_SMOLDOCLING_MODEL_PATH",
        "env_id": "LUCIDOTA_SMOLDOCLING_MODEL_ID",
        "local_dir": ROOT / "04_RUNTIME" / "models" / "smoldocling-256m-preview",
        "suggested_layout": "04_RUNTIME/models/smoldocling-256m-preview/",
    },
}


@dataclass
class CheckResult:
    name: str
    ok: bool
    action: str = ""
    detail: dict[str, Any] | None = None


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def out_path() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / f"infra_bootstrap_{stamp()}.json"


def run(cmd: list[str], *, timeout: int = 300, check: bool = False, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=check, env=env)


def sudo_password() -> str | None:
    return os.environ.get("LUCIDOTA_SUDO_PASSWORD")


def sudo_prefix() -> list[str]:
    pw = sudo_password()
    if pw:
        return ["sudo", "-S", "-p", ""]
    return ["sudo", "-n"]


def run_sudo(cmd: list[str], *, timeout: int = 1200) -> subprocess.CompletedProcess[str]:
    full = sudo_prefix() + cmd
    env = None
    stdin = None
    if sudo_password():
        stdin = subprocess.PIPE
    proc = subprocess.run(full, text=True, input=(sudo_password() + "\n") if sudo_password() else None, capture_output=True, timeout=timeout, env=env)
    return proc


def which_ollama() -> str | None:
    for candidate in [shutil.which("ollama"), "/usr/local/bin/ollama", "/usr/bin/ollama"]:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def ollama_list(binary: str) -> str:
    proc = run([binary, "list"], timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "ollama list failed")
    return proc.stdout


def install_ollama() -> CheckResult:
    script = "/tmp/ollama_install.sh"
    dl = run(["curl", "-fsSL", "https://ollama.com/install.sh", "-o", script], timeout=120)
    if dl.returncode != 0:
        return CheckResult("ollama_install", False, "download_failed", {"stderr": dl.stderr[-800:], "stdout": dl.stdout[-800:]})
    proc = run_sudo(["sh", script], timeout=1800)
    ok = proc.returncode == 0
    detail = {"stdout": proc.stdout[-1200:], "stderr": proc.stderr[-1200:], "script": script}
    if ok:
        svc = run_sudo(["systemctl", "enable", "--now", "ollama"], timeout=180)
        detail["systemctl"] = {"returncode": svc.returncode, "stdout": svc.stdout[-800:], "stderr": svc.stderr[-800:]}
        ok = svc.returncode == 0 or "already" in (svc.stderr + svc.stdout).lower()
    return CheckResult("ollama_install", ok, "official_vendor_install", detail)


def ensure_ollama(binary: str | None) -> CheckResult:
    if binary:
        try:
            ver = run([binary, "--version"], timeout=20)
            if ver.returncode != 0:
                return CheckResult("ollama_binary", False, "version_failed", {"stderr": ver.stderr[-400:], "stdout": ver.stdout[-400:]})
        except FileNotFoundError:
            return CheckResult("ollama_binary", False, "missing", {"expected": ["/usr/local/bin/ollama", "/usr/bin/ollama"]})
        return CheckResult("ollama_binary", True, "present", {"binary": binary, "version": (ver.stdout or ver.stderr).strip()[:200], "service": service_state()})
    inst = install_ollama()
    if not inst.ok:
        return inst
    binary = which_ollama()
    if not binary:
        return CheckResult("ollama_binary", False, "installed_but_missing_from_path", {"hint": "ollama should be on PATH after install"})
    return CheckResult("ollama_binary", True, "installed", {"binary": binary, "version": run([binary, "--version"], timeout=20).stdout.strip()[:200], "service": service_state()})


def service_state() -> dict[str, Any]:
    state = {"active": None, "enabled": None}
    for key, cmd in (("active", ["systemctl", "is-active", "ollama"]), ("enabled", ["systemctl", "is-enabled", "ollama"])):
        if shutil.which("systemctl"):
            r = run(cmd, timeout=15)
            state[key] = (r.stdout or r.stderr).strip() or r.returncode
    return state


def pull_ollama_models(binary: str, targets: list[str]) -> CheckResult:
    pulled = []
    skipped = []
    errors = []
    try:
        current = ollama_list(binary)
    except Exception as exc:
        current = f"list_error:{exc}"
    current_text = current.lower()
    for model in targets:
        model_l = model.lower()
        if model_l in current_text:
            skipped.append(model)
            continue
        proc = run([binary, "pull", model], timeout=2400)
        if proc.returncode == 0:
            pulled.append(model)
        else:
            errors.append({"model": model, "stderr": proc.stderr[-1200:], "stdout": proc.stdout[-1200:]})
    ok = not errors
    return CheckResult("ollama_models", ok, "pulled_missing_models", {"targets": targets, "pulled": pulled, "skipped": skipped, "errors": errors, "current": current[-4000:]})


def cache_layout_for(name: str, patterns: tuple[str, ...]) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    for root in MODEL_CACHE_ROOTS:
        if not root.exists():
            continue
        for pat in patterns:
            for p in root.rglob(pat):
                if p.is_file():
                    hits.append({"path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p), "bytes": p.stat().st_size})
    return {"name": name, "hits": hits[:40], "present": bool(hits)}

def model_layout_check(spec: dict[str, Any], *, patterns: tuple[str, ...], execute: bool) -> CheckResult:
    env_path = os.environ.get(spec["env_path"])
    env_id = os.environ.get(spec["env_id"])
    local_dir: Path = spec["local_dir"]
    checks: dict[str, Any] = {
        "repo_id": spec["repo_id"],
        "alt_repo_id": spec.get("alt_repo_id"),
        "env_path": env_path,
        "env_id": env_id,
        "local_dir": str(local_dir.relative_to(ROOT)),
        "cache": cache_layout_for(spec["repo_id"], patterns),
    }
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = ROOT / p
        checks["resolved_path"] = str(p)
        checks["exists"] = p.exists()
        if p.exists():
            return CheckResult(spec["repo_id"], True, "env_path_checked", checks)
    if local_dir.exists():
        checks["resolved_path"] = str(local_dir)
        checks["exists"] = True
        return CheckResult(spec["repo_id"], True, "local_dir_present", checks)
    if checks["cache"]["present"]:
        return CheckResult(spec["repo_id"], True, "cache_present", checks)
    if execute:
        download = download_hf_model(spec["repo_id"], local_dir, alt_repo_id=spec.get("alt_repo_id"))
        checks["download"] = download.detail
        if download.ok:
            checks["resolved_path"] = str(local_dir)
            checks["exists"] = True
            return CheckResult(spec["repo_id"], True, "downloaded", checks)
        return CheckResult(spec["repo_id"], False, download.action, checks | {"download_error": download.detail})
    checks["layout_needed"] = {
        "expected_env": f'{spec["env_path"]} or {spec["env_id"]}',
        "suggested_layout": spec["suggested_layout"],
        "minimum_files": ["config.json", "tokenizer.json", "model.safetensors|pytorch_model.bin"],
        "note": "Choose and approve exact HF repo ID and local layout before execution.",
    }
    return CheckResult(spec["repo_id"], False, "missing_weights", checks)


def download_hf_model(repo_id: str, local_dir: Path, *, alt_repo_id: str | None = None) -> CheckResult:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        return CheckResult(repo_id, False, "huggingface_hub_missing", {"error": str(exc), "pip_hint": f"{sys.executable} -m pip install huggingface_hub"})
    local_dir.mkdir(parents=True, exist_ok=True)
    repos = [repo_id] + ([alt_repo_id] if alt_repo_id else [])
    errors = []
    for chosen_repo in repos:
        if not chosen_repo:
            continue
        try:
            snapshot_download(
                repo_id=chosen_repo,
                repo_type="model",
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            return CheckResult(chosen_repo, True, "snapshot_download", {"repo_id": chosen_repo, "local_dir": str(local_dir), "alt_repo_id": alt_repo_id})
        except Exception as exc:
            errors.append({"repo_id": chosen_repo, "error": str(exc)})
    return CheckResult(repo_id, False, "snapshot_download_failed", {"local_dir": str(local_dir), "errors": errors})


def smoldocling_check(auto_install: bool) -> CheckResult:
    # SmolDocling is served via the `docling` package (docling + docling-ibm-models).
    # There is no standalone `smoldocling` pip package; checking for `smoldocling`
    # module was wrong. Check for `docling` instead.
    for module in ("docling", "docling_ibm_models"):
        if importlib.util.find_spec(module) is None:
            detail: dict[str, Any] = {"module": module, "note": "install via: pip install docling"}
            if auto_install:
                proc = run([sys.executable, "-m", "pip", "install", "docling"], timeout=2400)
                detail["pip"] = {"returncode": proc.returncode, "stdout": proc.stdout[-800:], "stderr": proc.stderr[-800:]}
                if proc.returncode == 0:
                    return CheckResult("smoldocling", True, "docling_pip_installed", detail)
            return CheckResult("smoldocling", False, "missing_dependency", detail)
    model_dir = ROOT / "04_RUNTIME" / "models" / "smoldocling-256m-preview"
    detail = {"module": "docling", "module_present": True, "model_dir": str(model_dir), "model_dir_exists": model_dir.exists()}
    return CheckResult("smoldocling", True, "docling_present", detail)


def dbos_attach(*, execute: bool, auto_install_pool: bool) -> CheckResult:
    dsn = resolve_state_dsn()
    pool_present = importlib.util.find_spec("psycopg_pool") is not None
    if not pool_present and auto_install_pool:
        proc = run([sys.executable, "-m", "pip", "install", "psycopg-pool"], timeout=1200)
        pool_present = proc.returncode == 0 and importlib.util.find_spec("psycopg_pool") is not None
        install_detail = {"stdout": proc.stdout[-1200:], "stderr": proc.stderr[-1200:], "installed": pool_present}
    else:
        install_detail = {"installed": pool_present}
    detail: dict[str, Any] = {"dsn": dsn, "pool_present": pool_present, "install": install_detail}
    if not pool_present:
        return CheckResult("dbos_attach", False, "connection_pool_missing", detail)
    try:
        from psycopg_pool import ConnectionPool
    except Exception as exc:
        return CheckResult("dbos_attach", False, "import_failed", {**detail, "error": str(exc)})

    if not execute:
        detail["mode"] = "dry_run"
        return CheckResult("dbos_attach", True, "validated", detail)

    workflow_id = "infra_bootstrap"
    run_id = stamp()
    payload = {"workflow_id": workflow_id, "run_id": run_id, "phase": "dbos_attach", "status": "succeeded", "source": "infra_bootstrap", "detail": {"dsn": dsn, "pool": "ConnectionPool", "governor": "unchanged"}}
    with ConnectionPool(conninfo=dsn, min_size=1, max_size=2, timeout=5) as pool:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.execute("SELECT to_regclass('lucidota_control.workflow_event')")
                table = cur.fetchone()[0]
                if not table:
                    return CheckResult("dbos_attach", False, "workflow_event_table_missing", detail)
                cur.execute(
                    """
                    INSERT INTO lucidota_control.workflow_event(workflow_id, run_id, phase, status, source, detail)
                    VALUES (%s,%s,%s,%s,%s,%s::jsonb)
                    RETURNING event_id::text
                    """,
                    (workflow_id, run_id, "dbos_attach", "succeeded", "infra_bootstrap", json.dumps(payload, sort_keys=True)),
                )
                event_id = cur.fetchone()[0]
            conn.commit()
    detail.update({"mode": "execute", "workflow_event_id": event_id, "attached": True})
    return CheckResult("dbos_attach", True, "workflow_event_recorded", detail)


def ensure_ollama_models(execute: bool, binary: str | None) -> CheckResult:
    if not binary:
        return CheckResult("ollama_models", False, "ollama_missing", {"targets": list(DEFAULT_OLLAMA_MODELS), "note": "install ollama first"})
    if not execute:
        try:
            current = ollama_list(binary)
        except Exception as exc:
            return CheckResult("ollama_models", False, "list_failed", {"error": str(exc)})
        missing = [m for m in DEFAULT_OLLAMA_MODELS if m.lower() not in current.lower()]
        return CheckResult("ollama_models", not missing, "listed", {"targets": list(DEFAULT_OLLAMA_MODELS), "missing": missing, "current": current[-4000:]})
    return pull_ollama_models(binary, list(DEFAULT_OLLAMA_MODELS))


def ensure_local_models(execute: bool) -> list[CheckResult]:
    checks: list[CheckResult] = []
    checks.append(model_layout_check(REQUIRED_LOCAL_MODELS["embeddings"], patterns=("*bge*", "*BGE*", "*m3*", "*M3*"), execute=execute))
    checks.append(model_layout_check(REQUIRED_LOCAL_MODELS["encoder_ner"], patterns=("*ModernBERT*", "*modernbert*"), execute=execute))
    checks.append(model_layout_check(REQUIRED_LOCAL_MODELS["vision_parser"], patterns=("*SmolDocling*", "*smoldocling*", "*docling*"), execute=execute))
    return checks


def write_report(report: dict[str, Any]) -> Path:
    path = out_path()
    path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"REPORT_PATH={path.relative_to(ROOT)}")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="infra_bootstrap")
    ap.add_argument("--execute", action="store_true", help="run safe autofix actions rather than validate only")
    ap.add_argument("--auto-install-python-deps", action="store_true", default=True, help="pip install missing Python deps in .venv when obvious")
    ap.add_argument("--no-auto-install-python-deps", action="store_false", dest="auto_install_python_deps")
    ap.add_argument("--auto-install-ollama", action="store_true", default=True, help="install ollama if the binary is missing")
    ap.add_argument("--no-auto-install-ollama", action="store_false", dest="auto_install_ollama")
    args = ap.parse_args(argv)

    checks: list[CheckResult] = []
    checks.append(CheckResult("ollama_install", True, "skipped_not_used", {"note": "llama.cpp serves all models; ollama not used in this stack"}))
    checks.append(CheckResult("ollama_models", True, "skipped_not_used", {"note": "llama.cpp serves all models; ollama not used in this stack"}))
    checks.extend(ensure_local_models(args.execute))
    checks.append(smoldocling_check(auto_install=args.execute and args.auto_install_python_deps))
    checks.append(dbos_attach(execute=args.execute, auto_install_pool=args.execute and args.auto_install_python_deps))

    report = {
        "schema": "lucidota.infra_bootstrap.report.v1",
        "generated_at": now(),
        "root": str(ROOT),
        "execute": args.execute,
        "ollama_binary": None,
        "checks": [asdict(c) for c in checks],
        "passed": all(c.ok for c in checks),
        "blockers": [c.name for c in checks if not c.ok],
    }
    write_report(report)
    print(json.dumps(report, indent=2, sort_keys=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
