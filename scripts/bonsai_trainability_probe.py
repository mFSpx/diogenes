#!/usr/bin/env python3
"""Loader-proof trainability probe for the exact Prism ML Bonsai GGUF target.

This probe is intentionally narrow:
- it only accepts the exact repo/file pair already verified on the pod,
- it records a blocker when the target is not the exact Prism ML GGUF file,
- and it preserves the current blocked state because the loader evidence shows
  the exact target is not trainable with the available llama.cpp loader path.

The probe does not mutate graphs or databases.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lucidota_bonsai_ternary_handler import build_server_command

EXACT_REPO = "prism-ml/Ternary-Bonsai-8B-gguf"
EXACT_FILE = "Ternary-Bonsai-8B-Q2_0.gguf"
DEFAULT_MODEL_SOURCE = ROOT / "03_VAULT" / "models" / "prism-ml" / "Ternary-Bonsai-8B-gguf" / "model_source.json"
DEFAULT_LOADER_LOG = ROOT / "04_RUNTIME" / "inference_os" / "bonsai_standard_cuda_probe.log"
DEFAULT_RECEIPT = ROOT / "05_OUTPUTS" / "runtime" / "bonsai_trainability_probe_latest.json"
DEFAULT_MODEL_PATH = ROOT / "03_VAULT" / "models" / "prism-ml" / "Ternary-Bonsai-8B-gguf" / EXACT_FILE
DEFAULT_LOADER_BINARY = ROOT / "01_REPOS" / "prismml_llama.cpp" / "build-cuda" / "bin" / "llama-server"

INVALID_GGML_TYPE_RE = re.compile(r"invalid ggml type (?P<type>\d+)")
LOADING_MODEL_RE = re.compile(r"loading model '?(?P<path>[^']+)'?")
FAILED_MODEL_RE = re.compile(r"failed to load model from (?P<path>.+)")


def now_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_loader_failure(log_text: str, *, model_path: Path, loader_command: list[str]) -> dict[str, Any] | None:
    lines = [line.rstrip() for line in log_text.splitlines() if line.strip()]
    failure_mode = None
    loader_path = None
    evidence: list[str] = []

    for line in lines:
        if loader_path is None:
            m_load = LOADING_MODEL_RE.search(line)
            if m_load:
                loader_path = m_load.group("path").strip()
        m_type = INVALID_GGML_TYPE_RE.search(line)
        if m_type:
            failure_mode = f"invalid_ggml_type_{m_type.group('type')}"
            evidence.append(line)
        if "failed to load model" in line:
            evidence.append(line)
            m_fail = FAILED_MODEL_RE.search(line)
            if m_fail:
                loader_path = loader_path or m_fail.group("path").strip()

    if failure_mode is None and not evidence:
        return None

    if failure_mode is None:
        failure_mode = "loader_rejected_target"

    return {
        "failure_mode": failure_mode,
        "loader_path": loader_path or str(model_path),
        "loader_command": loader_command,
        "evidence_lines": evidence,
        "evidence_tail": evidence[-4:],
    }


def build_probe(
    *,
    model_source_path: Path = DEFAULT_MODEL_SOURCE,
    loader_log_path: Path = DEFAULT_LOADER_LOG,
    receipt_path: Path = DEFAULT_RECEIPT,
    model_path: Path = DEFAULT_MODEL_PATH,
    loader_binary: Path = DEFAULT_LOADER_BINARY,
) -> dict[str, Any]:
    source = load_json(model_source_path)
    model_exists = model_path.exists()
    exact_target_matched = source.get("model_id") == EXACT_REPO and source.get("selected_file") == EXACT_FILE and model_exists
    target_blockers: list[str] = []
    if not exact_target_matched:
        target_blockers.append("exact_target_mismatch")

    loader_command = build_server_command(
        binary=loader_binary,
        host="127.0.0.1",
        port=8082,
        ctx=16,
        ngl=0,
        model_path=model_path,
    )

    loader_log_text = loader_log_path.read_text(encoding="utf-8") if loader_log_path.exists() else ""
    loader_failure = extract_loader_failure(loader_log_text, model_path=model_path, loader_command=loader_command)

    blockers = list(target_blockers)
    if exact_target_matched:
        if loader_failure and loader_failure.get("failure_mode") == "invalid_ggml_type_42":
            blockers.append("training_unsupported_loader_type_42")
        else:
            blockers.append("training_unsupported_loader_evidence_missing")

    receipt = {
        "schema": "lucidota.bonsai.trainability_probe.v1",
        "generated_at": now_z(),
        "status": "BLOCKED",
        "trainable": False,
        "exact_target": {
            "expected_repo": EXACT_REPO,
            "expected_selected_file": EXACT_FILE,
            "repo": source.get("model_id"),
            "selected_file": source.get("selected_file"),
            "source_url": source.get("source_url"),
            "source_path": rel(model_source_path),
            "model_path": rel(model_path),
            "file_exists": bool(model_exists),
            "matched": bool(exact_target_matched),
        },
        "loader_failure": loader_failure,
        "probe_command": loader_command,
        "blockers": blockers,
        "model_calls_performed": False,
        "db_writes_performed": False,
        "canonical_graph_writes_performed": False,
        "receipt_path": rel(receipt_path),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(prog="bonsai-trainability-probe")
    ap.add_argument("--model-source", type=Path, default=DEFAULT_MODEL_SOURCE)
    ap.add_argument("--loader-log", type=Path, default=DEFAULT_LOADER_LOG)
    ap.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--loader-binary", type=Path, default=DEFAULT_LOADER_BINARY)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    receipt = build_probe(
        model_source_path=args.model_source,
        loader_log_path=args.loader_log,
        receipt_path=args.receipt,
        model_path=args.model_path,
        loader_binary=args.loader_binary,
    )
    print(json.dumps(receipt, sort_keys=True) if args.json else json.dumps(receipt, indent=2, sort_keys=True))
    return 4 if receipt["status"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
