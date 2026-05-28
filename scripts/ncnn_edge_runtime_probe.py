#!/usr/bin/env python3
"""Read-only ncnn edge-runtime capability probe for LUCIDOTA."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from spine_common import ROOT, now, receipt, rel

DEFAULT_REPO = ROOT / "01_REPOS" / "flywheel1412_ncnn"
EXPECTED = [
    "README.md",
    "LICENSE.txt",
    "CMakeLists.txt",
    "python/README.md",
    "benchmark/README.md",
    "benchmark/benchncnn_llm.cpp",
    "examples/whisper.cpp",
    "examples/piper.cpp",
    "tools/pnnx",
]


def git_value(repo: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(["git", "-C", str(repo), *args], text=True, capture_output=True, timeout=5, check=False)
        return proc.stdout.strip() or None
    except Exception:
        return None


def file_lines(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    return len(path.read_text(encoding="utf-8", errors="ignore").splitlines())


def probe(repo: Path = DEFAULT_REPO) -> dict[str, Any]:
    repo = repo.resolve()
    missing = [p for p in EXPECTED if not (repo / p).exists()]
    present = [p for p in EXPECTED if (repo / p).exists()]
    py_spec = importlib.util.find_spec("ncnn")
    cmake = shutil.which("cmake")
    vulkaninfo = shutil.which("vulkaninfo")
    payload = {
        "schema": "lucidota.ncnn_edge_runtime_probe.v1",
        "generated_at": now(),
        "repo_path": rel(repo),
        "source_url": "https://gitlab.com/flywheel1412/ncnn",
        "upstream_family": "Tencent/ncnn high-performance mobile neural-network inference framework",
        "commit": git_value(repo, "rev-parse", "HEAD") if (repo / ".git").exists() else None,
        "remote": git_value(repo, "remote", "get-url", "origin") if (repo / ".git").exists() else None,
        "expected_files_present": present,
        "expected_files_missing": missing,
        "readme_lines": file_lines(repo / "README.md"),
        "python_readme_lines": file_lines(repo / "python/README.md"),
        "benchmark_readme_lines": file_lines(repo / "benchmark/README.md"),
        "capabilities_seen": {
            "mobile_cpu_inference": (repo / "src/cpu.cpp").exists(),
            "vulkan_compute_hooks": (repo / "src/gpu.cpp").exists() and (repo / "src/simplevk.cpp").exists(),
            "python_wrapper_docs": (repo / "python/README.md").exists(),
            "benchmark_harness": (repo / "benchmark/benchncnn.cpp").exists(),
            "llm_benchmark_source": (repo / "benchmark/benchncnn_llm.cpp").exists(),
            "whisper_example": (repo / "examples/whisper.cpp").exists(),
            "piper_tts_example": (repo / "examples/piper.cpp").exists(),
            "pnnx_tooling_tree": (repo / "tools/pnnx").exists(),
        },
        "host_probe": {
            "python_ncnn_importable": py_spec is not None,
            "cmake_available": cmake is not None,
            "cmake_path": cmake,
            "vulkaninfo_available": vulkaninfo is not None,
            "vulkaninfo_path": vulkaninfo,
            "VULKAN_SDK_set": bool(os.environ.get("VULKAN_SDK")),
        },
        "no_brainer_commitments": [
            "Keep ncnn indexed as edge/mobile inference candidate, not current core LLM runtime.",
            "Add/read a probe receipt before any ncnn build, install, or adapter work.",
            "Prefer read-only capability detection before heavy submodule/build actions.",
            "Consider ncnn for mobile/vision/audio edge models and Vulkan/CPU benchmarks, while llama.cpp remains GGUF LLM baseline unless a concrete ncnn model artifact exists.",
            "If integrated later, wrap as deterministic bounded tool with receipts and explicit model file paths.",
        ],
        "build_performed": False,
        "install_performed": False,
        "model_calls_performed": False,
        "network_calls_performed": False,
    }
    blockers: list[str] = []
    if not repo.exists():
        blockers.append("repo_missing")
    if missing and repo.exists():
        blockers.append("expected_reference_files_missing")
    payload["blockers"] = blockers
    payload["verdict"] = "PASS" if not blockers else "FAIL"
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only ncnn edge/mobile inference candidate probe.")
    ap.add_argument("--repo", default=str(DEFAULT_REPO))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    payload = probe(Path(args.repo))
    receipt("ncnn_edge_runtime_probe", payload, root="05_OUTPUTS/ncnn")
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    print("NCNN_EDGE_RUNTIME_PROBE=" + payload["verdict"])
    return 0 if payload["verdict"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
