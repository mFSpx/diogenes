#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def receipt_path(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("RECEIPT_PATH="):
            return ROOT / line.split("=", 1)[1]
    raise AssertionError(stdout)


def make_fake_ncnn(root: Path) -> Path:
    files = [
        "README.md",
        "LICENSE.txt",
        "CMakeLists.txt",
        "python/README.md",
        "benchmark/README.md",
        "benchmark/benchncnn.cpp",
        "benchmark/benchncnn_llm.cpp",
        "examples/whisper.cpp",
        "examples/piper.cpp",
        "src/cpu.cpp",
        "src/gpu.cpp",
        "src/simplevk.cpp",
        "tools/pnnx/.keep",
    ]
    for rel in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("stub\n", encoding="utf-8")
    return root


def test_ncnn_probe_is_read_only_and_records_no_brainers(tmp_path: Path) -> None:
    repo = make_fake_ncnn(tmp_path / "ncnn")
    proc = subprocess.run(
        [sys.executable, "scripts/ncnn_edge_runtime_probe.py", "--repo", str(repo), "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert proc.returncode == 0, proc.stderr
    assert "NCNN_EDGE_RUNTIME_PROBE=PASS" in proc.stdout
    data = json.loads(receipt_path(proc.stdout).read_text(encoding="utf-8"))
    assert data["build_performed"] is False
    assert data["install_performed"] is False
    assert data["model_calls_performed"] is False
    assert data["capabilities_seen"]["vulkan_compute_hooks"] is True
    assert data["capabilities_seen"]["llm_benchmark_source"] is True
    assert any("bounded tool" in item for item in data["no_brainer_commitments"])
