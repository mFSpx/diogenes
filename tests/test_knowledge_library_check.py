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


def test_knowledge_library_check_passes_and_finds_drf() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/knowledge_library_check.py", "--check", "--query", "drf", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert proc.returncode == 0, proc.stderr
    assert "KNOWLEDGE_LIBRARY_CHECK=PASS" in proc.stdout
    assert "cybercrafter_drf" in proc.stdout
    data = json.loads(receipt_path(proc.stdout).read_text(encoding="utf-8"))
    assert data["verdict"] == "PASS"
    assert data["model_calls_performed"] is False
    assert data["network_calls_performed"] is False
    assert any(match["id"] == "cybercrafter_drf" for match in data["matches"])


def test_knowledge_library_index_has_blueprint_drf_and_router_entries() -> None:
    index = json.loads((ROOT / "00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/index.json").read_text(encoding="utf-8"))
    ids = {entry["id"]: entry for entry in index["entries"]}
    assert ids["blueprint_first_model_second"]["authority_class"] == "pseudolaw"
    assert ids["cybercrafter_drf"]["authority_class"] == "research_reference"
    assert (ROOT / ids["cybercrafter_drf"]["local_path"]).exists()
    assert (ROOT / ids["cybercrafter_drf"]["knowledge_card"]).exists()
    assert ids["dobybaxter_llm_router"]["authority_class"] == "research_reference"
    assert (ROOT / ids["dobybaxter_llm_router"]["local_path"]).exists()
    assert (ROOT / ids["dobybaxter_llm_router"]["knowledge_card"]).exists()


def test_knowledge_library_query_finds_llm_router() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/knowledge_library_check.py", "--check", "--query", "topology", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(receipt_path(proc.stdout).read_text(encoding="utf-8"))
    assert any(match["id"] == "dobybaxter_llm_router" for match in data["matches"])


def test_knowledge_library_has_syd_learned_candidate_tool() -> None:
    index = json.loads((ROOT / "00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/index.json").read_text(encoding="utf-8"))
    ids = {entry["id"]: entry for entry in index["entries"]}
    syd = ids["sydsec_syd"]
    assert syd["source_url"] == "https://gitlab.com/sydsec1/Syd"
    assert syd["status"] == "learned_candidate_tool"
    assert syd["authority_class"] == "candidate_tool"
    assert (ROOT / syd["local_path"]).exists()
    assert (ROOT / syd["knowledge_card"]).exists()


def test_knowledge_library_has_ncnn_reference_and_probe() -> None:
    index = json.loads((ROOT / "00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/index.json").read_text(encoding="utf-8"))
    ids = {entry["id"]: entry for entry in index["entries"]}
    ncnn = ids["flywheel1412_ncnn"]
    assert ncnn["source_url"] == "https://gitlab.com/flywheel1412/ncnn"
    assert ncnn["authority_class"] == "research_reference"
    assert (ROOT / ncnn["local_path"]).exists()
    assert (ROOT / ncnn["knowledge_card"]).exists()
    assert (ROOT / "scripts/ncnn_edge_runtime_probe.py").exists()
