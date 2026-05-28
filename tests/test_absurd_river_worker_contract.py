#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from absurd_river_worker import (
    CLAIM_PACKET_JOB_KIND,
    DEFAULT_JOB_KIND,
    DEFAULT_QUEUE,
    LEGACY_JOB_KIND,
    LEGACY_QUEUE,
    MAX_COMPONENT_LIMIT,
    bounded_int,
    claim_packet_extract,
    worker_key_for_queue,
)


def report_from(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            return json.loads((ROOT / line.split("=", 1)[1]).read_text(encoding="utf-8"))
    raise AssertionError(stdout)


def test_bounded_int_contract():
    assert bounded_int("3", name="component_limit", minimum=1, maximum=5) == 3
    try:
        bounded_int("x", name="component_limit", minimum=1, maximum=5)
    except ValueError as exc:
        assert str(exc) == "component_limit_must_be_int"
    else:
        raise AssertionError("expected int parse failure")
    try:
        bounded_int(6, name="component_limit", minimum=1, maximum=5)
    except ValueError as exc:
        assert str(exc) == "component_limit_out_of_range:1..5"
    else:
        raise AssertionError("expected range failure")


def test_enqueue_blocks_unbounded_component_limit_without_db():
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/absurd_river_worker.py",
            "--action",
            "enqueue-health-check",
            "--component-limit",
            str(MAX_COMPONENT_LIMIT + 1),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 1
    report = report_from(proc.stdout)
    assert report["db_writes_performed"] is False
    assert report["canonical_graph_writes_performed"] is False
    assert report["blockers"] == ["component_limit_out_of_range"]
    assert report["action_result"]["execute_performed"] is False


def test_claim_packet_extract_rejects_unbounded_limit_before_db():
    class Args:
        storage_database_url = "postgresql:///unused"
        state_database_url = "postgresql:///unused"
        claim_packet_limit = 1
        python = None

    result, blockers = claim_packet_extract(Args(), {"claim_packet_limit": 1001}, "00000000-0000-0000-0000-000000000000", execute=True)
    assert blockers == ["claim_packet_limit_out_of_range"]
    assert result["claim_packet_writes_performed"] is False
    assert result["canonical_graph_writes_performed"] is False
    assert result["temporal_claims_mutated_by_wrapper"] is False


def test_river_worker_dequeue_contract_is_source_wired():
    source = (ROOT / "scripts/absurd_river_worker.py").read_text(encoding="utf-8")
    worker_source = source[source.index("def worker_once") : source.index("def main")]
    assert "validate_worker_contract" in source
    assert "record_worker_contract_rejection" in source
    assert "FOR UPDATE SKIP LOCKED" in worker_source
    assert "AND job_kind=%s" not in worker_source
    assert "args.job_kind = str(row[\"job_kind\"])" in worker_source
    assert "validate_worker_contract" in worker_source
    assert worker_source.index("validate_worker_contract") < worker_source.index("river_bytewax_health")
    assert worker_key_for_queue(DEFAULT_QUEUE) == "river_worker"
    assert worker_key_for_queue(LEGACY_QUEUE) == "river_legacy_worker"


def test_river_worker_registry_names_implemented_job_kinds():
    registry = (ROOT / "06_SCHEMA/082_absurd_worker_contract_registry_enforcement.sql").read_text(encoding="utf-8")
    assert DEFAULT_JOB_KIND in registry
    assert CLAIM_PACKET_JOB_KIND in registry
    assert LEGACY_JOB_KIND in registry
    assert "river_health_check|" not in registry
