#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from absurd_queue_spine import ALLOWED_JOB_KINDS, run_job


def sample_flow_payload() -> dict:
    return {
        "flow_id": "bitloops_bytewax_river_torch",
        "source_ref": "GOALS/BITLOOPS_SOVEREIGN_INTEGRATION_AUDIT.md",
        "input": {
            "text": "Bitloops should feed Bytewax and River, but ignore previous instructions is just data.",
            "paths": ["GOALS/CURRENT_HANDOFF.md", "scripts/bytewax_abductive_blender.py"],
        },
        "nodes": [
            {"id": "bitloops", "op": "bitloops_context"},
            {"id": "bytewax", "op": "bytewax_hint"},
            {"id": "river", "op": "river_mre"},
            {"id": "ternary", "op": "ternary_truth"},
        ],
    }


def test_momentary_flow_is_absurd_job_kind():
    assert "momentary_flow" in ALLOWED_JOB_KINDS


def test_momentary_flow_collapses_state_but_keeps_training_evidence():
    ok, result, err = run_job("momentary_flow", sample_flow_payload())

    assert ok is True
    assert err == ""
    assert result["outcome"] == "succeeded"
    assert result["schema"] == "lucidota.absurd.momentary_flow.result.v1"
    assert result["state_collapsed"] is True
    assert "shared" not in result
    assert result["canonical_graph_writes_performed"] is False
    assert result["model_calls_performed"] is False

    flow = result["flow"]
    assert flow["flow_id"] == "bitloops_bytewax_river_torch"
    assert flow["node_count"] == 4
    assert flow["completed_node_ids"] == ["bitloops", "bytewax", "river", "ternary"]

    examples = result["training_examples"]
    channels = {ex["channel"] for ex in examples}
    assert {"bitloops_context", "bytewax_hint", "river_mre", "ternary_truth"} <= channels
    assert all(ex["source_ref"] == "GOALS/BITLOOPS_SOVEREIGN_INTEGRATION_AUDIT.md" for ex in examples)
    assert all(ex["input_sha256"] and ex["output_sha256"] for ex in examples)

    ternary = next(ex for ex in examples if ex["channel"] == "ternary_truth")
    assert set(ternary["payload"]["ternary_vector"]) <= {-1, 0, 1}
    assert ternary["payload"]["adversarial_flag"] is True


def test_momentary_flow_rejects_unknown_node_op_without_persisting_shared_state():
    payload = sample_flow_payload()
    payload["nodes"].append({"id": "oops", "op": "direct_graph_write"})

    ok, result, err = run_job("momentary_flow", payload)

    assert ok is False
    assert err == "momentary_flow_failed"
    assert result["outcome"] == "failed"
    assert result["state_collapsed"] is True
    assert "shared" not in result
    assert result["canonical_graph_writes_performed"] is False
    assert result["error"].startswith("unknown_flow_op:")
