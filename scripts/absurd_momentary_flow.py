#!/usr/bin/env python3
"""PocketFlow-style momentary graph runner for durable ABSURD jobs.

Shared state exists only during the job. Completion returns training evidence,
transition trace, and hashes; the shared store itself is deliberately discarded.
"""
from __future__ import annotations

import argparse, hashlib, json, re
from datetime import datetime, timezone
from typing import Any

SCHEMA = "lucidota.absurd.momentary_flow.result.v1"
INJECTION = re.compile(r"\b(ignore\s+(previous|prior|above)\s+instructions|system\s+prompt|exfiltrate|curl\b[^\n]{0,80}\|\s*(sh|bash)|rm\s+-rf\s+/)\b", re.I)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def ternary_vector(text: str, dims: int = 12) -> list[int]:
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
    return [(digest[i] % 3) - 1 for i in range(dims)]


def input_text(shared: dict[str, Any]) -> str:
    return str((shared.get("input") or {}).get("text") or "")


def source_ref(payload: dict[str, Any]) -> str:
    return str(payload.get("source_ref") or payload.get("flow_id") or "absurd_momentary_flow")


def add_training(shared: dict[str, Any], payload: dict[str, Any], node: dict[str, Any], before: Any, out: dict[str, Any]) -> None:
    shared["training_examples"].append({
        "channel": node["op"],
        "node_id": node["id"],
        "source_ref": source_ref(payload),
        "input_sha256": sha(before),
        "output_sha256": sha(out),
        "payload": out,
    })
    shared["trace"].append({"node_id": node["id"], "op": node["op"], "output_sha256": sha(out)})
    shared["completed_node_ids"].append(node["id"])
    shared["last"] = out


def run_node(shared: dict[str, Any], payload: dict[str, Any], node: dict[str, Any]) -> None:
    op, text, inp = node.get("op"), input_text(shared), shared.get("last")
    paths = list((shared.get("input") or {}).get("paths") or payload.get("paths") or [])
    adversarial = bool(INJECTION.search(text))
    if op == "bitloops_context":
        out = {"context_source": "bitloops_devql_ready", "selected_paths": paths[:16], "text_sha256": sha(text), "evidence": "context_only_no_authority"}
    elif op == "bytewax_hint":
        out = {"epistemic_flag": "BULLSHIT" if adversarial else "POSSIBLE", "hypothesis": "bitloops_context_feeds_absurd_bytewax_lane", "support_score": 0.55, "injection_flag": adversarial, "evidence": inp}
    elif op == "river_mre":
        out = {"mre_kind": "momentary_reproducible_evidence", "features": {"chars": len(text), "path_count": len(paths), "adversarial": adversarial}, "reward_hint": -0.25 if adversarial else 0.25, "river_update": "training_only_no_model_call", "evidence": inp}
    elif op == "ternary_truth":
        out = {"ternary_vector": ternary_vector(text + sha(inp)), "truth_state": "BULLSHIT" if adversarial else "SURE_MAYBE", "adversarial_flag": adversarial, "gnn_edge_candidates": [{"src": "bitloops", "dst": "bytewax", "rel": "feeds"}, {"src": "bytewax", "dst": "river", "rel": "trains"}], "evidence": inp}
    else:
        raise ValueError(f"unknown_flow_op:{op}")
    add_training(shared, payload, {"id": str(node.get("id") or op), "op": str(op)}, inp, out)


def run_momentary_flow(payload: dict[str, Any]) -> dict[str, Any]:
    shared = {"input": payload.get("input") or {}, "last": None, "trace": [], "training_examples": [], "completed_node_ids": []}
    nodes = payload.get("nodes") or []
    if not isinstance(nodes, list) or not nodes:
        nodes = [{"id": "bitloops", "op": "bitloops_context"}, {"id": "bytewax", "op": "bytewax_hint"}, {"id": "river", "op": "river_mre"}, {"id": "ternary", "op": "ternary_truth"}]
    try:
        for node in nodes:
            if not isinstance(node, dict):
                raise ValueError("flow_node_must_be_object")
            run_node(shared, payload, node)
        outcome, status, error = "succeeded", "PASS", ""
    except Exception as exc:
        outcome, status, error = "failed", "FAIL", str(exc)
    return {
        "schema": SCHEMA,
        "generated_at": now(),
        "status": status,
        "outcome": outcome,
        "error": error,
        "flow": {"flow_id": str(payload.get("flow_id") or "momentary_flow"), "node_count": len(nodes), "completed_node_ids": shared["completed_node_ids"]},
        "transition_trace": shared["trace"],
        "training_examples": shared["training_examples"],
        "state_collapsed": True,
        "model_calls_performed": False,
        "canonical_graph_writes_performed": False,
        "proof": "momentary shared state discarded; durable ABSURD job result carries evidence/training examples",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload-json", default="{}")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    result = run_momentary_flow(json.loads(a.payload_json))
    print(json.dumps(result, sort_keys=True) if a.json else json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["outcome"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
