#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> tuple[subprocess.CompletedProcess[str], dict]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    assert proc.stdout.lstrip().startswith("{")
    assert proc.stdout.rstrip().endswith("}")
    payload = json.loads(proc.stdout)
    return proc, payload


def test_main_json_capable_rails_emit_json_only_stdout():
    cases = [
        [str(ROOT / "luci"), "attempt", "--text", "fix one small broken thing and prove it", "--json"],
        [
            str(ROOT / "luci"),
            "learn",
            "--candidate-kind",
            "source",
            "--text",
            "study source candidate and prove it",
            "--artifact",
            "scripts/dev_journey_decision_points.py",
            "--json",
        ],
        [str(ROOT / "luci"), "source", "--text", "read the live world from Hacker News and arXiv", "--json"],
        [str(ROOT / "luci"), "delegate", "--kind", "review", "--text", "review JSON purity", "--provider", "both", "--json"],
        [
            str(ROOT / "luci"),
            "operator-route",
            "--raw-command",
            "create case from folder and build packet",
            "--case-id",
            "json-contract-smoke",
            "--source-folder",
            str(ROOT / "scripts"),
            "--base-dir",
            str(ROOT / "05_OUTPUTS" / "operator_cases"),
            "--receipt-dir",
            str(ROOT / "05_OUTPUTS" / "operator_receipts"),
            "--json",
        ],
        [str(ROOT / "luci"), "model", "admission", "--run-diogenes-gate", "--json"],
        [str(ROOT / "luci"), "model", "provider", "groq-chat", "--prompt", "say hello in one short sentence", "--max-tokens", "16", "--execute", "--json"],
        [str(ROOT / "luci"), "flow", "batch", "--dag", "04_RUNTIME/promptflow_smoke_flow", "--eval", "04_RUNTIME/promptflow_smoke_flow/data.jsonl", "--run-id", "json_contract_smoke", "--output-dir", "05_OUTPUTS/promptflow_traces", "--json"],
        [str(ROOT / "luci"), "graph", "promote", "--execute", "--json"],
        [
            str(ROOT / "luci"),
            "graph",
            "materialize",
            "--execute",
            "--json",
            "--command-envelope-uuid",
            "a38fd6b2-9418-4742-b2f0-a0fc70b5b4eb",
            "--candidate-payload-json",
            '{"term":"ENTITY","label":"front door materialize smoke","status":"staged"}',
            "--evidence-ref",
            "05_OUTPUTS/graph/graph_promotion_full_e2e_20260601T175707672290Z.json",
        ],
        [
            str(ROOT / "luci"),
            "graph",
            "edge",
            "--execute",
            "--json",
            "--command-envelope-uuid",
            "a38fd6b2-9418-4742-b2f0-a0fc70b5b4eb",
            "--source-uuid",
            "11111111-1111-4111-8111-111111111111",
            "--target-uuid",
            "22222222-2222-4222-8222-222222222222",
            "--evidence-ref",
            "05_OUTPUTS/graph/graph_promotion_full_e2e_20260601T175707672290Z.json",
        ],
    ]

    for cmd in cases:
        proc, payload = _run(cmd)
        assert proc.stderr == "" or "WARNING:" in proc.stderr or "Problem" in proc.stderr
        assert isinstance(payload, dict)
        assert payload
