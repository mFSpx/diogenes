import json
import subprocess
import sys

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "signal_aggregator.py"


def run_payload(payload, *args):
    proc = subprocess.run([sys.executable, str(SCRIPT), *args], input=json.dumps(payload), text=True, capture_output=True, cwd=ROOT, timeout=10)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_generic_priority_decay_suppresses_lower_tier_signal():
    rec = run_payload(
        {
            "signals": {
                "collapse": {"score": 0.95, "tier": 0, "hypotheses": {"RECOVER": 1}},
                "opportunity": {"score": 0.90, "tier": 2, "hypotheses": {"EXPLOIT": 1}},
            },
            "config": {"strategy": "priority_decay"},
        }
    )
    assert rec["stance"] == "RECOVER"
    assert rec["suppressed_signals"] == ["opportunity"]


def test_generic_dempster_shafer_contradiction_for_equal_conflict():
    rec = run_payload(
        {
            "signals": {
                "sensor_a": {"score": 0.99, "reliability": 1.0, "hypotheses": {"GO": 1}},
                "sensor_b": {"score": 0.99, "reliability": 1.0, "hypotheses": {"STOP": 1}},
            },
            "config": {"strategy": "dempster_shafer"},
        }
    )
    assert rec["stance"] == "CONTRADICTION"
    assert rec["conflict"] > 0.55


def test_generic_ensemble_uses_centrality_and_emits_receipt():
    rec = run_payload(
        {
            "signals": {
                "visible_core": {"score": 0.7, "tier": 1, "entity": "core", "hypotheses": {"EVADE": 1}, "centrality_weight": 0.5},
                "edge_gain": {"score": 0.68, "tier": 2, "entity": "edge", "hypotheses": {"EXPLOIT": 1}},
            },
            "graph": {"edges": [["core", "edge"], ["core", "db"], ["core", "api"]]},
        }
    )
    assert rec["schema"] == "lucidota.generic.signal_aggregate.v1"
    assert rec["stance"] in {"EVADE", "EXPLOIT", "CONTRADICTION"}
    assert rec["centrality"]["core"] == 1.0
    assert rec["priority_queue"]
