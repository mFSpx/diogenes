#!/usr/bin/env python3
"""Emit the Ahoy ABBA3 rule/strategy/training audit receipt."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ahoy_sim.engine.receipts import OUT, sha256_file, stamp, utc_now, write_json_receipt
from ahoy_sim.rules.rule_gaps import blocking_gaps
from ahoy_sim.rules.rule_manifest import MANIFEST_PATH, load_manifest

def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)

def main() -> int:
    ap = argparse.ArgumentParser(description="Emit the Ahoy ABBA3 rule/strategy/training audit receipt.")
    ap.add_argument("--rules-only", action="store_true")
    ap.add_argument("--out-dir", default=None, help="Receipt directory; defaults to 05_OUTPUTS/ahoy/abba3.")
    args = ap.parse_args()
    gaps = blocking_gaps()
    manifest = load_manifest()
    receipt = {
        "schema": "lucidota.ahoy.abba3.v1",
        "created_at": utc_now(),
        "verdict": "BLOCKED" if gaps else "PASS",
        "rules_verdict": "BLOCKED" if gaps else "PASS",
        "blocking_rule_gaps": gaps,
        "rules_abba": {
            "A_acquire_source_truth": {"manifest": str(MANIFEST_PATH.relative_to(ROOT)), "manifest_sha256": sha256_file(MANIFEST_PATH), "rule_count": len(manifest.get("rules", []))},
            "B_build_executable_model": {"modules": ["ahoy_sim/rules", "ahoy_sim/engine"], "status": "implemented_v1"},
            "B_break_validators": {"tests": ["tests/test_ahoy_*.py"], "status": "implemented"},
            "A_audit_against_source": {"status": "pass" if not gaps else "blocked", "blocking_gaps": [g["gap_id"] for g in gaps]},
        },
        "strategy_abba": None if args.rules_only else {
            "A_define_utilities": {"status": "implemented_v1", "policies": ["BluefinPolicy", "MolluskPolicy", "Smuggler/RiverPolicy"]},
            "B_build_policies": {"status": "heuristic_bounded"},
            "B_break_exploit_fixtures": {"status": "tests_present"},
            "A_audit_policy_legality": {"status": "engine_legal_action_authority"},
        },
        "training_abba": None if args.rules_only else {
            "A_define_schema_labels": {"schema": "06_SCHEMA/ahoy/ahoy_training_row.v1.json"},
            "B_collect_online_rows": {"status": "partial_allowed_only_with_allow_partial_rules"},
            "B_break_rows": {"status": "validators_present"},
            "A_audit_dataset": {"status": "pass" if not gaps else "blocked_for_full_training_until_rule_gaps_zero"},
        },
        "blockers": [g["gap_id"] for g in gaps],
    }
    out_dir = Path(args.out_dir) if args.out_dir else OUT / "abba3"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    path = out_dir / f"abba3_{stamp()}.json"
    write_json_receipt(path, receipt)
    print(json.dumps({"verdict": receipt["verdict"], "receipt": display_path(path), "blocking_gaps": len(gaps)}, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
