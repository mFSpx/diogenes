#!/usr/bin/env python3
"""Compare scenario-batch/holdout receipts and propose the next seed."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "goals"
GO25 = ["OBJECT", "EVENT", "EDGE"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_report(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return json.loads(p.read_text(encoding="utf-8"))


def rules_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    return list(report.get("promoted_rules") or report.get("decision_rule_candidates") or report.get("decision_tree_candidates") or [])


def rule_key(rule: dict[str, Any]) -> tuple:
    c = rule.get("condition") or {}
    a = rule.get("action") or {}
    return (
        c.get("family"),
        c.get("intent"),
        c.get("lane"),
        c.get("hygiene_label"),
        c.get("model_needed"),
        a.get("outbound_state"),
    )


def compare_reports(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    cur_rules = rules_from_report(current)
    base_rules = rules_from_report(baseline)
    current_condition_index: dict[tuple, list[dict[str, Any]]] = {}
    baseline_condition_index: dict[tuple, list[dict[str, Any]]] = {}
    for rule in cur_rules:
        c = rule.get("condition") or {}
        current_condition_index.setdefault((c.get("family"), c.get("intent"), c.get("lane"), c.get("hygiene_label"), c.get("model_needed")), []).append(rule)
    for rule in base_rules:
        c = rule.get("condition") or {}
        baseline_condition_index.setdefault((c.get("family"), c.get("intent"), c.get("lane"), c.get("hygiene_label"), c.get("model_needed")), []).append(rule)

    stable_rules: list[dict[str, Any]] = []
    new_rules: list[dict[str, Any]] = []
    lost_rules: list[dict[str, Any]] = []
    morphing_rules: list[dict[str, Any]] = []

    for cond_key, rules in current_condition_index.items():
        baseline_rules = baseline_condition_index.get(cond_key) or []
        if not baseline_rules:
            new_rules.extend(rules)
            continue
        for rule in rules:
            current_action = (rule.get("action") or {}).get("outbound_state")
            if any((base.get("action") or {}).get("outbound_state") == current_action for base in baseline_rules):
                stable_rules.append(rule)
            else:
                morphing_rules.append(
                    {
                        "condition": rule.get("condition") or {},
                        "baseline_action": (baseline_rules[0].get("action") or {}).get("outbound_state"),
                        "current_action": current_action,
                        "baseline_support": baseline_rules[0].get("support", 0),
                        "current_support": rule.get("support", 0),
                    }
                )

    for cond_key, rules in baseline_condition_index.items():
        if cond_key not in current_condition_index:
            lost_rules.extend(rules)

    scenario_focus = sorted(
        {
            *(str((rule.get("condition") or {}).get("family")) for rule in new_rules + morphing_rules if (rule.get("condition") or {}).get("family")),
            *(str((rule.get("condition") or {}).get("intent")) for rule in new_rules + morphing_rules if (rule.get("condition") or {}).get("intent")),
        }
    )
    if not scenario_focus:
        scenario_focus = ["stable"]

    return {
        "schema": "lucidota.goal_scenario_compare.v1",
        "generated_at": now(),
        "objective": "Compare scenario receipts over time and propose the next seed.",
        "intent": "scenario_compare",
        "ontology_mode": "GO25_STRICT",
        "ontology_terms": GO25,
        "evidence_refs": [rel(current.get("report_path") or current.get("receipt_path") or ""), rel(baseline.get("report_path") or baseline.get("receipt_path") or "")],
        "status": "PASS",
        "next_action": "Use the stable, morphing, and new rule sets to tighten the next scenario seed.",
        "model_calls_performed": False,
        "canonical_graph_writes_performed": False,
        "current_report": rel(current.get("report_path") or current.get("receipt_path") or ""),
        "baseline_report": rel(baseline.get("report_path") or baseline.get("receipt_path") or ""),
        "stable_rules": stable_rules,
        "new_rules": new_rules,
        "lost_rules": lost_rules,
        "morphing_rules": morphing_rules,
        "comparison_summary": {
            "stable": len(stable_rules),
            "new": len(new_rules),
            "lost": len(lost_rules),
            "morphing": len(morphing_rules),
        },
        "scenario_focus": scenario_focus,
        "next_seed": GO25,
    }


def write_report(report: dict[str, Any]) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"goal_scenario_compare_{stamp()}.json"
    report["receipt_path"] = rel(path)
    report["report_path"] = report["receipt_path"]
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    report = write_report(compare_reports(load_report(a.current), load_report(a.baseline)))
    print("REPORT_PATH=" + report["report_path"])
    print("GOAL_SCENARIO_COMPARE=PASS")
    print(json.dumps(report, sort_keys=True) if a.json else report["report_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
