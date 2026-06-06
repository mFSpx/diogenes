#!/usr/bin/env python3
"""ROOT-414 gauntlet scorer: mechanical benchmark checks + HITL-ready report."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "00_PROJECT_BRAIN" / "414_PRIMITIVE_CRIES" / "benchmarks"
CASES = BENCH / "cases"
SUBS = BENCH / "submissions"
REPORTS = BENCH / "reports"
CANONICAL_BPS = {0, 2, 4, 6, 10, 50, 69, 150}
TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9_?]{2,}\b")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def symbols(obj: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for v in obj.values():
            found |= symbols(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= symbols(v)
    elif isinstance(obj, str):
        found |= set(TOKEN_RE.findall(obj))
    return found


def packets(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def score_case(case: dict[str, Any], sub: Any) -> dict[str, Any]:
    expected = case["expected"]
    ps = packets(sub)
    all_sym = symbols(sub)
    feedback = []
    score = 0
    max_score = 100

    required = set(expected.get("required_primitives", []))
    forbidden = set(expected.get("forbidden_primitives", []))
    present_required = sorted(required & all_sym)
    missing_required = sorted(required - all_sym)
    forbidden_present = sorted(forbidden & all_sym)

    score += round(30 * (len(present_required) / len(required))) if required else 30
    if missing_required:
        feedback.append({"severity": "error", "code": "missing_required_primitives", "message": ", ".join(missing_required)})

    if forbidden_present:
        score -= 25
        feedback.append({"severity": "error", "code": "forbidden_primitives_present", "message": ", ".join(forbidden_present)})
    else:
        score += 20

    bps_values = [p.get("confidence_bps") for p in ps if isinstance(p.get("confidence_bps"), int)]
    allowed_bps = set(expected.get("allowed_bps", []))
    if any(v in allowed_bps for v in bps_values):
        score += 10
    else:
        feedback.append({"severity": "warn", "code": "bps_not_allowed", "message": f"seen={bps_values}, allowed={sorted(allowed_bps)}"})

    if any(v not in CANONICAL_BPS for v in bps_values):
        score -= 10
        feedback.append({"severity": "error", "code": "noncanonical_bps", "message": str(bps_values)})

    if expected.get("must_have_falsifier"):
        if any(str(p.get("falsifier", "")).strip() for p in ps):
            score += 15
        else:
            feedback.append({"severity": "error", "code": "missing_falsifier", "message": "No falsifier visible."})

    if expected.get("must_have_local_gate"):
        if any(p.get("local_gates") for p in ps) or any(g in all_sym for g in ["DOCUMENT_EXAMINATION", "VALIDITY_AUDIT", "TEMPORAL_PRECEDENCE", "CHAIN_OF_CUSTODY", "SOURCE_INDEPENDENCE"]):
            score += 15
        else:
            feedback.append({"severity": "error", "code": "missing_local_gate", "message": "No local gate visible."})

    lifecycles = set(expected.get("claim_lifecycle", []))
    seen_lifecycle = {str(p.get("claim_lifecycle", "")) for p in ps}
    if lifecycles and (lifecycles & seen_lifecycle or any(any(lc in str(p) for lc in lifecycles) for p in ps)):
        score += 10
    elif lifecycles:
        feedback.append({"severity": "warn", "code": "claim_lifecycle_mismatch", "message": f"seen={sorted(seen_lifecycle)}, expected_any={sorted(lifecycles)}"})

    return {
        "benchmark_id": case["benchmark_id"],
        "title": case.get("title", ""),
        "score": max(0, min(max_score, score)),
        "missing_required": missing_required,
        "forbidden_present": forbidden_present,
        "feedback": feedback,
        "judge_focus": case.get("judge_focus", []),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", help="case id or path; omit to list cases")
    ap.add_argument("submission", nargs="?", help="submission JSON path")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    if not args.case:
        cases = sorted(p.stem for p in CASES.glob("*.json"))
        print(json.dumps({"cases": cases}, indent=2) if args.json else "\n".join(cases))
        return 0
    case_path = Path(args.case)
    if not case_path.exists():
        case_path = CASES / f"{args.case}.json"
    if not args.submission:
        raise SystemExit("submission path required")
    sub_path = Path(args.submission)
    case = load_json(case_path)
    sub = load_json(sub_path)
    report = score_case(case, sub)
    out = REPORTS / f"{case['benchmark_id']}__{sub_path.stem}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({**report, "report_path": str(out)}, sort_keys=True) if args.json else json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
