#!/usr/bin/env python3
"""Mem0-style behavior observation, not blind reinforcement

Research-only dry-run scaffold. No DB writes. No graph writes.
"""
from __future__ import annotations
import argparse, json, datetime
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "05_OUTPUTS" / "research_integration"
MAX_INPUT_BYTES = 256 * 1024
PAYLOAD = {
  "candidate_classes": [
    "preference",
    "habit",
    "friction",
    "workflow",
    "relationship_tone",
    "goal",
    "operator_law"
  ],
  "hard_law": "repeated behavior != preference",
  "observations": [
    {
      "text": "Operator asks for progress bars repeatedly.",
      "class": "workflow",
      "not_preference_until_confirmed": True
    }
  ]
}

def stamp(): return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

def load_payload(raw: str | None) -> tuple[dict[str, Any], list[str]]:
    if not raw:
        return PAYLOAD, []
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists() or not p.is_file():
        return PAYLOAD, [f"input_missing:{raw}"]
    if p.stat().st_size > MAX_INPUT_BYTES:
        return PAYLOAD, [f"input_too_large:{p.stat().st_size}>{MAX_INPUT_BYTES}"]
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return PAYLOAD, ["input_json_must_be_object"]
    return data, []

def validate_payload(payload: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    observations = payload.get("observations")
    classes = payload.get("candidate_classes")
    if not isinstance(classes, list) or not all(isinstance(c, str) and c.strip() for c in classes):
        blockers.append("candidate_classes_must_be_nonempty_string_list")
        classes = []
    if not isinstance(observations, list):
        blockers.append("observations_must_be_list")
        observations = []
    if len(observations) > 200:
        blockers.append("dry_run_payload_over_200_observations")
    for i, obs in enumerate(observations[:200]):
        if not isinstance(obs, dict):
            blockers.append(f"observation_not_object:{i}")
            continue
        if not str(obs.get("text", "")).strip():
            blockers.append(f"observation_missing_text:{i}")
        if obs.get("class") not in set(classes):
            blockers.append(f"observation_unknown_class:{i}")
        if obs.get("not_preference_until_confirmed") is not True:
            blockers.append(f"observation_missing_confirmation_guard:{i}")
    return blockers

def main():
    ap=argparse.ArgumentParser(description='Mem0-style behavior observation, not blind reinforcement')
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--input")
    ap.add_argument("--out-dir", default=None)
    args=ap.parse_args()
    out_dir=Path(args.out_dir) if args.out_dir else DEFAULT_OUT / stamp()
    if not out_dir.is_absolute(): out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload, blockers = load_payload(args.input)
    blockers.extend(validate_payload(payload))
    report={"schema":'lucidota.research.behavioral_memory.v1',"generated_at":now(),"mode":"dry_run","purpose":'Mem0-style behavior observation, not blind reinforcement',"input":args.input,"payload":payload,"db_writes_performed":False,"graph_writes_performed":False,"canonical_mutation_allowed":False,"blockers":blockers,"status":"PASS" if not blockers else "FAIL"}
    out=out_dir/(Path(__file__).stem + "_report.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"REPORT_PATH={out.relative_to(ROOT)}")
    return 0 if not blockers else 4
if __name__ == "__main__":
    raise SystemExit(main())
