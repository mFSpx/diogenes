#!/usr/bin/env python3
"""activity_event/activity_tree_node shaped JSON; consent/revocation/sensitivity scopes required

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
  "activity_events": [
    {
      "source_ref": "sample",
      "actor": "operator",
      "action": "authored_decision",
      "object_ref": "research_decision_record",
      "sensitivity_scope": "operator_owned",
      "consent_scope": "local_only"
    }
  ],
  "activity_tree_nodes": [
    {
      "node_kind": "decision_record",
      "label": "Personal Context Spine",
      "evidence_refs": [
        "KRAMPUSCHEWING/LUCIDOTA_PERSONAL_CONTEXT_SPINE_CATCHME.md"
      ]
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
    blockers: list[str] = []
    if not p.exists() or not p.is_file():
        return PAYLOAD, [f"input_missing:{raw}"]
    size = p.stat().st_size
    if size > MAX_INPUT_BYTES:
        return PAYLOAD, [f"input_too_large:{size}>{MAX_INPUT_BYTES}"]
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return PAYLOAD, [f"input_json_invalid:{exc.msg}:line{exc.lineno}:col{exc.colno}"]
    if not isinstance(data, dict):
        return PAYLOAD, ["input_json_must_be_object"]
    return data, blockers

def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)

def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)

def validate_payload(payload: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    events = payload.get("activity_events")
    nodes = payload.get("activity_tree_nodes")
    if not isinstance(events, list):
        blockers.append("activity_events_must_be_list")
        events = []
    if not isinstance(nodes, list):
        blockers.append("activity_tree_nodes_must_be_list")
        nodes = []
    for i, event in enumerate(events[:200]):
        if not isinstance(event, dict):
            blockers.append(f"activity_event_not_object:{i}")
            continue
        for key in ("source_ref", "actor", "action", "object_ref", "sensitivity_scope", "consent_scope"):
            if not str(event.get(key, "")).strip():
                blockers.append(f"activity_event_missing_{key}:{i}")
        if event.get("consent_scope") not in {"local_only", "operator_approved", "revoked"}:
            blockers.append(f"activity_event_bad_consent_scope:{i}")
    for i, node in enumerate(nodes[:200]):
        if not isinstance(node, dict):
            blockers.append(f"activity_tree_node_not_object:{i}")
            continue
        if not str(node.get("node_kind", "")).strip():
            blockers.append(f"activity_tree_node_missing_node_kind:{i}")
        if not isinstance(node.get("evidence_refs", []), list):
            blockers.append(f"activity_tree_node_evidence_refs_must_be_list:{i}")
    if len(events) > 200 or len(nodes) > 200:
        blockers.append("dry_run_payload_over_200_rows")
    return blockers

def main():
    ap=argparse.ArgumentParser(description='activity_event/activity_tree_node shaped JSON; consent/revocation/sensitivity scopes required')
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--input")
    ap.add_argument("--out-dir", default=None)
    args=ap.parse_args()
    out_dir=Path(args.out_dir) if args.out_dir else DEFAULT_OUT / stamp()
    if not out_dir.is_absolute(): out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload, blockers = load_payload(args.input)
    blockers.extend(validate_payload(payload))
    report={"schema":'lucidota.research.activity_tree_ingest.v1',"generated_at":now(),"mode":"dry_run","purpose":'activity_event/activity_tree_node shaped JSON; consent/revocation/sensitivity scopes required',"input":args.input,"payload":payload,"db_writes_performed":False,"graph_writes_performed":False,"canonical_mutation_allowed":False,"blockers":blockers,"status":"PASS" if not blockers else "FAIL"}
    out=out_dir/(Path(__file__).stem + "_report.json")
    write_json_atomic(out, report)
    print(f"REPORT_PATH={display_path(out)}")
    return 0 if not blockers else 4
if __name__ == "__main__":
    raise SystemExit(main())
