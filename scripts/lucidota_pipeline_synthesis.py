#!/usr/bin/env python3
"""Executable LUCIDOTA pipeline synthesis map.

This is the "map the force multipliers" layer: it reads TICKLETRUNK plus the
already-existing pipeline contracts/job planner and emits an inspectable command
map, gap list, and receipt. It performs no DB writes and no canonical graph
writes.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline_contracts import stage_contracts
from work_order_importer import STAGES, planned_pipeline_jobs
from spine_common import ROOT, rel, sha256_json, write_json

OUT = ROOT / "05_OUTPUTS" / "pipeline_synthesis"
TICKLETRUNK = ROOT / "00_PROJECT_BRAIN" / "TICKLETRUNK.json"

FAMILIES: dict[str, dict[str, Any]] = {
    "command_envelope": {
        "purpose": "Accept and scope operator intent before effects.",
        "needles": ("command", "envelope", "control_packet", "operator_command", "kernel"),
    },
    "absurd_queue": {
        "purpose": "Durable workflow journal, queue jobs, replay, and worker execution.",
        "needles": ("absurd", "queue", "spine", "job_adapter", "worker_runner", "work_order"),
    },
    "case_pipeline": {
        "purpose": "Intake → parse → timeline → staging → graph candidates → case packet.",
        "needles": ("pipeline", "case_", "product_", "custody", "parse", "timeline", "staging", "graph_candidate"),
    },
    "stream_interstate": {
        "purpose": "Bytewax/River/CEP/marrow stream interstate and signal fan-in.",
        "needles": ("bytewax", "river", "stream", "cep", "marrow"),
    },
    "graph_promotion": {
        "purpose": "Graph staging, promotion gates, journal barriers, and materialization guards.",
        "needles": ("graph_promotion", "graph_material", "graph_journal", "canonical_graph", "graph_edge"),
    },
    "chrono": {
        "purpose": "Chrono-ledger temporal extraction, replay, conflict, and projection.",
        "needles": ("chrono", "timeline", "temporal"),
    },
    "krampus_korpus": {
        "purpose": "KRAMPUS/KORPUS proof-hoard archaeology and document recovery lanes.",
        "needles": ("krampus", "korpus"),
    },
    "model_inference": {
        "purpose": "Model registry, runtime budget, local runner, GPU/model readiness, and inference lanes.",
        "needles": ("model", "inference", "llama", "needle", "mamba", "deepseek", "gpu"),
    },
    "proof_audit": {
        "purpose": "Receipts, proof kernel, status ledger, oracle gates, and safety audits.",
        "needles": ("proof", "receipt", "audit", "oracle", "gate", "status_ledger"),
    },
    "tickletrunk_manifest": {
        "purpose": "Proof-hoard discovery, script buckets, and manifest refresh.",
        "needles": ("tickletrunk", "manifest", "script_bucket"),
    },
    "surfaces": {
        "purpose": "Operator/UI surfaces, command extraction, sidecars, and lineage guards.",
        "needles": ("surface", "html", "ui", "sidecar"),
    },
    "workflow_foundry": {
        "purpose": "Blueprint-first workflow mining, invariant candidates, and recovered workflows.",
        "needles": ("workflow", "blueprint", "foundry", "design_atom"),
    },
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def load_manifest(path: Path = TICKLETRUNK) -> dict[str, Any]:
    if not path.exists():
        return {"toolboxes": {}, "total_tools": 0, "warnings": ["tickletrunk_missing"]}
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    boxes = manifest.get("toolboxes") or {}
    for category, items in boxes.items():
        if isinstance(items, dict):
            items = items.get("items") or []
        for item in items or []:
            if isinstance(item, dict):
                row = dict(item)
                row.setdefault("category", category)
                entries.append(row)
    return entries


def haystack(entry: dict[str, Any]) -> str:
    parts = [
        entry.get("name"),
        entry.get("relative_path"),
        entry.get("category"),
        entry.get("kind"),
        entry.get("what_it_does"),
        entry.get("one_line_description"),
        " ".join(str(t) for t in entry.get("tags") or []),
    ]
    return " ".join(str(p or "") for p in parts).lower()


def family_hits(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    hits: dict[str, list[dict[str, Any]]] = {name: [] for name in FAMILIES}
    for entry in entries:
        h = haystack(entry)
        for family, spec in FAMILIES.items():
            if any(needle in h for needle in spec["needles"]):
                hits[family].append(entry)
    return hits


def compact_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": entry.get("relative_path") or entry.get("path"),
        "category": entry.get("category"),
        "kind": entry.get("kind"),
        "description": entry.get("one_line_description") or entry.get("what_it_does"),
        "known_use_count": entry.get("known_use_count", 0),
    }


def family_summary(hits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family, entries in hits.items():
        by_category = Counter(str(e.get("category", "UNKNOWN")) for e in entries)
        scripts = [e for e in entries if e.get("category") in {"SCRIPTS", "SCRAPERS"}]
        schemas = [e for e in entries if e.get("category") == "SCHEMAS"]
        tests = [e for e in entries if str(e.get("relative_path", "")).startswith("tests/")]
        workflows = [e for e in entries if e.get("category") == "WORKFLOWS"]
        out[family] = {
            "purpose": FAMILIES[family]["purpose"],
            "entry_count": len(entries),
            "scripts_count": len(scripts),
            "schemas_count": len(schemas),
            "tests_count": len(tests),
            "workflow_docs_count": len(workflows),
            "categories": dict(sorted(by_category.items())),
            "top_components": [compact_entry(e) for e in sorted(entries, key=lambda x: int(x.get("known_use_count") or 0), reverse=True)[:12]],
        }
    return out


def stage_map(case_id: str, source_folder: str | None) -> list[dict[str, Any]]:
    contracts = stage_contracts()
    planned = {
        job["payload"]["stage"]: job
        for job in planned_pipeline_jobs(case_id=case_id, source_folder=source_folder or "<SOURCE_FOLDER>")
    }
    rows: list[dict[str, Any]] = []
    for index, stage in enumerate(STAGES, start=1):
        contract = contracts[stage]
        rows.append({
            "index": index,
            "stage": stage,
            "lane": f"pipeline.{stage}",
            "component_name": contract.component_name,
            "receipt_type": contract.receipt_type,
            "required_inputs": list(contract.input_schema.get("required", [])),
            "required_outputs": list(contract.output_schema.get("required", [])),
            "planned_job": planned[stage],
        })
    return rows


def command_plan(case_id: str, source_folder: str | None, base_dir: str) -> list[dict[str, str]]:
    src = source_folder or "<SOURCE_FOLDER>"
    return [
        {
            "name": "verify_manifest",
            "command": "python3 scripts/tickletrunk_scan.py --check",
            "why": "Confirm proof-hoard access layer before new synthesis.",
        },
        {
            "name": "bucket_scripts",
            "command": "python3 scripts/script_bucket_manifest.py",
            "why": "Classify active/reusable/legacy scripts without moving anything.",
        },
        {
            "name": "new_case",
            "command": f"python3 scripts/lucidota_cli.py --base-dir {base_dir} new-case {case_id} --max-files 100",
            "why": "Create a scoped case workspace and import policy.",
        },
        {
            "name": "run_case_pipeline",
            "command": f"python3 scripts/lucidota_cli.py --base-dir {base_dir} run-pipeline {case_id} --source {src}",
            "why": "Execute intake→parse→timeline→staging→graph-candidate→case-packet with receipts.",
        },
        {
            "name": "plan_absurd_jobs",
            "command": f"python3 scripts/work_order_importer.py pipeline --case-id {case_id} --source-folder {src} --absurd-dir {base_dir}/{case_id}/absurd --json",
            "why": "Produce the local ABSURD dependency-ordered job plan before enqueue/drain.",
        },
        {
            "name": "enqueue_absurd_jobs",
            "command": f"python3 scripts/work_order_importer.py pipeline --case-id {case_id} --source-folder {src} --absurd-dir {base_dir}/{case_id}/absurd --execute --json",
            "why": "Write local queue state only; no DB or canonical graph writes.",
        },
        {
            "name": "drain_absurd_jobs",
            "command": f"python3 scripts/absurd_worker_runner.py {base_dir}/{case_id}/absurd --base-dir {base_dir}",
            "why": "Drain the local file-backed ABSURD jobs through existing stage dispatch.",
        },
    ]


def synthesize_next_actions(summary: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for family, row in summary.items():
        if row["entry_count"] == 0:
            instruction = f"Add or recover a blueprint for missing family `{family}`."
            action_type = "RECOVER_BLUEPRINT"
        elif row["scripts_count"] and not row["tests_count"]:
            instruction = f"Add a receipt-backed smoke test for `{family}` active scripts."
            action_type = "ADD_TEST_HARNESS"
        elif family in {"absurd_queue", "case_pipeline", "graph_promotion", "chrono"} and row["scripts_count"] and not row["schemas_count"]:
            instruction = f"Add an explicit schema/contract surface for `{family}`."
            action_type = "ADD_CONTRACT_SCHEMA"
        elif family == "model_inference":
            instruction = "Promote the Inference OS layer into an explicit residency/scheduler blueprint."
            action_type = "SYNTHESIZE_INFERENCE_OS_BLUEPRINT"
        else:
            continue
        seed = {"family": family, "action_type": action_type, "instruction": instruction}
        actions.append({
            "action_id": "action:" + sha256_json(seed)[:24],
            "action_type": action_type,
            "target_ref": family,
            "plain_language_instruction": instruction,
            "closure_gate": "python3 scripts/lucidota_pipeline_synthesis.py --no-write --json",
            "canonical_graph_writes_allowed": False,
        })
    return actions


def build_synthesis(
    *,
    manifest_path: Path = TICKLETRUNK,
    case_id: str = "pipeline-synthesis",
    source_folder: str | None = None,
    base_dir: str = "05_OUTPUTS/cases",
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    entries = flatten_entries(manifest)
    hits = family_hits(entries)
    summary = family_summary(hits)
    stages = stage_map(case_id, source_folder)
    payload = {
        "schema": "lucidota.pipeline_synthesis_map.v1",
        "generated_at": now_iso(),
        "source_manifest": rel(manifest_path),
        "manifest_generated_at": manifest.get("generated_at"),
        "manifest_total_tools": manifest.get("total_tools", len(entries)),
        "blueprint_first_model_second": True,
        "db_writes_performed": False,
        "canonical_graph_writes_performed": False,
        "stage_map": stages,
        "families": summary,
        "command_plan": command_plan(case_id, source_folder, base_dir),
        "next_actions": synthesize_next_actions(summary),
    }
    payload["family_count"] = len(summary)
    payload["planned_pipeline_job_count"] = len(stages)
    payload["next_action_count"] = len(payload["next_actions"])
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# LUCIDOTA Pipeline Synthesis Map",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "- Blueprint-first/model-second: true",
        "- DB writes performed: false",
        "- Canonical graph writes performed: false",
        "",
        "## Product case pipeline",
        "",
        "| # | stage | lane | receipt | inputs | outputs |",
        "|---:|---|---|---|---|---|",
    ]
    for row in payload["stage_map"]:
        lines.append(
            f"| {row['index']} | `{row['stage']}` | `{row['lane']}` | `{row['receipt_type']}` | "
            f"{', '.join(row['required_inputs'])} | {', '.join(row['required_outputs'])} |"
        )
    lines.extend(["", "## Architecture family coverage", ""])
    for family, row in payload["families"].items():
        lines.append(
            f"- `{family}` — entries={row['entry_count']} scripts={row['scripts_count']} "
            f"schemas={row['schemas_count']} tests={row['tests_count']} — {row['purpose']}"
        )
    lines.extend(["", "## Command plan", ""])
    for step in payload["command_plan"]:
        lines.append(f"- **{step['name']}**: `{step['command']}` — {step['why']}")
    lines.extend(["", "## Next actions", ""])
    for action in payload["next_actions"]:
        lines.append(f"- `{action['action_type']}` `{action['target_ref']}` — {action['plain_language_instruction']}")
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(payload: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"lucidota_pipeline_synthesis_{stamp()}"
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    payload["report_path"] = rel(json_path)
    payload["markdown_path"] = rel(md_path)
    write_json(json_path, payload)
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a receipt-backed synthesis map for LUCIDOTA pipeline automation.")
    parser.add_argument("--manifest", default=str(TICKLETRUNK))
    parser.add_argument("--case-id", default="pipeline-synthesis")
    parser.add_argument("--source-folder")
    parser.add_argument("--base-dir", default="05_OUTPUTS/cases")
    parser.add_argument("--out-dir", default=str(OUT))
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    payload = build_synthesis(
        manifest_path=manifest_path,
        case_id=args.case_id,
        source_folder=args.source_folder,
        base_dir=args.base_dir,
    )
    if not args.no_write:
        json_path, md_path = write_outputs(payload, Path(args.out_dir))
        print(f"REPORT_PATH={rel(json_path)}")
        print(f"MARKDOWN_PATH={rel(md_path)}")
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    print("PIPELINE_SYNTHESIS=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
