#!/usr/bin/env python3
"""Build Root-Rotor dedicated-analysis queue jobs from a manifest.

Each file listed in the manifest emits exactly one JSONL row. The builder does
not call external model APIs.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "GOALS" / "ROOT_ROTOR_ACTIVE_SOFTWARE_AUDIT_DUMP.manifest.json"
DEFAULT_QUEUE_JSONL = ROOT / "05_OUTPUTS" / "root_rotor_manual_queue.jsonl"
DEFAULT_NODE_DIR = ROOT / "05_OUTPUTS" / "root_rotor_nodes"
SCHEMA = "lucidota.root_rotor.manual_queue_job.v1"
TARGET_OUTPUT_CONTRACT = "lucidota.root_rotor.bible_node_payload.v1"
NODE_PAYLOAD_SCHEMA = TARGET_OUTPUT_CONTRACT

MANUAL_ROOTS = {
    "SYSTEM_ARCH": (1, "System Architecture Manual"),
    "RUNTIME_GOVERNOR": (2, "Runtime Governor Manual"),
    "AVIONICS": (3, "Algorithms Avionics Manual"),
    "FLIGHT_MAN": (4, "Operations Flight Manual"),
    "LEDGER": (5, "Ledger Amendment Manual"),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_slug(text: str) -> str:
    text = text.replace("\\", "/").strip("/")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return (text or "root_rotor_node")[:160].lower()


def manual_for_path(path: str) -> str:
    p = path.lower()
    if p.startswith("06_schema/") or p.endswith(".sql"):
        return "SYSTEM_ARCH"
    if "governor" in p or "bge" in p or "model_runtime" in p or p.startswith("build_automation/"):
        return "RUNTIME_GOVERNOR"
    if p.startswith("algos/") or p.startswith("math/") or "router" in p or "runtime.py" in p:
        return "AVIONICS"
    if p.startswith("goals/") or p.startswith("00_project_brain/"):
        return "LEDGER"
    return "FLIGHT_MAN"


def target_manual_volume_guess(path: str) -> dict[str, Any]:
    manual_id = manual_for_path(path)
    volume = MANUAL_ROOTS[manual_id][0]
    return {
        "manual_id": manual_id,
        "volume_id": volume,
        "volume_guess": f"{volume}.0.0",
    }


def route_model(path: str) -> tuple[str, list[str]]:
    p = path.lower()
    if p.endswith(".sql") or p.startswith("06_schema/"):
        return "groq", ["groq", "vibes:codestral"]
    if p.endswith(".rs") or p.endswith(".py"):
        return "codestral", ["vibes:codestral", "groq"]
    return "codestral", ["vibes:codestral", "groq"]


def node_contract_snippet() -> str:
    payload = {
        "schema": TARGET_OUTPUT_CONTRACT,
        "source_path": "<path>",
        "source_sha256": "<sha256>",
        "manual_id": "<SYSTEM_ARCH|RUNTIME_GOVERNOR|AVIONICS|FLIGHT_MAN|LEDGER>",
        "node_title": "<short exact title>",
        "node_kind": "<OBJECT|WORKFLOW|EVENT|RECEIPT|EDGE|STATE|BOX|CLAIM|SOURCE|LEDGER|SCHEMA|CONFIG|SCRIPT|ALGORITHM|MODEL|DAEMON|TEST|REFERENCE>",
        "ontology_tags": ["OBJECT"],
        "what_it_is_and_does": "<strict capability statement>",
        "exact_interactions": {
            "imports": [],
            "invokes": [],
            "reads": [],
            "writes": [],
            "db_schemas_or_tables": [],
            "apis_or_cli": []
        },
        "operating_limits_failure_modes": [],
        "integration_points": {
            "tests": [],
            "receipts": [],
            "schemas": [],
            "commands": []
        },
        "payload_asd_ste100": "<manual-ready controlled technical text>",
        "dependencies": [],
        "affects_nodes": [],
        "confidence": "high|medium|low",
        "evidence_refs": ["<source path and line/function evidence>"]
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def source_text_for_entry(entry: dict[str, Any], *, root: Path = ROOT, max_source_bytes: int = 100_000) -> tuple[str, bool]:
    path = root / str(entry["path"])
    try:
        data = path.read_bytes()[:max_source_bytes]
        truncated = int(entry.get("size_bytes") or 0) > len(data)
        return data.decode("utf-8", errors="replace"), truncated
    except OSError:
        return "", False


def build_prompt(entry: dict[str, Any], *, root: Path = ROOT, max_source_bytes: int = 100_000) -> str:
    path = str(entry["path"])
    sha = str(entry["sha256"])
    guess = target_manual_volume_guess(path)
    source_text, source_truncated = source_text_for_entry(entry, root=root, max_source_bytes=max_source_bytes)
    return ("You are a Root-Rotor analysis worker.\n\n"
            "Write one canonical node payload and do not invent facts.\n\n"
            "This prompt is ASD-STE100-aligned: use controlled terms, one meaning per\n"
            "wording, and short direct statements.\n\n"
            "Law-of-Root indexing applies; every output must map to a stable DB node\n"
            "coordinate and include evidence references.\n\n"
            f"source_path: {path}\n"
            f"source_sha256: {sha}\n"
            f"Bytes read: {entry.get('bytes_read')}\n"
            f"Size bytes: {entry.get('size_bytes')}\n"
            f"Target manual volume guess: {json.dumps(guess)}\n"
            f"source_text_truncated: {str(source_truncated).lower()}\n\n"
            "SOURCE_TEXT_BEGIN\n"
            f"{source_text}\n"
            "SOURCE_TEXT_END\n\n"
            "Target DB output schema contract (node payload):\n"
            f"{node_contract_snippet()}\n\n"
            "Deliver a JSON blob that conforms to lucidota.root_rotor.bible_node_payload.v1.\n"
            "Do not return pending_dedicated_model_analysis. Analyze SOURCE_TEXT.\n"
            "Fill what_it_is_and_does, exact_interactions, operating_limits_failure_modes, "
            "integration_points, payload_asd_ste100, node_kind, ontology_tags, dependencies, and affects_nodes from evidence only.\n"
            "Use canonical ontology tags where applicable: OBJECT, WORKFLOW, EVENT, RECEIPT, EDGE, STATE, CLAIM, BOX, COMMENT, SCAR, CHURN, LOOP, DAEMON, ECDYSIS.\n"
            "Default model routing should prefer Vibes/Codestral with Groq fallback.\n")


def queue_row(entry: dict[str, Any], node_dir: Path = DEFAULT_NODE_DIR, *, root: Path = ROOT, max_source_bytes: int = 100_000) -> dict[str, Any]:
    path = str(entry["path"])
    model, preference = route_model(path)
    guess = target_manual_volume_guess(path)
    manual_id = guess["manual_id"]
    target_file = node_dir / f"{safe_slug(f'{manual_id}__{path}')}.json"
    try:
        target_file_value = str(target_file.relative_to(ROOT))
    except ValueError:
        target_file_value = str(target_file)

    return {
        "schema": SCHEMA,
        "label": f"root_rotor_{safe_slug(path)}",
        "path": path,
        "sha256": str(entry["sha256"]),
        "size_bytes": int(entry["size_bytes"]),
        "bytes_read": int(entry["bytes_read"]),
        "target_manual_volume_guess": guess,
        "model": model,
        "model_preference": preference,
        "prompt": build_prompt(entry, root=root, max_source_bytes=max_source_bytes),
        "target_file": target_file_value,
        "target_output_contract": TARGET_OUTPUT_CONTRACT,
    }


def build_queue_rows(manifest_path: Path = DEFAULT_MANIFEST, *, node_dir: Path = DEFAULT_NODE_DIR, root: Path = ROOT, max_source_bytes: int = 100_000) -> list[dict[str, Any]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for entry in manifest.get("files", []):
        if not all(k in entry for k in ("path", "sha256", "size_bytes", "bytes_read")):
            continue
        rows.append(queue_row(entry, node_dir=node_dir, root=root, max_source_bytes=max_source_bytes))
    return rows


def write_jsonl(rows: list[dict[str, Any]], output_jsonl: Path) -> int:
    output_jsonl = output_jsonl.resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)


def build_and_write(
    manifest_path: Path = DEFAULT_MANIFEST,
    output_jsonl: Path = DEFAULT_QUEUE_JSONL,
    node_dir: Path = DEFAULT_NODE_DIR,
    root: Path = ROOT,
    max_source_bytes: int = 100_000,
) -> dict[str, Any]:
    rows = build_queue_rows(manifest_path, node_dir=node_dir, root=root, max_source_bytes=max_source_bytes)
    written = write_jsonl(rows, output_jsonl)
    return {
        "schema": "lucidota.root_rotor.manual_queue_builder.v1",
        "generated_at": now(),
        "manifest_path": str(manifest_path),
        "output_jsonl": str(output_jsonl),
        "jobs_planned": len(rows),
        "jobs_written": written,
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build per-file Root-Rotor manual-analysis queue rows.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_QUEUE_JSONL))
    parser.add_argument("--node-dir", default=str(DEFAULT_NODE_DIR))
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--max-source-bytes", type=int, default=100_000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = build_and_write(
        Path(args.manifest),
        output_jsonl=Path(args.output),
        node_dir=Path(args.node_dir),
        root=Path(args.root),
        max_source_bytes=args.max_source_bytes,
    )
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"ROOT_ROTOR_MANUAL_QUEUE={result['status']}")
        print(f"JOBS_PLANNED={result['jobs_planned']} JOBS_WRITTEN={result['jobs_written']}")
        print(f"OUTPUT_JSONL={result['output_jsonl']}")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
