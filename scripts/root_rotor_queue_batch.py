#!/usr/bin/env python3
"""Select the next unprocessed Root-Rotor manual jobs from the full queue."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT / "05_OUTPUTS" / "root_rotor_manual_queue.jsonl"
DEFAULT_OUTPUT = ROOT / "05_OUTPUTS" / "root_rotor_manual_queue_next.jsonl"
TARGET_SCHEMA = "lucidota.root_rotor.bible_node_payload.v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def output_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("schema") == TARGET_SCHEMA and "what_it_is_and_does" in payload


def resolve_target(path_value: str) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else ROOT / p


def select_next_jobs(queue_path: Path, *, limit: int, model: str | None = None) -> tuple[list[dict[str, Any]], int, int]:
    rows = load_jsonl(queue_path)
    selected: list[dict[str, Any]] = []
    skipped_existing = 0
    skipped_model = 0
    for row in rows:
        if model and row.get("model") != model:
            skipped_model += 1
            continue
        target_file = row.get("target_file")
        if target_file and output_is_valid(resolve_target(str(target_file))):
            skipped_existing += 1
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected, skipped_existing, skipped_model


def write_next_batch(queue_path: Path = DEFAULT_QUEUE, output_path: Path = DEFAULT_OUTPUT, *, limit: int, model: str | None = None) -> dict[str, Any]:
    selected, skipped_existing, skipped_model = select_next_jobs(queue_path, limit=limit, model=model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected), encoding="utf-8")
    return {
        "schema": "lucidota.root_rotor.queue_batch.v1",
        "generated_at": now(),
        "queue_path": str(queue_path),
        "output_path": str(output_path),
        "limit": limit,
        "model": model,
        "jobs_selected": len(selected),
        "jobs_skipped_existing_output": skipped_existing,
        "jobs_skipped_model": skipped_model,
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write next unprocessed Root-Rotor queue batch.")
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = write_next_batch(Path(args.queue), Path(args.output), limit=args.limit, model=args.model)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=False))
    else:
        print(f"ROOT_ROTOR_QUEUE_BATCH={result['status']}")
        print(f"JOBS_SELECTED={result['jobs_selected']} OUTPUT={result['output_path']}")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
