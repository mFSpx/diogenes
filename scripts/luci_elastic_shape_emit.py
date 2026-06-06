#!/usr/bin/env python3
"""Emit an elastic-shape receipt and optionally write it to Postgres."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import psycopg


ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "01_REPOS" / "lucidota_resonance"
DEFAULT_DSN = os.environ.get("LUCIDOTA_GO_STATE_DSN") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"


def run_receipt(args: list[str]) -> dict[str, Any]:
    cmd = [
        "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        str(CRATE / "Cargo.toml"),
        "--bin",
        "lucidota_elastic_shape",
        "--",
        *args,
    ]
    proc = subprocess.run(cmd, cwd=CRATE, text=True, capture_output=True, check=True)
    return json.loads(proc.stdout)


def write_db(dsn: str, receipt: dict[str, Any]) -> None:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lucidota_runtime.elastic_shape_receipt (
                    receipt_uuid,
                    artifact_uuid,
                    source,
                    source_hash,
                    signature,
                    collision_signature,
                    dimensions,
                    entropy_hint,
                    shape_vector,
                    active_resonances,
                    fidelity,
                    residual_mass,
                    residual_vector,
                    collision,
                    canon_status,
                    route_context,
                    detail
                )
                VALUES (
                    COALESCE(%(receipt_uuid)s::uuid, gen_random_uuid()),
                    %(artifact_uuid)s::uuid,
                    %(source)s,
                    %(source_hash)s,
                    %(signature)s,
                    %(collision_signature)s,
                    %(dimensions)s,
                    %(entropy_hint)s,
                    %(shape_vector)s::jsonb,
                    %(active_resonances)s::jsonb,
                    %(fidelity)s,
                    %(residual_mass)s,
                    %(residual_vector)s::jsonb,
                    %(collision)s,
                    %(canon_status)s,
                    %(route_context)s::jsonb,
                    %(detail)s::jsonb
                )
                """,
                {
                    "receipt_uuid": receipt.get("receipt_uuid"),
                    "artifact_uuid": receipt.get("artifact_uuid"),
                    "source": receipt.get("source", "Runtime"),
                    "source_hash": receipt.get("source_hash", ""),
                    "signature": receipt.get("signature", ""),
                    "collision_signature": receipt.get("collision_signature", ""),
                    "dimensions": receipt.get("dimensions", 0),
                    "entropy_hint": receipt.get("entropy_hint", 0.0),
                    "shape_vector": json.dumps(receipt.get("shape_vector", [])),
                    "active_resonances": json.dumps(receipt.get("active_resonances", [])),
                    "fidelity": receipt.get("fidelity", 0.0),
                    "residual_mass": receipt.get("residual_mass", 0.0),
                    "residual_vector": json.dumps(receipt.get("residual_vector", [])),
                    "collision": receipt.get("collision", False),
                    "canon_status": receipt.get("canon_status", "not_truth_runtime_only"),
                    "route_context": json.dumps(receipt.get("route_context", {})),
                    "detail": json.dumps(receipt.get("detail", {})),
                },
            )
        conn.commit()


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit a runtime elastic-shape receipt and optionally write it to Postgres.")
    ap.add_argument("--artifact-uuid", required=True)
    ap.add_argument("--source", default="Runtime")
    ap.add_argument("--synthetic", action="store_true", help="Mark the emitted receipt as Synthetic source.")
    ap.add_argument("--min-dims", type=int, default=2)
    ap.add_argument("--max-dims", type=int, default=128)
    ap.add_argument("--entropy-hint", type=float, default=0.0)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--write-db", action="store_true", default=True)
    ap.add_argument("--no-write-db", dest="write_db", action="store_false")
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--base-url", default="")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--signal", action="append", default=[], help="TOKEN=VALUE signal pair; may be repeated")
    args = ap.parse_args()

    receipt = run_receipt(
        [
            "--artifact-uuid",
            args.artifact_uuid,
            "--source",
            "Synthetic" if args.synthetic else args.source,
            "--min-dims",
            str(args.min_dims),
            "--max-dims",
            str(args.max_dims),
            "--entropy-hint",
            str(args.entropy_hint),
            "--threshold",
            str(args.threshold),
            *sum(([ "--signal", spec] for spec in args.signal), []),
        ]
    )
    if args.write_db:
        write_db(args.dsn, receipt)
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
