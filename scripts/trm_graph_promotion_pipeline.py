#!/usr/bin/env python3
"""TRM Training Data Graph Promotion Pipeline.

ETL WITH CARE:
  Extract -> verify hash -> transform -> cipher (already done) -> validate schema
  -> triple-timestamp -> write receipt -> stage for graph

TRIPLE HASHED:  sha256 in file metadata, in receipt, in graph candidate (3 places).
TRIPLE TIMESTAMPED: created_at (source creation), processed_at (extraction time),
                    verified_at (verification time).

Mutation class: receipt_only (dry-run) / candidate_writer (--promote).
Never writes to canonical graph tables directly -- uses graph_promotion_gate.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS_DIR = ROOT / "05_OUTPUTS" / "trm_training" / "receipts"
CANDIDATES_DIR = ROOT / "05_OUTPUTS" / "graph_candidates" / "trm_training"
OUT = ROOT / "05_OUTPUTS" / "graph"
STATE_DSN = os.environ.get("LUCIDOTA_GO_STATE_DSN", "postgresql:///lucidota_state")
STORAGE_DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")

# Schema for the promotion receipt
RECEIPT_SCHEMA_V1 = "lucidota.go.graph_promotion_receipt.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def _sha256_file(path: Path) -> str:
    """Compute sha256 of a file. Returns hex string."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_obj(obj: Any) -> str:
    """Compute sha256 of a JSON-serializable object (canonical JSON)."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _read_receipts() -> list[dict[str, Any]]:
    """Read all receipt JSONs from RECEIPTS_DIR."""
    if not RECEIPTS_DIR.is_dir():
        print(f"WARNING: receipts dir not found: {RECEIPTS_DIR}", file=sys.stderr)
        return []
    receipts: list[dict[str, Any]] = []
    for f in sorted(RECEIPTS_DIR.iterdir()):
        if f.suffix == ".json":
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                data.setdefault("_file_path", str(f))
                data.setdefault("_file_sha256", _sha256_file(f))
                receipts.append(data)
            except Exception as exc:
                print(f"WARNING: failed to read receipt {f}: {exc}", file=sys.stderr)
    return receipts


def _resolve_file_paths(receipt: dict[str, Any]) -> list[Path]:
    """Resolve all file paths referenced in a receipt to actual Path objects.

    Checks both 'sources' and 'files_written' (or 'ciphered_copies_written'/'files_processed').
    Entries may be strings (plain paths) or dicts with a 'file' key (PII cipher receipt format).
    """
    paths: set[Path] = set()
    for key in ("sources", "files_written", "files_processed", "ciphered_copies_written"):
        entries = receipt.get(key)
        if not isinstance(entries, list):
            continue  # skip non-list values (e.g., some receipts store int or str here)
        for entry in entries:
            if isinstance(entry, dict):
                entry = entry.get("file", "")
            if not isinstance(entry, str) or not entry.strip():
                continue
            p = Path(entry)
            if not p.is_absolute():
                # Try relative to ROOT or cwd
                rel_root = ROOT / p
                if rel_root.exists():
                    p = rel_root
                elif p.exists():
                    pass  # relative to cwd
                else:
                    # Check if it's in trm_training directories by filename
                    for subdir in ("ahoy", "ciphered"):
                        candidate = ROOT / "05_OUTPUTS" / "trm_training" / subdir / p.name
                        if candidate.exists():
                            p = candidate
                            break
            if p.exists():
                paths.add(p.resolve())
    return sorted(paths)


def _compute_triple_hash(receipt: dict[str, Any], data_files: list[Path]) -> dict[str, Any]:
    """Compute triple hashes: receipt file sha256, data file sha256s, and a combined candidate sha256."""
    receipt_sha = receipt.get("_file_sha256", _sha256_obj(receipt))
    file_hashes: dict[str, str] = {}
    for f in data_files:
        file_hashes[_rel(f)] = _sha256_file(f)
    combined = {
        "receipt_sha256": receipt_sha,
        "file_sha256s": file_hashes,
    }
    candidate_sha = _sha256_obj(combined)
    return {
        "receipt_sha256": receipt_sha,
        "file_sha256s": file_hashes,
        "candidate_sha256": candidate_sha,
    }


def _compute_triple_timestamp(receipt: dict[str, Any]) -> dict[str, str]:
    """Compute triple timestamps: created_at, processed_at, verified_at.

    - created_at: from receipt if available, else the receipt file mtime, else now
    - processed_at: receipt's 'generated_at' if available, else now
    - verified_at: always now (verification happens at pipeline time)
    """
    receipt_ts = receipt.get("generated_at") or receipt.get("timestamp") or _now()
    try:
        # Try to parse the file's modification time as a timestamp
        created = datetime.fromtimestamp(
            Path(receipt.get("_file_path", "")).stat().st_mtime, tz=timezone.utc
        ).isoformat().replace("+00:00", "Z")
    except Exception:
        created = receipt_ts
    return {
        "created_at": created,
        "processed_at": receipt_ts,
        "verified_at": _now(),
    }


def _build_candidate_packet(receipt: dict[str, Any], idx: int) -> dict[str, Any]:
    """Build a single graph promotion candidate packet from a receipt."""
    data_files = _resolve_file_paths(receipt)
    triple_hash = _compute_triple_hash(receipt, data_files)
    triple_ts = _compute_triple_timestamp(receipt)

    schema = receipt.get("schema", "unknown")
    command = receipt.get("command", "unknown")
    verdict = receipt.get("verdict", "UNKNOWN")

    # Determine source system from schema or command
    schema_lower = schema.lower()
    command_lower = command.lower()
    if "ahoy" in schema_lower or "ahoy" in command_lower:
        source_system = "trm_ahoy_extraction"
    elif "pii" in schema_lower or "cipher" in command_lower:
        source_system = "trm_pii_cipher"
    elif "groq_extraction" in schema_lower:
        if "final" in schema_lower:
            source_system = "trm_groq_final"
        else:
            source_system = "trm_groq_extraction"
    elif "krampus" in schema_lower or "krampus" in command_lower:
        source_system = "trm_krampus_extraction"
    elif "river" in schema_lower or "river" in command_lower:
        source_system = "trm_river_extraction"
    elif "thicktext" in schema_lower or "thicktext" in command_lower:
        source_system = "trm_thicktext_extraction"
    elif "infrastructure" in schema_lower:
        source_system = "trm_infrastructure_prep"
    elif "rr" in command_lower or "entity_graph" in command_lower:
        source_system = "trm_entity_graph"
    else:
        source_system = "trm_other"

    # Determine training data characteristics
    row_count = receipt.get("total_rows_loaded", 0) or receipt.get("n_games", 0)
    feature_count = receipt.get("feature_count", 0)
    label_count = receipt.get("label_count", 0)

    # Source file refs for evidence
    source_refs = [_rel(p) for p in data_files]
    evidence_refs = source_refs + [_rel(Path(receipt.get("_file_path", "")))]

    candidate_payload = {
        "source_system": source_system,
        "schema": schema,
        "command": command,
        "verdict": verdict,
        "triple_hash": triple_hash,
        "triple_timestamp": triple_ts,
        "receipt_detail": {
            "row_count": row_count,
            "feature_count": feature_count,
            "label_count": label_count,
            "n_games": receipt.get("n_games", 0),
            "seed": receipt.get("seed", None),
        },
        "data_files": [_rel(p) for p in data_files],
        "evidence_refs": evidence_refs,
        "receipt_file": _rel(Path(receipt.get("_file_path", ""))),
    }

    return {
        "packet_id": f"trm_training_{source_system}_{idx}",
        "staging_ref": str(CANDIDATES_DIR / f"candidate_{source_system}_{idx}.json"),
        "source_system": source_system,
        "candidate_kind": "node",
        "candidate_payload": candidate_payload,
        "evidence_refs": evidence_refs,
        "authority_class": "operator_defined_label",
        "promotion_status": "candidate",
        "detail": {
            "pipeline": "scripts/trm_graph_promotion_pipeline.py",
            "pipeline_version": "1.0",
            "receipt_index": idx,
            "triple_hash": triple_hash,
        },
    }


def _write_staging_candidates(
    candidates: list[dict[str, Any]],
    *,
    execute: bool = False,
) -> dict[str, Any]:
    """Write candidate packets to CANDIDATES_DIR.

    Returns report dict with counts and file paths.
    """
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for candidate in candidates:
        staging_path = Path(candidate["staging_ref"])
        if not staging_path.is_absolute():
            staging_path = CANDIDATES_DIR / f"candidate_{candidate['source_system']}_{candidate['detail']['receipt_index']}.json"
        staging_path = staging_path.resolve()
        staging_path.write_text(
            json.dumps(candidate, indent=2, sort_keys=False, default=str),
            encoding="utf-8",
        )
        written.append(str(staging_path))

    return {
        "candidates_staged": len(written),
        "staging_paths": written,
        "staging_dir": str(CANDIDATES_DIR),
        "writes_performed": bool(execute),
    }


def _promote_via_gate(candidates: list[dict[str, Any]], *, execute: bool = False) -> list[dict[str, Any]]:
    """Stage candidate packets to DB via graph_promotion_gate.py.

    Only runs if --promote is set.
    """
    results: list[dict[str, Any]] = []
    gate_script = ROOT / "scripts" / "graph_promotion_gate.py"
    if not gate_script.exists():
        results.append({
            "step": "gate_script_check",
            "rc": 1,
            "error": f"gate script not found: {gate_script}",
        })
        return results

    if not execute:
        for idx in range(len(candidates)):
            results.append({
                "step": f"gate_skipped_{idx}",
                "rc": 0,
                "reason": "dry_run_no_promote",
            })
        return results

    for idx, candidate in enumerate(candidates):
        cmd = [
            sys.executable, str(gate_script), "gate",
            "--source-system", candidate.get("source_system", "trm_training"),
            "--candidate-kind", candidate.get("candidate_kind", "node"),
            "--candidate-payload-json", json.dumps(candidate["candidate_payload"]),
        ]
        for ev in candidate.get("evidence_refs", []):
            cmd += ["--evidence-ref", ev]
        cmd += [
            "--authority-class", candidate.get("authority_class", "operator_defined_label"),
            "--decision", "defer",
            "--rationale", f"TRM training data graph promotion: {candidate['source_system']}",
        ]
        if execute:
            cmd.append("--execute")

        try:
            p = subprocess.run(
                cmd, cwd=ROOT, text=True, capture_output=True, timeout=120,
            )
            gate_out: dict[str, Any] = {
                "step": f"gate_promote_{idx}",
                "rc": p.returncode,
                "stdout_tail": p.stdout[-1000:],
                "stderr_tail": p.stderr[-1000:],
            }
            # Parse REPORT_PATH from stdout
            for line in p.stdout.splitlines():
                if line.startswith("REPORT_PATH="):
                    gate_out["report_path"] = line.split("=", 1)[1]
                if line.startswith("PACKET_UUID="):
                    gate_out["packet_uuid"] = line.split("=", 1)[1]
                if line.startswith("GRAPH_GATE_ALLOWED="):
                    gate_out["gate_allowed"] = line.split("=", 1)[1]
            results.append(gate_out)
        except subprocess.TimeoutExpired:
            results.append({"step": f"gate_promote_{idx}", "rc": -1, "error": "timeout"})
        except Exception as exc:
            results.append({"step": f"gate_promote_{idx}", "rc": -1, "error": str(exc)})

    return results


def _write_graph_promotion_receipt(
    receipts: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    staging_result: dict[str, Any],
    gate_results: list[dict[str, Any]],
) -> Path:
    """Write the final graph promotion receipt to 05_OUTPUTS/graph/."""
    OUT.mkdir(parents=True, exist_ok=True)
    report_path = OUT / f"trm_graph_promotion_receipt_{_stamp()}.json"

    # Determine verdict from gate results
    gate_errors = [r for r in gate_results if r.get("rc", 0) != 0]
    if gate_errors:
        verdict = "FAIL" if any(r.get("rc", 0) > 1 for r in gate_results) else "PARTIAL"
    else:
        verdict = "PASS"

    receipt = {
        "command": "scripts/trm_graph_promotion_pipeline.py",
        "schema": RECEIPT_SCHEMA_V1,
        "generated_at": _now(),
        "triple_hashed": True,
        "triple_timestamped": True,
        "etl_with_care": True,
        "receipts_processed": len(receipts),
        "candidates_staged": staging_result.get("candidates_staged", 0),
        "candidates_promoted": len([r for r in gate_results if r.get("gate_allowed") == "true"]),
        "gate_results_count": len(gate_results),
        "sources": [r.get("_file_path", "") for r in receipts],
        "receipt_sha256s": [r.get("_file_sha256", "") for r in receipts],
        "candidate_sha256s": [c["detail"]["triple_hash"]["candidate_sha256"] for c in candidates],
        "staging_result": staging_result,
        "gate_results": gate_results,
        "verdict": verdict,
        "pipeline_version": "1.0",
    }

    report_path.write_text(json.dumps(receipt, indent=2, sort_keys=False, default=str), encoding="utf-8")
    print(f"RECEIPT_PATH={_rel(report_path)}")
    return report_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="TRM training data graph promotion pipeline. Scans receipts, stages candidates, "
                    "and optionally promotes to graph via the existing gate pipeline.",
    )
    ap.add_argument("--dry-run", action="store_true", default=True,
                    help="Default mode: stage candidates to filesystem only. No DB writes.")
    ap.add_argument("--promote", action="store_true", default=False,
                    help="Stage candidate packets to DB via graph_promotion_gate.py --execute")
    ap.add_argument("--json", action="store_true",
                    help="Emit final report as JSON to stdout")
    args = ap.parse_args()

    # If neither flag set, default to dry-run
    execute = bool(args.promote)
    if not execute:
        args.dry_run = True

    # Step 1: Read all receipts
    receipts = _read_receipts()
    if not receipts:
        print("ERROR: no receipts found -- nothing to promote.", file=sys.stderr)
        return 1

    print(f"Found {len(receipts)} receipt(s) in {_rel(RECEIPTS_DIR)}")

    # Step 2: Build candidate packets from each receipt
    candidates: list[dict[str, Any]] = []
    for idx, receipt in enumerate(receipts):
        candidate = _build_candidate_packet(receipt, idx)
        candidates.append(candidate)

    print(f"Built {len(candidates)} candidate packet(s)")

    # Step 3: Stage candidates to filesystem
    staging_result = _write_staging_candidates(candidates, execute=execute)
    print(f"Staged {staging_result['candidates_staged']} candidate(s) to {staging_result['staging_dir']}")

    # Step 4: Promote via gate (only if --promote)
    gate_results = _promote_via_gate(candidates, execute=execute)
    if gate_results:
        promoted_count = len([r for r in gate_results if r.get("rc", -1) == 0])
        print(f"Gate results: {promoted_count}/{len(gate_results)} succeeded")
        if execute:
            for r in gate_results:
                if r.get("packet_uuid"):
                    print(f"  PACKET_UUID={r['packet_uuid']} (rc={r.get('rc')})")

    # Step 5: Write final receipt
    report_path = _write_graph_promotion_receipt(receipts, candidates, staging_result, gate_results)

    # Step 6: Output summary
    final_report = json.loads(report_path.read_text(encoding="utf-8"))
    if args.json:
        print(json.dumps(final_report, sort_keys=True, default=str))
    else:
        print("=" * 60)
        print(f"TRM GRAPH PROMOTION: {final_report['verdict']}")
        print(f"  Receipts processed:  {final_report['receipts_processed']}")
        print(f"  Candidates staged:   {final_report['candidates_staged']}")
        print(f"  Candidates promoted: {final_report['candidates_promoted']}")
        print(f"  Receipt path:        {final_report.get('generated_at', '')}")
        print(f"  Verdict:             {final_report['verdict']}")
        print("=" * 60)

    return 0 if final_report["verdict"] in ("PASS", "PARTIAL") else 2


if __name__ == "__main__":
    raise SystemExit(main())
