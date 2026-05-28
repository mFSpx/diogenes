#!/usr/bin/env python3
"""Bounded LUCIDOTA matrix executor.

Streams security quarantine manifests and RGAUNTLET JSONL without loading large
inputs. Writes dead-list/unblock receipts and bounded gauntlet proof batches.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "matrix_execution"
DEFAULT_MANIFEST = ROOT / "05_OUTPUTS/security/security_quarantine_manifest_20260520T065123Z.json"
DEFAULT_GAUNTLET = ROOT / "05_OUTPUTS/work_orders/lucidota_600_work_order_gauntlet_20260517T211943051910Z.jsonl"


def now_z() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")


def meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    with Path("/proc/meminfo").open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                out[parts[0].rstrip(":")] = int(parts[1]) // 1024
    return out


def resource_guard(min_mem_mb: int, min_swap_mb: int) -> dict[str, Any]:
    m = meminfo()
    available = m.get("MemAvailable", 0)
    swap_free = m.get("SwapFree", 0)
    ok = available >= min_mem_mb and swap_free >= min_swap_mb
    receipt = {
        "schema": "lucidota.matrix.resource_guard.v1",
        "generated_at": now_z(),
        "mem_available_mb": available,
        "swap_free_mb": swap_free,
        "min_mem_mb": min_mem_mb,
        "min_swap_mb": min_swap_mb,
        "ok": ok,
    }
    if not ok:
        raise SystemExit(f"resource_guard_failed:{receipt}")
    return receipt


def iter_json_array_for_key(path: Path, key: str, *, chunk_size: int = 65536, max_buffer: int = 8_000_000) -> Iterator[Any]:
    """Yield objects from a top-level JSON array named `key` using bounded buffering."""
    decoder = json.JSONDecoder()
    needle = '"' + key + '"'
    found_key = False
    in_array = False
    buf = ""
    eof = False
    with path.open("r", encoding="utf-8") as fh:
        while True:
            if not eof:
                chunk = fh.read(chunk_size)
                if chunk:
                    buf += chunk
                    if len(buf) > max_buffer:
                        # Keep this hard fail rather than silently buffering unbounded data.
                        raise RuntimeError(f"json_stream_buffer_exceeded:{key}:{len(buf)}")
                else:
                    eof = True
            if not found_key:
                idx = buf.find(needle)
                if idx == -1:
                    if eof:
                        return
                    buf = buf[-len(needle)-32:]
                    continue
                buf = buf[idx + len(needle):]
                found_key = True
            if found_key and not in_array:
                idx = buf.find("[")
                if idx == -1:
                    if eof:
                        return
                    continue
                buf = buf[idx + 1:]
                in_array = True
            while in_array:
                buf = buf.lstrip()
                if buf.startswith(","):
                    buf = buf[1:].lstrip()
                if buf.startswith("]"):
                    return
                try:
                    obj, end = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    break
                yield obj
                buf = buf[end:]
            if eof:
                return


def small_manifest_summary(path: Path, *, max_full_read_bytes: int = 5_000_000) -> dict[str, Any]:
    if path.stat().st_size > max_full_read_bytes:
        # For bigger future manifests, callers still get array streaming; summary is intentionally minimal.
        return {"summary_mode": "large_manifest_no_full_read", "size_bytes": path.stat().st_size}
    with path.open("r", encoding="utf-8") as fh:
        d = json.load(fh)
    keys = [
        "schema", "mode", "files_scanned", "included_files_scanned", "excluded_files_scanned",
        "included_findings_count", "excluded_findings_count", "counts", "excluded_counts",
        "clean_manifest", "brain_archaeology_full_ingest_allowed", "brain_archaeology_ingest_rule",
    ]
    return {k: d.get(k) for k in keys}


def process_quarantine(path: Path, *, batch_size: int) -> dict[str, Any]:
    out_stamp = stamp()
    summary = small_manifest_summary(path)
    dead_quarantined = OUT / f"security_quarantine_dead_quarantined_{out_stamp}.jsonl"
    dead_deferred = OUT / f"security_quarantine_dead_deferred_{out_stamp}.jsonl"
    sample_counts = collections.Counter()
    listed = 0
    files = {"quarantined": dead_quarantined.open("w", encoding="utf-8"), "deferred": dead_deferred.open("w", encoding="utf-8")}
    try:
        for finding in iter_json_array_for_key(path, "excluded_findings_sample"):
            decision = str(finding.get("decision") or "unknown")
            row = {
                "schema": "lucidota.security.dead_list_item.v1",
                "record_kind": "exact_manifest_sample",
                "generated_at": now_z(),
                "source_manifest": rel(path),
                "decision": decision,
                "path": finding.get("path"),
                "rule_names": finding.get("rule_names") or [],
                "excluded_from_ingest": True,
                "secret_values_printed": False,
            }
            target = files.get(decision)
            if target:
                target.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            sample_counts[decision] += 1
            listed += 1
            if listed % max(1, batch_size) == 0:
                for fh in files.values():
                    fh.flush()
    finally:
        for fh in files.values():
            fh.close()
    excluded_counts = summary.get("excluded_counts") or {}
    for decision in ("quarantined", "deferred"):
        total = int(excluded_counts.get(decision) or 0)
        missing = max(0, total - sample_counts.get(decision, 0))
        if missing:
            target_path = dead_quarantined if decision == "quarantined" else dead_deferred
            with target_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "schema": "lucidota.security.dead_list_item.v1",
                    "record_kind": "aggregate_unlisted_count",
                    "generated_at": now_z(),
                    "source_manifest": rel(path),
                    "decision": decision,
                    "unlisted_count": missing,
                    "reason": "source manifest records only excluded_findings_sample; aggregate count is final but exact paths are unavailable without rescanning",
                    "excluded_from_ingest": True,
                    "secret_values_printed": False,
                }, sort_keys=True) + "\n")
    phase05_unblocked = bool(summary.get("clean_manifest") and summary.get("brain_archaeology_full_ingest_allowed") and summary.get("included_findings_count") == 0)
    receipt = {
        "schema": "lucidota.matrix.quarantine_purge_receipt.v1",
        "generated_at": now_z(),
        "source_manifest": rel(path),
        "summary": summary,
        "dead_lists": {"quarantined": rel(dead_quarantined), "deferred": rel(dead_deferred)},
        "excluded_counts_final": excluded_counts,
        "exact_sample_items_written": listed,
        "phase05_unblocked_for_included_clean_files": phase05_unblocked,
        "quarantined_or_deferred_items_admitted": False,
        "secret_values_printed": False,
        "blockers": [] if phase05_unblocked else ["phase05_not_unblocked_by_manifest"],
    }
    receipt_path = OUT / f"quarantine_purge_receipt_{out_stamp}.json"
    receipt["receipt_path"] = rel(receipt_path)
    write_json(receipt_path, receipt)
    return receipt


def stream_gauntlet(path: Path, subsystems: set[int], statuses: set[str], limit: int) -> Iterator[dict[str, Any]]:
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                subsystem = int(row.get("subsystem_number"))
            except Exception:
                continue
            status = str(row.get("status") or "").upper()
            if subsystem in subsystems and status in statuses:
                row["_line_no"] = line_no
                yield row
                count += 1
                if limit and count >= limit:
                    return


def run_reaped(cmd: list[str], *, cwd: Path = ROOT, timeout: int = 120, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = now_z()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env={**os.environ, **(env or {})},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            stdout, stderr = proc.communicate(timeout=5)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
            stdout, stderr = proc.communicate()
    return {
        "cmd": cmd,
        "cwd": rel(cwd),
        "started_at": started,
        "finished_at": now_z(),
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "stdout_tail": (stdout or "")[-4000:],
        "stderr_tail": (stderr or "")[-4000:],
        "reaped": True,
    }


def proof_commands(command_timeout: int, active_subsystems: set[int]) -> list[dict[str, Any]]:
    command_map: dict[int, tuple[str, list[str], Path, dict[str, str]]] = {
        1: ("subsystem_01_lucidota_kernel", ["cargo", "test", "-p", "lucidota-kernel", "--quiet"], ROOT / "01_REPOS/lucidota_etl", {"CARGO_BUILD_JOBS": "1"}),
        2: ("subsystem_02_rust_workspace", ["cargo", "test", "--workspace", "--quiet"], ROOT / "01_REPOS/lucidota_etl", {"CARGO_BUILD_JOBS": "1"}),
        3: ("subsystem_03_claw_rust_workspace", ["cargo", "test", "--workspace", "--quiet"], ROOT / "01_REPOS/claudecode/rust", {"CARGO_BUILD_JOBS": "1"}),
        4: ("subsystem_04_absurd_queue_soak", [sys.executable, "scripts/spine_queue_soak_test.py", "--execute", "--jobs", "2"], ROOT, {}),
        5: ("subsystem_05_krampus_bounded_inventory", [sys.executable, "scripts/krampus_bounded_inventory.py", "--target", "KRAMPUSCHEWING", "--dry-run", "--max-files", "500"], ROOT, {}),
        6: ("subsystem_06_krampus_absurd_wrapper_audit", [sys.executable, "scripts/spine_krampus_worker.py", "--action", "audit", "--dry-run"], ROOT, {}),
        7: ("subsystem_07_document_parse_bakeoff", [sys.executable, "scripts/document_parse_bakeoff.py", "--dry-run"], ROOT, {}),
        14: ("subsystem_14_tickletrunk", [sys.executable, "scripts/tickletrunk_scan.py", "--check"], ROOT, {}),
        15: ("status_ledger_check", [sys.executable, "scripts/lucidota_status_ledger.py", "--check"], ROOT, {}),
        18: ("subsystem_18_gliner_dry_run", [sys.executable, "scripts/gliner_claim_packet_dry_run.py", "--text", "Operator routes KORPUS through Chrono-Ledger. Command Envelope Protocol preserves instruction provenance."], ROOT, {}),
    }
    commands: list[tuple[str, list[str], Path, dict[str, str]]] = [
        command_map[subsystem] for subsystem in sorted(active_subsystems) if subsystem in command_map
    ]
    if 15 not in active_subsystems:
        commands.append(("status_ledger_check", [sys.executable, "scripts/lucidota_status_ledger.py", "--check"], ROOT, {}))
    results = []
    for name, cmd, cwd, env in commands:
        results.append({"proof_name": name, **run_reaped(cmd, cwd=cwd, timeout=command_timeout, env=env)})
    return results


def scoped_git_status() -> list[str]:
    paths = [
        "scripts/matrix_stream_executor.py",
        "scripts/gauntlet_state_promoter.py",
        "scripts/chroma_gliner_bounded_probe.py",
        "scripts/spine_queue_soak_test.py",
        "scripts/absurd_queue_spine.py",
        "scripts/spine_krampus_worker.py",
        "scripts/spine_document_parse_worker.py",
        "scripts/korpus_krampii.py",
        "scripts/simplemem_candidate_index.py",
        "scripts/lucidota_security_quarantine_gate.py",
        "scripts/phase05_allowlisted_ingest.py",
        "scripts/safe_stress_test.py",
        "tests/test_quarantine_streaming_guards.py",
        ".lucidota_agents/specialists",
    ]
    cp = subprocess.run(["git", "status", "--short", "--", *paths], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return [line for line in cp.stdout.splitlines() if line.strip()]


def proof_for_task(task: dict[str, Any], proofs_by_name: dict[str, dict[str, Any]], quarantine_receipt: dict[str, Any] | None) -> tuple[str | None, dict[str, Any] | None]:
    subsystem = int(task.get("subsystem_number") or 0)
    if subsystem == 1:
        proof = proofs_by_name.get("subsystem_01_lucidota_kernel")
        return "subsystem_01_lucidota_kernel", proof
    if subsystem == 2:
        proof = proofs_by_name.get("subsystem_02_rust_workspace")
        return "subsystem_02_rust_workspace", proof
    if subsystem == 3:
        proof = proofs_by_name.get("subsystem_03_claw_rust_workspace")
        return "subsystem_03_claw_rust_workspace", proof
    if subsystem == 4:
        proof = proofs_by_name.get("subsystem_04_absurd_queue_soak")
        return "subsystem_04_absurd_queue_soak", proof
    if subsystem == 5:
        proof = proofs_by_name.get("subsystem_05_krampus_bounded_inventory")
        return "subsystem_05_krampus_bounded_inventory", proof
    if subsystem == 6:
        proof = proofs_by_name.get("subsystem_06_krampus_absurd_wrapper_audit")
        return "subsystem_06_krampus_absurd_wrapper_audit", proof
    if subsystem == 7:
        proof = proofs_by_name.get("subsystem_07_document_parse_bakeoff")
        return "subsystem_07_document_parse_bakeoff", proof
    if subsystem == 14:
        proof = proofs_by_name.get("subsystem_14_tickletrunk")
        return "subsystem_14_tickletrunk", proof
    if subsystem == 18:
        proof = proofs_by_name.get("subsystem_18_gliner_dry_run")
        return "subsystem_18_gliner_dry_run", proof
    if subsystem == 16 and quarantine_receipt:
        ok = bool(quarantine_receipt.get("phase05_unblocked_for_included_clean_files")) and not quarantine_receipt.get("blockers")
        proof = {
            "proof_name": "subsystem_16_quarantine_dead_list",
            "returncode": 0 if ok else 2,
            "timed_out": False,
            "cmd": ["python3", "scripts/matrix_stream_executor.py", "--quarantine-manifest", quarantine_receipt.get("source_manifest", "<manifest>")],
            "stdout_tail": "PHASE05_UNBLOCKED=" + str(ok).lower(),
            "stderr_tail": "",
            "reaped": True,
            "receipt_path": quarantine_receipt.get("receipt_path"),
        }
        return "subsystem_16_quarantine_dead_list", proof
    return None, None


def write_work_order_proof_packets(
    *,
    selected_batch: Path,
    proofs: list[dict[str, Any]],
    quarantine_receipt: dict[str, Any] | None,
    closure_path: Path,
    out_stamp: str,
) -> tuple[Path, dict[str, int]]:
    proofs_by_name = {p["proof_name"]: p for p in proofs}
    files_changed = scoped_git_status()
    packet_path = OUT / f"work_order_proof_packets_{out_stamp}.jsonl"
    counts = {"closed_for_review": 0, "still_blocked": 0}
    with selected_batch.open("r", encoding="utf-8") as src, packet_path.open("w", encoding="utf-8") as out:
        for line in src:
            if not line.strip():
                continue
            task = json.loads(line)
            proof_name, proof = proof_for_task(task, proofs_by_name, quarantine_receipt)
            proof_ok = bool(proof and proof.get("returncode") == 0 and not proof.get("timed_out"))
            status_after = "PASSED_FOR_REVIEW" if proof_ok else task.get("status")
            if proof_ok:
                counts["closed_for_review"] += 1
            else:
                counts["still_blocked"] += 1
            packet = {
                "schema": "lucidota.matrix.work_order_proof_packet.v1",
                "generated_at": now_z(),
                "work_order_id": task.get("work_order_id"),
                "status_before": task.get("status"),
                "status_after": status_after,
                "source_gauntlet_mutated": False,
                "files_read": [
                    rel(selected_batch),
                    quarantine_receipt.get("source_manifest") if quarantine_receipt else None,
                ],
                "files_changed_oracle": files_changed,
                "exact_commands_run": [proof.get("cmd")] if proof else [],
                "command_return_codes": [proof.get("returncode")] if proof else [],
                "stdout_summary": (proof.get("stdout_tail") or "")[-1000:] if proof else "",
                "stderr_summary": (proof.get("stderr_tail") or "")[-1000:] if proof else "no_proof_available_for_subsystem",
                "receipt_path": rel(packet_path),
                "supporting_receipts": [
                    quarantine_receipt.get("receipt_path") if quarantine_receipt else None,
                    rel(closure_path),
                ],
                "proof_name": proof_name,
                "reason_closure_is_justified": (
                    "bounded proof passed; source gauntlet left unchanged pending review"
                    if proof_ok
                    else "no passing proof packet available"
                ),
                "remaining_blocker": None if proof_ok else "proof_missing_or_failed",
            }
            packet["files_read"] = [x for x in packet["files_read"] if x]
            packet["supporting_receipts"] = [x for x in packet["supporting_receipts"] if x]
            out.write(json.dumps(packet, sort_keys=True, default=str) + "\n")
    return packet_path, counts


def process_gauntlet(path: Path, *, subsystems: set[int], statuses: set[str], limit: int, batch_size: int, execute_proofs: bool, command_timeout: int, quarantine_receipt: dict[str, Any] | None) -> dict[str, Any]:
    out_stamp = stamp()
    batch_path = OUT / f"gauntlet_execution_batch_{out_stamp}.jsonl"
    by_subsystem: dict[int, int] = collections.Counter()
    by_status: dict[str, int] = collections.Counter()
    selected = 0
    with batch_path.open("w", encoding="utf-8") as out:
        for row in stream_gauntlet(path, subsystems, statuses, limit):
            selected += 1
            subsystem = int(row["subsystem_number"])
            status = str(row.get("status") or "").upper()
            by_subsystem[subsystem] += 1
            by_status[status] += 1
            task = {
                "schema": "lucidota.matrix.gauntlet_task.v1",
                "generated_at": now_z(),
                "source_gauntlet": rel(path),
                "line_no": row.get("_line_no"),
                "work_order_id": row.get("work_order_id"),
                "subsystem_number": subsystem,
                "subsystem_name": row.get("subsystem_name"),
                "status": status,
                "gauntlet_step": row.get("gauntlet_step"),
                "severity": row.get("severity"),
                "target_paths": row.get("target_paths") or [],
                "commands": row.get("commands") or [],
                "recommended_littleworker": recommend_worker(subsystem, row),
                "receipt_required_before_pass": True,
            }
            out.write(json.dumps(task, sort_keys=True, default=str) + "\n")
            if selected % max(1, batch_size) == 0:
                out.flush()
    proofs = proof_commands(command_timeout, subsystems) if execute_proofs else []
    closure_path = OUT / f"gauntlet_closure_candidates_{out_stamp}.jsonl"
    passed_proofs = {p["proof_name"]: p for p in proofs if p.get("returncode") == 0 and not p.get("timed_out")}
    with closure_path.open("w", encoding="utf-8") as out:
        for proof_name, proof in passed_proofs.items():
            out.write(json.dumps({
                "schema": "lucidota.matrix.gauntlet_closure_candidate.v1",
                "generated_at": now_z(),
                "proof_name": proof_name,
                "candidate_status": "PASSED_FOR_REVIEW",
                "proof_returncode": proof.get("returncode"),
                "source_batch": rel(batch_path),
                "receipt_required_before_source_gauntlet_mutation": True,
            }, sort_keys=True, default=str) + "\n")
    proof_packet_path, proof_packet_counts = write_work_order_proof_packets(
        selected_batch=batch_path,
        proofs=proofs,
        quarantine_receipt=quarantine_receipt,
        closure_path=closure_path,
        out_stamp=out_stamp,
    )
    receipt = {
        "schema": "lucidota.matrix.gauntlet_execution_receipt.v1",
        "generated_at": now_z(),
        "source_gauntlet": rel(path),
        "selected_task_batch": rel(batch_path),
        "closure_candidates": rel(closure_path),
        "work_order_proof_packets": rel(proof_packet_path),
        "work_order_proof_packet_counts": proof_packet_counts,
        "selected_count": selected,
        "selected_by_subsystem": dict(sorted(by_subsystem.items())),
        "selected_by_status": dict(by_status),
        "proofs_executed": proofs,
        "quarantine_receipt": quarantine_receipt.get("receipt_path") if quarantine_receipt else None,
        "source_gauntlet_mutated": False,
        "limits": {"limit": limit, "batch_size": batch_size, "command_timeout": command_timeout},
        "blockers": [p["proof_name"] for p in proofs if p.get("returncode") != 0 or p.get("timed_out")],
    }
    receipt_path = OUT / f"gauntlet_execution_receipt_{out_stamp}.json"
    receipt["receipt_path"] = rel(receipt_path)
    write_json(receipt_path, receipt)
    return receipt


def recommend_worker(subsystem: int, row: dict[str, Any]) -> str:
    step = str(row.get("gauntlet_step") or "").lower()
    if subsystem == 1:
        return "22_status_ledger_closer"
    if subsystem == 2:
        return "07_cargo_parity_mapper"
    if subsystem == 3:
        return "07_cargo_parity_mapper"
    if subsystem == 4:
        return "19_sql_lock_doctor"
    if subsystem == 5:
        return "08_mime_interrogator"
    if subsystem == 6:
        return "15_chroma_chunk_drainer"
    if subsystem == 7:
        return "05_dead_letter_necromancer"
    if subsystem == 14:
        return "17_tickletrunk_reuse_scout"
    if subsystem == 16:
        return "14_security_quarantine_triager"
    if subsystem == 18:
        return "16_gliner_span_reconciler"
    if "schema" in step or subsystem == 26:
        return "09_schema_migration_surgeon"
    if "graph" in str(row.get("subsystem_name") or "").lower():
        return "10_graph_orphan_sweeper"
    if "ledger" in step:
        return "13_receipt_chain_auditor"
    return "21_work_order_prosecutor"


def parse_subsystems(raw: str) -> set[int]:
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quarantine-manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--gauntlet", default=str(DEFAULT_GAUNTLET))
    ap.add_argument("--subsystems", default="2,14,16,18")
    ap.add_argument("--statuses", default="BLOCKED,PENDING")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--min-mem-mb", type=int, default=1500)
    ap.add_argument("--min-swap-mb", type=int, default=2048)
    ap.add_argument("--command-timeout", type=int, default=180)
    ap.add_argument("--execute-proofs", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    guard = resource_guard(args.min_mem_mb, args.min_swap_mb)
    q_receipt = process_quarantine(Path(args.quarantine_manifest), batch_size=args.batch_size)
    g_receipt = process_gauntlet(
        Path(args.gauntlet),
        subsystems=parse_subsystems(args.subsystems),
        statuses={s.strip().upper() for s in args.statuses.split(",") if s.strip()},
        limit=args.limit,
        batch_size=args.batch_size,
        execute_proofs=args.execute_proofs,
        command_timeout=args.command_timeout,
        quarantine_receipt=q_receipt,
    )
    final = {
        "schema": "lucidota.matrix.execution_summary.v1",
        "generated_at": now_z(),
        "resource_guard": guard,
        "quarantine_receipt": q_receipt.get("receipt_path"),
        "gauntlet_receipt": g_receipt.get("receipt_path"),
        "phase05_unblocked_for_included_clean_files": q_receipt.get("phase05_unblocked_for_included_clean_files"),
        "gauntlet_selected_count": g_receipt.get("selected_count"),
        "proof_blockers": g_receipt.get("blockers"),
    }
    final_path = OUT / f"matrix_execution_summary_{stamp()}.json"
    final["receipt_path"] = rel(final_path)
    write_json(final_path, final)
    print(f"MATRIX_SUMMARY={rel(final_path)}")
    print(f"QUARANTINE_RECEIPT={q_receipt.get('receipt_path')}")
    print(f"GAUNTLET_RECEIPT={g_receipt.get('receipt_path')}")
    print("PHASE05_UNBLOCKED=" + str(bool(q_receipt.get("phase05_unblocked_for_included_clean_files"))).lower())
    return 0 if not g_receipt.get("blockers") else 2


if __name__ == "__main__":
    raise SystemExit(main())
