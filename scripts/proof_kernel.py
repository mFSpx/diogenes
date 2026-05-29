#!/usr/bin/env python3
"""Proof Kernel v1: byte custody without truth promotion or graph mutation."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORAGE = ROOT / "09_STORAGE" / "proof_kernel"
DEFAULT_OUT = ROOT / "05_OUTPUTS" / "proof_kernel"
SCHEMA = "lucidota.proof_object.v1"
RECEIPT_SCHEMA = "lucidota.proof_kernel.receipt.v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")


def storage_root(args: argparse.Namespace) -> Path:
    return Path(args.storage_root).resolve() if args.storage_root else DEFAULT_STORAGE


def out_root(args: argparse.Namespace) -> Path:
    return Path(args.output_dir).resolve() if args.output_dir else DEFAULT_OUT


def object_path(root: Path, digest: str, suffix: str = "") -> Path:
    name = digest + suffix
    return root / "objects" / digest[:2] / name


def ensure_storage(root: Path) -> None:
    (root / "objects").mkdir(parents=True, exist_ok=True)
    (root / "index").mkdir(parents=True, exist_ok=True)


def make_immutable(path: Path) -> None:
    try:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except PermissionError:
        pass


def store_bytes(root: Path, data: bytes, suffix: str = "") -> tuple[str, int, Path, bool]:
    ensure_storage(root)
    digest = sha256_bytes(data)
    path = object_path(root, digest, suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    if existed:
        if path.read_bytes() != data:
            raise RuntimeError(f"immutable_object_hash_collision:{digest}")
    else:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        make_immutable(tmp)
        tmp.replace(path)
        make_immutable(path)
    return digest, len(data), path, existed


def store_file(root: Path, source: Path) -> tuple[str, int, Path, bool]:
    if not source.exists():
        raise FileNotFoundError(f"missing_source:{source}")
    if not source.is_file():
        raise ValueError(f"source_not_file:{source}")
    data = source.read_bytes()
    digest = sha256_bytes(data)
    path = object_path(root, digest, source.suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    if existed:
        if path.read_bytes() != data:
            raise RuntimeError(f"immutable_object_hash_collision:{digest}")
    else:
        try:
            os.link(source, path)
        except OSError:
            shutil.copy2(source, path)
        make_immutable(path)
    return digest, len(data), path, existed


def index_path(root: Path) -> Path:
    ensure_storage(root)
    return root / "index" / "proofs.jsonl"


def iter_index(root: Path) -> list[dict[str, Any]]:
    path = index_path(root)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
        except json.JSONDecodeError:
            continue
    return rows


def append_index(root: Path, proof: dict[str, Any]) -> bool:
    rows = iter_index(root)
    if any(r.get("proof_id") == proof.get("proof_id") for r in rows):
        return False
    path = index_path(root)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(proof, sort_keys=True, ensure_ascii=False) + "\n")
    return True


def proof_id_for(source_type: str, digest: str, original: str) -> str:
    return "proof_" + hashlib.sha256(f"{source_type}\0{digest}\0{original}".encode("utf-8")).hexdigest()[:32]


def duplicate_for(root: Path, digest: str, proof_id: str) -> str | None:
    for row in iter_index(root):
        if row.get("sha256") == digest and row.get("proof_id") != proof_id:
            return str(row.get("proof_id"))
    return None


def write_receipt(args: argparse.Namespace, payload: dict[str, Any]) -> Path:
    out = out_root(args)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"proof_kernel_{payload.get('action','receipt')}_{payload.get('proof_id','none')}_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False), encoding="utf-8")
    print("REPORT_PATH=" + rel(path))
    print("PROOF_KERNEL=" + payload["verdict"])
    if payload.get("proof_id"):
        print("PROOF_ID=" + str(payload["proof_id"]))
    return path


def build_proof(root: Path, *, source_type: str, original: str, stored: Path, digest: str, size: int, extractor_status: str, extra: dict[str, Any]) -> dict[str, Any]:
    pid = proof_id_for(source_type, digest, original)
    duplicate_of = duplicate_for(root, digest, pid)
    proof = {
        "schema": SCHEMA,
        "proof_id": pid,
        "source_type": source_type,
        "original_path_or_uri": original,
        "stored_artifact_path": rel(stored),
        "sha256": digest,
        "size": size,
        "created_at": now(),
        "ingested_at": now(),
        "extractor_status": extractor_status,
        "custody_events": [
            {"event": "ingested", "at": now(), "storage": rel(stored), "sha256": digest},
            {"event": "immutability_set", "at": now(), "storage": rel(stored)},
        ],
        "duplicate_hash": duplicate_of is not None,
        "duplicate_of": duplicate_of,
        "truth_promotion": False,
        "canonical_graph_materialization": False,
        "canonical_graph_writes_performed": False,
    }
    proof.update(extra)
    return proof


def finish_ingest(args: argparse.Namespace, proof: dict[str, Any]) -> int:
    root = storage_root(args)
    inserted = append_index(root, proof)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "action": "ingest",
        "proof_id": proof["proof_id"],
        "proof_object": proof,
        "index_inserted": inserted,
        "storage_root": rel(root),
        "blockers": [],
        "truth_promotion": False,
        "canonical_graph_materialization": False,
        "canonical_graph_writes_performed": False,
        "verdict": "PASS",
    }
    write_receipt(args, receipt)
    return 0


def file_ingest(args: argparse.Namespace) -> int:
    root = storage_root(args)
    source = Path(args.source)
    if not source.is_absolute():
        source = (ROOT / source).resolve() if (ROOT / source).exists() else source.resolve()
    try:
        digest, size, stored, existed = store_file(root, source)
    except Exception as exc:
        payload = {"schema": RECEIPT_SCHEMA, "action": "file_ingest", "proof_id": None, "blockers": [str(exc)], "verdict": "FAIL", "canonical_graph_writes_performed": False}
        write_receipt(args, payload)
        return 4
    st = source.stat()
    proof = build_proof(root, source_type="file", original=str(source), stored=stored, digest=digest, size=size, extractor_status="stored", extra={"source_mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"), "storage_existed": existed})
    return finish_ingest(args, proof)


def text_ingest(args: argparse.Namespace) -> int:
    root = storage_root(args)
    if args.text_file:
        source = Path(args.text_file)
        if not source.is_absolute():
            source = (ROOT / source).resolve() if (ROOT / source).exists() else source.resolve()
        try:
            data = source.read_bytes()
        except Exception as exc:
            payload = {"schema": RECEIPT_SCHEMA, "action": "text_ingest", "proof_id": None, "blockers": [f"text_source_unreadable:{exc}"], "verdict": "FAIL", "canonical_graph_writes_performed": False}
            write_receipt(args, payload)
            return 4
        original = str(source)
    else:
        data = str(args.text).encode("utf-8")
        original = "operator_text"
    digest, size, stored, existed = store_bytes(root, data, ".txt")
    proof = build_proof(root, source_type="text_note", original=original, stored=stored, digest=digest, size=size, extractor_status="stored_exact_text", extra={"operator_id": args.operator_id, "session_id": args.session_id, "storage_existed": existed})
    return finish_ingest(args, proof)


def read_arg_bytes(value: str | None, file_value: str | None) -> tuple[bytes, str | None]:
    if file_value:
        p = Path(file_value)
        if not p.is_absolute():
            p = (ROOT / p).resolve() if (ROOT / p).exists() else p.resolve()
        return p.read_bytes(), str(p)
    return (value or "").encode("utf-8"), None


def command_output_ingest(args: argparse.Namespace) -> int:
    root = storage_root(args)
    try:
        stdout_bytes, stdout_source = read_arg_bytes(args.stdout, args.stdout_file)
        stderr_bytes, stderr_source = read_arg_bytes(args.stderr, args.stderr_file)
    except Exception as exc:
        payload = {"schema": RECEIPT_SCHEMA, "action": "command_output_ingest", "proof_id": None, "blockers": [f"command_output_unreadable:{exc}"], "verdict": "FAIL", "canonical_graph_writes_performed": False}
        write_receipt(args, payload)
        return 4
    stdout_hash, stdout_size, stdout_stored, _ = store_bytes(root, stdout_bytes, ".stdout")
    stderr_hash, stderr_size, stderr_stored, _ = store_bytes(root, stderr_bytes, ".stderr")
    command: Any
    if args.command_json:
        command = json.loads(args.command_json)
    else:
        command = args.command
    manifest = {
        "command": command,
        "cwd": args.cwd,
        "return_code": args.return_code,
        "stdout_sha256": stdout_hash,
        "stdout_size": stdout_size,
        "stdout_stored_artifact_path": rel(stdout_stored),
        "stderr_sha256": stderr_hash,
        "stderr_size": stderr_size,
        "stderr_stored_artifact_path": rel(stderr_stored),
        "stdout_source": stdout_source,
        "stderr_source": stderr_source,
        "redaction_status": args.redaction_status,
        "rerun_performed": False,
    }
    digest, size, stored, existed = store_bytes(root, canonical_json(manifest), ".command.json")
    proof = build_proof(root, source_type="command_output", original="command_output_manifest", stored=stored, digest=digest, size=size, extractor_status="stored_command_output_without_rerun", extra={"command_output": manifest, "storage_existed": existed})
    return finish_ingest(args, proof)


def diff_ingest(args: argparse.Namespace) -> int:
    root = storage_root(args)
    if args.patch_file:
        p = Path(args.patch_file)
        if not p.is_absolute():
            p = (ROOT / p).resolve() if (ROOT / p).exists() else p.resolve()
        try:
            patch_text = p.read_text(encoding="utf-8")
        except Exception as exc:
            payload = {"schema": RECEIPT_SCHEMA, "action": "diff_ingest", "proof_id": None, "blockers": [f"patch_unreadable:{exc}"], "verdict": "FAIL", "canonical_graph_writes_performed": False}
            write_receipt(args, payload)
            return 4
        original = str(p)
    else:
        patch_text = args.patch_text or ""
        original = "operator_patch_text"
    manifest = {
        "patch_text": patch_text,
        "changed_files": args.changed_file or [],
        "base_identifier": args.base_id,
        "head_identifier": args.head_id,
        "diff_sha256": sha256_bytes(patch_text.encode("utf-8")),
    }
    digest, size, stored, existed = store_bytes(root, canonical_json(manifest), ".diff.json")
    proof = build_proof(root, source_type="code_diff", original=original, stored=stored, digest=digest, size=size, extractor_status="stored_patch_text", extra={"code_diff": manifest, "storage_existed": existed})
    return finish_ingest(args, proof)


def lookup(args: argparse.Namespace) -> int:
    root = storage_root(args)
    rows = iter_index(root)
    matches = []
    if args.proof_id:
        matches = [r for r in rows if r.get("proof_id") == args.proof_id]
    elif args.sha256:
        matches = [r for r in rows if r.get("sha256") == args.sha256]
    payload = {
        "schema": RECEIPT_SCHEMA,
        "action": "lookup",
        "proof_id": args.proof_id,
        "sha256": args.sha256,
        "matches": matches,
        "blockers": [] if matches else ["proof_not_found"],
        "truth_promotion": False,
        "canonical_graph_materialization": False,
        "canonical_graph_writes_performed": False,
        "verdict": "PASS" if matches else "FAIL",
    }
    write_receipt(args, payload)
    return 0 if matches else 4


def main() -> int:
    p = argparse.ArgumentParser(description="Proof Kernel v1 byte-custody CLI")
    p.add_argument("--storage-root")
    p.add_argument("--output-dir")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("file-ingest")
    sp.add_argument("--source", required=True)
    sp.set_defaults(func=file_ingest)

    sp = sub.add_parser("text-ingest")
    g = sp.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--text-file")
    sp.add_argument("--operator-id", default="operator")
    sp.add_argument("--session-id", default="local-session")
    sp.set_defaults(func=text_ingest)

    sp = sub.add_parser("command-output-ingest")
    cg = sp.add_mutually_exclusive_group(required=True)
    cg.add_argument("--command")
    cg.add_argument("--command-json")
    sp.add_argument("--cwd", required=True)
    sp.add_argument("--return-code", type=int, required=True)
    sp.add_argument("--stdout", default="")
    sp.add_argument("--stdout-file")
    sp.add_argument("--stderr", default="")
    sp.add_argument("--stderr-file")
    sp.add_argument("--redaction-status", choices=["none", "redacted", "unknown"], default="none")
    sp.set_defaults(func=command_output_ingest)

    sp = sub.add_parser("diff-ingest")
    dg = sp.add_mutually_exclusive_group(required=True)
    dg.add_argument("--patch-text")
    dg.add_argument("--patch-file")
    sp.add_argument("--changed-file", action="append", default=[])
    sp.add_argument("--base-id")
    sp.add_argument("--head-id")
    sp.set_defaults(func=diff_ingest)

    sp = sub.add_parser("lookup")
    lg = sp.add_mutually_exclusive_group(required=True)
    lg.add_argument("--proof-id")
    lg.add_argument("--sha256")
    sp.set_defaults(func=lookup)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
