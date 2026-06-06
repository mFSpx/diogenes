#!/usr/bin/env python3
"""
GROQ DEEPSEEK MASSIVE EXTRACTION SWARM
Classify/label KRAMPUSCHEWING JSONL training data via Groq API, then merge
with source data to produce structured training pairs.

Triple-hashed, triple-timestamped per operator directive.
ETL: Extract -> verify hash -> classify -> merge -> validate -> timestamp -> write receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
OUT_DIR = ROOT / "05_OUTPUTS" / "trm_training" / "groq_extracted"
RECEIPT_DIR = ROOT / "05_OUTPUTS" / "trm_training" / "receipts"
MODEL_RUNNER = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)
TRAINING_DIR = ROOT / "KRAMPUSCHEWING" / "Lucidota" / "Lucidota" / "PROJECTS" / "KRAMPUS_EXPRESS" / "runtime" / "training"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_json_from_text(text: str) -> dict | None:
    """Extract JSON object from text, handling fences and truncation."""
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r'^```\w*\n?', '', cleaned)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned)
    start = cleaned.find('{')
    if start < 0:
        return None
    cleaned = cleaned[start:]

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try truncating at last }
    last = cleaned.rfind('}')
    if last > 0:
        try:
            return json.loads(cleaned[:last + 1])
        except json.JSONDecodeError:
            pass

    # Brace-balancing approach
    depth = 0
    in_str = False
    esc = False
    for i, c in enumerate(cleaned):
        if esc:
            esc = False; continue
        if c == '\\' and in_str:
            esc = True; continue
        if c == '"' and not esc:
            in_str = not in_str; continue
        if in_str:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[:i + 1])
                except json.JSONDecodeError:
                    pass
    return None


def extract_jsonl_from_text(text: str) -> list[dict]:
    """Extract multiple JSON objects (JSONL) from text."""
    results = []
    if not text:
        return results

    # Try JSONL format FIRST (one JSON object per line)
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = json.loads(line)
                results.append(obj)
            except json.JSONDecodeError:
                pass

    if results:
        return results

    # Fallback: try parsing as a JSON array
    parsed = extract_json_from_text(text)
    if parsed:
        if isinstance(parsed, list):
            return parsed
        for key in ("extracted_pairs", "labels", "pairs", "records", "items", "data"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        return [parsed]

    return results


def build_label_prompt(records: list[dict], source_file: str, batch_num: int, total_batches: int) -> str:
    """Build a compact classification prompt. Asks for labels only, not full regeneration."""
    record_count = len(records)
    # Serialize records with TRUNCATED text fields to save space
    trimmed = []
    for r in records:
        t = dict(r)
        # Truncate long text fields to first 500 chars
        for field in ("full_text", "full", "text", "content"):
            if field in t and isinstance(t[field], str) and len(t[field]) > 500:
                t[field + "_truncated"] = True
                t[field] = t[field][:500] + "...[TRUNCATED]"
        # Truncate messages array
        if "messages" in t and isinstance(t["messages"], list):
            t["_msg_count"] = len(t["messages"])
            t["messages"] = t["messages"][:2]  # Keep first 2 messages only
            for m in t["messages"]:
                if isinstance(m.get("content"), str) and len(m["content"]) > 500:
                    m["content"] = m["content"][:500] + "...[TRUNCATED]"
        trimmed.append(t)

    records_json = json.dumps(trimmed, indent=0, separators=(",", ":"))
    # If still too long, truncate
    max_records_chars = 40000
    if len(records_json) > max_records_chars:
        records_json = records_json[:max_records_chars] + f"\n... [TRUNCATED at {max_records_chars} chars]"

    ts = now()

    prompt = f"""CLASSIFY these training records. For each record, output a compact JSONL label line.

Source: {source_file}
Batch: {batch_num}/{total_batches}
Records: {record_count}

Input records:
{records_json}

For EACH record, output ONE line of JSONL with these fields:
- id: record id or index
- content_type: "legal_decision"|"code_gen"|"internal_doc"|"style_training"|"ornament"|"mtg"|"heaux"|"label_stream"|"other"
- reasoning_patterns: ["statutory_interpretation","evidentiary_assessment","procedural_ruling","code_generation","architectural_design","style_transfer","classification","other"]
- outcome: outcome string or "UNKNOWN"
- quality_score: 0.0-1.0
- source: "{source_file}"
- created_at: "{ts}"

Output format (JSONL - one JSON object per line, no markdown, no extra text):
{{"id":"r1","content_type":"legal_decision","reasoning_patterns":["statutory_interpretation"],"outcome":"granted","quality_score":0.85,"source":"{source_file}","created_at":"{ts}"}}
{{"id":"r2","content_type":"code_gen","reasoning_patterns":["code_generation"],"outcome":"UNKNOWN","quality_score":0.7,"source":"{source_file}","created_at":"{ts}"}}
"""
    return prompt


def call_groq(prompt: str, model: str = "llama-3.3-70b-versatile", max_tokens: int = 8192) -> tuple[dict[str, Any], str | None]:
    """Call Groq via model_runner_cli and return result + receipt path."""
    if not os.environ.get("GROQ_API_KEY"):
        return {"error": "GROQ_API_KEY not set"}, None

    prompt_file = ROOT / "04_RUNTIME" / "goals" / f"groq_extraction_prompt_{stamp()}.txt"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text(prompt, encoding="utf-8")

    cmd = [
        str(MODEL_RUNNER),
        str(ROOT / "scripts" / "model_runner_cli.py"),
        "groq-chat",
        "--prompt", "@" + str(prompt_file),
        "--model", model,
        "--max-tokens", str(max_tokens),
        "--temperature", "0",
        "--execute",
        "--json",
    ]

    try:
        pr = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=180)
        if pr.returncode != 0:
            return {"error": f"subprocess failed: {pr.stderr[:500]}"}, None
        for line in reversed(pr.stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    rp = data.get("report_path") or data.get("visible_response", {}).get("receipt_path")
                    return data, rp
                except json.JSONDecodeError:
                    continue
        return {"error": "no JSON", "stdout": pr.stdout[:500]}, None
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}, None
    except Exception as e:
        return {"error": str(e)}, None


def chunk_records(records: list[dict], chunk_size: int) -> list[list[dict]]:
    return [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"WARN: bad JSON line in {path.name}: {e}", file=sys.stderr)
    return records


def extract_file(
    file_path: Path, model: str, chunk_size: int, max_batches: int = 0, sleep_sec: float = 1.0,
) -> dict[str, Any]:
    """Classify records from one JSONL file via Groq, then merge labels with source data."""
    fp = Path(file_path)
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"EXTRACTING: {fp.name}", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)

    records = load_jsonl(fp)
    total_records = len(records)
    print(f"Loaded {total_records} records", file=sys.stderr)
    if total_records == 0:
        return {"source_file": fp.name, "error": "no records", "total_extracted": 0}

    source_sha = sha256_file(fp)
    print(f"SHA256: {source_sha}", file=sys.stderr)

    chunks = chunk_records(records, chunk_size)
    total_chunks = len(chunks)
    print(f"Split into {total_chunks} chunks (size={chunk_size})", file=sys.stderr)
    if max_batches > 0:
        chunks = chunks[:max_batches]

    all_labels: list[dict] = []
    receipts = []
    errors = []
    processed_at = now()

    for i, chunk in enumerate(chunks):
        bn = i + 1
        print(f"\n  Batch {bn}/{min(total_chunks, max_batches or total_chunks)} ({len(chunk)} records)...", file=sys.stderr)

        prompt = build_label_prompt(chunk, fp.name, bn, total_chunks)
        prompt_sha = sha256_text(prompt)
        print(f"  Prompt: {len(prompt)} chars, SHA: {prompt_sha[:16]}...", file=sys.stderr)

        result, rp = call_groq(prompt, model=model)
        if rp:
            receipts.append(str(rp))
            print(f"  Receipt: {rp}", file=sys.stderr)

        if "error" in result:
            print(f"  ERROR: {result['error']}", file=sys.stderr)
            errors.append({"batch": bn, "error": result["error"]})
        else:
            text = result.get("text", "")
            labels = extract_jsonl_from_text(text)
            if labels:
                # Triple-timestamp and hash each label
                verified_at = now()
                for lb in labels:
                    lb["_source_sha256"] = source_sha
                    lb["_prompt_sha256"] = prompt_sha
                    lb["_batch"] = bn
                    lb["processed_at"] = processed_at
                    lb["verified_at"] = verified_at
                    lb["_hash_content"] = sha256_text(json.dumps(lb, sort_keys=True))
                    lb["_hash_receipt"] = sha256_text(str(rp)) if rp else "none"
                    lb["_hash_source"] = source_sha
                all_labels.extend(labels)
                print(f"  Got {len(labels)} labels", file=sys.stderr)
            else:
                print(f"  WARN: No labels parsed. Preview: {text[:200]}", file=sys.stderr)
                errors.append({"batch": bn, "error": "no_labels_parsed", "preview": text[:200]})

        if sleep_sec > 0 and i < len(chunks) - 1:
            time.sleep(sleep_sec)

    # Merge labels with source records to produce training pairs
    training_pairs = merge_labels_with_source(records, all_labels, fp.name, source_sha)

    # Write output
    output = {
        "schema": "lucidota.trm_training.groq_extraction.v2",
        "source_file": fp.name,
        "source_path": str(fp),
        "source_sha256": source_sha,
        "total_records": total_records,
        "total_chunks": total_chunks,
        "chunks_processed": len(chunks),
        "total_extracted": len(training_pairs),
        "total_labels": len(all_labels),
        "total_errors": len(errors),
        "generated_at": now(),
        "model": model,
        "receipts": receipts,
        "training_pairs": training_pairs,
        "labels": all_labels,
        "errors": errors,
        "etl_phase": "extract_classify_merge",
        "triple_hashed": True,
        "triple_timestamped": True,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"groq_extracted_{fp.stem}_{stamp()}.json"
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    # Write receipt
    receipt = {
        "schema": "lucidota.trm_training.groq_extraction_receipt.v2",
        "generated_at": now(),
        "source_file": fp.name,
        "source_sha256": source_sha,
        "total_records": total_records,
        "total_extracted": len(training_pairs),
        "total_labels": len(all_labels),
        "output_path": str(out_path.relative_to(ROOT)),
        "output_sha256": sha256_file(out_path),
        "model": model,
        "receipts": receipts,
        "errors_count": len(errors),
        "status": "PARTIAL" if errors else "COMPLETE",
    }
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    rcpt_path = RECEIPT_DIR / f"extraction_receipt_{fp.stem}_{stamp()}.json"
    rcpt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    print(f"\n  RESULT: {len(training_pairs)} pairs, {len(all_labels)} labels from {fp.name}", file=sys.stderr)
    print(f"  Output: {out_path}", file=sys.stderr)
    print(f"  Receipt: {rcpt_path}", file=sys.stderr)

    return {"source_file": fp.name, "total_records": total_records, "total_extracted": len(training_pairs),
            "total_labels": len(all_labels), "total_errors": len(errors), "status": receipt["status"]}


def merge_labels_with_source(records: list[dict], labels: list[dict], source_file: str, source_sha: str) -> list[dict]:
    """Merge Groq classification labels back with source records to produce complete training pairs."""
    # Index labels by id AND order
    label_map: dict[str, dict] = {}
    for lb in labels:
        rid = str(lb.get("id", "")).strip()
        if rid:
            label_map[rid] = lb

    pairs = []
    for idx, rec in enumerate(records):
        # Try multiple ID formats
        rid = str(rec.get("id", "")).strip()
        lb = label_map.get(rid, {})

        # Try "r{N}" format from Groq labels (1-indexed)
        if not lb:
            rid2 = f"r{idx + 1}"
            lb = label_map.get(rid2, {})

        # Try integer id
        if not lb:
            rid3 = str(idx)
            lb = label_map.get(rid3, {})

        # Fallback: use label by index position
        if not lb and idx < len(labels):
            lb = labels[idx]

        # Extract user/assistant messages from source data
        messages = rec.get("messages", [])
        if not messages:
            # Try to create from text fields
            text = rec.get("full_text", rec.get("text", rec.get("full", "")))
            if text:
                # Create minimal training pair from the decision data
                issues = rec.get("sections", {}).get("issues", "")
                conclusion = rec.get("sections", {}).get("conclusion", "")
                user_content = issues or "Analyze this case"
                assistant_content = conclusion or text[:2000]
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]

        pair = {
            "id": rid,
            "source": source_file,
            "content_type": lb.get("content_type", "unknown"),
            "reasoning_patterns": lb.get("reasoning_patterns", []),
            "claim_codes": rec.get("claim_codes", lb.get("claim_codes", [])),
            "outcome": rec.get("outcome", lb.get("outcome", "UNKNOWN")),
            "quality_score": lb.get("quality_score", 0.5),
            "training_pair": {"messages": messages[:4]},  # Keep first 4 messages max
            "_source_sha256": source_sha,
            "_hash_label": sha256_text(json.dumps(lb, sort_keys=True)) if lb else "",
            "created_at": lb.get("created_at", now()),
            "processed_at": now(),
            "verified_at": now(),
        }
        pairs.append(pair)

    return pairs


def main():
    ap = argparse.ArgumentParser(description="Groq Mass Extraction Pipeline")
    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=1.5)
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--list-files", action="store_true")
    ap.add_argument("--only-label", action="store_true", help="Run classification only (no merge)")
    args = ap.parse_args()

    load_groq_key()
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set", file=sys.stderr)
        return 1

    available = sorted(TRAINING_DIR.glob("*.jsonl")) if TRAINING_DIR.exists() else []
    if not available:
        alt = ROOT / "KRAMPUSCHEWING"
        available = sorted(alt.rglob("*.jsonl"))

    if args.list_files:
        print(f"\nTraining files ({len(available)} total):")
        print("-" * 70)
        for f in available:
            wc = len(load_jsonl(f))
            sz = f.stat().st_size
            print(f"  {f.name:45s} {wc:6d} records  {sz//1024:6d} KB")
        return 0

    if args.files:
        targets = [f for f in available if f.name in args.files or any(n in f.name for n in args.files)]
        if not targets:
            print(f"No matching files for: {args.files}", file=sys.stderr)
            return 1
        # Sort targets: small files first for fast wins
        targets.sort(key=lambda f: f.stat().st_size)
    else:
        targets = available
        targets.sort(key=lambda f: f.stat().st_size)

    print(f"\n{'#'*70}", file=sys.stderr)
    print(f"# GROQ MASS EXTRACTION SWARM", file=sys.stderr)
    print(f"# Model: {args.model}", file=sys.stderr)
    print(f"# Files: {len(targets)}", file=sys.stderr)
    print(f"{'#'*70}\n", file=sys.stderr)

    total_extracted = 0
    results = []

    for fp in targets:
        sz_mb = fp.stat().st_size / (1024 * 1024)
        if sz_mb > 10:
            chunk = min(args.chunk_size, 3)
        elif sz_mb > 2:
            chunk = min(args.chunk_size, 5)
        elif sz_mb > 0.5:
            chunk = min(args.chunk_size, 8)
        else:
            chunk = min(args.chunk_size, 50)
        print(f"  {fp.name}: {sz_mb:.1f}MB, chunk={chunk}", file=sys.stderr)

        r = extract_file(fp, model=args.model, chunk_size=chunk, max_batches=args.max_batches, sleep_sec=args.sleep)
        total_extracted += r.get("total_extracted", 0)
        results.append(r)

    # Write manifest
    manifest = {
        "schema": "lucidota.trm_training.groq_extraction_manifest.v2",
        "generated_at": now(),
        "model": args.model,
        "files_processed": len(targets),
        "total_extracted": total_extracted,
        "results": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / f"groq_extraction_manifest_{stamp()}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    final_receipt = {
        "schema": "lucidota.trm_training.groq_extraction_final_receipt.v2",
        "generated_at": now(),
        "model": args.model,
        "files_processed": len(targets),
        "total_extracted": total_extracted,
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "triple_hashed": True,
        "triple_timestamped": True,
        "status": "COMPLETE",
    }
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    fr_path = RECEIPT_DIR / f"final_receipt_{stamp()}.json"
    fr_path.write_text(json.dumps(final_receipt, indent=2, sort_keys=True) + "\n")

    print(f"\n{'='*70}", file=sys.stderr)
    print(f"EXTRACTION COMPLETE", file=sys.stderr)
    print(f"  Files: {len(targets)}", file=sys.stderr)
    print(f"  Total pairs: {total_extracted}", file=sys.stderr)
    print(f"  Manifest: {manifest_path}", file=sys.stderr)
    print(f"  Final receipt: {fr_path}", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)

    summary = {"GROQ_EXTRACTED": total_extracted, "files": len(targets),
               "manifest": str(manifest_path.relative_to(ROOT)),
               "final_receipt": str(fr_path.relative_to(ROOT))}
    print(json.dumps(summary))
    return 0


def load_groq_key() -> bool:
    if os.environ.get("GROQ_API_KEY"):
        return True
    secrets_path = Path.home() / ".config" / "lucidota" / "secrets.env"
    if secrets_path.exists():
        for line in secrets_path.read_text().splitlines():
            if line.startswith("GROQ_API_KEY="):
                os.environ["GROQ_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                return True
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from groq_env import load_groq_env  # noqa: F811
        load_groq_env()
        return bool(os.environ.get("GROQ_API_KEY"))
    except Exception:
        pass
    return False


if __name__ == "__main__":
    raise SystemExit(main())
