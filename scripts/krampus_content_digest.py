#!/usr/bin/env python3
"""Krampus Content Digest Pipeline — turn raw RiverML extraction into per-file and per-repo digests.

Reads the RiverML JSONL extraction output, repo manifest files, and produces:
  - Global TF-IDF across all extracted chunks (not per-chunk)
  - Per-file content signatures (SHA256 + aggregated RiverML features)
  - Per-repo content signatures (SHA256 of file list + mean feature vector)
  - Embedding manifests for hot rail (Needle 26M / TRM 7M)
  - Postgres receipt facts

Usage:
  python3 scripts/krampus_content_digest.py [--execute] [--digest-only]

Mutation class: receipt_only
Receipt scope: LOCAL_FILE_PRODUCT, ABSURD_POSTGRES_RUNTIME
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psycopg2 as psycopg
    from psycopg2.extras import DictCursor as _DC
    dict_row = None  # type: ignore
    # Monkey-patch psycopg2 to have a row_factory interface similar to psycopg v3
    # We use DictCursor directly in the connection code below.
except ImportError:
    psycopg = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from ALGOS.shannon_entropy import shannon_entropy
except ImportError:
    def shannon_entropy(data: list) -> float:  # type: ignore
        if not data:
            return 0.0
        from collections import Counter
        c = Counter(data)
        total = len(data)
        return -sum((count / total) * math.log2(count / total) for count in c.values())

# --- Constants ---

RIVER_JSONL = ROOT / "05_OUTPUTS" / "brag" / "odysseus_full_code_river.jsonl"
REPO_MANIFEST_DIRS = [
    ROOT / "05_OUTPUTS" / "repo_extracts",
    ROOT / "KRAMPUSCHEWING" / "05_OUTPUTS" / "repo_extracts",
]
DIGEST_OUT = ROOT / "05_OUTPUTS" / "digest"
RECEIPT_SUBSYSTEM = "krampus_content_digest"
FEATURE_VERSION = "krampus_digest_v1"
DEFAULT_DB = "postgresql:///lucidota_state"

REPO_MANIFEST_GLOB = "*.json"

# Regex for tokenizing TF-IDF terms: alphanumeric sequences, respecting code identifiers
TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{1,63}")


# --- Helpers ---

def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def deterministic_uuid(namespace: str, value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"lucidota:{namespace}:{value}"))


def sha256_digest(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


# --- Read RiverML JSONL ---

def read_river_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all records from the RiverML JSONL extract file."""
    records = []
    if not path.exists():
        print(f"WARN: RiverML JSONL not found at {path}", file=sys.stderr)
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"READ {len(records)} records from {rel(path)}")
    return records


# --- Global TF-IDF Computation ---

def build_global_vocabulary(records: list[dict[str, Any]], min_df: int = 2, max_df_ratio: float = 0.85) -> dict[str, int]:
    """Build a global vocabulary from all chunk texts, filtering rare and very common terms.

    Returns a mapping of term -> index.
    """
    doc_freq: Counter[str] = Counter()
    token_counts: list[Counter[str]] = []
    n_docs = len(records)

    for rec in records:
        text = rec.get("text", "")
        tokens = TOKEN_RE.findall(text.lower())
        unique = set(tokens)
        for t in unique:
            doc_freq[t] += 1
        token_counts.append(Counter(tokens))

    # Filter: keep terms that appear in at least min_df docs and at most max_df_ratio docs
    max_docs = int(n_docs * max_df_ratio)
    vocab: dict[str, int] = {}
    for term, df in doc_freq.most_common():
        if df < min_df:
            continue
        if df > max_docs:
            continue
        if term not in vocab:
            vocab[term] = len(vocab)

    print(f"VOCAB size: {len(vocab)} terms (from {len(doc_freq)} unique, filtered min_df={min_df} max_df_ratio={max_df_ratio})")
    return vocab


def compute_global_tfidf(
    records: list[dict[str, Any]],
    vocab: dict[str, int],
    top_k: int = 50,
) -> list[dict[str, Any]]:
    """Compute global TF-IDF vectors for each record.

    Returns records augmented with 'global_tfidf' key: dict of term->score for top_k terms.
    """
    n_docs = len(records)
    doc_freq: Counter[str] = Counter()
    tokenized: list[Counter[str]] = []

    for rec in records:
        text = rec.get("text", "")
        tokens = TOKEN_RE.findall(text.lower())
        unique = set(tokens) & set(vocab.keys())
        for t in unique:
            doc_freq[t] += 1
        tokenized.append(Counter(tokens))

    augmented: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        tf = tokenized[i]
        # Compute TF-IDF for vocab terms present in this doc
        scores: dict[str, float] = {}
        max_tf = max(tf.values()) if tf else 1
        for term, count in tf.items():
            if term not in vocab:
                continue
            tf_val = count / max_tf  # normalized term frequency
            df = doc_freq.get(term, 1)
            idf = math.log((n_docs + 1) / (df + 1)) + 1  # smooth IDF
            scores[term] = round(tf_val * idf, 6)

        # Keep top_k terms by score
        top_terms = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        new_rec = dict(rec)
        new_rec["global_tfidf"] = dict(top_terms)
        augmented.append(new_rec)

    print(f"COMPUTED global TF-IDF for {len(augmented)} records (top_k={top_k})")
    return augmented


# --- Per-File Aggregation ---

def aggregate_by_file(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group records by source file and compute per-file digest.

    Returns dict of source_path -> file_digest.
    """
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        src = rec.get("source", "unknown")
        by_file[src].append(rec)

    file_digests: dict[str, dict[str, Any]] = {}
    for src, chunks in by_file.items():
        # Concatenate all chunk texts
        full_text = "\n".join(c.get("text", "") for c in chunks)

        # Aggregate structural features (average over chunks)
        agg_features: dict[str, float] = Counter()
        feature_count = 0
        for c in chunks:
            rf = c.get("riverml_features", {})
            if isinstance(rf, str):
                rf = json.loads(rf)
            for k, v in rf.items():
                if isinstance(v, (int, float)) and not k.startswith("subsys_") and not k.startswith("type_"):
                    agg_features[k] += float(v)
            feature_count += 1

        avg_features = {}
        if feature_count > 0:
            for k, v in agg_features.items():
                avg_features[k] = round(v / feature_count, 4)

        # Get SHA256 of source content (from last available chunk's sha256, or compute)
        source_sha256 = chunks[-1].get("sha256", "")
        if not source_sha256:
            source_sha256 = sha256_digest(full_text)

        # Merge global TF-IDF across chunks (take top terms across all chunks)
        merged_tfidf: dict[str, float] = {}
        for c in chunks:
            gt = c.get("global_tfidf", {})
            for term, score in gt.items():
                if term not in merged_tfidf or score > merged_tfidf[term]:
                    merged_tfidf[term] = score
        top_merged = dict(sorted(merged_tfidf.items(), key=lambda x: -x[1])[:100])

        # Subsystem from chunks
        subsystems = Counter()
        types = Counter()
        for c in chunks:
            rf = c.get("riverml_features", {})
            if isinstance(rf, str):
                rf = json.loads(rf)
            for k, v in rf.items():
                if k.startswith("subsys_") and v:
                    subsystems[k.replace("subsys_", "")] += float(v) if isinstance(v, (int, float)) else 1
                if k.startswith("type_") and v:
                    types[k.replace("type_", "")] += float(v) if isinstance(v, (int, float)) else 1

        dominant_subsystem = subsystems.most_common(1)[0][0] if subsystems else "unknown"

        file_digests[src] = {
            "source": src,
            "chunk_count": len(chunks),
            "total_chars": sum(len(c.get("text", "")) for c in chunks),
            "source_sha256": source_sha256,
            "file_sha256": sha256_digest(full_text),
            "dominant_subsystem": dominant_subsystem,
            "subsystem_distribution": dict(subsystems.most_common()),
            "type_distribution": dict(types.most_common()),
            "avg_features": avg_features,
            "top_terms": top_merged,
            "term_count": len(merged_tfidf),
            "chunks": [{"chunk_id": c.get("chunk_id"), "sha256": c.get("sha256")} for c in chunks],
        }

    print(f"AGGREGATED {len(file_digests)} files")
    return file_digests


# --- Read Repo Manifests ---

def read_repo_manifests() -> list[dict[str, Any]]:
    """Read all repo manifest JSON files."""
    manifests = []
    for d in REPO_MANIFEST_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.glob(REPO_MANIFEST_GLOB)):
            data = json.loads(p.read_text(encoding="utf-8"))
            manifests.append({"filename": p.name, "path": str(p), "data": data})
    print(f"READ {len(manifests)} repo manifest files")
    return manifests


def extract_repos_from_manifests(manifests: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Extract repo metadata from manifest files, returning a dict keyed by repo name."""
    repos: dict[str, dict[str, Any]] = {}
    for m in manifests:
        data = m["data"]
        # Handle different manifest formats
        if "repos" in data and isinstance(data["repos"], dict):
            # Format: { "repos": { "repo_name": { "total_files": N, ... } } }
            for name, info in data["repos"].items():
                repos[name] = {
                    "repo_name": name,
                    "manifest_source": m["filename"],
                    "total_files": info.get("total_files", 0),
                    "source_files": info.get("source_files", info.get("total_files", 0)),
                    "disk_gb": info.get("disk_gb", 0),
                    "lang": info.get("lang", "unknown"),
                }
        elif "repo" in data:
            # Format: { "repo": "name", "source_root": "...", "total_files": N, ... }
            name = data["repo"]
            repos[name] = {
                "repo_name": name,
                "manifest_source": m["filename"],
                "total_files": data.get("total_files", 0),
                "source_files": data.get("source_files", data.get("total_files", 0)),
                "disk_gb": round(data.get("total_bytes", 0) / (1024**3), 3),
                "lang": "mixed",
                "source_root": data.get("source_root", ""),
                "total_bytes": data.get("total_bytes", 0),
                "elapsed_seconds": data.get("elapsed_seconds", 0),
            }
    return repos


# --- Per-Repo Digest ---

def compute_repo_digests(
    file_digests: dict[str, dict[str, Any]],
    repo_metadata: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute per-repo content signatures combining file digests and repo metadata.

    Returns dict of repo_name -> repo_digest.
    """
    # Map source files to repos based on manifest metadata and source path patterns
    # First, build a reverse mapping from known repos
    repo_files: dict[str, list[dict[str, Any]]] = defaultdict(list)
    repo_metadata_by_name = dict(repo_metadata)

    # Try to assign files to repos based on manifests
    # The RiverML source paths are filenames within the Odysseus project
    # Additional repos come from manifests

    # Collect files that belong to the primary (Odysseus) repo
    if records:
        # Infer the primary repo name from the source paths
        source_samples = set()
        for rec in records[:100]:
            s = rec.get("source", "")
            if s:
                source_samples.add(s.split("/")[0] if "/" in s else s)

        primary_repo_name = "odysseus"  # default
        # Check if one of the manifests has "odysseus" or match
        for rname in repo_metadata_by_name:
            if rname.lower().replace("_", "").replace("-", "") in ("odysseus", "odysseusai"):
                primary_repo_name = rname
                break

        # Assign all RiverML records to primary repo
        for src, fd in file_digests.items():
            repo_files[primary_repo_name].append(fd)

    # Assign metadata-only repos (no file digests from RiverML)
    for rname, rmeta in repo_metadata_by_name.items():
        if rname not in repo_files:
            repo_files[rname] = []

    repo_digests: dict[str, dict[str, Any]] = {}
    for repo_name, files in repo_files.items():
        # Compute repo-level SHA256 from all file SHA256 hashes
        file_hash_string = "".join(
            sorted(f.get("file_sha256", f.get("source_sha256", "")) for f in files)
        )
        repo_sha256 = sha256_digest(file_hash_string) if file_hash_string else ""

        # Aggregate structural features across all files
        agg_avg_features: dict[str, float] = Counter()
        feature_count = 0
        for fd in files:
            af = fd.get("avg_features", {})
            for k, v in af.items():
                if isinstance(v, (int, float)):
                    agg_avg_features[k] += v
            feature_count += 1

        repo_avg_features: dict[str, float] = {}
        if feature_count > 0:
            for k, v in agg_avg_features.items():
                repo_avg_features[k] = round(v / feature_count, 4)

        # Aggregate top terms across all files in repo
        repo_term_scores: dict[str, float] = Counter()
        for fd in files:
            for term, score in fd.get("top_terms", {}).items():
                repo_term_scores[term] += score

        repo_top_terms = dict(sorted(repo_term_scores.items(), key=lambda x: -x[1])[:100])

        # Subsystem distribution
        subsys_dist: Counter[str] = Counter()
        for fd in files:
            sd = fd.get("subsystem_distribution", {})
            for subsys, count in sd.items():
                subsys_dist[subsys] += count

        # Merge with metadata
        meta = repo_metadata_by_name.get(repo_name, {})
        total_files = len(files)
        metadata_file_count = meta.get("total_files", 0)

        repo_digests[repo_name] = {
            "repo_name": repo_name,
            "repo_sha256": repo_sha256,
            "digested_file_count": total_files,
            "manifest_file_count": metadata_file_count,
            "manifest_source": meta.get("manifest_source", "none"),
            "suggested_language": meta.get("lang", "mixed"),
            "disk_gb": meta.get("disk_gb", 0),
            "file_count": max(total_files, metadata_file_count),
            "avg_structural_features": repo_avg_features,
            "top_terms": repo_top_terms,
            "unique_term_count": len(repo_term_scores),
            "subsystem_distribution": dict(subsys_dist.most_common()),
            "dominant_subsystem": subsys_dist.most_common(1)[0][0] if subsys_dist else "unknown",
            "feature_version": FEATURE_VERSION,
        }

    return repo_digests


# --- Embedding Manifest Generation ---

def generate_embedding_manifest(
    file_digests: dict[str, dict[str, Any]],
    repo_digests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Generate an embedding manifest for hot rail ingestion (Needle 26M / TRM 7M).

    Each entry is a document-level record with content text ready for embedding.
    """
    entries: list[dict[str, Any]] = []
    for src, fd in file_digests.items():
        terms_list = sorted(fd.get("top_terms", {}).keys(), key=lambda t: -fd["top_terms"][t])
        text_rep = "\n".join([
            f"source: {src}",
            f"subsystem: {fd.get('dominant_subsystem', 'unknown')}",
            f"sha256: {fd.get('file_sha256', '')}",
            f"top_terms: {' '.join(terms_list[:30])}",
        ])
        entries.append({
            "doc_id": deterministic_uuid("digest_file", src),
            "source": src,
            "sha256": fd.get("file_sha256", ""),
            "subsystem": fd.get("dominant_subsystem", "unknown"),
            "chunk_count": fd.get("chunk_count", 0),
            "embedding_text": text_rep,
            "features": fd.get("avg_features", {}),
            "n_terms": len(terms_list),
        })

    # Repo-level embedding entries
    for rname, rd in repo_digests.items():
        terms_list = sorted(rd.get("top_terms", {}).keys(), key=lambda t: -rd["top_terms"][t])
        text_rep = "\n".join([
            f"repo: {rname}",
            f"sha256: {rd.get('repo_sha256', '')}",
            f"files: {rd.get('digested_file_count', 0)}",
            f"subsystem: {rd.get('dominant_subsystem', 'unknown')}",
            f"top_terms: {' '.join(terms_list[:30])}",
        ])
        entries.append({
            "doc_id": deterministic_uuid("digest_repo", rname),
            "source": f"repo:{rname}",
            "sha256": rd.get("repo_sha256", ""),
            "subsystem": "repo_digest",
            "chunk_count": rd.get("digested_file_count", 0),
            "embedding_text": text_rep,
            "features": rd.get("avg_structural_features", {}),
            "n_terms": len(terms_list),
        })

    manifest = {
        "schema": "lucidota.krampus_digest.embedding_manifest.v1",
        "generated_at": now(),
        "feature_version": FEATURE_VERSION,
        "manifest": {
            "description": "Content digest embedding manifest for hot rail (Needle 26M / TRM 7M)",
            "pipeline": "krampus_content_digest",
            "total_entries": len(entries),
            "file_entries": len(file_digests),
            "repo_entries": len(repo_digests),
        },
        "entries": entries,
    }
    return manifest


# --- Output Writers ---

def write_digest_output(file_digests: dict[str, dict[str, Any]], repo_digests: dict[str, dict[str, Any]], embedding_manifest: dict[str, Any], args: argparse.Namespace) -> Path:
    """Write all digest output files to 05_OUTPUTS/digest/."""
    ts = stamp()
    DIGEST_OUT.mkdir(parents=True, exist_ok=True)

    # 1. Per-repo digest JSONL files
    repo_outputs = defaultdict(list)
    for src, fd in file_digests.items():
        rname = repo_digests.get("odysseus", {}).get("repo_name", "odysseus")
        repo_outputs[rname].append(fd)

    written_files = []
    for rname, entries in repo_outputs.items():
        out_path = DIGEST_OUT / f"{rname}_digest_{ts}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        written_files.append(str(rel(out_path)))
        print(f"WROTE {len(entries)} entries to {rel(out_path)}")

    # 2. Repo-level digest summary
    repo_summary_path = DIGEST_OUT / f"repo_digest_summary_{ts}.json"
    repo_summary = {
        "schema": "lucidota.krampus_digest.repo_summary.v1",
        "generated_at": now(),
        "feature_version": FEATURE_VERSION,
        "repos": repo_digests,
        "source_extract": rel(RIVER_JSONL),
    }
    repo_summary_path.write_text(
        json.dumps(repo_summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    written_files.append(str(rel(repo_summary_path)))
    print(f"WROTE repo digest summary to {rel(repo_summary_path)}")

    # 3. Embedding manifest
    embed_path = DIGEST_OUT / f"embedding_manifest_{ts}.json"
    embed_path.write_text(
        json.dumps(embedding_manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    written_files.append(str(rel(embed_path)))
    print(f"WROTE embedding manifest to {rel(embed_path)}")

    # 4. Receipt
    receipt = {
        "schema": "lucidota.krampus_digest.receipt.v1",
        "generated_at": now(),
        "feature_version": FEATURE_VERSION,
        "pipeline": "krampus_content_digest",
        "source_jsonl": rel(RIVER_JSONL),
        "records_processed": len(file_digests),
        "file_count": len(file_digests),
        "repo_count": len(repo_digests),
        "embedding_entry_count": embedding_manifest["manifest"]["total_entries"],
        "output_files": written_files,
        "receipt_scope": "LOCAL_FILE_PRODUCT",
        "mutation_class": "receipt_only",
        "execute_performed": args.execute,
    }
    receipt_path = DIGEST_OUT / f"content_digest_receipt_{ts}.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    written_files.append(str(rel(receipt_path)))
    print(f"WROTE receipt to {rel(receipt_path)}")

    return receipt_path


# --- Postgres Receipt Writing ---

def write_db_receipts(
    file_digests: dict[str, dict[str, Any]],
    repo_digests: dict[str, dict[str, Any]],
    receipt_path: Path,
    args: argparse.Namespace,
) -> int:
    """Write content-digest receipts to lucidota_control.runtime_status_fact."""
    if psycopg is None:
        print("WARN: psycopg not installed, skipping DB receipts", file=sys.stderr)
        return 0

    db_url = args.database_url or os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or DEFAULT_DB
    inserted = 0

    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                # Write per-repo digest fact
                for rname, rd in repo_digests.items():
                    fact_key = f"content_digest_repo:{rname}"
                    fact_value = json.dumps({
                        "repo_name": rname,
                        "repo_sha256": rd.get("repo_sha256", ""),
                        "digested_files": rd.get("digested_file_count", 0),
                        "manifest_files": rd.get("manifest_file_count", 0),
                        "unique_term_count": rd.get("unique_term_count", 0),
                        "dominant_subsystem": rd.get("dominant_subsystem", "unknown"),
                        "top_terms_sample": list(rd.get("top_terms", {}).keys())[:10],
                        "feature_version": FEATURE_VERSION,
                    })
                    evidence = json.dumps([str(rel(receipt_path)), f"digest/{receipt_path.name}"])
                    if args.execute:
                        cur.execute(
                            """
                            INSERT INTO lucidota_control.runtime_status_fact (subsystem, fact_key, fact_value, evidence_refs)
                            VALUES (%s, %s, %s::jsonb, %s::jsonb)
                            ON CONFLICT (subsystem, fact_key) DO UPDATE
                              SET fact_value = EXCLUDED.fact_value,
                                  evidence_refs = EXCLUDED.evidence_refs,
                                  derived_at = now()
                            """,
                            (RECEIPT_SUBSYSTEM, fact_key, fact_value, evidence),
                        )
                        inserted += 1

                # Write summary fact
                summary_fact_key = "content_digest_summary"
                summary_value = json.dumps({
                    "total_files_digested": len(file_digests),
                    "total_repos": len(repo_digests),
                    "total_records_processed": sum(
                        fd.get("chunk_count", 0) for fd in file_digests.values()
                    ),
                    "total_unique_terms": sum(
                        rd.get("unique_term_count", 0) for rd in repo_digests.values()
                    ),
                    "feature_version": FEATURE_VERSION,
                    "receipt_path": rel(receipt_path),
                })
                if args.execute:
                    cur.execute(
                        """
                        INSERT INTO lucidota_control.runtime_status_fact (subsystem, fact_key, fact_value, evidence_refs)
                        VALUES (%s, %s, %s::jsonb, %s::jsonb)
                        ON CONFLICT (subsystem, fact_key) DO UPDATE
                          SET fact_value = EXCLUDED.fact_value,
                              evidence_refs = EXCLUDED.evidence_refs,
                              derived_at = now()
                        """,
                        (RECEIPT_SUBSYSTEM, summary_fact_key, summary_value, json.dumps([rel(receipt_path)])),
                    )
                    inserted += 1

                if args.execute:
                    conn.commit()

        status = "EXECUTED" if args.execute else "DRY_RUN"
        print(f"DB_RECEIPTS: {inserted} facts {status.lower()}")
        return inserted

    except Exception as e:
        print(f"DB_ERROR: {e}", file=sys.stderr)
        return -1


# --- Main ---

def main() -> int:
    p = argparse.ArgumentParser(description="Krampus Content Digest Pipeline")
    p.add_argument("--execute", action="store_true", help="Actually write files and DB receipts")
    p.add_argument("--digest-only", action="store_true", help="Skip RiverML re-read; only aggregate existing digests")
    p.add_argument("--database-url", help="Postgres DSN for receipt writing (default: $ABSURD_SYSTEM_DATABASE_URL or postgresql:///lucidota_state)")
    p.add_argument("--vocab-min-df", type=int, default=2, help="Minimum document frequency for vocabulary terms (default: 2)")
    p.add_argument("--vocab-max-df-ratio", type=float, default=0.85, help="Maximum document frequency ratio for vocabulary terms (default: 0.85)")
    p.add_argument("--tfidf-top-k", type=int, default=50, help="Top-K TF-IDF terms per record (default: 50)")
    args = p.parse_args()

    start = datetime.now(timezone.utc)

    # Phase 1: Read RiverML extraction
    print("=" * 60)
    print("PHASE 1: Reading RiverML extraction")
    print("=" * 60)
    records = read_river_jsonl(RIVER_JSONL)
    if not records:
        print("ERROR: No RiverML records found. Cannot continue.", file=sys.stderr)
        return 1

    # Phase 2: Compute global TF-IDF
    print("\n" + "=" * 60)
    print("PHASE 2: Computing global TF-IDF")
    print("=" * 60)
    vocab = build_global_vocabulary(records, min_df=args.vocab_min_df, max_df_ratio=args.vocab_max_df_ratio)
    augmented = compute_global_tfidf(records, vocab, top_k=args.tfidf_top_k)

    # Phase 3: Aggregate by file
    print("\n" + "=" * 60)
    print("PHASE 3: Aggregating by file")
    print("=" * 60)
    file_digests = aggregate_by_file(augmented)

    # Phase 4: Read repo manifests and compute per-repo digests
    print("\n" + "=" * 60)
    print("PHASE 4: Computing per-repo digests")
    print("=" * 60)
    manifests = read_repo_manifests()
    repo_metadata = extract_repos_from_manifests(manifests)
    print(f"DISCOVERED {len(repo_metadata)} repos from manifests: {list(repo_metadata.keys())}")
    repo_digests = compute_repo_digests(file_digests, repo_metadata, records)

    # Phase 5: Generate embedding manifest
    print("\n" + "=" * 60)
    print("PHASE 5: Generating embedding manifest")
    print("=" * 60)
    embedding_manifest = generate_embedding_manifest(file_digests, repo_digests)

    # Phase 6: Write output
    print("\n" + "=" * 60)
    print("PHASE 6: Writing digest output")
    print("=" * 60)
    receipt_path = write_digest_output(file_digests, repo_digests, embedding_manifest, args)

    # Phase 7: Write DB receipts
    print("\n" + "=" * 60)
    print("PHASE 7: Writing DB receipts")
    print("=" * 60)
    db_count = write_db_receipts(file_digests, repo_digests, receipt_path, args)

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    print("\n" + "=" * 60)
    print(f"DIGEST COMPLETE in {elapsed:.2f}s")
    print(f"  Records processed: {len(records)}")
    print(f"  File digests: {len(file_digests)}")
    print(f"  Repo digests: {len(repo_digests)}")
    print(f"  Embedding entries: {embedding_manifest['manifest']['total_entries']}")
    print(f"  DB facts written: {db_count if db_count >= 0 else 'ERROR'}")
    print(f"  Output: {rel(DIGEST_OUT)}/")
    print(f"  Receipt: {rel(receipt_path)}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
