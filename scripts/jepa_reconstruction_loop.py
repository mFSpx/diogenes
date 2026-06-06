#!/usr/bin/env python3
"""
JEPA RECONSTRUCTION LOOP — JSON Cannon → Bonsai → Error Farming Engine

Implements the "Path of No Path" self-supervised learning loop:

  1. COMPRESS: Shape vectors from Malkovich Siphon (lucidota_elastic_shape.rs)
     are the low-fidelity JSON "graffiti" — the mathematical shadow of the original data.

  2. RECONSTITUTE: A 1-bit Bonsai 8B model (Hyde adapter, port 8082) receives the
     shape JSON and attempts to hallucinate the original text.

  3. ERROR FARM: Reconstruction error = 1 - similarity(original, hallucinated).
     High error → the shape lost information → feed into Darwin Hammer for evolution.

  4. JEPA ENERGY: The reconstruction error IS the JEPA energy.
     E = || original_embedding - predicted_embedding ||² in text space.
     Minimizing E forces the compressor to preserve semantically meaningful geometry.

The loop never stops because the Bytewax river never stops — infinite self-supervised
training data from the gap between reality and its compressed shadow.

Usage:
  python3 scripts/jepa_reconstruction_loop.py --limit 100
  python3 scripts/jepa_reconstruction_loop.py --daemon --poll-interval 30
  python3 scripts/jepa_reconstruction_loop.py --source indy_reads --limit 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIPHON_SHAPE_DIR = ROOT / "05_OUTPUTS" / "malkovich_siphon" / "shape_vectors"
INDY_SHAPE_DIR = ROOT / "05_OUTPUTS" / "malkovich_siphon" / "indy_reads"
RECEIPT_DIR = ROOT / "05_OUTPUTS" / "malkovich_siphon" / "receipts"
JEPA_RECEIPT_DIR = ROOT / "05_OUTPUTS" / "malkovich_siphon" / "jepa_reconstruction"
ERROR_POOL_DIR = ROOT / "05_OUTPUTS" / "malkovich_siphon" / "gauntlet" / "error_pool"

BONSAI_API = "http://127.0.0.1:8082/v1/chat/completions"
BONSAI_MODEL = "bonsai8b-q1-shared2"
DEFAULT_RECONSTRUCTION_TEMP = 0.7

# ── Reconstruction prompt template ──────────────────────────────────────────

RECONSTRUCT_PROMPT = """You are a signal reconstruction engine. Below is the geometric JSON shadow of a text passage — its shape_vector, fidelity, and active_resonances after being compressed through an Elastic Ontology Compressor.

The original text was destroyed. You must hallucinate it back from the geometry alone.

Shape vector (JSON):
{shape_json}

Source: {source}
Fidelity: {fidelity:.4f}
Dimensions: {dimensions}

Reconstruct the original passage. Output ONLY the reconstructed text, nothing else."""


def now_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════════
# RECONSTRUCTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_from_shape(shape_row: dict, api_url: str = BONSAI_API,
                           model: str = BONSAI_MODEL,
                           temperature: float = DEFAULT_RECONSTRUCTION_TEMP,
                           timeout: int = 30) -> dict[str, Any]:
    """Send shape JSON to Bonsai model, get reconstructed text back."""
    shape_vector = shape_row.get("shape_vector", [])
    source = shape_row.get("source", shape_row.get("book_label", "unknown"))
    fidelity = shape_row.get("fidelity", 1.0)
    dimensions = len(shape_vector)

    shape_json = json.dumps({
        "shape_vector": shape_vector[:16],  # first 16 dims — enough signal
        "fidelity": fidelity,
        "dimensions": dimensions,
        "residual_norm": shape_row.get("residual_norm", 0),
        "collision": shape_row.get("collision", False),
    }, separators=(",", ":"))

    prompt = RECONSTRUCT_PROMPT.format(
        shape_json=shape_json,
        source=source,
        fidelity=fidelity,
        dimensions=dimensions,
    )

    # For text sources, include the text_preview as ground truth reference
    original_text = shape_row.get("text_preview", shape_row.get("text", ""))[:500]

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 256,
        "stream": False,
    }

    try:
        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            response = json.loads(resp.read())

        reconstructed = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {
            "ok": True,
            "reconstructed": reconstructed,
            "original": original_text,
            "source": source,
            "fidelity": fidelity,
            "dimensions": dimensions,
            "shape_vector_preview": shape_vector[:8],
            "model": model,
        }
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.reason}", "source": source}
    except urllib.error.URLError as e:
        return {"ok": False, "error": f"Connection: {e.reason}", "source": source}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "source": source}


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def tokenize_simple(text: str) -> set[str]:
    """Simple word tokenization for overlap metrics."""
    words = text.lower().split()
    return {w.strip(".,;:!?\"'()[]{}*_#@$%^&-+=<>/\\|~`") for w in words if len(w) >= 3}


def jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity between two texts."""
    tokens_a = tokenize_simple(a)
    tokens_b = tokenize_simple(b)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def ngram_overlap(a: str, b: str, n: int = 3) -> float:
    """N-gram overlap fraction. More robust than Jaccard for short texts."""
    def ngrams(text: str, n: int) -> set[str]:
        words = text.lower().split()
        if len(words) < n:
            return {text.lower()}
        return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}

    ngrams_a = ngrams(a, n)
    ngrams_b = ngrams(b, n)
    if not ngrams_a or not ngrams_b:
        return 0.0
    return len(ngrams_a & ngrams_b) / max(len(ngrams_a), len(ngrams_b))


def compute_reconstruction_error(original: str, reconstructed: str) -> dict[str, float]:
    """Compute error metrics between original and reconstructed text."""
    if not original or not reconstructed:
        return {"error_kind": "empty_input", "jaccard_sim": 0.0, "ngram_overlap": 0.0, "reconstruction_error": 1.0}

    jaccard = jaccard_similarity(original, reconstructed)
    ngram = ngram_overlap(original, reconstructed, n=3)
    # Error = 1 - similarity. High error = good (rich training signal).
    error = 1.0 - max(jaccard, ngram * 0.7)

    # Length ratio penalty — if reconstructed is much shorter, it's giving up
    len_ratio = min(len(reconstructed), len(original)) / max(len(reconstructed), len(original), 1)
    if len_ratio < 0.2:
        error = min(1.0, error + 0.2)  # severe length mismatch = higher error

    return {
        "jaccard_similarity": round(jaccard, 4),
        "ngram_overlap_3": round(ngram, 4),
        "reconstruction_error": round(error, 4),
        "original_len": len(original),
        "reconstructed_len": len(reconstructed),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR POOL — feeds Darwin Hammer
# ═══════════════════════════════════════════════════════════════════════════════

def write_to_error_pool(result: dict, error: float, threshold: float = 0.6) -> Path | None:
    """If reconstruction error exceeds threshold, write to error pool for evolution.

    The error pool is the fuel for the Darwin Hammer — high-error cases are the
    "collisions" that drive spectral evolution and algorithmic mutation.
    """
    if error < threshold:
        return None

    ERROR_POOL_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": now_z(),
        "error": error,
        "source": result.get("source", "unknown"),
        "fidelity": result.get("fidelity", 1.0),
        "original_preview": result.get("original", "")[:200],
        "reconstructed_preview": result.get("reconstructed", "")[:200],
        "jaccard_similarity": 1.0 - error,
        "shape_vector_preview": result.get("shape_vector_preview", []),
    }
    path = ERROR_POOL_DIR / f"jepa_error_{now_z().replace(':', '')}_{sha256_short(result.get('original', ''))}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# LOADER — reads shape vectors with original text
# ═══════════════════════════════════════════════════════════════════════════════

def load_shape_samples(source: str = "all", limit: int = 100) -> list[dict]:
    """Load shape vectors that have original text attached."""
    samples = []

    if source in ("all", "indy_reads"):
        indy_file = INDY_SHAPE_DIR / "indy_shape_vectors.jsonl"
        if indy_file.exists():
            with open(indy_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("text_preview"):
                        row["_source_type"] = "indy_reads"
                        samples.append(row)

    if source in ("all", "ahoy"):
        ahoy_file = SIPHON_SHAPE_DIR / "ahoy_shape_vectors.jsonl"
        if ahoy_file.exists():
            with open(ahoy_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Ahoy doesn't have text_preview, but we can use the feature dict
                    row["_source_type"] = "ahoy"
                    samples.append(row)

    if limit and len(samples) > limit:
        # Stratified sample: take equal proportions from each source
        indy_samples = [s for s in samples if s["_source_type"] == "indy_reads"]
        ahoy_samples = [s for s in samples if s["_source_type"] == "ahoy"]
        per_source = max(1, limit // 2)
        samples = indy_samples[:per_source] + ahoy_samples[:per_source]
        samples = samples[:limit]

    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_reconstruction_cycle(source: str = "all", limit: int = 100,
                              error_threshold: float = 0.6,
                              temperature: float = DEFAULT_RECONSTRUCTION_TEMP,
                              dry_run: bool = False) -> dict[str, Any]:
    """Run one full JEPA reconstruction cycle."""
    samples = load_shape_samples(source=source, limit=limit)
    if not samples:
        return {"verdict": "IDLE", "reason": "no_samples_found", "samples_loaded": 0}

    print(f"JEPA RECONSTRUCTION — {len(samples)} samples from {source}")
    print(f"  Bonsai API: {BONSAI_API}")
    print(f"  Error threshold: {error_threshold}")

    if dry_run:
        indy_count = sum(1 for s in samples if s["_source_type"] == "indy_reads")
        ahoy_count = sum(1 for s in samples if s["_source_type"] == "ahoy")
        print(f"  DRY RUN — {indy_count} IndyReads + {ahoy_count} Ahoy samples")
        print(f"  Would call {BONSAI_MODEL} {len(samples)} times")
        return {
            "verdict": "DRY_RUN",
            "samples_loaded": len(samples),
            "indy_samples": indy_count,
            "ahoy_samples": ahoy_count,
            "would_call_model": BONSAI_MODEL,
            "would_call_count": len(samples),
        }

    results = []
    errors: list[float] = []
    error_pool_count = 0
    api_failures = 0

    JEPA_RECEIPT_DIR.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        if i > 0 and i % 25 == 0:
            print(f"  ... {i}/{len(samples)} reconstructions")

        # 1. Reconstruct
        recon = reconstruct_from_shape(sample, temperature=temperature)

        if not recon["ok"]:
            api_failures += 1
            results.append(recon)
            continue

        # 2. Compute error
        error_metrics = compute_reconstruction_error(
            recon["original"], recon["reconstructed"]
        )
        error = error_metrics["reconstruction_error"]
        errors.append(error)

        result_entry = {
            "id": sample.get("id", sample.get("chunk_ref", str(uuid.uuid4()))),
            "source": recon["source"],
            "source_type": sample.get("_source_type", "unknown"),
            "fidelity": recon["fidelity"],
            "original_preview": recon["original"][:200],
            "reconstructed": recon["reconstructed"][:300],
            "error_metrics": error_metrics,
            "model": recon["model"],
        }

        # 3. Feed error pool if high error
        pool_path = write_to_error_pool(recon, error, threshold=error_threshold)
        if pool_path:
            error_pool_count += 1
            result_entry["error_pool_path"] = str(pool_path)

        results.append(result_entry)

    # Aggregate stats
    avg_error = sum(errors) / len(errors) if errors else 0.0
    median_error = float(np.median(errors)) if errors else 0.0
    max_error = max(errors) if errors else 0.0
    min_error = min(errors) if errors else 0.0

    # Per-source breakdown
    source_errors: dict[str, list[float]] = defaultdict(list)
    for r, e in zip(results, errors + [0.0] * (len(results) - len(errors))):
        st = r.get("source_type", "unknown")
        source_errors[st].append(e)

    source_stats = {}
    for st, errs in source_errors.items():
        valid = [e for e in errs if e > 0]
        source_stats[st] = {
            "count": len(errs),
            "avg_error": sum(valid) / len(valid) if valid else 0.0,
            "max_error": max(valid) if valid else 0.0,
        }

    # Write cycle receipt
    receipt = {
        "schema": "lucidota.jepa.reconstruction_receipt.v1",
        "verdict": "PASS" if api_failures < len(samples) * 0.5 else f"DEGRADED ({api_failures} API failures)",
        "samples_loaded": len(samples),
        "reconstructions_attempted": len(samples),
        "reconstructions_succeeded": len(results) - api_failures,
        "api_failures": api_failures,
        "avg_reconstruction_error": round(avg_error, 4),
        "median_reconstruction_error": round(median_error, 4),
        "min_error": round(min_error, 4),
        "max_error": round(max_error, 4),
        "error_distribution": {
            "very_low (0.0-0.3)": sum(1 for e in errors if e < 0.3),
            "low (0.3-0.5)": sum(1 for e in errors if 0.3 <= e < 0.5),
            "medium (0.5-0.7)": sum(1 for e in errors if 0.5 <= e < 0.7),
            "high (0.7-0.9)": sum(1 for e in errors if 0.7 <= e < 0.9),
            "very_high (0.9-1.0)": sum(1 for e in errors if e >= 0.9),
        },
        "error_pool_entries": error_pool_count,
        "error_threshold": error_threshold,
        "per_source_stats": source_stats,
        "model_used": BONSAI_MODEL,
        "api_url": BONSAI_API,
        "timestamp": now_z(),
    }

    receipt_path = JEPA_RECEIPT_DIR / f"jepa_cycle_{now_z().replace(':', '')}.json"
    tmp = receipt_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    tmp.replace(receipt_path)

    # Also write to main malkovich receipt dir
    malkovich_receipt = RECEIPT_DIR / f"jepa_reconstruction_{now_z().replace(':', '')}.json"
    malkovich_tmp = malkovich_receipt.with_suffix(".tmp")
    malkovich_tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    malkovich_tmp.replace(malkovich_receipt)

    print(f"\n  JEPA CYCLE COMPLETE")
    print(f"  Reconstructions: {len(results) - api_failures}/{len(samples)}")
    print(f"  Avg error: {avg_error:.4f}  Median: {median_error:.4f}")
    print(f"  Error pool: {error_pool_count} entries (threshold={error_threshold})")
    for st, stats in source_stats.items():
        print(f"    {st}: {stats['count']} samples, avg_error={stats['avg_error']:.4f}")
    print(f"  Receipt: {receipt_path}")

    return receipt


def daemon_loop(source: str, limit: int, poll_interval: int,
                error_threshold: float, temperature: float) -> None:
    """Continuous JEPA reconstruction daemon."""
    print(f"JEPA RECONSTRUCTION DAEMON — polling every {poll_interval}s")
    print(f"  Source: {source}  Limit: {limit}")
    print(f"  Bonsai: {BONSAI_API}  Model: {BONSAI_MODEL}")
    print(f"  Error threshold: {error_threshold}")
    print()

    cycle = 0
    while True:
        cycle += 1
        print(f"[{now_z()}] JEPA Cycle {cycle}")
        try:
            receipt = run_reconstruction_cycle(
                source=source, limit=limit,
                error_threshold=error_threshold, temperature=temperature,
            )
            verdict = receipt.get("verdict", "UNKNOWN")
            error_pool = receipt.get("error_pool_entries", 0)
            avg_error = receipt.get("avg_reconstruction_error", 0)
            print(f"  {verdict}: avg_error={avg_error:.4f}, error_pool={error_pool}")
            print()
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            print()

        time.sleep(poll_interval)


def main():
    ap = argparse.ArgumentParser(
        description="JEPA RECONSTRUCTION LOOP — JSON Cannon → Bonsai → Error Farming"
    )
    ap.add_argument("--source", default="all",
                    choices=["all", "indy_reads", "ahoy"],
                    help="Which shape vector source to use (default: all)")
    ap.add_argument("--limit", type=int, default=100,
                    help="Max reconstructions per cycle (default: 100)")
    ap.add_argument("--error-threshold", type=float, default=0.6,
                    help="Minimum error to feed into Darwin Hammer pool (default: 0.6)")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="Model temperature for reconstruction (default: 0.7)")
    ap.add_argument("--daemon", action="store_true",
                    help="Run in continuous daemon mode")
    ap.add_argument("--poll-interval", type=int, default=30,
                    help="Seconds between daemon cycles (default: 30)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Load samples but don't call the model")
    ap.add_argument("--json", action="store_true",
                    help="Output receipt as JSON to stdout")
    ap.add_argument("--api-url", default=BONSAI_API,
                    help=f"Bonsai API URL (default: {BONSAI_API})")
    args = ap.parse_args()

    # Allow overriding API URL via module-level variable for reconstruct_from_shape
    import scripts.jepa_reconstruction_loop as _self
    _self.BONSAI_API = args.api_url

    if args.daemon:
        daemon_loop(args.source, args.limit, args.poll_interval,
                    args.error_threshold, args.temperature)
        return 0

    receipt = run_reconstruction_cycle(
        source=args.source, limit=args.limit,
        error_threshold=args.error_threshold,
        temperature=args.temperature,
        dry_run=args.dry_run,
    )

    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
