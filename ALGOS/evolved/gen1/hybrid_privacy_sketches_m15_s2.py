# DARWIN HAMMER — match 15, survivor 2
# gen: 1
# parent_a: privacy.py (gen0)
# parent_b: sketches.py (gen0)
# born: 2026-05-29T23:25:05Z

"""Hybrid privacy‑sketch module.

This module fuses the core mathematics of *privacy.py* and *sketches.py*.
The bridge is the use of a **Count‑Min Sketch (CMS)** matrix as a
compact estimator for the quantities that the privacy helpers need:
- The reconstruction‑risk score requires the ratio  
  `unique_quasi_identifiers / total_records`.  
  We approximate `unique_quasi_identifiers` with a cardinality estimate
  derived from the CMS (count of non‑zero cells) and `total_records`
  with a HyperLogLog‑style distinct‑count (exact set size for the demo).
- Differential‑privacy aggregation adds Laplace noise directly to the
  CMS cells before reconstructing the sum, preserving the linearity of
  the sketch.
- Anonymization is applied before MinHash LSH, ensuring that redacted
  tokens never influence the bucket keys.

The three public functions below demonstrate these hybrid operations."""

import hashlib
import random
import math
import sys
from pathlib import Path
from collections import defaultdict
from typing import Iterable, Dict, Set, List, Any

import numpy as np


def _cms_hash(item: str, depth: int, width: int) -> List[int]:
    """Return a list of column indices, one per hash row."""
    return [
        int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
        for d in range(depth)
    ]


def count_min_sketch(items: Iterable[str], width: int = 64, depth: int = 4) -> np.ndarray:
    """Build a Count‑Min Sketch matrix as a NumPy int array."""
    cms = np.zeros((depth, width), dtype=np.int64)
    for item in items:
        cols = _cms_hash(item, depth, width)
        for d, c in enumerate(cols):
            cms[d, c] += 1
    return cms


def _estimate_cardinality_from_cms(cms: np.ndarray) -> int:
    """
    Very coarse cardinality estimator: count distinct (row, col) cells
    that received at least one increment and divide by depth.
    """
    nonzero = np.count_nonzero(cms)
    depth = cms.shape[0]
    return max(1, nonzero // depth)


def hyperloglog_cardinality(items: Iterable[str]) -> int:
    """Exact distinct count (placeholder for a real HLL implementation)."""
    return len(set(items))


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Ratio‑based risk, clipped to [0,1]."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def hybrid_reconstruction_risk(
    records: List[Dict[str, Any]],
    quasi_keys: Set[str],
    width: int = 128,
    depth: int = 5,
) -> float:
    """
    Compute a privacy risk score using sketches.

    Steps
    -----
    1. Concatenate the values of the quasi‑identifier keys for each record
       into a single string and feed it to a CMS.
    2. Estimate the number of *unique* quasi‑identifier combinations from the CMS.
    3. Estimate the total number of records via a HyperLogLog‑style distinct count.
    4. Return the classic reconstruction‑risk ratio.
    """
    # 1. Build CMS over quasi‑identifier signatures
    signatures = [
        "||".join(str(record.get(k, "")) for k in sorted(quasi_keys))
        for record in records
    ]
    cms = count_min_sketch(signatures, width=width, depth=depth)

    # 2. Estimate unique quasi identifiers
    uniq_est = _estimate_cardinality_from_cms(cms)

    # 3. Total records (exact for the demo)
    total = hyperloglog_cardinality([str(i) for i in range(len(records))])

    # 4. Risk score
    return reconstruction_risk_score(uniq_est, total)


def dp_aggregate_sketch(
    values: Iterable[float],
    epsilon: float = 1.0,
    sensitivity: float = 1.0,
    width: int = 64,
    depth: int = 4,
) -> float:
    """
    Differential‑private sum using a noisy Count‑Min Sketch.

    The sketch is built over the string representation of each value.
    Laplace noise (scale = sensitivity/epsilon) is added to every cell.
    The unbiased estimator of the total sum is the average of the row sums.
    """
    # Build CMS where each item contributes its numeric value to the cell.
    cms = np.zeros((depth, width), dtype=np.float64)
    for v in values:
        item = f"{v:.12g}"
        cols = _cms_hash(item, depth, width)
        for d, c in enumerate(cols):
            cms[d, c] += v

    # Add Laplace noise
    scale = sensitivity / epsilon
    noise = np.random.laplace(0.0, scale, size=cms.shape)
    noisy_cms = cms + noise

    # Estimate sum: average of row sums (property of linear sketches)
    row_sums = noisy_cms.sum(axis=1)
    return float(row_sums.mean())


def anonymize_for_indexing(record: Dict[str, Any], redact_keys: Set[str] | None = None) -> Dict[str, Any]:
    """Redact sensitive fields before indexing."""
    redact = redact_keys or {"email", "phone", "ssn", "secret", "token", "password"}
    return {
        k: ("<redacted>" if k.lower() in redact else v) for k, v in record.items()
    }


def anonymized_minhash_lsh(
    docs: Dict[str, Set[str]],
    redact_keys: Set[str] | None = None,
) -> Dict[str, List[str]]:
    """
    Apply anonymization to each shingle before MinHash LSH bucketisation.

    The bucket key is the minimal SHA‑1 hash prefix (6 hex chars) among the
    redacted shingles of a document.
    """
    redact = redact_keys or {"email", "phone", "ssn", "secret", "token", "password"}
    buckets: Dict[str, List[str]] = defaultdict(list)

    for doc_id, shingles in docs.items():
        # Anonymize each shingle
        redacted = {
            "<redacted>" if sh.lower() in redact else sh for sh in shingles
        }
        # Compute minimal hash prefix
        min_hash = min(
            (hashlib.sha1(s.encode()).hexdigest()[:6] for s in redacted),
            default="empty",
        )
        buckets[min_hash].append(doc_id)

    return dict(buckets)


if __name__ == "__main__":
    # Smoke test for hybrid functions

    # Sample records for risk scoring
    sample_records = [
        {"id": i, "age": random.randint(20, 70), "zip": f"{random.randint(10000,99999)}"}
        for i in range(1, 101)
    ]
    risk = hybrid_reconstruction_risk(
        sample_records, quasi_keys={"age", "zip"}, width=128, depth=5
    )
    print(f"Hybrid reconstruction risk: {risk:.4f}")

    # Differential‑private aggregation
    values = [random.random() * 10 for _ in range(200)]
    noisy_sum = dp_aggregate_sketch(values, epsilon=0.5, sensitivity=1.0)
    print(f"Noisy DP sum (sketch): {noisy_sum:.4f}")

    # Anonymized MinHash LSH
    docs = {
        f"doc{i}": {f"word{random.randint(1,50)}" for _ in range(20)}
        for i in range(5)
    }
    lsh_index = anonymized_minhash_lsh(docs, redact_keys={"word10", "word20"})
    print("MinHash LSH buckets:", lsh_index)