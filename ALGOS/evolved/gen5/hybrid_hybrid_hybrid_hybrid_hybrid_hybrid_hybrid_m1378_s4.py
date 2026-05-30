# DARWIN HAMMER — match 1378, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_ternar_korpus_text_m1017_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_minimu_m14_s1.py (gen3)
# born: 2026-05-29T23:35:43Z

"""Hybrid Fusion of Text Feature Routing and Epistemic Decision Audit

Parents:
- hybrid_hybrid_hybrid_ternar_korpus_text_m1017_s0.py (text → minhash + entropy → Euclidean cost matrix → ternary routing & Voronoi)
- hybrid_hybrid_hybrid_decisi_hybrid_hybrid_minimu_m14_s1.py (epistemic certainty flags modify edge weights, regex‑based hygiene audit, Bayesian updates)

Mathematical Bridge:
Both parents operate on a complete graph whose edge weight is a Euclidean‑type distance.
Parent A supplies the raw symmetric cost matrix **C** where Cᵢⱼ = ‖vᵢ−vⱼ‖² for feature vectors *v*.
Parent B supplies a scalar epistemic multiplier ϕ(flag) ∈ (0, ∞) that rescales each edge according to the
certainty of its incident nodes.  The fused edge‑weight matrix **W** is therefore

    Wᵢⱼ = Cᵢⱼ · ½·(ϕ(flagᵢ)+ϕ(flagⱼ))

All subsequent algorithms (ternary routing, Voronoi partition, and the audit decision) use **W**
instead of the raw Euclidean cost, thereby unifying the two topologies into a single
matrix‑based system.
"""

import math
import random
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# 1. Text‑based feature extraction (Parent A)
# ----------------------------------------------------------------------


def _shingles(text: str, width: int = 5) -> List[str]:
    """Generate overlapping substrings (shingles) of given width."""
    return [text[i: i + width] for i in range(len(text) - width + 1)]


def minhash_signature(text: str, k: int = 64, width: int = 5) -> np.ndarray:
    """
    Very small deterministic minhash: return the k smallest 64‑bit hashes
    of the shingles as a float array normalised to [0, 1].
    """
    if not text:
        return np.zeros(k, dtype=float)
    shingles = _shingles(text.lower(), width)
    hashes = [hash(s) & 0xFFFFFFFFFFFFFFFF for s in shingles]
    hashes.sort()
    # pad / truncate to k elements
    sig = (hashes[:k] + [0] * k)[:k]
    return np.array(sig, dtype=float) / float(0xFFFFFFFFFFFFFFFF)


def shannon_entropy(text: str, max_len: int = 10_000) -> float:
    """Shannon entropy of the character distribution (clipped to max_len)."""
    if not text:
        return 0.0
    text = text[:max_len]
    counts = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def extract_features(text: str) -> np.ndarray:
    """
    Concatenate normalised minhash signature with a single entropy scalar.
    Resulting vector lives in ℝ^{k+1}.
    """
    sig = minhash_signature(text)                     # shape (k,)
    ent = np.array([shannon_entropy(text)], dtype=float)  # shape (1,)
    return np.concatenate([sig, ent])                # shape (k+1,)


# ----------------------------------------------------------------------
# 2. Epistemic certainty handling (Parent B)
# ----------------------------------------------------------------------


EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
# Mapping to a multiplier: lower multiplier → higher trust (cheaper edge)
_EPISTEMIC_MULTIPLIER: Dict[str, float] = {
    "FACT": 0.5,
    "PROBABLE": 0.8,
    "POSSIBLE": 1.0,
    "BULLSHIT": 1.5,
    "SURE_MAYBE": 1.2,
}


def epistemic_multiplier(flag: str) -> float:
    """Return the scalar factor associated with an epistemic flag."""
    return _EPISTEMIC_MULTIPLIER.get(flag.upper(), 1.0)


def build_epistemic_matrix(flags: List[str]) -> np.ndarray:
    """
    Construct a symmetric matrix Φ where Φᵢⱼ = ½·(ϕ(flagᵢ)+ϕ(flagⱼ)).
    This matrix will later be multiplied element‑wise with the Euclidean cost matrix.
    """
    n = len(flags)
    phi = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            val = 0.5 * (epistemic_multiplier(flags[i]) + epistemic_multiplier(flags[j]))
            phi[i, j] = phi[j, i] = val
    return phi


# ----------------------------------------------------------------------
# 3. Core matrix operations (shared between both parents)
# ----------------------------------------------------------------------


def cost_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    Compute the symmetric matrix C where Cᵢⱼ = ‖vᵢ−vⱼ‖².
    Vectors shape: (n, d)
    """
    # Using broadcasting: (v_i - v_j)^2 summed over dimensions
    diff = vectors[:, np.newaxis, :] - vectors[np.newaxis, :, :]   # (n, n, d)
    return np.einsum('ijk,ijk->ij', diff, diff)                    # (n, n)


def fused_weight_matrix(cost: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Fuse Euclidean costs with epistemic multipliers:
        W = C ⊙ Φ
    where ⊙ denotes element‑wise multiplication.
    """
    return cost * phi


# ----------------------------------------------------------------------
# 4. Hybrid algorithms
# ----------------------------------------------------------------------


def ternary_route(source: int, destination: int, W: np.ndarray) -> Tuple[int, float]:
    """
    Select an intermediate node k that minimises
        W[source, k] + W[k, destination].
    Returns (k, total_cost).  If source==destination, k == source.
    """
    if source == destination:
        return source, 0.0
    n = W.shape[0]
    best_k = source
    best_cost = math.inf
    for k in range(n):
        cost = W[source, k] + W[k, destination]
        if cost < best_cost:
            best_cost, best_k = cost, k
    return best_k, best_cost


def voronoi_partition(seeds: List[int], W: np.ndarray) -> List[int]:
    """
    Assign each node i to the nearest seed index according to weighted distance.
    Returns a list `assignment` where assignment[i] = seed_id (the element from `seeds`).
    """
    n = W.shape[0]
    assignment = [-1] * n
    for i in range(n):
        nearest_seed = min(seeds, key=lambda s: W[i, s])
        assignment[i] = nearest_seed
    return assignment


def hygiene_score(text: str) -> float:
    """
    Simple hygiene metric based on presence of evidence‑related keywords.
    Returns a score in [0, 1] where 1 means fully hygienic (many evidence tokens).
    """
    tokens = re.findall(r'\b\w+\b', text.lower())
    if not tokens:
        return 0.0
    evidence_hits = sum(1 for t in tokens if t in {
        "evidence", "verify", "verified", "confirm", "confirmed", "source",
        "sourced", "citation", "receipt", "hash", "sha256", "screenshot",
        "record", "log", "document", "proof", "fact", "facts", "check",
        "checked", "audit"
    })
    return min(1.0, evidence_hits / len(tokens))


def audit_decision(texts: List[str], flags: List[str]) -> List[Dict]:
    """
    For each text produce an audit dict containing:
        - entropy
        - hygiene_score
        - epistemic flag
        - Bayesian‑updated confidence (using a dummy likelihood model)
    """
    reports = []
    for txt, flg in zip(texts, flags):
        prior = epistemic_multiplier(flg) / max(_EPISTEMIC_MULTIPLIER.values())  # normalised prior ∈ (0,1]
        likelihood = hygiene_score(txt)  # treat hygiene as likelihood of being trustworthy
        marginal = prior * likelihood + (1 - prior) * 0.1  # false‑positive fixed at 0.1
        posterior = (likelihood * prior) / marginal if marginal else 0.0
        reports.append({
            "entropy": shannon_entropy(txt),
            "hygiene": hygiene_score(txt),
            "flag": flg,
            "posterior_confidence": posterior,
        })
    return reports


def hybrid_route_and_audit(source_idx: int, dest_idx: int,
                           texts: List[str], flags: List[str]) -> Dict:
    """
    End‑to‑end hybrid operation:
        1. Extract feature vectors.
        2. Build Euclidean cost matrix C.
        3. Build epistemic matrix Φ and fuse → W.
        4. Perform ternary routing (source → k → destination).
        5. Run audit on the three involved texts (source, k, destination).
    Returns a dict summarising the path and audit reports.
    """
    vectors = np.vstack([extract_features(t) for t in texts])          # (n, d)
    C = cost_matrix(vectors)                                          # (n, n)
    Φ = build_epistemic_matrix(flags)                                 # (n, n)
    W = fused_weight_matrix(C, Φ)                                     # (n, n)

    k, path_cost = ternary_route(source_idx, dest_idx, W)

    audit_reports = audit_decision([texts[source_idx], texts[k], texts[dest_idx]],
                                   [flags[source_idx], flags[k], flags[dest_idx]])

    return {
        "intermediate": k,
        "path_cost": path_cost,
        "audit": audit_reports,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample corpus
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. Evidence: captured in video.",
        "Plan the roadmap and schedule the test phases. Verify all checkpoints.",
        "Random gibberish without any meaningful content or proof.",
        "Confirmed source: hash sha256 abcdef1234567890. Document attached.",
        "Possible future work includes further analysis and audits."
    ]

    # Assign random epistemic flags to each document
    random.seed(42)
    sample_flags = [random.choice(EPISTEMIC_FLAGS) for _ in sample_texts]

    # Choose source and destination indices
    src, dst = 0, 3

    result = hybrid_route_and_audit(src, dst, sample_texts, sample_flags)

    print("Source index:", src)
    print("Destination index:", dst)
    print("Intermediate node:", result["intermediate"])
    print("Total weighted path cost:", result["path_cost"])
    print("\nAudit reports:")
    for i, rep in enumerate(result["audit"]):
        print(f"  Node {i} ({'source' if i==0 else 'intermediate' if i==1 else 'dest'}):")
        for k, v in rep.items():
            print(f"    {k}: {v}")
    # Demonstrate Voronoi partition with two seeds
    vectors = np.vstack([extract_features(t) for t in sample_texts])
    C = cost_matrix(vectors)
    Φ = build_epistemic_matrix(sample_flags)
    W = fused_weight_matrix(C, Φ)
    partitions = voronoi_partition(seeds=[0, 2], W=W)
    print("\nVoronoi assignment (seeds 0 and 2):", partitions)