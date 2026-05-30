# DARWIN HAMMER — match 4203, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_ternary_lens__capybara_optimizatio_m54_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m613_s2.py (gen5)
# born: 2026-05-29T23:54:15Z

"""Hybrid Lens‑Audit & Path‑Signature Surrogate
================================================

Parent A: ``hybrid_hybrid_ternary_lens__capybara_optimizatio_m54_s0.py``
    Provides a manifest loader and a *social interaction pruning* rule that
    nudges a candidate’s findings toward those of the current global best
    when classifications match.

Parent B: ``hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m613_s2.py``
    Supplies path‑signature extraction, Shannon entropy of the level‑2 signature,
    a MinHash‑derived discrete force series (integrated to a peak velocity) and
    an RBF surrogate whose Gaussian width is modulated by the entropy.

Mathematical Bridge
-------------------
The bridge is the **entropy scalar** obtained from the level‑2 signature of a
candidate’s (pruned) findings vector.  This scalar ``H`` multiplies the base
kernel width ``ε₀`` of the RBF surrogate (``ε = ε₀·(1+H)``).  Simultaneously,
the MinHash‑derived force series yields a peak velocity ``v_peak`` that is
appended to the signature‑based feature vector.  The final feature vector

    Φ = [sig₁, flatten(sig₂), H, v_peak]

is fed to the surrogate, producing a unified prediction that respects both
parent algorithms’ governing equations.

The module implements a self‑contained pipeline:
    1. Load a vendor manifest.
    2. Apply social‑interaction pruning.
    3. Extract path‑signature features and entropy.
    4. Generate a MinHash force series and integrate to obtain ``v_peak``.
    5. Predict a quality score via the entropy‑scaled RBF surrogate.
"""

import json
import math
import random
import sys
from pathlib import Path
import hashlib
from typing import Sequence, List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Constants & Types
# ----------------------------------------------------------------------
Vector = Sequence[float]
Candidate = Dict[str, Any]

# ----------------------------------------------------------------------
# Parent‑A utilities (manifest handling & pruning)
# ----------------------------------------------------------------------
def utc_now() -> str:
    """Current UTC timestamp in ISO‑8601 format (no microseconds)."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_manifest(path: Path) -> dict[str, Any]:
    """Read a JSON manifest and validate classifications."""
    data = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"usable_now", "research_only", "needs_conversion",
               "unsafe_for_fastpath", "unsupported"}
    for cand in data.get("vendors", []):
        cls = cand.get("classification")
        if cls not in allowed:
            raise SystemExit(f"invalid classification {cls!r} for {cand.get('candidate_key')}")
    return data


def social_interaction_pruning(candidate: Candidate,
                               g_best: Candidate,
                               k: float = 1.0,
                               r1: float | None = None,
                               seed: int | str | None = None) -> Candidate:
    """
    If ``candidate`` and ``g_best`` share the same classification, move each
    finding toward the global best according to the classic PSO‑style rule:

        f_i ← f_i + r1·(g_i – k·f_i)

    ``r1`` is a random coefficient in ``[0,1)``; ``k`` controls the pull‑strength.
    The function mutates and returns ``candidate``.
    """
    if r1 is None:
        rng = random.Random(seed)
        r1 = rng.random()
    else:
        rng = random.Random(seed)

    if candidate.get("classification") == g_best.get("classification"):
        f = candidate.get("findings", [])
        g = g_best.get("findings", [])
        # Guard against length mismatch
        if len(f) != len(g):
            min_len = min(len(f), len(g))
            f = f[:min_len]
            g = g[:min_len]
        candidate["findings"] = [fi + r1 * (gi - k * fi) for fi, gi in zip(f, g)]
    return candidate

# ----------------------------------------------------------------------
# Parent‑B utilities (signature, entropy, MinHash, RBF)
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Simple lead‑lag embedding for a 1‑D path.
    For a path ``x_0,…,x_n`` the lead‑lag vector is
    ``[x_0, x_0, x_1, x_1, …, x_n, x_n]`` shifted by one step.
    """
    if path.ndim != 1:
        raise ValueError("lead_lag_transform expects a 1‑D array")
    lead = path[:-1]
    lag = path[1:]
    return np.column_stack((lead, lag)).flatten()


def level_one_signature(path: np.ndarray) -> float:
    """Level‑1 signature is the total increment (integral of the path)."""
    return float(np.sum(np.diff(path)))


def level_two_signature(path: np.ndarray) -> np.ndarray:
    """
    Approximate the level‑2 signature matrix using the iterated integral:

        S^{(2)} = ∫_0^T ∫_0^t dX_s ⊗ dX_t

    For a discrete 1‑D path we use the outer product of increments.
    The result is a 2×2 symmetric matrix.
    """
    inc = np.diff(path)[:, None]  # column vector
    # Outer product of increments summed over time
    S = np.dot(inc.T, inc)
    # Ensure a 2×2 matrix (pad with zeros if needed)
    if S.shape != (2, 2):
        S = np.pad(S, ((0, 2 - S.shape[0]), (0, 2 - S.shape[1])), constant_values=0.0)
    return S


def signature_entropy(sig2: np.ndarray) -> float:
    """
    Compute Shannon entropy of the normalized eigen‑values of the level‑2 signature.
    """
    eigvals = np.linalg.eigvalsh(sig2)
    # Clip to avoid negative values due to numerical error
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    total = np.sum(eigvals)
    if total == 0.0:
        return 0.0
    probs = eigvals / total
    # Avoid log(0) by adding a tiny epsilon
    eps = np.finfo(float).eps
    return -np.sum(probs * np.log(probs + eps))


def path_signature_features(findings: List[float]) -> Tuple[float, np.ndarray, float]:
    """
    Given a list of findings, return:
        (sig1, sig2, entropy)
    where ``sig1`` is level‑1, ``sig2`` is the level‑2 matrix, and ``entropy``
    is the Shannon entropy of ``sig2``.
    """
    path = np.asarray(findings, dtype=float)
    if path.size < 2:
        # Pad with a zero to allow diff()
        path = np.pad(path, (0, 2 - path.size), constant_values=0.0)
    sig1 = level_one_signature(path)
    sig2 = level_two_signature(path)
    H = signature_entropy(sig2)
    return sig1, sig2, H


def minhash_force_series(vector: Vector, num_hashes: int = 8) -> List[float]:
    """
    Produce a deterministic pseudo‑random force series from a vector.
    Each hash is derived from SHA‑256 of ``vector`` concatenated with the hash index.
    The integer digest is mapped to a float in ``[-1, 1]``.
    """
    series = []
    base = json.dumps(vector, sort_keys=True).encode("utf-8")
    for i in range(num_hashes):
        h = hashlib.sha256(base + i.to_bytes(4, "little")).digest()
        # Take first 8 bytes as unsigned integer
        val = int.from_bytes(h[:8], "big")
        # Map to [-1, 1]
        series.append((val / (2**64 - 1)) * 2.0 - 1.0)
    return series


def integrate_force_series(series: List[float], dt: float = 1.0) -> Tuple[List[float], float]:
    """
    Simple Euler integration of a force series to obtain velocity.
    Returns the full velocity trajectory and the peak (max absolute) velocity.
    """
    vel = []
    v = 0.0
    for f in series:
        v += f * dt
        vel.append(v)
    v_peak = max(abs(v) for v in vel) if vel else 0.0
    return vel, v_peak


def rbf_surrogate_predict(phi: np.ndarray,
                          training_phi: np.ndarray,
                          training_y: np.ndarray,
                          epsilon_base: float = 1.0,
                          entropy: float = 0.0) -> float:
    """
    Gaussian RBF prediction.
    Kernel width is scaled by entropy: ``ε = ε₀·(1+entropy)``.
    Prediction is the weighted average of training targets.
    """
    epsilon = epsilon_base * (1.0 + entropy)
    if epsilon <= 0.0:
        epsilon = 1e-6
    # Compute Gaussian kernel between ``phi`` and each training example
    dists = np.linalg.norm(training_phi - phi, axis=1)
    kernels = np.exp(-(dists ** 2) / (2 * epsilon ** 2))
    # Avoid division by zero
    if np.sum(kernels) == 0.0:
        return float(np.mean(training_y))
    return float(np.dot(kernels, training_y) / np.sum(kernels))

# ----------------------------------------------------------------------
# Hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_candidate_score(candidate: Candidate,
                           global_best: Candidate,
                           training_phi: np.ndarray,
                           training_y: np.ndarray,
                           epsilon_base: float = 1.0,
                           k_prune: float = 1.0,
                           seed: int | str | None = None) -> float:
    """
    Full hybrid operation for a single candidate:

    1. Apply social‑interaction pruning toward ``global_best``.
    2. Extract signature features and entropy from the (pruned) findings.
    3. Generate a MinHash force series from the candidate key and integrate it
       to obtain ``v_peak``.
    4. Assemble the feature vector ``Φ`` and feed it to the entropy‑scaled RBF
       surrogate.

    Returns the surrogate’s predicted score.
    """
    # 1. Pruning
    pruned = social_interaction_pruning(candidate, global_best, k=k_prune, seed=seed)

    findings = pruned.get("findings", [])
    if not isinstance(findings, list) or not findings:
        findings = [0.0]  # fallback

    # 2. Signature & entropy
    sig1, sig2, H = path_signature_features(findings)

    # Flatten level‑2 matrix (row‑major)
    sig2_flat = sig2.flatten()

    # 3. MinHash force series → v_peak
    key = pruned.get("candidate_key", "unknown")
    force_series = minhash_force_series([key])
    _, v_peak = integrate_force_series(force_series)

    # 4. Assemble Φ
    phi = np.concatenate(([sig1], sig2_flat, [H, v_peak])).astype(float)

    # 5. Surrogate prediction
    score = rbf_surrogate_predict(phi, training_phi, training_y,
                                  epsilon_base=epsilon_base, entropy=H)
    return score


# ----------------------------------------------------------------------
# Simple in‑memory training data generator (for the smoke test)
# ----------------------------------------------------------------------
def _generate_dummy_training(num_samples: int = 20, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce synthetic training feature vectors and targets.
    The features follow the same layout as ``Φ`` used by the hybrid pipeline.
    Targets are a noisy linear combination of the components.
    """
    rng = np.random.default_rng(seed)
    # Components: sig1 (1), sig2 (4), H (1), v_peak (1) → total 7
    training_phi = rng.normal(size=(num_samples, 7))
    # Simple linear model + noise
    weights = np.array([0.3, 0.1, -0.2, 0.05, 0.07, 0.4, -0.1])
    training_y = training_phi @ weights + rng.normal(scale=0.1, size=num_samples)
    return training_phi, training_y


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 0. Locate a (possibly non‑existent) manifest – we fall back to a tiny stub.
    manifest_path = Path(__file__).with_name("dummy_manifest.json")
    if manifest_path.is_file():
        manifest = load_manifest(manifest_path)
    else:
        # Create a minimal manifest on‑the‑fly
        manifest = {
            "vendors": [
                {
                    "candidate_key": "alpha",
                    "classification": "usable_now",
                    "findings": [0.2, 0.5, -0.1, 0.3]
                },
                {
                    "candidate_key": "beta",
                    "classification": "usable_now",
                    "findings": [0.1, -0.4, 0.6]
                },
                {
                    "candidate_key": "gamma",
                    "classification": "research_only",
                    "findings": [0.0, 0.0, 0.0]
                }
            ]
        }

    vendors = manifest.get("vendors", [])
    if not vendors:
        sys.exit("No candidates found in manifest.")

    # Choose the first candidate as the initial global best
    global_best = vendors[0].copy()

    # Generate dummy training data for the surrogate
    train_phi, train_y = _generate_dummy_training()

    print(f"Timestamp: {utc_now()}")
    for cand in vendors:
        score = hybrid_candidate_score(cand, global_best,
                                       training_phi=train_phi,
                                       training_y=train_y,
                                       epsilon_base=0.5,
                                       k_prune=0.8,
                                       seed=123)
        print(f"Candidate {cand['candidate_key']!r} → predicted score: {score:.4f}")

    # Update global best based on the highest predicted score
    best_cand = max(vendors, key=lambda c: hybrid_candidate_score(c, global_best,
                                                                 training_phi=train_phi,
                                                                 training_y=train_y))
    print(f"Best candidate after evaluation: {best_cand['candidate_key']}")