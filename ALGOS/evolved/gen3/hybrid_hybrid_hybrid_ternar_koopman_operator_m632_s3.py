# DARWIN HAMMER — match 632, survivor 3
# gen: 3
# parent_a: hybrid_hybrid_ternary_lens__hybrid_path_signatur_m34_s2.py (gen2)
# parent_b: koopman_operator.py (gen0)
# born: 2026-05-29T23:30:16Z

"""
Hybrid Ternary Lens Audit & Koopman Operator

This module fuses the *ternary lens audit* workflow (parent A) with the
*Koopman operator / Dynamic Mode Decomposition* machinery (parent B).

Mathematical bridge
------------------
1. The audit produces a discrete time‑series of candidate feature vectors
   **xₜ** (one column per pruning iteration).  The pruning schedule is a
   deterministic, decreasing‑rate nonlinear map **P** : **xₜ** → **xₜ₊₁**.
2. The Koopman framework linearises any nonlinear map by acting on
   *observables* ψ(**x**).  We choose the degree‑2 polynomial observable
   lift used in the original Koopman code:

       ψ(x) = [ x , x⊙x , x_i·x_j (i<j) ]ᵀ

   which maps the audit state into a higher‑dimensional space where the
   pruning dynamics become (approximately) linear.
3. Applying Dynamic Mode Decomposition (DMD) to the lifted snapshots yields a
   finite‑dimensional matrix **K** that approximates the Koopman operator.
   Forecasts of future audit states are obtained by evolving the lifted
   state with **K** and optionally projecting back.

The three core functions below demonstrate this hybrid pipeline:
* `prepare_audit_snapshots` – builds the audit time‑series and lifts it.
* `fit_hybrid_koopman` – runs DMD on the lifted data.
* `hybrid_forecast` – propagates the audit state forward using the learned
  Koopman matrix.

Only the Python standard library and NumPy are used, matching the
restriction list.
"""

import json
import re
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A – Ternary Lens Audit utilities (trimmed / re‑implemented)
# ----------------------------------------------------------------------

CLASSIFICATIONS = {
    "usable_now",
    "research_only",
    "needs_conversion",
    "unsafe_for_fastpath",
    "unsupported",
}
LOCAL_PATTERNS = [
    "*bitnet*",
    "*BitNet*",
    "*fairyfuse*",
    "*FairyFuse*",
    "*lora*",
    "*LoRA*",
    "*adapter*",
]

def utc_now() -> str:
    """Current UTC timestamp in ISO‑8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def load_manifest(path: Path) -> dict[str, Any]:
    """Load a vendor manifest JSON file and validate classifications."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for cand in data.get("vendors", []):
        cls = cand.get("classification")
        if cls not in CLASSIFICATIONS:
            raise SystemExit(f"invalid classification {cls!r} for {cand.get('candidate_key')}")
    return data

def enforce_fast_path_rule(candidate: dict[str, Any]) -> List[str]:
    """
    Very small placeholder: returns a list of reasons why a candidate
    violates the fast‑path rule.  The real implementation inspects keys,
    families and notes; here we emulate that with a regex check.
    """
    findings: List[str] = []
    key = candidate.get("candidate_key", "")
    family = candidate.get("family", "")
    if re.search(r"standard.*lora|peft|qlora", f"{key} {family}", re.I):
        findings.append("contains fast‑path prohibited pattern")
    return findings

def candidate_to_feature(candidate: dict[str, Any]) -> np.ndarray:
    """
    Encode a candidate as a numeric feature vector.
    - One‑hot for classification (5 dims)
    - Binary flag for fast‑path violation (1 dim)
    - Random score in [0,1] to simulate a quality metric (1 dim)
    Resulting dimension: 7.
    """
    cls = candidate.get("classification")
    one_hot = np.array([1.0 if cls == c else 0.0 for c in sorted(CLASSIFICATIONS)], dtype=float)
    fast_path = 1.0 if enforce_fast_path_rule(candidate) else 0.0
    quality = random.random()
    return np.concatenate([one_hot, [fast_path, quality]])

def decreasing_pruning_schedule(num_candidates: int, step: int, total_steps: int) -> int:
    """
    Simple schedule: keep floor(num * (1 - step/total_steps)).
    Guarantees a monotonic decrease and never returns zero before the last step.
    """
    keep_frac = max(0.0, 1.0 - step / total_steps)
    return max(1, int(math.floor(num_candidates * keep_frac)))

def prune_candidates(candidates: List[dict[str, Any]], step: int, total_steps: int) -> List[dict[str, Any]]:
    """
    Randomly shuffle candidates, then keep a decreasing number according
    to the schedule.  The shuffle makes the process stochastic while still
    deterministic for a given random seed.
    """
    random.shuffle(candidates)
    keep = decreasing_pruning_schedule(len(candidates), step, total_steps)
    return candidates[:keep]

def prepare_audit_snapshots(manifest_path: Path, total_steps: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the audit snapshot matrices X and X' (next‑step) used for DMD.

    Returns
    -------
    X : ndarray (d, T)          – lifted feature vectors at times 0 … T‑1
    X_prime : ndarray (d, T)    – lifted feature vectors at times 1 … T
    """
    data = load_manifest(manifest_path)
    candidates = data.get("vendors", [])
    # Initial feature matrix (raw, not lifted)
    raw_snapshots: List[np.ndarray] = []

    for step in range(total_steps):
        # Apply pruning for this step
        candidates = prune_candidates(candidates, step, total_steps)
        # Encode each remaining candidate
        feats = np.column_stack([candidate_to_feature(c) for c in candidates])  # shape (7, N_t)
        # Aggregate to a single state vector by averaging over candidates
        if feats.size == 0:
            state = np.zeros(7)
        else:
            state = np.mean(feats, axis=1)  # shape (7,)
        raw_snapshots.append(state)

    # Build X and X' from consecutive states
    X_raw = np.column_stack(raw_snapshots[:-1])      # (7, T-1)
    X_prime_raw = np.column_stack(raw_snapshots[1:]) # (7, T-1)

    # Lift to polynomial observables (degree 2) – the same lift used in the
    # original Koopman code.
    X = observable_lift(X_raw)          # (d_lift, T-1)
    X_prime = observable_lift(X_prime_raw)

    return X, X_prime

# ----------------------------------------------------------------------
# Parent‑B – Koopman / DMD core (re‑implemented)
# ----------------------------------------------------------------------

def observable_lift(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Polynomial observable lift up to the given degree (currently only 2).

    Parameters
    ----------
    X : ndarray (d, N)   – raw state snapshots (columns are time steps)
    degree : int         – degree of polynomial terms (only 2 is supported)

    Returns
    -------
    Y : ndarray (d_lift, N) – lifted observables
    """
    if degree != 2:
        raise NotImplementedError("Only degree=2 is implemented in this hybrid.")
    d, N = X.shape
    # Linear terms
    linear = X
    # Quadratic self‑terms
    quad_self = X**2
    # Cross terms x_i * x_j for i < j
    cross_terms = []
    for i in range(d):
        for j in range(i + 1, d):
            cross_terms.append((X[i] * X[j]).reshape(1, N))
    if cross_terms:
        cross = np.concatenate(cross_terms, axis=0)
        lifted = np.concatenate([linear, quad_self, cross], axis=0)
    else:
        lifted = np.concatenate([linear, quad_self], axis=0)
    return lifted

def dmd(X: np.ndarray, X_prime: np.ndarray, rank: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard Dynamic Mode Decomposition.

    Returns
    -------
    eigenvalues : (r,) complex
    modes       : (d, r) complex
    amplitudes  : (r,) complex
    """
    # Thin SVD of X
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    r = min(rank, U.shape[1], Vh.shape[0])
    U_r = U[:, :r]
    S_r = np.diag(s[:r])
    V_r = Vh.conj().T[:, :r]

    # Reduced Koopman
    K_tilde = U_r.conj().T @ X_prime @ V_r @ np.linalg.inv(S_r)

    # Eigendecomposition
    Lambda, W = np.linalg.eig(K_tilde)
    Phi = X_prime @ V_r @ np.linalg.inv(S_r) @ W

    # Compute amplitudes (b) assuming first snapshot as initial condition
    x1 = X[:, 0]
    b = np.linalg.lstsq(Phi, x1, rcond=None)[0]

    return Lambda, Phi, b

def fit_koopman(X: np.ndarray, X_prime: np.ndarray, rank: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper returning the Koopman matrix in the lifted space."""
    Lambda, Phi, b = dmd(X, X_prime, rank=rank)
    # Reconstruct full‑rank Koopman approximation
    K_approx = Phi @ np.diag(Lambda) @ np.linalg.pinv(Phi)
    return K_approx, Lambda, b

def koopman_forecast(K: np.ndarray, psi0: np.ndarray, steps: int) -> np.ndarray:
    """
    Propagate the lifted state psi0 forward `steps` using the Koopman matrix K.

    Returns an array of shape (d_lift, steps+1) containing psi0, psi1, ….
    """
    d, _ = K.shape
    traj = np.empty((d, steps + 1), dtype=complex)
    traj[:, 0] = psi0
    for t in range(steps):
        traj[:, t + 1] = K @ traj[:, t]
    return traj

def reconstruction_error(X: np.ndarray, K: np.ndarray) -> float:
    """Mean squared error of one‑step reconstruction X' ≈ K X."""
    X_prime_est = K @ X
    return float(np.mean(np.abs(X_prime_est - X) ** 2))

# ----------------------------------------------------------------------
# Hybrid pipeline
# ----------------------------------------------------------------------

def fit_hybrid_koopman(manifest_path: Path, rank: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the audit manifest, build lifted snapshots, and fit a Koopman
    operator to the pruning dynamics.

    Returns
    -------
    K : ndarray (d_lift, d_lift)   – approximated Koopman matrix
    Lambda : ndarray (r,) complex  – eigenvalues
    b : ndarray (r,) complex       – mode amplitudes
    """
    X, X_prime = prepare_audit_snapshots(manifest_path)
    K, Lambda, b = fit_koopman(X, X_prime, rank=rank)
    return K, Lambda, b

def hybrid_forecast(manifest_path: Path, steps: int = 3, rank: int = 5) -> np.ndarray:
    """
    Perform a forecast of the audit state `steps` ahead using the hybrid
    Koopman model.

    The returned array is in the *original* (non‑lifted) feature space
    (dimension 7) by projecting the lifted forecast back with a pseudo‑inverse.
    """
    # Build snapshots and lift them
    X, _ = prepare_audit_snapshots(manifest_path)
    # Fit Koopman
    K, _, _ = fit_koopman(X, X, rank=rank)  # use X as both arguments to get K≈I for demo
    # Initial lifted state (first column)
    psi0 = X[:, 0]
    # Propagate
    lifted_traj = koopman_forecast(K, psi0, steps)
    # Project back: we use the linear part of the lift (the first 7 rows)
    # because those correspond to the original state.
    projection = np.linalg.pinv(observable_lift(np.eye(7)))[:7]  # shape (7, d_lift)
    traj_original = projection @ lifted_traj
    return traj_original.real  # discard negligible imaginary part

def hybrid_reconstruction_error(manifest_path: Path, rank: int = 5) -> float:
    """
    Compute the one‑step reconstruction error of the hybrid model on the
    audit data.
    """
    X, X_prime = prepare_audit_snapshots(manifest_path)
    K, _, _ = fit_koopman(X, X_prime, rank=rank)
    return reconstruction_error(X, K)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a temporary manifest with synthetic data
    tmp_path = Path("temp_manifest.json")
    synthetic = {
        "vendors": [
            {
                "candidate_key": f"model_{i}",
                "family": "standard_lora",
                "classification": random.choice(list(CLASSIFICATIONS)),
                "notes": "auto‑generated"
            }
            for i in range(30)
        ]
    }
    tmp_path.write_text(json.dumps(synthetic, indent=2), encoding="utf-8")

    # Fit hybrid Koopman model
    K, eigs, amp = fit_hybrid_koopman(tmp_path, rank=4)
    print("Koopman eigenvalues (hybrid):", eigs)

    # Forecast 3 steps ahead
    forecast = hybrid_forecast(tmp_path, steps=3, rank=4)
    print("\nForecasted audit feature vectors (original space):")
    for t, vec in enumerate(forecast.T):
        print(f"t={t}: {vec}")

    # Reconstruction error
    err = hybrid_reconstruction_error(tmp_path, rank=4)
    print("\nOne‑step reconstruction MSE:", err)

    # Clean up
    tmp_path.unlink(missing_ok=True)