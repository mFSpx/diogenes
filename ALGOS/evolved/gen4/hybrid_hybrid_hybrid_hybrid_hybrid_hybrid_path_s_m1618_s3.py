# DARWIN HAMMER — match 1618, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hybrid_decisi_fisher_localization_m153_s0.py (gen3)
# parent_b: hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s1.py (gen3)
# born: 2026-05-29T23:37:59Z

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Regex feature sets (Parent A)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted|kill|die|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis)\b",
    re.I,
)


# ----------------------------------------------------------------------
# Helper Functions (Parent A)
# ----------------------------------------------------------------------
def extract_features(text: str) -> np.ndarray:
    """Count occurrences of each regex category and return a float vector."""
    counts = [
        len(EVIDENCE_RE.findall(text)),
        len(PLANNING_RE.findall(text)),
        len(DELAY_RE.findall(text)),
        len(SUPPORT_RE.findall(text)),
        len(BOUNDARY_RE.findall(text)),
        len(OUTCOME_RE.findall(text)),
        len(IMPULSIVE_RE.findall(text)),
        len(SCARCITY_RE.findall(text)),
    ]
    return np.asarray(counts, dtype=float)


def _regularized_inverse(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute a numerically stable inverse using eigen‑value clipping.
    This is more robust than adding a tiny jitter to the diagonal.
    """
    # Symmetrize to avoid numerical asymmetry
    cov = (cov + cov.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Clip eigenvalues to a minimum of eps
    eigvals_clipped = np.clip(eigvals, eps, None)
    inv_cov = eigvecs @ np.diag(1.0 / eigvals_clipped) @ eigvecs.T
    return inv_cov


def fisher_information_score(feature_vec: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Compute the Fisher information I = μᵀ Σ⁻¹ μ using a regularized inverse.
    """
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    inv_cov = _regularized_inverse(cov_matrix)
    return float(feature_vec.T @ inv_cov @ feature_vec)


# ----------------------------------------------------------------------
# Lead‑lag Transform (Parent B) – vectorised implementation
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Convert a T×d path into a (2T‑1)×(2d) lead‑lag matrix.
    Vectorised version eliminates Python loops for speed and clarity.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("Path must be a 2‑D array of shape (T, d).")
    T, d = path.shape

    # Prepare repeated rows
    lead = np.repeat(path, 2, axis=0)[: 2 * T - 1]
    lag = np.empty_like(lead)

    # lag[t] = path[t] for even rows, path[t‑1] for odd rows (except first odd)
    lag[0::2] = path
    lag[1::2] = path[1:]

    # Concatenate lead and lag components
    return np.concatenate([lead, lag], axis=1)


# ----------------------------------------------------------------------
# Pheromone handling (Parent B) – monotonic time & proper decay
# ----------------------------------------------------------------------
def _monotonic_timestamp() -> float:
    """Monotonic timestamp in seconds (high‑resolution)."""
    return time.monotonic()


def decay_pheromone(value: float, half_life: float, elapsed: float) -> float:
    """Exponential decay based on half‑life."""
    if half_life <= 0:
        return 0.0
    return value * math.pow(0.5, elapsed / half_life)


class HybridSystem:
    """Encapsulates pheromone storage and the fused decision pipeline."""

    def __init__(self):
        # surface_key → (value, half_life, last_update)
        self._pheromones: Dict[str, Tuple[float, float, float]] = {}

    # ------------------------------------------------------------------
    # Pheromone API – additive exponential moving average
    # ------------------------------------------------------------------
    def update_pheromone(
        self,
        surface_key: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> None:
        """
        Add a new signal to the pheromone store.
        The existing value is first decayed, then the fresh signal is added.
        This yields an additive exponential moving average rather than a simple overwrite.
        """
        now = _monotonic_timestamp()
        old_val, old_hl, old_ts = self._pheromones.get(
            surface_key, (0.0, half_life_seconds, now)
        )
        elapsed = now - old_ts
        decayed = decay_pheromone(old_val, old_hl, elapsed)
        new_val = decayed + signal_value
        self._pheromones[surface_key] = (new_val, half_life_seconds, now)

    def get_pheromone(self, surface_key: str) -> float:
        """Return the current decayed pheromone value (or 0.0 if missing)."""
        entry = self._pheromones.get(surface_key)
        if entry is None:
            return 0.0
        val, half_life, ts = entry
        elapsed = _monotonic_timestamp() - ts
        return decay_pheromone(val, half_life, elapsed)

    # ------------------------------------------------------------------
    # Covariance estimation – Ledoit‑Wolf shrinkage (more stable)
    # ------------------------------------------------------------------
    @staticmethod
    def _shrinkage_covariance(samples: np.ndarray) -> np.ndarray:
        """
        Compute a shrinkage estimator of the covariance matrix.
        This reduces the risk of singularity for high‑dimensional data.
        """
        # Empirical covariance
        emp_cov = np.cov(samples, rowvar=False, bias=False)

        # Target: scaled identity
        mu = np.trace(emp_cov) / emp_cov.shape[0]
        target = mu * np.eye(emp_cov.shape[0])

        # Ledoit‑Wolf optimal shrinkage coefficient (closed form)
        X = samples - samples.mean(axis=0)
        beta = np.sum(np.square(X), axis=0).sum()
        gamma = np.linalg.norm(emp_cov - target, "fro") ** 2
        kappa = (beta - gamma) / (samples.shape[0] * gamma) if gamma != 0 else 0.0
        shrinkage = max(0.0, min(1.0, kappa))

        return shrinkage * target + (1 - shrinkage) * emp_cov

    # ------------------------------------------------------------------
    # Hybrid decision pipeline – deeper mathematical integration
    # ------------------------------------------------------------------
    def hybrid_decision(
        self,
        text: str,
        path: np.ndarray,
        surface_key: str,
        half_life_seconds: float = 300.0,
        alpha: float = 0.7,
        beta: float = 0.3,
    ) -> float:
        """
        Produce a scalar decision score by tightly coupling the two parent theories:

        1. Extract regex‑based feature vector μ from `text`.
        2. Transform `path` → lead‑lag matrix X.
        3. Estimate a robust covariance Σ̂ from X using shrinkage.
        4. Compute Fisher information I = μᵀ Σ̂⁻¹ μ.
        5. Update pheromone store with I (so the pheromone itself reflects information content).
        6. Retrieve the decayed pheromone P.
        7. Return a convex combination: score = α·I_norm + β·P_norm,
           where each term is normalized to unit variance across recent calls
           (running mean/variance kept per instance).

        The normalization step ensures that α and β truly control relative influence,
        regardless of raw magnitude differences.
        """
        # ---- 1. Feature extraction -------------------------------------------------
        mu = extract_features(text)

        # ---- 2. Lead‑lag transformation --------------------------------------------
        X = lead_lag_transform(path)

        # ---- 3. Covariance estimation -----------------------------------------------
        sigma_hat = self._shrinkage_covariance(X)

        # ---- 4. Fisher information --------------------------------------------------
        I_raw = fisher_information_score(mu, sigma_hat)

        # ---- 5. Pheromone update (injecting information content) --------------------
        self.update_pheromone(surface_key, I_raw, half_life_seconds)

        # ---- 6. Retrieve current pheromone -----------------------------------------
        P_raw = self.get_pheromone(surface_key)

        # ---- 7. Online normalization (running mean/var) ----------------------------
        # Store running statistics lazily
        stats = getattr(self, "_norm_stats", {})
        if surface_key not in stats:
            stats[surface_key] = {
                "mean_I": I_raw,
                "var_I": 1e-6,
                "mean_P": P_raw,
                "var_P": 1e-6,
                "count": 1,
            }
        else:
            s = stats[surface_key]
            s["count"] += 1
            # Welford's algorithm for mean/variance
            delta_I = I_raw - s["mean_I"]
            s["mean_I"] += delta_I / s["count"]
            s["var_I"] += delta_I * (I_raw - s["mean_I"])

            delta_P = P_raw - s["mean_P"]
            s["mean_P"] += delta_P / s["count"]
            s["var_P"] += delta_P * (P_raw - s["mean_P"])

        # Normalized values (add epsilon to avoid division by zero)
        eps = 1e-12
        s = stats[surface_key]
        I_norm = (I_raw - s["mean_I"]) / math.sqrt(s["var_I"] / max(s["count"] - 1, 1) + eps)
        P_norm = (P_raw - s["mean_P"]) / math.sqrt(s["var_P"] / max(s["count"] - 1, 1) + eps)

        # Store back
        self._norm_stats = stats

        # ---- 8. Weighted combination -------------------------------------------------
        score = alpha * I_norm + beta * P_norm
        return float(score)


# ----------------------------------------------------------------------
# CLI entry point (optional, kept for backward compatibility)
# ----------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Fisher‑Pheromone Path Fusion.")
    parser.add_argument("--text", type=str, required=True, help="Input text for regex features.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="JSON‑encoded list of [x1, x2, …] points, each point being a list of floats.",
    )
    parser.add_argument("--key", type=str, required=True, help="Surface key for pheromone tracking.")
    parser.add_argument(
        "--half-life",
        type=float,
        default=300.0,
        help="Half‑life in seconds for pheromone decay.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        path_array = np.asarray(json.loads(args.path), dtype=float)
    except Exception as exc:
        sys.exit(f"Failed to parse path JSON: {exc}")

    system = HybridSystem()
    score = system.hybrid_decision(
        text=args.text,
        path=path_array,
        surface_key=args.key,
        half_life_seconds=args.half_life,
    )
    print(json.dumps({"score": score}))


if __name__ == "__main__":
    main()