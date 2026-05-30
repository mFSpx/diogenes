# DARWIN HAMMER — match 1618, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_decisi_fisher_localization_m153_s0.py (gen3)
# parent_b: hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s1.py (gen3)
# born: 2026-05-29T23:37:59Z

"""
Hybrid Algorithm: Fisher‑Pheromone Path Fusion

Parents:
- hybrid_hybrid_hybrid_decisi_fisher_localization_m153_s0.py (regex feature extraction + Fisher information)
- hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s1.py (lead‑lag path transform + pheromone decay)

Mathematical Bridge:
Both parents operate on high‑dimensional representations of data.
Parent B lifts a sequential path into a lead‑lag space, yielding a matrix X∈ℝ^{(2T‑1)×2d}.
Parent A evaluates the quality of a feature vector μ via the Fisher information
I(μ)=μᵀ Σ⁻¹ μ, where Σ is the covariance of the underlying representation.
The fusion treats the lead‑lag matrix X as the stochastic sample that defines Σ,
computes the Fisher score of the regex‑derived feature vector, and finally
weights this score with a pheromone signal that decays exponentially with a
half‑life. The resulting scalar drives downstream decisions.
"""

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, Path as pathlib_Path
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
    """Count occurrences of each regex category and return a feature vector."""
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


def fisher_information_score(feature_vec: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Compute the Fisher information score I = μᵀ Σ⁻¹ μ.
    If Σ is singular, a small diagonal jitter is added.
    """
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    # Regularize
    jitter = 1e-8 * np.eye(cov_matrix.shape[0])
    inv_cov = np.linalg.inv(cov_matrix + jitter)
    score = float(feature_vec.T @ inv_cov @ feature_vec)
    return score

# ----------------------------------------------------------------------
# Lead‑lag Transform (Parent B)
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Convert a T×d path into a (2T‑1)×(2d) lead‑lag matrix.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("Path must be a 2‑D array of shape (T, d).")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

# ----------------------------------------------------------------------
# Pheromone handling (Parent B)
# ----------------------------------------------------------------------
def current_timestamp() -> float:
    """Return a monotonic timestamp (seconds since epoch)."""
    return pathlib_Path('/proc/self/cmdline').stat().st_ctime

def decay_pheromone(old_value: float, half_life: float, elapsed: float) -> float:
    """Exponential decay based on half‑life."""
    if half_life <= 0:
        return 0.0
    return old_value * math.pow(0.5, elapsed / half_life)

class HybridSystem:
    """Encapsulates pheromone storage and hybrid decision logic."""

    def __init__(self):
        self.pheromones = {}  # surface_key → dict with signal info

    # ------------------------------------------------------------------
    # Pheromone API
    # ------------------------------------------------------------------
    def update_pheromone(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> None:
        """
        Insert or refresh a pheromone entry. Existing entries decay based on elapsed time.
        """
        now = current_timestamp()
        entry = self.pheromones.get(surface_key)
        if entry is None:
            self.pheromones[surface_key] = {
                "signal_kind": signal_kind,
                "signal_value": signal_value,
                "half_life_seconds": half_life_seconds,
                "created_time": now,
            }
            return

        elapsed = now - entry["created_time"]
        decayed = decay_pheromone(entry["signal_value"], entry["half_life_seconds"], elapsed)
        # Replace with new value (could be combined; here we simply overwrite after decay)
        self.pheromones[surface_key] = {
            "signal_kind": signal_kind,
            "signal_value": signal_value + decayed,
            "half_life_seconds": half_life_seconds,
            "created_time": now,
        }

    def get_pheromone(self, surface_key: str) -> float:
        """Return the current (decayed) pheromone value for a key, or 0.0 if absent."""
        entry = self.pheromones.get(surface_key)
        if entry is None:
            return 0.0
        now = current_timestamp()
        elapsed = now - entry["created_time"]
        return decay_pheromone(entry["signal_value"], entry["half_life_seconds"], elapsed)

    # ------------------------------------------------------------------
    # Hybrid decision pipeline
    # ------------------------------------------------------------------
    def hybrid_decision(
        self,
        text: str,
        path: np.ndarray,
        surface_key: str,
        half_life_seconds: float = 300.0,
    ) -> float:
        """
        Produce a scalar decision score by:
        1. Extracting regex‑based features from `text`.
        2. Transforming `path` with lead‑lag and deriving its covariance Σ.
        3. Computing Fisher information I = μᵀ Σ⁻¹ μ.
        4. Fetching (and optionally updating) a pheromone value P for `surface_key`.
        5. Returning a weighted combination: score = α·I + β·P.
        """
        # 1. Feature extraction
        mu = extract_features(text)

        # 2. Lead‑lag transform and covariance
        ll = lead_lag_transform(path)
        # Use rows as samples; compute unbiased covariance (ddof=1)
        if ll.shape[0] < 2:
            # Degenerate case: fallback to identity covariance
            cov = np.eye(mu.shape[0])
        else:
            cov = np.cov(ll, rowvar=False)

        # 3. Fisher information
        fisher_score = fisher_information_score(mu, cov)

        # 4. Pheromone component
        pheromone_val = self.get_pheromone(surface_key)

        # Optional: refresh pheromone with the new Fisher score as signal
        self.update_pheromone(
            surface_key,
            signal_kind="fisher",
            signal_value=fisher_score,
            half_life_seconds=half_life_seconds,
        )

        # 5. Combine (simple linear blend)
        alpha = 0.7
        beta = 0.3
        combined = alpha * fisher_score + beta * pheromone_val
        return combined

# ----------------------------------------------------------------------
# Additional demonstration functions
# ----------------------------------------------------------------------
def simulate_random_path(T: int = 10, d: int = 3) -> np.ndarray:
    """Generate a random walk path of length T in d dimensions."""
    steps = np.random.randn(T, d)
    return np.cumsum(steps, axis=0)


def demo_hybrid():
    """Run a quick demonstration of the hybrid system."""
    hs = HybridSystem()
    sample_text = (
        "The audit confirmed the source and provided evidence. "
        "We plan the next steps and schedule a review tomorrow."
    )
    path = simulate_random_path()
    score = hs.hybrid_decision(sample_text, path, surface_key="demo")
    print(f"Hybrid decision score: {score:.4f}")

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_hybrid()