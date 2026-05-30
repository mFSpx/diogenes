# DARWIN HAMMER — match 4906, survivor 7
# gen: 6
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_ternar_m1557_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_regret_engine_m2_s4.py (gen3)
# born: 2026-05-29T23:58:54Z

import sys
import math
import random
import hashlib
import re
from pathlib import Path
from typing import List, Iterable, Tuple, Dict, Optional

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Audit risk, MinHash, Ternary routing
# ----------------------------------------------------------------------


def audit_risk(findings: Iterable[str]) -> float:
    """
    Compute audit risk as the proportion of findings that contain a
    high‑severity keyword.  Returns a value in [0, 1].
    """
    keywords = {"critical", "high", "risk", "issue", "vulnerability"}
    findings = list(findings)
    total = len(findings)
    if total == 0:
        return 0.0
    flagged = sum(
        1 for f in findings if any(k in f.lower() for k in keywords)
    )
    return flagged / total


def _hash_token(token: bytes, seed: int) -> int:
    """
    Deterministic hash of a token using Blake2b with a per‑seed
    personalization string.  Returns a 64‑bit unsigned integer.
    """
    h = hashlib.blake2b(token, person=seed.to_bytes(4, "little"))
    return int.from_bytes(h.digest()[:8], "big", signed=False)


def minhash_signature(tokens: Iterable[str], k: int = 64) -> np.ndarray:
    """
    Compute a simple MinHash signature of length *k*.
    The result is a float64 array normalised to [0, 1].
    """
    seeds = np.arange(k, dtype=np.uint32)
    # Initialise with the maximal possible 64‑bit value.
    min_hashes = np.full(k, np.iinfo(np.uint64).max, dtype=np.uint64)

    for token in tokens:
        token_bytes = token.encode("utf-8")
        for i, seed in enumerate(seeds):
            h = _hash_token(token_bytes, int(seed))
            if h < min_hashes[i]:
                min_hashes[i] = h

    # Normalise to the unit interval for stable dot‑products.
    return min_hashes.astype(np.float64) / np.iinfo(np.uint64).max


class TernaryRouter:
    """
    Holds a fixed set of ternary configuration vectors and selects the one
    maximising the dot‑product with a supplied weighted signature.
    """

    def __init__(self, num_outputs: int = 8, dim: int = 64, seed: int = 42):
        self.num_outputs = num_outputs
        self.dim = dim
        self._rng = np.random.default_rng(seed)
        self.configs = self._generate_configs()

    def _generate_configs(self) -> np.ndarray:
        """Matrix T of shape (num_outputs, dim) with entries in {‑1, 0, 1}."""
        return self._rng.integers(-1, 2, size=(self.num_outputs, self.dim), dtype=np.int8)

    def select(self, weighted_sig: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Return the index of the best configuration and its one‑hot encoding.
        """
        scores = self.configs @ weighted_sig
        idx = int(np.argmax(scores))
        one_hot = np.zeros(self.num_outputs, dtype=np.int8)
        one_hot[idx] = 1
        return idx, one_hot


def voronoi_aggregation(one_hot: np.ndarray, aux_vector: np.ndarray) -> float:
    """
    Perform a scalar aggregation of an auxiliary vector *x* with the chosen
    ternary pattern.  The result is a single float.
    """
    # one_hot is (num_outputs,) and aux_vector must have the same length.
    return float(np.dot(one_hot, aux_vector))


def circuit_breaker(agg: float, risk: float) -> float:
    """
    Final “circuit‑breaker” term.  It scales the aggregated value by the audit
    risk and returns a scalar.
    """
    return agg * risk


# ----------------------------------------------------------------------
# Parent B – Feature extraction, utility, regret‑softmax, Gini
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


def extract_feature_counts(text: str) -> Dict[str, int]:
    """
    Count occurrences of four regex‑based feature families in *text*.
    """
    return {
        "evidence": len(EVIDENCE_RE.findall(text)),
        "planning": len(PLANNING_RE.findall(text)),
        "delay": len(DELAY_RE.findall(text)),
        "support": len(SUPPORT_RE.findall(text)),
    }


def compute_utility(
    counts: Dict[str, int],
    pos_weights: Dict[str, float],
    neg_weights: Dict[str, float],
) -> np.ndarray:
    """
    Utility vector *u* = p·c – n·c where *c* are feature counts.
    The order follows the iteration order of *counts* (deterministic in Python 3.7+).
    """
    return np.array(
        [
            pos_weights[k] * counts[k] - neg_weights[k] * counts[k]
            for k in counts
        ],
        dtype=np.float64,
    )


def regret_softmax(u: np.ndarray) -> np.ndarray:
    """
    Regret‑weighted softmax.  Subtract the maximum for numerical stability,
    then normalise to a probability distribution.
    """
    if u.size == 0:
        return np.array([], dtype=np.float64)
    shifted = u - np.max(u)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


def gini_coefficient(probs: np.ndarray) -> float:
    """
    Gini coefficient of a probability distribution *probs*.
    Returns 0 for a uniform distribution and 1 for a degenerate one.
    """
    if probs.size == 0:
        return 0.0
    sorted_probs = np.sort(probs)
    n = probs.size
    cumulative = np.cumsum(sorted_probs)
    # The classic Gini formula adapted for a probability vector.
    gini = 1.0 - (2.0 / (n - 1)) * (np.sum(cumulative) / np.sum(sorted_probs) - (n + 1) / 2.0)
    return float(gini)


# ----------------------------------------------------------------------
# Deep Fusion – Bridging the two worlds
# ----------------------------------------------------------------------


class DeepFusion:
    """
    A tighter mathematical integration of the MinHash‑router pipeline (Parent A)
    with the regret‑softmax pipeline (Parent B).

    The key ideas are:

    1. **Projection** – a learned (or preset) linear map `W` projects the
       weighted MinHash signature into the *utility space* (dimension 4).
    2. **Regret‑aware routing** – the Gini coefficient of the regret
       distribution is fed back as a bias term when selecting a ternary
       configuration.
    3. **Joint scoring** – the final hybrid score combines the circuit‑breaker
       term and the regret measure after normalising both to comparable scales.
    """

    def __init__(
        self,
        num_router_outputs: int = 8,
        minhash_dim: int = 64,
        utility_dim: int = 4,
        alpha: float = 0.5,
        beta: float = 0.5,
        seed: int = 123,
    ):
        self.alpha = alpha
        self.beta = beta
        self.minhash_dim = minhash_dim
        self.utility_dim = utility_dim

        # Projection matrix W : utility_dim × minhash_dim.
        rng = np.random.default_rng(seed)
        self.W = rng.normal(loc=0.0, scale=1.0, size=(utility_dim, minhash_dim)).astype(
            np.float64
        )

        # Ternary router works on the original MinHash space.
        self.router = TernaryRouter(
            num_outputs=num_router_outputs, dim=minhash_dim, seed=seed
        )

    def _project_to_utility(self, weighted_sig: np.ndarray) -> np.ndarray:
        """
        Linear projection of the weighted MinHash signature into the utility
        space.  The result is a dense utility‑like vector that can be fed to
        the regret softmax.
        """
        return self.W @ weighted_sig  # shape (utility_dim,)

    def _regret_bias(self, probs: np.ndarray) -> float:
        """
        Convert the Gini coefficient into a scalar bias that nudges the router
        scores.  Larger inequality (higher Gini) yields a larger positive bias.
        """
        g = gini_coefficient(probs)
        # Scale to a modest range to avoid overwhelming the dot‑product.
        return 0.1 * g

    def compute_hybrid_score(
        self,
        text: str,
        audit_findings: List[str],
        aux_vector: Optional[np.ndarray] = None,
        pos_weights: Optional[Dict[str, float]] = None,
        neg_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        End‑to‑end computation of the hybrid score.

        Returns
        -------
        hybrid_score : float
            Weighted combination α·B̂ + β·Ĝ where B̂ and Ĝ are normalised.
        diagnostics : dict
            Intermediate values useful for debugging / inspection.
        """
        # ------------------------------------------------------------------
        # 1️⃣  MinHash + audit risk
        # ------------------------------------------------------------------
        tokens = re.findall(r"\w+", text.lower())
        s = minhash_signature(tokens, k=self.minhash_dim)          # raw signature
        r = audit_risk(audit_findings)                             # scalar risk
        s_w = r * s                                                 # weighted signature

        # ------------------------------------------------------------------
        # 2️⃣  Project to utility space and obtain regret distribution
        # ------------------------------------------------------------------
        u_proj = self._project_to_utility(s_w)                     # shape (4,)
        probs = regret_softmax(u_proj)                             # probability vector
        G = gini_coefficient(probs)                                # regret measure

        # ------------------------------------------------------------------
        # 3️⃣  Regret‑aware ternary routing
        # ------------------------------------------------------------------
        bias = self._regret_bias(probs)
        # Add bias uniformly to all router scores (equivalent to shifting the
        # argmax decision without changing relative ordering).
        scores = self.router.configs @ s_w + bias
        router_idx = int(np.argmax(scores))
        one_hot = np.zeros(self.router.num_outputs, dtype=np.int8)
        one_hot[router_idx] = 1

        # ------------------------------------------------------------------
        # 4️⃣  Circuit‑breaker term
        # ------------------------------------------------------------------
        if aux_vector is None:
            # Default auxiliary vector: simple identity scaling.
            aux_vector = np.ones(self.router.num_outputs, dtype=np.float64)
        else:
            aux_vector = np.asarray(aux_vector, dtype=np.float64)
            if aux_vector.shape != (self.router.num_outputs,):
                raise ValueError(
                    f"aux_vector must have shape ({self.router.num_outputs},)"
                )
        V = voronoi_aggregation(one_hot, aux_vector)              # scalar
        B = circuit_breaker(V, r)                                  # scalar

        # ------------------------------------------------------------------
        # 5️⃣  Normalisation of components
        # ------------------------------------------------------------------
        # Normalise B to [0, 1] using a sigmoid; this keeps the scale comparable
        # to G, which already lies in [0, 1].
        B_norm = 1.0 / (1.0 + math.exp(-B))
        G_norm = G  # already in [0, 1]

        hybrid_score = self.alpha * B_norm + self.beta * G_norm

        diagnostics = {
            "audit_risk": r,
            "minhash_weighted_norm": float(np.linalg.norm(s_w)),
            "projected_utility": u_proj.tolist(),
            "regret_probs": probs.tolist(),
            "gini": G,
            "router_index": router_idx,
            "router_bias": bias,
            "circuit_breaker_raw": B,
            "circuit_breaker_norm": B_norm,
            "hybrid_score": hybrid_score,
        }

        return hybrid_score, diagnostics


# ----------------------------------------------------------------------
# Example usage (can be removed or guarded by __name__ == "__main__")
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The audit revealed several critical issues. Evidence was logged, "
        "but the plan to remediate is delayed due to resource constraints."
    )
    findings = [
        "Critical vulnerability in module X",
        "High risk of data leakage",
        "Low severity warning",
    ]

    # Positive and negative weights for the four feature families.
    pos_w = {"evidence": 1.0, "planning": 0.8, "delay": -0.2, "support": 0.5}
    neg_w = {"evidence": -0.5, "planning": -0.3, "delay": 1.0, "support": -0.1}

    fusion = DeepFusion(num_router_outputs=8, minhash_dim=64, utility_dim=4)
    score, details = fusion.compute_hybrid_score(
        text=sample_text,
        audit_findings=findings,
        aux_vector=np.arange(8, dtype=np.float64),  # example auxiliary vector
        pos_weights=pos_w,
        neg_weights=neg_w,
    )
    print(f"Hybrid score: {score:.4f}")
    for k, v in details.items():
        print(f"{k}: {v}")