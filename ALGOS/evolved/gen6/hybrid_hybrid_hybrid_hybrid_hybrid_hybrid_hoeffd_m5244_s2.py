# DARWIN HAMMER — match 5244, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1822_s0.py (gen5)
# parent_b: hybrid_hybrid_hoeffding_tre_hybrid_hybrid_gliner_m220_s0.py (gen4)
# born: 2026-05-30T00:01:02Z

import math
import random
import time
import re
from dataclasses import dataclass, field
from typing import Tuple, List

import numpy as np

# ----------------------------------------------------------------------
# Regex‑based sparse feature extraction (Parent A)
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
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i)\b",
    re.I,
)


def gaussian_kernel(dist: float, sigma: float = 1.0) -> float:
    """Gaussian kernel with bandwidth sigma, returns value in (0,1]."""
    return math.exp(-0.5 * (dist / sigma) ** 2)


def regex_feature_extraction(text: str) -> np.ndarray:
    """Return a 5‑dimensional sparse feature vector."""
    evidence = len(EVIDENCE_RE.findall(text))
    planning = len(PLANNING_RE.findall(text))
    delay = len(DELAY_RE.findall(text))
    # Two placeholder dimensions for future extensions (currently zero)
    return np.array([evidence, planning, delay, 0.0, 0.0], dtype=float)


def similarity(prev_vec: np.ndarray, cur_vec: np.ndarray, sigma: float = 1.0) -> float:
    """Gaussian similarity between two feature vectors."""
    diff_norm = np.linalg.norm(cur_vec - prev_vec)
    return gaussian_kernel(diff_norm, sigma)


# ----------------------------------------------------------------------
# Diffusion‑forced LTC surrogate (Parent A)
# ----------------------------------------------------------------------
class DiffusionLTC:
    """
    Very light‑weight Long‑Term‑Current (LTC) surrogate.
    State `h` evolves as   h_{t+1} = (1-α)·h_t + α·λ(x_t)
    where λ is a diffusion forcing term.
    """

    def __init__(self, dim: int, n_centers: int = 8, alpha: float = 0.3, sigma: float = 2.0):
        self.dim = dim
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        # Randomly initialise RBF centres and isotropic weights
        self.centers = np.random.randn(n_centers, dim)
        self.weights = np.abs(np.random.randn(n_centers))  # non‑negative
        self.h = np.zeros(dim, dtype=float)

    def diffusion_lambda(self, x_t: np.ndarray) -> np.ndarray:
        """λ(x_t) = Σ_i w_i * exp(-||x_t-c_i||² / (2σ²))."""
        diffs = x_t - self.centers  # (m, d)
        sq_norms = np.einsum("ij,ij->i", diffs, diffs)
        rbf_vals = np.exp(-sq_norms / (2.0 * self.sigma ** 2))
        return np.dot(self.weights, rbf_vals) * np.ones(self.dim)

    def step(self, x_t: np.ndarray, confidence: float) -> np.ndarray:
        """
        Update internal state using confidence‑scaled diffusion.
        Higher confidence → stronger influence of the diffusion term.
        """
        λ = self.diffusion_lambda(x_t)
        # Confidence modulates the mixing coefficient α dynamically
        dyn_alpha = self.alpha * confidence
        self.h = (1.0 - dyn_alpha) * self.h + dyn_alpha * λ
        return self.h.copy()


# ----------------------------------------------------------------------
# Sparse Winner‑Take‑All (WTA) activation (Parent A)
# ----------------------------------------------------------------------
def sparse_wta_activation(vector: np.ndarray, k: int, confidence: float) -> np.ndarray:
    """Keep top‑k entries, scale them by confidence, zero‑out the rest."""
    if k <= 0:
        return np.zeros_like(vector)
    # argpartition gives unsorted top‑k indices; we keep them as‑is
    idx = np.argpartition(vector, -k)[-k:]
    activated = np.zeros_like(vector)
    activated[idx] = vector[idx] * confidence
    return activated


# ----------------------------------------------------------------------
# Hoeffding bound with Gini regularisation (Parent B)
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int, gini_coeff: float) -> float:
    """
    Hoeffding bound with an additive Gini regularisation term.
    The regularisation is scaled by π/6 to keep it comparable to the
    variance term.
    """
    if r <= 0 or not (0.0 < delta < 1.0) or n <= 0:
        raise ValueError("Invalid Hoeffding parameters")
    reg = gini_coeff * math.pi / 6.0
    return math.sqrt((r * r * math.log(1.0 / delta) + reg) / (2.0 * n))


@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


def should_split(
    best_gain: float,
    second_best_gain: float,
    r: float,
    delta: float,
    n: int,
    gini_coeff: float,
    tie_threshold: float = 0.05,
) -> SplitDecision:
    """Return a decision based on Hoeffding bound with confidence‑aware Gini."""
    eps = hoeffding_bound(r, delta, n, gini_coeff)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    if gap > eps:
        reason = "gap_exceeds_bound"
    elif eps < tie_threshold:
        reason = "tight_bound"
    else:
        reason = "await_more_data"
    return SplitDecision(split, eps, gap, reason)


# ----------------------------------------------------------------------
# Pheromone handling (Parent B)
# ----------------------------------------------------------------------
class PheromoneEntry:
    """Single pheromone trace with exponential decay."""

    __slots__ = (
        "uuid",
        "surface_key",
        "signal_kind",
        "signal_value",
        "half_life_seconds",
        "created_at",
        "last_decay",
    )

    def __init__(self, surface_key: str, signal_kind: str, signal_value: float, half_life_seconds: int):
        self.uuid = f"{random.getrandbits(128):032x}"
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = max(1, half_life_seconds)  # enforce >0
        self.created_at = time.time()
        self.last_decay = self.created_at

    def decay_factor(self, current_time: float) -> float:
        """Return exponential decay factor since last update."""
        elapsed = current_time - self.last_decay
        if elapsed < 0:
            elapsed = 0.0
        factor = 0.5 ** (elapsed / self.half_life_seconds)
        self.last_decay = current_time
        return factor

    def apply_decay(self, current_time: float) -> None:
        """Decay the stored signal value in‑place."""
        factor = self.decay_factor(current_time)
        self.signal_value *= factor


# ----------------------------------------------------------------------
# Fusion layer – deeper integration
# ----------------------------------------------------------------------
def confidence_modulated_gini(base_gini: float, confidence: float) -> float:
    """
    Map confidence ∈ (0,1] to a Gini scaling factor.
    We use a concave mapping so that low confidence heavily shrinks the Gini term,
    while high confidence keeps it near the base value.
    """
    # Ensure confidence is bounded away from zero for numerical stability
    conf = max(1e-6, min(1.0, confidence))
    # Example mapping: g_eff = base_gini * (1 - β·(1‑conf)) with β∈[0,1]
    beta = 0.7
    return max(1e-6, min(1.0, base_gini * (1.0 - beta * (1.0 - conf))))


def adapt_half_life(base_half_life: int, confidence: float, min_half_life: int = 5) -> int:
    """
    Increase half‑life with confidence (slower decay).
    Guarantees a minimum half‑life to avoid division‑by‑zero.
    """
    conf = max(1e-6, min(1.0, confidence))
    adapted = int(base_half_life / (conf + 1e-6))
    return max(min_half_life, adapted)


class HybridModel:
    """
    End‑to‑end hybrid model that fuses:
      • Regex‑driven sparse features (Parent A)
      • Diffusion‑forced LTC surrogate (Parent A)
      • Confidence‑scaled Gini regularisation (Parent B)
      • Confidence‑aware pheromone dynamics (Parent B)
      • Sparse WTA activation (Parent A)
    """

    def __init__(
        self,
        sigma_conf: float = 1.0,
        wta_k: int = 2,
        gini_base: float = 0.5,
        base_half_life: int = 30,
        r: float = 1.0,
        delta: float = 0.05,
    ):
        self.sigma_conf = sigma_conf
        self.wta_k = wta_k
        self.gini_base = gini_base
        self.base_half_life = base_half_life
        self.r = float(r)
        self.delta = float(delta)

        self.prev_feat = np.zeros(5, dtype=float)
        self.ltc = DiffusionLTC(dim=5)
        self.n_samples = 0

    def _generate_random_gains(self) -> Tuple[float, float]:
        """Generate synthetic gains ensuring best ≥ second."""
        g1 = random.random()
        g2 = random.random() * g1  # second ≤ first
        return g1, g2

    def step(self, cur_text: str, surface_key: str = "default") -> Tuple[SplitDecision, np.ndarray, PheromoneEntry]:
        """
        Execute one hybrid iteration:
          1. Extract sparse features.
          2. Compute confidence C.
          3. Update LTC state with confidence‑scaled diffusion.
          4. Apply sparse WTA on LTC output.
          5. Compute confidence‑modulated Gini and Hoeffding split decision.
          6. Create / update pheromone entry whose half‑life depends on C.
        Returns (split_decision, activated_vector, pheromone_entry).
        """
        # 1. Feature extraction
        cur_feat = regex_feature_extraction(cur_text)

        # 2. Confidence scalar (Gaussian similarity)
        C = similarity(self.prev_feat, cur_feat, sigma=self.sigma_conf)

        # 3. Diffusion‑forced LTC update (state incorporates confidence)
        ltc_out = self.ltc.step(cur_feat, confidence=C)

        # 4. Sparse WTA activation on LTC output
        activated = sparse_wta_activation(ltc_out, k=self.wta_k, confidence=C)

        # 5. Confidence‑aware Gini and Hoeffding split decision
        gini_eff = confidence_modulated_gini(self.gini_base, C)
        best_gain, second_gain = self._generate_random_gains()
        self.n_samples += 1
        split_decision = should_split(
            best_gain=best_gain,
            second_best_gain=second_gain,
            r=self.r,
            delta=self.delta,
            n=self.n_samples,
            gini_coeff=gini_eff,
        )

        # 6. Pheromone creation / adaptation
        half_life = adapt_half_life(self.base_half_life, C)
        pher = PheromoneEntry(
            surface_key=surface_key,
            signal_kind="confidence",
            signal_value=C,
            half_life_seconds=half_life,
        )
        # Immediately decay according to current timestamp to keep value realistic
        pher.apply_decay(time.time())

        # Update previous feature for next iteration
        self.prev_feat = cur_feat.copy()

        return split_decision, activated, pher

# ----------------------------------------------------------------------
# Example usage (can be removed in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    model = HybridModel()
    prev = np.zeros(5)
    texts = [
        "The plan includes verification of sources and a timeline.",
        "We need to wait for the audit report before proceeding.",
        "Evidence has been confirmed and logged.",
    ]
    for txt in texts:
        decision, vec, pher = model.step(txt, surface_key="example")
        print(f"Text: {txt!r}")
        print(f"  Decision: {decision}")
        print(f"  Activated vector: {vec}")
        print(f"  Pheromone half‑life: {pher.half_life_seconds}s, value: {pher.signal_value:.4f}")
        print("-" * 60)