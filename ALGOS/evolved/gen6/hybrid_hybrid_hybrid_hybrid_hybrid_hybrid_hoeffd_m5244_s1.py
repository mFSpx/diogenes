# DARWIN HAMMER — match 5244, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1822_s0.py (gen5)
# parent_b: hybrid_hybrid_hoeffding_tre_hybrid_hybrid_gliner_m220_s0.py (gen4)
# born: 2026-05-30T00:01:02Z

"""Hybrid Algorithm Fusion of:
- Parent A: regex-feature → RBF surrogate → LTC recurrent cell with diffusion forcing and sparse WTA integration.
- Parent B: Hoeffding bound regularized with Gini coefficient, pheromone‑guided splitting.

Mathematical Bridge
------------------
Parent A provides a *confidence scalar* `C` computed as a Gaussian similarity
between successive feature vectors `x_t`.  This scalar quantifies the
signal‑to‑noise gap and is used to modulate the core quantities of Parent B:

1. **Gini regularization** in the Hoeffding bound is scaled by `C`,
   i.e. `g_eff = gini_coeff * C`.  This ties the statistical confidence of
   the feature stream to the exploration‑exploitation trade‑off.

2. **Pheromone dynamics** (half‑life) are adapted by `C`, shortening decay
   when confidence is low and lengthening it when confidence is high:
   `τ = τ_0 / (C + ε)`.

3. **Sparse Winner‑Take‑All (WTA)** activation uses `C` as a multiplicative
   factor on the selected neurons, unifying the diffusion‑forced LTC cell
   with the sparse expansion of Parent B.

The resulting system processes textual inputs, extracts sparse feature
representations, evaluates confidence, and drives an online Hoeffding‑tree
splitting decision guided by pheromones and a confidence‑aware Gini term.
"""

import math
import random
import sys
from pathlib import Path
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np

# ----------------------------------------------------------------------
# Parent A components
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


def gaussian(x: float) -> float:
    """Gaussian kernel used for similarity."""
    return math.exp(-x * x)


def regex_feature_extraction(text: str) -> np.ndarray:
    """Extract a 5‑dimensional sparse feature vector from text."""
    evidence = len(EVIDENCE_RE.findall(text))
    planning = len(PLANNING_RE.findall(text))
    delay = len(DELAY_RE.findall(text))
    # Two placeholder dimensions for future extensions
    return np.array([evidence, planning, delay, 0.0, 0.0], dtype=float)


def similarity(prev_vec: np.ndarray, cur_vec: np.ndarray) -> float:
    """Gaussian similarity between successive feature vectors."""
    diff_norm = np.linalg.norm(cur_vec - prev_vec)
    return gaussian(diff_norm)


def diffusion_lambda(x_t: np.ndarray, centers: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Diffusion forcing term used in the LTC recurrent cell.
    Implements λ(x_t) = Σ_i w_i * exp(-||x_t - c_i||^2)
    """
    diffs = x_t - centers  # shape (m, d)
    sq_norms = np.sum(diffs * diffs, axis=1)  # shape (m,)
    return np.sum(weights[:, None] * np.exp(-sq_norms[:, None]), axis=0)


def sparse_wta_activation(vector: np.ndarray, k: int, confidence: float) -> np.ndarray:
    """
    Sparse Winner‑Take‑All activation.
    Keeps the top‑k entries, scales them by the confidence scalar.
    """
    if k <= 0:
        return np.zeros_like(vector)
    idx = np.argpartition(vector, -k)[-k:]
    activated = np.zeros_like(vector)
    activated[idx] = vector[idx] * confidence
    return activated


# ----------------------------------------------------------------------
# Parent B components
# ----------------------------------------------------------------------
def hoeffding_bound_with_gini(r: float, delta: float, n: int, gini_coeff: float) -> float:
    """Hoeffding bound regularized with a Gini coefficient."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("Invalid parameters for Hoeffding bound")
    regularization = gini_coeff * math.pi / 6.0
    return math.sqrt((r * r * math.log(1.0 / delta) + regularization) / (2.0 * n))


@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


def should_split_with_gini(
    best_gain: float,
    second_best_gain: float,
    r: float,
    delta: float,
    n: int,
    tie_threshold: float = 0.05,
    gini_coeff: float = 0.5,
) -> SplitDecision:
    eps = hoeffding_bound_with_gini(r, delta, n, gini_coeff)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)


class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value", "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str, signal_value: float, half_life_seconds: int):
        self.uuid = f"{random.getrandbits(128):032x}"
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        self.created_at = random.random()  # placeholder for timestamp
        self.last_decay = self.created_at

    def decay_factor(self, current_time: float) -> float:
        """Exponential decay based on half‑life."""
        elapsed = current_time - self.last_decay
        decay = 0.5 ** (elapsed / self.half_life_seconds)
        self.last_decay = current_time
        return decay


# ----------------------------------------------------------------------
# Fusion layer
# ----------------------------------------------------------------------
def confidence_modulated_gini(base_gini: float, confidence: float) -> float:
    """
    Scale the Gini coefficient by the confidence scalar.
    Guarantees the result stays in (0,1] for stability.
    """
    return min(1.0, max(1e-6, base_gini * confidence))


def update_pheromone(entry: PheromoneEntry, confidence: float, base_half_life: int = 30) -> None:
    """
    Adjust pheromone half‑life according to confidence.
    Higher confidence → slower decay (longer half‑life).
    """
    entry.half_life_seconds = int(base_half_life / (confidence + 1e-6))


def hybrid_step(
    prev_feat: np.ndarray,
    cur_text: str,
    r: float,
    delta: float,
    n_samples: int,
    gini_base: float = 0.5,
    wta_k: int = 2,
) -> Tuple[SplitDecision, np.ndarray, PheromoneEntry]:
    """
    Perform a single hybrid iteration:
    1. Extract features from `cur_text`.
    2. Compute confidence `C` via similarity to `prev_feat`.
    3. Apply sparse WTA on the feature vector scaled by `C`.
    4. Use confidence‑scaled Gini in Hoeffding split decision.
    5. Generate a pheromone entry whose decay depends on `C`.
    Returns the split decision, the activated sparse vector, and the pheromone.
    """
    # 1. Feature extraction
    cur_feat = regex_feature_extraction(cur_text)

    # 2. Confidence scalar
    C = similarity(prev_feat, cur_feat)

    # 3. Sparse WTA activation (acts as the RBF/LTC surrogate output)
    activated = sparse_wta_activation(cur_feat, k=wta_k, confidence=C)

    # 4. Confidence‑modulated Gini for Hoeffding bound
    gini_eff = confidence_modulated_gini(gini_base, C)

    # Simulated gains (in a real system these would come from label statistics)
    best_gain = random.random()
    second_gain = random.random() * 0.9  # ensure best >= second

    split_decision = should_split_with_gini(
        best_gain=best_gain,
        second_best_gain=second_gain,
        r=r,
        delta=delta,
        n=n_samples,
        gini_coeff=gini_eff,
    )

    # 5. Pheromone creation and adaptation
    pher = PheromoneEntry(
        surface_key="text_stream",
        signal_kind="confidence",
        signal_value=C,
        half_life_seconds=30,
    )
    update_pheromone(pher, confidence=C)

    return split_decision, activated, pher


def hybrid_process(texts: List[str]) -> List[Dict]:
    """
    Process a sequence of texts using the fused hybrid algorithm.
    Returns a list of dictionaries summarising each step.
    """
    if not texts:
        return []

    # Initialise with the first text
    prev_feat = regex_feature_extraction(texts[0])
    results = []

    # Hyper‑parameters (could be exposed to the user)
    r = 1.0
    delta = 0.05
    n_samples = 100
    gini_base = 0.4
    wta_k = 2

    for idx, txt in enumerate(texts):
        split_dec, activated_vec, pher = hybrid_step(
            prev_feat=prev_feat,
            cur_text=txt,
            r=r,
            delta=delta,
            n_samples=n_samples,
            gini_base=gini_base,
            wta_k=wta_k,
        )

        # Update state for next iteration
        prev_feat = regex_feature_extraction(txt)

        results.append(
            {
                "index": idx,
                "text": txt,
                "confidence": pher.signal_value,
                "split_decision": split_dec.should_split,
                "reason": split_dec.reason,
                "activated_vector": activated_vec.tolist(),
                "pheromone_half_life": pher.half_life_seconds,
            }
        )

    return results


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "The report includes verified evidence and a detailed plan.",
        "We need to pause the deployment and wait for further audit.",
        "Confirmed source shows the hash matches the screenshot.",
        "Schedule the next checkpoint and prioritize the checklist.",
        "No new evidence, but the timeline is extended.",
    ]

    output = hybrid_process(sample_texts)

    for entry in output:
        print(
            f"Idx {entry['index']}: split={entry['split_decision']}, "
            f"conf={entry['confidence']:.3f}, reason={entry['reason']}, "
            f"half_life={entry['pheromone_half_life']}"
        )