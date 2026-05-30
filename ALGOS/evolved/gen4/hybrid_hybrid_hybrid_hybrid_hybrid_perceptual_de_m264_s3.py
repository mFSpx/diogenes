# DARWIN HAMMER — match 264, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_liquid_m39_s0.py (gen3)
# parent_b: hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s5.py (gen3)
# born: 2026-05-29T23:27:55Z

"""
Hybrid Algorithm: regex‑feature → RBF surrogate → LTC recurrent cell with diffusion forcing

Parents:
* Parent A – regex feature extraction & Liquid‑Time‑Constant (LTC) recurrent cell with
  diffusion forcing.
* Parent B – Radial‑Basis Function (RBF) surrogate model + perceptual‑hash deduplication.

Mathematical Bridge
-------------------
1. **Feature Vector** – Text is parsed with the regexes from Parent A producing a
   5‑dimensional count vector **x**.
2. **Similarity α** – The similarity between successive vectors is computed with the
   same Gaussian RBF used in Parent B:
   `α = gaussian(‖x_t‑x_{t‑1}‖)`.  This α plays the role of the liquid‑time‑constant
   mixing coefficient in the LTC update.
3. **Diffusion λ** – The RBF surrogate from Parent B predicts a scalar diffusion
   coefficient `λ = f_RBF(x_t)`.  The surrogate is built from fixed centres and
   weights; its output modulates the stochastic forcing term of the LTC cell.
4. **Dedupe** – A lightweight perceptual hash (binary pattern derived from the
   median of the vector) identifies near‑duplicate inputs; duplicates are skipped
   so the LTC state is only updated on novel feature patterns.

Thus the hybrid evolves as  


h_{t+1} = (1-α)·h_t + α·tanh(W·x_t + U·h_t + b) + λ·η_t


where `η_t ~ N(0,1)` is Gaussian noise.  All components are pure NumPy / std‑lib,
fulfilling the fusion requirements.
"""

import re
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Dict

import numpy as np

# ----------------------------------------------------------------------
# Regex feature extraction (Parent A)
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


def extract_feature_vector(text: str) -> np.ndarray:
    """Return a 5‑dim count vector for the regex categories."""
    counts = [
        len(EVIDENCE_RE.findall(text)),
        len(PLANNING_RE.findall(text)),
        len(DELAY_RE.findall(text)),
        len(SUPPORT_RE.findall(text)),
        len(BOUNDARY_RE.findall(text)),
    ]
    return np.array(counts, dtype=float)


# ----------------------------------------------------------------------
# Perceptual hash deduplication (Parent B)
# ----------------------------------------------------------------------
def perceptual_hash(vec: np.ndarray, bits: int = 8) -> int:
    """
    Simple perceptual hash: compare each component to the median value.
    Returns an integer whose binary representation encodes the pattern.
    """
    if vec.size == 0:
        return 0
    median = np.median(vec)
    pattern = (vec > median).astype(int)
    # pad / truncate to `bits` length
    if pattern.size < bits:
        pattern = np.pad(pattern, (0, bits - pattern.size), constant_values=0)
    else:
        pattern = pattern[:bits]
    # pack bits into an int
    h = 0
    for bit in pattern:
        h = (h << 1) | int(bit)
    return h


# ----------------------------------------------------------------------
# Radial‑Basis Function surrogate (Parent B)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


@dataclass(frozen=True)
class RBFSurrogate:
    """RBF surrogate model."""
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Sequence[float]) -> float:
        """Predict a scalar using Gaussian RBFs."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )


# Pre‑defined surrogate (dummy training data)
# Centers are spread in the 5‑D feature space; weights are arbitrary.
_RBF_SURROGATE = RBFSurrogate(
    centers=[
        (0, 0, 0, 0, 0),
        (5, 0, 0, 0, 0),
        (0, 5, 0, 0, 0),
        (0, 0, 5, 0, 0),
        (0, 0, 0, 5, 0),
        (0, 0, 0, 0, 5),
        (3, 3, 3, 3, 3),
    ],
    weights=[0.2, -0.1, -0.1, -0.1, -0.1, -0.1, 0.5],
    epsilon=0.5,
)


def rbf_diffusion_coefficient(vec: np.ndarray) -> float:
    """
    Compute the diffusion coefficient λ for the LTC cell using the RBF surrogate.
    The output is forced to be non‑negative.
    """
    raw = _RBF_SURROGATE.predict(vec.tolist())
    return max(0.0, raw)


# ----------------------------------------------------------------------
# Liquid‑Time‑Constant recurrent cell (Parent A)
# ----------------------------------------------------------------------
class LTCCell:
    """
    Liquid‑Time‑Constant recurrent cell whose mixing coefficient α is derived
    from the Gaussian similarity of successive feature vectors, and whose
    stochastic forcing amplitude λ comes from the RBF surrogate.
    """

    def __init__(self, dim: int, hidden: int = 8, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((hidden, dim))
        self.U = rng.standard_normal((hidden, hidden))
        self.b = rng.standard_normal(hidden)
        self.state = np.zeros(hidden, dtype=float)

    @staticmethod
    def similarity(prev: np.ndarray, cur: np.ndarray, epsilon: float = 0.5) -> float:
        """Gaussian similarity α ∈ (0,1]."""
        dist = euclidean(prev, cur)
        return gaussian(dist, epsilon)

    def step(self, cur_vec: np.ndarray, prev_vec: np.ndarray) -> np.ndarray:
        """
        Perform one LTC update.
        Returns the new hidden state.
        """
        α = self.similarity(prev_vec, cur_vec)
        λ = rbf_diffusion_coefficient(cur_vec)
        noise = np.random.normal(0.0, 1.0, size=self.state.shape)
        linear = self.W @ cur_vec + self.U @ self.state + self.b
        candidate = np.tanh(linear)
        self.state = (1 - α) * self.state + α * candidate + λ * noise
        return self.state.copy()


# ----------------------------------------------------------------------
# High‑level hybrid pipeline
# ----------------------------------------------------------------------
def process_texts(texts: List[str]) -> List[np.ndarray]:
    """
    Run the full hybrid pipeline:
    1. Deduplicate via perceptual hash.
    2. Extract regex feature vector.
    3. Update LTC cell.
    Returns the list of hidden states after each unique input.
    """
    seen_hashes = set()
    ltc = LTCCell(dim=5)
    prev_vec = np.zeros(5, dtype=float)
    states: List[np.ndarray] = []

    for txt in texts:
        # 1. Feature extraction
        cur_vec = extract_feature_vector(txt)

        # 2. Dedupe – skip if hash repeats
        h = perceptual_hash(cur_vec)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # 3. LTC update
        state = ltc.step(cur_vec, prev_vec)
        states.append(state)
        prev_vec = cur_vec

    return states


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "Please verify the source and provide a citation.",
        "We need a plan and a checklist for the next phase.",
        "Can you hold off until tomorrow? I need more time.",
        "Call my friend Kai for support.",
        "Respect the boundary and stop contacting me.",
        # Duplicate (should be ignored)
        "Please verify the source and provide a citation.",
        # Slight variation (new hash)
        "Please verify the source and provide a proof.",
    ]

    hidden_states = process_texts(sample_texts)
    for i, h in enumerate(hidden_states, 1):
        print(f"Step {i}: hidden state = {h}")
    print(f"Processed {len(hidden_states)} unique inputs out of {len(sample_texts)} total.")