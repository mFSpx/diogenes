# DARWIN HAMMER — match 3180, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hard_truth_ma_kan_m27_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_percyphon_hyb_m1198_s0.py (gen5)
# born: 2026-05-29T23:48:16Z

"""Hybrid Stylometry‑Regex RBF Model
===================================

Parent A: *stylometry / LSM utilities* – provides a low‑dimensional
frequency vector `lsm_vector(text)` describing the distribution of
function‑word categories in a piece of text.

Parent B: *regex feature extraction & morphology‑driven RBF surrogate* – extracts
binary/count features from a text via regular expressions and evaluates them
through a Radial Basis Function (RBF) model whose centres and weights are
modulated by geometric morphology metrics (sphericity σ and flatness φ).

Mathematical Bridge
-------------------
1. **Feature Fusion** – The LSM vector (dimension `C`, number of function‑word
   categories) is concatenated with the regex‑derived feature vector (dimension
   `R`). The resulting hybrid feature vector  

   `x = [ lsm_vector ; regex_features ] ∈ ℝ^{C+R}`  

   serves as the input to the RBF surrogate.

2. **Morphology Modulation** – Given a 3‑D bounding box `(l, w, h)` the
   morphology metrics  

   `σ = V·π^{1/3}·(6V)^{2/3} / S`  with `V = l·w·h` and `S = 2(lw + lh + wh)`  

   `φ = min(l,w,h) / max(l,w,h)`  

   are computed. The sphericity `σ` scales the RBF centres: each centre `C_i`
   is multiplied by the factor `(1 + α·σ)`, where `α` is a configurable
   influence coefficient.

3. **RBF Evaluation** – With spread `γ` and weights `w_i` the hybrid output is  

   `y = Σ_i w_i · exp( -γ ‖ x - (1+α·σ)·C_i ‖² )`.

The implementation below realises this bridge, exposing three core functions
that illustrate the hybrid operation and a smoke‑test runnable as a script.
"""

import sys
import math
import random
import pathlib
import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import re

# ----------------------------------------------------------------------
# Parent A – stylometry / LSM utilities
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot cant wont dont didnt isnt arent was wasnt werent".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}

def words(text: str) -> List[str]:
    """Split text into lower‑case alphabetic tokens."""
    return [w for w in (text or "").lower().split() if w.isalpha()]

def lsm_vector(text: str) -> np.ndarray:
    """
    Return a dense numpy vector of length ``len(FUNCTION_CATS)`` where each entry
    is the relative frequency of the corresponding function‑word category.
    """
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    vec = []
    for cat in FUNCTION_CATS:
        vocab = FUNCTION_CATS[cat]
        freq = sum(cnt[w] for w in vocab) / total
        vec.append(freq)
    return np.array(vec, dtype=float)

def stable_hash(text: str) -> int:
    """Deterministic integer hash based on SHA‑256."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)

# ----------------------------------------------------------------------
# Parent B – regex feature extraction
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
    r"\b(?:ask|call|text|friend|friends|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety)\b",
    re.I,
)

def regex_features(text: str) -> np.ndarray:
    """
    Count occurrences of each regex pattern and return a dense vector of
    shape (5,). Order: evidence, planning, delay, support, boundary.
    """
    counts = [
        len(EVIDENCE_RE.findall(text)),
        len(PLANNING_RE.findall(text)),
        len(DELAY_RE.findall(text)),
        len(SUPPORT_RE.findall(text)),
        len(BOUNDARY_RE.findall(text)),
    ]
    return np.array(counts, dtype=float)

# ----------------------------------------------------------------------
# Morphology metrics (used to modulate RBF centres)
# ----------------------------------------------------------------------
def morphology_metrics(l: float, w: float, h: float) -> Tuple[float, float]:
    """
    Compute sphericity σ and flatness φ for a rectangular prism with side
    lengths l, w, h.
    """
    V = l * w * h
    S = 2 * (l * w + l * h + w * h)
    # Guard against degenerate geometry
    if S == 0:
        sigma = 0.0
    else:
        sigma = V * (math.pi ** (1.0 / 3.0)) * ((6 * V) ** (2.0 / 3.0)) / S
    phi = min(l, w, h) / max(l, w, h) if max(l, w, h) > 0 else 0.0
    return sigma, phi

# ----------------------------------------------------------------------
# RBF surrogate with morphology‑scaled centres
# ----------------------------------------------------------------------
@dataclass
class RBFModel:
    n_centers: int
    gamma: float
    alpha: float
    rng_seed: int = 0

    centres: np.ndarray = None   # shape (n_centers, dim)
    weights: np.ndarray = None   # shape (n_centers,)

    def _initialize(self, dim: int):
        rng = np.random.RandomState(self.rng_seed)
        self.centres = rng.normal(loc=0.0, scale=1.0, size=(self.n_centers, dim))
        self.weights = rng.normal(loc=0.0, scale=1.0, size=self.n_centers)

    def evaluate(self, x: np.ndarray, sigma: float) -> float:
        """
        Evaluate the RBF model on input vector ``x`` using morphology sphericity
        ``sigma`` to scale the centres.
        """
        if self.centres is None or self.weights is None:
            self._initialize(dim=x.shape[0])

        scale = 1.0 + self.alpha * sigma
        scaled_centres = self.centres * scale
        diffs = x - scaled_centres  # broadcasting over centres
        sq_norms = np.einsum('ij,ij->i', diffs, diffs)
        contributions = self.weights * np.exp(-self.gamma * sq_norms)
        return float(contributions.sum())

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_feature_vector(text: str) -> np.ndarray:
    """
    Concatenate the stylometry LSM vector with regex‑derived counts.
    Resulting dimension = len(FUNCTION_CATS) + 5.
    """
    lsm = lsm_vector(text)
    regex = regex_features(text)
    return np.concatenate([lsm, regex])

def hybrid_rbf_output(text: str,
                      dimensions: Tuple[float, float, float],
                      rbf_params: Dict) -> float:
    """
    Compute the hybrid RBF output for ``text`` given a 3‑D geometry
    ``dimensions = (l, w, h)`` and a dictionary ``rbf_params`` containing
    ``n_centers``, ``gamma`` and ``alpha`` (optional ``rng_seed``).
    """
    x = hybrid_feature_vector(text)
    sigma, _ = morphology_metrics(*dimensions)
    model = RBFModel(
        n_centers=rbf_params.get("n_centers", 8),
        gamma=rbf_params.get("gamma", 0.5),
        alpha=rbf_params.get("alpha", 1.0),
        rng_seed=rbf_params.get("rng_seed", stable_hash(text) % (2**32))
    )
    return model.evaluate(x, sigma)

def hybrid_predict(text: str,
                   dimensions: Tuple[float, float, float],
                   rbf_params: Dict) -> int:
    """
    Produce a binary decision (0/1) based on the sign of the hybrid RBF output.
    Positive output → 1, otherwise → 0.
    """
    score = hybrid_rbf_output(text, dimensions, rbf_params)
    return int(score > 0)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "I have verified the source and attached the screenshot. "
        "Next we should plan the next steps and schedule a meeting. "
        "If there is any delay, let me know."
    )
    # Example geometry: a modest rectangular box (units arbitrary)
    geom = (2.0, 1.5, 0.8)  # length, width, height

    params = {
        "n_centers": 6,
        "gamma": 0.8,
        "alpha": 0.3,
    }

    print("Hybrid feature vector (shape):", hybrid_feature_vector(sample_text).shape)
    print("Hybrid RBF output:", hybrid_rbf_output(sample_text, geom, params))
    print("Hybrid binary prediction:", hybrid_predict(sample_text, geom, params))
    sys.exit(0)