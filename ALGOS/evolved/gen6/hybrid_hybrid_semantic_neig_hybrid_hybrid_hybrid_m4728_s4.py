# DARWIN HAMMER — match 4728, survivor 4
# gen: 6
# parent_a: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1455_s1.py (gen5)
# born: 2026-05-29T23:57:44Z

"""Hybrid Semantic‑Morphology & Stylometry Circuit Breaker
Parent A: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s2.py
Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1455_s1.py

Mathematical bridge
-------------------
Both parents rely on vector‑based similarity measures.  
* Parent A computes a *semantic recovery priority* that mixes a morphology‑derived
  index with cosine similarity between a document vector and its semantic neighbours.  
* Parent B builds a *stylometry vector* from lexical‑category frequencies and later
  uses generic vector operations (e.g. dot‑product, cosine) to drive social‑interaction
  and predator‑evasion logic.

The fusion therefore uses a **common vector space**: a document is represented by a
concatenation of its semantic embedding (provided externally) and its stylometry
vector.  The unified “hybrid priority” is obtained by


H = α·R_morphology + (1‑α)·S_stylometry


where `R_morphology` is the recovery‑priority from Parent A and `S_stylometry` is a
normalized stylometry score derived from the cosine similarity of the combined
vector with a reference “healthy‑state” vector.  This scalar `H` drives a
circuit‑breaker that opens when the system’s ability to self‑right (morphology) and
its textual integrity (stylometry) fall below a dynamic threshold.

The implementation below provides three core functions that expose this hybrid
behaviour and a lightweight `HybridEndpointCircuitBreaker` class that uses the
combined priority to decide whether to open or close the circuit.
"""

import sys
import math
import random
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – morphology & semantic recovery
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized (0‑1) priority derived from morphology."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Standard cosine similarity, safe for zero‑vectors."""
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if den == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / den

# ----------------------------------------------------------------------
# Parent B – stylometry & lexical categories
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
}

def tokenise(text: str) -> List[str]:
    return [tok.strip(".,!?:;\"'()[]{}").lower() for tok in text.split() if tok]

def build_stylometry_vector(text: str) -> np.ndarray:
    """Return a normalized vector of category frequencies."""
    tokens = tokenise(text)
    total = len(tokens) or 1
    counts = {cat: 0 for cat in FUNCTION_CATS}
    for tok in tokens:
        for cat, vocab in FUNCTION_CATS.items():
            if tok in vocab:
                counts[cat] += 1
    vec = np.array([counts[cat] / total for cat in sorted(FUNCTION_CATS)])
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm

def stylometry_score(vec: np.ndarray, reference: np.ndarray) -> float:
    """Cosine similarity between a stylometry vector and a reference healthy vector."""
    return float(cosine_similarity(vec.tolist(), reference.tolist()))

# ----------------------------------------------------------------------
# Hybrid core – mathematical interface
# ----------------------------------------------------------------------
def compute_hybrid_priority(
    morphology: Morphology,
    semantic_embedding: List[float],
    document_text: str,
    alpha: float = 0.5,
    reference_stylometry: np.ndarray | None = None,
) -> float:
    """
    Combine morphology‑derived recovery priority with stylometry similarity.

    Parameters
    ----------
    morphology : Morphology
        Physical description used by Parent A.
    semantic_embedding : list[float]
        External semantic vector (e.g. from a language model).
    document_text : str
        Raw text whose stylometry is examined.
    alpha : float, default 0.5
        Weight of the morphology term (0‑1).  The stylometry term receives (1‑α).
    reference_stylometry : np.ndarray, optional
        Pre‑computed “healthy” stylometry vector.  If omitted a uniform vector is used.

    Returns
    -------
    float
        Hybrid priority in the range [0, 1].
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0 and 1")
    # 1️⃣ Morphology component (Parent A)
    r = recovery_priority(morphology)                     # 0‑1

    # 2️⃣ Stylometry component (Parent B)
    doc_vec = build_stylometry_vector(document_text)     # normalized
    if reference_stylometry is None:
        # uniform reference (all categories equally likely)
        reference_stylometry = np.ones_like(doc_vec) / np.linalg.norm(np.ones_like(doc_vec))
    s = stylometry_score(doc_vec, reference_stylometry)   # -1‑1 → map to 0‑1
    s = (s + 1.0) / 2.0

    # 3️⃣ Semantic neighbour influence (Parent A)
    # For illustration we treat the semantic embedding as a vector and compare it
    # to a fixed “ideal” embedding (all ones).  This mimics the neighbour similarity.
    ideal_emb = np.ones_like(semantic_embedding) / np.linalg.norm(np.ones_like(semantic_embedding))
    sem_sim = cosine_similarity(semantic_embedding, ideal_emb.tolist())  # -1‑1 → 0‑1
    sem_sim = (sem_sim + 1.0) / 2.0

    # Fuse everything: morphology, stylometry, semantic similarity
    hybrid = alpha * r + (1.0 - alpha) * (0.5 * s + 0.5 * sem_sim)
    return max(0.0, min(1.0, hybrid))

class HybridEndpointCircuitBreaker:
    """
    Circuit breaker that opens when the hybrid priority falls below a dynamic
    threshold.  The threshold itself adapts to recent hybrid scores (exponential
    moving average) to emulate the self‑righting behaviour of the serpentina
    morphology.
    """
    def __init__(self,
                 failure_threshold: int = 3,
                 alpha: float = 0.5,
                 ema_alpha: float = 0.2):
        self.failure_threshold = failure_threshold
        self.alpha = alpha                # weight for morphology in hybrid priority
        self.ema_alpha = ema_alpha        # smoothing for threshold adaptation
        self._failure_count = 0
        self._ema_threshold = 0.5         # start at a neutral midpoint

    def evaluate(self,
                 morphology: Morphology,
                 semantic_embedding: List[float],
                 document_text: str,
                 reference_stylometry: np.ndarray | None = None) -> bool:
        """
        Return True if the circuit remains closed (system healthy), False otherwise.
        """
        priority = compute_hybrid_priority(
            morphology,
            semantic_embedding,
            document_text,
            alpha=self.alpha,
            reference_stylometry=reference_stylometry,
        )

        # Update exponential moving average of the threshold
        self._ema_threshold = (
            self.ema_alpha * priority + (1.0 - self.ema_alpha) * self._ema_threshold
        )

        if priority < self._ema_threshold:
            self._failure_count += 1
        else:
            self._failure_count = max(0, self._failure_count - 1)

        return self._failure_count < self.failure_threshold

    def reset(self) -> None:
        self._failure_count = 0
        self._ema_threshold = 0.5

# ----------------------------------------------------------------------
# Demonstration functions (the required three)
# ----------------------------------------------------------------------
def demo_morphology_priority() -> None:
    """Show recovery_priority for a sample morphology."""
    m = Morphology(length=0.3, width=0.2, height=0.1, mass=2.5)
    print("Morphology recovery priority:", recovery_priority(m))

def demo_stylometry() -> None:
    """Compute and display a stylometry vector and its similarity to a reference."""
    txt = "I am the captain and I will lead the crew through the storm."
    vec = build_stylometry_vector(txt)
    ref = np.ones_like(vec) / np.linalg.norm(np.ones_like(vec))
    print("Stylometry vector:", vec)
    print("Similarity to uniform reference:", stylometry_score(vec, ref))

def demo_hybrid_breaker() -> None:
    """Run a short simulation of the hybrid circuit breaker."""
    morph = Morphology(length=0.4, width=0.25, height=0.15, mass=3.0)
    # Random semantic embedding (simulating a language‑model output)
    embedding = np.random.rand(8).tolist()
    text_good = "We have successfully completed the mission without any loss."
    text_bad = "Error error error! System failure, cannot recover."

    breaker = HybridEndpointCircuitBreaker(failure_threshold=2, alpha=0.6)

    for i, txt in enumerate([text_good, text_bad, text_bad, text_good, text_bad]):
        healthy = breaker.evaluate(morph, embedding, txt)
        state = "CLOSED (healthy)" if healthy else "OPEN (failure)"
        print(f"Step {i+1}: {state}")

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_morphology_priority()
    demo_stylometry()
    demo_hybrid_breaker()