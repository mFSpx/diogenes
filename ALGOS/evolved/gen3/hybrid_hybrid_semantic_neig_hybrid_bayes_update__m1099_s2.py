# DARWIN HAMMER — match 1099, survivor 2
# gen: 3
# parent_a: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s1.py (gen2)
# parent_b: hybrid_bayes_update_hybrid_krampus_brain_m15_s0.py (gen2)
# born: 2026-05-29T23:32:47Z

"""Hybrid Semantic-Bayesian Curvature Algorithm
Parents:
- hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s1.py (semantic neighbors, morphology‑based recovery priority)
- hybrid_bayes_update_hybrid_krampus_brain_m15_s0.py (feature extraction, Bayesian update)

Mathematical bridge:
The recovery priority R(m) derived from morphology serves as a *prior probability* for the
Bayesian update of cosine‑similarity scores S(i,j) between document vectors.  The posterior
P(i→j) = (S·R)/(S·R + (1−S)(1−R)) is then interpreted as an *Ollivier‑Ricci–like curvature* of the
edge (i,j).  This fuses the geometric intuition of the endpoint circuit with the probabilistic
evidence integration of the Bayes‑Krampus component.
"""

import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
import numpy as np

# ----------------------------------------------------------------------
# Morphology & recovery priority (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

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
    """Normalized to [0,1] – acts as a prior probability."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

# ----------------------------------------------------------------------
# Semantic neighbors (Parent A)
# ----------------------------------------------------------------------
def _cosine(a: list[float], b: list[float]) -> float:
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 0.0 if den == 0 else sum(x * y for x, y in zip(a, b)) / den

def semantic_neighbors(doc_id: str, vector: list[float], k: int = 5) -> list[tuple[str, float]]:
    """Return k most similar (cosine) neighbours, including synthetic random vectors."""
    candidates = [(doc_id, vector)]
    for i in range(1, k + 1):
        rnd_vec = np.random.rand(len(vector)).tolist()
        candidates.append((f"doc{i}", rnd_vec))
    # compute similarity excluding self
    sims = [(d, _cosine(vector, w)) for d, w in candidates if d != doc_id]
    return sorted(sims, key=lambda x: (-x[1], x[0]))[:k]

# ----------------------------------------------------------------------
# Feature extraction & Bayesian update (Parent B)
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> dict[str, float]:
    """Stub: produce a deterministic pseudo‑random feature map from the input text."""
    rng = random.Random(hash(text))
    keys = [
        "operator_visceral_ratio", "operator_tech_ratio", "operator_legal_osint_ratio",
        "psyche_forensic_shield_ratio", "psyche_poetic_entropy", "psyche_dissociative_index",
        "resilience_bureaucratic_weaponization_index", "resilience_resource_exhaustion_metric",
        "resilience_swarm_orchestration_density", "rainmaker_corporate_grit_tension",
        "rainmaker_countdown_density", "rainmaker_asset_structuring_weight",
        "telemetry_agent_symmetry_ratio", "telemetry_protocol_discipline",
        "telemetry_manic_velocity"
    ]
    return {k: rng.random() for k in keys}

def extract_master_vector(text: str) -> dict[str, float]:
    f = extract_full_features(text)
    return {
        "visceral_ratio": f.get("operator_visceral_ratio", 0.0),
        "tech_ratio": f.get("operator_tech_ratio", 0.0),
        "legal_osint_ratio": f.get("operator_legal_osint_ratio", 0.0),
        "forensic_shield_ratio": f.get("psyche_forensic_shield_ratio", 0.0),
        "poetic_entropy": f.get("psyche_poetic_entropy", 0.0),
        "dissociative_index": f.get("psyche_dissociative_index", 0.0),
        "bureaucratic_weaponization_index": f.get("resilience_bureaucratic_weaponization_index", 0.0),
        "resource_exhaustion_metric": f.get("resilience_resource_exhaustion_metric", 0.0),
        "swarm_orchestration_density": f.get("resilience_swarm_orchestration_density", 0.0),
        "corporate_grit_tension": f.get("rainmaker_corporate_grit_tension", 0.0),
        "countdown_density": f.get("rainmaker_countdown_density", 0.0),
        "asset_structuring_weight": f.get("rainmaker_asset_structuring_weight", 0.0),
        "agent_symmetry_ratio": f.get("telemetry_agent_symmetry_ratio", 0.0),
        "protocol_discipline": f.get("telemetry_protocol_discipline", 0.0),
        "manic_velocity": f.get("telemetry_manic_velocity", 0.0),
    }

def bayesian_update(prior: float, likelihood: float) -> float:
    """
    Binary Bayesian update:
        posterior = (likelihood * prior) /
                    (likelihood * prior + (1 - likelihood) * (1 - prior))
    Both arguments are assumed to be in [0,1].
    """
    eps = 1e-12
    prior = min(max(prior, 0.0), 1.0)
    likelihood = min(max(likelihood, 0.0), 1.0)
    numerator = likelihood * prior
    denominator = numerator + (1 - likelihood) * (1 - prior) + eps
    return numerator / denominator

# ----------------------------------------------------------------------
# Hybrid core: curvature as Bayesian‑posterior‑weighted similarity
# ----------------------------------------------------------------------
def hybrid_curvature(prior: float, similarity: float) -> float:
    """
    Treat similarity as likelihood and prior from morphology.
    The resulting posterior is interpreted as a curvature‑like weight.
    """
    return bayesian_update(prior, similarity)

def hybrid_neighbor_curvature(doc_id: str,
                              vector: list[float],
                              morphology: Morphology,
                              k: int = 5) -> list[tuple[str, float]]:
    """
    Compute curvature‑adjusted neighbor scores.
    Returns a list of (neighbor_id, curvature) sorted descending.
    """
    prior = recovery_priority(morphology)
    raw_neighbors = semantic_neighbors(doc_id, vector, k)
    return sorted(
        [(nid, hybrid_curvature(prior, sim)) for nid, sim in raw_neighbors],
        key=lambda x: -x[1]
    )

# ----------------------------------------------------------------------
# Hybrid endpoint circuit that records events and uses the hybrid scores
# ----------------------------------------------------------------------
class HybridEndpointCircuit:
    def __init__(self,
                 failure_threshold: int = 3,
                 morphology: Morphology | None = None,
                 alpha: float = 0.5):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()
        self.morphology = morphology
        self.alpha = alpha  # blending factor for external score vs internal curvature

    def record_event(self, success: bool) -> None:
        self.last_event_at = now_z()
        if not success:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.open = True
        else:
            # successful event reduces failure count
            self.failures = max(0, self.failures - 1)
            if self.failures < self.failure_threshold:
                self.open = False

    def hybrid_score(self,
                     doc_id: str,
                     vector: list[float],
                     k: int = 5) -> list[tuple[str, float]]:
        """
        Combine curvature‑adjusted neighbor scores with an external Bayesian
        evidence vector derived from a textual description.
        """
        # curvature part
        curv_scores = hybrid_neighbor_curvature(doc_id, vector, self.morphology or Morphology(1,1,1,1), k)

        # external evidence part (simulated via random text)
        evidence = extract_master_vector("dummy context")
        # compress evidence to a single scalar in [0,1] (simple mean)
        evidence_scalar = sum(evidence.values()) / len(evidence)

        # blend
        blended = [
            (nid,
             self.alpha * curvature + (1 - self.alpha) * evidence_scalar)
            for nid, curvature in curv_scores
        ]
        return sorted(blended, key=lambda x: -x[1])

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Z format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # create a simple morphology
    morph = Morphology(length=2.0, width=1.0, height=0.5, mass=3.0)

    # instantiate circuit
    circuit = HybridEndpointCircuit(failure_threshold=2, morphology=morph, alpha=0.7)

    # synthetic document vector
    doc_vec = np.random.rand(5).tolist()
    doc_id = "doc0"

    # simulate events
    circuit.record_event(success=False)
    circuit.record_event(success=False)  # should open the circuit
    print(f"Circuit open? {circuit.open}")

    # compute hybrid scores
    scores = circuit.hybrid_score(doc_id, doc_vec, k=4)
    print("Hybrid neighbor curvature scores:")
    for nid, sc in scores:
        print(f"  {nid}: {sc:.4f}")

    # recover priority sanity check
    print(f"Recovery priority (prior): {recovery_priority(morph):.4f}")

    sys.exit(0)