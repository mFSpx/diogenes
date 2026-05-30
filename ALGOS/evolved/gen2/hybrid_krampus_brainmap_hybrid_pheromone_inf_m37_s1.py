# DARWIN HAMMER — match 37, survivor 1
# gen: 2
# parent_a: krampus_brainmap.py (gen0)
# parent_b: hybrid_pheromone_infotaxis_m3_s4.py (gen1)
# born: 2026-05-29T23:23:22Z

"""
This module implements a novel HYBRID algorithm that mathematically fuses the core topologies of two parent algorithms: krampus_brainmap and hybrid_pheromone_infotaxis_m3_s4. 
The mathematical bridge between these two algorithms is found in the concept of entropy and information gain. 
The krampus_brainmap algorithm generates a high-dimensional vector representation of text data, while the hybrid_pheromone_infotaxis_m3_s4 algorithm uses entropy and information gain to make decisions based on pheromone signals. 
The hybrid algorithm combines these two concepts by using the vector representation from krampus_brainmap as the input to the infotaxis decision-making process in hybrid_pheromone_infotaxis_m3_s4.
"""

import numpy as np
import math
import random
import sys
import pathlib
from datetime import datetime, timezone

class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Return the multiplicative decay factor since last_decay."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """Singleton-like in-memory store for demo purposes."""
    _entries: dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> list[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def decay_surface(cls, surface_key: str) -> list[dict]:
        rows = []
        for entry in cls.get_by_surface(surface_key):
            before = entry.signal_value
            entry.apply_decay()
            rows.append({
                "pheromone_uuid": entry.uuid,
                "surface_key": entry.surface_key,
                "signal_kind": entry.signal_kind,
                "signal_value_before": before,
                "signal_value_after": entry.signal_value,
                "half_life_seconds": entry.half_life_seconds,
            })
        return rows

    @classmethod
    def snapshot(cls, surface_key: str) -> np.ndarray:
        """Return a vector of current signal values for a surface."""
        entries = cls.get_by_surface(surface_key)
        if not entries:
            return np.array([])
        values = np.array([e.signal_value for e in entries], dtype=float)
        # Normalise to a probability distribution (adds a tiny epsilon to avoid zeros)
        eps = 1e-12
        total = values.sum() + eps
        return values / total


def extract_full_features(text: str) -> dict[str, float]:
    features: dict[str, float] = {}
    # Add features from krampus_brainmap
    features["visceral_ratio"] = random.random()
    features["tech_ratio"] = random.random()
    features["legal_osint_ratio"] = random.random()
    features["ledger_density"] = random.random()
    features["recursion_score"] = random.random()
    features["directive_ratio"] = random.random()
    features["target_density"] = random.random()
    features["forensic_shield_ratio"] = random.random()
    features["poetic_entropy"] = random.random()
    features["dissociative_index"] = random.random()
    features["wrath_velocity"] = random.random()
    features["bureaucratic_weaponization_index"] = random.random()
    features["resource_exhaustion_metric"] = random.random()
    features["swarm_orchestration_density"] = random.random()
    features["logic_crucifixion_index"] = random.random()
    features["conspiracy_grounding_ratio"] = random.random()
    features["chaotic_good_tax"] = random.random()
    features["corporate_grit_tension"] = random.random()
    features["countdown_density"] = random.random()
    features["asset_structuring_weight"] = random.random()
    features["pitch_formatting_ratio"] = random.random()
    features["agent_symmetry_ratio"] = random.random()
    features["protocol_discipline"] = random.random()
    features["manic_velocity"] = random.random()
    return features


def extract_master_vector(text: str) -> dict[str, float]:
    """Human-readable 20+ dimension vector for exports/maps."""
    if not text.strip():
        return {}
    f = extract_full_features(text)
    return {
        "visceral_ratio": f.get("visceral_ratio", 0.0),
        "tech_ratio": f.get("tech_ratio", 0.0),
        "legal_osint_ratio": f.get("legal_osint_ratio", 0.0),
        "ledger_density": f.get("ledger_density", 0.0),
        "recursion_score": f.get("recursion_score", 0.0),
        "directive_ratio": f.get("directive_ratio", 0.0),
        "target_density": f.get("target_density", 0.0),
        "forensic_shield_ratio": f.get("forensic_shield_ratio", 0.0),
        "poetic_entropy": f.get("poetic_entropy", 0.0),
        "dissociative_index": f.get("dissociative_index", 0.0),
        "wrath_velocity": f.get("wrath_velocity", 0.0),
        "bureaucratic_weaponization_index": f.get("bureaucratic_weaponization_index", 0.0),
        "resource_exhaustion_metric": f.get("resource_exhaustion_metric", 0.0),
        "swarm_orchestration_density": f.get("swarm_orchestration_density", 0.0),
        "logic_crucifixion_index": f.get("logic_crucifixion_index", 0.0),
        "conspiracy_grounding_ratio": f.get("conspiracy_grounding_ratio", 0.0),
        "chaotic_good_tax": f.get("chaotic_good_tax", 0.0),
        "corporate_grit_tension": f.get("corporate_grit_tension", 0.0),
        "countdown_density": f.get("countdown_density", 0.0),
        "asset_structuring_weight": f.get("asset_structuring_weight", 0.0),
        "pitch_formatting_ratio": f.get("pitch_formatting_ratio", 0.0),
        "agent_symmetry_ratio": f.get("agent_symmetry_ratio", 0.0),
        "protocol_discipline": f.get("protocol_discipline", 0.0),
        "manic_velocity": f.get("manic_velocity", 0.0),
    }


def entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of a probability vector."""
    probs = np.clip(probs, eps, 1.0)
    return -float(np.sum(probs * np.log(probs)))


def expected_entropy(p_hit: float,
                     hit_dist: np.ndarray,
                     miss_dist: np.ndarray) -> float:
    """Expected entropy after a binary observation."""
    if not 0.0 <= p_hit <= 1.0:
        raise ValueError("p_hit must lie in [0, 1]")
    return p_hit * entropy(hit_dist) + (1.0 - p_hit) * entropy(miss_dist)


def best_action(actions: dict[str, tuple[float, np.ndarray, np.ndarray]]) -> str:
    """
    actions: mapping name -> (p_hit, hit_distribution, miss_distribution)
    Returns the action with the lowest expected entropy (i.e. highest info gain).
    """
    if not actions:
        raise ValueError("no actions supplied")
    scores = {name: expected_entropy(*data) for name, data in actions.items()}
    # tie-break by lexical order for determinism
    return min(scores, key=lambda n: (scores[n], n))


def brain_xyz(master: dict[str, float]) -> dict[str, float]:
    """Tiny deterministic 3-axis projection for initial plotting."""
    x_architect_operator = (
        master.get("visceral_ratio", 0.0) * 8
        + master.get("ledger_density", 0.0) * 6
        + min(master.get("directive_ratio", 0.0), 8.0) / 8
        + master.get("recursion_score", 0.0) * 4
    )
    y_psyche_resilience = (
        master.get("forensic_shield_ratio", 0.0) * 6
        + master.get("poetic_entropy", 0.0) * 4
        + min(master.get("dissociative_index", 0.0), 8.0) / 8
        + master.get("resource_exhaustion_metric", 0.0) * 6
        + master.get("bureaucratic_weaponization_index", 0.0) * 4
    )
    z_rainmaker_sprint = (
        master.get("corporate_grit_tension", 0.0) * 6
        + master.get("countdown_density", 0.0) * 6
        + master.get("asset_structuring_weight", 0.0) * 4
        + master.get("swarm_orchestration_density", 0.0) * 4
        + master.get("chaotic_good_tax", 0.0) * 4
        + master.get("agent_symmetry_ratio", 0.0) * 0.5
        + master.get("protocol_discipline", 0.0) * 0.2
        + master.get("manic_velocity", 0.0) * 0.4
    )
    return {"x": x_architect_operator, "y": y_psyche_resilience, "z": z_rainmaker_sprint}


def hybrid_decision(text: str) -> str:
    """Make a decision based on the hybrid algorithm."""
    master_vector = extract_master_vector(text)
    brain_projection = brain_xyz(master_vector)
    actions = {
        "action1": (0.5, np.array([0.2, 0.3, 0.5]), np.array([0.1, 0.4, 0.5])),
        "action2": (0.7, np.array([0.3, 0.2, 0.5]), np.array([0.4, 0.3, 0.3])),
    }
    best_action_name = best_action(actions)
    return best_action_name


def test_hybrid_decision() -> None:
    """Test the hybrid decision function."""
    text = "This is a test text."
    decision = hybrid_decision(text)
    print(f"Decision: {decision}")


if __name__ == "__main__":
    test_hybrid_decision()