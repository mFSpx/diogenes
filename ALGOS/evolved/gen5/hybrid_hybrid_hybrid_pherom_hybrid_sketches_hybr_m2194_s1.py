# DARWIN HAMMER — match 2194, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_pheromone_inf_hybrid_decision_hygi_m37_s3.py (gen2)
# parent_b: hybrid_sketches_hybrid_hybrid_bandit_m850_s0.py (gen4)
# born: 2026-05-29T23:41:24Z

"""
Hybrid Algorithm: hybrid_pheromone_bandit_entropy_sketch.py

Parents:
- hybrid_hybrid_pheromone_inf_hybrid_decision_hygi_m37_s3 (Pheromone infotaxis + decision hygiene + Shannon entropy)
- hybrid_sketches_hybrid_hybrid_bandit_m850_s0 (Count‑Min / HLL / MinHash sketches + bandit router + epistemic certainty flags)

Mathematical Bridge:
The bridge is the *entropy‑scaled inflow* of the Count‑Min sketch that records pheromone
usage.  Shannon entropy of the decision‑hygiene scores is used as a scalar that
modulates the confidence bounds of a UCB‑style bandit router.  Inflow rates (sketch
increments) are proportional to scores·(1+H), while outflow rates (UCB exploration
terms) are multiplied by a factor derived from epistemic‑certainty flags.
Thus the sketch provides a compact, streaming representation of pheromone
distribution, and the bandit’s expected‑reward + confidence bound calculation
is informed by both entropy and epistemic certainty, yielding a unified action
selection mechanism.
"""

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import numpy as np

# ----------------------------------------------------------------------
# Epistemic certainty flag (parent B)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int  # basis points, 0..10000 → 0%..100%
    authority_class: str
    rationale: str
    evidence_refs: tuple[str, ...] = ()
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc)
                                            .isoformat()
                                            .replace("+00:00", "Z"))

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")

    def as_dict(self) -> dict:
        return dict(self.__dict__)

# ----------------------------------------------------------------------
# Simple Count‑Min Sketch (parent B)
# ----------------------------------------------------------------------
class CountMinSketch:
    """
    A lightweight Count‑Min sketch using a fixed number of hash functions.
    """

    def __init__(self, width: int = 2_048, depth: int = 4):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int64)
        # deterministic seeds for reproducibility
        self.seeds = [i * 0x9e3779b9 for i in range(depth)]

    def _hash(self, item: str, seed: int) -> int:
        h = hashlib.blake2b(digest_size=8, key=seed.to_bytes(8, "little"))
        h.update(item.encode("utf-8"))
        return int.from_bytes(h.digest(), "little") % self.width

    def add(self, item: str, count: int = 1) -> None:
        for row, seed in enumerate(self.seeds):
            col = self._hash(item, seed)
            self.table[row, col] += count

    def estimate(self, item: str) -> int:
        estimates = []
        for row, seed in enumerate(self.seeds):
            col = self._hash(item, seed)
            estimates.append(self.table[row, col])
        return min(estimates)


# ----------------------------------------------------------------------
# Core mathematical helpers (parent A)
# ----------------------------------------------------------------------
def shannon_entropy(scores: np.ndarray) -> float:
    """
    Compute Shannon entropy of a non‑negative score vector.
    Scores are first normalized to a probability distribution.
    """
    if scores.ndim != 1:
        raise ValueError("scores must be a 1‑D array")
    total = scores.sum()
    if total == 0:
        return 0.0
    probs = scores / total
    # avoid log(0) by masking zero entries
    mask = probs > 0
    return -float(np.sum(probs[mask] * np.log2(probs[mask])))


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return a unit‑sum (probability) version of the vector."""
    total = vector.sum()
    if total == 0:
        return np.full_like(vector, 1.0 / len(vector))
    return vector / total


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def update_pheromone_sketch(
    sketch: CountMinSketch,
    actions: list[str],
    scores: np.ndarray,
    entropy: float,
) -> None:
    """
    Inflow: for each action, add a count proportional to its decision‑hygiene score
    scaled by (1 + entropy).  Entropy acts as a global multiplier that rewards
    higher uncertainty (more exploration) by increasing inflow.
    """
    multiplier = 1.0 + entropy  # entropy ≥0, so multiplier ≥1
    for act, sc in zip(actions, scores):
        increment = max(1, int(round(sc * multiplier)))
        sketch.add(act, increment)


def compute_ucb(
    propensity: np.ndarray,
    pulls: np.ndarray,
    total_pulls: int,
    flag: CertaintyFlag,
) -> np.ndarray:
    """
    Upper Confidence Bound (UCB) with epistemic modulation.
    The classic term sqrt(2 * ln(total) / pulls) is multiplied by a factor
    derived from the flag's confidence (higher confidence → smaller exploration).
    """
    # Avoid division by zero
    pulls = np.where(pulls == 0, 1, pulls)
    exploration = np.sqrt(2 * np.log(max(1, total_pulls)) / pulls)

    # Convert basis points to a [0,1] confidence factor; higher confidence ⇒ lower factor
    confidence = flag.confidence_bps / 10000.0
    epistemic_factor = 1.0 - confidence  # 0 (certain) → no extra exploration, 1 (uncertain) → full

    return propensity + epistemic_factor * exploration


def hybrid_action_selection(
    actions: list[str],
    scores: np.ndarray,
    sketch: CountMinSketch,
    propensity: np.ndarray,
    pulls: np.ndarray,
    flag: CertaintyFlag,
) -> str:
    """
    Perform a single hybrid decision step:
    1. Compute entropy of the hygiene scores.
    2. Update the pheromone sketch with entropy‑scaled inflow.
    3. Estimate current pheromone levels from the sketch (outflow proxy).
    4. Combine pheromone estimates with bandit UCB values to obtain a final
       selection probability distribution.
    5. Sample an action accordingly.
    """
    # 1. Entropy
    entropy = shannon_entropy(scores)

    # 2. Sketch update (inflow)
    update_pheromone_sketch(sketch, actions, scores, entropy)

    # 3. Pheromone estimates (outflow proxy)
    pheromone_counts = np.array([sketch.estimate(a) for a in actions], dtype=np.float64)
    pheromone_probs = normalize(pheromone_counts)

    # 4. Bandit UCB values
    total_pulls = int(pulls.sum())
    ucb_values = compute_ucb(propensity, pulls, total_pulls, flag)

    # Blend pheromone probabilities with normalized UCB values
    # Weight α for pheromone, (1‑α) for bandit; α derived from entropy (more entropy → rely more on bandit)
    alpha = 1.0 / (1.0 + entropy)  # entropy 0 → α=1 (pure pheromone), high entropy → α→0
    blended = alpha * pheromone_probs + (1 - alpha) * normalize(ucb_values)

    # 5. Sample action
    chosen = random.choices(actions, weights=blended, k=1)[0]
    return chosen


# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
def _smoke_test() -> None:
    actions = ["search", "store", "analyze", "report"]
    n = len(actions)

    # Random initial decision‑hygiene scores (positive)
    scores = np.random.rand(n) * 10 + 1

    # Bandit state
    propensity = np.random.rand(n) * 5  # estimated rewards
    pulls = np.zeros(n, dtype=int)      # number of times each arm was pulled

    # Epistemic flag (moderately certain)
    flag = CertaintyFlag(
        label="PROBABLE",
        confidence_bps=7000,  # 70 %
        authority_class="expert",
        rationale="simulation test",
    )

    sketch = CountMinSketch(width=1024, depth=4)

    print("=== Hybrid Decision Simulation ===")
    for step in range(10):
        # Update scores slightly to emulate changing hygiene assessments
        scores = np.clip(scores + np.random.randn(n), 0.1, None)

        # Select an action
        chosen = hybrid_action_selection(
            actions,
            scores,
            sketch,
            propensity,
            pulls,
            flag,
        )

        # Simulate observed reward (random for demo) and update bandit state
        reward = random.random()
        idx = actions.index(chosen)
        pulls[idx] += 1
        # Incremental update of propensity (simple average)
        propensity[idx] = ((propensity[idx] * (pulls[idx] - 1)) + reward) / pulls[idx]

        print(
            f"Step {step+1:2d}: chosen={chosen:8s} | reward={reward:.3f} | "
            f"entropy={shannon_entropy(scores):.3f} | pulls={pulls.tolist()}"
        )

    # Final sketch estimates
    final_estimates = {a: sketch.estimate(a) for a in actions}
    print("\nFinal pheromone (sketch) estimates:", final_estimates)


if __name__ == "__main__":
    _smoke_test()