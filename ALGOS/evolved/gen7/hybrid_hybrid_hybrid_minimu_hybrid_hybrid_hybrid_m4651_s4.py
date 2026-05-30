# DARWIN HAMMER — match 4651, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2535_s1.py (gen6)
# born: 2026-05-29T23:57:12Z

"""Hybrid Fusion of Minimum‑Cost Tree Bayes Update & Entropy‑Scaled NLMS

Parents:
- **Parent A** (`hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s1.py`):
  Bayesian edge posteriors, expected edge lengths, and a Count‑Min sketch
  providing expected rewards for actions.
- **Parent B** (`hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2535_s1.py`):
  Shannon entropy‑derived edge priors and an NLMS weight update whose
  learning rate is scaled by `exp(-H)` and a compatibility score `s`.

**Mathematical bridge**  
The entropy `H` computed from categorical evidence is used in three places:

1. **Prior scaling** – edge priors for the Bayesian update are set to  
   `πₑ = exp(-H) / Σₑ' exp(-H)`, i.e. a uniform prior weighted by `exp(-H)`.
2. **Learning‑rate scaling** – the NLMS step size `μ` becomes  
   `μ' = μ·exp(-H)`.
3. **Reward weighting** – the Count‑Min sketch estimate of an action’s
   reward supplies the observation vector `x` for NLMS; the compatibility
   score `s = xᵀ P x` (with a positive‑definite matrix `P`) further scales
   the update.

The fusion algorithm therefore:
* builds entropy‑scaled priors,
* updates edge posteriors with Bayesian inference,
* learns a weight vector via entropy‑scaled NLMS using sketch‑derived
  rewards,
* evaluates a hybrid tree cost that mixes expected edge lengths (posteriors)
  with expected rewards (sketch).

All components are pure NumPy / standard‑library code.
"""

import math
import random
import sys
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[int, int]          # (parent, child) identifiers
ActionID = str

# ----------------------------------------------------------------------
# 1. Entropy computation (Parent B)
# ----------------------------------------------------------------------
def compute_shannon_entropy(evidence: List[str]) -> float:
    """Shannon entropy H of a list of categorical tokens."""
    if not evidence:
        return 0.0
    cnt = Counter(evidence)
    total = len(evidence)
    entropy = -sum((c / total) * math.log2(c / total) for c in cnt.values())
    return entropy

# ----------------------------------------------------------------------
# 2. Count‑Min Sketch (Parent A)
# ----------------------------------------------------------------------
class CountMinSketch:
    """Simple Count‑Min sketch with d hash functions."""
    def __init__(self, width: int = 1000, depth: int = 5, seed: int = 0):
        self.width = width
        self.depth = depth
        self.tables = np.zeros((depth, width), dtype=np.float64)
        rng = random.Random(seed)
        self.seeds = [rng.randint(1, 2**31 - 1) for _ in range(depth)]

    def _hash(self, item: Any, i: int) -> int:
        h = hash((item, self.seeds[i]))
        return h % self.width

    def add(self, item: Any, count: float = 1.0) -> None:
        for i in range(self.depth):
            idx = self._hash(item, i)
            self.tables[i, idx] += count

    def estimate(self, item: Any) -> float:
        """Return the minimum count across hash tables (unbiased upper bound)."""
        return min(self.tables[i, self._hash(item, i)] for i in range(self.depth))

# ----------------------------------------------------------------------
# 3. Edge priors from entropy (Parent B) and Bayesian posterior (Parent A)
# ----------------------------------------------------------------------
def edge_priors_from_entropy(edges: List[Edge], evidence: List[str]) -> Dict[Edge, float]:
    """Uniform edge prior scaled by exp(-H)."""
    H = compute_shannon_entropy(evidence)
    if not edges:
        return {}
    weight = math.exp(-H)
    total = weight * len(edges)
    prior = weight / total
    return {e: prior for e in edges}

def bayes_edge_posteriors(
    edges: List[Edge],
    observations: Dict[Edge, Tuple[int, int]],  # (successes, trials)
    evidence: List[str],
    beta_prior: Tuple[float, float] = (1.0, 1.0)
) -> Dict[Edge, float]:
    """
    Compute posterior probability that an edge is “good”.
    Uses a Beta-Bernoulli model per edge, with priors derived from entropy.
    Returns the posterior mean for each edge.
    """
    priors = edge_priors_from_entropy(edges, evidence)
    posteriors: Dict[Edge, float] = {}
    for e in edges:
        a0, b0 = beta_prior
        # Scale prior mass by entropy weight (acts as pseudo‑counts)
        H = compute_shannon_entropy(evidence)
        scale = math.exp(-H)
        a = a0 * scale + observations.get(e, (0, 0))[0]  # successes
        b = b0 * scale + observations.get(e, (0, 0))[1] - observations.get(e, (0, 0))[0]  # failures
        post = a / (a + b) if (a + b) > 0 else 0.0
        # Blend with uniform prior from entropy for robustness
        posteriors[e] = priors[e] * 0.5 + post * 0.5
    return posteriors

# ----------------------------------------------------------------------
# 4. NLMS weight update with entropy‑scaled learning rate and compatibility (Parent B)
# ----------------------------------------------------------------------
def nlms_update(
    w: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float,
    evidence: List[str],
    P: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Normalized Least‑Mean‑Squares update:
        w ← w + μ'·s· (e·x) / (||x||² + ε)
    where:
        μ' = μ·exp(-H)                (entropy‑scaled learning rate)
        s = xᵀ P x                    (compatibility score)
        e = target - w·x              (prediction error)
    """
    H = compute_shannon_entropy(evidence)
    mu_prime = mu * math.exp(-H)
    norm_x2 = np.dot(x, x) + eps
    error = target - float(np.dot(w, x))
    s = float(np.dot(x, np.dot(P, x)))  # scalar compatibility
    update = (mu_prime * s * error / norm_x2) * x
    return w + update

# ----------------------------------------------------------------------
# 5. Hybrid tree cost mixing posteriors and sketch rewards (Parent A)
# ----------------------------------------------------------------------
def hybrid_tree_cost(
    root: int,
    edges: List[Edge],
    edge_lengths: Dict[Edge, float],
    posteriors: Dict[Edge, float],
    sketch: CountMinSketch,
    reward_scale: float = 1.0
) -> float:
    """
    Expected cost of a rooted tree:
        Σₑ  postₑ · (lengthₑ + reward_scale·Ê[rewardₑ])
    where Ê[rewardₑ] is the Count‑Min sketch estimate for edge identifier.
    """
    total = 0.0
    for e in edges:
        post = posteriors.get(e, 0.0)
        length = edge_lengths.get(e, 0.0)
        # Use edge tuple as sketch key
        reward_est = sketch.estimate(e)
        total += post * (length + reward_scale * reward_est)
    return total

# ----------------------------------------------------------------------
# 6. Convenience: build adjacency & sample data
# ----------------------------------------------------------------------
def build_sample_graph() -> Tuple[int, List[Edge], Dict[Edge, float]]:
    """Create a tiny tree for demonstration."""
    root = 0
    edges = [(0, 1), (0, 2), (1, 3), (1, 4)]
    # deterministic lengths
    edge_lengths = {
        (0, 1): 2.5,
        (0, 2): 3.0,
        (1, 3): 1.2,
        (1, 4): 0.9,
    }
    return root, edges, edge_lengths

# ----------------------------------------------------------------------
# 7. Main smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Evidence (categorical tokens)
    evidence = ["red", "blue", "red", "green", "blue", "red"]

    # 2. Graph
    root, edges, edge_lengths = build_sample_graph()

    # 3. Simulated observations per edge: (successes, trials)
    observations = {
        (0, 1): (8, 10),
        (0, 2): (3, 10),
        (1, 3): (7, 10),
        (1, 4): (2, 10),
    }

    # 4. Compute posterior edge probabilities
    post = bayes_edge_posteriors(edges, observations, evidence)

    # 5. Initialise Count‑Min sketch with synthetic rewards
    sketch = CountMinSketch(width=200, depth=4, seed=42)
    # Pretend each edge yields a reward proportional to its successes
    for e, (succ, trials) in observations.items():
        reward = succ / trials  # average reward
        sketch.add(e, count=reward * 10)  # scale to have integer‑like counts

    # 6. Hybrid tree cost
    cost = hybrid_tree_cost(root, edges, edge_lengths, post, sketch, reward_scale=5.0)
    print(f"Hybrid tree expected cost: {cost:.4f}")

    # 7. NLMS example: learn a weight vector for 3 features
    w = np.zeros(3)
    x = np.array([sketch.estimate((0, 1)), sketch.estimate((0, 2)), sketch.estimate((1, 3))])
    target = 1.0  # arbitrary desired output
    mu = 0.5
    P = np.eye(3) * 0.8  # positive‑definite compatibility matrix
    w_new = nlms_update(w, x, target, mu, evidence, P)
    print(f"Weight vector after NLMS update: {w_new}")

    # 8. Verify that the functions run without exception
    assert isinstance(cost, float)
    assert w_new.shape == (3,)
    print("Smoke test completed successfully.")