# DARWIN HAMMER — match 4700, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_krampu_m2657_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m1529_s2.py (gen6)
# born: 2026-05-29T23:57:35Z

"""
Hybrid Algorithm: TropicalNetwork + PheromoneEntry fused with Voronoi‑Bandit‑Honeybee‑Schoolfield

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_krampu_m2657_s1.py (TropicalNetwork & PheromoneEntry)
- hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m1529_s2.py (Voronoi partition, Ollivier‑Ricci curvature,
  honeybee store, Schoolfield temperature, bandit UCB)

Mathematical Bridge:
The TropicalNetwork produces a non‑negative vector 𝑦 = ReLU(W·x + b). After L1‑normalisation
𝑝_i = y_i / Σ_j y_j we obtain a probability distribution over network units.
PheromoneEntry objects generate a second distribution 𝑞_i proportional to their decayed signal
values. The Kullback‑Leibler divergence D_KL(p‖q) quantifies the mismatch between the learned
network representation and the environment‑driven pheromone field.

Each Voronoi region r (derived from a set of seed points) is assigned an Ollivier‑Ricci curvature
κ_r, which we treat as a raw expected reward. The honeybee store S_r modulates this reward
through a sigmoid σ(S_r)=1/(1+e^{‑S_r}). Temperature‑dependent Schoolfield scaling
λ_T(T) further adjusts the learning rate η. The final bandit‑style expected reward for region r
is

    ȓ_r = κ_r · σ(S_r) · exp(‑D_KL(p‖q_r))

where D_KL is computed between the global network distribution p and the region‑specific
pheromone distribution q_r. The bandit selects the region with the highest Upper‑Confidence
Bound (UCB) built from ȓ_r, its propensity and a confidence term.

The hybrid therefore intertwines:
* tropical neural computation,
* information‑theoretic alignment with pheromone fields,
* geometric curvature from Voronoi topology,
* biologically inspired store and temperature dynamics,
* and a bandit decision process.

The code below implements the core components and demonstrates a full forward‑update cycle.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Dict, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Shared immutable types (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class EngineEndpoint:
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: List[str]
    morphology: Morphology
    outbound_state: str = "draft_only"

# ----------------------------------------------------------------------
# Tropical Network (Parent A)
# ----------------------------------------------------------------------
class TropicalNetwork:
    """
    Simple ReLU‑style network: y_i = max(0, w_i·x + b_i)
    """
    def __init__(self, weights: np.ndarray, biases: np.ndarray):
        if weights.shape[0] != biases.shape[0]:
            raise ValueError("weights and biases must have same first dimension")
        self.weights = weights
        self.biases = biases

    def evaluate(self, input_vector: np.ndarray) -> np.ndarray:
        """Return network output vector."""
        out = np.maximum(0.0, self.weights @ input_vector + self.biases)
        return out

    def probability_distribution(self, input_vector: np.ndarray) -> np.ndarray:
        """L1‑normalise the ReLU output to obtain a probability distribution."""
        out = self.evaluate(input_vector)
        total = out.sum()
        if total == 0:
            # avoid division by zero – fall back to uniform distribution
            return np.full_like(out, 1.0 / out.size)
        return out / total

# ----------------------------------------------------------------------
# PheromoneEntry (Parent A)
# ----------------------------------------------------------------------
class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(random.getrandbits(128))
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def decay(self) -> None:
        """Apply exponential decay based on half‑life."""
        now = datetime.now(timezone.utc)
        elapsed = (now - self.last_decay).total_seconds()
        if elapsed <= 0:
            return
        decay_factor = 0.5 ** (elapsed / self.half_life_seconds)
        self.signal_value *= decay_factor
        self.last_decay = now

    def current_value(self) -> float:
        """Return the decayed signal value (decays on‑demand)."""
        self.decay()
        return self.signal_value

# ----------------------------------------------------------------------
# Voronoi & Curvature utilities (Parent B)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Blade = FrozenSet[int]
Multivector = Dict[Blade, float]

def voronoi_regions(points: List[Point], seeds: List[Point]) -> List[int]:
    """
    Assign each point to the index of the nearest seed.
    Returns a list region_id per point.
    """
    regions = []
    seed_arr = np.array(seeds)  # shape (k, 2)
    for pt in points:
        diffs = seed_arr - np.array(pt)
        dists = np.einsum('ij,ij->i', diffs, diffs)  # squared Euclidean
        region = int(np.argmin(dists))
        regions.append(region)
    return regions

def dummy_ollivier_ricci_curvature(adjacency: np.ndarray) -> np.ndarray:
    """
    Placeholder curvature estimator:
    κ_i = 1 - (degree_i / max_degree)
    Returns a curvature vector per node.
    """
    degrees = adjacency.sum(axis=1)
    max_deg = degrees.max() if degrees.max() > 0 else 1.0
    curvature = 1.0 - (degrees / max_deg)
    return curvature

# ----------------------------------------------------------------------
# Schoolfield temperature model (Parent B)
# ----------------------------------------------------------------------
def schoolfield_rate(T_celsius: float,
                     T0: float = 10.0,
                     Ha: float = 8000.0,
                     Hd: float = 20000.0,
                     Tl: float = -10.0,
                     R: float = 8.314) -> float:
    """
    Developmental rate λ_T(T) based on the Schoolfield model.
    T is in °C; internal calculations use Kelvin.
    """
    T = T_celsius + 273.15
    T0K = T0 + 273.15
    TlK = Tl + 273.15
    exp_term = math.exp(-Ha / (R * T)) - math.exp(-Hd / (R * T))
    denom = 1.0 + math.exp((Tl - T_celsius) / 10.0)
    return (exp_term / denom) * (T / T0K)

# ----------------------------------------------------------------------
# Bandit structures (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def build_hybrid_network(input_dim: int, hidden_dim: int) -> TropicalNetwork:
    """Create a TropicalNetwork with random weights/biases."""
    rng = np.random.default_rng()
    weights = rng.normal(loc=0.0, scale=1.0, size=(hidden_dim, input_dim))
    biases = rng.normal(loc=0.0, scale=0.5, size=hidden_dim)
    return TropicalNetwork(weights, biases)

def initialise_pheromones(regions: int) -> List[PheromoneEntry]:
    """Create one pheromone entry per Voronoi region."""
    entries = []
    for r in range(regions):
        entry = PheromoneEntry(
            surface_key=f"region_{r}",
            signal_kind="chem",
            signal_value=random.uniform(0.5, 1.5),
            half_life_seconds=random.randint(30, 120)
        )
        entries.append(entry)
    return entries

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Compute D_KL(p‖q) with smoothing to avoid log(0)."""
    p_s = np.clip(p, eps, 1.0)
    q_s = np.clip(q, eps, 1.0)
    return float(np.sum(p_s * np.log(p_s / q_s)))

def hybrid_step(network: TropicalNetwork,
                input_vec: np.ndarray,
                points: List[Point],
                seed_points: List[Point],
                adjacency: np.ndarray,
                stores: np.ndarray,
                temperature_c: float,
                pheromones: List[PheromoneEntry],
                eta0: float = 0.1) -> Tuple[int, np.ndarray]:
    """
    Perform one hybrid iteration:
    1. Network → probability p.
    2. Voronoi partition → region ids.
    3. For each region r:
         * Build pheromone distribution q_r (normalize current pheromone values of that region).
         * Compute D_KL(p‖q_r).
         * Compute curvature κ_r.
         * Compute sigmoid‑scaled store σ(S_r).
         * Compute temperature scaling λ_T.
         * Assemble expected reward ȓ_r.
    4. Apply UCB bandit to select region a.
    5. Update store S_a and adjacency A using η = η0·λ_T·σ(S_a).
    Returns selected region index and updated store vector.
    """
    # 1. Network distribution
    p = network.probability_distribution(input_vec)

    # 2. Voronoi assignment
    region_ids = np.array(voronoi_regions(points, seed_points))

    num_regions = len(seed_points)

    # 3. Gather pheromone values per region
    q_matrix = np.zeros((num_regions, p.size))
    for r in range(num_regions):
        # collect pheromone values whose surface_key matches the region
        vals = np.array([ph.current_value()
                         for ph in pheromones
                         if ph.surface_key == f"region_{r}"])
        if vals.size == 0:
            vals = np.array([1.0])  # fallback
        # broadcast to same size as network output for KL (simple uniform split)
        q = np.full(p.shape, vals.mean())
        q /= q.sum()
        q_matrix[r] = q

    # 4. Curvature per region (use dummy estimator on adjacency)
    curvature = dummy_ollivier_ricci_curvature(adjacency)  # length = num_nodes
    # map nodes -> region (simple nearest‑seed mapping)
    node_region = np.arange(num_regions)  # assume one node per region for demo
    κ_region = curvature[node_region]

    # 5. Store sigmoid and temperature scaling
    σ_S = 1.0 / (1.0 + np.exp(-stores))  # element‑wise
    λ_T = schoolfield_rate(temperature_c)

    # 6. Compute reward per region
    rewards = np.empty(num_regions)
    for r in range(num_regions):
        D = kl_divergence(p, q_matrix[r])
        rewards[r] = κ_region[r] * σ_S[r] * math.exp(-D)

    # 7. Upper Confidence Bound (UCB) selection
    # propensity = 1 (uniform) for simplicity; confidence = sqrt(2*log(t)/n)
    t = 1  # time step placeholder
    n = np.maximum(1, np.arange(1, num_regions + 1))
    confidence = np.sqrt(2 * np.log(t + 1) / n)
    ucb = rewards + confidence
    selected_region = int(np.argmax(ucb))

    # 8. Update store and adjacency for selected region
    η = eta0 * λ_T * σ_S[selected_region]
    delta_store = rewards[selected_region] - 0.0  # baseline = 0
    stores[selected_region] += delta_store

    # adjacency update: increase weights of edges incident to the selected node
    adjacency[selected_region, :] *= (1.0 + η * delta_store)
    adjacency[:, selected_region] *= (1.0 + η * delta_store)

    return selected_region, stores

# ----------------------------------------------------------------------
# Demonstration / Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dimensions
    INPUT_DIM = 5
    HIDDEN_DIM = 8
    NUM_POINTS = 30
    NUM_REGIONS = 4

    # 1. Build network
    net = build_hybrid_network(INPUT_DIM, HIDDEN_DIM)

    # 2. Random input
    rng = np.random.default_rng()
    x = rng.uniform(-1, 1, size=INPUT_DIM)

    # 3. Generate spatial points and seeds
    points = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(NUM_POINTS)]
    seeds = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(NUM_REGIONS)]

    # 4. Initialise adjacency (simple undirected graph)
    A = np.eye(NUM_REGIONS)  # start with self‑loops only

    # 5. Initialise honeybee stores (one per region)
    stores = np.zeros(NUM_REGIONS)

    # 6. Initialise pheromones
    pheros = initialise_pheromones(NUM_REGIONS)

    # 7. Run a few hybrid steps
    for step in range(5):
        temp = 20.0 + 5 * math.sin(step)  # fluctuating temperature
        chosen, stores = hybrid_step(
            network=net,
            input_vec=x,
            points=points,
            seed_points=seeds,
            adjacency=A,
            stores=stores,
            temperature_c=temp,
            pheromones=pheros,
            eta0=0.05
        )
        print(f"Step {step}: selected region {chosen}, stores={stores.round(3)}")
    print("Smoke test completed without errors.")