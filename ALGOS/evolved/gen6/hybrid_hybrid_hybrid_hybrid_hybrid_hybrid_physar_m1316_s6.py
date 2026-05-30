# DARWIN HAMMER — match 1316, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_sketch_hybrid_hybrid_minimu_m662_s2.py (gen3)
# parent_b: hybrid_hybrid_physarum_netw_hybrid_infotaxis_min_m967_s1.py (gen5)
# born: 2026-05-29T23:35:11Z

"""Hybrid Physarum‑Sheaf / Minimum‑Cost Tree with Epistemic Certainty and Sketch‑MinHash Fusion

Parents:
- hybrid_hybrid_hybrid_sketch_hybrid_hybrid_minimu_m662_s2.py (Count‑Min sketch, MinHash LSH,
  Bayesian edge scoring, epistemic certainty flags)
- hybrid_hybrid_physarum_netw_hybrid_infotaxis_min_m967_s1.py (Physarum‑Sheaf dynamics,
  MinHash‑based infotaxis, CertaintyFlag dataclass)

Mathematical Bridge:
The edge weight w_uv used in the minimum‑cost tree is modulated by three
quantities that originate from the two parent systems:

1. **Epistemic certainty** – each node carries a `CertaintyFlag`.  Its confidence
   (basis points) is normalised to [0,1] and denoted ϕ_u.  The combined certainty
   for an edge is ϕ_uv = (ϕ_u + ϕ_v)/2.

2. **Sketch‑derived sparsity** – a Count‑Min sketch of the incident tokens of a
   node yields an estimate s_u of the frequency of the most common token.
   The sparsity factor σ_uv = 1 / (1 + (s_u + s_v)/2) down‑weights edges that
   connect noisy/high‑frequency regions.

3. **Information similarity** – MinHash signatures of node neighbourhoods give
   a Jaccard‑like similarity J_uv.  In the Physarum‑Sheaf equations the
   transport gain α_uv = α₀·J_uv modulates the flux.

The hybrid edge weight is therefore

    w_uv = ℓ_uv · (1 – ϕ_uv) · σ_uv

where ℓ_uv is the Euclidean distance between node positions.  This weight feeds
into both the Bayesian edge‑score update (parent A) and the Physarum flux
computation (parent B).  The resulting flux q_uv drives the sheaf‑section update
while the Bayesian marginal updates the edge probabilities, closing the loop.

The module below implements this fused system with three public functions that
demonstrate the combined workflow.
"""

import math
import random
import sys
import pathlib
import hashlib
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Tuple, List, Dict, Iterable, Any

import numpy as np

# ----------------------------------------------------------------------
# Epistemic certainty infrastructure (from parent B)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int  # 0 .. 10000 (basis points)
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")
        if not self.generated_at:
            object.__setattr__(self, "generated_at",
                               datetime.now(timezone.utc).isoformat() + "Z")

# ----------------------------------------------------------------------
# Sketch / MinHash utilities (from parent A & B)
# ----------------------------------------------------------------------
def count_min_sketch(items: Iterable[Any], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Return a Count‑Min sketch table for the supplied items."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16)
            table[d][h % width] += 1
    return table

def minhash_signature(tokens: Iterable[str], num_perm: int = 64) -> Tuple[int, ...]:
    """Very light MinHash: for each permutation seed keep the minimum hash."""
    sig = []
    for seed in range(num_perm):
        min_h = None
        for t in tokens:
            h = int(hashlib.blake2b((seed.to_bytes(4, "big") + t.encode()), digest_size=8).hexdigest(), 16)
            if min_h is None or h < min_h:
                min_h = h
        sig.append(min_h if min_h is not None else 0)
    return tuple(sig)

def jaccard_from_minhash(sig1: Tuple[int, ...], sig2: Tuple[int, ...]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig1) != len(sig2):
        raise ValueError("signatures must be same length")
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)

# ----------------------------------------------------------------------
# Geometry & Bayesian helpers (from parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

def euclidean_length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    if marginal <= 0.0:
        raise ValueError("marginal must be > 0")
    return (likelihood * prior) / marginal

# ----------------------------------------------------------------------
# Core hybrid structures
# ----------------------------------------------------------------------
class HybridNetwork:
    """Container for nodes, positions, certainty, sketches, and sheaf sections."""

    def __init__(self,
                 positions: Dict[str, Point],
                 tokens: Dict[str, List[str]],
                 certainty: Dict[str, CertaintyFlag]):
        self.positions = positions                     # node -> (x,y)
        self.tokens = tokens                           # node -> list of tokens (for sketches)
        self.certainty = certainty                     # node -> CertaintyFlag
        # Pre‑compute sketches and MinHash signatures
        self.sketches: Dict[str, List[List[int]]] = {
            n: count_min_sketch(tok) for n, tok in tokens.items()
        }
        self.minhash_sigs: Dict[str, Tuple[int, ...]] = {
            n: minhash_signature(tok) for n, tok in tokens.items()
        }
        # Sheaf sections are vectors; initialise to normalized MinHash signatures
        dim = len(next(iter(self.minhash_sigs.values())))
        self.sheaf: Dict[str, np.ndarray] = {
            n: np.array(sig, dtype=float) / (2**64 - 1) for n, sig in self.minhash_sigs.items()
        }

    # ------------------------------------------------------------------
    # Hybrid edge weight computation
    # ------------------------------------------------------------------
    def edge_weight(self, u: str, v: str) -> float:
        """Hybrid weight w_uv = ℓ_uv · (1‑ϕ_uv) · σ_uv."""
        # Euclidean distance
        ℓ = euclidean_length(self.positions[u], self.positions[v])

        # Epistemic certainty factor ϕ_uv
        φ_u = self.certainty[u].confidence_bps / 10000.0
        φ_v = self.certainty[v].confidence_bps / 10000.0
        φ_uv = (φ_u + φ_v) / 2.0

        # Sketch sparsity σ_uv
        # Estimate most frequent token count from the first row of each sketch
        s_u = max(self.sketches[u][0]) if self.sketches[u] else 0
        s_v = max(self.sketches[v][0]) if self.sketches[v] else 0
        σ = 1.0 / (1.0 + (s_u + s_v) / 2.0)

        return ℓ * (1.0 - φ_uv) * σ

    # ------------------------------------------------------------------
    # Physarum‑Sheaf flux computation
    # ------------------------------------------------------------------
    def flux_matrix(self, α0: float = 0.5) -> Dict[Tuple[str, str], float]:
        """Compute flux q_uv for every unordered node pair."""
        flux: Dict[Tuple[str, str], float] = {}
        nodes = list(self.positions.keys())
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                w = self.edge_weight(u, v)
                # Jaccard similarity from MinHash signatures
                J = jaccard_from_minhash(self.minhash_sigs[u], self.minhash_sigs[v])
                α = α0 * J
                # Simple Physarum flux: q = α * (potential_u - potential_v)
                # Here we use the sheaf vector norm as a proxy for potential.
                pot_u = np.linalg.norm(self.sheaf[u])
                pot_v = np.linalg.norm(self.sheaf[v])
                q = α * (pot_u - pot_v) * w
                flux[(u, v)] = q
                flux[(v, u)] = -q  # antisymmetric
        return flux

    # ------------------------------------------------------------------
    # Sheaf section update using Physarum dynamics
    # ------------------------------------------------------------------
    def update_sheaf(self,
                     flux: Dict[Tuple[str, str], float],
                     step: float = 0.01) -> None:
        """Perform one explicit Euler step of the sheaf dynamics."""
        delta: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros_like(next(iter(self.sheaf.values()))))
        for (u, v), q in flux.items():
            if q == 0.0:
                continue
            diff = self.sheaf[v] - self.sheaf[u]
            delta[u] += q * diff
        # Apply updates
        for node, inc in delta.items():
            self.sheaf[node] += step * inc
            # Re‑normalise to keep values bounded
            norm = np.linalg.norm(self.sheaf[node])
            if norm > 0:
                self.sheaf[node] /= norm

    # ------------------------------------------------------------------
    # Bayesian edge probability update
    # ------------------------------------------------------------------
    def bayesian_edge_probs(self,
                            prior: float = 0.5,
                            false_positive: float = 0.1) -> Dict[Tuple[str, str], float]:
        """Return posterior probabilities for each edge being in the minimum‑cost tree."""
        probs: Dict[Tuple[str, str], float] = {}
        for (u, v) in self.edge_set():
            w = self.edge_weight(u, v)
            # Interpret weight as likelihood: larger weight → less likely to be in min‑cost tree
            likelihood = math.exp(-w)  # simple monotonic mapping
            marginal = bayes_marginal(prior, likelihood, false_positive)
            posterior = bayes_update(prior, likelihood, marginal)
            probs[(u, v)] = posterior
        return probs

    def edge_set(self) -> List[Tuple[str, str]]:
        nodes = list(self.positions.keys())
        return [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]

# ----------------------------------------------------------------------
# Public API – three demonstrative functions
# ----------------------------------------------------------------------
def build_hybrid_network() -> HybridNetwork:
    """Construct a tiny synthetic network with random geometry and tokens."""
    random.seed(42)
    positions = {
        f"N{i}": (random.uniform(0, 10), random.uniform(0, 10))
        for i in range(5)
    }
    # Each node gets a bag of tokens; some overlap to create similarity.
    base_tokens = ["alpha", "beta", "gamma", "delta", "epsilon"]
    tokens = {}
    for node in positions:
        # Random subset plus a node‑specific token
        subset = random.sample(base_tokens, k=3)
        subset.append(f"token_{node}")
        tokens[node] = subset
    # Certainty flags with varying confidence
    flags = {}
    for i, node in enumerate(positions):
        label = EPISTEMIC_FLAGS[i % len(EPISTEMIC_FLAGS)]
        confidence = (i + 1) * 2000  # 2000, 4000, … up to 10000
        flags[node] = CertaintyFlag(label=label,
                                    confidence_bps=confidence,
                                    authority_class="synthetic",
                                    rationale="demo")
    return HybridNetwork(positions, tokens, flags)

def run_hybrid_iteration(net: HybridNetwork) -> None:
    """Execute one full hybrid iteration: flux → sheaf update → Bayesian probs."""
    flux = net.flux_matrix(α0=0.7)
    net.update_sheaf(flux, step=0.05)
    posteriors = net.bayesian_edge_probs(prior=0.6, false_positive=0.05)
    # Print a concise summary
    print("Edge posteriors (sorted):")
    for (u, v), p in sorted(posteriors.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {u}-{v}: {p:.3f}")

def extract_sheaf_vectors(net: HybridNetwork) -> Dict[str, np.ndarray]:
    """Return a copy of the current sheaf sections for external analysis."""
    return {n: vec.copy() for n, vec in net.sheaf.items()}

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    network = build_hybrid_network()
    print("Initial sheaf vectors (norms):")
    for n, vec in network.sheaf.items():
        print(f"  {n}: {np.linalg.norm(vec):.3f}")
    for iteration in range(3):
        print(f"\n--- Iteration {iteration + 1} ---")
        run_hybrid_iteration(network)
    # Final sheaf snapshot
    final = extract_sheaf_vectors(network)
    print("\nFinal sheaf vector norms:")
    for n, vec in final.items():
        print(f"  {n}: {np.linalg.norm(vec):.3f}")