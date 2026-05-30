# DARWIN HAMMER — match 1525, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_fisher_m22_s2.py (gen4)
# parent_b: distributed_leader_election.py (gen0)
# born: 2026-05-29T23:37:03Z

"""Hybrid algorithm merging pheromone‑entropy/Fisher information (Parent A) with
local randomized leader election (Parent B).

Mathematical bridge:
- From Parent A we obtain a probability distribution **p** over recent pheromone
  signals. Its entropy *H(p)* and Fisher information *I(p)* quantify uncertainty
  and sensitivity of the distribution.
- From Parent B the broadcast probability *b(phase,step)=1/2^{phase‑step}* drives
  the random selection of candidate leaders.
- The hybrid replaces *b* with a *information‑aware* broadcast probability  

  **b̂ = b · (1 + λ·I(p))**,  

  where λ∈[0,1] scales the influence of Fisher information.  Thus, when the
  pheromone distribution is highly informative (large *I(p)*) the algorithm
  becomes more aggressive in electing leaders, otherwise it behaves like the
  original maximal‑independent‑set routine.

- Additionally, each node is embedded into a ternary‑lens vector (values
  -1,0,1) derived from a hash of the node identifier.  The dot product of two
  such vectors provides a lightweight “decision‑hygiene” score that can be
  used to break ties when multiple candidates compete.

The code below implements this hybrid, exposing three core functions:
`calculate_pheromone_probabilities`, `fisher_information`, and
`maximal_independent_set_hybrid`.  A small smoke test runs when the module is
executed directly."""


import math
import random
import sys
from pathlib import Path
import re
from collections.abc import Mapping, Hashable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – pheromone handling utilities
# ----------------------------------------------------------------------
TERNARY_DIMS = 12

_REGEX_CATALOG = [
    re.compile(
        r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
        r"sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|"
        r"triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|"
        r"first|after|review)\b",
        re.I,
    ),
]


def calculate_pheromone_probabilities(surface_key: str, limit: int, db_url: str | None = None):
    """
    Return the most recent ``limit`` pheromone signal values for ``surface_key``
    normalized to a probability distribution.

    If ``db_url`` is falsy (e.g. during a smoke test) a synthetic uniform
    distribution of length ``limit`` is returned.
    """
    if not db_url:
        # Fallback: uniform synthetic distribution
        return [1.0 / limit] * limit

    # Real implementation would query a PostgreSQL database.
    # The import is kept inside the function to avoid mandatory external
    # dependencies when the fallback path is used.
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT signal_value FROM lucidota_runtime.surface_pheromone
                   WHERE surface_key=%s ORDER BY created_at DESC LIMIT %s""",
                (surface_key, limit),
            )
            rows = cur.fetchall()
            pheromones = [r["signal_value"] for r in rows]

    total = sum(pheromones)
    if total == 0:
        raise ValueError("Pheromone sum is zero, cannot normalize.")
    return [p / total for p in pheromones]


def entropy(probabilities, eps: float = 1e-12) -> float:
    """
    Shannon entropy of a discrete probability distribution.
    """
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    return -sum(
        (p / total) * math.log(max(p / total, eps)) for p in probabilities
    )


# ----------------------------------------------------------------------
# Fisher information derived from a probability vector
# ----------------------------------------------------------------------
def fisher_information(probabilities, eps: float = 1e-12) -> float:
    """
    Compute a scalar Fisher information measure for a discrete distribution.

    For a categorical distribution with parameters θ_i = p_i, the Fisher
    information matrix is diagonal with entries (∂log p_i / ∂θ_i)^2 * p_i
    = 1 / p_i.  Summing the diagonal yields Σ 1/p_i, which diverges for very
    small p_i.  To keep the quantity bounded we use a softened version:

        I(p) = Σ ( (∂p_i/∂θ_i)^2 / (p_i + eps) )
              ≈ Σ ( (1 - p_i)^2 / (p_i + eps) )

    This captures how sensitive the distribution is to perturbations while
    remaining numerically stable.
    """
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    probs = np.array(probabilities) / total
    info = np.sum(((1.0 - probs) ** 2) / (probs + eps))
    return float(info)


# ----------------------------------------------------------------------
# Ternary lens embedding (decision hygiene)
# ----------------------------------------------------------------------
def ternary_lens_vector(node: Hashable, dims: int = TERNARY_DIMS) -> np.ndarray:
    """
    Map a node identifier to a fixed‑length ternary vector with entries
    -1, 0, or 1.  The mapping is deterministic via Python's built‑in hash.
    """
    h = hash(node)
    vec = np.empty(dims, dtype=int)
    for i in range(dims):
        # Extract two bits at a time and map to -1,0,1
        digit = (h >> (2 * i)) & 0b11
        vec[i] = -1 if digit == 0 else (1 if digit == 1 else 0)
    return vec


def hygiene_score(node_a: Hashable, node_b: Hashable) -> float:
    """
    Simple similarity score based on the dot product of ternary lens vectors.
    Higher values indicate more compatible decisions.
    """
    v_a = ternary_lens_vector(node_a)
    v_b = ternary_lens_vector(node_b)
    return float(np.dot(v_a, v_b))


# ----------------------------------------------------------------------
# Parent B – leader election utilities (adapted)
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, set[Node]]


def broadcast_probability(phase: int, step: int) -> float:
    """Return p = 1 / 2^{phase‑step}, clamped to [0, 1]."""
    if phase < 1 or step < 1:
        raise ValueError("phase and step must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))


def hybrid_broadcast_probability(
    phase: int,
    step: int,
    fisher_scalar: float,
    lam: float = 0.5,
) -> float:
    """
    Information‑aware broadcast probability.

    ``lam`` controls the strength of the Fisher‑information term (0 → original
    behaviour, 1 → maximal influence).  The scalar is normalised by the
    maximum possible Fisher information for the given distribution size to keep
    the factor in a reasonable range.
    """
    base = broadcast_probability(phase, step)
    # Normalise fisher_scalar to [0,1] using a simple heuristic:
    norm_fisher = math.tanh(fisher_scalar / 10.0)  # smooth saturation
    return min(1.0, base * (1.0 + lam * norm_fisher))


def maximal_independent_set_hybrid(
    graph: Graph,
    phases: int = 8,
    seed: int | str | None = None,
    pheromone_probs: list[float] | None = None,
    lam: float = 0.5,
) -> set[Node]:
    """
    Hybrid maximal independent set (MIS) algorithm.

    - ``pheromone_probs`` supplies the pheromone distribution for the current
      surface.  If omitted, a uniform distribution of length ``phases`` is used.
    - Fisher information derived from that distribution modulates the broadcast
      probability via ``hybrid_broadcast_probability``.
    - When multiple candidates broadcast simultaneously, the node with the
      highest ``hygiene_score`` against the current leader set wins, providing a
      deterministic tie‑breaker.
    """
    rng = random.Random(seed)
    undecided = set(graph)
    leaders: set[Node] = set()
    blocked: set[Node] = set()

    # Default pheromone distribution if none supplied
    if pheromone_probs is None:
        pheromone_probs = [1.0 / phases] * phases

    fisher_scalar = fisher_information(pheromone_probs)

    for phase in range(1, phases + 1):
        if not undecided:
            break

        p = hybrid_broadcast_probability(phases, phase, fisher_scalar, lam=lam)
        broadcasts = {n for n in undecided if rng.random() < p}

        # Resolve conflicts using hygiene scores
        new_leaders: set[Node] = set()
        for n in broadcasts:
            # n is a candidate if none of its neighbours also broadcast
            if not (graph.get(n, set()) & broadcasts):
                new_leaders.add(n)
            else:
                # Conflict: pick the node with the best hygiene score against
                # the already‑selected leaders (or against itself if none yet)
                competitors = {n} | (graph.get(n, set()) & broadcasts)
                best = max(
                    competitors,
                    key=lambda cand: hygiene_score(cand, tuple(leaders)[0])
                    if leaders
                    else 0,
                )
                new_leaders.add(best)

        leaders.update(new_leaders)

        # Block neighbours of newly elected leaders
        newly_blocked = set().union(
            *(graph.get(n, set()) for n in new_leaders), new_leaders
        )
        blocked.update(newly_blocked)
        undecided -= blocked

    # Final sweep for any remaining undecided nodes
    for n in sorted(undecided, key=str):
        if not (graph.get(n, set()) & leaders):
            leaders.add(n)

    return leaders


# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Construct a tiny graph
    example_graph: Graph = {
        "A": {"B", "C"},
        "B": {"A", "D"},
        "C": {"A", "D"},
        "D": {"B", "C"},
        "E": set(),
    }

    # Synthetic pheromone distribution (e.g., recent signal strengths)
    synthetic_pheromones = [0.2, 0.1, 0.4, 0.15, 0.15]

    # Run the hybrid MIS algorithm
    leaders = maximal_independent_set_hybrid(
        example_graph,
        phases=5,
        seed=42,
        pheromone_probs=synthetic_pheromones,
        lam=0.7,
    )

    print("Selected leaders:", leaders)

    # Demonstrate entropy and Fisher information on the same distribution
    ent = entropy(synthetic_pheromones)
    fisher = fisher_information(synthetic_pheromones)
    print(f"Entropy: {ent:.4f}, Fisher information: {fisher:.4f}")

    # Show a hygiene score between two nodes
    print("Hygiene score A vs B:", hygiene_score("A", "B"))