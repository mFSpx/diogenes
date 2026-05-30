# DARWIN HAMMER — match 3342, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_sketch_epistemic_certainty_m2_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m449_s0.py (gen4)
# born: 2026-05-29T23:49:24Z

"""Hybrid Sheaf‑Certainty + Thompson‑Curvature‑Voronoi Module
Parents:
- hybrid_hybrid_hybrid_sketch_epistemic_certainty_m2_s3.py (sheaf sections + certainty scalars)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m449_s0.py (Thompson‑sampling bandit, Ollivier‑Ricci curvature, Voronoi partitioning)

Mathematical bridge:
Each vertex v carries a sheaf section s(v)∈ℝ^{d_v} together with a certainty c(v)∈[0,1].
For an edge e=(u,v) with restriction matrix R_{u→v},
the raw coboundary discrepancy is δ_e = R_{u→v}·s(u) – s(v)′.
We weight δ_e by the geometric mean √(c(u)c(v)) and by the Ollivier‑Ricci curvature κ_e
computed from the neighbourhood distributions of u and v.  The resulting scalar

    w_e = √(c(u)c(v))·(1+κ_e)·‖δ_e‖₂

feeds directly into a Thompson‑sampling bandit whose actions correspond to edges.
Thus the bandit’s posterior updates are driven by a confidence‑ and curvature‑aware
information‑loss metric, while the Voronoi partition of the feature space can be used
to initialise or cluster the sheaf vectors.  The three core functions below realise
this fusion."""

import math
import random
import sys
import pathlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Certainty infrastructure (from epistemic_certainty.py)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
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
            object.__setattr__(
                self,
                "generated_at",
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: Iterable[str] = (),
) -> CertaintyFlag:
    return CertaintyFlag(
        label=label,
        confidence_bps=confidence_bps,
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(evidence_refs),
    )


# ----------------------------------------------------------------------
# Sheaf‑section + certainty data structures
# ----------------------------------------------------------------------
@dataclass
class SheafVertex:
    """A vertex of the sheaf‑graph."""
    id: str
    section: np.ndarray                     # s(v) ∈ ℝ^{d}
    certainty: CertaintyFlag                # epistemic certainty


def certainty_scalar(flag: CertaintyFlag) -> float:
    """Map confidence basis points (0..10000) to a probability‑like scalar."""
    return flag.confidence_bps / 10000.0


# ----------------------------------------------------------------------
# Bandit infrastructure (from hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m449_s0.py)
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 with trailing “Z”."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "thompson_sampling"


@dataclass
class BanditUpdate:
    """Single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float = 1.0


class ThompsonBandit:
    """Lightweight Thompson‑sampling bandit for Bernoulli‑type rewards."""

    def __init__(self, actions: List[str], prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self._alpha: Dict[str, float] = {a: prior_alpha for a in actions}
        self._beta: Dict[str, float] = {a: prior_beta for a in actions}
        self._actions = actions

    def sample(self) -> BanditAction:
        """Draw a posterior sample for each action and return the best."""
        samples = {
            a: np.random.beta(self._alpha[a], self._beta[a]) for a in self._actions
        }
        best = max(samples, key=samples.get)
        # Propensity is the probability that this action would be drawn;
        # we approximate it with the empirical frequency over many draws.
        # Here we simply set it to the sampled value.
        return BanditAction(
            action_id=best,
            propensity=samples[best],
            expected_reward=samples[best],
            confidence_bound=1.0 / (self._alpha[best] + self._beta[best]),
        )

    def update(self, upd: BanditUpdate) -> None:
        """Bayesian update of Beta posterior."""
        if upd.reward > 0:
            self._alpha[upd.action_id] += upd.reward
        else:
            self._beta[upd.action_id] += 1 - upd.reward  # assume reward∈{0,1}
        # optional: incorporate propensity weighting (ignored for simplicity)


# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------
def weighted_coboundary_norm(
    vertices: Dict[str, SheafVertex],
    edges: List[Tuple[str, str]],
    restrictions: Dict[Tuple[str, str], np.ndarray],
) -> float:
    """
    Compute Σ_{(u,v)∈E} √(c_u·c_v)·‖R_{u→v}·s(u) – s(v)′‖₂.
    s(v)′ is the pull‑back of s(v) to the edge space via the transpose of R_{v→u}
    (if defined); otherwise we treat s(v)′ as s(v) padded/truncated to match dimensions.
    """
    total = 0.0
    for (u, v) in edges:
        ru = restrictions.get((u, v))
        if ru is None:
            raise ValueError(f"Missing restriction matrix for edge ({u},{v})")
        su = vertices[u].section
        sv = vertices[v].section

        # Pull‑back of sv to the source space (simple projection)
        # We use the pseudo‑inverse of ru for the pull‑back.
        try:
            ru_pinv = np.linalg.pinv(ru)
            sv_prime = ru_pinv @ sv
        except np.linalg.LinAlgError:
            sv_prime = sv  # fallback

        delta = ru @ su - sv_prime
        cu = certainty_scalar(vertices[u].certainty)
        cv = certainty_scalar(vertices[v].certainty)
        weight = math.sqrt(cu * cv)
        total += weight * np.linalg.norm(delta, 2)
    return total


def approximate_ollivier_ricci(
    vertices: Dict[str, SheafVertex],
    edges: List[Tuple[str, str]],
    neighbor_radius: int = 1,
) -> Dict[Tuple[str, str], float]:
    """
    Very coarse approximation of Ollivier‑Ricci curvature κ_e for each edge.
    For each vertex we construct a probability distribution over its immediate
    neighbours by normalising the Euclidean distances of their sheaf sections.
    The 1‑Wasserstein distance between the two distributions is then used:

        κ_e = 1 - W1(dist_u, dist_v) / d(u,v)

    Here d(u,v)=1 for all edges, so κ_e = 1 - W1.
    """
    # Build neighbour lists
    neigh: Dict[str, List[str]] = {vid: [] for vid in vertices}
    for (u, v) in edges:
        neigh[u].append(v)
        neigh[v].append(u)

    # Distribution over neighbours: weight inversely proportional to distance in section space
    distributions: Dict[str, np.ndarray] = {}
    for vid, nlist in neigh.items():
        if not nlist:
            distributions[vid] = np.array([1.0])  # isolated node
            continue
        dists = np.array(
            [np.linalg.norm(vertices[vid].section - vertices[n].section, 2) for n in nlist]
        )
        # Avoid division by zero
        inv = 1.0 / (dists + 1e-12)
        prob = inv / inv.sum()
        distributions[vid] = prob

    # Compute κ for each edge
    curvature: Dict[Tuple[str, str], float] = {}
    for (u, v) in edges:
        pu = distributions[u]
        pv = distributions[v]
        # Align lengths by padding with zeros (very rough)
        maxlen = max(len(pu), len(pv))
        pu_pad = np.pad(pu, (0, maxlen - len(pu)), constant_values=0.0)
        pv_pad = np.pad(pv, (0, maxlen - len(pv)), constant_values=0.0)
        w1 = np.linalg.norm(pu_pad - pv_pad, 1) / 2.0  # L1 distance /2 = earth mover's distance for 1‑D
        curvature[(u, v)] = 1.0 - w1
    return curvature


def hybrid_thompson_step(
    vertices: Dict[str, SheafVertex],
    edges: List[Tuple[str, str]],
    restrictions: Dict[Tuple[str, str], np.ndarray],
    bandit: ThompsonBandit,
) -> BanditAction:
    """
    Perform a single Thompson‑sampling decision where each edge is an action.
    The posterior parameters for an edge are nudged by a hybrid metric:

        α_e ← 1 + w_e
        β_e ← 1 + (1 - κ_e)

    where w_e is the confidence‑weighted coboundary norm for edge e and κ_e its
    curvature.  The bandit then samples and returns the chosen action.
    """
    # Compute hybrid edge weights
    w_norm = weighted_coboundary_norm(vertices, edges, restrictions)
    curv = approximate_ollivier_ricci(vertices, edges)

    # Update bandit priors on‑the‑fly (simple additive scheme)
    for (u, v) in edges:
        edge_id = f"{u}->{v}"
        # Edge‑specific norm contribution (proportional share)
        edge_norm = w_norm / len(edges)
        κ = curv.get((u, v), 0.0)
        # Adjust priors directly in the bandit's internal dicts
        bandit._alpha[edge_id] = bandit._alpha.get(edge_id, 1.0) + edge_norm
        bandit._beta[edge_id] = bandit._beta.get(edge_id, 1.0) + (1.0 - κ)

    # Sample the best action according to the updated posteriors
    return bandit.sample()


def voronoi_partition(
    points: np.ndarray,
    seeds: np.ndarray,
) -> np.ndarray:
    """
    Very lightweight Voronoi partitioning: assign each point to the nearest seed.
    Returns an array of region indices of shape (n_points,).
    """
    # Compute squared Euclidean distances for efficiency
    dists = np.sum((points[:, None, :] - seeds[None, :, :]) ** 2, axis=2)
    return np.argmin(dists, axis=1)


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
def _build_demo_graph() -> Tuple[
    Dict[str, SheafVertex],
    List[Tuple[str, str]],
    Dict[Tuple[str, str], np.ndarray],
]:
    """Construct a tiny three‑node graph with random sections and certainties."""
    rng = np.random.default_rng(seed=42)
    vertices: Dict[str, SheafVertex] = {}
    for vid in ("A", "B", "C"):
        dim = rng.integers(2, 5)               # random dimension 2‑4
        section = rng.normal(size=dim)
        flag = certainty(
            label="FACT",
            confidence_bps=rng.integers(3000, 10000),
            authority_class="demo",
            rationale="synthetic",
        )
        vertices[vid] = SheafVertex(id=vid, section=section, certainty=flag)

    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    restrictions: Dict[Tuple[str, str], np.ndarray] = {}
    for (u, v) in edges:
        du = vertices[u].section.shape[0]
        dv = vertices[v].section.shape[0]
        # Random linear map from ℝ^{du} → ℝ^{dv}
        restrictions[(u, v)] = rng.normal(size=(dv, du))
        # Also store the opposite direction for completeness
        restrictions[(v, u)] = rng.normal(size=(du, dv))
    return vertices, edges, restrictions


def main() -> None:
    # Build demo data structures
    vertices, edges, restrictions = _build_demo_graph()

    # Initialise a Thompson bandit with one action per directed edge
    actions = [f"{u}->{v}" for (u, v) in edges]
    bandit = ThompsonBandit(actions=actions, prior_alpha=1.0, prior_beta=1.0)

    # Perform a hybrid step
    chosen = hybrid_thompson_step(vertices, edges, restrictions, bandit)

    # Print results
    print("Chosen action:", chosen.action_id)
    print("Propensity (sampled posterior):", chosen.propensity)
    print("Expected reward (sampled):", chosen.expected_reward)

    # Demonstrate Voronoi partitioning on the raw section vectors
    all_sections = np.vstack([v.section for v in vertices.values()])
    # Pick two random seeds from the existing sections
    seed_indices = np.random.choice(all_sections.shape[0], size=2, replace=False)
    seeds = all_sections[seed_indices]
    regions = voronoi_partition(all_sections, seeds)
    print("Voronoi region assignment:", dict(zip(vertices.keys(), regions)))


if __name__ == "__main__":
    main()