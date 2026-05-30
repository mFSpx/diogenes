# DARWIN HAMMER — match 48, survivor 5
# gen: 2
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s0.py (gen1)
# parent_b: epistemic_certainty.py (gen0)
# born: 2026-05-29T23:23:58Z

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple, Union

# ----------------------------------------------------------------------
# Epistemic certainty helpers (parent B)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


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
        if not 0 <= int(self.confidence_bps) <= 10_000:
            raise ValueError("confidence_bps must be in 0..10000")
        if not self.generated_at:
            object.__setattr__(
                self,
                "generated_at",
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            )

    def as_dict(self) -> Dict[str, Union[str, int, Tuple[str, ...]]]:
        return {
            "label": self.label,
            "confidence_bps": self.confidence_bps,
            "authority_class": self.authority_class,
            "rationale": self.rationale,
            "evidence_refs": self.evidence_refs,
            "generated_at": self.generated_at,
        }


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
        confidence_bps=int(confidence_bps),
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(str(x) for x in evidence_refs if x is not None),
    )


def filesystem_observation(*, sha256: str, path: str, mtime_utc: str | None = None) -> CertaintyFlag:
    refs = [f"sha256:{sha256}", f"path:{path}"]
    if mtime_utc:
        refs.append(f"mtime:{mtime_utc}")
    return certainty(
        "FACT",
        confidence_bps=10_000,
        authority_class="filesystem_observation",
        rationale="Local file bytes were hashed and copied into CAS; this proves byte custody, not semantic truth.",
        evidence_refs=refs,
    )


def parser_extraction(*, sha256: str, extract_method: str, injection_detected: bool = False) -> CertaintyFlag:
    if injection_detected:
        return certainty(
            "BULLSHIT",
            confidence_bps=9_000,
            authority_class="prompt_injection_signature",
            rationale="Untrusted source text matched instruction‑injection signatures; preserve bytes but treat embedded directives as hostile data.",
            evidence_refs=[f"sha256:{sha256}", f"extract:{extract_method}"],
        )
    return certainty(
        "PROBABLE",
        confidence_bps=6_500,
        authority_class="parser_extraction",
        rationale="Text was deterministically extracted from a local artifact; semantic claims remain unverified until corroborated.",
        evidence_refs=[f"sha256:{sha256}", f"extract:{extract_method}"],
    )


def abductive_hypothesis(*, evidence_refs: Iterable[str] = (), rationale: str = "Abductive bridge candidate") -> CertaintyFlag:
    return certainty(
        "POSSIBLE",
        confidence_bps=3_500,
        authority_class="abductive_hypothesis",
        rationale=rationale,
        evidence_refs=evidence_refs,
    )


def comment_prior(*, evidence_refs: Iterable[str] = (), rationale: str = "Operator or system comment; useful but not proof") -> CertaintyFlag:
    return certainty(
        "SURE_MAYBE",
        confidence_bps=2_500,
        authority_class="comment_prior",
        rationale=rationale,
        evidence_refs=evidence_refs,
    )


def contradiction(*, evidence_refs: Iterable[str] = (), rationale: str = "Contradiction or falsification marker") -> CertaintyFlag:
    return certainty(
        "BULLSHIT",
        confidence_bps=8_000,
        authority_class="contradiction_scan",
        rationale=rationale,
        evidence_refs=evidence_refs,
    )


# ----------------------------------------------------------------------
# Bayesian helpers (parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]


def euclidean(a: Point, b: Point) -> float:
    """Straight‑line distance."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = P(E|H)·P(H) + P(E|¬H)·(1‑P(H))."""
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = P(E|H)·P(H) / P(E)."""
    if marginal == 0.0:
        return 0.0
    return (likelihood * prior) / marginal


# ----------------------------------------------------------------------
# Hybrid utilities – deeper integration
# ----------------------------------------------------------------------
def confidence_to_probability(flag: CertaintyFlag) -> float:
    """Convert basis‑point confidence to a probability in [0, 1]."""
    return flag.confidence_bps / 10_000.0


def _lookup_edge_flag(
    edge: Edge,
    edge_flags: Dict[Edge, CertaintyFlag],
) -> CertaintyFlag:
    """Return the flag for an undirected edge, trying both orientations."""
    if edge in edge_flags:
        return edge_flags[edge]
    rev = (edge[1], edge[0])
    if rev in edge_flags:
        return edge_flags[rev]
    raise KeyError(f"Edge flag missing for {edge} (or its reverse)")


def compute_edge_posterior(
    source_node: str,
    target_node: str,
    node_flags: Dict[str, CertaintyFlag],
    edge_flags: Dict[Edge, CertaintyFlag],
    false_positive: float,
) -> float:
    """
    Compute the Bayesian posterior weight for a directed traversal
    from ``source_node`` to ``target_node``.
    """
    prior = confidence_to_probability(node_flags[source_node])
    likelihood = confidence_to_probability(
        _lookup_edge_flag((source_node, target_node), edge_flags)
    )
    marginal = bayes_marginal(prior, likelihood, false_positive)
    return bayes_update(prior, likelihood, marginal)


@dataclass
class HybridTree:
    nodes: Dict[str, Point]
    edges: List[Edge]
    root: str
    node_flags: Dict[str, CertaintyFlag]
    edge_flags: Dict[Edge, CertaintyFlag]
    false_positive_map: Dict[Edge, float] = field(default_factory=dict)
    path_weight: float = 0.2

    def __post_init__(self) -> None:
        # Ensure every edge has a false‑positive entry; default to 0.05.
        default_fp = 0.05
        for e in self.edges:
            if e not in self.false_positive_map and (e[1], e[0]) not in self.false_positive_map:
                self.false_positive_map[e] = default_fp

        # Build adjacency list for undirected traversal.
        self._adj: Dict[str, List[str]] = {n: [] for n in self.nodes}
        for a, b in self.edges:
            self._adj[a].append(b)
            self._adj[b].append(a)

    def _traverse(self) -> Tuple[float, float]:
        """
        Depth‑first walk from ``root`` accumulating:
        * ``material`` – sum of Euclidean distances weighted by posteriors.
        * ``path_penalty`` – sum of (depth * path_weight).
        Returns a tuple (material, path_penalty).
        """
        visited = set()
        stack: List[Tuple[str, int, float]] = [(self.root, 0, 1.0)]  # node, depth, cumulative_posterior
        material = 0.0
        path_penalty = 0.0

        while stack:
            node, depth, cum_post = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            path_penalty += depth * self.path_weight

            for neighbor in self._adj[node]:
                if neighbor in visited:
                    continue
                # Posterior for this directed edge
                fp = self.false_positive_map.get((node, neighbor), self.false_positive_map.get((neighbor, node), 0.05))
                post = compute_edge_posterior(
                    source_node=node,
                    target_node=neighbor,
                    node_flags=self.node_flags,
                    edge_flags=self.edge_flags,
                    false_positive=fp,
                )
                # Weight distance by posterior and propagate cumulative factor
                dist = euclidean(self.nodes[node], self.nodes[neighbor])
                weighted = dist * post
                material += weighted * cum_post
                stack.append((neighbor, depth + 1, cum_post * post))

        return material, path_penalty

    def total_cost(self) -> float:
        """Hybrid cost = material component + path penalty."""
        material, path_penalty = self._traverse()
        return material + path_penalty

    def aggregate_certainty(self) -> float:
        """
        Compute a global certainty for the whole tree.
        Uses log‑space to avoid underflow:
        log_cert = Σ log(posterior_edge)
        Returns exp(log_cert) ∈ [0,1].
        """
        log_sum = 0.0
        visited_edges = set()
        for a, b in self.edges:
            # Ensure each undirected edge is counted once
            edge_key = tuple(sorted((a, b)))
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)

            fp = self.false_positive_map.get((a, b), self.false_positive_map.get((b, a), 0.05))
            post_ab = compute_edge_posterior(a, b, self.node_flags, self.edge_flags, fp)
            post_ba = compute_edge_posterior(b, a, self.node_flags, self.edge_flags, fp)
            # Symmetrize by averaging in log‑space
            avg_log = 0.5 * (math.log(max(post_ab, 1e-12)) + math.log(max(post_ba, 1e-12)))
            log_sum += avg_log

        return math.exp(log_sum)


# ----------------------------------------------------------------------
# Public API – thin wrappers for backward compatibility
# ----------------------------------------------------------------------
def hybrid_tree_cost_with_certainty(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    node_flags: Dict[str, CertaintyFlag],
    edge_flags: Dict[Edge, CertaintyFlag],
    false_positive_map: Dict[Edge, float] | None = None,
    path_weight: float = 0.2,
) -> float:
    """
    Backward‑compatible wrapper that builds a :class:`HybridTree` and returns its total cost.
    """
    tree = HybridTree(
        nodes=nodes,
        edges=edges,
        root=root,
        node_flags=node_flags,
        edge_flags=edge_flags,
        false_positive_map=false_positive_map or {},
        path_weight=path_weight,
    )
    return tree.total_cost()


def aggregate_tree_certainty(
    nodes: Dict[str, Point],
    edges: List[Edge],
    node_flags: Dict[str, CertaintyFlag],
    edge_flags: Dict[Edge, CertaintyFlag],
    false_positive_map: Dict[Edge, float] | None = None,
) -> float:
    """
    Backward‑compatible wrapper that returns the global certainty of a tree.
    """
    tree = HybridTree(
        nodes=nodes,
        edges=edges,
        root=next(iter(nodes)),  # root is irrelevant for aggregate certainty
        node_flags=node_flags,
        edge_flags=edge_flags,
        false_positive_map=false_positive_map or {},
    )
    return tree.aggregate_certainty()