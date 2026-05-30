# DARWIN HAMMER — match 1195, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_bayes_claim_k_m166_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s3.py (gen3)
# born: 2026-05-29T23:33:37Z

import math
import numpy as np
from dataclasses import dataclass, replace
from typing import List, Tuple, Dict, Callable, Optional


# ----------------------------------------------------------------------
# 1. Probabilistic acceptance (Parent A)
# ----------------------------------------------------------------------
def acceptance_probability(delta_energy: float, temperature: float) -> float:
    """Metropolis‑style acceptance probability.

    Returns a value in (0, 1] (never exactly 0 to keep log‑domain safe)."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    # Clamp to avoid exp(-inf) = 0 which would break log‑domain later
    prob = math.exp(-delta_energy / temperature)
    return max(prob, 1e-12)


def hoeffding_decision(num_samples: int, epsilon: float, delta: float = 0.05) -> bool:
    """Hoeffding bound decision: True if the bound is tighter than *epsilon*."""
    if num_samples <= 0:
        return False
    bound = math.sqrt(math.log(2.0 / delta) / (2.0 * num_samples))
    return bound < epsilon


# ----------------------------------------------------------------------
# 2. Bayesian edge reliability (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class EdgeBetaPrior:
    """Beta prior parameters for a Bernoulli edge reliability."""
    alpha: float = 1.0
    beta: float = 1.0


def bayesian_edge_update(
    prior: EdgeBetaPrior,
    successes: int,
    failures: int,
) -> Tuple[float, float, EdgeBetaPrior]:
    """Return posterior mean, variance and updated prior.

    The variance is used as a *confidence* modifier: lower variance ⇒ higher trust."""
    new_alpha = prior.alpha + successes
    new_beta = prior.beta + failures
    total = new_alpha + new_beta
    posterior_mean = new_alpha / total
    # Beta variance formula
    posterior_var = (new_alpha * new_beta) / (total**2 * (total + 1))
    return posterior_mean, posterior_var, EdgeBetaPrior(new_alpha, new_beta)


# ----------------------------------------------------------------------
# 3. Morphology & Recovery Priority (Parent B)
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


def righting_time_index(
    m: Morphology,
    b: float = 1.0 / 3.0,
    k: float = 0.35,
    neck_lever: float = 1.0,
) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority in (0, 1] (never exactly 0)."""
    raw = righting_time_index(m) / max_index
    return max(min(raw, 1.0), 1e-12)


# ----------------------------------------------------------------------
# 4. SSIM endpoint circuit breaker (Parent B)
# ----------------------------------------------------------------------
def _mean_std(arr: np.ndarray) -> Tuple[float, float]:
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    return mu, sigma


def ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 255.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index Measure for 1‑D signals."""
    if len(x) != len(y):
        raise ValueError("signals must have the same length")
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    mu_x, sigma_x = _mean_std(x_arr)
    mu_y, sigma_y = _mean_std(y_arr)
    cov_xy = float(np.cov(x_arr, y_arr, ddof=1)[0, 1])

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)

    # Clamp to avoid exact 0 (log‑domain safety)
    return max(numerator / denominator, 1e-12)


# ----------------------------------------------------------------------
# 5. Tropical max‑plus algebra utilities (Parent A)
# ----------------------------------------------------------------------
def t_add(a: float, b: float) -> float:
    """Tropical addition = max."""
    return max(a, b)


def t_mul(a: float, b: float) -> float:
    """Tropical multiplication = ordinary addition."""
    return a + b


def tropical_floyd_warshall(dist: np.ndarray) -> np.ndarray:
    """All‑pairs tropical max‑plus shortest‑path (max‑plus “distance”)."""
    n = dist.shape[0]
    # Copy to avoid mutating caller
    d = dist.copy()
    for k in range(n):
        # Vectorised inner loops for speed
        via_k = d[:, k, None] + d[k, None, :]
        d = np.maximum(d, via_k)
    return d


# ----------------------------------------------------------------------
# 6. Deeply integrated hybrid model
# ----------------------------------------------------------------------
@dataclass
class HybridEdge:
    """Container for per‑edge raw cost and all confidence sources."""
    raw_cost: float
    delta_energy: float
    temperature: float
    successes: int
    failures: int
    ssim_val: float
    morphology: Morphology
    prior: EdgeBetaPrior = EdgeBetaPrior()


class HybridModel:
    """Encapsulates the full fusion pipeline with log‑domain tropical algebra."""

    def __init__(
        self,
        edges: Dict[Tuple[int, int], HybridEdge],
        num_nodes: int,
        epsilon_hoeffding: float = 0.01,
        hoeffding_delta: float = 0.05,
    ):
        """
        Args:
            edges: Mapping (i, j) → HybridEdge describing the directed graph.
            num_nodes: Number of vertices (0 … n‑1).
            epsilon_hoeffding: Desired Hoeffding error bound.
            hoeffding_delta: Failure probability for Hoeffding test.
        """
        self.edges = edges
        self.n = num_nodes
        self.epsilon = epsilon_hoeffding
        self.hoeffding_delta = hoeffding_delta

    # ------------------------------------------------------------------
    # 6.1 Confidence aggregation in log‑domain
    # ------------------------------------------------------------------
    @staticmethod
    def _log_confidence(
        p_accept: float,
        posterior_mean: float,
        posterior_var: float,
        ssim_val: float,
        recovery_pri: float,
    ) -> float:
        """
        Combine heterogeneous confidences into a single log‑weight.

        The variance term penalises uncertain posteriors:
            weight = mean * exp(-λ * var)
        with λ = 1 (can be tuned). All factors are forced into (0, 1] then
        log‑transformed; the product becomes a sum.
        """
        lam = 1.0
        var_factor = math.exp(-lam * posterior_var)  # in (0,1]
        combined = (
            max(p_accept, 1e-12)
            * max(posterior_mean, 1e-12)
            * var_factor
            * max(ssim_val, 1e-12)
            * max(recovery_pri, 1e-12)
        )
        return math.log(combined)

    # ------------------------------------------------------------------
    # 6.2 Build log‑weight matrix respecting Hoeffding decision
    # ------------------------------------------------------------------
    def _build_log_weight_matrix(self) -> np.ndarray:
        """Return an (n, n) matrix where entry (i, j) = log‑weight or -inf."""
        log_w = np.full((self.n, self.n), -np.inf, dtype=float)

        for (i, j), edge in self.edges.items():
            # Hoeffding test on the number of observations (successes+failures)
            if not hoeffding_decision(
                edge.successes + edge.failures, self.epsilon, self.hoeffding_delta
            ):
                # Not statistically reliable → discard edge
                continue

            p_acc = acceptance_probability(edge.delta_energy, edge.temperature)
            post_mean, post_var, new_prior = bayesian_edge_update(
                edge.prior, edge.successes, edge.failures
            )
            rec_pri = recovery_priority(edge.morphology)

            log_conf = self._log_confidence(
                p_acc, post_mean, post_var, edge.ssim_val, rec_pri
            )
            # In tropical algebra, multiplication = addition, so we add log_conf to cost
            log_w[i, j] = -log_conf  # negative because lower cost = higher confidence

        return log_w

    # ------------------------------------------------------------------
    # 6.3 Weighted tropical max‑plus path computation
    # ------------------------------------------------------------------
    def max_plus_path_cost(self, start: int, end: int) -> Optional[float]:
        """
        Compute the tropical max‑plus path cost from *start* to *end* using
        the fused confidences.

        Returns:
            The maximal (i.e. most reliable) tropical cost, or None if unreachable.
        """
        # Base cost matrix (raw costs) – unreachable edges are +inf
        base = np.full((self.n, self.n), np.inf, dtype=float)
        for (i, j), edge in self.edges.items():
            base[i, j] = edge.raw_cost

        # Convert to tropical “distance” (max‑plus) by negating costs:
        # In max‑plus, larger numbers are “better”; we treat lower raw cost as better,
        # so we use -raw_cost.
        tropical_dist = np.where(np.isfinite(base), -base, -np.inf)

        # Fuse log‑weights (negative log‑confidences) as additive penalties
        log_w = self._build_log_weight_matrix()
        # Combine: tropical distance + log weight penalty
        combined = tropical_dist + log_w  # element‑wise (still max‑plus semantics)

        # All‑pairs tropical closure
        closure = tropical_floyd_warshall(combined)

        result = closure[start, end]
        if result == -np.inf:
            return None
        # Convert back: higher tropical value = lower effective cost
        # Return the effective cost = -result
        return -result

    # ------------------------------------------------------------------
    # 6.4 Utility to expose updated priors after a run
    # ------------------------------------------------------------------
    def updated_priors(self) -> Dict[Tuple[int, int], EdgeBetaPrior]:
        """Return the posterior Beta priors for all edges that survived Hoeffding."""
        out: Dict[Tuple[int, int], EdgeBetaPrior] = {}
        for (i, j), edge in self.edges.items():
            if hoeffding_decision(
                edge.successes + edge.failures, self.epsilon, self.hoeffding_delta
            ):
                _, _, post = bayesian_edge_update(
                    edge.prior, edge.successes, edge.failures
                )
                out[(i, j)] = post
        return out


# ----------------------------------------------------------------------
# 7. Example usage (can be removed in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple 3‑node example
    morph = Morphology(length=2.0, width=1.5, height=0.5, mass=3.0)

    edges = {
        (0, 1): HybridEdge(
            raw_cost=4.0,
            delta_energy=0.8,
            temperature=1.2,
            successes=12,
            failures=3,
            ssim_val=ssim([0, 1, 2], [0, 1, 2]),
            morphology=morph,
        ),
        (1, 2): HybridEdge(
            raw_cost=2.5,
            delta_energy=0.3,
            temperature=1.2,
            successes=8,
            failures=2,
            ssim_val=ssim([10, 20, 30], [10, 20, 31]),
            morphology=morph,
        ),
        (0, 2): HybridEdge(
            raw_cost=7.0,
            delta_energy=1.5,
            temperature=1.2,
            successes=5,
            failures=5,
            ssim_val=ssim([5, 5, 5], [5, 5, 5]),
            morphology=morph,
        ),
    }

    model = HybridModel(edges, num_nodes=3, epsilon_hoeffding=0.05)
    cost = model.max_plus_path_cost(start=0, end=2)
    print("Hybrid max‑plus path cost (0→2):", cost)
    print("Updated priors:", model.updated_priors())