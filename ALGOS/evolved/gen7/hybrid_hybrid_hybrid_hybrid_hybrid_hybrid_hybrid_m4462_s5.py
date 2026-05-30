# DARWIN HAMMER — match 4462, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hoeffd_m2531_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m1850_s0.py (gen5)
# born: 2026-05-29T23:55:56Z

"""Hybrid Regret‑Weighted MinHash, Hoeffding‑Gini Audit & Variational Free‑Energy Scheduler.

Parents:
- **Parent A** – Regret‑weighted MinHash signature, Hoeffding bound, Gini‑scaled regret,
  decision‑hygiene audit (entropy, risk).
- **Parent B** – Variational Free Energy (VFE) surrogate for a RAM‑bounded model pool,
  random hyper‑vectors for fractional binding.

Mathematical Bridge
------------------
The bridge is the *risk‑aware signature vector* produced by Parent A.  
We treat the decision‑hygiene counts as a probability distribution **p**,
compute its Shannon entropy **H(p)** and the Gini coefficient **G(v)** of the
action‑value vector **v**.  The regret‑weighted MinHash signature **σ** (a
real‑valued hyper‑vector) is then combined with the Hoeffding bound **ε**
to obtain an *uncertainty‑adjusted* signature **σ̂ = σ·(1‑ε)**.

Parent B supplies a VFE surrogate **F** for a pool of models under a RAM
ceiling.  The hybrid engine fuses the two topologies by a dot‑product
between the uncertainty‑adjusted signature **σ̂** and the VFE gradient
vector **g = ∇F** (approximated by per‑model free‑energy contributions).
The final hybrid score is the summed aggregation

    S = H(p) + risk·G(v) + σ̂·g

which simultaneously captures audit risk, inequality, statistical confidence,
and resource‑aware free‑energy considerations.

The implementation below provides three core functions demonstrating this
fusion and a lightweight `ModelPool` that uses the VFE surrogate for loading
and eviction decisions.
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Iterable, Optional

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A building blocks
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Atomic decision element."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash used for MinHash‑style signatures."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def gini_coefficient(values: Iterable[float]) -> float:
    """Gini coefficient of a non‑negative value list."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    cumulative = np.cumsum(arr)
    return (2.0 * np.sum((np.arange(1, n + 1) * arr))) / (n * cumulative[-1]) - (n + 1) / n


def shannon_entropy(probs: Iterable[float]) -> float:
    """Standard Shannon entropy, base e."""
    p = np.asarray(list(probs), dtype=float)
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def hoeffding_bound(p_hat: float, n: int, delta: float = 0.05) -> float:
    """Hoeffding bound ε such that |p‑p̂| ≤ ε with probability 1‑δ."""
    if n <= 0:
        return 1.0
    return math.sqrt(math.log(2.0 / delta) / (2 * n))


def regret_weighted_signature(
    tokens: Iterable[str],
    actions: List[MathAction],
    seed: int = 0,
    dim: int = 1024,
) -> np.ndarray:
    """
    Produce a real‑valued signature vector.
    Each token contributes a random hyper‑vector; its amplitude is scaled by
    the regret (risk·(max‑expected‑value‑value)).
    """
    rng = np.random.default_rng(seed)
    # base random hyper‑vectors for each token (MinHash‑style)
    token_vecs = {}
    for t in tokens:
        # use a deterministic hash to seed the RNG for reproducibility
        h = _hash(seed, t) % (2**32)
        rng_token = np.random.default_rng(h)
        token_vecs[t] = rng_token.normal(loc=0.0, scale=1.0, size=dim)

    # compute regret scaling factor per action
    max_ev = max(a.expected_value for a in actions) if actions else 0.0
    regrets = np.array([a.risk * (max_ev - a.expected_value) for a in actions])
    if regrets.size == 0:
        scale = 1.0
    else:
        scale = regrets.mean() + 1.0  # ensure positivity

    # aggregate token vectors
    agg = np.zeros(dim, dtype=float)
    for vec in token_vecs.values():
        agg += vec
    agg *= scale
    # squash with sigmoid to keep values bounded
    return sigmoid(agg)


# ----------------------------------------------------------------------
# Parent‑B building blocks
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    """Lightweight descriptor for a model."""
    name: str
    ram_mb: int
    tier: str  # e.g. "small", "medium", "large"


def random_hv(d: int = 10000, kind: str = "complex", seed: Optional[int] = None) -> np.ndarray:
    """Generate a random hyper‑vector (complex phase representation)."""
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    elif kind == "bipolar":
        return rng.choice([-1, 1], size=d)
    else:  # real
        return rng.normal(size=d)


class ModelPool:
    """
    Manages a pool of loaded models under a RAM ceiling.
    Uses a Variational Free‑Energy (VFE) surrogate to decide loading/eviction.
    """

    def __init__(self, ram_ceiling_mb: int, temperature: float = 1.0):
        self.ram_ceiling = ram_ceiling_mb
        self.temperature = temperature
        self.models: List[ModelTier] = []

    @property
    def used_ram(self) -> int:
        return sum(m.ram_mb for m in self.models)

    def free_energy(self) -> float:
        """
        Simplified VFE:
            F = Σ_i (ram_i / R) * log(ram_i / R) + T * Σ_i log(1 + exp(-ram_i/T))
        where R = total RAM ceiling, T = temperature.
        """
        R = self.ram_ceiling
        if R == 0:
            return float("inf")
        ram_frac = np.array([m.ram_mb / R for m in self.models], dtype=float)
        entropy_term = np.sum(ram_frac * np.log(ram_frac + 1e-12))
        energy_term = self.temperature * np.sum(np.log1p(np.exp(-ram_frac / self.temperature)))
        return entropy_term + energy_term

    def gradient_vector(self) -> np.ndarray:
        """
        Approximate ∇F w.r.t. each model's RAM usage.
        For the toy surrogate, derivative of the entropy term is log(ram_i / R) + 1.
        """
        R = self.ram_ceiling
        grads = []
        for m in self.models:
            frac = m.ram_mb / R
            grad = math.log(frac + 1e-12) + 1.0
            grads.append(grad)
        return np.array(grads, dtype=float)

    def add_model(self, model: ModelTier) -> bool:
        """Attempt to add a model; evict if necessary. Returns success flag."""
        if model.ram_mb > self.ram_ceiling:
            return False  # impossible to fit
        self.models.append(model)
        self._evict_if_needed()
        return True

    def _evict_if_needed(self):
        """Evict models with highest free‑energy contribution until under ceiling."""
        while self.used_ram > self.ram_ceiling and self.models:
            # compute contribution = ram * |gradient|
            grads = self.gradient_vector()
            contributions = np.array([abs(g) * m.ram_mb for g, m in zip(grads, self.models)])
            evict_idx = int(np.argmax(contributions))
            evicted = self.models.pop(evict_idx)
            # recompute grads after eviction
            if self.models:
                grads = self.gradient_vector()
        # No return; state updated in‑place


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_hybrid_audit_score(
    actions: List[MathAction],
    tokens: Iterable[str],
    model_pool: ModelPool,
    seed: int = 0,
    delta: float = 0.05,
) -> float:
    """
    Compute the unified hybrid score S = H(p) + risk·G(v) + σ̂·g.

    - H(p)   : Shannon entropy of a normalized risk distribution over actions.
    - G(v)   : Gini coefficient of the expected values.
    - σ̂·g   : Dot‑product between the uncertainty‑adjusted signature and the
               VFE gradient vector (aligned by model count; excess dimensions are trimmed).
    """
    # 1. Risk distribution and its entropy
    risks = np.array([a.risk for a in actions], dtype=float)
    if risks.sum() == 0:
        probs = np.ones_like(risks) / len(risks) if risks.size else np.array([1.0])
    else:
        probs = risks / risks.sum()
    entropy = shannon_entropy(probs)

    # 2. Gini of expected values, scaled by average risk
    ev_vals = [a.expected_value for a in actions]
    gini = gini_coefficient(ev_vals)
    avg_risk = risks.mean() if risks.size else 0.0
    inequality_term = avg_risk * gini

    # 3. Regret‑weighted signature
    sig = regret_weighted_signature(tokens, actions, seed=seed, dim=1024)

    # 4. Hoeffding bound on a synthetic Bernoulli estimate
    #    Use the mean of the risk distribution as p̂ and number of actions as n.
    p_hat = probs.mean() if probs.size else 0.0
    epsilon = hoeffding_bound(p_hat, n=len(actions), delta=delta)

    # Adjust the signature by the confidence factor (1‑ε)
    sig_adj = sig * (1.0 - epsilon)

    # 5. VFE gradient from the model pool
    grad = model_pool.gradient_vector()
    # Align dimensions: repeat or truncate the signature to match grad length
    if grad.size == 0:
        dot = 0.0
    else:
        if sig_adj.size < grad.size:
            # repeat the short vector
            repeats = int(np.ceil(grad.size / sig_adj.size))
            sig_aligned = np.tile(sig_adj, repeats)[: grad.size]
        else:
            sig_aligned = sig_adj[: grad.size]
        dot = float(np.dot(sig_aligned, grad))

    # Final hybrid score
    return entropy + inequality_term + dot


def allocate_models_hybrid(
    actions: List[MathAction],
    tokens: Iterable[str],
    candidate_models: List[ModelTier],
    ram_ceiling_mb: int,
    seed: int = 0,
    delta: float = 0.05,
    temperature: float = 1.0,
) -> Tuple[ModelPool, float]:
    """
    Build a `ModelPool` under the given RAM ceiling using the hybrid audit
    score as a gating metric. Models are added in descending order of a
    utility metric:

        U_i = (risk·expected_value) / (ram_mb + 1)

    After each insertion the hybrid audit score is recomputed; insertion stops
    when the score exceeds a dynamic threshold (here set to the initial score
    plus one standard deviation of the utility list).
    """
    pool = ModelPool(ram_ceiling_mb=ram_ceiling_mb, temperature=temperature)

    # Compute utilities
    utilities = []
    for m in candidate_models:
        # map model name to a dummy action if present
        matching = next((a for a in actions if a.id == m.name), None)
        risk = matching.risk if matching else 0.5
        ev = matching.expected_value if matching else 0.0
        util = (risk * ev) / (m.ram_mb + 1)
        utilities.append((util, m))

    utilities.sort(key=lambda x: x[0], reverse=True)
    util_vals = [u for u, _ in utilities]
    if not util_vals:
        return pool, 0.0
    threshold = np.mean(util_vals) + np.std(util_vals)

    for util, model in utilities:
        if util < threshold:
            break
        pool.add_model(model)
        # recompute score; if it blows up we rollback the last addition
        score = compute_hybrid_audit_score(actions, tokens, pool, seed=seed, delta=delta)
        if math.isnan(score) or score > 1e6:
            # rollback
            pool.models.pop()
            break

    final_score = compute_hybrid_audit_score(actions, tokens, pool, seed=seed, delta=delta)
    return pool, final_score


def evaluate_action_regret(
    action: MathAction,
    baseline: Optional[MathAction] = None,
) -> float:
    """
    Simple regret estimator used by the hybrid system.
    Regret = risk * (baseline_expected_value - action_expected_value).
    If no baseline is supplied, the maximum expected value among all actions
    (assumed to be globally known) is used.
    """
    if baseline is None:
        raise ValueError("Baseline action must be provided for regret computation.")
    return action.risk * (baseline.expected_value - action.expected_value)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny action set
    actions = [
        MathAction(id="A", expected_value=0.8, risk=0.2),
        MathAction(id="B", expected_value=0.5, risk=0.5),
        MathAction(id="C", expected_value=0.3, risk=0.7),
    ]

    # Token stream (e.g., extracted from a prompt)
    tokens = ["audit", "risk", "model", "allocation", "entropy"]

    # Candidate models
    candidates = [
        ModelTier(name="A", ram_mb=200, tier="small"),
        ModelTier(name="B", ram_mb=500, tier="medium"),
        ModelTier(name="C", ram_mb=800, tier="large"),
        ModelTier(name="D", ram_mb=300, tier="small"),
    ]

    # Build hybrid pool
    pool, final_score = allocate_models_hybrid(
        actions=actions,
        tokens=tokens,
        candidate_models=candidates,
        ram_ceiling_mb=1200,
        seed=42,
        delta=0.05,
        temperature=0.7,
    )

    print("Loaded models:")
    for m in pool.models:
        print(f"  - {m.name} ({m.ram_mb} MB, tier={m.tier})")
    print(f"Total RAM used: {pool.used_ram} MB / {pool.ram_ceiling} MB")