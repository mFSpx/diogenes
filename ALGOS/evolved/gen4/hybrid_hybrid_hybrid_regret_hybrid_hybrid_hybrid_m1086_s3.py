# DARWIN HAMMER — match 1086, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_regret_engine_hybrid_bandit_router_m38_s5.py (gen3)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hoeffding_tre_m44_s3.py (gen3)
# born: 2026-05-29T23:32:53Z

"""HybridRegretBanditEndpointTropical
Integrates:
- Parent A: Regret‑Weighted Bandit with MinHash similarity and a “dance” signal.
- Parent B: Linear State‑Space health scoring + tropical (max‑plus) network + Hoeffding‑bound tree split.

Mathematical bridge
-------------------
1. **Regret‑Bandit score** for each action *i*  
   \(R_i = \mathbb{E}[v_i] - c_i - r_i + \tilde v_i\)  

   \(g(R_i)=\sigma(R_i)=\frac{1}{1+e^{-R_i}}\)  

   \(S_i = g(R_i)\,(1+\operatorname{sim}(\sigma_i,\sigma_{\text{ref}}))\;\cdot\;d\)  

   where `sim` is the MinHash Jaccard estimate and `d` is the
   honey‑bee store “dance” control signal (∈[0,1]).

2. **State‑Space health score** – the vector of bandit scores
   \(\mathbf{s}_t = [S_{i_1},\dots,S_{i_n}]\) is taken as the observation
   at discrete time *t*.  An SSM with matrices built from the current
   endpoint pool maps this observation to a scalar health score  

   \(y_t = \mathbf{w}^{\top}\mathbf{s}_t + b\)

   where \(\mathbf{w}\) contains the `righting_time_index` of each endpoint
   (higher ⇒ healthier) and \(b\) encodes normalized failure‑rate.

3. **Tropical network & Hoeffding split** – the health scores feed a
   two‑layer tropical (max‑plus) network  

   \(z^{(1)}_k = \max_j (W^{(1)}_{kj}+y_t) + b^{(1)}_k\)  

   \(g_t = \max_k (W^{(2)}_{k}+z^{(1)}_k) + b^{(2)}\)

   The scalar \(g_t\) is a “gain” candidate.  A Hoeffding bound applied to
   the running mean of gains decides, with confidence \(1-\delta\), whether a
   decision‑tree node should split.

The three core functions below realise this pipeline:
`compute_regret_bandit_scores`, `compute_health_scores`,
and `tropical_hoeffding_update`.  They are deliberately lightweight yet
preserve the mathematical essence of both parents.  The module can be
used as a stand‑alone hybrid decision engine."""

import hashlib
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – regret‑weighted bandit utilities
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MathAction:
    """Action description used by the regret‑bandit component."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome for an action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash used by MinHash."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Return a MinHash signature of length *k* for the given token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        # all‑zero signature – similarity will be zero
        return [0] * k
    sig = []
    for i in range(k):
        # smallest hash value for seed i
        min_h = min(_hash(i, t) for t in toks)
        sig.append(min_h)
    return sig


def jaccard_minhash(sig_a: List[int], sig_b: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("Signatures must be of equal length")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def sigmoid(x: float) -> float:
    """Standard logistic sigmoid."""
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class StoreState:
    """Honey‑bee store state – only the `dance` signal is needed."""
    dance: float = 1.0  # bounded in [0,1]; 1 means full influence


def compute_regret_bandit_scores(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    store: StoreState,
    reference_action_ids: List[str],
    minhash_k: int = 128,
) -> Dict[str, float]:
    """
    Compute the hybrid regret‑bandit score S_i for every action.

    Returns
    -------
    dict
        Mapping ``action.id -> S_i``.
    """
    # 1. Build a lookup for counterfactuals
    cf_lookup: Dict[str, MathCounterfactual] = {
        cf.action_id: cf for cf in counterfactuals
    }

    # 2. Compute MinHash signatures for all actions (tokenised by id characters)
    signatures: Dict[str, List[int]] = {}
    for act in actions:
        tokens = list(act.id)  # simplistic tokenisation
        signatures[act.id] = minhash_signature(tokens, k=minhash_k)

    # 3. Reference signature – median of the chosen reference actions
    ref_sigs = [signatures[aid] for aid in reference_action_ids if aid in signatures]
    if not ref_sigs:
        # fall back to a zero signature
        ref_sig = [0] * minhash_k
    else:
        # element‑wise median (as int)
        ref_sig = [int(np.median([s[i] for s in ref_sigs])) for i in range(minhash_k)]

    # 4. Compute S_i
    scores: Dict[str, float] = {}
    for act in actions:
        # raw regret term
        cf = cf_lookup.get(act.id)
        counterfactual = cf.outcome_value if cf else 0.0
        R = act.expected_value - act.cost - act.risk + counterfactual

        # sigmoid weighting
        g = sigmoid(R)

        # similarity term
        sim = jaccard_minhash(signatures[act.id], ref_sig)

        # final hybrid score
        S = g * (1.0 + sim) * store.dance
        scores[act.id] = S
    return scores


# ----------------------------------------------------------------------
# Parent B – endpoint SSM, tropical network, Hoeffding split
# ----------------------------------------------------------------------


@dataclass
class Endpoint:
    """Endpoint description used by the health‑score SSM."""
    failures: int
    failure_threshold: int
    righting_time_index: float  # higher ⇒ healthier

    @property
    def failure_rate(self) -> float:
        """Normalized failure rate in [0, 1]."""
        if self.failure_threshold == 0:
            return 0.0
        return min(1.0, self.failures / self.failure_threshold)


def build_ssm_matrices(endpoints: List[Endpoint]) -> Tuple[np.ndarray, float]:
    """
    Construct a simple linear SSM from the endpoint pool.

    Returns
    -------
    A : ndarray, shape (n_endpoints,)
        Weight vector containing the `righting_time_index` of each endpoint.
    b : float
        Bias term equal to the negative mean failure rate (more failures → lower health).
    """
    if not endpoints:
        raise ValueError("At least one endpoint required")
    A = np.array([ep.righting_time_index for ep in endpoints], dtype=float)
    mean_failure = np.mean([ep.failure_rate for ep in endpoints])
    b = -mean_failure
    return A, b


def compute_health_scores(
    bandit_scores: Dict[str, float],
    endpoints: List[Endpoint],
) -> np.ndarray:
    """
    Map the vector of bandit scores to a scalar health score per time step.

    For demonstration we treat each *action* as a separate time step and
    compute a health score by projecting the score onto the endpoint weight
    vector A (dot product) plus bias b.

    Returns
    -------
    ndarray, shape (n_actions,)
        Health scores y_t for each action (ordered by insertion order of
        ``bandit_scores``).
    """
    A, b = build_ssm_matrices(endpoints)

    # Align the length of A with the number of actions by repetition if needed
    scores_vec = np.array(list(bandit_scores.values()), dtype=float)
    if len(A) < len(scores_vec):
        repeats = int(np.ceil(len(scores_vec) / len(A)))
        A_ext = np.tile(A, repeats)[: len(scores_vec)]
    else:
        A_ext = A[: len(scores_vec)]

    y = A_ext * scores_vec + b
    return y


@dataclass
class TreeNode:
    """Very small Hoeffding‑tree node tracking gain statistics."""
    count: int = 0
    sum_gain: float = 0.0
    sum_sq_gain: float = 0.0
    split: bool = False
    split_threshold: float = 0.0  # will be set by Hoeffding bound


def tropical_network(gains_input: np.ndarray) -> np.ndarray:
    """
    Two‑layer tropical (max‑plus) network.

    Layer 1: weight matrix W1 ∈ ℝ^{h×1}, bias b1 ∈ ℝ^{h}
    Layer 2: weight vector w2 ∈ ℝ^{h}, bias b2 ∈ ℝ

    Because the input is scalar per time step, the max‑plus operations reduce
    to simple affine transforms with a max over the hidden units.
    """
    h = 8  # hidden dimensionality
    rng = np.random.default_rng(42)

    # Random weights/biases – fixed seed for reproducibility
    W1 = rng.uniform(-1.0, 1.0, size=(h, 1))
    b1 = rng.uniform(-0.5, 0.5, size=h)

    W2 = rng.uniform(-1.0, 1.0, size=h)
    b2 = rng.uniform(-0.2, 0.2)

    # First layer: max_j (W1_j * x + b1_j)
    # Since x is a vector, we broadcast
    z1 = np.max(W1.squeeze() * gains_input[:, None] + b1, axis=1)

    # Second layer: max_k (W2_k + z1_k) + b2
    # Here we treat each time step independently, so we compute per step
    # using the same hidden representation (z1 is already per step)
    # To keep dimensions consistent we expand W2 and compute max over hidden units.
    gains = np.max(W2 + z1[:, None], axis=1) + b2
    return gains


def hoeffding_bound(R: float, n: int, delta: float) -> float:
    """
    Hoeffding bound for a random variable bounded in [0, R].

    Returns the maximal deviation ε such that
    P(|mean - true_mean| > ε) ≤ δ.
    """
    if n <= 0:
        return float("inf")
    return R * math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def tropical_hoeffding_update(
    health_scores: np.ndarray,
    node: TreeNode,
    delta: float = 0.05,
    gain_range: Tuple[float, float] = (0.0, 1.0),
) -> None:
    """
    Feed health scores through the tropical network, update the node statistics,
    and decide whether to split according to a Hoeffding bound.

    The function mutates ``node`` in place.
    """
    gains = tropical_network(health_scores)

    for g in gains:
        node.count += 1
        node.sum_gain += g
        node.sum_sq_gain += g * g

    # Empirical mean gain
    mean_gain = node.sum_gain / node.count

    # Upper bound on the range of gain values
    R = gain_range[1] - gain_range[0]

    eps = hoeffding_bound(R, node.count, delta)

    # Simple split criterion: if mean gain exceeds a threshold by > ε
    # For demo we set threshold at 0.6 (arbitrary)
    threshold = 0.6
    if not node.split and mean_gain - eps > threshold:
        node.split = True
        node.split_threshold = threshold


# ----------------------------------------------------------------------
# High‑level hybrid pipeline
# ----------------------------------------------------------------------


def hybrid_step(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    store: StoreState,
    reference_action_ids: List[str],
    endpoints: List[Endpoint],
    node: TreeNode,
) -> Tuple[Dict[str, float], np.ndarray, TreeNode]:
    """
    Execute one complete hybrid iteration:
    1. Regret‑bandit scoring.
    2. Health‑score projection via a linear SSM.
    3. Tropical network + Hoeffding split update.

    Returns
    -------
    bandit_scores : dict
        Action ID → hybrid bandit score S_i.
    health_scores : ndarray
        Scalar health scores y_t (one per action).
    node : TreeNode
        Updated tree node (may have `split == True`).
    """
    # 1. Regret‑bandit scores
    bandit_scores = compute_regret_bandit_scores(
        actions, counterfactuals, store, reference_action_ids
    )

    # 2. Health scores from SSM
    health_scores = compute_health_scores(bandit_scores, endpoints)

    # 3. Tropical + Hoeffding update
    tropical_hoeffding_update(health_scores, node)

    return bandit_scores, health_scores, node


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Define a tiny action set
    actions = [
        MathAction(id="A1", expected_value=1.2, cost=0.1, risk=0.05),
        MathAction(id="A2", expected_value=0.8, cost=0.2, risk=0.02),
        MathAction(id="A3", expected_value=1.0, cost=0.15, risk=0.1),
    ]

    # Counterfactuals (some may be missing)
    counterfactuals = [
        MathCounterfactual(action_id="A1", outcome_value=0.3),
        MathCounterfactual(action_id="A3", outcome_value=-0.1),
    ]

    store = StoreState(dance=0.9)

    reference_action_ids = ["A1", "A2"]  # used for MinHash reference signature

    # Endpoint pool – arbitrary but consistent with health scoring