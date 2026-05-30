# DARWIN HAMMER — match 4385, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_endpoi_m416_s2.py (gen5)
# parent_b: hybrid_hybrid_liquid_time_c_hybrid_fisher_locali_m2189_s1.py (gen3)
# born: 2026-05-29T23:55:16Z

"""Hybrid Fusion of Tropical‑Network/SSM with Liquid‑Time‑Constant, MinHash and Fisher‑SSIM

Parents:
- **Parent A** (`hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_endpoi_m416_s2.py`):
  Tropical (max‑plus) network feeding a state‑space model (SSM) and a curvature‑driven
  brain‑map. Core operation: `TropicalNetwork.evaluate` → SSM update → SSIM between
  SSM output and tropical output.

- **Parent B** (`hybrid_hybrid_liquid_time_c_hybrid_fisher_locali_m2189_s1.py`):
  Liquid‑Time‑Constant (LTC) recurrent cell whose time constant is modulated by a
  MinHash similarity; a Fisher information score weights this similarity, and an
  SSIM metric is used for routing decisions.

**Mathematical Bridge**
We treat the tropical network output **T(x)** as a target for the LTC‑SSM dynamics.
The effective liquid time constant **τ** is defined as  

\[
\tau = \frac{\tau_0}{1 + \alpha \, \mathcal{F}(\theta) \, \mathrm{J}(S_1,S_2)},
\]

where `τ0` is a base constant, `α` a scaling factor, `\mathcal{F}` the Fisher
information score for a tunable parameter `θ`, and `J` the MinHash‑based Jaccard
similarity between two token sets. The state update blends the current state
`h` with the tropical target using this τ, and the structural similarity index
(SSIM) between the updated state and the tropical output provides a hybrid
metric for routing/selection.

The module implements this fused system and demonstrates three core functions:
`ltc_step_hybrid`, `tropical_ssm_forward`, and `hybrid_metric`. """

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Iterable, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Tropical Network
# ----------------------------------------------------------------------
class TropicalNetwork:
    """Max‑plus (tropical) linear layer."""
    def __init__(self, weights: np.ndarray, biases: np.ndarray):
        self.weights = weights  # shape (out_dim, in_dim)
        self.biases = biases    # shape (out_dim,)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Tropical (max‑plus) affine transformation."""
        out = np.empty(self.weights.shape[0], dtype=float)
        for i in range(self.weights.shape[0]):
            out[i] = max(0.0, np.dot(self.weights[i], x) + self.biases[i])
        return out

# ----------------------------------------------------------------------
# Parent B – MinHash utilities
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1

def _hash(seed: int, token: str) -> int:
    """Hash a token with a 4‑byte seed using Blake2b (64‑bit output)."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(
        np.frombuffer(
            hashlib.blake2b(data, digest_size=8).digest(),
            dtype=np.uint8
        ).view(np.uint64), "big"
    )

def minhash_signature(tokens: Iterable[str], num_perm: int = 128) -> List[int]:
    """Compute MinHash signature for a set of tokens."""
    signature = [MAX64] * num_perm
    for token in tokens:
        for i in range(num_perm):
            h = _hash(i, token)
            if h < signature[i]:
                signature[i] = h
    return signature

def minhash_similarity(sig1: List[int], sig2: List[int]) -> float:
    """Jaccard similarity estimated from two MinHash signatures."""
    if len(sig1) != len(sig2):
        raise ValueError("Signatures must be of equal length")
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)

# ----------------------------------------------------------------------
# Fisher information (scalar placeholder)
# ----------------------------------------------------------------------
def fisher_score(samples: np.ndarray) -> float:
    """
    Approximate Fisher information for a univariate Gaussian model.
    For a Gaussian with unknown mean μ and known variance σ²,
    I(μ) = 1/σ². We estimate σ² from the sample variance.
    """
    if samples.size == 0:
        return 0.0
    var = np.var(samples, ddof=1)
    return 0.0 if var == 0 else 1.0 / var

# ----------------------------------------------------------------------
# SSIM for 1‑D signals (simplified)
# ----------------------------------------------------------------------
def ssim_1d(x: np.ndarray, y: np.ndarray, C1: float = 0.01**2, C2: float = 0.03**2) -> float:
    """Structural Similarity Index for 1‑D arrays."""
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    return numerator / denominator if denominator != 0 else 0.0

# ----------------------------------------------------------------------
# Hybrid Liquid‑Time‑Constant (LTC) cell
# ----------------------------------------------------------------------
@dataclass
class LTCState:
    """State vector of the hybrid LTC‑SSM system."""
    h: np.ndarray  # hidden state
    tau0: float    # base time constant

def ltc_step_hybrid(
    state: LTCState,
    input_vec: np.ndarray,
    tropical_out: np.ndarray,
    fisher_val: float,
    jaccard: float,
    alpha: float = 1.0,
    dt: float = 0.1,
) -> LTCState:
    """
    Perform one hybrid LTC step.
    τ = τ0 / (1 + α * Fisher * Jaccard)
    h_{t+1} = h_t + dt/τ * ( -h_t + tropical_out )
    """
    tau = state.tau0 / (1.0 + alpha * fisher_val * jaccard)
    dh = (-state.h + tropical_out) * (dt / tau)
    new_h = state.h + dh
    return LTCState(h=new_h, tau0=state.tau0)

# ----------------------------------------------------------------------
# Fusion core: tropical → SSM → LTC step → metric
# ----------------------------------------------------------------------
def tropical_ssm_forward(
    tropical_net: TropicalNetwork,
    ltc_state: LTCState,
    input_vec: np.ndarray,
) -> Tuple[LTCState, np.ndarray]:
    """
    Compute tropical network output and feed it to the LTC‑SSM dynamics.
    Returns the updated state and the tropical output.
    """
    t_out = tropical_net.evaluate(input_vec)
    # For demonstration we use a dummy Fisher score based on the tropical output
    fisher_val = fisher_score(t_out)
    # Dummy token sets derived from quantised tropical output
    tokens_a = {f"tok_{int(v)}" for v in t_out}
    tokens_b = {f"tok_{int(v*1.1)}" for v in t_out}
    sig_a = minhash_signature(tokens_a)
    sig_b = minhash_signature(tokens_b)
    jaccard = minhash_similarity(sig_a, sig_b)
    new_state = ltc_step_hybrid(
        state=ltc_state,
        input_vec=input_vec,
        tropical_out=t_out,
        fisher_val=fisher_val,
        jaccard=jaccard,
    )
    return new_state, t_out

def hybrid_metric(state: LTCState, tropical_out: np.ndarray) -> float:
    """
    Compute a hybrid quality metric:
    metric = α * Fisher + β * SSIM(state, tropical_out) + γ * Jaccard
    For simplicity we reuse the Fisher from the current state vector.
    """
    fisher_val = fisher_score(state.h)
    # Re‑compute Jaccard using the same tokenisation scheme as in the forward step
    tokens_state = {f"tok_{int(v)}" for v in state.h}
    tokens_trop = {f"tok_{int(v)}" for v in tropical_out}
    jaccard = minhash_similarity(
        minhash_signature(tokens_state),
        minhash_signature(tokens_trop),
    )
    ssim_val = ssim_1d(state.h, tropical_out)
    α, β, γ = 0.4, 0.4, 0.2
    return α * fisher_val + β * ssim_val + γ * jaccard

def hybrid_forward(
    tropical_net: TropicalNetwork,
    init_state: LTCState,
    input_seq: List[np.ndarray],
) -> List[Tuple[np.ndarray, float]]:
    """
    Run a sequence of inputs through the hybrid system.
    Returns a list of (state_vector, hybrid_metric) tuples.
    """
    results = []
    state = init_state
    for vec in input_seq:
        state, t_out = tropical_ssm_forward(tropical_net, state, vec)
        metric = hybrid_metric(state, t_out)
        results.append((state.h.copy(), metric))
    return results

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Create a tiny tropical network (3→3)
    W = np.random.randn(3, 3)
    b = np.random.randn(3)
    tropical = TropicalNetwork(weights=W, biases=b)

    # Initialise LTC state
    init_h = np.zeros(3)
    ltc_state = LTCState(h=init_h, tau0=1.0)

    # Random input sequence of length 5
    inputs = [np.random.randn(3) for _ in range(5)]

    # Run hybrid forward pass
    outputs = hybrid_forward(tropical, ltc_state, inputs)

    for idx, (h_vec, metric) in enumerate(outputs):
        print(f"Step {idx}: state={h_vec}, hybrid_metric={metric:.4f}")