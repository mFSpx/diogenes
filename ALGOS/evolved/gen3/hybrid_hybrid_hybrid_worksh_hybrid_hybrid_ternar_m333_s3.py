# DARWIN HAMMER — match 333, survivor 3
# gen: 3
# parent_a: hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s1.py (gen2)
# parent_b: hybrid_hybrid_ternary_route_variational_free_ene_m21_s0.py (gen2)
# born: 2026-05-29T23:28:34Z

import sys
import math
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core constants
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
BASE_TAU: float = 1.0          # baseline liquid time constant
ALPHA: float = 5.0             # gating steepness
LAMBDA: float = 0.7            # VFE weighting factor
MINHASH_K: int = 64            # number of hash functions for MinHash
MAX64: int = (1 << 64) - 1     # mask for 64‑bit hashing
SEED_BASE: int = 123456789     # deterministic base seed for all RNGs

# ----------------------------------------------------------------------
# Deterministic RNG
# ----------------------------------------------------------------------
_rng = np.random.default_rng(SEED_BASE)

# ----------------------------------------------------------------------
# Utility: weekday‑dependent weight vector (Parent A)
# ----------------------------------------------------------------------
def weekday_weight_vector(groups: Tuple[str, ...], dow: int) -> np.ndarray:
    """
    Normalised weight vector w(d) for the given weekday index ``dow`` (0=Sun … 6=Sat).

    A sinusoidal pattern with a small amplitude ensures the vector never collapses
    to a one‑hot configuration, preserving gradient flow.
    """
    n = len(groups)
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


def today_weekday() -> int:
    """Return today's weekday index compatible with ``weekday_weight_vector``."""
    return (date.today().weekday() + 1) % 7  # 0 = Sunday


# ----------------------------------------------------------------------
# MinHash utilities (Parent A)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """
    Simple 64‑bit hash based on Python's built‑in hash combined with a seed.
    The function is deterministic across interpreter runs because the seed
    is mixed into the tuple before hashing.
    """
    mixed = hash((seed, token)) & MAX64
    return mixed


def minhash_signature(token_set: Set[str], k: int = MINHASH_K) -> List[int]:
    """
    Compute a MinHash signature of length ``k`` for ``token_set``.
    Each entry is the minimum hash value observed for a distinct random seed.
    """
    signature: List[int] = []
    for i in range(k):
        min_val = min(_hash(i, token) for token in token_set)
        signature.append(min_val)
    return signature


def minhash_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """
    Estimate Jaccard similarity from two MinHash signatures.
    """
    if len(sig_a) != len(sig_b):
        raise ValueError("Signatures must have equal length")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def groupwise_similarity(
    token_set: Set[str],
    reference_map: Dict[str, Set[str]],
) -> np.ndarray:
    """
    Compute a similarity vector ``s⃗`` where each entry corresponds to a group.
    ``reference_map`` maps group names to their own reference token sets.
    """
    sims = []
    for grp in GROUPS:
        ref = reference_map.get(grp, set())
        if not ref or not token_set:
            sims.append(0.0)
            continue
        sig_a = minhash_signature(token_set)
        sig_b = minhash_signature(ref)
        sims.append(minhash_similarity(sig_a, sig_b))
    return np.array(sims, dtype=np.float64)


# ----------------------------------------------------------------------
# Variational Free‑Energy utilities (Parent B)
# ----------------------------------------------------------------------
def gaussian_kl(mean_q: np.ndarray, var_q: np.ndarray,
                mean_p: np.ndarray, var_p: np.ndarray) -> np.ndarray:
    """
    KL divergence KL(q‖p) for diagonal‑covariance Gaussians.
    Returns a vector of per‑dimension KL terms.
    """
    return 0.5 * (
        np.log(var_p / var_q) +
        (var_q + (mean_q - mean_p) ** 2) / var_p -
        1.0
    )


def variational_free_energy(
    state_means: Dict[str, np.ndarray],
    state_vars: Dict[str, np.ndarray],
    prior_means: Dict[str, np.ndarray],
    prior_vars: Dict[str, np.ndarray],
    weight_vec: np.ndarray,
    sim_vec: np.ndarray,
) -> float:
    """
    Compute a deeper VFE that respects the group structure:

        F = Σ_g ( -‖μ_g‖² )  –  λ * Σ_g ( w_g * s_g * KL_g )

    The reconstruction term is still a proxy (negative L2 norm) but summed per group.
    """
    recon = 0.0
    weighted_kl = 0.0
    for i, grp in enumerate(GROUPS):
        μ = state_means[grp]
        recon -= np.sum(μ ** 2)                     # proxy log‑likelihood
        kl_vec = gaussian_kl(μ, state_vars[grp],
                             prior_means[grp], prior_vars[grp])
        # collapse per‑dimension KL into a scalar (sum) before weighting
        kl_scalar = np.sum(kl_vec)
        weighted_kl += weight_vec[i] * sim_vec[i] * kl_scalar
    return recon - LAMBDA * weighted_kl


# ----------------------------------------------------------------------
# Deep gating – per‑group LTC modulation (core of Parent A)
# ----------------------------------------------------------------------
def gating_factors(weight_vec: np.ndarray, sim_vec: np.ndarray) -> np.ndarray:
    """
    Compute a per‑group gating vector g⃗ where

        g_i = σ( α * (w_i * s_i - 0.5) )

    The sigmoid is numerically stable via ``np.exp`` with clipping.
    """
    prod = weight_vec * sim_vec                     # element‑wise product ∈ [0,1]
    # shift to centre the sigmoid around 0.5 of the product
    z = ALPHA * (prod - 0.5)
    # stable sigmoid
    g = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
    return g


def effective_time_constants(base_tau: float, gating_vec: np.ndarray) -> np.ndarray:
    """
    τ⃗ = τ₀ / g⃗  (larger gating → smaller effective τ).
    A tiny epsilon prevents division‑by‑zero.
    """
    eps = 1e-9
    return base_tau / (gating_vec + eps)


# ----------------------------------------------------------------------
# Dummy ternary router (stand‑in for the external service in Parent B)
# ----------------------------------------------------------------------
def dummy_route_command(text: str, intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic router that returns a Dirichlet‑sampled distribution over three actions.
    The RNG seed is derived from a stable hash of ``(text, intent)``.
    """
    seed = hash((text, intent)) & MAX64
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(3)).astype(float)
    return {
        "action_distribution": {"a": float(probs[0]),
                                "b": float(probs[1]),
                                "c": float(probs[2])},
        "selected_action": ["a", "b", "c"][int(np.argmax(probs))],
        "intent": intent,
        "text": text,
        "context_summary": {k: str(v)[:30] for k, v in context.items()}
    }


# ----------------------------------------------------------------------
# Hybrid processing pipeline (the fused algorithm)
# ----------------------------------------------------------------------
def _init_group_states(dim: int) -> Tuple[
    Dict[str, np.ndarray], Dict[str, np.ndarray],
    Dict[str, np.ndarray], Dict[str, np.ndarray]
]:
    """
    Initialise per‑group Gaussian states (mean, var) and priors.
    All means start at zero; variances are unit.
    """
    means = {}
    vars_ = {}
    prior_means = {}
    prior_vars = {}
    for grp in GROUPS:
        means[grp] = np.zeros(dim, dtype=np.float64)
        vars_[grp] = np.ones(dim, dtype=np.float64)
        prior_means[grp] = np.zeros(dim, dtype=np.float64)
        prior_vars[grp] = np.ones(dim, dtype=np.float64)
    return means, vars_, prior_means, prior_vars


def hybrid_process(
    packet: Dict[str, Any],
    reference_map: Dict[str, Set[str]] | None = None,
    latent_dim: int = 8,
) -> Dict[str, Any]:
    """
    End‑to‑end hybrid routine with a deeper mathematical coupling:

    1. Extract ``text`` and ``intent``.
    2. Tokenise ``text`` (simple whitespace split) → token set.
    3. Build weekday weight vector ``w(d)``.
    4. Compute a *groupwise* MinHash similarity vector ``s⃗``.
    5. Derive per‑group gating ``g⃗`` and effective time constants ``τ⃗``.
    6. Run the dummy ternary router.
    7. Interpret the router distribution as a soft‑argmax Gaussian mean for each group.
    8. Evaluate a group‑aware variational free‑energy.
    9. Return a rich result dictionary.
    """
    # ------------------------------------------------------------------
    # 1‑2. Basic extraction and tokenisation
    # ------------------------------------------------------------------
    text = str(packet.get("text", ""))
    intent = str(packet.get("intent", "unknown"))
    context = packet.get("context", {})

    tokens = set(text.lower().split())

    # ------------------------------------------------------------------
    # 3. Weekday weight vector
    # ------------------------------------------------------------------
    dow = today_weekday()
    w_vec = weekday_weight_vector(GROUPS, dow)          # shape (G,)

    # ------------------------------------------------------------------
    # 4. Groupwise MinHash similarity
    # ------------------------------------------------------------------
    if reference_map is None:
        # fallback: empty reference for every group → zero similarity
        reference_map = {g: set() for g in GROUPS}
    s_vec = groupwise_similarity(tokens, reference_map)  # shape (G,)

    # ------------------------------------------------------------------
    # 5. Gating and effective time constants
    # ------------------------------------------------------------------
    g_vec = gating_factors(w_vec, s_vec)                # shape (G,)
    tau_vec = effective_time_constants(BASE_TAU, g_vec)  # shape (G,)

    # ------------------------------------------------------------------
    # 6. Dummy router
    # ------------------------------------------------------------------
    router_out = dummy_route_command(text, intent, context)

    # ------------------------------------------------------------------
    # 7. Soft‑argmax Gaussian means per group
    # ------------------------------------------------------------------
    # Convert the three‑action distribution into a scalar in [0,1]
    dist_vals = np.array(list(router_out["action_distribution"].values()))
    soft_argmax = np.dot(dist_vals, np.arange(1, 4)) / 3.0   # weighted average ∈ [0,1]

    # Initialise per‑group latent states
    means, vars_, prior_means, prior_vars = _init_group_states(latent_dim)

    # Inject the soft‑argmax value into each group's mean (simple coupling)
    for grp in GROUPS:
        means[grp] = soft_argmax * np.ones(latent_dim, dtype=np.float64)

    # ------------------------------------------------------------------
    # 8. Variational free‑energy with deeper coupling
    # ------------------------------------------------------------------
    free_energy = variational_free_energy(
        state_means=means,
        state_vars=vars_,
        prior_means=prior_means,
        prior_vars=prior_vars,
        weight_vec=w_vec,
        sim_vec=s_vec,
    )

    # ------------------------------------------------------------------
    # 9. Assemble result
    # ------------------------------------------------------------------
    result: Dict[str, Any] = {
        "weekday": dow,
        "weight_vector": w_vec.tolist(),
        "similarity_vector": s_vec.tolist(),
        "gating_vector": g_vec.tolist(),
        "effective_tau_vector": tau_vec.tolist(),
        "router_output": router_out,
        "soft_argmax_scalar": float(soft_argmax),
        "variational_free_energy": float(free_energy),
        "latent_means": {g: means[g].tolist() for g in GROUPS},
        "latent_vars": {g: vars_[g].tolist() for g in GROUPS},
    }
    return result


# ----------------------------------------------------------------------
# Example entry‑point (can be removed in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    example_packet = {
        "text": "Schedule a meeting with the analytics team tomorrow.",
        "intent": "schedule_meeting",
        "context": {"user_id": 42, "timezone": "UTC"},
    }

    # Minimal reference token sets per group for demonstration
    reference_tokens = {
        "codex": {"code", "function", "python", "algorithm"},
        "groq": {"query", "search", "index", "document"},
        "cohere": {"language", "model", "generation", "text"},
        "local_models": {"offline", "edge", "device", "cache"},
    }

    out = hybrid_process(example_packet, reference_map=reference_tokens)
    print(json.dumps(out, indent=2))