# DARWIN HAMMER — match 21, survivor 2
# gen: 2
# parent_a: hybrid_ternary_router_ssim_m1_s1.py (gen1)
# parent_b: variational_free_energy.py (gen0)
# born: 2026-05-29T23:22:53Z

"""
Hybrid Ternary‑Router / Variational Free‑Energy (HTR‑VFE)

Parent A: ``hybrid_ternary_router_ssim_m1_s1.py`` – provides a ternary‑router
that maps an input text to an output text and a structural similarity index
(SSIM) that quantifies reconstruction quality between the two character streams.

Parent B: ``variational_free_energy.py`` – defines the variational free‑energy
(F) for Gaussian generative models, together with KL divergence and a gradient
descent belief‑update.

Mathematical bridge
-------------------
Both parents evaluate a *reconstruction* error:

* SSIM(x, y) ∈ [0, 1] measures how well the router’s output y reproduces the
  input x.
* In the free‑energy formulation the reconstruction loss is the negative
  log‑likelihood of the observation under a Gaussian centred at the current
  belief (μ_q).

We map the SSIM score to a pseudo‑observation noise variance

    σ_obs² = ε + (1 – SSIM)·R²

where R is the dynamic range of the character codes (≈255) and ε≈1e‑6 prevents
division by zero.  A high SSIM yields a small σ_obs (high precision), making
the free‑energy term penalise belief deviations strongly; a low SSIM yields a
large σ_obs, relaxing the penalty.

The hybrid algorithm therefore:

1. Routes the packet with the ternary router.
2. Encodes input and output texts as integer vectors.
3. Computes SSIM → similarity.
4. Treats the router output vector as the current belief mean μ_q.
5. Evaluates the variational free energy F(μ_q) using the input vector as the
   observation and σ_obs derived from similarity.
6. Optionally performs a gradient‑descent belief update (perception) that moves
   μ_q toward the observation, mimicking active‑inference dynamics.

The three public functions below showcase this pipeline.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

import math
import numpy as np

# ---------------------------------------------------------------------------
# Ternary‑router interface (parent A)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.fairyfuse.fairyfuse_backend import route_command  # type: ignore

# ---------------------------------------------------------------------------
# SSIM implementation (parent A)
# ---------------------------------------------------------------------------


def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index between two 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError("samples must have equal length")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) *
                                                    (vx + vy + c2))


# ---------------------------------------------------------------------------
# Gaussian KL & Free‑Energy (parent B)
# ---------------------------------------------------------------------------


def kl_gaussian(mu_q: np.ndarray, sigma_q: np.ndarray,
                mu_p: np.ndarray, sigma_p: np.ndarray) -> float:
    """KL[N(μ_q,σ_q²)‖N(μ_p,σ_p²)] – summed over all dimensions."""
    mu_q = np.asarray(mu_q, dtype=float)
    sigma_q = np.asarray(sigma_q, dtype=float)
    mu_p = np.asarray(mu_p, dtype=float)
    sigma_p = np.asarray(sigma_p, dtype=float)

    if np.any(sigma_q <= 0) or np.any(sigma_p <= 0):
        raise ValueError("Standard deviations must be strictly positive.")

    kl = (np.log(sigma_p / sigma_q) +
          (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2.0 * sigma_p ** 2) - 0.5)
    return float(np.sum(kl))


def free_energy_gaussian(mu_q: np.ndarray, sigma_q: np.ndarray,
                         mu_p: np.ndarray, sigma_p: np.ndarray,
                         obs: np.ndarray, sigma_obs: np.ndarray) -> float:
    """Variational free energy for a Gaussian generative model."""
    mu_q = np.asarray(mu_q, dtype=float)
    sigma_q = np.asarray(sigma_q, dtype=float)
    mu_p = np.asarray(mu_p, dtype=float)
    sigma_p = np.asarray(sigma_p, dtype=float)
    obs = np.asarray(obs, dtype=float)
    sigma_obs = np.asarray(sigma_obs, dtype=float)

    if np.any(sigma_obs <= 0):
        raise ValueError("sigma_obs must be strictly positive.")

    # KL term (belief vs prior)
    kl = kl_gaussian(mu_q, sigma_q, mu_p, sigma_p)

    # Reconstruction / negative log‑likelihood term
    recon = 0.5 * np.sum(((obs - mu_q) ** 2) / (sigma_obs ** 2) +
                         np.log(2.0 * np.pi * sigma_obs ** 2))

    return kl + recon


def belief_update(mu_q: np.ndarray, sigma_q: np.ndarray,
                 obs: np.ndarray, A: np.ndarray,
                 sigma_obs: np.ndarray, eta: float = 0.1) -> np.ndarray:
    """
    One gradient‑descent step on free energy w.r.t. μ_q (perception).

    dF/dμ_q = -(Aᵀ·Σ_obs⁻¹·(obs - A·μ_q))

    Parameters
    ----------
    mu_q : current belief mean (d,)
    sigma_q : belief std (d,) – not updated here
    obs : observation (k,)
    A : observation matrix (k, d)
    sigma_obs : observation noise std (k,)
    eta : learning rate

    Returns
    -------
    Updated μ_q (d,)
    """
    mu_q = np.atleast_1d(np.asarray(mu_q, dtype=float))
    obs = np.atleast_1d(np.asarray(obs, dtype=float))
    A = np.atleast_2d(np.asarray(A, dtype=float))
    sigma_obs = np.atleast_1d(np.asarray(sigma_obs, dtype=float))

    if sigma_obs.shape[0] != obs.shape[0]:
        raise ValueError("sigma_obs must match observation dimensionality.")
    if A.shape[0] != obs.shape[0] or A.shape[1] != mu_q.shape[0]:
        raise ValueError("A shape incompatible with obs and mu_q.")

    # Precision matrix (diagonal)
    prec = 1.0 / (sigma_obs ** 2)  # (k,)

    # Gradient: -(Aᵀ·(prec·(obs - A·μ_q)))
    pred = A @ mu_q                     # (k,)
    error = obs - pred                  # (k,)
    weighted_error = prec * error       # (k,)
    grad = -A.T @ weighted_error        # (d,)

    return mu_q - eta * grad


# ---------------------------------------------------------------------------
# Hybrid operations
# ---------------------------------------------------------------------------


def _text_to_vector(text: str) -> np.ndarray:
    """Encode a string as a vector of Unicode code points (uint8 range)."""
    return np.fromiter((ord(c) for c in text), dtype=np.float64)


def hybrid_route_packet(packet: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the ternary router, compute SSIM similarity and the corresponding
    variational free energy.

    The router output vector is interpreted as the current belief mean μ_q.
    The input vector is the observation.  A simple identity observation matrix
    (A = I) is used, and a zero‑mean wide prior (μ_p=0, σ_p=10) is assumed.
    """
    # 1. Route via ternary router
    text_in = str(packet.get("text_surface") or packet.get("raw_command") or
                  packet.get("text") or "")
    intent = str(packet.get("normalized_intent") or packet.get("intent") or
                 "bytewax_rete_bandit")
    context = {
        "source": packet.get("source"),
        "source_ref": packet.get("source_ref"),
        "ontology_terms": packet.get("ontology_terms") or [],
        "epistemic_flag": packet.get("epistemic_flag"),
        "payload": packet.get("payload") or {},
    }
    route = route_command(text_in[:4096], intent, context).to_dict()
    route["engine_channel"] = "cpu_fairyfuse_ternary"
    route["outbound_state"] = "draft_only"

    # 2. Encode texts
    text_out = str(route.get("text_surface") or route.get("raw_command") or
                   route.get("text") or "")
    vec_in = _text_to_vector(text_in)
    vec_out = _text_to_vector(text_out)

    # Pad to equal length for SSIM (required by the metric)
    if vec_in.size != vec_out.size:
        max_len = max(vec_in.size, vec_out.size)
        vec_in = np.pad(vec_in, (0, max_len - vec_in.size), constant_values=0)
        vec_out = np.pad(vec_out, (0, max_len - vec_out.size), constant_values=0)

    # 3. SSIM similarity
    similarity = ssim(vec_in, vec_out)
    route["similarity"] = similarity

    # 4. Free‑energy evaluation
    # Prior (wide Gaussian)
    mu_p = np.zeros_like(vec_out)
    sigma_p = np.full_like(vec_out, 10.0)

    # Observation noise derived from similarity
    eps = 1e-6
    sigma_obs = np.full_like(vec_in, eps + (1.0 - similarity) * 255.0 ** 2)

    # Belief variance – keep it modest
    sigma_q = np.full_like(vec_out, 1.0)

    # Identity observation matrix
    A = np.eye(vec_out.shape[0])

    fe = free_energy_gaussian(mu_q=vec_out,
                              sigma_q=sigma_q,
                              mu_p=mu_p,
                              sigma_p=sigma_p,
                              obs=vec_in,
                              sigma_obs=sigma_obs)
    route["free_energy"] = fe

    # Store vectors for downstream use (optional)
    route["_vec_input"] = vec_in.tolist()
    route["_vec_output"] = vec_out.tolist()

    return route


def hybrid_belief_refine(packet: Dict[str, Any],
                         eta: float = 0.05) -> Dict[str, Any]:
    """
    Perform a single perception update on the router's belief (output vector)
    using the free‑energy gradient.  The updated belief replaces the output
    vector in the returned route dict and a new free‑energy value is computed.
    """
    route = hybrid_route_packet(packet)

    vec_in = np.array(route["_vec_input"], dtype=float)
    vec_out = np.array(route["_vec_output"], dtype=float)

    # Re‑derive σ_obs from the stored similarity
    similarity = route["similarity"]
    sigma_obs = np.full_like(vec_in, 1e-6 + (1.0 - similarity) * 255.0 ** 2)

    # Identity observation matrix
    A = np.eye(vec_out.shape[0])

    sigma_q = np.full_like(vec_out, 1.0)
    mu_p = np.zeros_like(vec_out)
    sigma_p = np.full_like(vec_out, 10.0)

    # Gradient descent on μ_q
    mu_q_new = belief_update(mu_q=vec_out,
                             sigma_q=sigma_q,
                             obs=vec_in,
                             A=A,
                             sigma_obs=sigma_obs,
                             eta=eta)

    # Re‑compute free energy with the refined belief
    fe_new = free_energy_gaussian(mu_q=mu_q_new,
                                  sigma_q=sigma_q,
                                  mu_p=mu_p,
                                  sigma_p=sigma_p,
                                  obs=vec_in,
                                  sigma_obs=sigma_obs)

    # Update route dict
    route["refined_output_vector"] = mu_q_new.tolist()
    route["free_energy_refined"] = fe_new
    route["belief_update_step"] = eta

    return route


def evaluate_packet(packet: Dict[str, Any]) -> Dict[str, float]:
    """
    Convenience wrapper returning the key scalar metrics for a packet:
    * similarity (SSIM)
    * free_energy (initial)
    * free_energy_refined (after one perception step)
    """
    initial = hybrid_route_packet(packet)
    refined = hybrid_belief_refine(packet)

    return {
        "similarity": float(initial["similarity"]),
        "free_energy_initial": float(initial["free_energy"]),
        "free_energy_refined": float(refined["free_energy_refined"]),
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _random_packet() -> Dict[str, Any]:
    """Generate a synthetic packet compatible with the ternary router."""
    return {
        "text_surface": "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(80)),
        "normalized_intent": "bytewax_rete_bandit",
        "source": "synthetic_test",
        "payload": {},
    }


if __name__ == "__main__":
    pkt = _random_packet()
    print("=== Hybrid routing + free‑energy evaluation ===")
    result = evaluate_packet(pkt)
    for k, v in result.items():
        print(f"{k}: {v:.6f}")
    # Demonstrate the full refined route dict
    full = hybrid_belief_refine(pkt)
    print("\nSample of refined route keys:", list(full.keys())[:10])