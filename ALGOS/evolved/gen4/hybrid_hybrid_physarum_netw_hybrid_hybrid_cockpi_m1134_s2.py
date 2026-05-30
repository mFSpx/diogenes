# DARWIN HAMMER — match 1134, survivor 2
# gen: 4
# parent_a: hybrid_physarum_network_hybrid_hybrid_bandit_m11_s2.py (gen3)
# parent_b: hybrid_hybrid_cockpit_metri_hard_truth_math_m27_s0.py (gen2)
# born: 2026-05-29T23:33:06Z

"""
Hybrid module unifying:
- physarum_network (flux‑based conductance dynamics) and hybrid_hybrid_bandit_router (bandit propensity updates)
- cockpit honesty/evidence metrics (anti_slop_ratio, cockpit_honesty) with hard‑truth telemetry (stylometry‑derived trust)

Mathematical bridge:
The scalar *trust* derived from cockpit metrics is used as a multiplicative factor in two places:
1. It scales the bandit propensity matrix, thus modulating the probability of selecting an edge.
2. It modulates the gain term of the physarum conductance update, i.e. the effective reinforcement
   becomes  α·trust  instead of the raw α.

Consequently the flux through an edge influences the bandit reward, while the bandit reward
(and the trust‑weighted gain) influences the conductance that determines the pressure field.
The system therefore forms a closed loop where evidence‑based trust controls exploration
and the physarum dynamics shape exploitation.
"""

import math
import random
import sys
from pathlib import Path
from typing import Tuple, Dict

import numpy as np

# ---------------------------------------------------------------------------
# Parent A – physarum & bandit primitives
# ---------------------------------------------------------------------------
def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    """Flux through an edge given its conductance, length and endpoint pressures."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float, dt: float = 1.0,
                       gain: float = 1.0, decay: float = 0.05) -> float:
    """Physarum conductance ODE discretisation."""
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non‑negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


# ---------------------------------------------------------------------------
# Parent B – cockpit metrics and stylometry‑based trust
# ---------------------------------------------------------------------------
def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Ratio of supported claims, clamped to [0, 1]."""
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0,
                claims_with_evidence / total_claims_emitted))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Fraction of displayed items that are known‑good, clamped to [0, 1]."""
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok / total))


def stylometry_vector(text: str) -> np.ndarray:
    """
    Very lightweight stylometry: returns a 3‑dimensional feature vector
    [character count, vowel ratio, average word length].
    """
    if not text:
        return np.zeros(3)
    chars = len(text)
    vowels = sum(c.lower() in "aeiou" for c in text)
    words = text.split()
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0.0
    return np.array([chars, vowels / max(chars, 1), avg_word_len])


def trust_from_metrics(displayed_ok: int, unknown_displayed_as_ok: int,
                       claims_with_evidence: int, total_claims_emitted: int,
                       sample_text: str) -> float:
    """
    Combine cockpit honesty, anti‑slop ratio and a simple stylometry‑derived signal
    into a single trust scalar in [0, 1].
    """
    honesty = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)
    evidence = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    style_feat = stylometry_vector(sample_text)
    # Map stylometry to [0,1] via a sigmoid on a linear combination
    w = np.array([0.001, 5.0, 0.05])  # hand‑tuned weights
    style_score = 1.0 / (1.0 + math.exp(-np.dot(w, style_feat)))
    return honesty * evidence * style_score


# ---------------------------------------------------------------------------
# Hybrid class integrating both worlds
# ---------------------------------------------------------------------------
class HybridBanditPhysarumCockpit:
    """
    Core object holding:
    - conductance matrix (physarum)
    - propensity matrix (bandit)
    - trust scalar (cockpit + stylometry)
    The trust scalar modulates both propensity and conductance gain.
    """

    def __init__(self,
                 d_in: int,
                 d_out: int,
                 seed: int = 0,
                 base_eta: float = 0.01,
                 alpha: float = 1.0,
                 beta: float = 0.05,
                 dt: float = 1.0,
                 store_decay: float = 0.99) -> None:
        self.rng = np.random.default_rng(seed)
        self.base_eta = base_eta          # learning rate for propensity smoothing
        self.alpha = alpha                # base physarum gain
        self.beta = beta                  # physarum decay
        self.dt = dt
        self.store_decay = store_decay
        self.d_in = d_in
        self.d_out = d_out
        self.conductance = np.ones((d_in, d_out))
        self.propensity = np.ones((d_in, d_out)) / (d_out)  # normalized
        self.trust = 1.0

    # -----------------------------------------------------------------------
    # Core hybrid operations
    # -----------------------------------------------------------------------
    def compute_trust(self,
                      displayed_ok: int,
                      unknown_displayed_as_ok: int,
                      claims_with_evidence: int,
                      total_claims_emitted: int,
                      sample_text: str) -> None:
        """Update internal trust scalar."""
        self.trust = trust_from_metrics(displayed_ok,
                                        unknown_displayed_as_ok,
                                        claims_with_evidence,
                                        total_claims_emitted,
                                        sample_text)

    def select_action(self) -> Tuple[int, int]:
        """
        Stochastic selection of an (input, output) edge proportional to
        propensity * conductance (i.e. the effective flow capacity).
        Returns the flat index (i, j).
        """
        weights = self.propensity * self.conductance
        flat_weights = weights.ravel()
        flat_weights_sum = flat_weights.sum()
        if flat_weights_sum == 0:
            # fallback to uniform
            probs = np.full_like(flat_weights, 1 / flat_weights.size)
        else:
            probs = flat_weights / flat_weights_sum
        choice = self.rng.choice(flat_weights.size, p=probs)
        i, j = divmod(choice, self.d_out)
        return i, j

    def hybrid_update(self,
                      action: Tuple[int, int],
                      reward: float,
                      edge_length: float,
                      pressure_a: float,
                      pressure_b: float) -> None:
        """
        Perform a single hybrid update:
        1. Compute flux on the selected edge.
        2. Update conductance using trust‑scaled gain.
        3. Update propensity using a trust‑weighted exponential moving average.
        """
        i, j = action
        # 1. Flux
        q = flux(self.conductance[i, j], edge_length, pressure_a, pressure_b)

        # 2. Conductance update – gain is α·trust
        gain = self.alpha * self.trust
        self.conductance[i, j] = update_conductance(
            self.conductance[i, j], q, dt=self.dt, gain=gain, decay=self.beta)

        # 3. Propensity update – simple bandit rule with trust‑scaled learning rate
        eta = self.base_eta * self.trust
        # Incremental update towards reward (treated as desirability)
        self.propensity[i, j] = (1 - eta) * self.propensity[i, j] + eta * reward

        # Normalise propensity rows to keep them a probability distribution
        row_sum = self.propensity[i, :].sum()
        if row_sum > 0:
            self.propensity[i, :] /= row_sum

    # -----------------------------------------------------------------------
    # Helper for external inspection
    # -----------------------------------------------------------------------
    def current_state(self) -> Dict[str, np.ndarray]:
        """Return a snapshot of the internal matrices and trust."""
        return {
            "conductance": self.conductance.copy(),
            "propensity": self.propensity.copy(),
            "trust": np.array([self.trust])
        }


# ---------------------------------------------------------------------------
# Demonstration functions (the required three+ functions)
# ---------------------------------------------------------------------------
def hybrid_step(model: HybridBanditPhysarumCockpit,
                displayed_ok: int,
                unknown_displayed_as_ok: int,
                claims_with_evidence: int,
                total_claims_emitted: int,
                sample_text: str,
                edge_length: float,
                pressure_a: float,
                pressure_b: float) -> Tuple[int, int, float]:
    """
    Executes one full hybrid iteration:
    - refresh trust,
    - select an action,
    - compute a synthetic reward (here: flux magnitude),
    - perform hybrid_update,
    Returns the chosen edge indices and the reward used.
    """
    model.compute_trust(displayed_ok,
                        unknown_displayed_as_ok,
                        claims_with_evidence,
                        total_claims_emitted,
                        sample_text)

    action = model.select_action()
    i, j = action
    # Synthetic reward proportional to absolute flux (encourages high flow)
    q = flux(model.conductance[i, j], edge_length, pressure_a, pressure_b)
    reward = abs(q)

    model.hybrid_update(action, reward, edge_length, pressure_a, pressure_b)
    return i, j, reward


def ideal_velocity(trust: float, base_velocity: float = 1.0) -> float:
    """
    Bridge to the hard‑truth telemetry side: the ideal velocity field is
    modulated by trust, mirroring the formulation in the original cockpit‑hard‑truth hybrid.
    """
    return base_velocity * trust


def run_demo(iterations: int = 10) -> None:
    """Simple smoke‑test that runs the hybrid algorithm for a few steps."""
    d_in, d_out = 4, 5
    model = HybridBanditPhysarumCockpit(d_in, d_out, seed=42)

    for step in range(iterations):
        # Randomly generated cockpit‑style statistics
        displayed_ok = random.randint(0, 20)
        unknown_ok = random.randint(0, 5)
        claims_with_evidence = random.randint(0, 15)
        total_claims = random.randint(1, 20)
        sample_text = "The quick brown fox jumps over the lazy dog."

        # Physical parameters for the selected edge (same for simplicity)
        edge_len = random.uniform(0.5, 2.0)
        pressure_a = random.uniform(0.0, 5.0)
        pressure_b = random.uniform(0.0, 5.0)

        i, j, reward = hybrid_step(model,
                                   displayed_ok,
                                   unknown_ok,
                                   claims_with_evidence,
                                   total_claims,
                                   sample_text,
                                   edge_len,
                                   pressure_a,
                                   pressure_b)

        vel = ideal_velocity(model.trust)
        print(f"Step {step+1:02d}: edge ({i},{j}) reward={reward:.4f} trust={model.trust:.4f} vel={vel:.4f}")

    # Final state sanity check
    state = model.current_state()
    assert state["conductance"].shape == (d_in, d_out)
    assert state["propensity"].shape == (d_in, d_out)
    assert 0.0 <= state["trust"][0] <= 1.0


if __name__ == "__main__":
    run_demo()