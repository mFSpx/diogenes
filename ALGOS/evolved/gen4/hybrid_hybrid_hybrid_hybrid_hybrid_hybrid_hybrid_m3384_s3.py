# DARWIN HAMMER — match 3384, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hybrid_cockpi_hybrid_hybrid_fisher_m583_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_rlct_g_hybrid_cockpit_metri_m457_s2.py (gen3)
# born: 2026-05-29T23:49:38Z

"""Hybrid algorithm merging:
- Parent A: cockpit honesty/trust metrics, Fisher‑JEPA energy, constant‑velocity rectified flow.
- Parent B: Real Log Canonical Threshold (RLCT) estimation from loss curves and pheromone decay dynamics.

Mathematical bridge:
The scalar *trust* from cockpit metrics is interpreted as a pheromone concentration that
decays over time (Parent B).  The RLCT estimate, a measure of model complexity,
rescales the Fisher information (Parent A) and the trust‑weighted velocity,
producing a unified step:
    Δx = (trust·pheromone_decay) * (1 + λ_RLCT) * (x1‑x0) + λ_RLCT·Fisher(θ)
where λ_RLCT is the RLCT scalar.  This couples the information‑geometry term
(Fisher score) with the search‑dynamics term (velocity) under a common
pheromone‑modulated trust factor.  The JEPA energy is then evaluated on the
resulting candidate, closing the hybrid loop.
"""

import math
import random
import sys
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------
def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Proportion of claims backed by evidence."""
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Raw trust metric in [0,1] based on displayed vs unknown claims."""
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok / total))


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian kernel used for Fisher score."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    """Scalar Fisher information for a given latent theta."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def trust_weighted_velocity(x0: float, x1: float, trust: float) -> float:
    """Straight‑line velocity scaled by trust."""
    return trust * (x1 - x0)


def jeap_energy(candidate: float, prev_candidate: float, fisher: float) -> float:
    """JEPA‑style energy: squared deviation of candidate from a predictor that steps by Fisher."""
    predictor = np.array([prev_candidate + fisher])
    encoder = np.array([candidate])
    return np.sum((encoder - predictor) ** 2)


def hybrid_flow_loss(model_prediction: float, target: float, trust: float) -> float:
    """Simple loss that penalises deviation weighted by lack of trust."""
    return (model_prediction - target) ** 2 * (1.0 - trust)


# ----------------------------------------------------------------------
# Parent B building blocks
# ----------------------------------------------------------------------
class PheromoneSystem:
    """Manages pheromone signals with exponential decay."""
    def __init__(self):
        self.signals = {}

    def calculate(self, surface_key: str, kind: str, init_value: float, half_life_seconds: float) -> float:
        """Create or retrieve a signal and decay it by one second."""
        if surface_key not in self.signals:
            self.signals[surface_key] = {}
        if kind not in self.signals[surface_key]:
            self.signals[surface_key][kind] = init_value
        # decay for a single time step (Δt = 1 s)
        decayed = self.signals[surface_key][kind] * math.pow(0.5, 1.0 / half_life_seconds)
        self.signals[surface_key][kind] = decayed
        return decayed

    def update(self, surface_key: str, kind: str, delta: float) -> None:
        """Add delta to an existing signal."""
        self.signals.setdefault(surface_key, {}).setdefault(kind, 0.0)
        self.signals[surface_key][kind] += delta


def estimate_rlct(train_losses_per_n: list, n_values: list) -> float:
    """Estimate the Real Log Canonical Threshold from loss vs dataset size."""
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)

    if np.any(ns <= np.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    if losses.shape != ns.shape:
        raise ValueError("train_losses_per_n and n_values must have the same length")

    y = np.log(np.maximum(losses, 1e-300))
    x = np.log(np.log(ns))

    x_c = x - x.mean()
    y_c = y - y.mean()
    var_x = (x_c ** 2).sum()
    if var_x < 1e-15:
        raise ValueError("n_values have no variance in log(log(n)) space")
    return float((x_c * y_c).sum() / var_x)


# ----------------------------------------------------------------------
# Hybrid core functions (the required three+ functions)
# ----------------------------------------------------------------------
def compute_trust_pheromone(displayed_ok: int,
                            unknown_displayed_as_ok: int,
                            pheromone_sys: PheromoneSystem,
                            surface_key: str,
                            kind: str,
                            half_life_seconds: float = 10.0) -> float:
    """
    Combine raw cockpit honesty with a decayed pheromone concentration.
    The result stays in [0,1].
    """
    raw_trust = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)
    pheromone = pheromone_sys.calculate(surface_key, kind, init_value=raw_trust, half_life_seconds=half_life_seconds)
    combined = max(0.0, min(1.0, raw_trust * pheromone))
    return combined


def hybrid_step(x0: float,
                x1: float,
                theta: float,
                trust: float,
                rlct: float,
                prev_candidate: float) -> dict:
    """
    Perform a single hybrid update:
      * trust‑weighted velocity scaled by (1+rlct)
      * Fisher information scaled by rlct
      * JEPA energy evaluation
    Returns a dictionary with the new candidate and diagnostics.
    """
    # Velocity term (rectified flow) modulated by trust and RLCT
    velocity = trust_weighted_velocity(x0, x1, trust) * (1.0 + rlct)

    # Fisher term (information geometry) scaled by RLCT
    fisher = fisher_score(theta) * rlct

    # Candidate position after this step
    candidate = x0 + velocity + fisher

    # Energy of the candidate w.r.t. predictor
    energy = jeap_energy(candidate, prev_candidate, fisher)

    return {
        "candidate": candidate,
        "velocity": velocity,
        "fisher_scaled": fisher,
        "energy": energy,
        "rlct": rlct,
        "trust": trust
    }


def update_system_from_history(pheromone_sys: PheromoneSystem,
                               surface_key: str,
                               kind: str,
                               train_losses: list,
                               n_vals: list) -> float:
    """
    Estimate RLCT from the provided training history, use it to
    reinforce the pheromone signal (larger RLCT → stronger signal),
    and return the estimated RLCT.
    """
    rlct_est = estimate_rlct(train_losses, n_vals)
    # Reinforce pheromone proportionally to RLCT (clipped to a reasonable range)
    delta = max(0.0, min(1.0, rlct_est)) * 0.1  # small boost
    pheromone_sys.update(surface_key, kind, delta)
    return rlct_est


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy data for the test
    x_start = 0.0
    x_target = 5.0
    theta = 0.3

    displayed_ok = 80
    unknown_ok = 20

    # Instantiate pheromone manager
    pheromone = PheromoneSystem()

    # Compute trust combined with pheromone decay
    trust = compute_trust_pheromone(displayed_ok,
                                    unknown_ok,
                                    pheromone,
                                    surface_key="cockpit",
                                    kind="trust",
                                    half_life_seconds=15.0)

    # Simulated loss history for RLCT estimation
    n_history = [100, 200, 400, 800, 1600]
    loss_history = [0.9, 0.7, 0.55, 0.42, 0.33]

    rlct = update_system_from_history(pheromone,
                                      surface_key="cockpit",
                                      kind="trust",
                                      train_losses=loss_history,
                                      n_vals=n_history)

    # Perform a hybrid step
    step_info = hybrid_step(x_start,
                            x_target,
                            theta,
                            trust,
                            rlct,
                            prev_candidate=x_start)

    # Print diagnostics
    print("Trust (pheromone‑modulated):", trust)
    print("Estimated RLCT:", rlct)
    print("Hybrid step diagnostics:")
    for k, v in step_info.items():
        print(f"  {k}: {v}")

    # Verify that loss function behaves without error
    loss = hybrid_flow_loss(step_info["candidate"], target=x_target, trust=trust)
    print("Hybrid flow loss:", loss)