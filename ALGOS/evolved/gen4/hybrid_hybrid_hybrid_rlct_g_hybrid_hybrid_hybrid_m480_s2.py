# DARWIN HAMMER — match 480, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_rlct_grokking_hybrid_pheromone_inf_m208_s1.py (gen2)
# parent_b: hybrid_hybrid_hybrid_endpoi_epistemic_certainty_m59_s0.py (gen3)
# born: 2026-05-29T23:29:27Z

"""
Hybrid Algorithm: RLCT‑Grokking + Pheromone Infotaxis ↔ Morphology‑Based Epistemic Certainty

Parents:
- hybrid_hybrid_rlct_grokking_hybrid_pheromone_inf_m208_s1 (RLCT estimation,
  Hodgkin‑Huxley‑style energy, pheromone‑driven infotaxis)
- hybrid_hybrid_hybrid_endpoi_epistemic_certainty_m59_s0 (morphology metrics,
  righting‑time based recovery priority, epistemic certainty flags)

Mathematical Bridge:
Both parents manipulate *uncertainty* as a scalar that guides decision making.
RLCT provides an estimate of the information‑theoretic free‑energy curvature,
while the morphology side supplies a confidence weight (recovery_priority) that
quantifies epistemic certainty about a physical state.  We fuse them by
mapping the RLCT‑derived entropy term 𝑆 = –log |RLCT|  onto a *certainty factor*
𝜙 ∈ [0,1] and then weighting the morphology‑derived recovery priority with a
pheromone‑modulated exploration term.  The resulting hybrid score drives
energy‑aware exploration‑exploitation in a unified state‑space model.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent B – Morphology & Epistemic Certainty utilities
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Maps righting‑time index to a confidence weight in [0,1]."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Parent A – RLCT estimation, Hodgkin‑Huxley energy, pheromone infotaxis
# ----------------------------------------------------------------------
class PheromoneRLCTSystem:
    def __init__(self):
        # pheromone signal per discrete location (e.g., spatial grid)
        self.pheromone_signals: Dict[str, float] = {}

    # ---- RLCT estimation ------------------------------------------------
    @staticmethod
    def estimate_rlct_from_losses(train_losses_per_n: List[float],
                                  n_values: List[float]) -> float:
        """Linear regression of log(loss) on log(log(n))."""
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

    # ---- Hodgkin‑Huxley style currents ----------------------------------
    @staticmethod
    def sodium_current(V: float, m: float, h: float,
                       g_Na: float = 120.0, E_Na: float = 50.0) -> float:
        return g_Na * (m ** 3) * h * (V - E_Na)

    @staticmethod
    def potassium_current(V: float, n: float,
                          g_K: float = 36.0, E_K: float = -77.0) -> float:
        return g_K * (n ** 4) * (V - E_K)

    @staticmethod
    def optimize_energy(V: float, m: float, h: float, n: float,
                        g_Na: float = 120.0, E_Na: float = 50.0,
                        g_K: float = 36.0, E_K: float = -77.0) -> float:
        """Total ionic energy (sum of currents) used as a proxy for free energy."""
        return (PheromoneRLCTSystem.sodium_current(V, m, h, g_Na, E_Na) +
                PheromoneRLCTSystem.potassium_current(V, n, g_K, E_K))

    # ---- Pheromone handling ---------------------------------------------
    def decay_pheromones(self, half_life: float = 3600.0, dt: float = 1.0) -> None:
        """Exponential decay of all stored pheromone signals."""
        decay_factor = 0.5 ** (dt / half_life)
        for loc in list(self.pheromone_signals.keys()):
            self.pheromone_signals[loc] *= decay_factor
            if self.pheromone_signals[loc] < 1e-12:
                del self.pheromone_signals[loc]

    def deposit_pheromone(self, location: str, amount: float) -> None:
        """Add pheromone to a location, creating the entry if needed."""
        self.pheromone_signals[location] = self.pheromone_signals.get(location, 0.0) + amount

    def expected_entropy(self) -> float:
        """Entropy of the normalized pheromone distribution (information gain)."""
        total = sum(self.pheromone_signals.values())
        if total == 0:
            return 0.0
        probs = np.array(list(self.pheromone_signals.values())) / total
        return -float(np.sum(probs * np.log(np.maximum(probs, 1e-300))))


# ----------------------------------------------------------------------
# Hybrid Operations (bridge between the two parents)
# ----------------------------------------------------------------------
def compute_certainty_factor(rlct: float) -> float:
    """
    Convert RLCT (a curvature measure) into a certainty factor φ ∈ [0,1].
    Small |RLCT| → high uncertainty → low φ.
    """
    if rlct == 0:
        return 0.0
    entropy_like = -math.log(abs(rlct) + 1e-12)  # larger entropy for smaller |rlct|
    # Map entropy to [0,1] via a sigmoid that saturates at 1 for low entropy.
    phi = 1.0 / (1.0 + math.exp(entropy_like - 2.0))
    return max(0.0, min(1.0, phi))


def hybrid_score(V: float, m: float, h: float, n: float,
                 train_losses: List[float], n_vals: List[float],
                 morphology: Morphology,
                 pheromone_system: PheromoneRLCTSystem,
                 location: str) -> float:
    """
    Unified decision metric that blends:
      • Energy from Hodgkin‑Huxley currents (A)
      • RLCT‑derived certainty factor (A ↔ B)
      • Morphology recovery priority (B)
      • Pheromone‑based expected entropy (A)

    The score is higher for configurations that are energetically favorable,
    morphologically reliable, and guided by informative pheromone cues.
    """
    # 1. Core physical energy (lower is better)
    energy = pheromone_system.optimize_energy(V, m, h, n)

    # 2. RLCT → certainty factor
    rlct = pheromone_system.estimate_rlct_from_losses(train_losses, n_vals)
    phi = compute_certainty_factor(rlct)

    # 3. Morphology confidence
    priority = recovery_priority(morphology)

    # 4. Pheromone influence (higher entropy → more exploration, we penalize)
    entropy = pheromone_system.expected_entropy()
    pheromone_weight = math.exp(-entropy)  # low entropy → stronger exploitation

    # 5. Combine (weights chosen heuristically)
    score = ( -energy * 0.4               # we want low energy → high score
              + phi * 0.3
              + priority * 0.2
              + pheromone_weight * 0.1 )
    # Deposit pheromone proportional to the achieved score to close the loop
    pheromone_system.deposit_pheromone(location, amount=score)
    return score


def update_system_state(V: float, m: float, h: float, n: float,
                       train_losses: List[float], n_vals: List[float],
                       morphology: Morphology,
                       pheromone_system: PheromoneRLCTSystem,
                       location: str,
                       dt: float = 1.0) -> Tuple[float, float]:
    """
    Perform a single hybrid update:
      1. Decay pheromones.
      2. Compute hybrid score.
      3. Adjust membrane potential V towards a target that minimizes the score.
    Returns the new (V, score).
    """
    pheromone_system.decay_pheromones(dt=dt)

    score = hybrid_score(V, m, h, n,
                         train_losses, n_vals,
                         morphology,
                         pheromone_system,
                         location)

    # Simple gradient‑free adjustment: move V opposite to the sign of the score derivative
    # (here we approximate by a small step toward lower energy)
    grad_estimate = (PheromoneRLCTSystem.sodium_current(V + 0.1, m, h) +
                     PheromoneRLCTSystem.potassium_current(V + 0.1, n) -
                     PheromoneRLCTSystem.sodium_current(V, m, h) -
                     PheromoneRLCTSystem.potassium_current(V, n))
    V_new = V - 0.05 * grad_estimate  # learning rate 0.05
    return V_new, score


def summarize_state(pheromone_system: PheromoneRLCTSystem,
                    morphology: Morphology) -> dict:
    """
    Produce a diagnostic dictionary containing:
      - total pheromone mass
      - entropy of pheromone distribution
      - morphology indices
      - latest RLCT estimate (if possible)
    """
    total_pheromone = sum(pheromone_system.pheromone_signals.values())
    entropy = pheromone_system.expected_entropy()
    sphericity = sphericity_index(morphology.length,
                                  morphology.width,
                                  morphology.height)
    flatness = flatness_index(morphology.length,
                              morphology.width,
                              morphology.height)
    righting = righting_time_index(morphology)

    # Dummy RLCT estimate (requires training data; we return None if unavailable)
    rlct = None
    try:
        # Use placeholder data if the system has at least one pheromone entry
        if pheromone_system.pheromone_signals:
            dummy_losses = [random.random() for _ in range(5)]
            dummy_ns = [np.e ** (i + 2) for i in range(5)]
            rlct = pheromone_system.estimate_rlct_from_losses(dummy_losses, dummy_ns)
    except Exception:
        rlct = None

    return {
        "total_pheromone": total_pheromone,
        "pheromone_entropy": entropy,
        "sphericity": sphericity,
        "flatness": flatness,
        "righting_time_index": righting,
        "recovery_priority": recovery_priority(morphology),
        "rlct_estimate": rlct,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise components
    pheromone_system = PheromoneRLCTSystem()
    morphology = Morphology(length=2.0, width=1.0, height=0.5, mass=3.0)

    # Dummy training loss curve
    train_losses = [0.9, 0.7, 0.55, 0.43, 0.35]
    n_vals = [10, 20, 40, 80, 160]

    # Initial biophysical state
    V, m_gate, h_gate, n_gate = -65.0, 0.05, 0.6, 0.32
    location = "cell_001"

    # Run a few hybrid updates
    for step in range(5):
        V, score = update_system_state(V, m_gate, h_gate, n_gate,
                                       train_losses, n_vals,
                                       morphology, pheromone_system,
                                       location, dt=1.0)
        print(f"Step {step+1:02d} | V={V:.3f} mV | Score={score:.4f}")

    # Final diagnostics
    diagnostics = summarize_state(pheromone_system, morphology)
    print("\nDiagnostics:")
    for k, v in diagnostics.items():
        print(f"  {k}: {v}")