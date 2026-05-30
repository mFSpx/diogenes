# DARWIN HAMMER — match 1243, survivor 5
# gen: 5
# parent_a: hybrid_pheromone_infotaxis_m3_s4.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1025_s1.py (gen4)
# born: 2026-05-29T23:34:51Z

"""Hybrid Pheromone‑Infotaxis‑NLMS Algorithm
================================================
This module fuses the **PheromoneStore** logic from *hybrid_pheromone_infotaxis_m3_s4.py*
(parent A) with the **NLMS‑driven decision‑hygiene** dynamics from
*hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1025_s1.py* (parent B).

Mathematical bridge
-------------------
* The pheromone entries on a given surface provide a **signal vector** `I`.
* Parent B treats an input vector `x` together with an external signal `I` and a
  scalar state `s` in the NLMS‑style weight update  

\[
\mathbf{w}_{t+1}= \mathbf{w}_t + 
\frac{\mu}{\epsilon+\|\mathbf{x}_t\|^2}\;e_t\;\mathbf{x}_t,
\qquad e_t = d_t - \mathbf{w}_t^\top\mathbf{x}_t .
\]

* We extend `x` by concatenating the **decayed pheromone values** `I` and the
  current **morphology‑driven priority scalar** `s`.  
  Thus the hybrid input becomes  

\[
\mathbf{z}_t = [\mathbf{x}_t; \mathbf{I}_t; s_t] .
\]

* The diffusion forcing from parent B adds Gaussian noise to the pheromone
  component before it is fed to the NLMS update:

\[
\tilde{\mathbf{I}}_t = \sqrt{\bar\alpha[t]}\,\mathbf{I}_t
                      + \sqrt{1-\bar\alpha[t]}\,\boldsymbol\varepsilon_t .
\]

* The **EndpointCircuitBreaker** (parent B) acts as a gate: weight updates are
  performed only while the circuit is closed (`allow() == True`).  Failures are
  recorded when the decay of pheromones falls below a threshold, demonstrating
  a feedback loop between the two subsystems.

The three core functions below illustrate this hybrid dynamics:
`decay_surface_pheromones`, `nlms_weight_update`, and `hybrid_step`.  The script
ends with a smoke test that exercises the whole pipeline without external
dependencies.

"""

import sys
import math
import random
import pathlib
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Pheromone handling
# ----------------------------------------------------------------------
class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Multiplicative decay since the last decay event."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """In‑memory singleton store for demo purposes."""
    _entries: Dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> List[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def decay_surface(cls, surface_key: str) -> List[Dict]:
        rows = []
        for entry in cls.get_by_surface(surface_key):
            before = entry.signal_value
            entry.apply_decay()
            rows.append({
                "pheromone_uuid": entry.uuid,
                "surface_key": entry.surface_key,
                "signal_kind": entry.signal_kind,
                "signal_value_before": before,
                "signal_value_after": entry.signal_value,
                "age_seconds": entry.age_seconds(),
            })
        return rows

# ----------------------------------------------------------------------
# Parent B – Decision hygiene & NLMS machinery
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Simple circuit breaker used to gate weight updates."""
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc)

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc)

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc)

    def allow(self) -> bool:
        return not self.open


class Morphology:
    """Placeholder morphology that yields a scalar priority."""
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

    def priority(self) -> float:
        """A simple heuristic: larger volume → higher priority, normalized."""
        volume = self.length * self.width * self.height
        return math.tanh(volume / (self.mass + 1e-6))


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
# Global mutable state for the NLMS learner (for demonstration)
_NLMS_W = None          # weight vector, initialized lazily
_NLMS_BIAS = 0.0
_MU = 0.5               # learning rate
_EPS = 1e-6             # regularisation term to avoid division by zero

def decay_surface_pheromones(surface_key: str) -> np.ndarray:
    """
    Decay all pheromones on *surface_key* and return the vector of
    decayed signal values ordered by entry UUID.
    """
    rows = PheromoneStore.decay_surface(surface_key)
    # Preserve ordering for reproducibility
    rows.sort(key=lambda r: r["pheromone_uuid"])
    values = np.array([r["signal_value_after"] for r in rows], dtype=float)
    return values

def nlms_weight_update(x: np.ndarray, I: np.ndarray, s: float,
                      desired: float) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS update using the concatenated input
    z = [x; I; s].

    Returns the updated weight vector and the current error.
    """
    global _NLMS_W, _NLMS_BIAS

    # Build the hybrid input vector
    z = np.concatenate([x, I, np.array([s], dtype=float)])

    if _NLMS_W is None:
        _NLMS_W = np.zeros_like(z)

    # Prediction and error
    y = np.dot(_NLMS_W, z) + _NLMS_BIAS
    e = desired - y

    # NLMS adaptation
    norm_sq = np.dot(z, z)
    step = _MU / (norm_sq + _EPS)
    _NLMS_W += step * e * z
    _NLMS_BIAS += step * e  # optional bias adaptation

    return _NLMS_W.copy(), e

def hybrid_step(surface_key: str,
                x: np.ndarray,
                desired_output: float,
                morphology: Morphology,
                breaker: EndpointCircuitBreaker,
                diffusion_schedule: List[float]) -> Dict:
    """
    One hybrid iteration:

    1. Decay pheromones on *surface_key* → raw signal vector I_raw.
    2. Apply diffusion forcing (Gaussian noise) according to the schedule.
    3. Compute priority scalar s from *morphology*.
    4. If the circuit breaker allows, run NLMS weight update.
    5. Record success/failure in the breaker based on the magnitude of the
       error (large error → failure).
    6. Return a diagnostic dictionary.
    """
    # 1. Decay pheromones
    I_raw = decay_surface_pheromones(surface_key)

    # 2. Diffusion forcing (parent B)
    t = len(diffusion_schedule) - 1  # use last schedule entry for simplicity
    alpha = diffusion_schedule[t] if diffusion_schedule else 0.0
    noise = np.random.normal(size=I_raw.shape)
    I_noisy = math.sqrt(alpha) * I_raw + math.sqrt(1 - alpha) * noise

    # 3. Morphology‑driven priority
    s = morphology.priority()

    # 4. Conditional NLMS update
    if breaker.allow():
        w, error = nlms_weight_update(x, I_noisy, s, desired_output)
        # 5. Simple heuristic: treat |error| > 1.0 as a failure
        if abs(error) > 1.0:
            breaker.record_failure()
            status = "failure"
        else:
            breaker.record_success()
            status = "success"
    else:
        w = _NLMS_W if _NLMS_W is not None else np.zeros_like(
            np.concatenate([x, I_noisy, np.array([s])]))
        error = None
        status = "blocked"

    # 6. Diagnostic payload
    return {
        "surface_key": surface_key,
        "decayed_pheromone": I_raw.tolist(),
        "noisy_pheromone": I_noisy.tolist(),
        "priority_s": s,
        "weight_vector": w.tolist(),
        "error": None if error is None else float(error),
        "circuit_status": status,
        "breaker_open": breaker.open,
    }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a surface and populate it with a few pheromones
    surface = "test_surface"
    for i in range(3):
        entry = PheromoneEntry(
            surface_key=surface,
            signal_kind="chem",
            signal_value=random.uniform(0.5, 2.0),
            half_life_seconds=30 + i * 10
        )
        PheromoneStore.add(entry)

    # Dummy external input vector x (e.g., sensor readings)
    x_vec = np.array([0.1, 0.4, 0.7])

    # Desired output for NLMS (could be a reward signal)
    d_target = 0.8

    # Morphology instance
    morph = Morphology(length=1.2, width=0.8, height=0.5, mass=2.0)

    # Circuit breaker
    cb = EndpointCircuitBreaker(failure_threshold=2)

    # Simple diffusion schedule (ᾱ[t]) – linearly increasing to 0.9
    diffusion_sched = np.linspace(0.1, 0.9, num=5).tolist()

    # Run a few hybrid steps
    for step in range(5):
        result = hybrid_step(
            surface_key=surface,
            x=x_vec,
            desired_output=d_target,
            morphology=morph,
            breaker=cb,
            diffusion_schedule=diffusion_sched
        )
        print(f"Step {step+1}: error={result['error']}, status={result['circuit_status']}")
        # Slight pause to let decay accumulate (optional)
        # time.sleep(1)

    print("Final weight vector:", result["weight_vector"])
    sys.exit(0)