# DARWIN HAMMER — match 1445, survivor 4
# gen: 5
# parent_a: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s1.py (gen1)
# parent_b: hybrid_hybrid_diffusion_for_hybrid_hybrid_gliner_m369_s0.py (gen4)
# born: 2026-05-29T23:36:22Z

"""Hybrid Self‑Righting Circuit Breaker with Diffusion‑Pheromone Dynamics.

Parents:
- hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s1.py (morphology‑based recovery priority & circuit breaker)
- hybrid_hybrid_diffusion_for_hybrid_hybrid_gliner_m369_s0.py (diffusion noise schedule & pheromone decay)

Mathematical bridge:
The recovery priority p∈[0,1] derived from morphology (Parent A) scales the diffusion noise variance σ² = σ₀²·(1‑p) and simultaneously modulates pheromone signal decay by a factor (1‑p). Thus the same scalar couples the stochastic corruption of inputs with the deterministic fading of pheromone cues, yielding a unified dynamical system."""
import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

# ---------- Parent A utilities ----------
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

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Scaled to [0,1]; higher priority → more resilient."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

class EndpointCircuitBreaker:
    """Simple circuit breaker whose threshold adapts to recovery priority."""
    def __init__(self, failure_threshold: int = 3, morphology: Morphology = None):
        self.base_threshold = failure_threshold
        self.morphology = morphology
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def _adjusted_threshold(self) -> float:
        """Higher recovery priority raises the effective threshold."""
        if self.morphology is None:
            return self.base_threshold
        p = recovery_priority(self.morphology)
        return self.base_threshold * (1.0 + p)  # linear scaling

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc).isoformat()

    def record_failure(self) -> None:
        self.failures += 1
        self.last_event_at = datetime.now(timezone.utc).isoformat()
        if self.failures >= self._adjusted_threshold():
            self.open = True

    def can_proceed(self) -> bool:
        return not self.open

# ---------- Parent B utilities ----------
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"

class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(pathlib.Path('/proc/self/cmdline').stat().st_ino)  # cheap unique id
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc).timestamp()
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return datetime.now(timezone.utc).timestamp() - self.last_decay

    def decay_factor(self) -> float:
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self, priority: float = 0.0) -> None:
        """
        Decay is damped by recovery priority:
        effective_factor = decay_factor ** (1‑p)
        """
        base_factor = self.decay_factor()
        effective_factor = base_factor ** (1.0 - priority)
        self.signal_value *= effective_factor
        self.last_decay = datetime.now(timezone.utc).timestamp()

class PheromoneStore:
    """In‑memory store for pheromone entries."""
    def __init__(self):
        self._entries: Dict[str, PheromoneEntry] = {}

    def add(self, entry: PheromoneEntry) -> None:
        self._entries[entry.uuid] = entry

    def get_all(self) -> List[PheromoneEntry]:
        return list(self._entries.values())

    def decay_all(self, priority: float = 0.0) -> None:
        for e in self._entries.values():
            e.apply_decay(priority)

# ---------- Hybrid core ----------
def diffusion_noise_schedule(step: int, total_steps: int,
                             base_sigma: float = 1.0,
                             priority: float = 0.0) -> float:
    """
    Gaussian noise variance that linearly anneals over steps.
    The recovery priority reduces the noise amplitude:
        sigma = base_sigma * (1 - p) * (1 - step/total_steps)
    """
    if not (0.0 <= priority <= 1.0):
        raise ValueError("priority must be in [0,1]")
    progress = step / max(1, total_steps)
    sigma = base_sigma * (1.0 - priority) * (1.0 - progress)
    return max(0.0, sigma)

def corrupt_signal(signal: np.ndarray, sigma: float) -> np.ndarray:
    """Add isotropic Gaussian noise with variance sigma²."""
    if sigma <= 0.0:
        return signal
    noise = np.random.normal(loc=0.0, scale=sigma, size=signal.shape)
    return signal + noise

def hybrid_step(state: Dict[str, Any],
                morphology: Morphology,
                pheromone_store: PheromoneStore,
                total_steps: int = 10) -> Dict[str, Any]:
    """
    Perform one hybrid iteration:
    1. Compute recovery priority p.
    2. Generate diffusion noise σ based on p and current step.
    3. Corrupt the input vector.
    4. Decay pheromones with p‑dependent factor.
    5. Update circuit breaker based on a simple success/failure test.
    Returns updated state.
    """
    step = state.get("step", 0)
    cb: EndpointCircuitBreaker = state["circuit_breaker"]
    input_vec: np.ndarray = state["input"]

    # 1. priority
    p = recovery_priority(morphology)

    # 2. diffusion sigma
    sigma = diffusion_noise_schedule(step, total_steps, base_sigma=1.0, priority=p)

    # 3. corrupt input
    corrupted = corrupt_signal(input_vec, sigma)

    # 4. pheromone decay
    pheromone_store.decay_all(priority=p)

    # 5. simplistic success test: if L2 norm decreased, count as success
    if np.linalg.norm(corrupted) < np.linalg.norm(input_vec):
        cb.record_success()
    else:
        cb.record_failure()

    # Prepare next state
    new_state = {
        "step": step + 1,
        "circuit_breaker": cb,
        "input": corrupted,
        "priority": p,
        "sigma": sigma,
        "pheromones": pheromone_store.get_all(),
    }
    return new_state

# ---------- Demonstration functions ----------
def init_hybrid_system(morphology: Morphology,
                       input_dim: int = 5) -> Tuple[Dict[str, Any], PheromoneStore]:
    """Create initial state and a pheromone store populated with a demo entry."""
    cb = EndpointCircuitBreaker(failure_threshold=3, morphology=morphology)
    init_vec = np.random.randn(input_dim)
    store = PheromoneStore()
    demo_entry = PheromoneEntry(
        surface_key="demo",
        signal_kind="signal",
        signal_value=1.0,
        half_life_seconds=30
    )
    store.add(demo_entry)
    state = {"step": 0, "circuit_breaker": cb, "input": init_vec}
    return state, store

def run_hybrid_simulation(steps: int = 10) -> None:
    """Run the hybrid system for a given number of steps, printing key metrics."""
    morph = Morphology(length=0.5, width=0.3, height=0.2, mass=2.0)
    state, store = init_hybrid_system(morph, input_dim=3)

    for _ in range(steps):
        state = hybrid_step(state, morph, store, total_steps=steps)
        cb: EndpointCircuitBreaker = state["circuit_breaker"]
        print(f"Step {state['step']}: sigma={state['sigma']:.3f}, "
              f"priority={state['priority']:.3f}, "
              f"circuit_open={cb.open}, failures={cb.failures}")

def evaluate_recovery_effect():
    """Show how varying morphology changes priority and thus diffusion."""
    for mass in [0.5, 1.0, 2.0, 5.0]:
        morph = Morphology(length=0.5, width=0.3, height=0.2, mass=mass)
        p = recovery_priority(morph)
        sigma = diffusion_noise_schedule(step=0, total_steps=10, priority=p)
        print(f"mass={mass:.1f} → priority={p:.3f}, initial sigma={sigma:.3f}")

# ---------- Smoke test ----------
if __name__ == "__main__":
    print("=== Hybrid Simulation Demo ===")
    run_hybrid_simulation(steps=12)
    print("\n=== Recovery Priority Evaluation ===")
    evaluate_recovery_effect()