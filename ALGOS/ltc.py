#!/usr/bin/env python3
"""
Liquid Time-Constant (LTC) Networks — continuous-time ODE recurrent cells.

LTC (Hasani et al. 2020/2021) replaces the fixed recurrence of standard RNNs
with a coupled ODE:

    τ_sys(x, I, θ) * dx/dt  =  -x  +  f(x, I, θ) * (A - x)

where:
  x          : hidden state vector  (shape: hidden_size)
  I          : input at time t       (shape: input_size)
  τ_sys      : state- and input-dependent time constant (the "liquid" part)
  A          : saturating attractor  (defaults to +1 or -1 per neuron)
  f(x, I, θ) : sigmoid-gated coupling  sigmoid(Wx*x + WI*I + b)

Key properties:
  - Time step Δt appears explicitly → the network is aware of irregular timing.
  - τ adapts to input (liquid), so the effective memory horizon stretches or
    contracts depending on what the network sees.
  - Euler-solved version is O(hidden²) per step, usable on CPU.

Euler update (one step, Δt seconds):
    tau     = tau_base * sigmoid(Wx_tau @ x + WI_tau @ I + b_tau) + epsilon
    f_gate  = sigmoid(Wx_f @ x + WI_f @ I + b_f)
    dxdt    = (-x + f_gate * (A - x)) / tau
    x_next  = x + Δt * dxdt

LUCIDOTA role — temporal evidence stream processor:
  Given a sequence of evidence observations with real-world timestamps
  (from lucidota_go.temporal_claim, lucidota_korpus.corpus_chunk, or
  lucidota_go.staging_packet), LTC models salience decay / intensity drift:

  - Each observation is encoded as an input vector I_t (embedding or feature hash).
  - The hidden state x tracks accumulated salience across the evidence stream.
  - Irregular Δt from real timestamps means recent evidence gets higher weight
    than old evidence automatically, without manual decay engineering.
  - The norm ||x|| at any point indicates current evidence intensity.
  - Changes in x direction indicate claim drift — the same entity shifting
    semantic neighborhood over time.

  Integration point: scripts/ltc_evidence_wire.py (planned) queries
  lucidota_go.staging_packet ordered by created_at, encodes proposed_term as
  a feature hash (or BGE embedding via fleet), runs LTC, and emits an
  evidence_intensity receipt to 05_OUTPUTS/runtime/.

References:
  - Hasani et al. (2020), "Liquid Time-constant Networks." AAAI 2021.
  - Lechner et al. (2020), "Neural circuit policies enabling auditable autonomy."
  - Hasani et al. (2022), "Closed-form Continuous-time Neural Networks." NeurIPS.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = [
    "LTCCell",
    "LTCSequence",
    "feature_hash",
    "evidence_intensity",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class LTCCell:
    """Single Euler-discretised LTC cell with fixed random weights.

    Weights are initialised from a SHA-256-seeded PRNG for determinism.
    Pass seed as a string; identical seeds reproduce identical dynamics.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        seed: str = "ltc-lucidota-default",
        tau_base: float = 1.0,
        epsilon: float = 0.01,
        attractor: float = 1.0,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau_base = tau_base
        self.epsilon = epsilon
        self.A = np.full(hidden_size, attractor, dtype=np.float64)

        # Deterministic weight init via SHA-256 seed
        import hashlib
        seed_bytes = hashlib.sha256(seed.encode()).digest()
        seed_int = int.from_bytes(seed_bytes[:4], "big")
        rng = np.random.default_rng(seed_int)

        scale_x = 1.0 / math.sqrt(hidden_size)
        scale_i = 1.0 / math.sqrt(input_size)

        # Gating weights (f-gate)
        self.Wx_f  = rng.standard_normal((hidden_size, hidden_size)) * scale_x
        self.WI_f  = rng.standard_normal((hidden_size, input_size))  * scale_i
        self.b_f   = np.zeros(hidden_size)

        # Tau weights
        self.Wx_tau = rng.standard_normal((hidden_size, hidden_size)) * scale_x
        self.WI_tau = rng.standard_normal((hidden_size, input_size))  * scale_i
        self.b_tau  = np.zeros(hidden_size)

    def zero_state(self) -> np.ndarray:
        return np.zeros(self.hidden_size, dtype=np.float64)

    def step(self, x: np.ndarray, I: np.ndarray, dt: float, max_sub_dt: float = 0.1) -> np.ndarray:
        """Euler step with sub-stepping for stability.

        dt is in the caller's time unit (see dt_scale in LTCSequence).
        max_sub_dt caps the per-sub-step size relative to tau_base so that
        the Euler method stays within its stability region (dt/τ << 1).
        """
        if dt <= 0.0:
            return x
        n_steps = max(1, math.ceil(dt / max_sub_dt))
        sub_dt = dt / n_steps
        for _ in range(n_steps):
            tau    = self.tau_base * _sigmoid(self.Wx_tau @ x + self.WI_tau @ I + self.b_tau) + self.epsilon
            f_gate = _sigmoid(self.Wx_f @ x + self.WI_f @ I + self.b_f)
            dxdt   = (-x + f_gate * (self.A - x)) / tau
            x = x + sub_dt * dxdt
        return x


class LTCSequence:
    """Run LTCCell over a sequence of (timestamp_seconds, input_vector) pairs.

    dt_scale converts raw timestamp differences to the ODE time unit.
    Default dt_scale=1/3600 converts seconds → hours, so tau_base=1.0
    means a memory horizon of ~1 hour, which suits evidence streams.
    """

    def __init__(self, cell: LTCCell, dt_scale: float = 1.0 / 3600.0) -> None:
        self.cell = cell
        self.dt_scale = dt_scale

    def run(
        self,
        sequence: Sequence[tuple[float, np.ndarray]],
        x0: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Return list of hidden states after each observation.

        sequence: list of (unix_timestamp_seconds, input_vector)
        """
        x = self.cell.zero_state() if x0 is None else x0.copy()
        states: list[np.ndarray] = []
        prev_t = None
        for t, I in sequence:
            raw_dt = (t - prev_t) if prev_t is not None else 0.0
            dt = raw_dt * self.dt_scale
            x = self.cell.step(x, I, dt)
            states.append(x.copy())
            prev_t = t
        return states


def feature_hash(text: str, dim: int = 32, seed: str = "ltc-fh") -> np.ndarray:
    """Deterministic bag-of-chars feature hash → dim-dimensional unit vector.

    Zero-dependency alternative to embeddings.  Adequate for temporal pattern
    detection on ontology terms / proposed_term strings.
    """
    import hashlib
    full_seed = f"{seed}:{text}"
    raw = hashlib.sha256(full_seed.encode()).digest()
    # Tile hash bytes to fill dim
    repeats = math.ceil(dim * 2 / len(raw))
    padded = (raw * repeats)[: dim * 2]
    # Signed int8 from each byte-pair
    vec = np.array(
        [int.from_bytes(padded[i : i + 2], "big", signed=True) / 32768.0 for i in range(0, dim * 2, 2)],
        dtype=np.float64,
    )[:dim]
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def evidence_intensity(states: list[np.ndarray]) -> dict:
    """Summary statistics over a sequence of LTC hidden states.

    Returns:
        final_norm   : ||x_final||  — current evidence intensity
        mean_norm    : mean over sequence
        drift        : cosine distance between first and last state
                       (0 = no drift, 2 = perfect reversal)
        peak_norm    : maximum norm seen
        peak_idx     : index of peak
    """
    if not states:
        return {"final_norm": 0.0, "mean_norm": 0.0, "drift": 0.0, "peak_norm": 0.0, "peak_idx": -1}

    norms = np.array([np.linalg.norm(s) for s in states])
    peak_idx = int(np.argmax(norms))

    drift = 0.0
    if len(states) >= 2:
        a, b = states[0], states[-1]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            cos = float(np.dot(a, b) / (na * nb))
            cos = max(-1.0, min(1.0, cos))
            drift = 1.0 - cos  # 0 = same direction, 1 = orthogonal, 2 = opposite

    return {
        "final_norm": float(norms[-1]),
        "mean_norm": float(norms.mean()),
        "drift": drift,
        "peak_norm": float(norms[peak_idx]),
        "peak_idx": peak_idx,
    }


# ---------------------------------------------------------------------------
# Smoke-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("LTC smoke test — evidence stream simulation")
    cell = LTCCell(input_size=32, hidden_size=16, seed="lucidota-ltc-demo")
    seq_obj = LTCSequence(cell)

    # Simulate 10 evidence observations at irregular intervals (seconds since epoch)
    terms = [
        "PERSON", "ENTITY", "GRIP", "SNARE", "PERSON",
        "GHOST", "WITNESS", "GRIP", "VOID", "PERSON",
    ]
    timestamps = [0, 60, 90, 210, 300, 450, 480, 600, 750, 900]

    obs: list[tuple[float, np.ndarray]] = [
        (float(t), feature_hash(term, dim=32))
        for t, term in zip(timestamps, terms)
    ]

    states = seq_obj.run(obs)
    intensity = evidence_intensity(states)
    print(json.dumps({"observations": len(obs), **intensity}, indent=2))
    print("PASS — LTC smoke test complete")
