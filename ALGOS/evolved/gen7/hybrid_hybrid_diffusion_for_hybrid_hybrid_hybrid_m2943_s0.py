# DARWIN HAMMER — match 2943, survivor 0
# gen: 7
# parent_a: hybrid_diffusion_forcing_hybrid_hybrid_minimu_m918_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1315_s2.py (gen6)
# born: 2026-05-29T23:46:48Z

"""Hybrid Diffusion‑Forcing + Associative‑Memory Model

Parents
-------
* **Parent A** – *hybrid_diffusion_forcing_hybrid_hybrid_minimu_m918_s2.py*  
  Implements a diffusion‑forcing loss where an epistemic `CertaintyFlag` is mapped to a
  noise schedule `ᾱ_t`. The schedule controls the variance of Gaussian noise added to each
  token of a sequence.

* **Parent B** – *hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1315_s2.py*  
  Provides a sheaf‑based representation, a dense associative memory (DAM) whose energy
  `E(x) = -β‖x‑P‖²` measures similarity to stored patterns, and a linear random
  transformation (`TTT`) that can be interpreted as a learned embedding.

Mathematical Bridge
-------------------
The bridge is the **certainty‑driven scaling** of both the diffusion noise and the
associative‑memory energy.  A `CertaintyFlag` supplies a scalar confidence `c∈[0,1]`
(`confidence_bps/10000`).  We use `c` to:

1. **Modulate the diffusion schedule** – higher certainty yields a slower diffusion
   (smaller noise) via `ᾱ_t = (1‑c)·schedule(t) + c`.
2. **Weight the associative‑memory energy** – the contribution of the DAM term to the
   total loss is multiplied by `c`, reflecting that with higher epistemic confidence we
   trust the memory reconstruction more.

The resulting hybrid loss combines a diffusion‑forcing term `L_DF` and a DAM term `L_DAM`:


L_total = Σ_t w_t · ‖x_t – x̂_t‖²  +  c · (‑β)·‖x̂ – P‖²


where `w_t = 1‑ᾱ_t` are schedule‑derived weights, `x̂_t` is the denoised estimate at
step `t`, and `P` are the stored patterns.  This unified formulation preserves the core
topologies of both parents while enabling epistemic certainty to steer the joint
optimization.

The module below implements the bridge and provides three public functions that
demonstrate the hybrid operation.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Tuple, Dict, List

# ----------------------------------------------------------------------
# Parent‑A building blocks
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

@dataclass(frozen=True)
class CertaintyFlag:
    """Epistemic certainty descriptor."""
    label: str
    confidence_bps: int                # basis points, 0 … 10000 → confidence ∈ [0,1]
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")

    @property
    def confidence(self) -> float:
        """Return confidence as a float in [0,1]."""
        return self.confidence_bps / 10000.0


def _cosine_schedule(T: int) -> np.ndarray:
    """Nichol & Dhariwal cosine schedule ᾱ_t for t = 0…T."""
    steps = np.arange(T + 1, dtype=np.float64)
    f = np.cos(((steps / T) + 0.008) / 1.008 * math.pi / 2) ** 2
    return np.clip(f, a_min=1e-8, a_max=1.0)


def _linear_schedule(T: int) -> np.ndarray:
    """Ho et al. linear schedule ᾱ_t for t = 0…T."""
    return np.linspace(1.0, 0.0, T + 1)


def confidence_to_noise_schedule(T: int,
                                 certainty_flag: CertaintyFlag,
                                 schedule: str = "cosine") -> np.ndarray:
    """
    Map a CertaintyFlag to a diffusion noise schedule ᾱ_t ∈ (0,1].

    The schedule is first generated (cosine or linear) and then blended with the
    epistemic confidence `c`:

        ᾱ_t = (1‑c)·schedule_t + c

    Higher confidence pushes the schedule towards 1 (no noise).
    """
    if schedule == "cosine":
        base = _cosine_schedule(T)
    elif schedule == "linear":
        base = _linear_schedule(T)
    else:
        raise ValueError(f"unknown schedule {schedule!r}")

    c = certainty_flag.confidence
    blended = (1.0 - c) * base + c
    # Ensure monotonicity and bounds
    blended = np.clip(blended, a_min=1e-8, a_max=1.0)
    return blended


def aggregate_certainty(flags: Iterable[CertaintyFlag]) -> float:
    """
    Produce a scalar summarising overall epistemic certainty of a collection
    of CertaintyFlag objects.  Simple arithmetic mean of the confidence values.
    """
    confidences = [f.confidence for f in flags]
    if not confidences:
        return 0.0
    return float(np.mean(confidences))


# ----------------------------------------------------------------------
# Parent‑B building blocks
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


class Sheaf:
    """Simple sheaf storing a vector per node."""
    def __init__(self, node_dims: dict, edges: list):
        self.node_dims = node_dims
        self.edges = edges
        self.sections: Dict[Any, np.ndarray] = {}

    def set_section(self, node: Any, value: np.ndarray) -> None:
        self.sections[node] = value

    def get_section(self, node: Any) -> np.ndarray:
        return self.sections.get(node)


class DenseAssociativeMemory:
    """Energy‑based associative memory."""
    def __init__(self, patterns: np.ndarray, beta: float = 1.0):
        """
        Parameters
        ----------
        patterns : (P, D) array of stored patterns.
        beta : scaling factor for the energy.
        """
        self.patterns = patterns  # shape (num_patterns, dim)
        self.beta = beta

    def energy(self, input_vector: np.ndarray) -> float:
        """
        Compute the energy -β‖x‑P‖² summed over all stored patterns.
        Lower (more negative) energy means higher similarity.
        """
        diff = input_vector - self.patterns  # broadcast (P, D)
        sq = np.square(diff)
        return -self.beta * np.sum(sq)


class TTT:
    """Random linear transform (Tensor‑Tensor‑Tensor)."""
    def __init__(self, d_in: int, d_out: int = None, scale: float = 0.01, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.d_in = d_in
        self.d_out = d_out if d_out is not None else d_in
        self.scale = scale

    def transform(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Apply a random linear map W·x where W ~ N(0, scale²).
        """
        if input_vector.shape[-1] != self.d_in:
            raise ValueError("input dimension mismatch")
        W = self.rng.standard_normal((self.d_out, self.d_in)) * self.scale
        return W @ input_vector


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def hybrid_diffusion_forcing(sequence: np.ndarray,
                             certainty_flag: CertaintyFlag,
                             T: int = 1000,
                             schedule: str = "cosine",
                             rng: np.random.Generator = None) -> Tuple[np.ndarray, float]:
    """
    Perform a single forward diffusion pass on `sequence` using a certainty‑aware schedule.
    Returns the noisy sequence and the diffusion‑forcing loss (MSE weighted by schedule).

    Parameters
    ----------
    sequence : (N, D) array – N tokens, D dimensional embeddings.
    certainty_flag : CertaintyFlag governing the schedule.
    T : number of diffusion steps (default 1000).
    schedule : "cosine" or "linear".
    rng : optional NumPy Generator; if None, a new default_rng is created.

    Returns
    -------
    noisy_seq : (N, D) array after adding Gaussian noise.
    loss : scalar diffusion‑forcing loss.
    """
    if rng is None:
        rng = np.random.default_rng()
    alpha_bar = confidence_to_noise_schedule(T, certainty_flag, schedule)  # (T+1,)
    # Use the final timestep as the noise magnitude for the whole sequence
    sigma = math.sqrt(1.0 - alpha_bar[-1])
    noise = rng.normal(loc=0.0, scale=sigma, size=sequence.shape)
    noisy_seq = sequence + noise

    # Weighted MSE loss
    weights = 1.0 - alpha_bar  # (T+1,)
    weight = float(weights[-1])  # scalar weight for the final step
    loss = weight * np.mean(np.square(noise))
    return noisy_seq, loss


def compute_associative_energy(vector: np.ndarray,
                               dam: DenseAssociativeMemory) -> float:
    """
    Compute the associative memory energy for a single vector.
    """
    return dam.energy(vector)


def hybrid_total_loss(sequence: np.ndarray,
                      certainty_flag: CertaintyFlag,
                      dam: DenseAssociativeMemory,
                      ttt: TTT,
                      T: int = 1000,
                      schedule: str = "cosine",
                      rng: np.random.Generator = None) -> float:
    """
    End‑to‑end hybrid loss combining diffusion forcing and associative memory.

    Steps
    -----
    1. Diffusion‑forcing pass → noisy_seq, L_DF.
    2. Linear transform via TTT → transformed_seq.
    3. Energy of transformed_seq under DAM → E_DAM (negative value).
    4. Total loss = L_DF  -  c·E_DAM   (since E_DAM ≤ 0, subtraction rewards similarity).

    The epistemic confidence `c` scales the contribution of the DAM term.
    """
    if rng is None:
        rng = np.random.default_rng()
    # 1. Diffusion forcing
    noisy_seq, loss_df = hybrid_diffusion_forcing(sequence,
                                                  certainty_flag,
                                                  T=T,
                                                  schedule=schedule,
                                                  rng=rng)

    # 2. Apply TTT transform token‑wise and average to a single representation
    transformed = np.vstack([ttt.transform(tok) for tok in noisy_seq])
    # Collapse to a single vector (e.g., mean embedding)
    rep = transformed.mean(axis=0)

    # 3. Associative memory energy
    energy = compute_associative_energy(rep, dam)  # negative or zero

    # 4. Combine
    c = certainty_flag.confidence
    total = loss_df - c * energy  # energy is negative → term adds to loss when similarity is high
    return float(total)


# ----------------------------------------------------------------------
# Auxiliary morphology utilities (reused from Parent B)
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Return a dimensionless measure of how spherical an object is."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    """Return a dimensionless measure of how flat an object is."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a dummy certainty flag
    flag = CertaintyFlag(
        label="FACT",
        confidence_bps=8000,          # 0.8 confidence
        authority_class="expert",
        rationale="test case"
    )

    # Random sequence: 10 tokens, 64‑dim embeddings
    rng = np.random.default_rng(42)
    seq = rng.normal(size=(10, 64))

    # Dummy patterns for the associative memory (5 patterns, 64‑dim)
    patterns = rng.normal(size=(5, 64))
    dam = DenseAssociativeMemory(patterns=patterns, beta=0.5)

    # TTT transformer (64 → 64)
    ttt = TTT(d_in=64, scale=0.02, seed=123)

    # Compute hybrid loss
    loss = hybrid_total_loss(sequence=seq,
                             certainty_flag=flag,
                             dam=dam,
                             ttt=ttt,
                             T=500,
                             schedule="cosine",
                             rng=rng)

    print(f"Hybrid total loss: {loss:.6f}")

    # Demonstrate morphology utilities (unrelated but included for completeness)
    sph = sphericity_index(2.0, 2.0, 2.0)
    flat = flatness_index(2.0, 2.0, 1.0)
    print(f"Sphericity index: {sph:.4f}, Flatness index: {flat:.4f}")