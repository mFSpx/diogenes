# DARWIN HAMMER — match 683, survivor 5
# gen: 5
# parent_a: physarum_network.py (gen0)
# parent_b: hybrid_hybrid_hdc_hybrid_hy_hybrid_hybrid_rbf_su_m182_s0.py (gen4)
# born: 2026-05-29T23:30:32Z

import numpy as np
import hashlib
from typing import List, Tuple, Sequence, Callable, Optional


# ----------------------------------------------------------------------
# Utility functions – now vectorised with NumPy for speed and clarity
# ----------------------------------------------------------------------
def random_vector(dim: int = 10_000, seed: Optional[int] = None) -> np.ndarray:
    """Generate a bipolar (+1 / -1) random vector."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=dim) * 2 - 1  # maps {0,1} -> {-1,+1}


def symbol_vector(symbol: str, dim: int = 10_000) -> np.ndarray:
    """Deterministic bipolar vector derived from a hash of *symbol*."""
    h = hashlib.sha256(symbol.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(h, "big")
    return random_vector(dim, seed)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element‑wise binding (Hadamard product)."""
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    return a * b


def bundle(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Superposition (majority vote) of bipolar vectors."""
    if not vectors:
        raise ValueError("at least one vector is required")
    stacked = np.stack(vectors)
    sums = stacked.sum(axis=0)
    return np.where(sums >= 0, 1, -1)


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two real‑valued vectors."""
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    return float(np.linalg.norm(a - b))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian kernel."""
    return np.exp(-((epsilon * r) ** 2))


# ----------------------------------------------------------------------
# Physarum‑style flux and conductance dynamics
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float,
         pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Compute flux on an edge using Ohm‑like law."""
    if edge_length <= 0:
        raise ValueError("edge_length must be positive")
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float,
                       dt: float = 1.0, gain: float = 1.0,
                       decay: float = 0.05) -> float:
    """Physarum conductance adaptation."""
    if dt < 0 or decay < 0:
        raise ValueError("dt and decay must be non‑negative")
    new_c = conductance + dt * (gain * abs(q) - decay * conductance)
    return max(0.0, new_c)


# ----------------------------------------------------------------------
# Radial‑Basis‑Function surrogate with online weight adaptation
# ----------------------------------------------------------------------
class RBFSurrogate:
    def __init__(self,
                 centers: List[Tuple[float, ...]],
                 weights: List[float],
                 epsilon: float = 1.0,
                 lr: float = 0.01):
        if len(centers) != len(weights):
            raise ValueError("centers and weights must have the same length")
        self.centers = [np.asarray(c, dtype=float) for c in centers]
        self.weights = np.asarray(weights, dtype=float)
        self.epsilon = epsilon
        self.lr = lr  # learning rate for online updates

    def _kernel_row(self, x: np.ndarray) -> np.ndarray:
        """Compute kernel values between *x* and all centers."""
        dists = np.array([euclidean(x, c) for c in self.centers])
        return np.exp(-((self.epsilon * dists) ** 2))

    def predict(self, x: Sequence[float]) -> float:
        """RBF prediction for input *x*."""
        x_arr = np.asarray(x, dtype=float)
        k = self._kernel_row(x_arr)
        return float(k @ self.weights)

    def adapt(self,
              x: Sequence[float],
              target: float,
              confidence: float = 1.0) -> None:
        """
        Simple stochastic gradient step on squared error,
        scaled by *confidence* (0‑1). Higher confidence → larger step.
        """
        x_arr = np.asarray(x, dtype=float)
        k = self._kernel_row(x_arr)
        pred = float(k @ self.weights)
        error = target - pred
        # Gradient w.r.t. weights is -2 * error * k
        grad = -2.0 * error * k
        self.weights -= self.lr * confidence * grad


# ----------------------------------------------------------------------
# Deeply integrated hybrid model
# ----------------------------------------------------------------------
class HybridFluxRBF:
    """
    A unified model that couples physarum flux dynamics with an RBF surrogate.
    The magnitude of flux modulates a confidence term that scales both
    weight adaptation and the influence of the bound vector on the surrogate.
    """

    def __init__(self,
                 dim: int,
                 centers: List[Tuple[float, ...]],
                 weights: List[float],
                 epsilon: float = 1.0,
                 conductance: float = 1.0,
                 edge_length: float = 1.0,
                 pressure_a: float = 1.0,
                 pressure_b: float = 0.0,
                 dt: float = 1.0,
                 gain: float = 1.0,
                 decay: float = 0.05,
                 lr: float = 0.01):
        self.dim = dim
        self.conductance = conductance
        self.edge_length = edge_length
        self.pressure_a = pressure_a
        self.pressure_b = pressure_b
        self.dt = dt
        self.gain = gain
        self.decay = decay

        # internal state vectors
        self.vec_a = random_vector(dim)
        self.vec_b = random_vector(dim)

        # surrogate
        self.rbf = RBFSurrogate(centers, weights, epsilon, lr)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def _confidence_from_flux(self, q: float) -> float:
        """Map flux magnitude to a confidence in [0,1] using a sigmoid."""
        # scale factor chosen empirically; can be exposed as a parameter
        scale = 5.0
        return 1.0 / (1.0 + np.exp(-scale * (abs(q) - 0.5)))

    def step(self,
             external_target: Optional[float] = None,
             external_input: Optional[Sequence[float]] = None) -> Tuple[float, np.ndarray, float]:
        """
        Perform one hybrid iteration:

        1. Compute flux and update conductance.
        2. Bind the two internal vectors.
        3. Bundle a history of bound vectors (here we reuse the current bound).
        4. Build an RBF input that concatenates the bundled vector (as float) with
           an optional external input.
        5. Predict with the surrogate, then optionally adapt it using *external_target*.

        Returns
        -------
        q : float
            Current flux value.
        bound_vector : np.ndarray
            Result of binding ``vec_a`` and ``vec_b``.
        prediction : float
            Surrogate output after this step.
        """
        # 1. Flux & conductance update
        q = flux(self.conductance,
                 self.edge_length,
                 self.pressure_a,
                 self.pressure_b)
        self.conductance = update_conductance(self.conductance,
                                              q,
                                              self.dt,
                                              self.gain,
                                              self.decay)

        # 2. Binding
        bound_vector = bind(self.vec_a, self.vec_b)

        # 3. Bundling – in a richer implementation we would keep a sliding
        #    window of past bound vectors; for simplicity we reuse the current one.
        bundled_vector = bound_vector.astype(float)  # convert to real‑valued for RBF

        # 4. Assemble RBF input
        if external_input is not None:
            rbf_input = np.concatenate([bundled_vector, np.asarray(external_input, dtype=float)])
        else:
            rbf_input = bundled_vector

        # 5. Surrogate prediction
        prediction = self.rbf.predict(rbf_input)

        # 6. Optional online adaptation
        if external_target is not None:
            confidence = self._confidence_from_flux(q)
            self.rbf.adapt(rbf_input, external_target, confidence)

        # 7. Refresh internal vectors for next iteration (simple random walk)
        self.vec_a = bind(self.vec_a, random_vector(self.dim))
        self.vec_b = bind(self.vec_b, random_vector(self.dim))

        return q, bound_vector, prediction

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def set_pressures(self, pa: float, pb: float) -> None:
        self.pressure_a = pa
        self.pressure_b = pb

    def set_edge_length(self, length: float) -> None:
        if length <= 0:
            raise ValueError("edge_length must be positive")
        self.edge_length = length

    def get_state(self) -> dict:
        """Expose a snapshot of the internal state for inspection."""
        return {
            "conductance": self.conductance,
            "pressure_a": self.pressure_a,
            "pressure_b": self.pressure_b,
            "edge_length": self.edge_length,
            "vec_a": self.vec_a.copy(),
            "vec_b": self.vec_b.copy(),
            "rbf_weights": self.rbf.weights.copy(),
        }


# ----------------------------------------------------------------------
# Simple demonstration when run as a script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a modest‑size model for quick demo
    dim_demo = 256
    centers_demo = [tuple(np.random.rand(dim_demo)) for _ in range(5)]
    weights_demo = [0.0 for _ in range(5)]

    model = HybridFluxRBF(
        dim=dim_demo,
        centers=centers_demo,
        weights=weights_demo,
        epsilon=0.8,
        conductance=1.0,
        edge_length=1.2,
        pressure_a=1.0,
        pressure_b=0.3,
        dt=0.5,
        gain=1.2,
        decay=0.02,
        lr=0.005,
    )

    # Simulate a few steps with synthetic targets
    for step in range(10):
        # synthetic external input: a low‑dim real vector
        ext_input = np.random.rand(10)
        # synthetic target that slowly grows
        target = step * 0.1
        q, bound, pred = model.step(external_target=target, external_input=ext_input)
        print(f"Step {step:02d} | flux={q:.4f} | pred={pred:.4f} | conf={model._confidence_from_flux(q):.3f}")

    # Inspect final state
    state = model.get_state()
    print("\nFinal conductance:", state["conductance"])
    print("Final RBF weights:", state["rbf_weights"])