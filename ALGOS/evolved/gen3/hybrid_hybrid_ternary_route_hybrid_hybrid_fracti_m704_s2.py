# DARWIN HAMMER — match 704, survivor 2
# gen: 3
# parent_a: hybrid_ternary_router_hybrid_minimum_cost__m36_s3.py (gen2)
# parent_b: hybrid_hybrid_fractional_hd_hybrid_hoeffding_tre_m12_s1.py (gen2)
# born: 2026-05-29T23:30:35Z

import numpy as np
import math
import json
import datetime
from typing import Iterable, Optional, Any, Tuple, Callable


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def emit_json(obj: Any) -> None:
    """Print a JSON object with deterministic key order."""
    print(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str))


def random_hv(d: int, kind: str = "real", seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a deterministic random hyper‑vector.

    Parameters
    ----------
    d : int
        Dimensionality of the vector.
    kind : {"real", "bipolar", "complex"}
        Distribution type.
    seed : Optional[int]
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        Normalised hyper‑vector (complex for ``complex`` kind).
    """
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        hv = np.exp(1j * theta)
    elif kind == "bipolar":
        hv = rng.choice(np.array([-1.0, 1.0]), size=d)
    elif kind == "real":
        hv = rng.standard_normal(d)
        hv /= np.linalg.norm(hv) + 1e-30
    else:
        raise ValueError(f"Unsupported kind {kind!r}")
    return hv


def bind(x: np.ndarray, hv: np.ndarray) -> np.ndarray:
    """Circular convolution via FFT (binding)."""
    return np.fft.ifft(np.fft.fft(x) * np.fft.fft(hv))


def unbind(z: np.ndarray, hv: np.ndarray) -> np.ndarray:
    """Approximate inverse binding using conjugate division."""
    hv_f = np.fft.fft(hv)
    mag2 = np.abs(hv_f) ** 2 + 1e-30
    inv_hv_f = np.conj(hv_f) / mag2
    return np.fft.ifft(np.fft.fft(z) * inv_hv_f)


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """
    Classic Hoeffding bound.

    Parameters
    ----------
    r : float
        Range of the bounded random variable (must be > 0).
    delta : float
        Desired failure probability (0 < delta < 1).
    n : int
        Sample size (must be > 0).

    Returns
    -------
    float
        Upper bound on the deviation with probability ``1‑delta``.
    """
    if r <= 0:
        raise ValueError("r must be positive")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1)")
    if n <= 0:
        raise ValueError("n must be positive")
    return math.sqrt((r ** 2 * math.log(1.0 / delta)) / (2.0 * n))


def gini_coefficient(values: Iterable[float]) -> float:
    """
    Compute the Gini coefficient for a non‑negative sequence.

    Returns 0 for an empty or all‑zero input.
    """
    xs = np.asarray(list(values), dtype=float)
    if xs.size == 0 or np.allclose(xs, 0):
        return 0.0
    if np.any(xs < 0):
        raise ValueError("values must be non‑negative")
    xs_sorted = np.sort(xs)
    n = xs_sorted.size
    cumulative = np.cumsum(xs_sorted)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)


class HybridHoeffdingTree:
    """
    Core implementation shared by all hybrid variants.

    The class encapsulates the mathematical bridge between the
    fractional exponent ``alpha`` and the Hoeffding bound, using the
    Gini coefficient as a *dynamic* scaling factor.
    """

    def __init__(
        self,
        alpha: float,
        r: float = 1.0,
        delta: float = 0.01,
        hv_kind: str = "real",
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        alpha : float
            Base fractional exponent (0 ≤ alpha ≤ 1 recommended).
        r : float, optional
            Range for Hoeffding bound (default 1.0).
        delta : float, optional
            Failure probability for Hoeffding bound (default 0.01).
        hv_kind : str, optional
            Kind of random hyper‑vector used for binding (default "real").
        seed : Optional[int], optional
            Seed for reproducible hyper‑vector generation.
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha should lie in [0,1] for stable scaling")
        self.alpha = alpha
        self.r = r
        self.delta = delta
        self.hv_kind = hv_kind
        self.seed = seed

    def _scaled_alpha(self, gini: float) -> float:
        """
        Map Gini ∈ [0,1] to a scaling factor that never collapses to zero.
        The linear map ``0.5 + 0.5·gini`` ensures a minimum weight of 0.5.
        """
        return self.alpha * (0.5 + 0.5 * gini)

    def _noise(self, n: int) -> np.ndarray:
        """
        Generate zero‑mean Gaussian noise whose standard deviation equals
        the Hoeffding bound for the current sample size.
        """
        bound = hoeffding_bound(self.r, self.delta, n)
        return np.random.normal(0.0, bound, size=n)

    def transform(self, values: np.ndarray) -> np.ndarray:
        """
        Apply the hybrid transformation to ``values``.

        The same random hyper‑vector is used for both binding and unbinding,
        guaranteeing a consistent algebraic relationship.
        """
        if values.ndim != 1:
            raise ValueError("values must be a one‑dimensional array")
        n = values.size

        # 1️⃣ Compute statistics
        gini = gini_coefficient(values)
        scaled_alpha = self._scaled_alpha(gini)

        # 2️⃣ Generate a *single* hyper‑vector for the whole operation
        hv = random_hv(n, kind=self.hv_kind, seed=self.seed)

        # 3️⃣ Core hybrid computation
        bound_part = bind(values, hv) * scaled_alpha
        unbound_part = unbind(values, hv) * (1.0 - scaled_alpha)
        hybrid = bound_part + unbound_part

        # 4️⃣ Add calibrated Hoeffding noise
        hybrid += self._noise(n)

        # Preserve the original dtype (real → real, complex → complex)
        if np.isrealobj(values):
            hybrid = np.real_if_close(hybrid, tol=1e-9)
        return hybrid

    # Convenience wrappers matching the original function names
    def ternary_router(self, values: np.ndarray) -> np.ndarray:
        """Alias for ``transform`` kept for backward compatibility."""
        return self.transform(values)

    def fractional(self, values: np.ndarray) -> np.ndarray:
        """Alias for ``transform`` kept for backward compatibility."""
        return self.transform(values)

    def ternary(self, values: np.ndarray) -> np.ndarray:
        """Alias for ``transform`` kept for backward compatibility."""
        return self.transform(values)


if __name__ == "__main__":
    # Demo / sanity‑check
    np.random.seed(42)
    sample = np.random.normal(0.0, 1.0, 1000)
    alpha = 0.5

    model = HybridHoeffdingTree(alpha=alpha, hv_kind="real", seed=123)

    res_router = model.ternary_router(sample)
    res_fractional = model.fractional(sample)
    res_ternary = model.ternary(sample)

    print("Hybrid Ternary Router Hoeffding Tree mean:", float(res_router.mean()))
    print("Hybrid Fractional‑Hoeffding Tree mean:", float(res_fractional.mean()))
    print("Hybrid Ternary Hoeffding Tree mean:", float(res_ternary.mean()))