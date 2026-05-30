# DARWIN HAMMER — match 5409, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_caputo_hybrid_nlms_omni_cha_m725_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s2.py (gen3)
# born: 2026-05-30T00:01:52Z

import math
import sys
from datetime import datetime
from collections import deque
import numpy as np

# -------------------- Clifford Algebra Utilities --------------------

def _blade_sign(indices):
    """Return sorted blade and its sign after reordering.
    Duplicated indices cancel (e_i^2 = 1) and change sign accordingly.
    """
    lst = list(indices)
    sign = 1
    i = 0
    while i < len(lst):
        j = i + 1
        while j < len(lst):
            if lst[i] > lst[j]:
                lst[i], lst[j] = lst[j], lst[i]
                sign *= -1
            elif lst[i] == lst[j]:
                # cancel duplicate basis vector
                lst.pop(j)
                lst.pop(i)
                sign *= 1  # e_i*e_i = 1, no sign change
                i -= 1
                break
            j += 1
        i += 1
    return frozenset(sorted(lst)), sign

def _multiply_blades(blade_a, blade_b):
    """Geometric product of two basis blades."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return result, sign

# -------------------- Multivector Class --------------------

class Multivector:
    """Simple multivector in Cl(n,0). Keys are frozenset of basis indices."""

    def __init__(self, components=None, n=0):
        self.n = int(n)
        self.components = {}
        if components:
            for blade, val in components.items():
                if abs(val) > 1e-15:
                    self.components[frozenset(blade)] = float(val)

    def copy(self):
        return Multivector(self.components.copy(), self.n)

    # ----- Algebraic operations -----
    def __add__(self, other):
        res = self.copy()
        for b, v in other.components.items():
            res.components[b] = res.components.get(b, 0.0) + v
            if abs(res.components[b]) < 1e-15:
                del res.components[b]
        return res

    def __sub__(self, other):
        res = self.copy()
        for b, v in other.components.items():
            res.components[b] = res.components.get(b, 0.0) - v
            if abs(res.components[b]) < 1e-15:
                del res.components[b]
        return res

    def __rmul__(self, scalar):
        return self.scalar_mul(scalar)

    def scalar_mul(self, scalar):
        if abs(scalar) < 1e-15:
            return Multivector(n=self.n)
        return Multivector({b: v * scalar for b, v in self.components.items()}, self.n)

    def geometric_product(self, other):
        """Full geometric product (Clifford multiplication)."""
        result = Multivector(n=self.n)
        for ba, va in self.components.items():
            for bb, vb in other.components.items():
                blade, sign = _multiply_blades(ba, bb)
                result.components[blade] = result.components.get(blade, 0.0) + sign * va * vb
        # prune near‑zero entries
        result.components = {b: v for b, v in result.components.items() if abs(v) > 1e-15}
        return result

    def inner_product(self, other):
        """Scalar (grade‑0) part of the geometric product."""
        return self.geometric_product(other).components.get(frozenset(), 0.0)

    # ----- Utility -----
    def norm_sq(self):
        """Squared Euclidean norm (sum of squares of all grades)."""
        return sum(v * v for v in self.components.values())

    def __repr__(self):
        return f"Multivector({self.components})"

# -------------------- Fractional Derivative (Grünwald‑Letnikov) --------------------

def grunwald_letnikov_coeffs(alpha, length):
    """Return coefficients for the Grünwald‑Letnikov approximation of order α."""
    coeffs = np.zeros(length, dtype=float)
    coeffs[0] = 1.0
    for k in range(1, length):
        coeffs[k] = coeffs[k - 1] * (-(alpha - k + 1) / k)
    return coeffs

def fractional_derivative_history(history, alpha):
    """Compute D^α w_t using stored history (list of Multivector)."""
    if not history:
        return Multivector(n=history[0].n if history else 0)
    coeffs = grunwald_letnikov_coeffs(alpha, len(history))
    result = Multivector(n=history[0].n)
    for c, w in zip(coeffs, reversed(history)):
        result = result + (c * w)
    factor = 1.0 / math.gamma(1 - alpha)
    return factor * result

# -------------------- Hybrid Model --------------------

class HybridModel:
    """
    Hybrid NLMS with Caputo‑like fractional memory and geometric‑product weight update.
    """

    def __init__(self, alpha, n, mu=0.1, eps=1e-6, max_history=50):
        """
        alpha : fractional order (0 < α < 1)
        n     : dimension of the underlying vector space
        mu    : base step size
        eps   : regularisation term to avoid division by zero
        max_history : maximum length of stored weight history
        """
        self.alpha = float(alpha)
        self.n = int(n)
        self.mu = float(mu)
        self.eps = float(eps)
        self.max_history = int(max_history)

        # initialise weight as scalar 0 multivector
        self.w = Multivector({frozenset(): 0.0}, n=self.n)

        # store past weights for fractional derivative
        self._weight_history = deque(maxlen=self.max_history)

    # ----- Liquid time constant (day‑of‑week modulation) -----
    @staticmethod
    def liquid_time_constant(day_of_week):
        """
        Map day_of_week ∈ {1,…,7} to a positive scalar.
        Use a smooth sinusoidal modulation centred at 1.0.
        """
        if not 1 <= day_of_week <= 7:
            raise ValueError("day_of_week must be in 1..7")
        # shift so that Monday (1) gives slightly lower constant, Sunday (7) higher
        theta = 2 * math.pi * (day_of_week - 1) / 7.0
        return 1.0 + 0.2 * math.sin(theta)

    # ----- Core update routine -----
    def step(self, x_vec, d_scalar, day_of_week):
        """
        Perform one adaptation step.
        x_vec : iterable of length n (real numbers) – input vector
        d_scalar : desired scalar output
        day_of_week : integer 1..7
        Returns updated scalar output estimate.
        """
        # Convert input to a multivector (grade‑1 vector)
        x = Multivector({frozenset([i]): float(val) for i, val in enumerate(x_vec)}, n=self.n)

        # Estimate output using scalar part of w·x
        y = self.w.inner_product(x)

        # Compute error
        e = float(d_scalar) - y

        # Normalisation term (||x||^2 + eps)
        norm_sq = x.norm_sq() + self.eps

        # Fractional derivative of the weight (memory term)
        D_alpha_w = fractional_derivative_history(list(self._weight_history), self.alpha)

        # Liquid time constant modulation
        tau = self.liquid_time_constant(day_of_week)

        # Weight update rule:
        # w_{t+1} = w_t + (mu * tau) * (e * x) / norm_sq  -  (mu * tau) * D^α w_t
        correction = (self.mu * tau) * (e * x)  # e is scalar, geometric product distributes
        memory_term = (self.mu * tau) * D_alpha_w

        self.w = self.w + (correction.scalar_mul(1.0 / norm_sq)) - memory_term

        # Store current weight for future fractional derivative computation
        self._weight_history.append(self.w.copy())

        return y

# -------------------- Convenience Wrapper --------------------

def hybrid_nlms_caputo_geometric_product(alpha, n, inputs, targets, days):
    """
    Run the hybrid algorithm over a sequence.
    inputs : list of input vectors (each iterable of length n)
    targets: list of desired scalar outputs
    days   : list of day_of_week integers (same length as inputs)
    Returns list of output estimates.
    """
    if not (len(inputs) == len(targets) == len(days)):
        raise ValueError("inputs, targets and days must have equal length")
    model = HybridModel(alpha=alpha, n=n)
    outputs = []
    for x, d, day in zip(inputs, targets, days):
        y = model.step(x, d, day)
        outputs.append(y)
    return outputs

# -------------------- Unit Tests --------------------

if __name__ == "__main__":
    import unittest

    class TestHybridAlgorithm(unittest.TestCase):
        def test_basic_convergence(self):
            alpha = 0.4
            n = 3
            # Simple linear system: output = sum(x_i)
            inputs = [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.5, 0.5, 0.0],
            ]
            targets = [1.0, 1.0, 1.0, 3.0, 1.0]
            days = [1, 2, 3, 4, 5]  # Monday‑Friday
            outputs = hybrid_nlms_caputo_geometric_product(alpha, n, inputs, targets, days)
            # After a few iterations the estimate for the last sample should be close to 1.0
            self.assertAlmostEqual(outputs[-1], 1.0, delta=0.2)

        def test_liquid_time_constant_range(self):
            for dow in range(1, 8):
                tc = HybridModel.liquid_time_constant(dow)
                self.assertTrue(0.8 <= tc <= 1.2)

        def test_fractional_derivative_coeffs(self):
            coeffs = grunwald_letnikov_coeffs(0.5, 5)
            # First coefficient must be 1, others should follow known pattern
            self.assertAlmostEqual(coeffs[0], 1.0)
            self.assertTrue(all(abs(c) < 1.0 for c in coeffs[1:]))

    unittest.main(argv=[sys.argv[0]], exit=False)