# DARWIN HAMMER — match 1707, survivor 0
# gen: 3
# parent_a: hybrid_geometric_product_hybrid_model_vram_sc_m22_s1.py (gen2)
# parent_b: ttt_linear.py (gen0)
# born: 2026-05-29T23:38:23Z

"""
Fusion of hybrid_geometric_product_hybrid_model_vram_sc_m22_s1.py and ttt_linear.py.

The mathematical bridge between the two parents is the integration of the 
Clifford geometric product into the TTT-Linear model's update rule. By 
representing the weight matrix W as a multivector and using the geometric 
product for updates, we can leverage the properties of Clifford algebras 
to optimize the model's performance while minimizing memory usage.

The geometric product combines the inner and outer products, allowing us to 
update the weight matrix W while preserving its geometric structure. This 
approach enables the model to adapt to changing memory requirements while 
maintaining a stable and efficient update rule.

This fusion combines the governing equations of both parents, allowing for 
a novel hybrid algorithm that adapts to changing memory requirements.
"""

import numpy as np
import math
import random
import sys
import pathlib

def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble-sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                lst.pop(j)
                lst.pop(j)  # was j+1, now at j after pop
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, components, n):
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def grade(self, k):
        """Return a new Multivector keeping only grade-k blades."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items()
             if len(blade) == k},
            self.n,
        )

    def scalar_part(self):
        """Return the scalar (grade-0) coefficient."""
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other):
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector({k: v for k, v in result.items() if v != 0.0}, self.n)

def init_ttt(d_in, d_out=None, scale=0.01, seed=0):
    """Initialize W shape (d_out, d_in).

    d_out defaults to d_in. Small random initialization; scale controls
    the initial magnitude so the first few gradient steps are interpretable.
    """
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_geometric_product_loss(W, x, target=None):
    """Self-supervised loss for TTT with geometric product update.

    If target is None, use reconstruction loss: ||W x - x||^2.
    x shape: (d_in,). Returns scalar float.

    The reconstruction objective treats the identity mapping as the target.
    The weight matrix learns to be a good compressor of the input distribution
    seen so far — if W@x ≈ x holds, W has absorbed enough structure.
    """
    W_multivector = Multivector(W, 2)
    x_multivector = Multivector({frozenset([0]): x}, 2)
    W_x_multivector = Multivector()
    for blade, coef in W_multivector.components.items():
        for blade_x, coef_x in x_multivector.components.items():
            W_x_multivector.components[tuple(sorted(blade + blade_x))] = coef * coef_x
    W_x_multivector = W_x_multivector.grade(1)
    loss = np.linalg.norm(W_x_multivector.scalar_part() - target)
    return loss

def ttt_geometric_product_step(W, x, eta=0.01):
    """Update W using geometric product and gradient descent.

    W shape: (d_out, d_in).
    x shape: (d_in,). Returns updated W.
    """
    W_multivector = Multivector(W, 2)
    x_multivector = Multivector({frozenset([0]): x}, 2)
    W_x_multivector = Multivector()
    for blade, coef in W_multivector.components.items():
        for blade_x, coef_x in x_multivector.components.items():
            W_x_multivector.components[tuple(sorted(blade + blade_x))] = coef * coef_x
    W_x_multivector = W_x_multivector.grade(1)
    grad_W_loss = -2 * x_multivector.scalar_part()
    W_new = W + eta * grad_W_loss
    return W_new

def ttt_geometric_product_forward(W, x):
    """Forward pass using geometric product update.

    W shape: (d_out, d_in).
    x shape: (d_in,). Returns output.
    """
    W_multivector = Multivector(W, 2)
    x_multivector = Multivector({frozenset([0]): x}, 2)
    W_x_multivector = Multivector()
    for blade, coef in W_multivector.components.items():
        for blade_x, coef_x in x_multivector.components.items():
            W_x_multivector.components[tuple(sorted(blade + blade_x))] = coef * coef_x
    W_x_multivector = W_x_multivector.grade(1)
    output = W_x_multivector.scalar_part()
    return output

if __name__ == "__main__":
    d_in, d_out = 10, 20
    W = init_ttt(d_in, d_out)
    x = np.random.rand(d_in)
    target = np.random.rand(d_in)
    loss = ttt_geometric_product_loss(W, x, target)
    W_new = ttt_geometric_product_step(W, x)
    output = ttt_geometric_product_forward(W_new, x)
    print("Loss:", loss)
    print("Updated W:", W_new)
    print("Output:", output)