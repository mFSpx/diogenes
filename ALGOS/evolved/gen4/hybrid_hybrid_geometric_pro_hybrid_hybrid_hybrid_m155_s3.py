# DARWIN HAMMER — match 155, survivor 3
# gen: 4
# parent_a: hybrid_geometric_product_hybrid_model_vram_sc_m22_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_ternar_ttt_linear_m5_s4.py (gen3)
# born: 2026-05-29T23:27:21Z

import numpy as np
from typing import Dict, FrozenSet, Tuple, Any

# ----------------------------------------------------------------------
# Blade utilities – robust handling of index cancellation and sign
# ----------------------------------------------------------------------
def _blade_sign(indices: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    """
    Sort indices and cancel pairs of equal indices (which square to +1).
    Returns the sorted tuple of remaining indices and the sign (+1 / -1)
    induced by the swaps needed to bring the original list into the sorted order.
    """
    # Count occurrences
    counts: Dict[int, int] = {}
    for i in indices:
        counts[i] = counts.get(i, 0) + 1

    # Cancel even occurrences
    remaining = [i for i, c in counts.items() if c % 2 == 1]
    # The remaining list may contain duplicates (odd count >1) – keep one copy per odd count
    # Build the list respecting original multiplicities (odd only)
    cleaned = []
    for i in indices:
        if counts[i] % 2 == 1:
            cleaned.append(i)
            counts[i] = 0  # ensure we keep only one copy
    # Sort while tracking sign via bubble‑sort swaps
    lst = list(cleaned)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign = -sign
    return tuple(lst), sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """
    Clifford (geometric) product of two basis blades.
    Returns (resulting_blade, sign).
    """
    combined = tuple(list(blade_a) + list(blade_b))
    sorted_idxs, sign = _blade_sign(combined)
    return frozenset(sorted_idxs), sign


# ----------------------------------------------------------------------
# Multivector helpers
# ----------------------------------------------------------------------
def vector_to_mv(v: np.ndarray) -> Dict[FrozenSet[int], float]:
    """Convert a 1‑D array into a multivector containing only grade‑1 blades."""
    mv: Dict[FrozenSet[int], float] = {}
    for i, coeff in enumerate(v):
        if coeff != 0.0:
            mv[frozenset({i})] = float(coeff)
    return mv


def mv_to_vector(mv: Dict[FrozenSet[int], float], dim: int) -> np.ndarray:
    """Extract the vector (grade‑1) part of a multivector."""
    vec = np.zeros(dim, dtype=float)
    for blade, coeff in mv.items():
        if len(blade) == 1:
            idx = next(iter(blade))
            vec[idx] = coeff
    return vec


def geometric_product(a: Dict[FrozenSet[int], float],
                      b: Dict[FrozenSet[int], float]) -> Dict[FrozenSet[int], float]:
    """Full Clifford (geometric) product of two multivectors."""
    result: Dict[FrozenSet[int], float] = {}
    for blade_a, coef_a in a.items():
        for blade_b, coef_b in b.items():
            blade_out, sign = _multiply_blades(blade_a, blade_b)
            result[blade_out] = result.get(blade_out, 0.0) + coef_a * coef_b * sign
    return result


# ----------------------------------------------------------------------
# SSIM utilities (1‑D version) and its analytic gradient
# ----------------------------------------------------------------------
def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural Similarity Index Measure for 1‑D signals."""
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x, ddof=0)
    sigma_y = np.std(y, ddof=0)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    return numerator / denominator


def ssim_grad(pred: np.ndarray, target: np.ndarray,
              dynamic_range: float = 255.0,
              k1: float = 0.01,
              k2: float = 0.03) -> np.ndarray:
    """
    Analytic gradient of SSIM w.r.t. the prediction vector `pred`.
    Returns d(SSIM)/d(pred) of shape (len(pred),).
    """
    N = pred.size
    mu_p = np.mean(pred)
    mu_t = np.mean(target)

    sigma_p2 = np.mean((pred - mu_p) ** 2)
    sigma_t2 = np.mean((target - mu_t) ** 2)
    sigma_pt = np.mean((pred - mu_p) * (target - mu_t))

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    # Pre‑compute repeated terms
    A = 2 * mu_p * mu_t + c1
    B = 2 * sigma_pt + c2
    C = mu_p ** 2 + mu_t ** 2 + c1
    D = sigma_p2 + sigma_t2 + c2

    ssim_val = (A * B) / (C * D)

    # Derivatives of the components
    dmu_p = 1.0 / N
    dsigma_p2 = 2.0 * (pred - mu_p) / N - 2.0 * (np.mean(pred - mu_p) * dmu_p)  # simplifies to 2*(pred-mu_p)/N
    dsigma_pt = (target - mu_t) / N - (np.mean(target - mu_t) * dmu_p)

    # Chain rule
    dA = 2 * mu_t * dmu_p
    dB = 2 * dsigma_pt
    dC = 2 * mu_p * dmu_p
    dD = dsigma_p2

    # Gradient of the quotient (A*B)/(C*D)
    grad = (dA * B + A * dB) * (C * D) - (A * B) * (dC * D + C * dD)
    grad /= (C * D) ** 2
    return grad


# ----------------------------------------------------------------------
# Hybrid model – deeper integration of geometric algebra
# ----------------------------------------------------------------------
class GTTHybrid:
    """
    Geometric‑TT‑Hybrid model.
    The weight is stored as a multivector (dictionary) that can act on input vectors
    via the full geometric product.  The training step combines a reconstruction loss
    with a SSIM‑based perceptual loss, using the exact analytic gradient of SSIM.
    """

    def __init__(self,
                 dim: int,
                 scale: float = 0.01,
                 seed: int = 0,
                 alpha: float = 0.1,
                 beta: float = 0.1,
                 lr: float = 1e-3):
        """
        Parameters
        ----------
        dim : int
            Dimensionality of the input / output vectors.
        scale : float
            Standard‑deviation of the initial random coefficients.
        seed : int
            Random seed for reproducibility.
        alpha, beta : float
            Weighting of reconstruction and SSIM losses respectively.
        lr : float
            Base learning rate.
        """
        rng = np.random.default_rng(seed)
        # Initialise a random multivector: scalar part zero, vector part random
        self.dim = dim
        self.W_mv: Dict[FrozenSet[int], float] = {frozenset({i}): float(rng.standard_normal() * scale)
                                                  for i in range(dim)}
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lr = float(lr)

    # ------------------------------------------------------------------
    # Forward pass using geometric product
    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply the current multivector weight to input vector x."""
        x_mv = vector_to_mv(x)
        out_mv = geometric_product(self.W_mv, x_mv)
        return mv_to_vector(out_mv, self.dim)

    # ------------------------------------------------------------------
    # Loss computation (reconstruction + SSIM)
    # ------------------------------------------------------------------
    def loss(self, pred: np.ndarray, target: np.ndarray) -> Tuple[float, float, float]:
        """Return total loss and its two components."""
        rec = np.mean((pred - target) ** 2)                     # MSE reconstruction
        ssim_val = ssim(pred, target)
        ssim_loss = 1.0 - ssim_val
        total = self.alpha * rec + self.beta * ssim_loss
        return total, rec, ssim_loss

    # ------------------------------------------------------------------
    # Training step – analytic gradients for both terms
    # ------------------------------------------------------------------
    def step(self, x: np.ndarray, target: np.ndarray = None) -> float:
        """
        Perform one adaptation step.
        If `target` is None, the input itself is used as the self‑supervised target.
        Returns the total hybrid loss after the update.
        """
        if target is None:
            target = x

        # Forward pass
        pred = self.forward(x)

        # ----- Reconstruction gradient (MSE) -----
        residual = pred - target                     # shape (dim,)
        grad_rec_W = 2.0 * np.outer(residual, x)      # dMSE/dW (matrix form)

        # ----- SSIM gradient -----
        grad_ssim_pred = ssim_grad(pred, target)     # dSSIM/dpred
        # Chain rule: d(1‑SSIM)/dW = -grad_ssim_pred @ x.T
        grad_ssim_W = -np.outer(grad_ssim_pred, x)

        # Combine gradients with weighting
        grad_W = self.alpha * grad_rec_W + self.beta * grad_ssim_W

        # ------------------------------------------------------------------
        # Map matrix gradient back onto the multivector representation.
        # Only the vector (grade‑1) blades are updated; scalar part stays zero.
        # ------------------------------------------------------------------
        for i in range(self.dim):
            blade = frozenset({i})
            # The column i of grad_W corresponds to derivative w.r.t. coefficient of blade i
            # because forward = W_mv ⋅ x (geometric product reduces to linear map on vectors)
            self.W_mv[blade] -= self.lr * grad_W[i, i]  # diagonal captures self‑interaction
            # Off‑diagonal contributions affect the same blade via the symmetric part;
            # we accumulate them for better approximation.
            self.W_mv[blade] -= self.lr * (grad_W[i, :].sum() - grad_W[i, i]) / (self.dim - 1 + 1e-12)

        total_loss, rec, ssim_l = self.loss(pred, target)
        return total_loss

# ----------------------------------------------------------------------
# Simple sanity test (executed when run as script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    dim = 8
    model = GTTHybrid(dim=dim, scale=0.05, seed=1, alpha=0.7, beta=0.3, lr=5e-4)

    # Random input
    x = np.random.rand(dim).astype(float)

    # Perform a few adaptation steps
    for epoch in range(5):
        loss_val = model.step(x)
        print(f"Epoch {epoch + 1}: loss = {loss_val:.6f}")

    # Final prediction
    y_hat = model.forward(x)
    print("Final prediction:", y_hat)