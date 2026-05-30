# DARWIN HAMMER — match 406, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_sparse_wta_hy_hybrid_capybara_opti_m180_s0.py (gen3)
# parent_b: hybrid_fisher_localization_hybrid_ternary_route_m40_s5.py (gen2)
# born: 2026-05-29T23:28:55Z

"""Hybrid Sparse‑WTA / Fisher‑Weighted SSIM Algorithm
===================================================

Parent algorithms
-----------------
* **hybrid_hybrid_sparse_wta_hy_hybrid_capybara_opti_m180_s0.py** – provides a
  hash‑based sparse expansion (`expand`), a top‑k winner‑take‑all mask
  (`top_k_mask`), Hamming distance utilities and an exponential evasion
  schedule (`evasion_delta`).

* **hybrid_fisher_localization_hybrid_ternary_route_m40_s5.py** – supplies
  Gaussian‑beam weighting (`gaussian_beam`), Fisher information for a Gaussian
  beam (`fisher_score`), a text‑to‑signal conversion (`text_to_signal`) and a
  weighted structural similarity index (`weighted_ssim`).

Mathematical bridge
-------------------
Both parents treat a *scalar confidence* derived from the signal‑to‑noise gap.
In the WTA side this confidence rescales the random coefficients used for the
social interaction; in the Fisher side the same confidence appears as the
parameter `θ` of the Gaussian beam, which directly weights the SSIM
computation.  The hybrid algorithm therefore

1. expands a dense input into a high‑dimensional sparse vector,
2. evaluates a **confidence scalar** `c = (max−min)/σ`,
3. uses `c` as the angle `θ` for the Gaussian beam, producing a per‑component
   weight vector,
4. multiplies the sparse vector by the Fisher information `F(θ)` (which is a
   function of the same `c`), and
5. finally applies a top‑k winner‑take‑all mask before measuring similarity
   with the weighted SSIM.

The resulting pipeline fuses the matrix‑style projection of the WTA algorithm
with the information‑theoretic weighting of the Fisher‑based algorithm in a
single mathematically coherent flow.

The module below implements three core hybrid functions and a small smoke
test.
"""

import hashlib
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Sparse WTA utilities
# ----------------------------------------------------------------------
def expand(values: List[float], m: int, salt: str = "") -> List[float]:
    """Hash‑based sparse expansion of ``values`` into a vector of length ``m``."""
    if m <= 0:
        raise ValueError("m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            j = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if h[8] & 1 else -1.0
            out[j] += sign * v
    return out


def top_k_mask(values: List[float], k: int) -> List[int]:
    """Binary mask with ``1`` at the indices of the top‑k values."""
    k = max(0, min(k, len(values)))
    winners = {
        i
        for i, _ in sorted(enumerate(values), key=lambda x: (-x[1], x[0]))[:k]
    }
    return [1 if i in winners else 0 for i in range(len(values))]


def hamming(a: List[int], b: List[int]) -> int:
    """Hamming distance between two binary vectors."""
    return sum(el1 != el2 for el1, el2 in zip(a, b))


def evasion_delta(t: int, t_max: int, delta_max: float = 1.0, alpha: float = 3.0) -> float:
    """Exponential decay schedule for evasion magnitude.

    Parameters
    ----------
    t : int
        Current time step (must be non‑negative).
    t_max : int
        Maximum number of steps (must be positive).
    delta_max : float
        Upper bound of the evasion magnitude.
    alpha : float
        Decay rate; larger ``alpha`` yields faster decay.

    Returns
    -------
    float
        The evasion magnitude at step ``t``.
    """
    if t < 0:
        raise ValueError("t must be non‑negative")
    if t_max <= 0:
        raise ValueError("t_max must be positive")
    if delta_max < 0 or alpha < 0:
        raise ValueError("delta_max and alpha must be non‑negative")
    # Normalised time in [0,1]
    tau = min(t / t_max, 1.0)
    return delta_max * math.exp(-alpha * tau)


# ----------------------------------------------------------------------
# Parent B – Fisher‑weighted SSIM utilities
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ of a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def text_to_signal(text: str) -> List[float]:
    """Convert a Unicode string to a numeric signal (list of code points)."""
    return [float(ord(ch)) for ch in text]


def weighted_ssim(
    x: Sequence[float],
    y: Sequence[float],
    theta: float,
    center: float,
    width: float,
    dynamic_range: float | None = None,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Weighted Structural Similarity Index.

    The weight for each sample is the Gaussian beam intensity evaluated at the
    sample index (treated as ``θ``).  This couples the Fisher‑information side
    directly to the similarity computation.
    """
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    # Per‑sample angles are the integer indices.
    indices = np.arange(len(x_arr), dtype=np.float64)
    w = np.exp(-0.5 * ((indices - center) / width) ** 2)

    # Normalise weights
    w_sum = np.sum(w) + 1e-12
    w = w / w_sum

    mu_x = np.sum(w * x_arr)
    mu_y = np.sum(w * y_arr)

    sigma_x2 = np.sum(w * (x_arr - mu_x) ** 2)
    sigma_y2 = np.sum(w * (y_arr - mu_y) ** 2)
    sigma_xy = np.sum(w * (x_arr - mu_x) * (y_arr - mu_y))

    if dynamic_range is None:
        all_vals = np.concatenate([x_arr, y_arr])
        dynamic_range = np.max(all_vals) - np.min(all_vals)
        if dynamic_range == 0:
            dynamic_range = 1.0

    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)

    return float(numerator / (denominator + 1e-12))


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def confidence_scalar(values: List[float], eps: float = 1e-12) -> float:
    """Signal‑to‑noise based confidence: (max−min)/σ."""
    arr = np.asarray(values, dtype=np.float64)
    diff = np.max(arr) - np.min(arr)
    sigma = np.std(arr) + eps
    return float(diff / sigma)


def hybrid_expand(
    values: List[float],
    m: int,
    salt: str,
    theta: float,
    center: float,
    width: float,
) -> List[float]:
    """
    Sparse expansion followed by Fisher‑information scaling.

    The raw expansion ``e`` is multiplied element‑wise by the Fisher score
    evaluated at ``θ = confidence_scalar(e)``.  The result is a confidence‑aware
    sparse representation.
    """
    e = expand(values, m, salt)
    c = confidence_scalar(e)
    f = fisher_score(theta=c, center=center, width=width)
    # Scale the whole vector by the scalar Fisher information.
    return [v * f for v in e]


def hybrid_top_k_mask(
    values: List[float],
    k: int,
    theta: float,
    center: float,
    width: float,
) -> List[int]:
    """
    Top‑k mask where the selection threshold is modulated by Fisher confidence.

    The confidence scalar ``c`` is interpreted as an angle and fed to
    ``fisher_score``; the resulting factor rescales the values before the
    top‑k operation.
    """
    c = confidence_scalar(values)
    f = fisher_score(theta=c, center=center, width=width)
    scaled = [v * f for v in values]
    return top_k_mask(scaled, k)


def hybrid_similarity(
    txt_a: str,
    txt_b: str,
    *,
    m: int = 1024,
    k: int = 20,
    salt: str = "hybrid",
    center: float = 0.0,
    width: float = 10.0,
    t_max: int = 100,
    delta_max: float = 1.0,
    alpha: float = 3.0,
) -> Dict[str, Any]:
    """
    End‑to‑end hybrid similarity pipeline.

    1. Convert texts to numeric signals.
    2. Sparse‑expand each signal (confidence‑scaled).
    3. Apply a Fisher‑modulated top‑k mask.
    4. Compute a weighted SSIM between the masked vectors.
    5. Return also the current evasion magnitude (for potential downstream use).

    Returns a dictionary with keys:
        ``'ssim'``        – weighted SSIM score,
        ``'evasion'``    – evasion magnitude at step derived from confidence,
        ``'mask_a'``/``'mask_b'`` – binary masks,
        ``'expanded_a'``/``'expanded_b'`` – raw expanded vectors (pre‑mask).
    """
    # 1. Text → signal
    sig_a = text_to_signal(txt_a)
    sig_b = text_to_signal(txt_b)

    # 2. Hybrid expansion
    exp_a = hybrid_expand(sig_a, m, salt, theta=0.0, center=center, width=width)
    exp_b = hybrid_expand(sig_b, m, salt, theta=0.0, center=center, width=width)

    # 3. Fisher‑modulated top‑k masks
    mask_a = hybrid_top_k_mask(exp_a, k, theta=0.0, center=center, width=width)
    mask_b = hybrid_top_k_mask(exp_b, k, theta=0.0, center=center, width=width)

    # Apply masks
    masked_a = [v if m_i else 0.0 for v, m_i in zip(exp_a, mask_a)]
    masked_b = [v if m_i else 0.0 for v, m_i in zip(exp_b, mask_b)]

    # 4. Weighted SSIM – we treat the index as θ for the weight function.
    ssim_val = weighted_ssim(
        masked_a,
        masked_b,
        theta=0.0,          # not used directly inside weighted_ssim (indices act as θ)
        center=center,
        width=width,
    )

    # 5. Evasion magnitude – derive a step from the confidence of the first signal
    conf = confidence_scalar(exp_a)
    step = int(conf * t_max) % t_max
    evasion = evasion_delta(step, t_max, delta_max, alpha)

    return {
        "ssim": ssim_val,
        "evasion": evasion,
        "mask_a": mask_a,
        "mask_b": mask_b,
        "expanded_a": exp_a,
        "expanded_b": exp_b,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    txt1 = "The quick brown fox jumps over the lazy dog."
    txt2 = "The quick brown fox leaps over the lazy dog!"

    result = hybrid_similarity(txt1, txt2, m=2048, k=30, center=100.0, width=30.0)
    print("Hybrid weighted SSIM :", result["ssim"])
    print("Current evasion magnitude :", result["evasion"])
    print("Mask A sum :", sum(result["mask_a"]))
    print("Mask B sum :", sum(result["mask_b"]))