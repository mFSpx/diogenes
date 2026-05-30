# DARWIN HAMMER — match 5191, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_model_vram_sc_hybrid_geometric_pro_m4_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_serpentina_se_m872_s3.py (gen5)
# born: 2026-05-30T00:00:28Z

"""Hybrid Module: VRAM‑Adaptive Geometric‑Product SSIM & Fisher Scoring

Parents:
- hybrid_hybrid_model_vram_sc_hybrid_geometric_pro_m4_s2.py (VRAM scheduler + geometric product)
- hybrid_hybrid_hybrid_minimu_hybrid_serpentina_se_m872_s3.py (Morphology‑aware SSIM, Fisher score)

Mathematical Bridge:
The geometric product combines two feature vectors **u**, **v** into a multivector  
  G(u,v) = u·v + u∧v, where “·” is the Euclidean inner product (scalar part) and “∧” is the outer
product (bivector part).  For image‑based pipelines the vectors are derived from
gradient or intensity statistics of the input images.

The VRAM scheduler supplies a memory‑aware scaling factor 𝛼 ∈ (0,1] that modulates
the influence of the geometric product on downstream similarity measures.
When free GPU memory is abundant, α≈1 and the full geometric contribution is used;
under memory pressure α is reduced, effectively blending the scalar‑only term.

Both parents expose similarity‑oriented metrics (SSIM, Fisher score).  The hybrid
function `adaptive_ssim` injects the scaled geometric product into the SSIM
denominator, while `adaptive_fisher` uses the same α‑scaled product to weight
the Fisher information.  This creates a unified system where memory constraints
directly affect the geometric enrichment of image similarity and statistical
scoring."""
import math
import random
import sys
import pathlib
import datetime
import subprocess
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Callable, Any, Dict

# ----------------------------------------------------------------------
# Basic utilities from Parent A (VRAM scheduler)
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Z format."""
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def gpu_memory() -> Dict[str, Any]:
    """
    Query GPU memory via `nvidia-smi`.  Returns a dict with keys:
    - status: "ok" | "missing" | "error"
    - free: free memory in MiB (float) if status == "ok"
    """
    if not pathlib.Path("/usr/bin/nvidia-smi").exists():
        return {"status": "missing", "message": "nvidia-smi not found"}
    try:
        cp = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5,
        )
        if cp.returncode != 0:
            return {"status": "error", "stderr": cp.stderr[:500]}
        free = float(cp.stdout.strip().splitlines()[0])
        return {"status": "ok", "free": free}
    except Exception as e:
        return {"status": "error", "exception": str(e)}


def vram_adaptive_factor(threshold_mib: float = 1024.0) -> float:
    """
    Return a scaling factor α based on free GPU memory.
    If free >= threshold → α = 1.0
    Else α linearly decays to 0.2 at zero free memory.
    """
    info = gpu_memory()
    if info.get("status") != "ok":
        return 0.5  # conservative default
    free = info["free"]
    if free >= threshold_mib:
        return 1.0
    # linear interpolation between (0,0.2) and (threshold,1.0)
    return 0.2 + 0.8 * max(free, 0.0) / threshold_mib


# ----------------------------------------------------------------------
# Geometric product (Parent A core)
# ----------------------------------------------------------------------
def geometric_product(a: np.ndarray, b: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute the geometric product of two equal‑length real vectors.
    Returns (scalar_part, bivector_part) where:
    - scalar_part = a·b (dot product)
    - bivector_part = a∧b represented as the exterior product matrix,
      flattened to a 1‑D array of length n*(n‑1)/2.
    """
    if a.shape != b.shape:
        raise ValueError("Vectors must have the same shape")
    scalar = float(np.dot(a, b))
    n = a.shape[0]
    # Build bivector components (i<j)
    biv = []
    for i in range(n):
        for j in range(i + 1, n):
            biv.append(a[i] * b[j] - a[j] * b[i])
    return scalar, np.array(biv, dtype=float)


# ----------------------------------------------------------------------
# Morphology & similarity utilities (Parent B core)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]


def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float = 0.0


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def gaussian_beam(theta: float, center: float, width: float, sphericity: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return sphericity * math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, sphericity: float,
                 eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width, sphericity), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03,
         morphology: Morphology = None) -> float:
    """
    Structural Similarity Index, optionally scaled by morphology‑derived factors.
    """
    if x.shape != y.shape:
        raise ValueError("Input images must have the same dimensions")
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2

    if morphology:
        scale = sphericity_index(morphology.length, morphology.width, morphology.height)
        C1 *= scale
        C2 *= scale

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Hybrid Functions (new)
# ----------------------------------------------------------------------
def extract_feature_vector(image: np.ndarray, patch_size: int = 8) -> np.ndarray:
    """
    Simple feature extractor: flatten the mean‑subtracted central patch.
    """
    h, w = image.shape[:2]
    ph = h // 2 - patch_size // 2
    pw = w // 2 - patch_size // 2
    patch = image[ph:ph + patch_size, pw:pw + patch_size].astype(float)
    return (patch - patch.mean()).ravel()


def adaptive_ssim(img1: np.ndarray, img2: np.ndarray,
                  morphology: Morphology = None) -> float:
    """
    SSIM that incorporates a memory‑aware geometric product of image features.
    The geometric scalar part scales the SSIM numerator; the bivector norm
    attenuates the denominator via the VRAM factor α.
    """
    base_ssim = ssim(img1, img2, morphology=morphology)

    # Feature vectors
    f1 = extract_feature_vector(img1)
    f2 = extract_feature_vector(img2)

    # Geometric product
    scalar, biv = geometric_product(f1, f2)
    biv_norm = np.linalg.norm(biv)

    # VRAM‑adaptive factor
    alpha = vram_adaptive_factor()

    # Fuse: boost similarity when scalar is large, penalize when bivector is large
    fused = base_ssim * (1 + alpha * scalar / (1 + biv_norm))
    return float(np.clip(fused, -1.0, 1.0))


def adaptive_fisher(theta: float, center: float, width: float,
                    morphology: Morphology = None) -> float:
    """
    Fisher information score modulated by morphology and VRAM factor.
    The sphericity index provides a physical scaling; α reduces the score
    under low‑memory conditions.
    """
    if morphology is None:
        raise ValueError("Morphology must be provided")
    sphericity = sphericity_index(morphology.length, morphology.width, morphology.height)
    raw = fisher_score(theta, center, width, sphericity)
    alpha = vram_adaptive_factor()
    return raw * alpha


def geometric_ssim_gradient(img: np.ndarray, morphology: Morphology = None) -> np.ndarray:
    """
    Compute a gradient map where each pixel's value is the SSIM between its
    local neighbourhood and the image centre, weighted by the geometric product
    of gradient vectors.  This demonstrates a per‑pixel hybrid operation.
    """
    h, w = img.shape[:2]
    grad_y, grad_x = np.gradient(img.astype(float))
    centre_feat = extract_feature_vector(img)

    out = np.zeros((h, w), dtype=float)
    patch_sz = 8
    half = patch_sz // 2
    for i in range(half, h - half):
        for j in range(half, w - half):
            local_patch = img[i - half:i + half, j - half:j + half]
            local_feat = extract_feature_vector(local_patch)
            scalar, biv = geometric_product(centre_feat, local_feat)
            biv_norm = np.linalg.norm(biv)
            alpha = vram_adaptive_factor()
            # Local SSIM with centre (using full images for simplicity)
            local_ssim = ssim(img, local_patch, morphology=morphology)
            out[i, j] = local_ssim * (1 + alpha * scalar / (1 + biv_norm))
    return out


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create two synthetic images
    rng = np.random.default_rng(42)
    img_a = (rng.random((64, 64)) * 255).astype(np.uint8)
    img_b = img_a.copy()
    # Introduce slight noise
    img_b = np.clip(img_b + rng.integers(-5, 6, size=img_b.shape), 0, 255).astype(np.uint8)

    morph = Morphology(length=10.0, width=8.0, height=5.0)

    print("VRAM adaptive factor:", vram_adaptive_factor())
    print("Adaptive SSIM:", adaptive_ssim(img_a, img_b, morphology=morph))
    print("Adaptive Fisher (theta=0.3):", adaptive_fisher(theta=0.3, center=0.0, width=1.0, morphology=morph))

    # Gradient map test (small size for speed)
    grad_map = geometric_ssim_gradient(img_a, morphology=morph)
    print("Gradient map stats – min:", grad_map.min(), "max:", grad_map.max())