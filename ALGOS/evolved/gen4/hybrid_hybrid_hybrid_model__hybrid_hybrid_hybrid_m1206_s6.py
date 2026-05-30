# DARWIN HAMMER — match 1206, survivor 6
# gen: 4
# parent_a: hybrid_hybrid_model_vram_sc_hybrid_ssim_doomsday_m214_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_privac_hybrid_infotaxis_min_m106_s1.py (gen3)
# born: 2026-05-29T23:34:36Z

"""Hybrid Algorithm integrating GPU‑VRAM signal SSIM (Parent A) with
Model‑Pool entropy‑guided MinHash similarity (Parent B).

Mathematical bridge:
- Both parents quantify similarity: Parent A via Structural Similarity Index
  (SSIM) between two 1‑D signals, Parent B via Jaccard similarity approximated
  by MinHash signatures.
- We fuse them by defining a *Hybrid Similarity* 𝛴 = α·SSIM(x, y) + β·Ĵ(x, y)
  where Ĵ is the MinHash‑based Jaccard estimate and α,β∈[0,1] are weighting
  coefficients.  The entropy of the model‑signature distribution, H(p), is
  used as a dynamic weight modifier: α←α·(1‑H/Hmax), β←β·(H/Hmax).
- This unified metric drives VRAM‑aware model‑pool decisions: a model whose
  signature is most similar to the current GPU‑memory signal (high 𝛴) is kept
  in memory, while less similar models are evicted when the RAM ceiling is
  approached.
"""

import numpy as np
import math
import random
import sys
import pathlib
from typing import Dict, List, Tuple, Sequence
from dataclasses import dataclass

# ----------------------------------------------------------------------
# Helper utilities (shared by both parents)
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Z format."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _rel(path: pathlib.Path | str) -> str:
    """Return a path relative to the repository root (best‑effort)."""
    try:
        root = pathlib.Path(__file__).resolve().parents[2]
        return str(pathlib.Path(path).resolve().relative_to(root))
    except Exception:
        return str(path)

# ----------------------------------------------------------------------
# Parent A – SSIM on GPU‑VRAM & periodic signals
# ----------------------------------------------------------------------
def generate_gpu_memory_signal(length: int = 256,
                               base_mb: int = 4096,
                               variance_mb: int = 512,
                               seed: int | None = None) -> np.ndarray:
    """Simulate a 1‑D GPU‑memory usage signal.

    The signal is a noisy sinusoid around ``base_mb`` to mimic periodic
    allocation/deallocation patterns.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    sinusoid = np.sin(2 * math.pi * t / length)
    noise = rng.normal(0, variance_mb * 0.05, size=length)
    signal = base_mb + variance_mb * sinusoid + noise
    return signal.astype(np.float64)


def generate_periodic_signal(length: int = 256,
                             amplitude: float = 500.0,
                             frequency: float = 1.0,
                             phase: float = 0.0,
                             seed: int | None = None) -> np.ndarray:
    """Generate a clean periodic reference signal (sine wave)."""
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    signal = amplitude * np.sin(2 * math.pi * frequency * t / length + phase)
    # Add tiny jitter to avoid perfect alignment (helps SSIM stability)
    jitter = rng.normal(0, amplitude * 0.01, size=length)
    return (signal + jitter).astype(np.float64)


def _ssim_component(x: np.ndarray, y: np.ndarray, C1: float, C2: float) -> Tuple[float, float, float]:
    """Return (luminance, contrast, structure) components of SSIM."""
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    l = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)
    c = (2 * math.sqrt(sigma_x) * math.sqrt(sigma_y) + C2) / (sigma_x + sigma_y + C2)
    s = (sigma_xy + C2 / 2) / (math.sqrt(sigma_x) * math.sqrt(sigma_y) + C2 / 2)
    return l, c, s


def compute_ssim(x: np.ndarray, y: np.ndarray,
                 K1: float = 0.01, K2: float = 0.03,
                 L: float = 65535) -> float:
    """Simplified SSIM for 1‑D signals."""
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    l, c, s = _ssim_component(x, y, C1, C2)
    return float(l * c * s)


# ----------------------------------------------------------------------
# Parent B – MinHash, entropy and model‑pool management
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    """Descriptor of a model with its VRAM footprint."""
    name: str
    ram_mb: int
    tier: str  # e.g. "T1", "T2", "T3"


# Example tier definitions (could be extended by the user)
TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1")
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2")
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2")
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3")


class ModelPool:
    """RAM‑aware pool that uses hybrid similarity to decide eviction."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self.signatures: Dict[str, np.ndarray] = {}  # MinHash signatures per model

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def _entropy(self) -> float:
        """Shannon entropy of the distribution of signature bit‑counts."""
        if not self.signatures:
            return 0.0
        # Count number of 1‑bits per signature (simple proxy for diversity)
        counts = np.array([np.sum(sig) for sig in self.signatures.values()], dtype=float)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-12))

    def _hybrid_score(self,
                      gpu_signal: np.ndarray,
                      periodic_signal: np.ndarray,
                      model_sig: np.ndarray,
                      alpha: float = 0.5,
                      beta: float = 0.5) -> float:
        """Combined SSIM + MinHash Jaccard score for a single model."""
        # SSIM between GPU memory and periodic reference
        ssim_val = compute_ssim(gpu_signal, periodic_signal)

        # Jaccard estimate via MinHash between the GPU signal (hashed) and model signature
        gpu_sig = minhash_signature(gpu_signal)
        intersect = np.sum(np.minimum(gpu_sig, model_sig))
        union = np.sum(np.maximum(gpu_sig, model_sig))
        jaccard_est = intersect / (union + 1e-12)

        # Entropy‑driven weighting
        H = self._entropy()
        Hmax = math.log2(len(self.signatures)) if self.signatures else 1.0
        w = H / (Hmax + 1e-12)  # in [0,1]
        a = alpha * (1 - w)
        b = beta * w

        return a * ssim_val + b * jaccard_est

    def load(self,
             model: ModelTier,
             gpu_signal: np.ndarray,
             periodic_signal: np.ndarray) -> None:
        """Load a model, evicting the least‑similar one if needed."""
        # Compute (or retrieve) model signature
        if model.name not in self.signatures:
            # For demo purposes we hash the model name string into a synthetic signature
            rng = np.random.default_rng(hash(model.name) & 0xffffffff)
            self.signatures[model.name] = rng.integers(0, 2, size=128, dtype=np.uint8)

        # If there is enough room, simply load
        if model.ram_mb + self._used() <= self.ram_ceiling_mb:
            self.loaded[model.name] = model
            return

        # Otherwise evict the model with the lowest hybrid score
        scores = {
            name: self._hybrid_score(gpu_signal, periodic_signal, self.signatures[name])
            for name in self.loaded
        }
        # Identify eviction candidate
        evict_name = min(scores, key=scores.get)
        del self.loaded[evict_name]
        del self.signatures[evict_name]
        # Retry loading (now space is guaranteed)
        self.load(model, gpu_signal, periodic_signal)

    def info(self) -> str:
        """Human‑readable snapshot of the pool."""
        lines = [
            f"RAM ceiling: {self.ram_ceiling_mb} MiB",
            f"Used RAM   : {self._used()} MiB",
            f"Loaded models ({len(self.loaded)}): " + ", ".join(self.loaded),
            f"Current entropy: {self._entropy():.3f} bits",
        ]
        return "\n".join(lines)


# ----------------------------------------------------------------------
# MinHash utilities (Parent B)
# ----------------------------------------------------------------------
def _hash_functions(num_funcs: int = 128, max_val: int = 2**31 - 1) -> List[Tuple[int, int]]:
    """Generate a list of (a, b) coefficients for universal hashing."""
    rng = random.Random(0xDEADBEEF)
    funcs = []
    for _ in range(num_funcs):
        a = rng.randint(1, max_val - 1)
        b = rng.randint(0, max_val - 1)
        funcs.append((a, b))
    return funcs


_HASH_FUNCS = _hash_functions()


def _minhash_one(value: int, a: int, b: int, prime: int = 2**31 - 1) -> int:
    """Hash a single integer under a universal hash."""
    return (a * value + b) % prime


def minhash_signature(data: Sequence[float],
                      num_perm: int = 128,
                      prime: int = 2**31 - 1) -> np.ndarray:
    """Return a binary MinHash signature for a numeric sequence.

    The sequence is first quantised to 32‑bit ints via hashing of its string
    representation; then the minimum hash per permutation defines a bit.
    """
    # Quantise each element
    hashed_vals = np.array([hash(str(v)) & 0xffffffff for v in data], dtype=np.uint32)

    signature = np.zeros(num_perm, dtype=np.uint8)
    for i, (a, b) in enumerate(_HASH_FUNCS[:num_perm]):
        hashed_perm = (_minhash_one(hashed_vals, a, b, prime))
        signature[i] = int(np.min(hashed_perm) % 2)  # 0/1 bit
    return signature


def entropy(dist: Sequence[float]) -> float:
    """Shannon entropy of a probability distribution (base‑2)."""
    probs = np.array(dist, dtype=float)
    probs = probs / (probs.sum() + 1e-12)
    return -np.sum(probs * np.log2(probs + 1e-12))


# ----------------------------------------------------------------------
# Demonstration functions (at least three)
# ----------------------------------------------------------------------
def hybrid_similarity(gpu_signal: np.ndarray,
                      periodic_signal: np.ndarray,
                      model_sig: np.ndarray,
                      alpha: float = 0.6,
                      beta: float = 0.4) -> float:
    """Public wrapper returning the combined similarity score."""
    ssim_val = compute_ssim(gpu_signal, periodic_signal)
    gpu_sig = minhash_signature(gpu_signal)
    intersect = np.sum(np.minimum(gpu_sig, model_sig))
    union = np.sum(np.maximum(gpu_sig, model_sig))
    jaccard_est = intersect / (union + 1e-12)
    return alpha * ssim_val + beta * jaccard_est


def schedule_models(models: List[ModelTier],
                    gpu_signal: np.ndarray,
                    periodic_signal: np.ndarray,
                    ceiling_mb: int = 6000) -> ModelPool:
    """Create a ModelPool and load models respecting the hybrid score."""
    pool = ModelPool(ram_ceiling_mb=ceiling_mb)
    for mdl in models:
        pool.load(mdl, gpu_signal, periodic_signal)
    return pool


def demo_hybrid_workflow() -> None:
    """Run a lightweight end‑to‑end demonstration."""
    # 1. Generate signals
    gpu_sig = generate_gpu_memory_signal(length=256, seed=42)
    periodic_sig = generate_periodic_signal(length=256, amplitude=500, seed=99)

    # 2. Build a small catalogue of models
    catalogue = [
        TIER_T1_QWEN_0_5B,
        TIER_T2_REASONING,
        TIER_T2_TOOL,
        TIER_T3_QWEN_7B,
    ]

    # 3. Initialise pool with hybrid‑aware scheduling
    pool = schedule_models(catalogue, gpu_sig, periodic_sig, ceiling_mb=8000)

    # 4. Show results
    print("=== Hybrid Model‑Pool Snapshot ===")
    print(pool.info())
    # 5. Compute explicit hybrid similarity for the first loaded model
    first_name = next(iter(pool.loaded))
    model_sig = pool.signatures[first_name]
    score = hybrid_similarity(gpu_sig, periodic_sig, model_sig)
    print(f"\nHybrid similarity for model '{first_name}': {score:.4f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_hybrid_workflow()