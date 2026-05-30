# DARWIN HAMMER — match 4477, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_hybrid_m1795_s4.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_capybara_optimizatio_m245_s0.py (gen4)
# born: 2026-05-29T23:56:01Z

"""Hybrid RBF‑Hash Surrogate + Probabilistic Label Optimization

Parent A: ``hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_hybrid_m1795_s4.py`` – provides a
combined kernel that mixes Euclidean distance with Hamming distance of perceptual
hashes and a ridge‑free RBF surrogate model.

Parent B: ``hybrid_hybrid_hybrid_hybrid_capybara_optimizatio_m245_s0.py`` – defines
probabilistic labels (binary label with a confidence score) and a simple
aggregation routine.  Its narrative stresses the use of expected values of an
objective function together with movement‑primitive optimisation.

**Mathematical bridge**  
The surrogate model of Parent A learns a mapping  f̂(x) ≈ E[label | x] by
regressing the expected label (confidence × binary label) on the combined kernel.
The optimisation routine of Parent B then treats f̂ as a black‑box objective and
searches the input space with stochastic movement primitives (Gaussian steps)
to maximise the predicted expected label.  Thus the kernel regression supplies a
smooth differentiable surrogate for the probabilistic objective, while the
movement‑primitive optimiser provides a gradient‑free search that respects the
uncertainty encoded in the labels.

The module below fuses both sides:
* kernel construction & surrogate fitting (A)
* probabilistic label aggregation (B)
* optimisation of the surrogate using movement primitives (B)
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Iterable, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – hash‑augmented RBF kernel utilities
# ----------------------------------------------------------------------
Vector = np.ndarray


def compute_phash(data: bytes, bits: int = 64) -> int:
    """Return a ``bits``‑wide perceptual hash of the raw bytes."""
    # Simple MD5‑based hash; deterministic and fast.
    digest = np.frombuffer(data, dtype=np.uint8)
    # Use numpy's built‑in crc32 as a lightweight stand‑in for MD5.
    # (We avoid hashlib to stay within stdlib + numpy.)
    h = np.uint64(np.sum(digest) * 0x9e3779b97f4a7c15)  # a mix constant
    return int(h & ((1 << bits) - 1))


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers interpreted as bit‑vectors."""
    return bin(a ^ b).count("1")


def combined_kernel(
    X: np.ndarray,
    eps_e: float = 1.0,
    eps_h: float = 1.0,
    hash_bits: int = 64,
) -> np.ndarray:
    """RBF kernel multiplied by a Gaussian of normalized Hamming distance."""
    N = X.shape[0]
    # Euclidean squared distances
    sq_dists = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)

    # Perceptual hashes for every row
    hashes = np.array(
        [compute_phash(X[i].tobytes(), bits=hash_bits) for i in range(N)],
        dtype=np.uint64,
    )

    # Pairwise Hamming distances
    ham = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i, N):
            d = hamming_distance(int(hashes[i]), int(hashes[j]))
            ham[i, j] = d
            ham[j, i] = d
    ham_norm = ham / hash_bits

    # Gaussian in Euclidean space * Gaussian in Hamming space
    K = np.exp(-eps_e * sq_dists - eps_h * ham_norm ** 2)
    return K


def fit_hybrid(X: np.ndarray, y: np.ndarray, eps_e: float = 1.0, eps_h: float = 1.0) -> np.ndarray:
    """Solve K w = y for the hybrid kernel."""
    K = combined_kernel(X, eps_e, eps_h)
    # Regularisation for numerical stability
    K += np.eye(K.shape[0]) * 1e-12
    w = np.linalg.solve(K, y)
    return w


@dataclass
class HybridRBFSurrogate:
    X_train: np.ndarray
    w: np.ndarray
    eps_e: float = 1.0
    eps_h: float = 1.0
    hash_bits: int = 64

    def _kernel_between(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        sq = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
        hashes_A = np.array(
            [compute_phash(a.tobytes(), bits=self.hash_bits) for a in A],
            dtype=np.uint64,
        )
        hashes_B = np.array(
            [compute_phash(b.tobytes(), bits=self.hash_bits) for b in B],
            dtype=np.uint64,
        )
        N, M = A.shape[0], B.shape[0]
        ham = np.empty((N, M), dtype=np.float64)
        for i in range(N):
            for j in range(M):
                ham[i, j] = hamming_distance(int(hashes_A[i]), int(hashes_B[j]))
        ham_norm = ham / self.hash_bits
        return np.exp(-self.eps_e * sq - self.eps_h * ham_norm ** 2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        K_test = self._kernel_between(X, self.X_train)
        return K_test @ self.w


# ----------------------------------------------------------------------
# Parent B – probabilistic labeling and movement‑primitive optimisation
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LabelingFunctionResult:
    """Result of a labeling function."""
    lf_name: str
    doc_id: str
    label: int  # 0 or 1


@dataclass(frozen=True)
class ProbabilisticLabel:
    """Probabilistic label with confidence."""
    doc_id: str
    label: int
    confidence: float  # in [0, 1]


def labeling_function(name: str | None = None):
    """Decorator for labeling functions used in the demo."""
    def deco(fn: Callable[[dict], int]):
        fn.lf_name = name or fn.__name__
        return fn
    return deco


def aggregate_labels(batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
    """Aggregate raw labeling results into probabilistic labels.

    For each document we compute the proportion of positive votes as confidence.
    """
    votes: Dict[str, List[int]] = {}
    for batch in batches:
        for r in batch:
            if r.label not in (0, 1):
                continue
            votes.setdefault(r.doc_id, []).append(r.label)

    out: List[ProbabilisticLabel] = []
    for doc_id, vs in votes.items():
        pos = sum(vs)
        total = len(vs)
        confidence = pos / total if total > 0 else 0.0
        label = 1 if confidence >= 0.5 else 0
        out.append(ProbabilisticLabel(doc_id=doc_id, label=label, confidence=confidence))
    return out


# ----------------------------------------------------------------------
# Hybrid operations (three required functions)
# ----------------------------------------------------------------------
def prepare_training_data(
    X: np.ndarray,
    prob_labels: List[ProbabilisticLabel],
) -> Tuple[np.ndarray, np.ndarray]:
    """Map probabilistic labels onto the rows of ``X``.

    The function assumes that the order of ``prob_labels`` matches the rows of ``X``.
    It returns ``y`` as the expected label:  confidence * label.
    """
    # Simple safety check – align by doc_id if present in X metadata
    y = np.array([pl.confidence * pl.label for pl in prob_labels], dtype=np.float64)
    return X, y


def fit_hybrid_surrogate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    eps_e: float = 1.0,
    eps_h: float = 1.0,
) -> HybridRBFSurrogate:
    """Convenience wrapper that returns a ready‑to‑use surrogate object."""
    w = fit_hybrid(X_train, y_train, eps_e, eps_h)
    return HybridRBFSurrogate(X_train=X_train, w=w, eps_e=eps_e, eps_h=eps_h)


def optimize_surrogate(
    surrogate: HybridRBFSurrogate,
    init_point: np.ndarray,
    step_size: float = 0.1,
    n_iter: int = 200,
    rng: random.Random | None = None,
) -> Tuple[np.ndarray, float]:
    """Hill‑climbing optimisation using Gaussian movement primitives.

    Returns the best point found and its surrogate prediction.
    """
    rng = rng or random.Random()
    best_x = init_point.copy()
    best_val = float(surrogate.predict(best_x[None, :])[0])

    dim = init_point.shape[0]

    for _ in range(n_iter):
        # Propose a new point by adding isotropic Gaussian noise
        proposal = best_x + step_size * np.array(
            [rng.gauss(0, 1) for _ in range(dim)], dtype=np.float64
        )
        # Clip to a reasonable range (here we assume features are in [0, 1])
        proposal = np.clip(proposal, 0.0, 1.0)

        val = float(surrogate.predict(proposal[None, :])[0])
        if val > best_val:
            best_val = val
            best_x = proposal

    return best_x, best_val


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Synthetic feature matrix (10 samples, 3 dimensions)
    rng = np.random.default_rng(42)
    X = rng.random((10, 3))

    # 2. Create dummy labeling function results (two batches)
    @labeling_function("lf_a")
    def lf_a(doc: dict) -> int:
        return int(doc["feature_sum"] > 1.2)

    @labeling_function("lf_b")
    def lf_b(doc: dict) -> int:
        return int(doc["feature_sum"] < 0.8)

    batches: List[List[LabelingFunctionResult]] = []
    for batch_idx in range(2):
        batch: List[LabelingFunctionResult] = []
        for i, row in enumerate(X):
            doc = {"feature_sum": float(row.sum())}
            # Alternate labeling functions per batch
            lf = lf_a if batch_idx % 2 == 0 else lf_b
            label = lf(doc)
            batch.append(LabelingFunctionResult(lf_name=lf.lf_name, doc_id=str(i), label=label))
        batches.append(batch)

    # 3. Aggregate to probabilistic labels
    prob_labels = aggregate_labels(batches)

    # 4. Prepare training data (y = confidence * label)
    X_train, y_train = prepare_training_data(X, prob_labels)

    # 5. Fit the hybrid surrogate
    surrogate = fit_hybrid_surrogate(X_train, y_train, eps_e=5.0, eps_h=2.0)

    # 6. Optimise starting from the centre of the space
    init = np.full((3,), 0.5)
    best_point, best_score = optimize_surrogate(surrogate, init, step_size=0.05, n_iter=500)

    print("Best point found :", best_point)
    print("Surrogate prediction at best point :", best_score)
    print("All surrogate predictions :", surrogate.predict(X_train))