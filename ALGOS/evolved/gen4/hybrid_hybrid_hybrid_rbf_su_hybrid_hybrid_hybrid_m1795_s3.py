# DARWIN HAMMER — match 1795, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_rbf_surrogate_perceptual_dedupe_m57_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_regret_engine_m2_s4.py (gen3)
# born: 2026-05-29T23:39:02Z

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
import datetime as dt
import re
import hashlib

import numpy as np

Vector = np.ndarray  

def compute_phash(data: bytes, bits: int = 64) -> int:
    digest = hashlib.md5(data).digest()
    h_int = int.from_bytes(digest, byteorder="big")
    return h_int & ((1 << bits) - 1)


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def combined_kernel(
    X: np.ndarray,
    eps_e: float = 1.0,
    eps_h: float = 1.0,
    hash_bits: int = 64,
) -> np.ndarray:
    N = X.shape[0]
    sq_dists = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)  

    hashes = np.array([compute_phash(X[i].tobytes(), bits=hash_bits) for i in range(N)], dtype=np.uint64)
    ham_matrix = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i, N):
            d = hamming_distance(int(hashes[i]), int(hashes[j]))
            ham_matrix[i, j] = d
            ham_matrix[j, i] = d

    ham_norm = ham_matrix / hash_bits

    K = np.exp(-eps_e * sq_dists - eps_h * ham_norm ** 2)
    return K


def fit_hybrid(X: np.ndarray, y: np.ndarray, eps_e: float = 1.0, eps_h: float = 1.0) -> np.ndarray:
    K = combined_kernel(X, eps_e, eps_h)
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        K_test = self._kernel_between(X, self.X_train)
        return K_test @ self.w

    def _kernel_between(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        sq = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)

        hashes_A = np.array([compute_phash(a.tobytes(), bits=self.hash_bits) for a in A], dtype=np.uint64)
        hashes_B = np.array([compute_phash(b.tobytes(), bits=self.hash_bits) for b in B], dtype=np.uint64)
        N, M = len(A), len(B)
        ham = np.empty((N, M), dtype=np.float64)
        for i in range(N):
            for j in range(M):
                ham[i, j] = hamming_distance(int(hashes_A[i]), int(hashes_B[j]))
        ham_norm = ham / self.hash_bits

        return np.exp(-self.eps_e * sq - self.eps_h * ham_norm ** 2)


EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)


def extract_counts(text: str) -> np.ndarray:
    return np.array(
        [
            len(EVIDENCE_RE.findall(text)),
            len(PLANNING_RE.findall(text)),
            len(DELAY_RE.findall(text)),
            len(SUPPORT_RE.findall(text)),
        ],
        dtype=np.float64,
    )


def regret_softmax(u: np.ndarray) -> np.ndarray:
    shift = np.max(u)
    exps = np.exp(u - shift)
    return exps / np.sum(exps)


def gini_coefficient(p: np.ndarray) -> float:
    if p.ndim != 1:
        raise ValueError("p must be a one‑dimensional array")
    sorted_p = np.sort(p)
    n = len(p)
    cum = np.cumsum(sorted_p)
    gini = 1.0 - 2.0 * np.sum(cum) / (n * np.sum(sorted_p)) + (1.0 / n)
    return float(gini)


def compute_utilities(
    counts: np.ndarray,
    pos_weights: np.ndarray,
    neg_weights: np.ndarray,
    surrogate_pred: float,
    alpha: float = 0.5,
) -> np.ndarray:
    utilities = (pos_weights * counts) - (neg_weights * counts) + alpha * surrogate_pred
    return utilities


def hybrid_decision(
    text: str,
    pos_weights: np.ndarray,
    neg_weights: np.ndarray,
    X_train: np.ndarray,
    w: np.ndarray,
    eps_e: float = 1.0,
    eps_h: float = 1.0,
    alpha: float = 0.5,
) -> float:
    counts = extract_counts(text)
    surrogate = HybridRBFSurrogate(X_train, w, eps_e, eps_h)
    surrogate_pred = surrogate.predict(np.array([counts]))[0]
    utilities = compute_utilities(counts, pos_weights, neg_weights, surrogate_pred, alpha)
    probabilities = regret_softmax(utilities)
    return gini_coefficient(probabilities)


def improved_hybrid_decision(
    text: str,
    pos_weights: np.ndarray,
    neg_weights: np.ndarray,
    X_train: np.ndarray,
    w: np.ndarray,
    eps_e: float = 1.0,
    eps_h: float = 1.0,
    alpha: float = 0.5,
    hash_bits: int = 128, # Increased hash bits for better perceptual hash
) -> float:
    counts = extract_counts(text)
    surrogate = HybridRBFSurrogate(X_train, w, eps_e, eps_h, hash_bits)
    surrogate_pred = surrogate.predict(np.array([counts]))[0]
    utilities = compute_utilities(counts, pos_weights, neg_weights, surrogate_pred, alpha)
    probabilities = regret_softmax(utilities)
    return gini_coefficient(probabilities)


def main():
    # Example usage
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0.5, 0.6, 0.7])
    w = fit_hybrid(X_train, y_train)

    pos_weights = np.array([0.2, 0.3, 0.1, 0.4])
    neg_weights = np.array([0.1, 0.2, 0.3, 0.4])

    text = "This is an example text with evidence and planning."

    decision = improved_hybrid_decision(text, pos_weights, neg_weights, X_train, w)
    print(decision)


if __name__ == "__main__":
    main()