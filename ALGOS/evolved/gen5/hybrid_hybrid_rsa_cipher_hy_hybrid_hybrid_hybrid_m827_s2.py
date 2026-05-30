# DARWIN HAMMER — match 827, survivor 2
# gen: 5
# parent_a: hybrid_rsa_cipher_hybrid_hybrid_decisi_m137_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_hard_t_m92_s0.py (gen4)
# born: 2026-05-29T23:32:28Z

"""Hybrid RSA‑Metric + RBF‑Surrogate Model

Parents:
* **Parent A** – RSA primitive combined with a hygiene‑entropy metric derived from a
  9‑dimensional feature vector extracted from free‑text.
* **Parent B** – Radial‑Basis‑Function (RBF) surrogate that learns a mapping
  from low‑dimensional feature vectors (e.g., stylometric fingerprints) to a scalar
  output by solving a dense linear system K·w = y.

Mathematical bridge:
The scalar hygiene‑entropy metric *Sₕ* (Parent A) is first scaled to an integer
*m* and RSA‑encrypted to a ciphertext *c*.  The ciphertext serves as the target
label *y* for the RBF surrogate (Parent B).  Consequently the surrogate learns a
function  **f̂**  such that  

  f̂(x) ≈ c = RSAₑ,ₙ(m) ,  m = ⌊scale·Sₕ⌋ .

During inference the surrogate predicts an encrypted integer, which can be
RSA‑decrypted with the private exponent *d* to recover the original scaled metric,
and finally un‑scaled to obtain *Sₕ*.  This unifies the two topologies:
feature extraction → metric → RSA encryption → surrogate learning → prediction →
RSA decryption → metric recovery.

The module implements the full pipeline with three public functions:
`hybrid_train`, `hybrid_predict`, and `hybrid_decrypt`.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utility types
# ----------------------------------------------------------------------
Vector = Sequence[float]

# ----------------------------------------------------------------------
# RSA primitive (from Parent A)
# ----------------------------------------------------------------------


def rsa_encrypt(message: int, e: int, n: int) -> int:
    """RSA encryption: c = m^e mod n."""
    if not 0 <= message < n:
        raise ValueError("message must be in [0, n)")
    return pow(message, e, n)


def rsa_decrypt(ciphertext: int, d: int, n: int) -> int:
    """RSA decryption: m = c^d mod n."""
    if not 0 <= ciphertext < n:
        raise ValueError("ciphertext must be in [0, n)")
    return pow(ciphertext, d, n)


# ----------------------------------------------------------------------
# Feature extraction & hybrid metric (Parent A)
# ----------------------------------------------------------------------


def extract_features(text: str) -> List[float]:
    """
    Very light‑weight extraction of nine numeric cues from free text.
    The implementation mimics the original nine regex counts but uses simple
    word‑based heuristics for demonstration.
    Returns a list of nine float counts.
    """
    tokens = text.lower().split()
    # 1. evidence‑related words
    evidence = sum(tok in {"evidence", "proof", "testimony"} for tok in tokens)
    # 2. hygiene‑related words
    hygiene = sum(tok in {"clean", "sanitize", "hygiene"} for tok in tokens)
    # 3. entropy‑related words (information richness)
    entropy = sum(tok in {"entropy", "information", "data"} for tok in tokens)
    # 4. numeric tokens
    numeric = sum(tok.isdigit() for tok in tokens)
    # 5. question marks
    qmarks = text.count("?")
    # 6. exclamation marks
    excl = text.count("!")
    # 7. uppercase words (shouting)
    upper = sum(tok.isupper() for tok in tokens)
    # 8. length bucket (short vs long)
    length_bucket = 1 if len(text) < 100 else 2
    # 9. placeholder for future cue
    placeholder = 0
    return [
        float(evidence),
        float(hygiene),
        float(entropy),
        float(numeric),
        float(qmarks),
        float(excl),
        float(upper),
        float(length_bucket),
        float(placeholder),
    ]


def shannon_entropy(counts: Sequence[float]) -> float:
    """Compute Shannon entropy of a non‑negative count vector."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log(p, 2) for p in probs)


def compute_hygiene_metric(v: Sequence[float]) -> float:
    """
    Compute the hybrid metric Sₕ = s·(1+H/H_max) where:
      * s = v·w₁ + v·w₂  (two dot‑products with fixed weight vectors)
      * H = Shannon entropy of v
      * H_max = log2(len(v))  (max possible entropy)
    """
    # Fixed weight vectors (chosen arbitrarily for illustration)
    w1 = np.array([0.2, 0.1, 0.15, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1])
    w2 = np.array([0.05, 0.25, 0.1, 0.1, 0.05, 0.15, 0.1, 0.1, 0.1])
    v_arr = np.array(v, dtype=float)
    s = float(v_arr @ w1 + v_arr @ w2)
    H = shannon_entropy(v)
    H_max = math.log2(len(v)) if len(v) > 1 else 1.0
    return s * (1.0 + H / H_max)


def scale_metric_to_int(metric: float, scale: int = 1_000_000) -> int:
    """Fixed‑point scaling of a positive float to a non‑negative integer."""
    if metric < 0:
        raise ValueError("Metric must be non‑negative for RSA encoding")
    return int(metric * scale)


def descale_int_to_metric(value: int, scale: int = 1_000_000) -> float:
    """Inverse of `scale_metric_to_int`."""
    return value / scale


# ----------------------------------------------------------------------
# RBF Surrogate (Parent B)
# ----------------------------------------------------------------------


def gaussian_kernel(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))


def build_kernel_matrix(X: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """
    Construct the symmetric kernel matrix K where
        K[i, j] = exp(- (epsilon * ||x_i - x_j||)^2 )
    """
    n = X.shape[0]
    K = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            r = np.linalg.norm(X[i] - X[j])
            val = gaussian_kernel(r, epsilon)
            K[i, j] = val
            K[j, i] = val
    return K


class RBFSurrogate:
    """Container for a trained RBF surrogate."""

    def __init__(self, X: np.ndarray, weights: np.ndarray):
        self.X = X          # training points (augmented feature vectors)
        self.weights = weights  # solution of K·w = y

    def predict(self, X_new: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
        """
        Predict outputs for new inputs using the trained surrogate:
            ŷ = Σ_j w_j * exp(- (epsilon * ||x_new - x_j||)^2 )
        Returns a 1‑D array of predictions.
        """
        preds = np.empty(X_new.shape[0], dtype=float)
        for i, x in enumerate(X_new):
            diffs = self.X - x  # shape (n_train, dim)
            dists = np.linalg.norm(diffs, axis=1)
            kernels = np.exp(-((epsilon * dists) ** 2))
            preds[i] = float(np.dot(self.weights, kernels))
        return preds


# ----------------------------------------------------------------------
# Hybrid pipeline tying both parents together
# ----------------------------------------------------------------------


def hybrid_train(
    texts: List[str],
    rsa_pub: Tuple[int, int],
    scale: int = 1_000_000,
    epsilon: float = 1.0,
) -> RBFSurrogate:
    """
    Train an RBF surrogate where the target labels are RSA‑encrypted hygiene metrics.

    Parameters
    ----------
    texts : list of raw strings – each becomes a 9‑dim feature vector.
    rsa_pub : (e, n) RSA public key.
    scale : integer scaling factor for metric → integer conversion.
    epsilon : RBF kernel shape parameter.

    Returns
    -------
    RBFSurrogate instance ready for prediction.
    """
    e, n = rsa_pub
    # 1. Feature extraction → metric → integer → RSA encryption
    metrics = []
    encrypted = []
    for txt in texts:
        v = extract_features(txt)
        S = compute_hygiene_metric(v)
        m_int = scale_metric_to_int(S, scale)
        c = rsa_encrypt(m_int, e, n)
        metrics.append(S)
        encrypted.append(c)

    # 2. Build augmented feature vectors.
    #    For illustration we concatenate the original 9‑dim vector with a simple
    #    stylometric fingerprint (character frequency of a‑z, length 26).
    X_aug = []
    for txt in texts:
        v = np.array(extract_features(txt), dtype=float)
        # stylometric fingerprint: normalized counts of letters a‑z
        letters = [ch for ch in txt.lower() if "a" <= ch <= "z"]
        freq = np.zeros(26, dtype=float)
        for ch in letters:
            freq[ord(ch) - 97] += 1.0
        if freq.sum() > 0:
            freq /= freq.sum()
        X_aug.append(np.concatenate([v, freq]))
    X_aug = np.vstack(X_aug)  # shape (N, 35)

    # 3. Solve K·w = y where y = encrypted ciphertexts (treated as floats)
    K = build_kernel_matrix(X_aug, epsilon)
    y = np.array(encrypted, dtype=float)
    weights = np.linalg.solve(K, y)

    return RBFSurrogate(X_aug, weights)


def hybrid_predict(
    model: RBFSurrogate,
    new_texts: List[str],
    epsilon: float = 1.0,
) -> List[int]:
    """
    Predict RSA‑encrypted hygiene metrics for unseen texts.

    Returns a list of integer ciphertexts (still encrypted).
    """
    X_new = []
    for txt in new_texts:
        v = np.array(extract_features(txt), dtype=float)
        letters = [ch for ch in txt.lower() if "a" <= ch <= "z"]
        freq = np.zeros(26, dtype=float)
        for ch in letters:
            freq[ord(ch) - 97] += 1.0
        if freq.sum() > 0:
            freq /= freq.sum()
        X_new.append(np.concatenate([v, freq]))
    X_new = np.vstack(X_new)
    preds_float = model.predict(X_new, epsilon)
    # Cast predictions back to ints (they should be very close to true ciphertexts)
    return [int(round(val)) for val in preds_float]


def hybrid_decrypt(
    ciphertexts: List[int],
    rsa_priv: Tuple[int, int],
    scale: int = 1_000_000,
) -> List[float]:
    """
    RSA‑decrypt a list of ciphertexts and descale to obtain the original hygiene metric.
    """
    d, n = rsa_priv
    metrics = []
    for c in ciphertexts:
        m_int = rsa_decrypt(c, d, n)
        S = descale_int_to_metric(m_int, scale)
        metrics.append(S)
    return metrics


# ----------------------------------------------------------------------
# Simple RSA key generation (for testing only; NOT cryptographically secure)
# ----------------------------------------------------------------------


def _generate_small_rsa_key(bits: int = 512) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Generate a tiny RSA key pair suitable for demonstration.
    Returns ((e, n), (d, n)).
    """
    # Very small prime generator – deterministic for repeatability
    def _next_prime(start: int) -> int:
        def _is_prime(p: int) -> bool:
            if p < 2:
                return False
            for i in range(2, int(math.isqrt(p)) + 1):
                if p % i == 0:
                    return False
            return True

        p = start
        while not _is_prime(p):
            p += 1
        return p

    random.seed(0)
    # Choose two distinct primes roughly bits/2 size
    p = _next_prime(random.getrandbits(bits // 2))
    q = _next_prime(p + 1)
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 65537
    # Compute modular inverse of e modulo phi
    def egcd(a: int, b: int) -> Tuple[int, int, int]:
        if b == 0:
            return (a, 1, 0)
        g, y, x = egcd(b, a % b)
        return (g, x, y - (a // b) * x)

    g, x, _ = egcd(e, phi)
    if g != 1:
        raise RuntimeError("e and phi are not coprime")
    d = x % phi
    return (e, n), (d, n)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # 1. Generate a toy RSA key pair
    pub_key, priv_key = _generate_small_rsa_key(bits=256)

    # 2. Training corpus (tiny)
    train_texts = [
        "Evidence shows the clean data has low entropy.",
        "Proof of concept: sanitize the system!?",
        "Testimony indicates high information content.",
        "The device was sanitized; no evidence of breach.",
    ]

    # 3. Train the hybrid model
    model = hybrid_train(train_texts, rsa_pub=pub_key, scale=1_000_000, epsilon=0.5)

    # 4. New unseen texts
    test_texts = [
        "Clean evidence suggests no breach.",
        "Entropy is high; data needs sanitization.",
    ]

    # 5. Predict encrypted metrics
    encrypted_preds = hybrid_predict(model, test_texts, epsilon=0.5)

    # 6. Decrypt to obtain hygiene metrics
    recovered_metrics = hybrid_decrypt(encrypted_preds, rsa_priv=priv_key, scale=1_000_000)

    # 7. Display results
    for txt, metric in zip(test_texts, recovered_metrics):
        print(f"Text: {txt}\nRecovered hygiene metric: {metric:.6f}\n")
    sys.exit(0)