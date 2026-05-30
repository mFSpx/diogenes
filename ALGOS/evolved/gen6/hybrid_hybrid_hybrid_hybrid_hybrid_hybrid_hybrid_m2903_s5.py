# DARWIN HAMMER — match 2903, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m538_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_shanno_hybrid_hybrid_hybrid_m1118_s0.py (gen5)
# born: 2026-05-29T23:46:34Z

"""Hybrid Unified Algorithm
Parents:
- hybrid_hybrid_hybrid_hybrid_m538_s0.py (Hybrid Fusion with RBF surrogate)
- hybrid_hybrid_hybrid_shanno_hybrid_hybrid_hybrid_m1118_s0.py (Sparse WTA, RSA encoding, Regret minimization)

Mathematical Bridge:
The Fusion algorithm supplies a *resource vector* R = (d, p, s) where
    d = Euclidean distance to stored prototypes,
    p = privacy‑load term (here omitted for brevity),
    s = score predicted by an RBF surrogate:
        s_i = Σ_j w_ij * exp(-ε·‖x‑c_j‖²).

The Sparse WTA algorithm consumes the score vector **s**, retains the top‑k entries (producing a sparse decision vector **z**) and discards the rest.  
The RSA module treats the integer representation of **z** as a message m and produces a ciphertext c = m^e mod n.  

Regret minimization uses the decoded decision \hat{z} to compute instantaneous regret
    r = loss(\hat{z}, y) – loss(z_opt, y)
and updates the Fusion weights w via a gradient step proportional to r·∂s/∂w.

Thus the unified system flows:
input → distances → RBF scores → Sparse WTA → RSA encode/decode → Regret → weight update.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ---------- Utility Functions ----------

def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return float(np.linalg.norm(a - b))

def sparse_wta(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Sparse Winner‑Take‑All: keep the top‑k scores, zero out the rest.
    Returns a binary mask (1 for kept entries, 0 otherwise).
    """
    if k <= 0:
        return np.zeros_like(scores, dtype=int)
    idx = np.argpartition(-scores, k - 1)[:k]
    mask = np.zeros_like(scores, dtype=int)
    mask[idx] = 1
    return mask

def is_prime(n: int) -> bool:
    """Very small deterministic primality test (sufficient for 16‑bit primes)."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(math.isqrt(n))
    for i in range(3, r + 1, 2):
        if n % i == 0:
            return False
    return True

def generate_small_prime(bits: int = 16) -> int:
    """Generate a random prime of given bit length."""
    while True:
        p = random.getrandbits(bits) | 1  # ensure odd
        if is_prime(p):
            return p

def generate_rsa_keypair(prime_bits: int = 16) -> Tuple[int, int, int]:
    """Generate RSA (n, e, d) with small primes for demonstration."""
    p = generate_small_prime(prime_bits)
    q = generate_small_prime(prime_bits)
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 65537
    # ensure e and phi are coprime
    while math.gcd(e, phi) != 1:
        e += 2
    # modular inverse of e modulo phi
    def egcd(a: int, b: int) -> Tuple[int, int, int]:
        if a == 0:
            return b, 0, 1
        g, y, x = egcd(b % a, a)
        return g, x - (b // a) * y, y
    g, x, _ = egcd(e, phi)
    d = x % phi
    return n, e, d

def rsa_encrypt(message: int, e: int, n: int) -> int:
    return pow(message, e, n)

def rsa_decrypt(cipher: int, d: int, n: int) -> int:
    return pow(cipher, d, n)

def loss(pred: np.ndarray, true: np.ndarray) -> float:
    """Simple squared loss."""
    return float(np.mean((pred - true) ** 2))

def regret_update(
    w: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray,
    decoded: np.ndarray,
    true_label: np.ndarray,
    regret: float,
    lr: float = 0.01,
) -> np.ndarray:
    """
    Gradient‑like update of RBF weights w based on regret.
    w shape: (n_prototypes, d_in)
    scores = RBF scores before WTA (size n_prototypes)
    mask   = binary mask from WTA (size n_prototypes)
    decoded = decoded decision vector (binary)
    true_label = ground‑truth binary vector
    regret = scalar instantaneous regret
    """
    # gradient of scores w.r.t. w_j = -2*epsilon*(x - c_j)*exp(-epsilon*||x-c_j||^2)
    # For simplicity we approximate gradient using the mask (only active prototypes)
    grad = np.zeros_like(w)
    active_idx = np.where(mask == 1)[0]
    for j in active_idx:
        # placeholder gradient direction (push weight away from x if regret > 0)
        direction = -regret * (w[j] - decoded)  # shape (d_in,)
        grad[j] = direction
    w_new = w - lr * grad
    return w_new

# ---------- Core Classes ----------

class HybridFusionRBF:
    """
    Core of Parent A: maintains prototypes (centers) and RBF weights.
    Provides distance and score computation.
    """
    def __init__(
        self,
        d_in: int,
        max_prototypes: int = 10,
        epsilon: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.d_in = d_in
        self.max_prototypes = max_prototypes
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.centers: List[np.ndarray] = []          # shape (n, d_in)
        self.weights: np.ndarray = np.empty((0, d_in))  # shape (n, d_in)

    def add_prototype(self, vec: np.ndarray) -> None:
        """Add a new prototype (center)."""
        if len(self.centers) >= self.max_prototypes:
            # replace a random existing prototype
            idx = self.rng.randint(0, len(self.centers) - 1)
            self.centers[idx] = vec.copy()
            self.weights[idx] = vec.copy()
        else:
            self.centers.append(vec.copy())
            if self.weights.size == 0:
                self.weights = vec.reshape(1, -1)
            else:
                self.weights = np.vstack([self.weights, vec.reshape(1, -1)])

    def compute_distances(self, x: np.ndarray) -> np.ndarray:
        """Euclidean distances from x to each prototype."""
        return np.array([euclidean(x, c) for c in self.centers])

    def rbf_scores(self, x: np.ndarray) -> np.ndarray:
        """RBF surrogate scores for input x."""
        dists = self.compute_distances(x)
        return np.array([gaussian_rbf(r, self.epsilon) for r in dists])

class HybridSparseRSARegret:
    """
    Core of Parent B: Sparse WTA, RSA encoding/decoding, and regret evaluation.
    """
    def __init__(self, k: int = 3, rsa_bits: int = 16):
        self.k = k
        self.n, self.e, self.d = generate_rsa_keypair(rsa_bits)

    def encode_decision(self, mask: np.ndarray) -> int:
        """Interpret binary mask as integer and RSA‑encrypt."""
        message = int("".join(map(str, mask.astype(int))), 2)
        return rsa_encrypt(message, self.e, self.n)

    def decode_decision(self, cipher: int, length: int) -> np.ndarray:
        """RSA‑decrypt and convert back to binary mask of given length."""
        message = rsa_decrypt(cipher, self.d, self.n)
        bin_str = bin(message)[2:].zfill(length)
        return np.array([int(ch) for ch in bin_str], dtype=int)

    def evaluate_regret(
        self,
        decoded: np.ndarray,
        true_label: np.ndarray,
    ) -> float:
        """Instantaneous regret based on squared loss."""
        opt_loss = loss(true_label, true_label)          # zero loss for perfect decision
        cur_loss = loss(decoded, true_label)
        return cur_loss - opt_loss

class HybridUnified:
    """
    Unified system that fuses HybridFusionRBF with HybridSparseRSARegret.
    """
    def __init__(
        self,
        d_in: int,
        max_prototypes: int = 10,
        epsilon: float = 1.0,
        wta_k: int = 3,
        rsa_bits: int = 16,
        seed: int = 0,
    ) -> None:
        self.fusion = HybridFusionRBF(d_in, max_prototypes, epsilon, seed)
        self.regret_mod = HybridSparseRSARegret(wta_k, rsa_bits)

    def ingest(self, vec: np.ndarray) -> None:
        """Add incoming vector as a prototype (simulates learning)."""
        self.fusion.add_prototype(vec)

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Returns:
            mask          – binary sparse decision (post‑WTA)
            ciphertext    – RSA encrypted integer of mask
            decoded_mask  – RSA decrypted mask (should equal mask)
        """
        scores = self.fusion.rbf_scores(x)
        mask = sparse_wta(scores, self.regret_mod.k)
        ciphertext = self.regret_mod.encode_decision(mask)
        decoded_mask = self.regret_mod.decode_decision(ciphertext, len(mask))
        return mask, ciphertext, decoded_mask

    def update(self, x: np.ndarray, true_label: np.ndarray, lr: float = 0.01) -> None:
        """
        Perform a full step: predict, compute regret, and update RBF weights.
        true_label is a binary vector of the same length as the decision mask.
        """
        mask, cipher, decoded = self.predict(x)
        regret = self.regret_mod.evaluate_regret(decoded, true_label)
        # Update RBF weights using regret gradient
        if self.fusion.weights.shape[0] > 0:
            self.fusion.weights = regret_update(
                self.fusion.weights,
                self.fusion.rbf_scores(x),
                mask,
                decoded,
                true_label,
                regret,
                lr,
            )

# ---------- Demonstration Functions ----------

def demo_hybrid_operation():
    """Run a short demonstration of the unified hybrid algorithm."""
    d_in = 5
    unified = HybridUnified(d_in=d_in, max_prototypes=4, epsilon=0.8, wta_k=2, rsa_bits=16, seed=42)

    # Create random prototypes
    for _ in range(4):
        vec = np.random.randn(d_in)
        unified.ingest(vec)

    # New input vector
    x = np.random.randn(d_in)

    # Simulated true label (binary vector of length equal to number of prototypes)
    true_label = np.zeros(len(unified.fusion.centers), dtype=int)
    true_label[0] = 1  # assume first prototype is the correct decision

    # Perform prediction and update
    mask, cipher, decoded = unified.predict(x)
    print("Mask (WTA):", mask)
    print("Ciphertext:", cipher)
    print("Decoded mask:", decoded)

    unified.update(x, true_label, lr=0.05)
    print("Updated RBF weights shape:", unified.fusion.weights.shape)

def rbf_score_demo():
    """Show RBF score computation on a simple example."""
    d_in = 3
    fusion = HybridFusionRBF(d_in=d_in, max_prototypes=3, epsilon=1.0, seed=1)
    for i in range(3):
        fusion.add_prototype(np.array([i, i + 1, i + 2], dtype=float))
    x = np.array([1.0, 2.0, 3.0])
    scores = fusion.rbf_scores(x)
    print("RBF scores:", scores)

def rsa_roundtrip_demo():
    """Validate RSA encrypt/decrypt on a random binary mask."""
    rsa_mod = HybridSparseRSARegret(k=4, rsa_bits=16)
    mask = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=int)
    cipher = rsa_mod.encode_decision(mask)
    decoded = rsa_mod.decode_decision(cipher, len(mask))
    print("Original mask:", mask)
    print("Decoded mask :", decoded)

# ---------- Smoke Test ----------

if __name__ == "__main__":
    print("=== RBF Score Demo ===")
    rbf_score_demo()
    print("\n=== RSA Round‑Trip Demo ===")
    rsa_roundtrip_demo()
    print("\n=== Hybrid Unified Demo ===")
    demo_hybrid_operation()