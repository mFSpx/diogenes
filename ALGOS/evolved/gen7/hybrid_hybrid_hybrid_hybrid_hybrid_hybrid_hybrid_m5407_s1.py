# DARWIN HAMMER — match 5407, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hard_t_ternary_router_m906_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2494_s6.py (gen6)
# born: 2026-05-30T00:01:42Z

"""Hybrid Algorithm: Fusion of DARWIN HAMMER's Text Stylometry (Parent A) and
Count‑Min Sketch / Multivector Koopman Dynamics (Parent B)

Mathematical Bridge
-------------------
* Parent A* provides a high‑dimensional sparse representation **v**∈ℝⁿ
  derived from linguistic function‑category counts and punctuation frequencies.
* *Parent B* interprets a Count‑Min sketch **S** as a multivector **m**∈ℝᵈ by
  flattening the sketch table (each bucket becomes a basis‑blade coefficient).
* The bridge is a **bilinear projection**  B∈ℝ^{d×n} that maps the text vector
  into the multivector space: **m₀ = B · v**.  The resulting multivector is then
  evolved in time by a learned Koopman operator **K**∈ℝ^{d×d},
  producing **m₁ = K · m₀**.  A linear decoder **D**∈ℝ^{p×d} projects the
  evolved multivector back to a low‑dimensional “morphology” space
  **y = D · m₁**.  A variational free‑energy (VFE) term combines the
  reconstruction error ‖y − ŷ‖² with a KL‑divergence between the Bayesian
  Beta posterior of each sketch bucket and a uniform prior, yielding a scalar
  quality score for the hybrid pipeline.

The implementation below contains the core pipeline:
1. Text → vector (Parent A)
2. Vector → Count‑Min sketch update
3. Sketch → multivector (flatten)
4. Bilinear projection **B**
5. Koopman evolution **K**
6. Decoding **D**
7. Bayesian Beta update per bucket
8. Variational free‑energy evaluation
"""

import math
import random
import sys
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Parent A – stylometric vectorisation
# ----------------------------------------------------------------------
FUNCTION_CATS = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*"

def text_to_vector(text: str) -> np.ndarray:
    """Return a sparse count vector of length |FUNCTION_CATS|+|PUNCT|."""
    vec = np.zeros(len(FUNCTION_CATS) + len(PUNCT), dtype=np.float64)
    words = text.lower().split()
    # category counts
    for i, (cat, wordset) in enumerate(FUNCTION_CATS.items()):
        vec[i] = sum(1 for w in words if w in wordset)
    # punctuation counts
    for j, ch in enumerate(PUNCT):
        vec[len(FUNCTION_CATS) + j] = text.count(ch)
    return vec

# ----------------------------------------------------------------------
# Parent B – Count‑Min sketch interpreted as a multivector
# ----------------------------------------------------------------------
class CountMinSketch:
    def __init__(self, depth: int = 4, width: int = 256, seed: int = 0):
        self.depth = depth
        self.width = width
        self.table = np.zeros((depth, width), dtype=np.int64)
        rng = random.Random(seed)
        self.salts = [rng.randint(1, 2**31 - 1) for _ in range(depth)]

    def _hash(self, item: any, i: int) -> int:
        h = hash((item, self.salts[i]))
        return h % self.width

    def update(self, item: any, inc: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(item, i)
            self.table[i, idx] += inc

    def query(self, item: any) -> int:
        """Return the minimum count estimate for *item*."""
        mins = []
        for i in range(self.depth):
            idx = self._hash(item, i)
            mins.append(self.table[i, idx])
        return min(mins)

    def flatten(self) -> np.ndarray:
        """Interpret the sketch as a multivector by flattening."""
        return self.table.ravel().astype(np.float64)

# ----------------------------------------------------------------------
# Bayesian Beta parameters per bucket (simple independent priors)
# ----------------------------------------------------------------------
class BetaPosterior:
    """Maintain independent Beta(a,b) for each sketch bucket."""
    def __init__(self, shape: tuple, prior_a: float = 1.0, prior_b: float = 1.0):
        self.a = np.full(shape, prior_a, dtype=np.float64)
        self.b = np.full(shape, prior_b, dtype=np.float64)

    def update(self, counts: np.ndarray) -> None:
        """Increment posterior with observed counts (treated as successes)."""
        self.a += counts
        self.b += 1.0  # each observation adds one failure pseudo‑count

    def kl_to_uniform(self) -> float:
        """KL( Beta(a,b) || Beta(1,1) ) summed over all buckets."""
        a, b = self.a, self.b
        term1 = (a - 1) * (np.log(a) - np.log(a + b))
        term2 = (b - 1) * (np.log(b) - np.log(a + b))
        term3 = - (np.log(math.gamma(a)) + np.log(math.gamma(b)) - np.log(math.gamma(a + b)))
        term4 = np.log(math.gamma(2)) - (np.log(math.gamma(1)) + np.log(math.gamma(1)))  # KL to Uniform(1,1)
        return np.sum(term1 + term2 + term3 + term4)

# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------
def bilinear_projection(B: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Project text vector v (size n) into multivector space (size d) via B (d×n)."""
    return B @ v

def learn_koopman(X: np.ndarray, X_next: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """
    Least‑squares estimate of Koopman operator K solving K·X ≈ X_next.
    X, X_next are d×N matrices (N samples).
    """
    Xt = X.T
    X_next_t = X_next.T
    # K = X_next * X.T * (X * X.T + reg*I)^-1
    cov = X @ Xt
    inv = np.linalg.inv(cov + reg * np.eye(cov.shape[0]))
    K = X_next @ Xt @ inv
    return K

def variational_free_energy(recon_error: float, beta_post: BetaPosterior) -> float:
    """Combine reconstruction error with KL divergence of Beta posteriors."""
    kl = beta_post.kl_to_uniform()
    return recon_error + kl

# ----------------------------------------------------------------------
# Demonstration pipeline (three public functions)
# ----------------------------------------------------------------------
def hybrid_initialize(vector_dim: int, sketch_depth: int = 4, sketch_width: int = 256,
                     seed: int = 42) -> dict:
    """
    Initialise all structures required for the hybrid pipeline.
    Returns a dictionary with keys:
    - 'B' : bilinear projection matrix (d×n)
    - 'sketch' : CountMinSketch instance
    - 'beta' : BetaPosterior instance matching sketch shape
    - 'decoder' : linear decoder D (p×d)
    """
    rng = np.random.default_rng(seed)
    d = sketch_depth * sketch_width
    B = rng.normal(scale=0.01, size=(d, vector_dim)).astype(np.float64)
    sketch = CountMinSketch(depth=sketch_depth, width=sketch_width, seed=seed)
    beta = BetaPosterior(shape=(sketch_depth, sketch_width))
    p = 32  # dimensionality of decoded morphology space
    D = rng.normal(scale=0.01, size=(p, d)).astype(np.float64)
    return {'B': B, 'sketch': sketch, 'beta': beta, 'decoder': D}

def hybrid_step(text: str, state: dict, K: np.ndarray = None) -> tuple:
    """
    Execute one hybrid iteration:
    1. Convert *text* → vector v.
    2. Update sketch with each token (as hashable items).
    3. Flatten sketch → multivector m_raw.
    4. Project v via B → m_proj and add to m_raw (fusion).
    5. If a Koopman K is supplied, evolve the fused multivector.
    6. Decode to morphology space y.
    7. Update Beta posterior and compute VFE.
    Returns (y, vfe, m_fused).
    """
    # 1. text → vector
    v = text_to_vector(text)

    # 2. sketch update (each word as an item)
    for token in text.split():
        state['sketch'].update(token, inc=1)

    # 3. raw multivector from sketch
    m_raw = state['sketch'].flatten()

    # 4. bilinear projection
    m_proj = bilinear_projection(state['B'], v)

    # Fusion (simple addition)
    m_fused = m_raw + m_proj

    # 5. Koopman evolution (optional)
    if K is not None:
        m_evolved = K @ m_fused
    else:
        m_evolved = m_fused

    # 6. Decode
    y = state['decoder'] @ m_evolved

    # 7. Bayesian update & VFE
    counts = state['sketch'].table.astype(np.float64)
    state['beta'].update(counts)
    recon_error = np.mean((y - np.mean(y))**2)  # simple variance as proxy
    vfe = variational_free_energy(recon_error, state['beta'])

    return y, vfe, m_fused

def hybrid_train_koopman(samples: list, state: dict) -> np.ndarray:
    """
    Given a list of texts, build paired multivector samples (t, t+1)
    and learn a Koopman operator K.
    """
    d = state['sketch'].depth * state['sketch'].width
    X = np.zeros((d, len(samples) - 1), dtype=np.float64)
    X_next = np.zeros_like(X)

    # Populate sketch sequentially, capturing multivector after each text
    for idx, txt in enumerate(samples):
        # update sketch with current text
        for token in txt.split():
            state['sketch'].update(token, inc=1)
        m = state['sketch'].flatten()
        if idx > 0:
            X_next[:, idx - 1] = m
        if idx < len(samples) - 1:
            X[:, idx] = m

    # Learn K
    K = learn_koopman(X, X_next)
    return K

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "I love the quick brown fox!",
        "She does not like the lazy dog.",
        "They will run, and they will win."
    ]

    # Initialise hybrid system
    vec_dim = len(FUNCTION_CATS) + len(PUNCT)
    state = hybrid_initialize(vector_dim=vec_dim, sketch_depth=4, sketch_width=64, seed=123)

    # Train a Koopman operator on the sample sequence
    K = hybrid_train_koopman(sample_texts, state)

    # Run a hybrid step on a new sentence
    test_sentence = "We can achieve great things together."
    y, vfe, m = hybrid_step(test_sentence, state, K=K)

    print("Decoded morphology vector (shape {}):".format(y.shape))
    print(y)
    print("\nVariational Free Energy:", vfe)
    print("\nFused multivector norm:", np.linalg.norm(m))