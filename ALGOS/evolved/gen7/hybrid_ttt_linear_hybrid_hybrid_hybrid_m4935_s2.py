# DARWIN HAMMER — match 4935, survivor 2
# gen: 7
# parent_a: ttt_linear.py (gen0)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m1940_s0.py (gen6)
# born: 2026-05-29T23:58:58Z

import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable, Iterable, Tuple
import hashlib
import random

# ----------------------------------------------------------------------
# Fixed linguistic categories (unchanged)
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none other some such than that the their them these they this those what which while".split()
    ),
}

# ----------------------------------------------------------------------
# Helper: deterministic pseudo‑random generator based on a seed string
# ----------------------------------------------------------------------
def _seed_from_string(s: str) -> np.random.Generator:
    """Create a NumPy Generator seeded from a string."""
    # Use SHA256 to obtain a 64‑bit integer seed
    h = hashlib.sha256(s.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "little", signed=False)
    return np.random.default_rng(seed)


# ----------------------------------------------------------------------
# Embedding utilities
# ----------------------------------------------------------------------
def _bag_of_words_vector(text: str, vocab: List[str]) -> np.ndarray:
    """Simple bag‑of‑words embedding (counts) using a fixed vocabulary."""
    vec = np.zeros(len(vocab), dtype=np.float32)
    for token in text.lower().split():
        if token in vocab:
            vec[vocab.index(token)] += 1.0
    return vec


def _linguistic_feature_vector(text: str, dim: int = 8) -> np.ndarray:
    """
    Encode the presence of each functional category as a one‑hot vector.
    If a word belongs to multiple categories, the first match wins.
    The vector is padded/truncated to ``dim``.
    """
    vec = np.zeros(dim, dtype=np.float32)
    for i, token in enumerate(text.lower().split()):
        if i >= dim:
            break
        for idx, (cat, words) in enumerate(FUNCTION_CATS.items()):
            if token in words:
                vec[i] = idx + 1  # categories are 1‑based, 0 = “other”
                break
    return vec


def combined_embedding(
    text: str,
    vocab: List[str],
    embed_dim: int = 64,
) -> np.ndarray:
    """
    Produce a single fixed‑size embedding that concatenates:
    1. A bag‑of‑words vector (size ``len(vocab)``)
    2. A linguistic feature vector (size ``embed_dim - len(vocab)``)
    """
    bow = _bag_of_words_vector(text, vocab)
    ling = _linguistic_feature_vector(text, dim=embed_dim - len(vocab))
    return np.concatenate([bow, ling])


# ----------------------------------------------------------------------
# Core Hybrid TTT implementation
# ----------------------------------------------------------------------
@dataclass
class HybridTTT:
    """
    Hybrid TTT (Transform‑Then‑Transform) model that fuses:
    * a regret‑weighted linear predictor,
    * a linguistic similarity component,
    * and a deterministic weight initialization.
    """

    input_dim: int
    output_dim: int | None = None
    lr: float = 0.01
    seed: int = 0
    vocab: List[str] = field(default_factory=lambda: ["hello", "world", "test", "python", "is", "fun", "this", "a", "i"])
    embed_dim: int = 64

    # internal state (filled in __post_init__)
    W: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.output_dim is None:
            self.output_dim = self.input_dim
        rng = np.random.default_rng(self.seed)
        # Small random init – scale chosen to keep early updates stable
        self.W = rng.standard_normal((self.output_dim, self.input_dim)) * 0.01

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------
    def _embed(self, text: str) -> np.ndarray:
        """Return a deterministic embedding of ``text`` with shape (input_dim,)."""
        emb = combined_embedding(text, self.vocab, embed_dim=self.embed_dim)
        # If the embedding is larger than ``input_dim`` we truncate,
        # otherwise we pad with zeros.
        if emb.shape[0] > self.input_dim:
            return emb[: self.input_dim]
        elif emb.shape[0] < self.input_dim:
            pad = np.zeros(self.input_dim - emb.shape[0], dtype=emb.dtype)
            return np.concatenate([emb, pad])
        return emb

    def _linguistic_score(self, a: str, b: str) -> float:
        """Cosine similarity of the linguistic feature vectors."""
        fa = _linguistic_feature_vector(a, dim=self.embed_dim - len(self.vocab))
        fb = _linguistic_feature_vector(b, dim=self.embed_dim - len(self.vocab))
        if np.all(fa == 0) or np.all(fb == 0):
            return 0.0
        return float(np.dot(fa, fb) / (np.linalg.norm(fa) * np.linalg.norm(fb)))

    def _regret_score(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Squared error of the linear predictor ``W`` on ``x`` with target ``y``.
        This is the classic regret‑weighted term.
        """
        pred = self.W @ x
        return float(np.linalg.norm(pred - y) ** 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def step(self, x_text: str, y_text: str) -> None:
        """
        Perform a single hybrid update.
        ``x_text`` – current observation,
        ``y_text`` – desired target (often the same as ``x_text`` for reconstruction).
        """
        x = self._embed(x_text)
        y = self._embed(y_text)

        # Compute scores
        ling_score = self._linguistic_score(x_text, y_text)
        regret = self._regret_score(x, y)

        # Gradient‑like update that respects both components.
        # The linguistic term modulates the direction, the regret term scales magnitude.
        grad = (self.W @ x - y) + ling_score * (self.W @ x - x)
        self.W -= self.lr * (regret + ling_score) * grad

    def fit(self, texts: Iterable[str]) -> None:
        """
        Train on a sequence of texts using self‑reconstruction as the target.
        """
        for txt in texts:
            self.step(txt, txt)

    def predict(self, text: str) -> np.ndarray:
        """Return the linear prediction for ``text``."""
        x = self._embed(text)
        return self.W @ x

    def similarity(self, a: str, b: str) -> float:
        """
        Hybrid similarity that blends linguistic cosine similarity with
        regret‑weighted distance, yielding a value in [0, 1].
        """
        ling = self._linguistic_score(a, b)
        x = self._embed(a)
        y = self._embed(b)
        regret = self._regret_score(x, y)
        # Normalise regret to (0, 1] using a smooth exponential map
        regret_norm = 1.0 - np.exp(-regret)
        return (ling + (1.0 - regret_norm)) / 2.0

# ----------------------------------------------------------------------
# Example usage (kept minimal for reproducibility)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Deterministic dimensions: we keep them modest for the demo.
    model = HybridTTT(input_dim=64, seed=42)

    corpus = [
        "hello world",
        "this is a test",
        "python is fun",
        "i love python",
        "the quick brown fox jumps over the lazy dog",
    ]

    model.fit(corpus)

    # Show a few predictions and similarities
    for s in corpus[:3]:
        pred = model.predict(s)
        print(f"Input: {s!r}")
        print(f"Prediction norm: {np.linalg.norm(pred):.4f}")

    sim = model.similarity("hello world", "python is fun")
    print(f"\nHybrid similarity between 'hello world' and 'python is fun': {sim:.4f}")