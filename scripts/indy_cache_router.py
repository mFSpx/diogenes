```python
"""indy_cache_router.py – a tiny Anthropic prompt‑cache router.

The module defines:
* STATIC_PREFIX – a constant string that is cached once.
* CacheRouter – predicts cacheability and routes tasks using River.
* build_payload – creates an Anthropic‑compatible request payload.
"""

import json, os, time, hashlib
from river.linear_model import LogisticRegression
from river.optim import FTRL
from river.bandit import Exp3

# --------------------------------------------------------------------------- #
# 1.  Cached static prefix (persona stamp + GO‑25 + IO‑25 terms)
# --------------------------------------------------------------------------- #
STATIC_PREFIX = (
    "You are a helpful, knowledgeable AI assistant. "
    "Ontology GO-25: metabolic process, signal transduction, cellular component organization. "
    "IO-25: input‑output handling, error recovery, state persistence."
)

# --------------------------------------------------------------------------- #
# 2.  CacheRouter
# --------------------------------------------------------------------------- #
class CacheRouter:
    """Predict cache hits and route tasks to groq / local / self."""

    _log_path = "/tmp/lucidota_cache_route.jsonl"
    _arms = ("groq", "local", "self")
    _cache_thr = 0.217

    def __init__(self):
        # Logistic regression with FTRL (l2=0.01)
        self.clf = LogisticRegression(optimizer=FTRL(l2=0.01))
        # Exp3 bandit with 3 arms
        self.bandit = Exp3(n_arms=3, gamma=0.1)

    # ------------------------------------------------------------------- #
    # 2.1  Cache prediction
    # ------------------------------------------------------------------- #
    def _features(self, context_hash: str, token_count: int) -> dict:
        """Simple numeric features for the models."""
        # deterministic integer from the hash string
        h = int(hashlib.sha256(context_hash.encode()).hexdigest(), 16) % 1_000_000
        return {"hash_mod": h, "tokens": token_count}

    def predict_cache(self, context_hash: str, token_count: int) -> bool:
        """Return True if the model estimates a cache‑hit probability > 0.217."""
        feats = self._features(context_hash, token_count)
        #