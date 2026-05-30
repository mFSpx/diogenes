# DARWIN HAMMER — match 1293, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_geomet_m18_s0.py (gen3)
# parent_b: hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s3.py (gen4)
# born: 2026-05-29T23:35:00Z

"""Hybrid Stylometry–SSIM Fusion Module

Parents:
- hybrid_hybrid_hard_truth_ma_hybrid_hybrid_geomet_m18_s0.py (stylometric feature extraction,
  geometric representation, Voronoi‑like clustering)
- hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s3.py (Structural Similarity Index
  Measure (SSIM) and ternary routing logic)

Mathematical Bridge:
Stylometric categories are mapped to a high‑dimensional feature vector **v**∈ℝⁿ.
We reshape **v** into a square matrix **I**∈ℝ^{k×k} (zero‑padding if necessary) and treat it as a
synthetic “image”.  The SSIM function from the second parent operates on two such
matrices, yielding a similarity score that quantifies stylistic proximity.  This score
feeds the routing decision and can also be used as a regularisation term in a linear
model loss (ttt_loss).  Thus the core operations of both parents are fused:
stylometric → geometric matrix → SSIM → routing / loss.
"""

import sys
import math
import random
import pathlib
import re
import json
from datetime import datetime, timezone
from collections import Counter
import numpy as np

# ----------------------------------------------------------------------
# Stylometric utilities (Parent A)
# ----------------------------------------------------------------------
FUNCTION_CATS = {
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
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

def words(text: str):
    """Return a list of lowercase words (ASCII letters + optional apostrophe)."""
    # Replace punctuation with spaces, keep apostrophes inside words
    cleaned = re.sub(rf"[{re.escape(PUNCT)}]", " ", text)
    tokens = re.findall(r"\b[a-zA-Z]+'?[a-zA-Z]*\b", cleaned)
    return [t.lower() for t in tokens]

def stylometric_vector(text: str) -> np.ndarray:
    """
    Produce a normalized feature vector whose components correspond to the
    relative frequencies of each FUNCTION_CATS category.
    """
    token_list = words(text)
    total = len(token_list) or 1
    cat_counts = []
    token_set = set(token_list)
    for cat, vocab in FUNCTION_CATS.items():
        count = sum(1 for w in token_set if w in vocab)
        cat_counts.append(count / total)
    return np.array(cat_counts, dtype=float)

# ----------------------------------------------------------------------
# SSIM and routing utilities (Parent B)
# ----------------------------------------------------------------------
def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index Measure for two equally‑shaped arrays."""
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    ssim_map = ((2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)) * \
               ((2 * sigma_xy + C2) / (sigma_x ** 2 + sigma_y ** 2 + C2))
    return float(ssim_map)

def vector_to_image(vec: np.ndarray) -> np.ndarray:
    """
    Reshape a 1‑D feature vector into a square 2‑D array (synthetic image).
    Zero‑pad if necessary.
    """
    n = vec.size
    k = int(math.ceil(math.sqrt(n)))
    padded = np.zeros(k * k, dtype=vec.dtype)
    padded[:n] = vec
    return padded.reshape((k, k))

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def stylometric_ssim(text_a: str, text_b: str) -> float:
    """
    Compute SSIM between two texts by converting their stylometric vectors
    into square matrices.
    """
    vec_a = stylometric_vector(text_a)
    vec_b = stylometric_vector(text_b)
    img_a = vector_to_image(vec_a)
    img_b = vector_to_image(vec_b)
    # Scale to typical image dynamic range for SSIM stability
    scale = 255.0 / max(img_a.max(), img_b.max(), 1e-8)
    return ssim(img_a * scale, img_b * scale, dynamic_range=255.0)

def ttt_loss_hybrid(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> float:
    """
    Linear prediction loss (Parent B) enriched with a stylometric regulariser.
    The regulariser penalises deviation from a target stylometric similarity.
    """
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    mse = np.mean(residual ** 2)

    # Optional stylometric regularisation: compare predicted vector's style to original
    pred_style = stylometric_vector(" ".join(map(str, pred)))
    orig_style = stylometric_vector(" ".join(map(str, x)))
    style_sim = stylometric_ssim(
        " ".join(map(str, pred_style)),
        " ".join(map(str, orig_style))
    )
    # Convert similarity (0..1) to a penalty (higher similarity → lower penalty)
    style_penalty = (1.0 - style_sim)
    return float(mse + 0.1 * style_penalty)

def route_packet_hybrid(packet: dict) -> dict:
    """
    Hybrid routing: use stylometric SSIM between the incoming text and a
    reference catalogue to decide the outbound channel.
    """
    text = str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or "")
    # Simple reference corpus (could be loaded from file; here we hard‑code)
    reference_texts = {
        "formal": "Thus, the experiment demonstrates a statistically significant improvement.",
        "informal": "Yo! That thing was pretty cool, ain't it?",
    }
    # Compute similarity to each reference
    sims = {k: stylometric_ssim(text, v) for k, v in reference_texts.items()}
    best_label = max(sims, key=sims.get)
    route = {
        "engine_channel": f"cpu_fairyfuse_{best_label}",
        "similarity_score": sims[best_label],
        "outbound_state": "draft_only",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    return route

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_a = "I think that the results are quite promising and could lead to further research."
    sample_b = "Yo! That thing was pretty cool, ain't it?"
    print("Stylometric SSIM:", stylometric_ssim(sample_a, sample_b))

    # Linear model demo
    dim = stylometric_vector(sample_a).size
    W = np.eye(dim) + 0.01 * np.random.randn(dim, dim)
    x = stylometric_vector(sample_a)
    loss = ttt_loss_hybrid(W, x)
    print("Hybrid TTT loss:", loss)

    # Routing demo
    pkt = {"text_surface": sample_a}
    route = route_packet_hybrid(pkt)
    print("Routing decision:", route)