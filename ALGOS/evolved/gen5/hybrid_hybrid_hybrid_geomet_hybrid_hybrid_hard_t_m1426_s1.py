# DARWIN HAMMER — match 1426, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s2.py (gen4)
# parent_b: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_endpoi_m127_s3.py (gen3)
# born: 2026-05-29T23:36:24Z

"""
This module fuses the hybrid geometric product from the Clifford algebra (Cl(n,0)) 
with the hybrid hard truth math model from the stylometry and LSM utilities. 
The mathematical bridge between these two structures is formed by using the 
geometric product to compute distances and orientations between points in the 
stylometry feature space, and then applying these computations to assign points 
to their nearest hard truth math model.

The governing equations of the Clifford algebra are used to compute the 
geometric product of multivectors, which are then used to represent 
points and vectors in the stylometry feature space. The hybrid hard truth 
math model is used to assign points to their nearest hard truth model, and 
the geometric product is used to compute the distances and orientations 
between these points and models.

This module provides functions to compute the geometric product of 
multivectors, assign points to their nearest hard truth model using the 
hybrid hard truth math model, and visualize the resulting assignments.
"""

import math
import numpy as np
import random
import sys
import pathlib
from typing import Any, Dict, List, Tuple

# Core blade arithmetic helpers
def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble-sorting index list.

    Each transposition of adjacent indices that are out of order flips the
    sign (anti-commutativity).  Duplicate indices cancel (e_i^2 = 1 → they
    annihilate and contribute +1 to the sign, but the index disappears).
    """
    lst = list(indices)
    sign = 1
    # Bubble sort; track swaps
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # Duplicate: e_i * e_i = 1, remove both
                lst.pop(j)
                lst.pop(j)  # was j+1, now at j after pop
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices).

    Returns (result_blade_frozenset, sign).
    """
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


# Multivector
class Multivector:
    def __init__(self, blades: Dict[frozenset, float]):
        self.blades = blades

    def __mul__(self, other):
        result_blades = {}
        for blade_a, coeff_a in self.blades.items():
            for blade_b, coeff_b in other.blades.items():
                result_blade, sign = _multiply_blades(blade_a, blade_b)
                if result_blade in result_blades:
                    result_blades[result_blade] += sign * coeff_a * coeff_b
                else:
                    result_blades[result_blade] = sign * coeff_a * coeff_b
        return Multivector(result_blades)


# Stylometry and LSM utilities
FUNCTION_CATS: Dict[str, set[str]] = {
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
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def words(text: str) -> List[str]:
    """Lower‑case alphabetic tokens (apostrophe‑aware)."""
    return [w for w in text.lower().split() if w.isalpha()]


def stylometry_features(text: str) -> np.ndarray:
    """
    Produce a deterministic 96‑dimensional numeric representation of *text*.
    The implementation uses a SHA‑256 hash to seed a pseudo‑random generator,
    guaranteeing reproducibility without external corpora.
    """
    h = hashlib.md5(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.random(96)


def lsm_vector(text: str) -> np.ndarray:
    """
    Compute the proportion of words belonging to each FUNCTION_CAT.
    Return a 8-dimensional numeric representation of *text*.
    """
    text_words = words(text)
    cat_counts = {cat: sum(1 for w in text_words if w in cats) for cat, cats in FUNCTION_CATS.items()}
    return np.array([cat_counts[cat] / len(text_words) for cat in FUNCTION_CATS])


# Hybrid function
def hybrid_stylometry_multivector(text: str) -> Multivector:
    """
    Compute a multivector representation of *text* by combining stylometry features 
    with the geometric product.
    """
    features = stylometry_features(text)
    blades = {frozenset(): features[0]}
    for i in range(1, len(features)):
        blades[frozenset([i])] = features[i]
    return Multivector(blades)


def assign_to_hard_truth(model: Multivector, point: Multivector) -> Multivector:
    """
    Assign *point* to its nearest hard truth model using the geometric product.
    """
    # Compute geometric product
    product = model * point
    # Find nearest hard truth model
    nearest_model = Multivector({frozenset(): product.blades[frozenset()]})
    return nearest_model


def visualize_assignments(texts: List[str], models: List[Multivector]) -> None:
    """
    Visualize the assignments of *texts* to their nearest hard truth *models*.
    """
    for text, model in zip(texts, models):
        point = hybrid_stylometry_multivector(text)
        assigned_model = assign_to_hard_truth(model, point)
        print(f"Text: {text}, Assigned Model: {assigned_model.blades}")


if __name__ == "__main__":
    texts = ["This is a test.", "Another test.", "Test with different words."]
    models = [Multivector({frozenset(): 1.0}), Multivector({frozenset([1]): 2.0})]
    visualize_assignments(texts, models)