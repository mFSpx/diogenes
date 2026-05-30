# DARWIN HAMMER — match 1426, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s2.py (gen4)
# parent_b: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_endpoi_m127_s3.py (gen3)
# born: 2026-05-29T23:36:24Z

import math
import random
import sys
import pathlib
from typing import Any, Dict, List, Tuple, Iterable
import numpy as np

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
        "no not never none neither cannot cant wont dont didnt isnt arent was wasnt werent".split()
    ),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only quite rather too".split()
    ),
}

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = i
        while j < n - 1:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                del lst[j : j + 2]
                n -= 2
                i = -1  # restart outer loop
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(
    blade_a: frozenset, blade_b: frozenset
) -> Tuple[frozenset, int]:
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    def __init__(self, components: Dict[frozenset, float] = None):
        self.components: Dict[frozenset, float] = {}
        if components:
            for b, v in components.items():
                if abs(v) > 1e-15:
                    self.components[frozenset(b)] = float(v)

    @staticmethod
    def scalar(value: float) -> "Multivector":
        return Multivector({frozenset(): value})

    @staticmethod
    def vector(coeffs: Iterable[float]) -> "Multivector":
        comp = {}
        for i, c in enumerate(coeffs):
            if abs(c) > 1e-15:
                comp[frozenset({i})] = float(c)
        return Multivector(comp)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = self.components.copy()
        for b, v in other.components.items():
            result[b] = result.get(b, 0.0) + v
            if abs(result[b]) < 1e-15:
                del result[b]
        return Multivector(result)

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __neg__(self) -> "Multivector":
        return Multivector({b: -v for b, v in self.components.items()})

    def __mul__(self, other: "Multivector") -> "Multivector":
        result: Dict[frozenset, float] = {}
        for ba, va in self.components.items():
            for bb, vb in other.components.items():
                bc, sign = _multiply_blades(ba, bb)
                coeff = va * vb * sign
                result[bc] = result.get(bc, 0.0) + coeff
        result = {b: v for b, v in result.items() if abs(v) > 1e-15}
        return Multivector(result)

    def grade(self, k: int) -> "Multivector":
        comp = {b: v for b, v in self.components.items() if len(b) == k}
        return Multivector(comp)

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def reverse(self) -> "Multivector":
        comp = {}
        for b, v in self.components.items():
            k = len(b)
            sign = 1 if (k * (k - 1) // 2) % 2 == 0 else -1
            comp[b] = v * sign
        return Multivector(comp)

    def norm(self) -> float:
        prod = self * self.reverse()
        return math.sqrt(abs(prod.scalar_part()))

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for b, v in sorted(self.components.items(), key=lambda x: (len(x[0]), x[0])):
            if not b:
                blade = "1"
            else:
                blade = "*".join(f"e{i}" for i in sorted(b))
            terms.append(f"{v:.3g}{'*' if blade!='1' else ''}{blade}")
        return " + ".join(terms)


def text_to_multivector(text: str) -> Multivector:
    tokens = text.split()
    stylometry_vector = np.zeros(96)
    for i, token in enumerate(tokens):
        for j, func_cat in enumerate(FUNCTION_CATS):
            if token.lower() in FUNCTION_CATS[func_cat]:
                stylometry_vector[j] += 1
    return Multivector.vector(stylometry_vector)


def geometric_similarity(mv1: Multivector, mv2: Multivector) -> float:
    return mv1 * mv2


def assign_texts_to_nodes(texts: List[str], node_texts: List[str]) -> List[int]:
    node_multivectors = [text_to_multivector(text) for text in node_texts]
    assignments = []
    for text in texts:
        text_multivector = text_to_multivector(text)
        similarities = [geometric_similarity(text_multivector, node_multivector) for node_multivector in node_multivectors]
        assignments.append(np.argmax(similarities))
    return assignments


def main():
    texts = ["This is a test text.", "This is another test text."]
    node_texts = ["Node 1 text.", "Node 2 text."]
    assignments = assign_texts_to_nodes(texts, node_texts)
    print(assignments)


if __name__ == "__main__":
    main()