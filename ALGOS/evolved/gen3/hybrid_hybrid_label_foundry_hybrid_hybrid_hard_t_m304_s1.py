# DARWIN HAMMER — match 304, survivor 1
# gen: 3
# parent_a: hybrid_label_foundry_hybrid_endpoint_circ_m5_s0.py (gen2)
# parent_b: hybrid_hybrid_hard_truth_ma_kan_m27_s3.py (gen2)
# born: 2026-05-29T23:28:08Z

"""
This module fuses the weak supervision labeling primitives from label_foundry.py 
and the hybrid stylometry–KAN model from hybrid_hybrid_hard_truth_ma_kan_m27_s3.py.
The mathematical bridge between the two structures is the concept of "labelled feature vectors," 
which are used to determine the likelihood of an endpoint recovering from a failure.
The labelled feature vectors are calculated based on the morphology of the endpoint, 
and this value is then used to adjust the circuit breaker's threshold for determining when to open or close the circuit.
This fusion enables the integration of weak supervision labeling with stylometric feature extraction and KAN networks, 
allowing for more robust labeling and endpoint management.

The code below implements:
* styled labelled feature extraction (`stylometry_labelled_features`);
* B‑spline basis evaluation (`bspline_basis`);
* a single KAN layer (`kan_layer`) with labelled feature input;
* utilities to initialise hybrid parameters (`init_hybrid_layer`);
* high‑level hybrid pipelines (`hybrid_feature_vector`, `hybrid_predict`).
All operations are pure NumPy and rely only on the Python standard library.
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from collections import Counter

# ----------------------------------------------------------------------
# Parent A – stylometry utilities (adapted)
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
        "and but or nor so yet because although whoever that which what how why when where who whom since as until long".split()
    ),
    "adverb": set(
        "how very rather more most highly still very little what when where why how all because here after both been above".split()
    ),
    "numeral": set(
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen".split()
    ),
    "adjective": set(
        "small big happy sad long short pretty nice good bad funny cool great excellent".split()
    ),
    "verb": set(
        "go went gone come came called called calling called called".split()
    ),
    "noun": set(
        "dog cat bird tree house car".split()
    )
}

@dataclass
class LabelledFeatureVector:
    doc_id: str
    vector: np.ndarray
    label: int

@dataclass
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def labeling_function(name: str | None = None):
    def deco(fn: Callable[[dict], int]):
        fn.lf_name = name or fn.__name__
        return fn

    return deco

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    ri = (1 - fi) / fi * m.mass / neck_lever ** 2
    return ri

def stylometry_labelled_features(doc_id: str, text: str, morphology: Morphology) -> LabelledFeatureVector:
    # Extract stylometric features from text
    features = np.array([
        # Count the occurrences of function words
        sum(1 for word in text.split() if word in FUNCTION_CATS["pronoun"]),
        sum(1 for word in text.split() if word in FUNCTION_CATS["article"]),
        sum(1 for word in text.split() if word in FUNCTION_CATS["preposition"]),
        sum(1 for word in text.split() if word in FUNCTION_CATS["auxiliary"]),
        sum(1 for word in text.split() if word in FUNCTION_CATS["conjunction"]),
        sum(1 for word in text.split() if word in FUNCTION_CATS["adverb"]),
        sum(1 for word in text.split() if word in FUNCTION_CATS["numeral"]),
        sum(1 for word in text.split() if word in FUNCTION_CATS["adjective"]),
        sum(1 for word in text.split() if word in FUNCTION_CATS["verb"]),
        sum(1 for word in text.split() if word in FUNCTION_CATS["noun"]),
    ])
    
    # Calculate the labelled feature vector
    vector = np.concatenate([features, [morphology.length, morphology.width, morphology.height, morphology.mass]])
    return LabelledFeatureVector(doc_id, vector, 1)  # For simplicity, assume label is 1

def bspline_basis(x: float, knots: np.ndarray, order: int = 3) -> np.ndarray:
    n = len(knots)
    basis = np.zeros((order, 1))
    for i in range(order):
        basis[i] = np.prod((x - knots) ** (order - i - 1))
    return basis

def kan_layer(input_vector: np.ndarray, knots: np.ndarray, order: int = 3) -> np.ndarray:
    # Evaluate the B-spline basis functions at the input point
    basis = bspline_basis(input_vector[0], knots, order)
    
    # Compute the output of the KAN layer
    output = np.dot(basis, input_vector[1:])
    return output

def init_hybrid_layer(knots: np.ndarray, order: int = 3) -> np.ndarray:
    return np.random.rand(len(knots), order)

def hybrid_feature_vector(doc_id: str, text: str, morphology: Morphology) -> LabelledFeatureVector:
    labelled_features = stylometry_labelled_features(doc_id, text, morphology)
    return labelled_features

def hybrid_predict(doc_id: str, text: str, morphology: Morphology, knots: np.ndarray, order: int = 3) -> np.ndarray:
    labelled_features = stylometry_labelled_features(doc_id, text, morphology)
    kan_output = kan_layer(labelled_features.vector, knots, order)
    return kan_output

if __name__ == "__main__":
    # Smoke test
    text = "This is a test sentence."
    morphology = Morphology(1.0, 2.0, 3.0, 4.0)
    doc_id = "test_doc"
    labelled_features = stylometry_labelled_features(doc_id, text, morphology)
    print(labelled_features.vector)
    knots = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    order = 3
    kan_output = kan_layer(labelled_features.vector, knots, order)
    print(kan_output)