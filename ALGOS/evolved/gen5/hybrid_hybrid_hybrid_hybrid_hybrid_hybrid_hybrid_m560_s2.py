# DARWIN HAMMER — match 560, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_gliner_hybrid_possum_filter_m323_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hard_t_ollivier_ricci_curva_m393_s2.py (gen4)
# born: 2026-05-29T23:29:46Z

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from datetime import datetime, timezone
import math
import uuid

@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"

class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Return the multiplicative decay factor since last_decay."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor

def stylometric_feature_extraction(texts: List[str]) -> np.ndarray:
    FUNCTION_CATS = {
        "pronoun": {
            "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
            "he", "him", "his", "she", "her", "hers", "they", "them", "their",
            "theirs", "we", "us", "our", "ours"
        },
        "article": {"a", "an", "the"},
        "preposition": {
            "about", "above", "after", "against", "around", "as", "at", "before",
            "behind", "below", "between", "by", "during", "for", "from", "in",
            "into", "of", "off", "on", "onto", "over", "through", "to", "under",
            "with", "without"
        },
        "auxiliary": {
            "am", "are", "be", "been", "being", "can", "could", "did", "do",
            "does", "had", "has", "have", "is", "may", "might"
        }
    }
    features = []
    for text in texts:
        feature_vector = [0] * len(FUNCTION_CATS)
        for i, (category, words) in enumerate(FUNCTION_CATS.items()):
            for word in words:
                if word.lower() in text.lower():
                    feature_vector[i] += 1
        features.append(feature_vector)
    return np.array(features)

def ollivier_ricci_curvature(graph: np.ndarray) -> np.ndarray:
    num_vertices = graph.shape[0]
    curvature = np.zeros((num_vertices, num_vertices))
    for i in range(num_vertices):
        for j in range(num_vertices):
            if i != j:
                curvature[i, j] = graph[i, j] / (1 + graph[i, j])
    return curvature

def compute_similarity_matrix(feature_vectors: np.ndarray) -> np.ndarray:
    num_vectors = feature_vectors.shape[0]
    similarity_matrix = np.zeros((num_vectors, num_vectors))
    for i in range(num_vectors):
        for j in range(num_vectors):
            if i != j:
                similarity_matrix[i, j] = 1 - (np.linalg.norm(feature_vectors[i] - feature_vectors[j]) / 
                                               (np.linalg.norm(feature_vectors[i]) + np.linalg.norm(feature_vectors[j])))
    return similarity_matrix

def hybrid_algorithm(texts: List[str], spans: List[Span], distance_threshold: float) -> List[Span]:
    feature_vectors = stylometric_feature_extraction(texts)
    similarity_matrix = compute_similarity_matrix(feature_vectors)
    curvature = ollivier_ricci_curvature(similarity_matrix)
    filtered_spans = []
    for i, span in enumerate(spans):
        similar_spans = [s for j, s in enumerate(spans) if i != j and similarity_matrix[i, j] > (1 - distance_threshold)]
        if not similar_spans:
            filtered_spans.append(span)
    return filtered_spans

if __name__ == "__main__":
    texts = ["This is a sample text.", "Another sample text.", "A third sample text."]
    spans = [Span(0, 10, "sample text", "label", 0.5), Span(15, 25, "another sample", "label", 0.7), Span(30, 40, "third sample", "label", 0.3)]
    distance_threshold = 0.5
    filtered_spans = hybrid_algorithm(texts, spans, distance_threshold)
    print(filtered_spans)