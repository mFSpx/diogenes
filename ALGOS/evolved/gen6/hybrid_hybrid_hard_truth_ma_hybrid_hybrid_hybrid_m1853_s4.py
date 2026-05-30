# DARWIN HAMMER — match 1853, survivor 4
# gen: 6
# parent_a: hybrid_hard_truth_math_hybrid_minimum_cost__m12_s6.py (gen2)
# parent_b: hybrid_hybrid_hybrid_distri_hybrid_hybrid_hybrid_m602_s0.py (gen5)
# born: 2026-05-29T23:39:22Z

import re
import math
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Linguistic categories
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
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}


def words(text: str) -> List[str]:
    """Extract lower‑case alphabetic tokens from a string."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def lsm_vector(text: str) -> Dict[str, float]:
    """
    Produce a lightweight semantic (LSM) vector for *text*.
    The vector contains the relative frequency of each functional category.
    """
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }


def compute_phash(values: List[float]) -> int:
    """Perceptual hash of a numeric vector (max 64 bits)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()


def broadcast_probability(phase: int, step: int) -> float:
    """Probability used by the bandit for exploration."""
    if phase < 1 or step < 1:
        raise ValueError("phase and step must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))


def gini_coefficient(rewards: List[float]) -> float:
    """Gini coefficient of a reward batch (0 = equal, 1 = maximal inequality)."""
    if not rewards:
        return 0.0
    rewards_arr = np.array(rewards, dtype=float)
    mean = np.mean(rewards_arr)
    if mean == 0:
        return 0.0
    n = len(rewards_arr)
    diffs = np.abs(rewards_arr[:, None] - rewards_arr[None, :])
    return diffs.sum() / (2 * n * n * mean)


def schoolfield_rate(temperature: float) -> float:
    """Simple temperature‑performance model."""
    return 1.0 / (1.0 + math.exp(temperature - 20.0))


# ----------------------------------------------------------------------
# Graph construction and feature handling
# ----------------------------------------------------------------------
def build_graph(elements: List[List[float]], hamming_thresh: int = 2) -> Dict[str, Dict[str, float]]:
    """
    Build an undirected graph where nodes are string indices.
    An edge exists if the Hamming distance between perceptual hashes is ≤ *hamming_thresh*.
    Edge weights are initialised to 1.0 and later overwritten by semantic similarity.
    """
    graph: Dict[str, Dict[str, float]] = {}
    hashes: Dict[str, int] = {str(i): compute_phash(el) for i, el in enumerate(elements)}

    for i in range(len(elements)):
        graph[str(i)] = {}
        for j in range(i + 1, len(elements)):
            if hamming_distance(hashes[str(i)], hashes[str(j)]) <= hamming_thresh:
                graph[str(i)][str(j)] = 1.0
                graph[str(j)][str(i)] = 1.0
    return graph


def node_feature_vector(
    element: List[float],
    label: str | None = None,
    recent_rewards: List[float] | None = None,
) -> Dict[str, float]:
    """
    Produce a unified feature vector for a node.
    Components:
        • Linguistic LSM vector derived from *label* (or a placeholder if missing).
        • Temperature performance term from the element's mean value.
        • Gini‑scaled exploration term derived from *recent_rewards*.
    """
    # 1️⃣ Linguistic component
    text = label if label is not None else "placeholder"
    feat = lsm_vector(text)

    # 2️⃣ Temperature component (scaled into [0,1] via Schoolfield)
    temp = np.mean(element) if element else 0.0
    feat["temperature"] = schoolfield_rate(temp)

    # 3️⃣ Gini‑based scaling (optional)
    if recent_rewards:
        gini = gini_coefficient(recent_rewards)
        # The more unequal the rewards, the smaller the contribution
        feat["gini_scale"] = 1.0 - gini
    else:
        feat["gini_scale"] = 1.0

    return feat


def lsm_score(a: Dict[str, float], b: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """
    Cosine‑like similarity between two LSM‑style vectors.
    Returns the overall similarity and per‑category contributions.
    """
    detail: Dict[str, float] = {}
    for cat in FUNCTION_CATS:
        av = a.get(cat, 0.0)
        bv = b.get(cat, 0.0)
        denom = av + bv + 1e-6
        score = 1.0 - (abs(av - bv) / denom)
        detail[cat] = max(0.0, min(1.0, score))
    overall = sum(detail.values()) / len(FUNCTION_CATS)
    return overall, detail


def integrate_graph(
    graph: Dict[str, Dict[str, float]],
    elements: List[List[float]],
    labels: List[str] | None = None,
    reward_history: List[float] | None = None,
) -> None:
    """
    Replace raw edge weights with semantic similarity scores.
    The similarity is a weighted blend of linguistic, temperature and Gini components.
    """
    if labels is None:
        labels = [None] * len(elements)

    # Pre‑compute node feature vectors
    node_feats: Dict[str, Dict[str, float]] = {}
    for idx, (el, lab) in enumerate(zip(elements, labels)):
        node_feats[str(idx)] = node_feature_vector(el, lab, reward_history)

    for node, neighbors in graph.items():
        for nbr in list(neighbors.keys()):
            sim, _ = lsm_score(node_feats[node], node_feats[nbr])
            # Blend with the original topological weight (currently always 1.0)
            graph[node][nbr] = 0.6 * sim + 0.4 * neighbors[nbr]


# ----------------------------------------------------------------------
# Adaptive NLMS weight update
# ----------------------------------------------------------------------
def nlms_weight_update(rewards: List[float], base_step: float) -> float:
    """
    Adjust *base_step* using the Gini coefficient of the recent reward batch.
    Returns the adapted step size.
    """
    gini = gini_coefficient(rewards)
    adapted = base_step * (1.0 - gini)
    return max(0.0, adapted)


# ----------------------------------------------------------------------
# High‑level hybrid bandit class
# ----------------------------------------------------------------------
class HybridBandit:
    """
    A minimal contextual multi‑armed bandit that fuses:
        • Graph‑based topology (edges reflect similarity).
        • Linguistic‑semantic node descriptors.
        • Adaptive NLMS learning rate driven by reward inequality.
    """

    def __init__(
        self,
        elements: List[List[float]],
        labels: List[str] | None = None,
        hamming_thresh: int = 2,
        base_step: float = 0.1,
    ):
        self.elements = elements
        self.labels = labels if labels is not None else [None] * len(elements)
        self.graph = build_graph(elements, hamming_thresh)
        self.base_step = base_step
        self.rewards_history: List[float] = []
        self.phase = 1  # used by broadcast_probability
        self.step_counter = 0

        # Initialise semantic edge weights
        integrate_graph(self.graph, self.elements, self.labels)

    def select_arm(self) -> str:
        """
        Choose a node (arm) using an ε‑greedy style policy where ε is given by
        broadcast_probability(self.phase, self.step_counter).
        """
        eps = broadcast_probability(self.phase, self.step_counter + 1)
        if random.random() < eps:
            choice = random.choice(list(self.graph.keys()))
        else:
            # Exploit: pick node with highest average edge weight
            avg_weights = {
                node: sum(w for w in nbrs.values()) / max(1, len(nbrs))
                for node, nbrs in self.graph.items()
            }
            choice = max(avg_weights, key=avg_weights.get)
        self.step_counter += 1
        return choice

    def update(self, node: str, reward: float) -> None:
        """
        Record *reward*, adapt the NLMS step size and refresh the graph
        with the latest reward distribution.
        """
        self.rewards_history.append(reward)
        # Keep a short sliding window for stability
        if len(self.rewards_history) > 50:
            self.rewards_history.pop(0)

        # Adapt learning rate
        self.base_step = nlms_weight_update(self.rewards_history, self.base_step)

        # Re‑integrate graph using the updated reward‑driven Gini term
        integrate_graph(
            self.graph,
            self.elements,
            self.labels,
            reward_history=self.rewards_history,
        )

    def get_edge_weights(self) -> Dict[Tuple[str, str], float]:
        """Convenient view of the symmetric edge weights."""
        edges: Dict[Tuple[str, str], float] = {}
        for u, nbrs in self.graph.items():
            for v, w in nbrs.items():
                if (v, u) not in edges:
                    edges[(u, v)] = w
        return edges


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
def smoke_test() -> None:
    # Synthetic numeric elements and optional textual labels
    elements = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [2.5, 3.5, 4.5],
    ]
    labels = ["the quick brown fox", "jumps over the lazy dog", "lorem ipsum dolor", "sample text"]

    bandit = HybridBandit(elements, labels, hamming_thresh=2, base_step=0.1)

    # Simulate a few interaction rounds
    for _ in range(10):
        arm = bandit.select_arm()
        # Dummy reward: higher for nodes with larger mean value
        reward = np.mean(elements[int(arm)]) + random.gauss(0, 0.5)
        bandit.update(arm, reward)

    # Print final edge weights for inspection
    for (u, v), w in bandit.get_edge_weights().items():
        print(f"Edge ({u}, {v}) weight: {w:.4f}")


if __name__ == "__main__":
    smoke_test()