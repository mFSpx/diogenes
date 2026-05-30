# DARWIN HAMMER — match 3185, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m2716_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_percyphon_hyb_m1198_s3.py (gen5)
# born: 2026-05-29T23:48:30Z

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Dict

EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str = ""
    rationale: str = ""
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")

EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)

REGEX_PATTERNS = [
    ("evidence", EVIDENCE_RE),
    ("planning", PLANNING_RE),
    ("delay", DELAY_RE),
    ("support", SUPPORT_RE),
    ("boundary", BOUNDARY_RE),
]

def extract_regex_features(text: str) -> np.ndarray:
    counts = np.array([len(pat.findall(text)) for _, pat in REGEX_PATTERNS], dtype=float)
    norm = np.linalg.norm(counts) + 1e-12
    return counts / norm

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float = 1.0

def sphericity_index(morph: Morphology) -> float:
    l, w, h = morph.length, morph.width, morph.height
    SA = 2 * (l * w + w * h + h * l)
    V = l * w * h
    if V <= 0:
        return 0.0
    return (SA ** 3) / (36 * np.pi * (V ** 2) + 1e-12)

def certainty_weighted_coboundary(flags: List[CertaintyFlag]) -> np.ndarray:
    n = len(flags)
    if n < 2:
        raise ValueError("At least two certainty flags are required")
    conf = np.array([f.confidence_bps / 10000.0 for f in flags], dtype=float)
    C = np.diag(conf)
    D = np.zeros((n - 1, n), dtype=float)
    for i in range(n - 1):
        D[i, i] = 1.0
        D[i, i + 1] = -1.0
    W = C @ D
    return W

def rotor_from_operator(op: np.ndarray) -> np.ndarray:
    rows, cols = op.shape
    if rows != cols:
        size = max(rows, cols)
        pad = ((0, size - rows), (0, size - cols))
        op_sq = np.pad(op, pad, mode='constant')
    else:
        op_sq = op.copy()
    q, _ = np.linalg.qr(op_sq)
    return q

def rotate_feature_vector(vec: np.ndarray, rotor: np.ndarray) -> np.ndarray:
    dim = rotor.shape[0]
    if vec.shape[0] < dim:
        vec_padded = np.pad(vec, (0, dim - vec.shape[0]), mode='constant')
    else:
        vec_padded = vec[:dim]
    return rotor @ vec_padded

def token_entropy(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    freq: Dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    total = len(tokens)
    probs = np.array(list(freq.values()), dtype=float) / total
    return -np.sum(probs * np.log2(probs + 1e-12))

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a) + 1e-12
    norm_b = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (norm_a * norm_b))

def pruning_probability(step: int, decay: float = 0.01) -> float:
    return np.exp(-decay * max(step, 0))

def hybrid_decision_metric(
    text: str,
    morph: Morphology,
    flags: List[CertaintyFlag],
    step: int = 0,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
) -> float:
    W = certainty_weighted_coboundary(flags)
    R = rotor_from_operator(W)
    f = extract_regex_features(text)
    f_rotated = rotate_feature_vector(f, R)
    sim = cosine_similarity(f_rotated, np.ones_like(f_rotated))
    H = token_entropy(text)
    S = sphericity_index(morph)
    p = pruning_probability(step)
    return p * (alpha * sim + beta * H + gamma * S)

import re