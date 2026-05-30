# DARWIN HAMMER — match 733, survivor 3
# gen: 3
# parent_a: hybrid_hybrid_decision_hygi_hybrid_sketches_rlct_m6_s3.py (gen2)
# parent_b: ternary_lens_audit.py (gen0)
# born: 2026-05-29T23:30:40Z

"""Hybrid Decision‑Hygiene, Sketch‑RLCT & Ternary‑Lens Audit Module
=================================================================

Parent A
--------
* ``hybrid_decision_hygiene_shannon_entropy_m12_s1.py`` – extracts
  decision‑hygiene token frequencies, computes Shannon entropy *H* and
  supplies a Count‑Min sketch together with a HyperLogLog estimator.  The
  core mathematical objects are **log‑counts**.

Parent B
--------
* ``ternary_lens_audit.py`` – loads a vendor manifest, applies a set of
  regex‑based policy rules and produces a list of rule‑violations.  The
  policy can be expressed as a binary vector **v** and a weight vector
  **w**, giving a linear penalty *w·v*.

Mathematical Bridge
-------------------
Both parents manipulate **log‑count** information:

* Entropy uses `p_i = c_i / Σc` and `H = - Σ p_i log p_i`.
* The Count‑Min sketch yields an **approximate log‑likelihood**
  `ℓ̂ = Σ_j log ŷ_j` where `ŷ_j` are the sketch frequencies.
* HyperLogLog supplies an estimate `N̂` of distinct tokens; `λ = log N̂`
  appears in RLCT free‑energy terms `λ·log n`.
* The audit rules produce a binary violation vector `v ∈ {0,1}^k`
  (one entry per rule).  Assigning a configurable weight `w_i` to each
  rule gives a penalty `P = Σ_i w_i v_i`.

The unified hybrid free‑energy therefore becomes


F_hybrid = ℓ̂  -  H  +  λ·log n  +  Σ_i w_i v_i


where
* `n` – total token count,
* `ℓ̂` – sketch log‑likelihood,
* `H` – decision‑hygiene entropy,
* `λ = log N̂` – RLCT coefficient from HyperLogLog,
* `v_i` – audit‑rule violation flags,
* `w_i` – user‑defined rule weights.

The module below implements this fusion with three public functions
demonstrating the combined computation.

"""

import math
import random
import sys
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Decision‑hygiene regexes (Parent A)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(r"\b(?:plan|checklist|steps?|sequence|roadmap|schedule|timeline)\b", re.I)
DELAY_RE = re.compile(r"\b(?:delay|latency|wait|postpone|defer|hold|stall)\b", re.I)

DECISION_CATEGORIES = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
}

# ----------------------------------------------------------------------
# Simple Count‑Min Sketch (Parent A)
# ----------------------------------------------------------------------
class CountMinSketch:
    """Very small Count‑Min sketch using pairwise‑independent hash functions."""

    def __init__(self, width: int = 2 ** 10, depth: int = 5, seed: int = 0):
        self.width = width
        self.depth = depth
        self.tables = np.zeros((depth, width), dtype=np.int64)
        random.seed(seed)
        self.seeds = [random.randint(1, 2 ** 31 - 1) for _ in range(depth)]

    def _hash(self, item: str, i: int) -> int:
        h = hash((item, self.seeds[i]))
        return (h ^ (h >> 16)) % self.width

    def add(self, item: str, increment: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(item, i)
            self.tables[i, idx] += increment

    def estimate(self, item: str) -> int:
        """Return the minimum count across rows (standard CMS estimate)."""
        return min(self.tables[i, self._hash(item, i)] for i in range(self.depth))

    def total(self) -> int:
        return int(self.tables.sum())


# ----------------------------------------------------------------------
# Very small HyperLogLog estimator (Parent A)
# ----------------------------------------------------------------------
class HyperLogLog:
    """HyperLogLog with 2**p registers, using built‑in hash."""

    def __init__(self, p: int = 10):
        self.p = p
        self.m = 1 << p
        self.registers = np.zeros(self.m, dtype=np.uint8)

    @staticmethod
    def _rho(w: int, max_width: int) -> int:
        """Position of leftmost 1-bit (1‑based)."""
        rho = 1
        while rho <= max_width and (w & (1 << (max_width - rho))) == 0:
            rho += 1
        return rho

    def add(self, item: str) -> None:
        x = hash(item) & 0xFFFFFFFFFFFFFFFF  # 64‑bit hash
        idx = x >> (64 - self.p)
        w = x << self.p & 0xFFFFFFFFFFFFFFFF
        rank = self._rho(w, 64 - self.p)
        self.registers[idx] = max(self.registers[idx], rank)

    def estimate(self) -> float:
        """Return cardinality estimate using the original HLL formula."""
        alpha_m = 0.7213 / (1 + 1.079 / self.m)
        Z = 1.0 / np.sum(2.0 ** -self.registers)
        E = alpha_m * self.m * self.m * Z
        # Small range correction
        if E <= 2.5 * self.m:
            V = np.count_nonzero(self.registers == 0)
            if V != 0:
                E = self.m * math.log(self.m / V)
        return E


# ----------------------------------------------------------------------
# Audit rule engine (Parent B)
# ----------------------------------------------------------------------
CLASSIFICATIONS = {
    "usable_now",
    "research_only",
    "needs_conversion",
    "unsafe_for_fastpath",
    "unsupported",
}
LOCAL_PATTERNS = [
    "*bitnet*",
    "*BitNet*",
    "*fairyfuse*",
    "*FairyFuse*",
    "*lora*",
    "*LoRA*",
    "*adapter*",
]

def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    for candidate in data.get("vendors", []):
        classification = candidate.get("classification")
        if classification not in CLASSIFICATIONS:
            raise ValueError(f"invalid classification {classification!r}")
    return data


def enforce_fast_path_rule(candidate: dict) -> List[str]:
    findings: List[str] = []
    key = candidate.get("candidate_key", "")
    family = candidate.get("family", "")
    notes = candidate.get("notes", "")

    if re.search(r"standard.*lora|peft|qlora", key + " " + family, re.I):
        if candidate.get("classification") != "unsafe_for_fastpath" or candidate.get("fast_path_compatible"):
            findings.append("STANDARD_LORA_RULE_VIOLATION")
    if re.search(r"fp16|fp32", notes, re.I) and candidate.get("fast_path_compatible"):
        findings.append("FP_HOTPATH_CONFLICT")
    if candidate.get("fast_path_compatible") and candidate.get("benchmark_required") and not candidate.get("benchmark_evidence"):
        findings.append("FAST_PATH_CLAIM_NEEDS_BENCHMARK_EVIDENCE")
    return findings


def audit_candidate(candidate: dict) -> Tuple[bool, List[str]]:
    """Return (is_clean, list_of_violations)."""
    violations = enforce_fast_path_rule(candidate)
    return (len(violations) == 0, violations)


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def decision_hygiene_counts(texts: Iterable[str]) -> Tuple[Dict[str, int], int]:
    """
    Count decision‑hygiene tokens across *texts*.

    Returns
    -------
    counts : dict mapping category name → count
    total  : total number of matched tokens
    """
    counts = defaultdict(int)
    total = 0
    for txt in texts:
        for cat, regex in DECISION_CATEGORIES.items():
            matches = regex.findall(txt)
            c = len(matches)
            if c:
                counts[cat] += c
                total += c
    return dict(counts), total


def shannon_entropy(counts: Dict[str, int]) -> float:
    """Compute Shannon entropy `H = - Σ p_i log p_i` from raw counts."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array(list(counts.values())) / total
    return -float(np.sum(probs * np.log(probs + 1e-12)))


def sketch_log_likelihood(tokens: Iterable[str], cms: CountMinSketch) -> float:
    """
    Feed *tokens* into *cms* and return the approximate log‑likelihood

        ℓ̂ = Σ_j log (frequencŷ_j + 1)

    The `+1` prevents log(0) for unseen items.
    """
    for tok in tokens:
        cms.add(tok, 1)
    ll = 0.0
    for tok in tokens:
        est = cms.estimate(tok) + 1  # Laplace smoothing
        ll += math.log(est)
    return ll


def hyperloglog_lambda(tokens: Iterable[str], hll: HyperLogLog) -> float:
    """Add *tokens* to *hll* and return λ = log(N̂) where N̂ is the cardinality estimate."""
    for tok in tokens:
        hll.add(tok)
    N_hat = hll.estimate()
    return math.log(N_hat + 1e-12)


def audit_violations(manifest: dict, weight_map: Dict[str, float] = None) -> Tuple[float, List[str]]:
    """
    Evaluate all candidates in *manifest* with the audit rules.

    Returns
    -------
    penalty : weighted sum Σ w_i v_i
    details : list of human‑readable violation strings
    """
    if weight_map is None:
        weight_map = {
            "STANDARD_LORA_RULE_VIOLATION": 2.0,
            "FP_HOTPATH_CONFLICT": 1.5,
            "FAST_PATH_CLAIM_NEEDS_BENCHMARK_EVIDENCE": 3.0,
        }

    total_penalty = 0.0
    all_details: List[str] = []

    for cand in manifest.get("vendors", []):
        clean, violations = audit_candidate(cand)
        for v in violations:
            w = weight_map.get(v, 1.0)
            total_penalty += w
            all_details.append(f"{cand.get('candidate_key','<unknown>')}: {v} (w={w})")
    return total_penalty, all_details


def hybrid_free_energy(
    texts: Iterable[str],
    tokens: Iterable[str],
    manifest_path: Path,
    cms_width: int = 1024,
    cms_depth: int = 5,
    hll_p: int = 10,
) -> Tuple[float, dict]:
    """
    Compute the unified hybrid free‑energy metric.

    Parameters
    ----------
    texts
        Iterable of raw strings used for decision‑hygiene analysis.
    tokens
        Token stream (e.g. whitespace split) used for sketch & HLL.
    manifest_path
        Path to a vendor manifest JSON file (Parent B input).

    Returns
    -------
    F_hybrid : float
        The combined metric `ℓ̂ - H + λ·log n + penalty`.
    details : dict
        Intermediate values useful for debugging/inspection.
    """
    # 1. Decision‑hygiene entropy
    counts, dh_total = decision_hygiene_counts(texts)
    H = shannon_entropy(counts)

    # 2. Sketch log‑likelihood
    cms = CountMinSketch(width=cms_width, depth=cms_depth, seed=42)
    ℓ_hat = sketch_log_likelihood(tokens, cms)

    # 3. HyperLogLog λ term
    hll = HyperLogLog(p=hll_p)
    λ = hyperloglog_lambda(tokens, hll)

    # 4. Audit penalty
    manifest = load_manifest(manifest_path)
    penalty, audit_details = audit_violations(manifest)

    n = sum(1 for _ in tokens)  # total token count
    F = ℓ_hat - H + λ * math.log(max(n, 1)) + penalty

    details = {
        "entropy_H": H,
        "sketch_log_likelihood": ℓ_hat,
        "lambda": λ,
        "log_n": math.log(max(n, 1)),
        "audit_penalty": penalty,
        "audit_details": audit_details,
        "decision_counts": counts,
        "total_decision_tokens": dh_total,
        "total_tokens": n,
    }
    return F, details


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal in‑memory manifest to avoid filesystem dependencies
    dummy_manifest = {
        "vendors": [
            {
                "candidate_key": "example_lora_v1",
                "family": "standard_lora",
                "classification": "unsafe_for_fastpath",
                "fast_path_compatible": False,
                "benchmark_required": False,
                "benchmark_evidence": None,
                "notes": "",
            },
            {
                "candidate_key": "fast_fp16_adapter",
                "family": "adapter_family",
                "classification": "usable_now",
                "fast_path_compatible": True,
                "benchmark_required": True,
                "benchmark_evidence": None,
                "notes": "fp16 optimized",
            },
        ]
    }

    # Write temporary manifest file
    tmp_path = Path("tmp_manifest.json")
    tmp_path.write_text(json.dumps(dummy_manifest), encoding="utf-8")

    sample_texts = [
        "We need to verify the source and provide evidence for the plan.",
        "The delay was noted, but the checklist is ready.",
    ]
    sample_tokens = "We need to verify the source and provide evidence for the plan . The delay was noted but the checklist is ready .".split()

    F, dbg = hybrid_free_energy(
        texts=sample_texts,
        tokens=sample_tokens,
        manifest_path=tmp_path,
        cms_width=256,
        cms_depth=4,
        hll_p=8,
    )

    print(f"Hybrid free‑energy: {F:.4f}")
    for k, v in dbg.items():
        print(f"{k}: {v}")

    # Clean up temporary file
    try:
        tmp_path.unlink()
    except OSError:
        pass