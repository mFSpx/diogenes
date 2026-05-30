# DARWIN HAMMER — match 52, survivor 2
# gen: 2
# parent_a: hybrid_fisher_localization_krampus_chrono_m17_s1.py (gen1)
# parent_b: jepa_energy.py (gen0)
# born: 2026-05-29T23:23:52Z

"""Hybrid Fisher‑JEPA algorithm.

Parents:
- **hybrid_fisher_localization_krampus_chrono_m17_s1.py** (Algorithm A) – extracts
  candidate timestamps from text and scores them with a Fisher‑information
  based “information density”.
- **jepa_energy.py** (Algorithm B) – defines a Joint Embedding Predictive Architecture
  (JEPA) where an encoder maps raw observations to unit‑norm representations and a
  predictor uses a latent variable *z* to forecast future representations. Energy is
  the squared L2 distance between the true and predicted representations.

Mathematical bridge:
The Fisher score *F(θ)* is a scalar that quantifies how informative a particular
timestamp is. In JEPA the latent variable *z* encodes hidden causes. We treat the
Fisher score of a timestamp candidate as the latent *z* supplied to the predictor.
Thus each date candidate yields a JEPA energy

    E(candidate) = ‖ encoder(t) – predictor( encoder(t_prev), F(θ) ) ‖²

where *t* is the candidate timestamp (seconds since the epoch) and *t_prev* is a
reference timestamp (here the Fisher centre). The hybrid therefore fuses the
information‑density weighting of A with the representation‑space prediction of B.

The module provides three high‑level functions that illustrate the hybrid operation
and a small smoke test.
"""

import math
import random
import sys
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Algorithm A – Fisher‑based date extraction
# ---------------------------------------------------------------------------

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def parse_loose_datetime(raw: str) -> datetime | None:
    text = raw.strip().strip("'\"`[]()")
    if not text:
        return None
    try:
        val = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if val.tzinfo is None:
            val = val.replace(tzinfo=timezone.utc)
        return val.astimezone(timezone.utc)
    except ValueError:
        return None


def chrono_candidates_for_path(path: Path, text_sample: str = "") -> list[dict]:
    """Extract candidate timestamps from a text snippet."""
    candidates = []
    pattern = r"(?im)^\s*(?:date|created|created_at|created at|generated|timestamp|time|filed|updated|modified)\s*[:=]\s*[\"']?((?:20|19)\d{2}[-_/]\d{1,2}[-_/]\d{1,2}(?:[T\s_]\d{1,2}:?\d{2}(?::?\d{2})?(?:\s?(?:Z|[+-]\d{2}:?\d{2}))?)?)"
    for match in re.finditer(pattern, text_sample):
        raw = match.group(1)
        parsed = parse_loose_datetime(raw)
        if parsed:
            candidates.append({
                "timestamp": parsed,
                "source": "content_frontmatter",
                "raw": raw,
                "path": path,
            })
    return candidates


# ---------------------------------------------------------------------------
# Algorithm B – JEPA core components
# ---------------------------------------------------------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def encoder(x: np.ndarray, W_enc: np.ndarray, b_enc: np.ndarray | None = None) -> np.ndarray:
    """Linear encoder with unit‑sphere normalization."""
    s = W_enc @ x
    if b_enc is not None:
        s = s + b_enc
    return _normalize(s)


def predictor(s_y: np.ndarray, z: np.ndarray, W_pred: np.ndarray, b_pred: np.ndarray | None = None) -> np.ndarray:
    """Linear predictor that receives past representation and latent z."""
    inp = np.concatenate([s_y, z])
    p = W_pred @ inp
    if b_pred is not None:
        p = p + b_pred
    return _normalize(p)


def jepa_energy(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                W_enc: np.ndarray, W_pred: np.ndarray,
                b_enc: np.ndarray | None = None, b_pred: np.ndarray | None = None) -> float:
    """JEPA energy = squared L2 distance between true and predicted representations."""
    s_x = encoder(x, W_enc, b_enc)
    s_y = encoder(y, W_enc, b_enc)
    p_hat = predictor(s_y, z, W_pred, b_pred)
    diff = s_x - p_hat
    return float(np.dot(diff, diff))


# ---------------------------------------------------------------------------
# Hybrid operations
# ---------------------------------------------------------------------------

def hybrid_fisher_jepa_scores(fisher_center: float,
                              fisher_width: float,
                              path: Path,
                              text_sample: str,
                              W_enc: np.ndarray,
                              W_pred: np.ndarray) -> list[dict]:
    """
    For each datetime candidate extracted from *text_sample* compute:
      * Fisher score (information density)
      * JEPA energy where the Fisher score is used as the latent variable *z*.

    Returns a list of dictionaries ordered by increasing JEPA energy (more plausible dates first).
    """
    candidates = chrono_candidates_for_path(path, text_sample)
    results = []

    # reference observation (past) – we use the centre timestamp as a scalar vector
    y_vec = np.array([fisher_center], dtype=float)

    for cand in candidates:
        t = cand["timestamp"].timestamp()
        theta = (t - fisher_center) / fisher_width
        F = fisher_score(theta, 0.0, 1.0)          # scalar Fisher information
        # Observation vectors are 1‑D (timestamp expressed as seconds)
        x_vec = np.array([t], dtype=float)
        # Latent variable is a 1‑D vector containing the Fisher score
        z_vec = np.array([F], dtype=float)

        energy = jepa_energy(x_vec, y_vec, z_vec, W_enc, W_pred)
        results.append({
            "candidate": cand,
            "fisher_score": F,
            "jepa_energy": energy,
        })

    # Sort by JEPA energy (lower is better)
    results.sort(key=lambda r: r["jepa_energy"])
    return results


def hybrid_best_candidate(fisher_center: float,
                          fisher_width: float,
                          path: Path,
                          text_sample: str,
                          W_enc: np.ndarray,
                          W_pred: np.ndarray) -> dict | None:
    """
    Return the most plausible datetime candidate according to the hybrid metric.
    If no candidates are found, returns None.
    """
    scored = hybrid_fisher_jepa_scores(fisher_center, fisher_width, path,
                                       text_sample, W_enc, W_pred)
    if not scored:
        return None
    best = scored[0]
    # Enrich with a ready‑to‑use datetime object
    best["datetime"] = best["candidate"]["timestamp"]
    return best


def hybrid_batch_energy(fisher_center: float,
                        fisher_width: float,
                        path: Path,
                        text_sample: str,
                        W_enc: np.ndarray,
                        W_pred: np.ndarray,
                        n_samples: int = 5) -> float:
    """
    Sample *n_samples* random subsets of the candidate list (by shuffling) and
    compute the mean JEPA energy across the chosen best candidate of each subset.
    This provides a stochastic estimate of the hybrid objective.
    """
    candidates = chrono_candidates_for_path(path, text_sample)
    if not candidates:
        return float('nan')

    energies = []
    for _ in range(n_samples):
        random.shuffle(candidates)
        # keep only the first half as a random subset
        subset = candidates[: max(1, len(candidates)//2)]
        # compute hybrid scores for the subset
        sub_path = path  # path is unchanged
        sub_text = "\n".join([c["raw"] for c in subset])
        best = hybrid_best_candidate(fisher_center, fisher_width,
                                     sub_path, sub_text, W_enc, W_pred)
        if best:
            energies.append(best["jepa_energy"])
    return float(np.mean(energies)) if energies else float('nan')


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create a temporary file with several date lines
    tmp_path = Path("tmp_dates.txt")
    tmp_path.write_text(
        "created: 2022-03-15T12:30:00Z\n"
        "updated: 2023-07-01 08:45\n"
        "timestamp = 2021-11-05\n"
        "date: 2020-01-01T00:00:00Z\n"
    )

    sample_text = tmp_path.read_text()

    # Fisher parameters (centre = now, width = 1 year in seconds)
    now_ts = datetime.now(timezone.utc).timestamp()
    one_year = 365.25 * 24 * 3600
    fisher_center = now_ts
    fisher_width = one_year

    # JEPA weight matrices (simple dimensions)
    d_in = 1                 # scalar timestamp
    d_rep = 4                # representation size
    d_latent = 1             # Fisher score dimension

    rng = np.random.default_rng(42)
    W_enc = rng.normal(scale=0.1, size=(d_rep, d_in))
    W_pred = rng.normal(scale=0.1, size=(d_rep, d_rep + d_latent))

    # Run hybrid functions
    scores = hybrid_fisher_jepa_scores(fisher_center, fisher_width,
                                       tmp_path, sample_text,
                                       W_enc, W_pred)
    print("Hybrid scores (sorted by JEPA energy):")
    for entry in scores:
        cand = entry["candidate"]
        print(f"  {cand['raw']} -> Fisher={entry['fisher_score']:.4e}, Energy={entry['jepa_energy']:.4e}")

    best = hybrid_best_candidate(fisher_center, fisher_width,
                                 tmp_path, sample_text,
                                 W_enc, W_pred)
    if best:
        print("\nBest candidate according to hybrid metric:")
        print(f"  raw: {best['candidate']['raw']}")
        print(f"  datetime (UTC): {best['datetime'].isoformat()}")
        print(f"  fisher_score: {best['fisher_score']:.4e}")
        print(f"  jepa_energy: {best['jepa_energy']:.4e}")
    else:
        print("\nNo date candidates found.")

    batch_energy = hybrid_batch_energy(fisher_center, fisher_width,
                                       tmp_path, sample_text,
                                       W_enc, W_pred, n_samples=7)
    print(f"\nMean JEPA energy over stochastic batches: {batch_energy:.4e}")

    # Clean up temporary file
    tmp_path.unlink(missing_ok=True)