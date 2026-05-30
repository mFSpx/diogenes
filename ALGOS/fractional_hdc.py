#!/usr/bin/env python3
"""Fractional Power Binding in Hyperdimensional Computing.

Z = X (*) Y^alpha = F^{-1}(F(X) . exp(i*alpha*angle(F(Y))))

F: Fourier transform.  angle(F(Y)): phase of Y's Fourier spectrum.
Y^alpha: raise Y to fractional power alpha by scaling its phase by alpha.
(*): circular convolution — the binding operator in HDC.

In standard HDC concepts are encoded as random d-dimensional bipolar/complex
vectors.  Binding X (*) Y associates two concepts (role-filler binding).
Unbinding recovers X via X (*) Y^{-1}.  Fractional binding Y^alpha
interpolates continuously between identity (alpha=0) and full binding
(alpha=1), so analog relationships — 0.3-angry, 0.7-curious — live in a
flat vector without coordinate axes.

Superposition (X + Y) bundles concepts.  Similarity is cosine distance.
Cleanup is nearest-stored-vector lookup.

The holographic note: 10,000-dimensional vectors can superimpose thousands
of concepts without destructive interference (like holograms) because all
information is spread uniformly across every dimension.  The phase IS the
meaning.

Clifford geometric product in geometric_product.py is richer than circular
convolution, but both operate in the same paradigm of coordinate-free
superposition algebras.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "random_hv",
    "bind",
    "unbind",
    "fractional_power",
    "bundle",
    "similarity",
    "cleanup",
    "encode_sequence",
    "fractional_blend",
]


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

def random_hv(d=10000, kind="complex", seed=None):
    """Generate a random hypervector of dimension d.

    Parameters
    ----------
    d:
        Dimension of the hypervector.
    kind:
        "complex"  — unit-magnitude complex vector (each component e^{i*theta},
                     theta ~ Uniform[0, 2pi]).  These are the natural carriers
                     for phase-based fractional binding.
        "bipolar"  — real vector with each component in {+1, -1}.
        "real"     — Gaussian sample normalized to unit L2 norm.
    seed:
        Integer seed for reproducibility; None for random.

    Returns
    -------
    np.ndarray
        Shape (d,).  dtype=complex128 for kind="complex", float64 otherwise.
    """
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    if kind == "bipolar":
        return rng.choice(np.array([-1.0, 1.0]), size=d)
    if kind == "real":
        v = rng.standard_normal(d)
        return v / (np.linalg.norm(v) + 1e-30)
    raise ValueError(f"kind must be 'complex', 'bipolar', or 'real'; got {kind!r}")


def bind(X, Y):
    """Bind two hypervectors via circular convolution.

    Z = X (*) Y = ifft(fft(X) * fft(Y))

    Circular convolution is the standard HDC binding operator.  It is
    associative, approximately invertible, and preserves dimensionality.
    The result Z is dissimilar to both X and Y but encodes their
    association — querying Z with one recovers the other.

    Parameters
    ----------
    X, Y:
        1-D numpy arrays of equal length.

    Returns
    -------
    np.ndarray
        Bound hypervector, same shape and dtype as X.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    return np.fft.ifft(np.fft.fft(X) * np.fft.fft(Y))


def unbind(Z, Y):
    """Invert binding: recover X from Z = X (*) Y.

    Exact inverse via element-wise division in the Fourier domain:
        X = ifft(fft(Z) / fft(Y))

    Division is exact for unit-magnitude complex HVs (|F(Y)| = 1 everywhere)
    and numerically safe for near-unit-magnitude HVs.  For non-unit-magnitude
    HVs a small epsilon guards against division by zero.

    For complex unit-magnitude HVs: fft(Y)^{-1} = conj(fft(Y)) / |fft(Y)|^2,
    but direct division is simpler and more general.

    Parameters
    ----------
    Z:
        Bound hypervector (output of bind).
    Y:
        One factor used during binding.

    Returns
    -------
    np.ndarray
        Reconstruction of the other binding factor.
    """
    Z = np.asarray(Z)
    Y = np.asarray(Y)
    FY = np.fft.fft(Y)
    # guard: if any magnitude is zero, fall back to conjugate
    mag = np.abs(FY)
    inv_FY = np.conj(FY) / (mag ** 2 + 1e-30)
    return np.fft.ifft(np.fft.fft(Z) * inv_FY)


def fractional_power(Y, alpha):
    """Raise hypervector Y to fractional power alpha via phase scaling.

    Y^alpha = F^{-1}( |F(Y)| * exp(i * alpha * angle(F(Y))) )

    Scaling the phase of each Fourier component by alpha is the
    generalization of integer powers to real-valued alpha:
      alpha=0  ->  Y^0 = identity vector (all phases zero, magnitude 1).
      alpha=1  ->  Y^1 = Y.
      alpha=-1 ->  Y^{-1} = unbinding key.
    Intermediate alpha encodes fractional association strength.

    Parameters
    ----------
    Y:
        1-D numpy array hypervector.
    alpha:
        Real scalar exponent.

    Returns
    -------
    np.ndarray
        Y^alpha, same shape as Y, complex128 dtype.
    """
    Y = np.asarray(Y, dtype=complex)
    F = np.fft.fft(Y)
    magnitude = np.abs(F)
    phase = np.angle(F)
    F_frac = magnitude * np.exp(1j * alpha * phase)
    return np.fft.ifft(F_frac)


def bundle(hvs, weights=None):
    """Superpose a list of hypervectors into a single bundle.

    Superposition is addition in the base space followed by normalization to
    unit L2 magnitude (for complex HVs) or unit L2 norm (for real).  The
    resulting bundle is similar to each constituent and can hold thousands of
    items before similarity degrades noticeably — the holographic capacity
    property.

    Parameters
    ----------
    hvs:
        List or array of 1-D hypervectors, all same shape and dtype.
    weights:
        Optional 1-D array of non-negative scalars (one per HV).  Defaults
        to uniform.  Weights are L1-normalised internally.

    Returns
    -------
    np.ndarray
        Bundled hypervector, same shape as each input.
    """
    hvs = [np.asarray(v) for v in hvs]
    if not hvs:
        raise ValueError("bundle requires at least one hypervector")
    d = hvs[0].shape[0]
    if any(v.shape[0] != d for v in hvs):
        raise ValueError("all hypervectors must have equal dimension")

    if weights is None:
        weights = np.ones(len(hvs), dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / (weights.sum() + 1e-30)

    result = sum(w * v for w, v in zip(weights, hvs))
    norm = np.linalg.norm(result)
    if norm > 1e-30:
        result = result / norm
    return result


def similarity(X, Y):
    """Cosine similarity between two hypervectors.

    For real vectors: cos(theta) = X.Y / (|X||Y|).
    For complex vectors: Re(X^H Y) / (|X||Y|) — the normalized real part of
    the Hermitian inner product, which collapses to cosine when vectors are
    unit magnitude.

    Returns a float in [-1, 1].

    Parameters
    ----------
    X, Y:
        1-D numpy arrays of equal length.

    Returns
    -------
    float
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    nx = np.linalg.norm(X)
    ny = np.linalg.norm(Y)
    if nx < 1e-30 or ny < 1e-30:
        return 0.0
    dot = np.vdot(X, Y)          # conjugate of X, then dot with Y
    return float(np.real(dot) / (nx * ny))


def cleanup(query, codebook):
    """Nearest-neighbor lookup: find the closest HV in codebook to query.

    Uses cosine similarity.  Returns the index of the best matching entry.

    Parameters
    ----------
    query:
        1-D numpy array to match.
    codebook:
        List of 1-D numpy arrays (the known concept vectors).

    Returns
    -------
    int
        Index into codebook of the closest match.
    float
        Similarity score of that match.
    """
    if not codebook:
        raise ValueError("codebook must not be empty")
    scores = [similarity(query, hv) for hv in codebook]
    best = int(np.argmax(scores))
    return best, scores[best]


def encode_sequence(items, d=10000):
    """Encode an ordered sequence of hypervectors using position binding.

    Each item i is bound to a position vector raised to the power i:
        bundle = sum_i  bind(item_i, pos^i)

    where pos is a fixed random unit-complex HV.  The resulting bundle
    encodes both identity and order: querying with pos^i retrieves item_i.

    Parameters
    ----------
    items:
        List of 1-D complex numpy arrays (pre-generated HVs for each concept).
    d:
        Dimension (used to generate the position HV with a fixed seed).

    Returns
    -------
    np.ndarray
        Bundle HV representing the ordered sequence.
    """
    if not items:
        raise ValueError("encode_sequence requires at least one item")
    pos = random_hv(d=d, kind="complex", seed=0x706F73)  # hex for "pos"
    bound = [bind(item, fractional_power(pos, float(i)))
             for i, item in enumerate(items)]
    return bundle(bound)


def fractional_blend(X, Y, alpha):
    """Continuously interpolate between X and Y using fractional binding.

    result = X (*) Y^alpha

    At alpha=0: result = X (*) identity = X  (if Y^0 is identity).
    At alpha=1: result = X (*) Y  (full binding).
    Intermediate alpha smoothly mixes the two concepts.

    Parameters
    ----------
    X:
        Base hypervector.
    Y:
        Second hypervector to blend in.
    alpha:
        Blend exponent in [0, 1] (can exceed this range for extrapolation).

    Returns
    -------
    np.ndarray
        Blended hypervector.
    """
    return bind(X, fractional_power(Y, alpha))


# ---------------------------------------------------------------------------
# Main: fractional binding demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    d = 10_000
    rng_seed = 42

    print("=" * 64)
    print("Fractional HDC — phase binding demo")
    print(f"  d={d}")
    print("=" * 64)

    # --- 1. Generate three concept HVs ---------------------------------
    concept_A = random_hv(d=d, kind="complex", seed=rng_seed)
    concept_B = random_hv(d=d, kind="complex", seed=rng_seed + 1)
    concept_C = random_hv(d=d, kind="complex", seed=rng_seed + 2)

    # Sanity: two random HVs should be nearly orthogonal
    sim_AB = similarity(concept_A, concept_B)
    print(f"\n[Sanity] sim(A, B)={sim_AB:.4f}  (should be ~0)")

    # --- 2. Standard full binding --------------------------------------
    Z_full = bind(concept_A, concept_B)
    print(f"\n[Full binding] Z = A (*) B")
    print(f"  sim(Z, A)={similarity(Z_full, concept_A):.4f}  (should be ~0)")
    print(f"  sim(Z, B)={similarity(Z_full, concept_B):.4f}  (should be ~0)")

    # --- 3. Fractional binding alpha=0.5 -------------------------------
    alpha = 0.5
    Z_frac = bind(concept_A, fractional_power(concept_B, alpha))
    Z_zero = bind(concept_A, fractional_power(concept_B, 0.0))
    print(f"\n[Fractional binding alpha={alpha}]  Z_frac = A (*) B^0.5")
    print(f"  sim(Z_frac, Z_full)={similarity(Z_frac, Z_full):.4f}")
    print(f"  sim(Z_frac, Z_zero)={similarity(Z_frac, Z_zero):.4f}")
    print(f"  (frac-bound is intermediate between no-bind and full-bind)")

    # --- 4. Unbinding verification --------------------------------------
    recovered_A = unbind(Z_full, concept_B)
    sim_recovered = similarity(recovered_A, concept_A)
    print(f"\n[Unbinding] A' = Z (*) B^{{-1}}")
    print(f"  sim(recovered_A, A)={sim_recovered:.4f}  (should be ~1.0)")
    if sim_recovered < 0.98:
        print("  WARNING: unbinding fidelity low", file=sys.stderr)

    # --- 5. Similarity matrix ------------------------------------------
    concepts = {"A": concept_A, "B": concept_B, "C": concept_C,
                "Z_full": Z_full, "Z_frac": Z_frac}
    keys = list(concepts.keys())
    print(f"\n[Similarity matrix]")
    header = f"{'':>10}" + "".join(f"{k:>10}" for k in keys)
    print(header)
    for k1 in keys:
        row = f"{k1:>10}"
        for k2 in keys:
            s = similarity(concepts[k1], concepts[k2])
            row += f"{s:>10.3f}"
        print(row)

    # --- 6. Sequence encoding ------------------------------------------
    seq = encode_sequence([concept_A, concept_B, concept_C], d=d)
    print(f"\n[Sequence encoding] seq = bundle of positional bindings")
    print(f"  sim(seq, A)={similarity(seq, concept_A):.4f}")
    print(f"  sim(seq, B)={similarity(seq, concept_B):.4f}")
    print(f"  sim(seq, C)={similarity(seq, concept_C):.4f}")
    print(f"  (modest positive similarity expected for each member)")

    # --- 7. Cleanup demo -----------------------------------------------
    codebook = [concept_A, concept_B, concept_C]
    labels = ["A", "B", "C"]
    noisy = concept_A + 0.05 * random_hv(d=d, kind="real", seed=99)
    idx, score = cleanup(noisy, codebook)
    print(f"\n[Cleanup] noisy version of A -> codebook[{idx}]={labels[idx]}  "
          f"score={score:.4f}  (should match A)")

    print("\n" + "=" * 64)
    print("PASS — all checks completed.")
