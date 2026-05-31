#!/usr/bin/env python3
"""
Joint Embedding Predictive Architecture (JEPA) — Energy-Based Latent Variable Prediction.

Standard generative models predict raw observations (pixels, tokens) from a latent variable.
They measure error in observation space, forcing the model to account for irreducible
perceptual variance (lighting, texture, pose jitter) that carries no semantic content.

JEPA (LeCun 2022) reframes prediction entirely in representation space:

    E(x, y, z) = || s_theta(x) - p_phi(s_theta(y), z) ||_2^2

  s_theta : encoder — maps an observation to an abstract representation (unit sphere).
  p_phi   : predictor — maps (encoded past s_theta(y), latent z) to predicted representation
            of the future observation x.
  z       : latent variable encoding unobserved causes — elapsed time, hidden action,
            unknown dynamics. It is the physics you cannot see.
  E = 0   : the predicted representation perfectly matches the actual representation.

Learning signal: minimize E over the empirical data distribution.  No pixel reconstruction,
no generative log-likelihood, no adversarial discriminator.

Why this matters:
  - The encoder is forced to build a representation that supports *prediction*, not
    reconstruction.  Irrelevant pixel variance is discarded; geometry is preserved.
  - The latent z models uncertainty about what happens between y and x without requiring
    the model to hallucinate the pixels of that uncertainty.
  - The visual cortex does something similar: you predict abstract geometric outcomes,
    not the raw photon map.

Connection to variational_free_energy.py:
  Free Energy (Friston) = prediction error + complexity (KL divergence in probabilistic space).
  JEPA energy = prediction error in representation space (L2, no KL term).
  Same principle — minimize surprise — different geometry.  Free Energy is Bayesian inference
  over generative models; JEPA is metric learning over representation manifolds.
  As representations become more abstract, the two converge in spirit.

Collapse trap:
  A degenerate encoder s_theta(x) = constant for all x achieves E = 0 trivially.
  This is representation collapse.  VICReg regularization (Bardes et al. 2022)
  penalizes low variance and off-diagonal covariance to keep representations spread.

LUCIDOTA role — evidence representation health monitor:
  1. Embedding audit: run JEPA energy over a batch of corpus_chunk embeddings
     (from lucidota_korpus.corpus_chunk.embedding JSONB). High energy = good
     representation (predictions are wrong → model is uncertain → learning).
     Near-zero energy across all pairs = COLLAPSE ALERT — BGE fleet or the
     embedding pipeline is returning degenerate representations.
  2. Staging candidate scoring: given a proposed_term embedding (from staging_packet)
     and a context window of recent staging candidates, JEPA energy scores how
     "surprising" the candidate is relative to the context. High energy = novel
     claim worth closer inspection. Low energy = redundant / already staged.
  3. Integration point: scripts/jepa_evidence_wire.py (planned) queries
     lucidota_korpus.corpus_chunk for recent embeddings, runs vicreg_loss /
     energy_single to detect collapse and score novelty, writes receipt to
     05_OUTPUTS/runtime/jepa_health_<ts>.json.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "encoder",
    "predictor",
    "jepa_energy",
    "jepa_loss_batch",
    "collapse_check",
    "vicreg_regularizer",
    "init_jepa",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(v):
    """Project vector v onto the unit sphere.  Returns v unchanged if near-zero."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


# ---------------------------------------------------------------------------
# Core JEPA components
# ---------------------------------------------------------------------------

def encoder(x, W_enc, b_enc=None):
    """Linear encoder: s = W_enc @ x + b_enc, normalized to unit sphere.

    The encoder maps raw observations into an abstract representation space.
    Normalization keeps all representations on the unit sphere — the geometry
    that the predictor must learn to navigate.

    Parameters
    ----------
    x     : array shape (d_in,)
    W_enc : array shape (d_rep, d_in)
    b_enc : array shape (d_rep,) or None — zero bias if None

    Returns
    -------
    array shape (d_rep,) — unit-norm representation
    """
    x = np.asarray(x, dtype=float)
    W_enc = np.asarray(W_enc, dtype=float)
    s = W_enc @ x
    if b_enc is not None:
        s = s + np.asarray(b_enc, dtype=float)
    return _normalize(s)


def predictor(s_y, z, W_pred, b_pred=None):
    """Linear predictor: p = W_pred @ concat(s_y, z) + b_pred, normalized.

    The predictor maps an encoded past observation s_y and a latent variable z
    to a predicted representation of the future observation.  It operates entirely
    in representation space — it never touches raw pixels.

    Parameters
    ----------
    s_y    : array shape (d_rep,)   — encoded past, from encoder()
    z      : array shape (d_latent,) — latent variable (hidden cause / action)
    W_pred : array shape (d_rep, d_rep + d_latent)
    b_pred : array shape (d_rep,) or None

    Returns
    -------
    array shape (d_rep,) — predicted future representation, unit-norm
    """
    s_y = np.asarray(s_y, dtype=float)
    z = np.asarray(z, dtype=float)
    W_pred = np.asarray(W_pred, dtype=float)
    inp = np.concatenate([s_y, z])
    p = W_pred @ inp
    if b_pred is not None:
        p = p + np.asarray(b_pred, dtype=float)
    return _normalize(p)


def jepa_energy(x, y, z, W_enc, W_pred, b_enc=None, b_pred=None):
    """Compute the JEPA energy E(x, y, z).

    E(x, y, z) = || s_theta(x) - p_phi(s_theta(y), z) ||_2^2

    E = 0 iff the predicted representation of x (given past y and latent z)
    perfectly matches the actual representation of x.  The energy is a measure
    of prediction error in abstract representation space.

    Parameters
    ----------
    x      : array shape (d_in,)    — future observation
    y      : array shape (d_in,)    — past observation
    z      : array shape (d_latent,) — latent variable
    W_enc  : array shape (d_rep, d_in)
    W_pred : array shape (d_rep, d_rep + d_latent)
    b_enc  : array shape (d_rep,) or None
    b_pred : array shape (d_rep,) or None

    Returns
    -------
    float — scalar energy in [0, 4] on unit sphere (max L2^2 between two unit vecs is 4)
    """
    s_x = encoder(x, W_enc, b_enc)
    s_y = encoder(y, W_enc, b_enc)
    p_hat = predictor(s_y, z, W_pred, b_pred)
    diff = s_x - p_hat
    return float(np.dot(diff, diff))


def jepa_loss_batch(X_batch, Y_batch, Z_batch, W_enc, W_pred, b_enc=None, b_pred=None):
    """Mean JEPA energy over a mini-batch.

    Parameters
    ----------
    X_batch : array shape (batch, d_in)    — future observations
    Y_batch : array shape (batch, d_in)    — past observations
    Z_batch : array shape (batch, d_latent) — latent variables
    W_enc   : array shape (d_rep, d_in)
    W_pred  : array shape (d_rep, d_rep + d_latent)
    b_enc   : array shape (d_rep,) or None
    b_pred  : array shape (d_rep,) or None

    Returns
    -------
    float — mean energy across the batch
    """
    X_batch = np.asarray(X_batch, dtype=float)
    Y_batch = np.asarray(Y_batch, dtype=float)
    Z_batch = np.asarray(Z_batch, dtype=float)
    batch = X_batch.shape[0]
    total = 0.0
    for i in range(batch):
        total += jepa_energy(X_batch[i], Y_batch[i], Z_batch[i],
                             W_enc, W_pred, b_enc, b_pred)
    return total / batch


# ---------------------------------------------------------------------------
# Collapse diagnostics
# ---------------------------------------------------------------------------

def collapse_check(W_enc, X_samples):
    """Check for representation collapse: variance of encoded representations.

    A collapsed encoder maps all inputs to (approximately) the same point on
    the unit sphere.  This zeroes the energy trivially without learning anything.
    The diagnostic: compute the mean per-dimension variance of the encoded batch.

    Parameters
    ----------
    W_enc     : array shape (d_rep, d_in)
    X_samples : array shape (n_samples, d_in)

    Returns
    -------
    (mean_variance, is_collapsed)
      mean_variance : float — mean variance across representation dimensions
      is_collapsed  : bool  — True if mean_variance < 0.01
    """
    X_samples = np.asarray(X_samples, dtype=float)
    W_enc = np.asarray(W_enc, dtype=float)
    reps = np.array([encoder(x, W_enc) for x in X_samples])  # (n, d_rep)
    mean_var = float(np.mean(np.var(reps, axis=0)))
    return mean_var, mean_var < 0.01


# ---------------------------------------------------------------------------
# VICReg regularizer
# ---------------------------------------------------------------------------

def vicreg_regularizer(Z1, Z2, lambda_var=25.0, lambda_cov=1.0):
    """VICReg regularization to prevent representation collapse.

    Bardes et al. (2022) VICReg enforces:
      - Variance: each dimension of the representation must have std >= 1.
        Penalizes collapsing all inputs onto a single point.
      - Covariance: off-diagonal entries of the feature covariance matrix
        are pushed toward zero.  Decorrelates dimensions, preventing
        all information from piling into a single direction.

    Note: the similarity (invariance) term is the JEPA energy itself and is
    not included here — this is the regularization complement only.

    Parameters
    ----------
    Z1         : array shape (batch, d) — first set of representations
    Z2         : array shape (batch, d) — second set (e.g. predicted)
    lambda_var : float — weight on variance term (VICReg default: 25)
    lambda_cov : float — weight on covariance term (VICReg default: 1)

    Returns
    -------
    float — scalar regularization loss
    """
    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)
    batch, d = Z1.shape

    def _var_term(Z):
        # Per-dimension std; penalize when std < 1 via hinge
        std = np.sqrt(np.var(Z, axis=0) + 1e-4)
        return float(np.mean(np.maximum(0.0, 1.0 - std)))

    def _cov_term(Z):
        # Normalized covariance; penalize off-diagonal entries
        Z_centered = Z - Z.mean(axis=0)
        cov = (Z_centered.T @ Z_centered) / (batch - 1)       # (d, d)
        diag_mask = np.eye(d, dtype=bool)
        off_diag = cov[~diag_mask]
        return float(np.sum(off_diag ** 2) / d)

    var_loss = _var_term(Z1) + _var_term(Z2)
    cov_loss = _cov_term(Z1) + _cov_term(Z2)
    return lambda_var * var_loss + lambda_cov * cov_loss


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_jepa(d_in, d_rep, d_latent, scale=0.01, seed=0):
    """Initialize JEPA encoder and predictor weight matrices.

    Small-scale random init — avoids large initial energy gradients while
    keeping representations spread (not yet collapsed).

    Parameters
    ----------
    d_in     : int   — input observation dimension
    d_rep    : int   — representation dimension
    d_latent : int   — latent variable dimension
    scale    : float — std dev for weight initialization
    seed     : int

    Returns
    -------
    dict with keys:
      "W_enc"  : array shape (d_rep, d_in)
      "W_pred" : array shape (d_rep, d_rep + d_latent)
      "b_enc"  : array shape (d_rep,)
      "b_pred" : array shape (d_rep,)
    """
    rng = np.random.default_rng(seed)
    return {
        "W_enc":  rng.normal(0.0, scale, size=(d_rep, d_in)),
        "W_pred": rng.normal(0.0, scale, size=(d_rep, d_rep + d_latent)),
        "b_enc":  np.zeros(d_rep),
        "b_pred": np.zeros(d_rep),
    }


# ---------------------------------------------------------------------------
# Main: 2D toy world — rotation prediction
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------ setup
    # World: a point moves on the unit circle.
    # Past observation y = [cos(theta), sin(theta)].
    # Latent z = [delta_angle] — the rotation the agent does not observe.
    # Future observation x = y rotated by z.
    # JEPA learns: given (y, z), predict the representation of x.
    # ------------------------------------------------------------------ setup

    rng = np.random.default_rng(seed=42)

    d_in = 2       # 2D point on circle
    d_rep = 8      # representation dimension
    d_latent = 1   # latent: angle delta (scalar)
    n_train = 512
    batch_size = 64
    n_epochs = 200
    lr = 0.05
    lambda_var = 5.0
    lambda_cov = 0.5

    # Generate dataset: random angles and random deltas
    thetas = rng.uniform(0, 2 * np.pi, n_train)
    deltas = rng.uniform(-np.pi / 4, np.pi / 4, n_train)
    Y_data = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    X_data = np.stack([np.cos(thetas + deltas), np.sin(thetas + deltas)], axis=1)
    Z_data = deltas[:, None]  # shape (n_train, 1)

    # Initialize weights
    params = init_jepa(d_in, d_rep, d_latent, scale=0.1, seed=0)
    W_enc = params["W_enc"]
    W_pred = params["W_pred"]
    b_enc = params["b_enc"]
    b_pred = params["b_pred"]

    print("JEPA Energy — 2D rotation world")
    print(f"  d_in={d_in}  d_rep={d_rep}  d_latent={d_latent}")
    print(f"  n_train={n_train}  batch={batch_size}  epochs={n_epochs}  lr={lr}")
    print()

    # Finite-difference gradient descent (clean, no autograd)
    eps_fd = 1e-4

    def total_loss(W_enc_, W_pred_, X_b, Y_b, Z_b):
        base = jepa_loss_batch(X_b, Y_b, Z_b, W_enc_, W_pred_, b_enc, b_pred)
        # VICReg: encode both batches, compute regularizer
        Z1 = np.array([encoder(x, W_enc_, b_enc) for x in X_b])
        Z2 = np.array([encoder(y, W_enc_, b_enc) for y in Y_b])
        reg = vicreg_regularizer(Z1, Z2, lambda_var=lambda_var, lambda_cov=lambda_cov)
        return base + reg

    history = []
    n_batches = n_train // batch_size

    for epoch in range(n_epochs):
        idx = rng.permutation(n_train)
        epoch_loss = 0.0

        for b in range(n_batches):
            sl = idx[b * batch_size: (b + 1) * batch_size]
            X_b = X_data[sl]
            Y_b = Y_data[sl]
            Z_b = Z_data[sl]

            loss0 = total_loss(W_enc, W_pred, X_b, Y_b, Z_b)
            epoch_loss += loss0

            # Finite-difference gradient for W_enc
            grad_enc = np.zeros_like(W_enc)
            for i in range(W_enc.shape[0]):
                for j in range(W_enc.shape[1]):
                    W_enc[i, j] += eps_fd
                    lp = total_loss(W_enc, W_pred, X_b, Y_b, Z_b)
                    W_enc[i, j] -= eps_fd
                    grad_enc[i, j] = (lp - loss0) / eps_fd

            # Finite-difference gradient for W_pred
            grad_pred = np.zeros_like(W_pred)
            for i in range(W_pred.shape[0]):
                for j in range(W_pred.shape[1]):
                    W_pred[i, j] += eps_fd
                    lp = total_loss(W_enc, W_pred, X_b, Y_b, Z_b)
                    W_pred[i, j] -= eps_fd
                    grad_pred[i, j] = (lp - loss0) / eps_fd

            W_enc -= lr * grad_enc
            W_pred -= lr * grad_pred

        avg_loss = epoch_loss / n_batches
        history.append(avg_loss)

        if epoch % 40 == 0 or epoch == n_epochs - 1:
            mean_var, collapsed = collapse_check(W_enc, X_data[:200])
            print(f"  epoch {epoch:3d}  loss={avg_loss:.6f}  "
                  f"rep_var={mean_var:.4f}  collapsed={collapsed}")

    # ------------------------------------------------------------------ final
    print()
    final_energy = jepa_loss_batch(X_data[:100], Y_data[:100], Z_data[:100],
                                   W_enc, W_pred, b_enc, b_pred)
    mean_var, collapsed = collapse_check(W_enc, X_data)
    energy_drop = history[0] / (history[-1] + 1e-12)

    print(f"Final JEPA energy (100 samples): {final_energy:.6f}")
    print(f"Representation variance         : {mean_var:.6f}")
    print(f"Collapsed                       : {collapsed}")
    print(f"Energy reduction ratio          : {energy_drop:.2f}x")
    print()

    # Verify energy formula on a single example
    theta0 = 0.3
    delta0 = 0.5
    y0 = np.array([np.cos(theta0), np.sin(theta0)])
    x0 = np.array([np.cos(theta0 + delta0), np.sin(theta0 + delta0)])
    z0 = np.array([delta0])
    e_single = jepa_energy(x0, y0, z0, W_enc, W_pred, b_enc, b_pred)
    s_x0 = encoder(x0, W_enc, b_enc)
    s_y0 = encoder(y0, W_enc, b_enc)
    p0 = predictor(s_y0, z0, W_pred, b_pred)
    print("Single-example check:")
    print(f"  theta={theta0:.2f}  delta={delta0:.2f}")
    print(f"  ||s_theta(x)||={np.linalg.norm(s_x0):.6f}  (unit norm)")
    print(f"  ||p_phi(s_y,z)||={np.linalg.norm(p0):.6f}  (unit norm)")
    print(f"  E(x,y,z)={e_single:.6f}")
    print()
    print("Note: Free Energy (Friston) minimizes prediction error + KL in prob space.")
    print("      JEPA energy minimizes prediction error in representation space (L2).")
    print("      Same imperative — minimize surprise — different geometry.")
