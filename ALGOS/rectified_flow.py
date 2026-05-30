#!/usr/bin/env python3
"""
Rectified Flow Matching — straight-line generative transport.

Standard diffusion (DDPM) corrupts data along curved, variance-exploding paths,
requiring hundreds of denoising steps at inference.  Rectified Flow (Liu et al.
2022) replaces the curved schedule with a straight-line interpolant between source
and target distributions:

    Z_t = t * X_1 + (1 - t) * X_0,   t in [0, 1]

where X_0 ~ source (noise) and X_1 ~ target (data).  The constant-velocity
vector field that exactly follows these straight paths is simply (X_1 - X_0),
so the training objective is:

    L = E_{t, X_0, X_1} [ || v_theta(Z_t, t) - (X_1 - X_0) ||^2 ]

At inference, Euler integration from Z_0 = X_0 (noise) to Z_1 ~ X_1 (data)
along the learned field v_theta converges in far fewer steps than DDPM because
the target trajectories are already nearly straight.

Connection to Diffusion Forcing (Chen et al. 2024):
  DF assigns per-token independent noise levels t_i, letting the denoiser see a
  causally ordered mix of clean and noisy tokens simultaneously.  Rectified flow
  generalises the noising schedule to the full segment [0, 1] with straight paths.
  Combining the two: assign per-token t_i ~ [0, 1], compute Z_{t_i} via the
  rectified interpolant, and regress v_theta on (X_1 - X_0) at each token
  independently — straight paths + causal masking in one objective.

Optimal transport interpretation:
  Among all couplings of X_0 and X_1, straight paths minimise the expected kinetic
  energy E[||X_1 - X_0||^2], which is the Benamou–Brenier OT cost.  A rectified
  flow therefore learns an approximate OT map implicitly from unpaired samples.

Straightness metric:
  A perfectly straight trajectory has total arc-length equal to the chord length.
  straightness = 1 - arc / chord,  in [0, 1].  The closer to 1, the fewer Euler
  steps are needed for a given accuracy.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "interpolant",
    "flow_target",
    "flow_loss",
    "euler_solve",
    "midpoint_solve",
    "straightness",
    "sample_t",
]


# ---------------------------------------------------------------------------
# Core interpolant and target
# ---------------------------------------------------------------------------

def interpolant(x0, x1, t):
    """Straight-line interpolant: Z_t = t * x1 + (1 - t) * x0.

    Broadcasts t over a leading batch dimension.  If x0 has shape (B, d) and t
    has shape (B,), t is reshaped to (B, 1) so the multiply broadcasts correctly.

    Parameters
    ----------
    x0 : array-like, shape (..., d) — source samples (noise side, t=0).
    x1 : array-like, shape (..., d) — target samples (data side, t=1).
    t  : float or array-like, shape () or (...,) — interpolation parameter in [0, 1].

    Returns
    -------
    np.ndarray same shape as x0.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    t  = np.asarray(t,  dtype=np.float64)
    # Reshape t so it broadcasts against the last (feature) dimension.
    if x0.ndim > t.ndim:
        t = t[..., np.newaxis]
    return t * x1 + (1.0 - t) * x0


def flow_target(x0, x1):
    """Constant-velocity target for v_theta: simply (x1 - x0).

    The exact vector field that follows straight-line paths from x0 to x1 at
    unit speed is constant along the path, equal to the displacement.

    Parameters
    ----------
    x0 : array-like, shape (B, d).
    x1 : array-like, shape (B, d).

    Returns
    -------
    np.ndarray shape (B, d) — the constant velocity direction.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    return x1 - x0


def flow_loss(v_pred, x0, x1):
    """MSE between predicted velocity and the straight-line target.

    L = mean over batch of || v_pred - (x1 - x0) ||^2

    Parameters
    ----------
    v_pred : array-like, shape (B, d) — model velocity prediction at some Z_t, t.
    x0     : array-like, shape (B, d) — source samples.
    x1     : array-like, shape (B, d) — target samples.

    Returns
    -------
    float scalar — mean squared error over the batch.
    """
    v_pred = np.asarray(v_pred, dtype=np.float64)
    target = flow_target(x0, x1)
    diff   = v_pred - target
    return float(np.mean(diff ** 2))


# ---------------------------------------------------------------------------
# ODE solvers
# ---------------------------------------------------------------------------

def euler_solve(v_fn, x0, steps=10):
    """Euler integration of the learned vector field v_fn from t=0 to t=1.

    Discretises [0, 1] into `steps` equal intervals; at each step applies:
        z_{k+1} = z_k + (1/steps) * v_fn(z_k, t_k)

    Parameters
    ----------
    v_fn  : callable (z, t) -> np.ndarray — learned velocity field.
              z shape (B, d), t scalar float in [0, 1].
    x0    : array-like, shape (B, d) — initial state (source noise).
    steps : int — number of Euler integration steps.

    Returns
    -------
    np.ndarray shape (steps+1, B, d) — full trajectory including start and end.
    """
    x0   = np.asarray(x0, dtype=np.float64)
    dt   = 1.0 / steps
    ts   = np.linspace(0.0, 1.0 - dt, steps)          # t values at which v is evaluated
    traj = np.empty((steps + 1,) + x0.shape, dtype=np.float64)
    traj[0] = x0
    z = x0.copy()
    for k, t in enumerate(ts):
        v = v_fn(z, float(t))
        z = z + dt * np.asarray(v, dtype=np.float64)
        traj[k + 1] = z
    return traj


def midpoint_solve(v_fn, x0, steps=10):
    """Heun (improved Euler / midpoint) solver — higher accuracy at same step count.

    Each step is a two-stage Runge-Kutta:
        k1 = v_fn(z_k,         t_k)
        k2 = v_fn(z_k + dt*k1, t_k + dt)
        z_{k+1} = z_k + 0.5 * dt * (k1 + k2)

    This halves the local truncation error relative to plain Euler, which is
    particularly valuable for rectified flow because the field is smooth and
    nearly constant — Heun converges with 2-4 steps where Euler needs 8-10.

    Parameters
    ----------
    v_fn  : callable (z, t) -> np.ndarray.
    x0    : array-like, shape (B, d).
    steps : int — number of Heun steps.

    Returns
    -------
    np.ndarray shape (steps+1, B, d).
    """
    x0   = np.asarray(x0, dtype=np.float64)
    dt   = 1.0 / steps
    ts   = np.linspace(0.0, 1.0 - dt, steps)
    traj = np.empty((steps + 1,) + x0.shape, dtype=np.float64)
    traj[0] = x0
    z = x0.copy()
    for k, t in enumerate(ts):
        k1 = np.asarray(v_fn(z,          float(t)),      dtype=np.float64)
        k2 = np.asarray(v_fn(z + dt * k1, float(t + dt)), dtype=np.float64)
        z  = z + 0.5 * dt * (k1 + k2)
        traj[k + 1] = z
    return traj


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def straightness(trajectory):
    """Measure how straight a trajectory is.

    straightness = chord_length / arc_length

    where arc_length = sum of step norms and chord_length = start-to-end distance.

    Computed per sample and averaged over the batch.

    A perfectly straight path has arc == chord, so straightness == 1.0.
    A highly curved path has arc >> chord, so straightness << 1.0.
    The formula is numerically stable even for near-straight paths (arc ~= chord)
    because division never subtracts nearly-equal quantities.

    Parameters
    ----------
    trajectory : array-like, shape (T+1, B, d) or (T+1, d).
                 Full trajectory from euler_solve / midpoint_solve.
                 If 2-D (T+1, d), treated as a single sample.

    Returns
    -------
    float in (0, 1].  Samples where start == end (arc ~ 0) contribute 1.0
    (degenerate still point, trivially "straight").
    """
    traj = np.asarray(trajectory, dtype=np.float64)  # (T+1, ..., d)
    # Normalise to 3-D: (T+1, B, d) — works for both batched and unbatched.
    if traj.ndim == 2:
        traj = traj[:, np.newaxis, :]          # (T+1, 1, d)
    # segs: (T, B, d)
    segs = np.diff(traj, axis=0)
    # Per-sample arc-length: sum of step norms over T.  shape (B,).
    seg_norms = np.linalg.norm(segs, axis=-1)      # (T, B)
    arc_lens  = seg_norms.sum(axis=0)              # (B,)
    # Per-sample chord length.  shape (B,).
    chords = np.linalg.norm(traj[-1] - traj[0], axis=-1)  # (B,)
    # chord/arc in [0, 1]; degenerate (arc~0 or chord~0) treated as 1.0.
    valid      = arc_lens > 1e-12
    per_sample = np.where(valid, chords / np.where(valid, arc_lens, 1.0), 1.0)
    per_sample = np.clip(per_sample, 0.0, 1.0)
    return float(per_sample.mean())


# ---------------------------------------------------------------------------
# Time sampling
# ---------------------------------------------------------------------------

def sample_t(batch_size, mode='uniform'):
    """Sample t values in [0, 1] for flow matching training.

    Parameters
    ----------
    batch_size : int — number of t values to draw.
    mode       : str — 'uniform' or 'logit-normal'.
                 'logit-normal' concentrates mass near t=0 and t=1, where the
                 flow matching loss is harder to learn; it improves convergence
                 on real image datasets (Esser et al. 2024 / Stable Diffusion 3).

    Returns
    -------
    np.ndarray shape (batch_size,) with values in (0, 1).
    """
    if mode == 'uniform':
        return np.random.uniform(0.0, 1.0, size=batch_size)
    elif mode == 'logit-normal':
        # Draw u ~ N(0, 1), map via sigmoid: t = sigma(u) = 1 / (1 + exp(-u)).
        # This produces a distribution on (0,1) that is more uniform in logit
        # space — heavier tails toward 0 and 1 than Beta(0.5,0.5).
        u = np.random.normal(0.0, 1.0, size=batch_size)
        return 1.0 / (1.0 + np.exp(-u))
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'uniform' or 'logit-normal'.")


# ---------------------------------------------------------------------------
# Main: linear toy problem
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Source: X0 ~ N(0, I)   Target: X1 ~ N(3, 0.5*I)   dim=2   batch=256
    # v_theta is a linear model: v(z, t) = W @ z + b
    # Trained with SGD for 200 steps; straightness printed every 50 steps.
    # -----------------------------------------------------------------------

    rng = np.random.default_rng(seed=0)
    np.random.seed(0)

    DIM       = 2
    BATCH     = 256
    STEPS     = 200
    LR        = 0.05
    EULER_N   = 20          # inference steps for straightness measurement

    # Linear v_theta: v(z, t) = z @ W.T + b   (W shape (DIM, DIM), b shape (DIM,))
    W = rng.standard_normal((DIM, DIM)) * 0.01
    b = rng.standard_normal(DIM)        * 0.01

    def v_theta(z, t):
        """Linear field: ignores t (constant field ansatz for this toy problem)."""
        return z @ W.T + b

    def draw_batch():
        x0 = rng.standard_normal((BATCH, DIM))
        x1 = rng.standard_normal((BATCH, DIM)) * np.sqrt(0.5) + 3.0
        return x0, x1

    print("Rectified Flow Matching — linear toy demo")
    print(f"  Source: N(0, I)   Target: N(3, 0.5*I)   dim={DIM}   batch={BATCH}")
    print(f"  v_theta: linear (W, b)   steps={STEPS}   lr={LR}")
    print()
    print(f"{'Step':>6}  {'Loss':>10}  {'Straightness':>13}  {'Mean Z1':>10}")
    print("-" * 48)

    for step in range(1, STEPS + 1):
        x0, x1 = draw_batch()
        t_vals  = sample_t(BATCH, mode='logit-normal')           # (B,)
        z_t     = interpolant(x0, x1, t_vals)                    # (B, DIM)
        target  = flow_target(x0, x1)                            # (B, DIM)

        v_pred  = z_t @ W.T + b                                  # forward pass

        # MSE loss
        loss_val = flow_loss(v_pred, x0, x1)

        # Gradients (manual for numpy-only linear model)
        residual = v_pred - target                                # (B, DIM)
        dL_dv    = 2.0 * residual / BATCH                        # (B, DIM)
        dW       = dL_dv.T @ z_t                                 # (DIM, DIM)
        db       = dL_dv.sum(axis=0)                             # (DIM,)

        W -= LR * dW
        b -= LR * db

        if step % 50 == 0 or step == 1:
            # Measure straightness on a held-out batch using the current field.
            x0_eval, _ = draw_batch()
            traj = euler_solve(v_theta, x0_eval, steps=EULER_N)  # (EULER_N+1, B, DIM)
            s    = straightness(traj)
            mean_z1 = float(np.mean(traj[-1]))
            print(f"{step:>6}  {loss_val:>10.6f}  {s:>13.6f}  {mean_z1:>10.4f}")

    print()
    # Final evaluation with Heun solver (fewer steps, higher accuracy).
    x0_final, _ = draw_batch()
    traj_euler  = euler_solve(v_theta,    x0_final, steps=4)
    traj_heun   = midpoint_solve(v_theta, x0_final, steps=4)
    s_euler = straightness(traj_euler)
    s_heun  = straightness(traj_heun)
    mean_euler = float(np.mean(traj_euler[-1]))
    mean_heun  = float(np.mean(traj_heun[-1]))
    print("Final inference comparison (4 steps):")
    print(f"  Euler   straightness={s_euler:.6f}  mean_z1={mean_euler:.4f}")
    print(f"  Heun    straightness={s_heun:.6f}  mean_z1={mean_heun:.4f}")
    print(f"  Target mean = 3.0  (both should be close)")
