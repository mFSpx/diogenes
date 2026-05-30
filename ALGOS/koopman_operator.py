#!/usr/bin/env python3
"""
Koopman Operator — linearize nonlinear dynamics in observable space.

Given a nonlinear dynamical system  x_{t+1} = f(x_t),  the Koopman operator K
acts on *observables* (functions of state) rather than on the state directly:

    K psi(x) = psi(f(x))

Because K is linear (even when f is not), the evolution of any observable
satisfies  psi(x_t) = K^t psi(x_0),  a pure matrix power.  In practice we
pick a finite dictionary of observables and approximate K from data using
Dynamic Mode Decomposition (DMD).

DMD sketch
----------
Given snapshot matrices  X  and  X' = f(X)  (both shape d x T):

1. Thin SVD of X:  X = U S V^T,  truncated to rank r.
2. Project: K_tilde = U^T X' V S^{-1}   (r x r reduced operator).
3. Eigendecompose K_tilde: K_tilde W = W Lambda.
4. Lift eigenvectors back: Phi = X' V S^{-1} W   (DMD modes, d x r).
5. Full-rank approximation: K_approx = Phi diag(Lambda) pinv(Phi).

Forecast
--------
    psi(x_t) = K^t psi(x_0)

For state-space use with no lifting this is just  x_t ≈ K^t x_0.

Observable lifting (polynomial)
--------------------------------
observable_lift(x, degree=2) maps a d-dimensional state to a 1-D vector
containing  [x, x^2, cross-terms x_i*x_j for i<j], turning the system into
a higher-dimensional linear one before applying DMD.

References
----------
- Koopman (1931), "Hamiltonian Systems and Transformations in Hilbert Space."
- Schmid (2010), "Dynamic Mode Decomposition of Numerical and Experimental Data."
- Mezic (2013), "Analysis of Fluid Flows via Spectral Properties of the Koopman
  Operator."
- Brunton et al. (2022), "Modern Koopman Theory for Dynamical Systems."
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "dmd",
    "fit_koopman",
    "koopman_forecast",
    "reconstruction_error",
    "observable_lift",
]


# ---------------------------------------------------------------------------
# Core DMD
# ---------------------------------------------------------------------------

def dmd(X, X_prime, rank=10):
    """Standard Dynamic Mode Decomposition.

    Parameters
    ----------
    X : array (d, T)
        Snapshot matrix; column t is the state at time t.
    X_prime : array (d, T)
        Shifted snapshot matrix; column t is the state at time t+1.
    rank : int
        Truncation rank for the SVD.  Clamped to min(d, T) if too large.

    Returns
    -------
    eigenvalues : ndarray (r,) complex
        DMD eigenvalues (Koopman eigenvalues in the truncated basis).
    eigenvectors : ndarray (d, r) complex
        DMD modes (columns), full-state representation.
    K_approx : ndarray (d, d) complex
        Full approximate Koopman matrix  K ≈ Phi diag(lambda) pinv(Phi).
    """
    X = np.asarray(X, dtype=float)
    X_prime = np.asarray(X_prime, dtype=float)
    d, T = X.shape
    r = min(rank, d, T)

    # Step 1 — thin SVD of X, truncated to rank r.
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    U = U[:, :r]
    S = S[:r]
    Vt = Vt[:r, :]

    S_inv = np.diag(1.0 / S)

    # Step 2 — reduced (projected) operator  K_tilde = U^T X' V S^{-1}.
    K_tilde = U.T @ X_prime @ Vt.T @ S_inv   # (r, r)

    # Step 3 — eigendecompose K_tilde.
    eigenvalues, W = np.linalg.eig(K_tilde)   # W columns are right eigenvectors

    # Step 4 — lift eigenvectors back to full state space (DMD modes).
    Phi = X_prime @ Vt.T @ S_inv @ W          # (d, r)

    # Step 5 — reconstruct approximate K.
    # K_approx x ≈ Phi diag(lambda) pinv(Phi) x
    Phi_pinv = np.linalg.pinv(Phi)
    K_approx = Phi @ np.diag(eigenvalues) @ Phi_pinv   # (d, d)

    return eigenvalues, Phi, K_approx


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit_koopman(trajectory, rank=10):
    """Fit a Koopman operator from a state trajectory.

    Parameters
    ----------
    trajectory : array (T, d)
        Time-ordered sequence of states.
    rank : int
        DMD truncation rank.

    Returns
    -------
    dict with keys:
        K            - (d, d) complex ndarray, approximate Koopman matrix
        eigenvalues  - (r,) complex ndarray
        eigenvectors - (d, r) complex ndarray  (DMD modes)
        rank         - int, effective rank used
    """
    trajectory = np.asarray(trajectory, dtype=float)
    T, d = trajectory.shape
    if T < 2:
        raise ValueError("trajectory must have at least 2 time steps")

    # Build snapshot matrices: columns are time steps.
    X = trajectory[:-1].T      # (d, T-1)
    X_prime = trajectory[1:].T # (d, T-1)

    r = min(rank, d, T - 1)
    eigenvalues, eigenvectors, K = dmd(X, X_prime, rank=r)

    return {
        "K": K,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "rank": r,
    }


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

def koopman_forecast(x0, model, steps):
    """Forecast forward in time using the Koopman matrix.

    Applies  x_{t+1} = K x_t  iteratively, which is equivalent to
    K^t x_0 in exact arithmetic.

    Parameters
    ----------
    x0 : array (d,)
        Initial state.
    model : dict
        Output of fit_koopman.
    steps : int
        Number of steps to forecast.  Returns steps+1 points including x0.

    Returns
    -------
    ndarray (steps, d)  — forecast states at times 1 … steps (x0 excluded).
    """
    K = model["K"]
    x = np.asarray(x0, dtype=complex)
    out = np.empty((steps, len(x)), dtype=complex)
    for t in range(steps):
        x = K @ x
        out[t] = x
    return out.real


# ---------------------------------------------------------------------------
# Reconstruction error
# ---------------------------------------------------------------------------

def reconstruction_error(trajectory, model):
    """MSE between actual trajectory and Koopman forecast from trajectory[0].

    Parameters
    ----------
    trajectory : array (T, d)
        Ground-truth states.
    model : dict
        Output of fit_koopman.

    Returns
    -------
    float — mean squared error over all time steps 1 … T-1.
    """
    trajectory = np.asarray(trajectory, dtype=float)
    T, d = trajectory.shape
    steps = T - 1
    predicted = koopman_forecast(trajectory[0], model, steps)   # (steps, d)
    actual = trajectory[1:]                                       # (steps, d)
    return float(np.mean((predicted - actual) ** 2))


# ---------------------------------------------------------------------------
# Observable lifting
# ---------------------------------------------------------------------------

def observable_lift(x, degree=2):
    """Polynomial feature lift for Koopman observable dictionary.

    Constructs  [x, x^2, x_i * x_j for all i < j]  (degree-2 monomials).
    Higher degree raises the dimension but can improve linearity of the lift.

    Parameters
    ----------
    x : array (d,)
        Raw state vector.
    degree : int
        Polynomial degree.  Currently only degree 1 and 2 are implemented;
        higher values fall back to degree 2 with a warning.

    Returns
    -------
    ndarray (n_features,) — lifted observable vector.
    """
    x = np.asarray(x, dtype=float).ravel()
    d = len(x)

    if degree == 1:
        return x.copy()

    if degree > 2:
        import warnings
        warnings.warn(
            f"observable_lift: degree {degree} not implemented; using degree=2.",
            stacklevel=2,
        )

    # degree == 2
    parts = [x, x ** 2]
    if d > 1:
        cross = np.array([x[i] * x[j] for i in range(d) for j in range(i + 1, d)])
        parts.append(cross)

    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Main: demonstration on a 2-D nonlinear oscillator
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    rng = np.random.default_rng(42)

    # 2-D nonlinear oscillator:  x_{t+1} = R(omega) x_t + epsilon * nonlinear_term
    # where R(omega) is a rotation by omega radians and the nonlinear term is
    # a small cubic coupling.  At epsilon=0 this is linear; at epsilon>0 Koopman
    # must work harder.
    omega = 0.3          # radians per step
    epsilon = 0.05       # nonlinearity strength
    T_train = 200
    T_forecast = 20

    def step(x):
        c, s = np.cos(omega), np.sin(omega)
        R = np.array([[c, -s], [s, c]])
        nl = np.array([x[1] ** 3, -x[0] ** 3])
        return R @ x + epsilon * nl

    # Generate training trajectory.
    x = np.array([1.0, 0.0])
    traj = [x.copy()]
    for _ in range(T_train - 1):
        x = step(x)
        traj.append(x.copy())
    traj = np.array(traj)   # (T_train, 2)

    # Fit Koopman.
    model = fit_koopman(traj, rank=10)
    print(f"Koopman fit: rank={model['rank']}, "
          f"K shape={model['K'].shape}, "
          f"eigenvalue magnitudes (first 5): "
          f"{np.abs(model['eigenvalues'][:5]).round(4)}")

    # Forecast 20 steps from a fresh initial condition.
    x0 = np.array([0.8, 0.6])
    forecast = koopman_forecast(x0, model, T_forecast)   # (20, 2)

    # Ground-truth for the same 20 steps.
    x = x0.copy()
    ground_truth = []
    for _ in range(T_forecast):
        x = step(x)
        ground_truth.append(x.copy())
    ground_truth = np.array(ground_truth)

    mse = float(np.mean((forecast - ground_truth) ** 2))
    print(f"Forecast MSE over {T_forecast} steps: {mse:.6f}")

    # Reconstruction error on the training trajectory.
    train_err = reconstruction_error(traj, model)
    print(f"Training reconstruction MSE: {train_err:.6f}")

    # Sanity: spectral radius of K should be near 1 for a conservative system.
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(model["K"]))))
    print(f"Spectral radius of K: {spectral_radius:.6f}")

    # Quick observable_lift sanity check.
    sample = np.array([1.0, 2.0])
    lifted = observable_lift(sample, degree=2)
    # expect: [1, 2, 1, 4, 2]  (x, x^2, x0*x1)
    print(f"observable_lift([1,2], degree=2) = {lifted}")

    sys.exit(0)
