#!/usr/bin/env python3
"""
Caputo Fractional Derivative — power-law memory kernel for SSMs and decay models.

Standard exponential decay forgets with rate e^{-lambda t}: memory is Markovian
and geometric. The Caputo fractional derivative replaces exponential memory with
a power-law kernel, giving infinite non-local history with algebraic (slow) decay.

Definition (Caputo, 1967):

    D^alpha f(t) = 1/Gamma(1 - alpha) * integral_0^t f'(tau) / (t - tau)^alpha dtau

where 0 < alpha < 1. As alpha -> 1 the operator approaches the classical first
derivative; as alpha -> 0 it approaches the identity.  The kernel (t - tau)^{-alpha}
is non-integrable at tau = t (weakly singular) so numerical schemes must handle the
singularity — the trapezoidal rule with a left-endpoint correction is used here.

Power-law decay kernel (the fractional analogue of e^{-lambda t}):

    phi(t; alpha) = t^{alpha - 1} / Gamma(alpha)

This appears as the impulse response of a fractional integrator and as the weight
each past gradient receives in the Caputo integral.

Fractional SSM step replaces the Markovian recurrence

    h_t = A h_{t-1} + B x_t

with the Caputo-weighted sum over the full history:

    h_t = sum_{k=0}^{T-1} w_k * (A h_k + B x_t)

where w_k = phi(t - tau_k; alpha) / sum_j phi(t - tau_j; alpha) is the
normalized fractional kernel evaluated at each past time.  This gives the SSM
algebraically-decaying long-range memory without a hidden state blowup.

Lanczos Gamma approximation (Lanczos 1964, g=7 coefficients from Numerical Recipes):

    Gamma(z+1) = sqrt(2*pi) * (z + g + 0.5)^{z + 0.5} * e^{-(z + g + 0.5)} * A_g(z)

    A_g(z) = c_0 + c_1/(z+1) + c_2/(z+2) + ... + c_g/(z+g)
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "gamma_lanczos",
    "caputo_derivative",
    "fractional_decay",
    "fractional_ssm_step",
]

# Lanczos g=7 coefficients (Numerical Recipes 3rd ed., table 6.1)
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])


def gamma_lanczos(z):
    """Lanczos approximation of Gamma(z) for z > 0.

    Uses g=7 Lanczos coefficients from Numerical Recipes.  Accurate to ~15
    significant figures for real z > 0.  For z < 0.5 uses the reflection
    formula Gamma(z)*Gamma(1-z) = pi/sin(pi*z) to stay in the stable region.

    Parameters
    ----------
    z:
        Positive real argument.

    Returns
    -------
    float
        Gamma(z).
    """
    z = float(z)
    if z < 0.5:
        # Reflection formula
        return np.pi / (np.sin(np.pi * z) * gamma_lanczos(1.0 - z))
    z -= 1.0
    x = _LANCZOS_C[0]
    for i in range(1, _LANCZOS_G + 2):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return float(np.sqrt(2.0 * np.pi) * (t ** (z + 0.5)) * np.exp(-t) * x)


def caputo_derivative(f_vals, t_vals, alpha):
    """Numerical Caputo derivative via trapezoidal rule.

    Approximates D^alpha f at every point in t_vals:

        D^alpha f(t_k) = 1/Gamma(1-alpha) * sum_{j=0}^{k-1}
            integral_{t_j}^{t_{j+1}} f'(tau) / (t_k - tau)^alpha dtau

    The integrand is approximated by the midpoint-slope of the (f, t) segment
    divided by the kernel at the segment midpoint.  At k=0 the derivative is
    zero (no history).

    Parameters
    ----------
    f_vals:
        1-D numpy array of function values f(t_j).
    t_vals:
        1-D numpy array of time points (same length, monotone increasing).
    alpha:
        Fractional order, 0 < alpha < 1.

    Returns
    -------
    np.ndarray
        Same shape as f_vals.  Element k is D^alpha f(t_k).
    """
    f_vals = np.asarray(f_vals, dtype=float)
    t_vals = np.asarray(t_vals, dtype=float)
    n = len(f_vals)
    prefactor = 1.0 / gamma_lanczos(1.0 - alpha)
    result = np.zeros(n)
    for k in range(1, n):
        tk = t_vals[k]
        acc = 0.0
        for j in range(k):
            dt = t_vals[j + 1] - t_vals[j]
            f_prime = (f_vals[j + 1] - f_vals[j]) / dt        # slope on segment
            t_mid = 0.5 * (t_vals[j] + t_vals[j + 1])
            kernel = (tk - t_mid) ** (-alpha)
            acc += f_prime * kernel * dt
        result[k] = prefactor * acc
    return result


def fractional_decay(t_array, alpha, scale=1.0):
    """Power-law decay: t^(alpha-1) / Gamma(alpha).

    The fractional analogue of exponential decay.  This is the impulse response
    of a Caputo fractional integrator of order alpha.  For small alpha the decay
    is very slow (long memory); as alpha -> 1 the kernel sharpens toward a Dirac
    at t=0.

    Parameters
    ----------
    t_array:
        1-D numpy array of positive time values.
    alpha:
        Fractional order, 0 < alpha < 1.
    scale:
        Optional amplitude multiplier.

    Returns
    -------
    np.ndarray
        Decay values, same shape as t_array.
    """
    t_array = np.asarray(t_array, dtype=float)
    g_alpha = gamma_lanczos(alpha)
    return scale * (t_array ** (alpha - 1.0)) / g_alpha


def fractional_ssm_step(h_history, x, alpha, A, B, C, dt=0.1):
    """SSM step with Caputo memory kernel.

    Replaces the Markovian recurrence h_t = A h_{t-1} + B x with a fractional
    convolution over the full history:

        h_new = sum_{k=0}^{T-1} w_k * (A @ h_k + B @ x)

    where w_k = phi(T*dt - k*dt; alpha) are the fractional decay weights and
    phi is the power-law kernel t^{alpha-1}/Gamma(alpha).  Weights are L1-
    normalised so the contribution magnitudes stay comparable across history
    lengths.

    Output:
        y_new = C @ h_new

    Parameters
    ----------
    h_history:
        Array of shape (T_past, state_dim) — all past hidden states.
    x:
        Input vector of shape (input_dim,).
    alpha:
        Fractional order, 0 < alpha < 1.  Lower = longer memory.
    A:
        State transition matrix, shape (state_dim, state_dim).
    B:
        Input matrix, shape (state_dim, input_dim).
    C:
        Output matrix, shape (output_dim, state_dim).
    dt:
        Time step size (uniform grid assumed).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (h_new, y_new) where h_new has shape (state_dim,) and y_new has
        shape (output_dim,).
    """
    h_history = np.asarray(h_history, dtype=float)
    x = np.asarray(x, dtype=float)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)

    T_past = h_history.shape[0]
    if T_past == 0:
        h_new = B @ x
        y_new = C @ h_new
        return h_new, y_new

    # Time offsets from the current step back to each past step (oldest first)
    # tau_k = (T_past - k) * dt  for k = 0 ... T_past-1
    offsets = np.arange(T_past, 0, -1, dtype=float) * dt   # shape (T_past,)
    weights = fractional_decay(offsets, alpha)              # power-law kernel
    weights = weights / (weights.sum() + 1e-30)             # L1 normalise

    # Candidate update for each past state: A @ h_k + B @ x
    Bx = B @ x                                              # (state_dim,)
    candidates = (A @ h_history.T).T + Bx                  # (T_past, state_dim)

    # Weighted sum over history
    h_new = (weights[:, np.newaxis] * candidates).sum(axis=0)
    y_new = C @ h_new
    return h_new, y_new


# ---------------------------------------------------------------------------
# main: compare exponential vs fractional decay over 50 steps
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    steps = 50
    alpha = 0.4          # fractional order (lower = longer memory)
    lam = 1.0            # exponential decay rate

    t = np.linspace(0.1, 5.0, steps)   # avoid t=0 (kernel singularity)

    exp_decay = np.exp(-lam * t)
    frac_decay = fractional_decay(t, alpha)
    # normalise both to start at 1 for fair comparison
    frac_decay_norm = frac_decay / frac_decay[0]

    # long-range signal retained: ratio at final vs initial step
    exp_ratio = exp_decay[-1] / exp_decay[0]
    frac_ratio = frac_decay_norm[-1] / frac_decay_norm[0]

    print("=" * 58)
    print("Caputo fractional derivative — decay comparison")
    print(f"  steps={steps}  alpha={alpha}  lambda={lam}")
    print("=" * 58)
    print(f"  {'Step':>5}  {'t':>6}  {'exp_decay':>12}  {'frac_decay':>12}")
    print("-" * 58)
    for i in [0, 9, 19, 29, 39, 49]:
        print(f"  {i:>5}  {t[i]:>6.2f}  {exp_decay[i]:>12.6f}  {frac_decay_norm[i]:>12.6f}")
    print("-" * 58)
    print(f"  Signal retained at step {steps}: exp={exp_ratio:.6f}  frac={frac_ratio:.6f}")
    print(f"  Fractional retains {frac_ratio/exp_ratio:.1f}x more long-range signal")
    print()

    # Quick Caputo derivative smoke test on a known function: f(t) = t
    # D^alpha [t] = t^{1-alpha} / Gamma(2 - alpha)
    t2 = np.linspace(0.0, 2.0, 30)
    f2 = t2.copy()
    numerical = caputo_derivative(f2, t2, alpha)
    analytical = t2 ** (1.0 - alpha) / gamma_lanczos(2.0 - alpha)
    mid = len(t2) // 2
    err = abs(numerical[mid] - analytical[mid]) / (abs(analytical[mid]) + 1e-30)
    print("Caputo D^alpha[t] smoke test  (f=t, alpha={:.2f})".format(alpha))
    print(f"  numerical[{mid}]={numerical[mid]:.6f}  analytical[{mid}]={analytical[mid]:.6f}  rel_err={err:.4f}")

    # SSM step smoke test
    state_dim, input_dim, output_dim = 4, 2, 2
    rng = np.random.default_rng(42)
    A_mat = 0.9 * np.eye(state_dim)
    B_mat = rng.standard_normal((state_dim, input_dim)) * 0.1
    C_mat = rng.standard_normal((output_dim, state_dim)) * 0.1
    T_hist = 10
    h_hist = rng.standard_normal((T_hist, state_dim)) * 0.1
    x_in = rng.standard_normal(input_dim)
    h_new, y_new = fractional_ssm_step(h_hist, x_in, alpha, A_mat, B_mat, C_mat)
    print()
    print("SSM step smoke test")
    print(f"  h_new shape={h_new.shape}  ||h_new||={np.linalg.norm(h_new):.4f}")
    print(f"  y_new={y_new}")
    print("=" * 58)
