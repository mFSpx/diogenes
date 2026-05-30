#!/usr/bin/env python3
"""
Real Log Canonical Threshold (RLCT) and Grokking -- Singular Learning Theory.

Watanabe's free energy asymptotic:
    F_n(w) ~ n*L_n(w0) + lambda*log(n) - (m-1)*log(log(n))

lambda is the RLCT: measures geometric degeneracy (singularity) of the loss
landscape at the true parameter w0.  Lower lambda = more singular geometry =
simpler internal structure = better generalization.

Grokking is a phase transition: the network crosses a critical dataset size
n_crit where the singular geometry collapses from a complex memorization
manifold (high lambda) to a simple generalization point (low lambda).  The
free energy of the generalizing solution eventually beats the memorizing
solution once n is large enough for lambda*log(n) to dominate.

The RLCT of any neural network is strictly less than its nominal parameter
count due to symmetries (permutation of hidden units, scaling symmetry in
weight space).  For ReLU networks, lambda is related to the number of distinct
linear regions (activation patterns) the network can express on the training
distribution.

Algebraic geometry note:
  For networks with polynomial activations the loss L(w) is a polynomial in w.
  The RLCT is a birational invariant of the zero set {w : L(w) = 0}.
  Watanabe proved generalisation error = lambda/n asymptotically, replacing
  the classical 1/n rate by a geometry-adjusted term.  The zeta function
  zeta(z) = integral |L(w)|^z phi(w) dw has a pole at z = -lambda, which
  is the RLCT.  For singular models (all deep nets) lambda < d/2 strictly.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "bayesian_information_criterion",
    "waic_estimate",
    "estimate_rlct_from_losses",
    "free_energy_asymptotic",
    "grokking_threshold",
    "activation_pattern_count",
]


def bayesian_information_criterion(log_likelihood, n_params, n_samples):
    """Standard BIC.

    BIC = -2 * log_likelihood + n_params * log(n_samples)

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood evaluated at the MLE.
    n_params : int or float
        Number of free parameters.
    n_samples : int or float
        Dataset size n.

    Returns
    -------
    float
        BIC score.  Lower is better.

    Notes
    -----
    BIC penalises complexity by n_params * log(n) which matches the leading
    singular term in the free energy only when lambda = n_params / 2 (regular
    models).  For singular models (deep nets) BIC over-penalises -- WAIC and
    the RLCT-corrected free energy are the right tools.
    """
    return -2.0 * float(log_likelihood) + float(n_params) * np.log(float(n_samples))


def waic_estimate(log_likelihoods_per_sample):
    """Widely Applicable Information Criterion (Watanabe 2010).

    WAIC = -2 * ( sum_i log mean_j exp(ll[i,j])
                - sum_i var_j(ll[i,j]) )

    Parameters
    ----------
    log_likelihoods_per_sample : array-like, shape (n_samples, n_posterior_draws)
        Log p(x_i | w^(j)) for each data point i and posterior draw j.

    Returns
    -------
    float
        WAIC score.  Lower is better.

    Notes
    -----
    WAIC is valid for singular statistical models (no regular-model assumption).
    It equals the free energy asymptotically for models where the true
    distribution lies on the singularity of the parameter space.  The variance
    term is the functional variance, which estimates the effective complexity
    actually used by the posterior (not the nominal parameter count).
    """
    ll = np.asarray(log_likelihoods_per_sample, dtype=np.float64)  # (n, J)
    # log mean_j exp(ll[i,j])  -- stable via log-sum-exp
    max_ll = ll.max(axis=1, keepdims=True)
    lppd = np.log(np.exp(ll - max_ll).mean(axis=1)) + max_ll.squeeze(axis=1)
    # var_j(ll[i,j]) for each i
    p_waic = ll.var(axis=1)
    return float(-2.0 * (lppd.sum() - p_waic.sum()))


def estimate_rlct_from_losses(train_losses_per_n, n_values):
    """Estimate the RLCT lambda from a training-loss curve over dataset sizes.

    Fits the regression:
        log(loss) ~ lambda * log(log(n)) + const

    via ordinary least squares.

    Parameters
    ----------
    train_losses_per_n : array-like, shape (k,)
        Training loss values L(n) at each dataset size in n_values.  Values
        should be total free energies F_n (not per-sample) so they grow with n.
    n_values : array-like, shape (k,)
        Dataset sizes n (must all be > e ~ 2.718 for log(log(n)) > 0).

    Returns
    -------
    float
        Estimated RLCT (slope of log(F_n) vs log(log(n))).

    Notes
    -----
    From F_n ~ n*L0 + lambda*log(n), taking logs of both sides and noting
    that for large n the dominant variation is log(lambda*log(n)) ~ log(log(n)),
    the slope in log-log(log(n)) space approximates lambda up to a constant.
    Proper RLCT estimation requires thermodynamic integration (MCMC over
    inverse temperature); this is a lightweight proxy.
    """
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)
    if np.any(ns <= np.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    if len(losses) != len(ns):
        raise ValueError("train_losses_per_n and n_values must have the same length")
    y = np.log(np.maximum(losses, 1e-300))
    x = np.log(np.log(ns))
    # OLS: lambda_hat = cov(x, y) / var(x)
    x_c = x - x.mean()
    y_c = y - y.mean()
    var_x = (x_c ** 2).sum()
    if var_x < 1e-15:
        raise ValueError("n_values have no variance in log(log(n)) space")
    return float((x_c * y_c).sum() / var_x)


def free_energy_asymptotic(n, L0, lambda_rlct, m=1):
    """Asymptotic Bayesian free energy from Singular Learning Theory.

    F_n ~ n * L0 + lambda * log(n) - (m - 1) * log(log(n))

    Parameters
    ----------
    n : float or int
        Dataset size.
    L0 : float
        True risk (expected loss at the true parameter w0).
    lambda_rlct : float
        RLCT of the model at w0.
    m : int
        Multiplicity of the RLCT pole (usually 1 for generic singularities).

    Returns
    -------
    float
        Asymptotic free energy estimate.

    Notes
    -----
    The m > 1 correction appears when the RLCT is achieved by multiple poles
    of the zeta function simultaneously (a codimension > 1 event in model
    space).  In practice m = 1 for almost all architectures.
    """
    n = float(n)
    if n <= 1.0:
        raise ValueError("n must be > 1")
    fe = n * float(L0) + float(lambda_rlct) * np.log(n)
    if int(m) > 1:
        if n <= np.e:
            raise ValueError("n must be > e when m > 1 (log(log(n)) undefined)")
        fe -= (int(m) - 1) * np.log(np.log(n))
    return float(fe)


def grokking_threshold(lambda_memorize, lambda_generalize, n_train):
    """Find the critical dataset size where generalisation beats memorisation.

    Solves numerically for the smallest n where:
        n * L_gen + lambda_gen * log(n) < n * L_mem + lambda_mem * log(n)

    Physical setup for the crossing to exist (memo wins early, gen wins late):

      Memorisation (local optimum found first by SGD):
        L_mem > 0  -- overfits to noise, elevated expected loss on unseen data.
        lambda_mem < lambda_gen  -- geometrically simple manifold (low RLCT).
        F_mem = n * L_mem + lambda_mem * log(n)

      Generalisation (global optimum, harder to reach):
        L_gen = 0  -- learns true structure, zero expected loss asymptotically.
        lambda_gen > lambda_mem  -- richer geometry needed to express the true
                                    pattern; more parameter directions are used.
        F_gen = lambda_gen * log(n)

      At small n: n * L_mem dominates F_mem and is small; lambda_gen * log(n)
      is also small but already exceeds n * L_mem + lambda_mem * log(n) for
      n small enough.  The exact crossing satisfies:
        lambda_gen * log(n_crit) = n_crit * L_mem + lambda_mem * log(n_crit)
        (lambda_gen - lambda_mem) * log(n_crit) = n_crit * L_mem
        n_crit / log(n_crit) = (lambda_gen - lambda_mem) / L_mem

      At small n the left side n/log(n) is small (minimum 2/log(2) ~ 2.89 at
      n=2), so memo wins when (lambda_gen - lambda_mem) / L_mem > 2.89.
      As n grows n/log(n) grows unboundedly, so gen eventually wins.

    Parameters
    ----------
    lambda_memorize : float
        RLCT of the memorisation solution.  Must be LESS than lambda_generalize
        for the correct crossing to occur (memo is geometrically simpler).
    lambda_generalize : float
        RLCT of the generalisation solution.  Must be GREATER than
        lambda_memorize (the true global minimum uses richer geometry).
    n_train : int
        Maximum n to search over.

    Returns
    -------
    int
        Smallest n >= 2 where the generalisation free energy is lower,
        or n_train if no crossing found within range.

    Notes
    -----
    Internal constants: L_mem = 0.05 (memo has elevated true risk from
    overfitting), L_gen = 0.0 (gen achieves zero true risk).
    Crossing requires lambda_gen > lambda_mem AND L_mem > L_gen = 0.
    For the default scenario in main (lambda_mem=1.2, lambda_gen=5.0,
    L_mem=0.05): RHS = (5.0-1.2)/0.05 = 76, so n_crit ~ 500.
    """
    lambda_mem = float(lambda_memorize)
    lambda_gen = float(lambda_generalize)
    if lambda_gen <= lambda_mem:
        raise ValueError(
            "lambda_generalize must be > lambda_memorize for a grokking crossing to exist. "
            "The generalising global minimum has higher RLCT (richer geometry) but lower true risk."
        )
    # Memo has elevated true risk (overfits to noise).
    # Gen achieves the true risk of zero (learns the data-generating structure).
    L_mem = 0.05
    L_gen = 0.0

    def fe_diff(n):
        # F_gen - F_mem  (negative => gen wins)
        f_gen = n * L_gen + lambda_gen * np.log(n)
        f_mem = n * L_mem + lambda_mem * np.log(n)
        return f_gen - f_mem

    # At small n fe_diff > 0 (memo wins).  Scan for the first crossing.
    for n in range(2, int(n_train) + 1):
        if fe_diff(n) < 0.0:
            return n
    return int(n_train)


def activation_pattern_count(W_layers, x_samples):
    """Count distinct ReLU activation patterns across samples.

    Each sample x produces a binary signature: for each neuron, 1 if the
    pre-activation is positive, 0 otherwise.  Distinct signatures correspond
    to distinct linear regions of the network.  The number of reachable
    linear regions is a proxy for effective model complexity and is related
    (via the RLCT of piecewise-linear models) to the true parameter cost.

    Parameters
    ----------
    W_layers : list of np.ndarray
        Weight matrices, each shape (d_out, d_in).  Biases are omitted; the
        pattern geometry is dominated by the weight directions.
    x_samples : np.ndarray, shape (n_samples, d_in)
        Input samples.

    Returns
    -------
    int
        Number of distinct activation patterns observed.

    Notes
    -----
    For a single-layer ReLU net the number of distinct patterns is bounded
    by O(n^d) (Cover's theorem) but the RLCT reflects the actual number
    reachable from the training distribution, not the worst case.
    Networks that generalise tend to collapse to far fewer patterns -- the
    RLCT shrinks correspondingly.  High-variance (memorising) weights produce
    many more distinct patterns than low-variance (generalising) weights.
    """
    x = np.asarray(x_samples, dtype=np.float64)
    patterns = set()
    for xi in x:
        h = xi.copy()
        signature = []
        for W in W_layers:
            W = np.asarray(W, dtype=np.float64)
            pre = W @ h
            activation = (pre > 0).astype(np.int8)
            signature.extend(activation.tolist())
            h = pre * activation  # ReLU: zero out negatives
        patterns.add(tuple(signature))
    return len(patterns)


# ---------------------------------------------------------------------------
# Main: grokking simulation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    print("=" * 64)
    print("RLCT / Grokking -- Singular Learning Theory demo")
    print("=" * 64)

    # --- BIC sanity check ---------------------------------------------------
    bic = bayesian_information_criterion(log_likelihood=-120.0, n_params=50, n_samples=500)
    print(f"\nBIC (ll=-120, params=50, n=500): {bic:.4f}")

    # --- WAIC on a toy posterior ---------------------------------------------
    # Simulate 200 data points, 30 posterior draws
    ll_samples = rng.normal(loc=-1.5, scale=0.3, size=(200, 30))
    waic = waic_estimate(ll_samples)
    print(f"WAIC (synthetic posterior): {waic:.4f}")

    # --- RLCT estimation from free-energy curve ----------------------------
    # Supply *total* free energies F_n (growing with n) to the estimator.
    # Grokking scenario:
    #   Memorisation: low RLCT (lambda_mem=1.2) but elevated true risk (L_mem=0.05).
    #   Generalisation: high RLCT (lambda_gen=5.0) but zero true risk (L_gen=0).
    # SLT interpretation: the true global minimum (gen) has richer geometry
    # (more parameter symmetries active), so higher lambda.  The local memo
    # minimum is simpler (lower lambda) but fails to reach true risk zero.
    n_vals = np.array([30, 60, 120, 250, 500, 1000, 2000, 4000], dtype=float)
    lambda_mem_val = 1.2   # RLCT of memorising local minimum (low)
    lambda_gen_val = 5.0   # RLCT of generalising global minimum (high)
    L0_mem = 0.05          # True risk of memo solution (overfits to noise)
    L0_gen = 0.0           # True risk of gen solution (learns true structure)

    fe_curve_mem = np.array([
        free_energy_asymptotic(n, L0_mem, lambda_mem_val) + rng.normal(0, 0.5)
        for n in n_vals
    ])
    fe_curve_gen = np.array([
        free_energy_asymptotic(n, L0_gen, lambda_gen_val) + rng.normal(0, 0.5)
        for n in n_vals
    ])

    est_lambda_mem = estimate_rlct_from_losses(np.maximum(fe_curve_mem, 1e-6), n_vals)
    est_lambda_gen = estimate_rlct_from_losses(np.maximum(fe_curve_gen, 1e-6), n_vals)
    print(f"\nRLCT estimation (from total free-energy curves):")
    print(f"  Memorisation  -- true lambda={lambda_mem_val:.1f}  estimated={est_lambda_mem:.3f}")
    print(f"  Generalisation -- true lambda={lambda_gen_val:.1f}  estimated={est_lambda_gen:.3f}")
    print(f"  (gen lambda > mem lambda: gen solution uses richer geometry)")

    # --- Free energy at two dataset sizes: before and after n_crit ----------
    for n_demo, label in [(20, "small n (memo wins)"), (1000, "large n (gen wins)")]:
        fm = free_energy_asymptotic(n_demo, L0_mem, lambda_mem_val)
        fg = free_energy_asymptotic(n_demo, L0_gen, lambda_gen_val)
        winner = "generalisation" if fg < fm else "memorisation"
        print(f"\nFree energy at n={n_demo} ({label}):")
        print(f"  Memorisation  F_n = {fm:.2f}  (lambda={lambda_mem_val}, L0={L0_mem})")
        print(f"  Generalisation F_n = {fg:.2f}  (lambda={lambda_gen_val}, L0={L0_gen})")
        print(f"  Winner: {winner}")

    # --- Grokking threshold -------------------------------------------------
    # Crossing condition: n/log(n) = (lambda_gen - lambda_mem) / L_mem
    # RHS = (5.0 - 1.2) / 0.05 = 76  =>  n_crit ~ 500
    # Internal constants in grokking_threshold: L_mem=0.05, L_gen=0.
    L_mem_inner = 0.05
    L_gen_inner = 0.0
    n_crit = grokking_threshold(
        lambda_memorize=lambda_mem_val,
        lambda_generalize=lambda_gen_val,
        n_train=50000,
    )
    rhs = (lambda_gen_val - lambda_mem_val) / L_mem_inner
    print(f"\nGrokking transition:")
    print(f"  lambda_mem={lambda_mem_val}  lambda_gen={lambda_gen_val}")
    print(f"  L_mem={L_mem_inner} (memo overfits: elevated true risk)")
    print(f"  L_gen={L_gen_inner} (gen learns true structure: zero true risk)")
    print(f"  Crossing: n/log(n) = (lambda_gen - lambda_mem)/L_mem = {rhs:.1f}")
    print(f"  n_crit = {n_crit}  (gen free energy < memo free energy for all n >= n_crit)")

    # Verify the crossing by scanning free energies around n_crit
    checkpoints = sorted({max(2, n_crit - 200), max(2, n_crit - 1), n_crit, n_crit + 1, n_crit + 200})
    for n_check in checkpoints:
        fm = free_energy_asymptotic(n_check, L_mem_inner, lambda_mem_val)
        fg = free_energy_asymptotic(n_check, L_gen_inner, lambda_gen_val)
        flag = "<-- gen wins" if fg < fm else "    mem wins"
        print(f"    n={n_check:6d}  F_mem={fm:12.2f}  F_gen={fg:12.2f}  {flag}")

    # --- Activation pattern count as RLCT proxy ----------------------------
    # Memorising network: full-rank random weights, high entropy, many linear
    # regions => high effective RLCT.
    # Generalising network: rank-1 weight matrices (outer product structure)
    # simulate a collapsed, low-dimensional representation with far fewer
    # reachable activation patterns => low effective RLCT.
    d_in, d_h1, d_h2 = 8, 16, 16

    # High-rank random weights -- many distinct activation patterns.
    W1_mem = rng.normal(0, 1.0, size=(d_h1, d_in))
    W2_mem = rng.normal(0, 1.0, size=(d_h2, d_h1))

    # Low-rank (rank-2) weights -- constrained geometry, few patterns.
    # Simulate via sum of two rank-1 components.
    u1, v1 = rng.normal(0, 1.0, (d_h1, 1)), rng.normal(0, 1.0, (1, d_in))
    u2, v2 = rng.normal(0, 1.0, (d_h1, 1)), rng.normal(0, 1.0, (1, d_in))
    W1_gen = u1 @ v1 + u2 @ v2
    p1, q1 = rng.normal(0, 1.0, (d_h2, 1)), rng.normal(0, 1.0, (1, d_h1))
    p2, q2 = rng.normal(0, 1.0, (d_h2, 1)), rng.normal(0, 1.0, (1, d_h1))
    W2_gen = p1 @ q1 + p2 @ q2

    x_test = rng.normal(0, 1.0, size=(500, d_in))
    pat_mem = activation_pattern_count([W1_mem, W2_mem], x_test)
    pat_gen = activation_pattern_count([W1_gen, W2_gen], x_test)
    print(f"\nActivation pattern count (proxy for RLCT):")
    print(f"  Memorising  (full-rank random weights, rank={d_in}): {pat_mem} distinct patterns")
    print(f"  Generalising (rank-2 collapsed weights):              {pat_gen} distinct patterns")
    print(f"  Ratio mem/gen: {pat_mem / max(pat_gen, 1):.1f}x")
    print(f"  More patterns => higher effective RLCT => worse generalisation per sample")

    print("\n" + "=" * 64)
    print("Algebraic geometry note:")
    print("  L(w) is polynomial in w for finite polynomial-activation nets.")
    print("  RLCT = smallest pole of zeta(z) = integral |L(w)|^z phi(w) dw.")
    print("  It is a birational invariant of the zero-set {L(w)=0}.")
    print("  Watanabe: generalisation error = lambda/n (not 1/n classical).")
    print("  Deep nets are always singular: lambda < d/2 strictly.")
    print("=" * 64)
