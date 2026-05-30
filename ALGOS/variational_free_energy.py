#!/usr/bin/env python3
"""
Variational Free Energy — Active Inference / Free Energy Principle.

F = E_{q(s)}[ln q(s) - ln p(o,s)]
  = KL[q(s) || p(s|o)] - ln p(o)

F is always an upper bound on surprise (-ln p(o)).

Minimizing F tightens two screws simultaneously:
  - UPDATE BELIEFS (perception): move q(s) toward the true posterior p(s|o).
    The gap KL[q || p(s|o)] shrinks. The agent becomes less wrong about the world.
  - ALTER THE WORLD (action): change o until p(o|s) matches q's predictions.
    Surprise -ln p(o) shrinks because the world is steered into familiar territory.

This is why perception and action are the same computation in biological brains —
both are gradient descent on the same scalar F.

Thermodynamic note: F is literally Helmholtz free energy from statistical physics.
  F = U - TS   (energy minus entropy, a.k.a. "useful energy")
The brain runs as a heat engine on prediction error.  Surprise = metabolic cost.
A well-calibrated model is an efficient engine.  A confused model burns hot.

All distributions are Gaussian throughout (tractable closed forms, no quadrature).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "kl_gaussian",
    "free_energy_gaussian",
    "belief_update",
    "active_inference_step",
    "precision_weighted_update",
]


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------


def kl_gaussian(
    mu_q: float | np.ndarray,
    sigma_q: float | np.ndarray,
    mu_p: float | np.ndarray,
    sigma_p: float | np.ndarray,
) -> float:
    """KL divergence KL[N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)].

    Closed form for univariate Gaussians (scalar or array; arrays are summed):

        KL = ln(sigma_p/sigma_q) + (sigma_q^2 + (mu_q - mu_p)^2) / (2 sigma_p^2) - 1/2

    Parameters
    ----------
    mu_q, sigma_q:
        Mean and standard deviation of the variational distribution q.
    mu_p, sigma_p:
        Mean and standard deviation of the prior p.

    Returns
    -------
    float — sum of KL over all dimensions if array inputs.
    """
    mu_q = np.asarray(mu_q, dtype=float)
    sigma_q = np.asarray(sigma_q, dtype=float)
    mu_p = np.asarray(mu_p, dtype=float)
    sigma_p = np.asarray(sigma_p, dtype=float)

    if np.any(sigma_q <= 0) or np.any(sigma_p <= 0):
        raise ValueError("Standard deviations must be strictly positive.")

    kl = (
        np.log(sigma_p / sigma_q)
        + (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2.0 * sigma_p ** 2)
        - 0.5
    )
    return float(np.sum(kl))


def free_energy_gaussian(
    mu_q: float | np.ndarray,
    sigma_q: float | np.ndarray,
    mu_p_s: float | np.ndarray,
    sigma_p_s: float | np.ndarray,
    obs: float | np.ndarray,
    sigma_obs: float | np.ndarray,
) -> float:
    """Variational free energy for a Gaussian generative model.

    F = KL[q(s) || p(s)] + reconstruction_loss

    where reconstruction_loss is the negative log-likelihood of the observation
    under the generative model evaluated at the current belief mean:

        reconstruction_loss = 0.5 * ((obs - mu_q)^2 / sigma_obs^2 + ln(2 pi sigma_obs^2))

    This is an ELBO decomposition: the KL term penalises beliefs that deviate
    from the prior; the reconstruction term penalises beliefs that predict the
    observation poorly.

    Parameters
    ----------
    mu_q, sigma_q:
        Variational posterior mean and standard deviation.
    mu_p_s, sigma_p_s:
        Prior mean and standard deviation over hidden states s.
    obs:
        Observed scalar or vector.
    sigma_obs:
        Observation noise standard deviation.

    Returns
    -------
    float — variational free energy F.
    """
    obs = np.asarray(obs, dtype=float)
    mu_q = np.asarray(mu_q, dtype=float)
    sigma_obs = np.asarray(sigma_obs, dtype=float)

    if np.any(sigma_obs <= 0):
        raise ValueError("sigma_obs must be strictly positive.")

    kl = kl_gaussian(mu_q, sigma_q, mu_p_s, sigma_p_s)

    # Negative log-likelihood of obs under N(mu_q, sigma_obs^2)
    # This is the "prediction error" term — does the belief predict the observation?
    recon = float(
        np.sum(
            0.5 * ((obs - mu_q) ** 2 / sigma_obs ** 2)
            + 0.5 * np.log(2.0 * np.pi * sigma_obs ** 2)
        )
    )

    return kl + recon


def belief_update(
    mu_q: float | np.ndarray,
    sigma_q: float | np.ndarray,
    obs: float | np.ndarray,
    A: float | np.ndarray,
    sigma_obs: float | np.ndarray,
    eta: float = 0.1,
) -> float | np.ndarray:
    """Gradient descent step on F w.r.t. mu_q (perception update).

    The gradient of the reconstruction term w.r.t. mu_q is:

        dF/d(mu_q) = -(obs - A @ mu_q) / sigma_obs^2 * A^T

    This is the classic prediction-error signal weighted by precision (1/sigma_obs^2).
    Positive prediction error (obs > A @ mu_q) pulls the belief upward.

    Parameters
    ----------
    mu_q:
        Current belief mean, shape (d,) or scalar.
    sigma_q:
        Current belief standard deviation (not updated here — full covariance
        update requires a second-order step, which is left to the caller).
    obs:
        Observation, shape (k,) or scalar.
    A:
        Observation matrix mapping states to observations, shape (k, d) or scalar.
        If scalar, treated as 1x1 matrix.
    sigma_obs:
        Observation noise std, shape (k,) or scalar.
    eta:
        Gradient descent step size (learning rate).

    Returns
    -------
    mu_q_new — updated belief mean, same shape as mu_q.
    """
    mu_q = np.atleast_1d(np.asarray(mu_q, dtype=float))
    obs = np.atleast_1d(np.asarray(obs, dtype=float))
    A = np.atleast_2d(np.asarray(A, dtype=float))
    sigma_obs = np.atleast_1d(np.asarray(sigma_obs, dtype=float))

    if np.any(sigma_obs <= 0):
        raise ValueError("sigma_obs must be strictly positive.")

    # Prediction error: shape (k,)
    pred_error = obs - A @ mu_q
    precision = 1.0 / (sigma_obs ** 2)

    # Gradient of reconstruction loss w.r.t. mu_q: shape (d,)
    # dF/d(mu_q) = -A^T @ diag(precision) @ pred_error
    grad = -A.T @ (precision * pred_error)

    mu_q_new = mu_q - eta * grad
    return mu_q_new.squeeze() if mu_q_new.shape == (1,) else mu_q_new


def active_inference_step(
    mu_q: float | np.ndarray,
    sigma_q: float | np.ndarray,
    obs: float | np.ndarray,
    A: float | np.ndarray,
    sigma_obs: float | np.ndarray,
    world_fn,
    eta_belief: float = 0.1,
    eta_action: float = 0.05,
) -> tuple:
    """One full active inference step: update beliefs AND compute action gradient.

    Perception half: gradient descent on F w.r.t. mu_q (see belief_update).
    Action half: finite-difference gradient of F w.r.t. action a, then a -= eta_action * dF/da.

    The key insight is that the agent tries to reduce F by both moving its model
    of the world (belief update) AND by moving the world itself (action).
    Both reduce the same objective — this is the duality at the heart of active inference.

    Parameters
    ----------
    mu_q, sigma_q:
        Current belief mean and standard deviation.
    obs:
        Current observation.
    A:
        Observation matrix.
    sigma_obs:
        Observation noise standard deviation.
    world_fn:
        Callable (action: float) -> new_obs. Encapsulates world dynamics.
        Finite differences probe this function to estimate dF/da.
    eta_belief:
        Belief update step size.
    eta_action:
        Action gradient step size.

    Returns
    -------
    (mu_q_new, action, F_value)
      mu_q_new — updated belief mean after perception step.
      action    — gradient-descent action step taken.
      F_value   — free energy at the current belief before updating.
    """
    obs = np.asarray(obs, dtype=float)
    mu_q = np.asarray(mu_q, dtype=float)

    # Current free energy (prior mean = mu_q itself as a flat prior approximation)
    F_current = free_energy_gaussian(
        mu_q, sigma_q, mu_q, sigma_q * 10.0, obs, sigma_obs
    )

    # --- Perception update ---
    mu_q_new = belief_update(mu_q, sigma_q, obs, A, sigma_obs, eta=eta_belief)

    # --- Action gradient via finite difference ---
    eps_fd = 1e-3
    # Probe F under a small positive action
    obs_plus = np.asarray(world_fn(eps_fd), dtype=float)
    F_plus = free_energy_gaussian(
        mu_q_new, sigma_q, mu_q, sigma_q * 10.0, obs_plus, sigma_obs
    )
    obs_minus = np.asarray(world_fn(-eps_fd), dtype=float)
    F_minus = free_energy_gaussian(
        mu_q_new, sigma_q, mu_q, sigma_q * 10.0, obs_minus, sigma_obs
    )

    dF_da = (F_plus - F_minus) / (2.0 * eps_fd)
    action = -eta_action * dF_da  # gradient descent: move against the gradient

    return mu_q_new, float(action), float(F_current)


def precision_weighted_update(
    predictions: list,
    obs_list: list,
    precisions: list,
) -> list:
    """Hierarchical precision-weighted prediction error minimization.

    In hierarchical predictive coding each level sends predictions downward and
    receives precision-weighted prediction errors upward.  Higher precision means
    the level trusts its own predictions less and defers more to incoming data.

    The update at each level is:

        error_i = precision_i * (obs_i - prediction_i)

    This is the signal that gets backpropagated up the hierarchy to update
    higher-level representations.  When precision is high, small discrepancies
    generate large update signals; low precision attenuates noisy channels.

    Parameters
    ----------
    predictions:
        List of predicted observations at each hierarchical level.
    obs_list:
        List of actual observations at each level (same length as predictions).
    precisions:
        List of inverse variances (1/sigma^2) at each level.

    Returns
    -------
    List of precision-weighted prediction errors, one per level.
    """
    if not (len(predictions) == len(obs_list) == len(precisions)):
        raise ValueError("predictions, obs_list, and precisions must have equal length.")

    errors = []
    for pred, obs, prec in zip(predictions, obs_list, precisions):
        pred = np.asarray(pred, dtype=float)
        obs = np.asarray(obs, dtype=float)
        prec = float(prec)
        if prec < 0:
            raise ValueError("Precision (inverse variance) must be non-negative.")
        errors.append(prec * (obs - pred))
    return errors


# ---------------------------------------------------------------------------
# Demo: 1D agent tracking a drifting state
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Setup -------------------------------------------------------------------
    # True world: hidden state s drifts (random walk, scale 0.3 per step).
    # Agent observes o = s + N(0, sigma_obs^2).
    # Agent belief: q(s) = N(mu_q, sigma_q^2).
    # Prior: p(s) = N(0, sigma_prior^2) — broad prior.
    #
    # Phase 1 (steps 0..14): belief update only. Agent tracks reality.
    #   F falls as the belief closes the gap with the true posterior.
    # Phase 2 (steps 15..29): agent also acts.
    #   Action nudges true_state toward the current belief (efference copy).
    #   world_fn is deterministic for the finite-difference probe — noise only
    #   enters the real observation, not the gradient estimate.
    #   Action is clipped to [-0.5, 0.5] per step to keep the demo legible.
    # -------------------------------------------------------------------------

    rng = np.random.default_rng(seed=7)

    sigma_obs = 1.0
    sigma_prior = 5.0
    sigma_q = 1.0          # belief uncertainty (held fixed in this demo)
    drift_scale = 0.3      # random-walk scale per step
    A = np.array([[1.0]])  # identity: obs = s + noise
    ACTION_CLIP = 0.5      # max action magnitude per step

    true_state = 3.0       # start displaced from prior
    mu_q = 0.0             # agent starts at prior mean

    F_THRESHOLD = 2.5      # act only when F stays above this after belief update

    print("Variational Free Energy — active inference demo")
    print(f"  sigma_obs={sigma_obs}  sigma_prior={sigma_prior}  drift={drift_scale}")
    print(f"  True state starts at {true_state:.2f}, belief at {mu_q:.2f}")
    print(f"  Steps 0-14: perception only.  Steps 15-29: perception + action.")
    print()
    print(f"{'Step':>4}  {'True s':>7}  {'Belief':>7}  {'|Error|':>7}  {'F':>7}  {'Action':>7}")
    print("-" * 52)

    for step in range(30):
        # World generates a noisy observation
        obs = true_state + rng.normal(0, sigma_obs)

        # Free energy before any update (what the agent starts the step with)
        F_val = free_energy_gaussian(
            mu_q, sigma_q, 0.0, sigma_prior, obs, sigma_obs
        )

        # --- Perception: gradient descent on F w.r.t. mu_q ---
        mu_q_arr = belief_update(
            np.array([mu_q]), sigma_q, np.array([obs]),
            A, sigma_obs, eta=0.3,
        )
        mu_q = float(mu_q_arr)

        # --- Action (phase 2 only): nudge the world if F is still high ---
        action_taken = 0.0
        if step >= 15:
            F_after = free_energy_gaussian(
                mu_q, sigma_q, 0.0, sigma_prior, obs, sigma_obs
            )
            if F_after > F_THRESHOLD:
                # Deterministic world_fn for finite-difference probe:
                # hypothetically, action a shifts true_state by a, observed cleanly.
                # No RNG inside — gradient estimates must not be noisy.
                def world_fn(a: float, _s: float = true_state) -> float:
                    return _s + a   # noiseless prediction for the gradient probe

                _, raw_action, _ = active_inference_step(
                    mu_q, sigma_q, obs, A, sigma_obs, world_fn,
                    eta_belief=0.0,   # belief already updated above
                    eta_action=0.4,
                )
                # Clip to keep the demo bounded
                action_taken = float(np.clip(raw_action, -ACTION_CLIP, ACTION_CLIP))
                true_state += action_taken

        print(
            f"{step:>4}  {true_state:>7.3f}  {mu_q:>7.3f}  "
            f"{abs(true_state - mu_q):>7.3f}  {F_val:>7.3f}  {action_taken:>7.4f}"
        )

        # World drifts after agent acts
        true_state += drift_scale * rng.normal()

    print()
    print("Precision-weighted hierarchy demo (3 levels):")
    preds = [2.0, 5.0, 10.0]
    obs_h = [2.3, 4.5, 11.0]
    precs = [4.0, 1.0, 0.25]   # high precision at bottom (sensory), low at top (abstract)
    errors = precision_weighted_update(preds, obs_h, precs)
    for lvl, (p, o, pr, e) in enumerate(zip(preds, obs_h, precs, errors)):
        print(f"  Level {lvl}: pred={p:.1f}  obs={o:.1f}  precision={pr:.2f}  "
              f"weighted_error={float(e):.4f}")
