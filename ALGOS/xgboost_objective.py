#!/usr/bin/env python3
r"""XGBoost objective mathematics and a small sklearn-compatible wrapper.

Algorithm 1: eXtreme Gradient Boosting (XGBoost)

Ensemble prediction after t rounds:
    \hat y_i^{(t)} = \hat y_i^{(t-1)} + f_t(x_i),    f_t \in \mathcal{F}

Regularized objective:
    Obj^{(t)} = \sum_i l(y_i, \hat y_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)

Second-order Taylor approximation used to select the next tree:
    Obj^{(t)} \approx \sum_i [g_i f_t(x_i) + 1/2 h_i f_t(x_i)^2] + \Omega(f_t)

where
    g_i = \partial_{\hat y^{(t-1)}} l(y_i, \hat y_i^{(t-1)})
    h_i = \partial^2_{\hat y^{(t-1)}} l(y_i, \hat y_i^{(t-1)})

Tree regularization:
    \Omega(f_t) = \gamma T + 1/2 \lambda \sum_{j=1}^T w_j^2

For a leaf j with instance set I_j:
    G_j = \sum_{i\in I_j} g_i,  H_j = \sum_{i\in I_j} h_i
    w_j^* = -G_j / (H_j + \lambda)
    Obj_leaf^* = -1/2 \sum_j G_j^2 / (H_j + \lambda) + \gamma T

Split gain:
    Gain = 1/2 [G_L^2/(H_L+\lambda) + G_R^2/(H_R+\lambda)
                - (G_L+G_R)^2/(H_L+H_R+\lambda)] - \gamma

This file keeps the math explicit while delegating production tree building to
xgboost.XGBClassifier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

XGBOOST_MATH = __doc__ or ""


def sigmoid(margin: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-margin))


def binary_logistic_grad_hess(y_true: np.ndarray, margin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and second gradients for binary logistic loss in margin space."""
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h


def optimal_leaf_weight(gradient_sum: float, hessian_sum: float, reg_lambda: float = 1.0) -> float:
    return -float(gradient_sum) / (float(hessian_sum) + float(reg_lambda))


def split_gain(
    left_gradient: float,
    left_hessian: float,
    right_gradient: float,
    right_hessian: float,
    *,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
) -> float:
    gl, hl = float(left_gradient), float(left_hessian)
    gr, hr = float(right_gradient), float(right_hessian)
    parent = (gl + gr) ** 2 / (hl + hr + reg_lambda)
    children = gl**2 / (hl + reg_lambda) + gr**2 / (hr + reg_lambda)
    return 0.5 * (children - parent) - gamma


@dataclass(frozen=True)
class XGBoostConfig:
    objective: str = "multi:softprob"
    n_estimators: int = 96
    learning_rate: float = 0.08
    max_depth: int = 5
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_lambda: float = 1.0
    min_child_weight: float = 1.0
    random_state: int = 414
    n_jobs: int = -1
    eval_metric: str = "mlogloss"


def build_xgb_classifier(num_class: int, config: XGBoostConfig | None = None) -> Any:
    """Return an XGBClassifier using the objective above.

    Import is local so this ALGOS unit remains readable on machines without the
    optional ML stack installed.
    """
    import xgboost as xgb

    cfg = config or XGBoostConfig()
    objective = cfg.objective if num_class > 2 else "binary:logistic"
    kwargs = {
        "objective": objective,
        "n_estimators": cfg.n_estimators,
        "learning_rate": cfg.learning_rate,
        "max_depth": cfg.max_depth,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "reg_lambda": cfg.reg_lambda,
        "min_child_weight": cfg.min_child_weight,
        "random_state": cfg.random_state,
        "n_jobs": cfg.n_jobs,
        "eval_metric": cfg.eval_metric,
        "tree_method": "hist",
    }
    if num_class > 2:
        kwargs["num_class"] = int(num_class)
    return xgb.XGBClassifier(**kwargs)
