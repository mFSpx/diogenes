#!/usr/bin/env python3
r"""SHAP/Shapley attribution mathematics and tree-model helpers.

Algorithm 2: SHAP (SHapley Additive exPlanations)

For feature set F and model value function f_S(x_S), the Shapley value of
feature j is

    \phi_j = \sum_{S \subseteq F \setminus \{j\}}
             |S|!(|F|-|S|-1)! / |F|!
             [f_{S\cup\{j\}}(x_{S\cup\{j\}}) - f_S(x_S)]

The additive explanation model is

    f(x) = \phi_0 + \sum_{j=1}^{|F|} \phi_j

for the model output space being explained. For XGBoost tree ensembles we use
TreeSHAP: either shap.TreeExplainer when the package is available or XGBoost's
built-in pred_contribs=True contribution tensor. Both return the same additive
contract in margin/raw-output space for tree models.
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Callable, Iterable

import numpy as np

SHAP_MATH = __doc__ or ""


def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    return math.factorial(subset_size) * math.factorial(feature_count - subset_size - 1) / math.factorial(feature_count)


def exact_shapley_value(
    value_fn: Callable[[frozenset[int]], float],
    feature_index: int,
    feature_count: int,
) -> float:
    """Exact generic Shapley value by enumerating every coalition.

    Intended for small didactic/state-vector cases. TreeSHAP is used for real
    XGBoost models because enumeration is O(2^F).
    """
    others = [i for i in range(feature_count) if i != feature_index]
    total = 0.0
    for k in range(len(others) + 1):
        for subset in combinations(others, k):
            s = frozenset(subset)
            total += shapley_kernel_weight(k, feature_count) * (value_fn(s | {feature_index}) - value_fn(s))
    return total


def xgboost_tree_shap_tensor(booster: Any, matrix: Any) -> np.ndarray:
    """Deterministic TreeSHAP tensor from an XGBoost Booster.

    Output shape is XGBoost-version dependent:
      * binary/regression: rows x (features + bias)
      * multiclass: rows x classes x (features + bias)
    """
    import xgboost as xgb

    dmatrix = matrix if isinstance(matrix, xgb.DMatrix) else xgb.DMatrix(matrix)
    return booster.predict(dmatrix, pred_contribs=True)


def mean_abs_shap_by_feature(shap_tensor: np.ndarray, feature_names: list[str]) -> list[dict[str, float | str]]:
    """Aggregate a TreeSHAP contribution tensor into ranked feature importances."""
    values = np.asarray(shap_tensor)
    if values.ndim == 3:
        # rows x classes x (features+bias) -> rows/classes mean absolute per feature
        contrib = np.abs(values[:, :, :-1]).mean(axis=(0, 1))
    elif values.ndim == 2:
        contrib = np.abs(values[:, :-1]).mean(axis=0)
    else:
        raise ValueError(f"unsupported_shap_tensor_shape:{values.shape}")
    n = min(len(feature_names), contrib.shape[0])
    ranked = sorted(
        ({"feature": feature_names[i], "mean_abs_shap": float(contrib[i])} for i in range(n)),
        key=lambda row: row["mean_abs_shap"],
        reverse=True,
    )
    return ranked


def try_shap_tree_explainer(model: Any, matrix: Any) -> Any | None:
    """Optional shap.TreeExplainer path; returns None if unavailable/incompatible."""
    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(model)
        return explainer.shap_values(matrix)
    except Exception:
        return None
