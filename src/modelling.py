"""Shared model specification, metrics and calibration utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Predictor sets
# --------------------------------------------------------------------------
# Every predictor below is knowable *before* the ball is struck. The source
# column `shot_type` (the outcome) and `goal_mouth_location` (defined only once
# the ball reaches the frame) are excluded throughout to prevent leakage.

NUMERIC = [
    "distance_proxy",
    "angle_proxy",
    "player_x",
    "player_y",
    "lateral_offset",
    "centrality",
    "minute",
    "score_diff_before",
]
BINARY = [
    "inside_box",
    "inside_six_yard",
    "central_corridor",
    "is_home_shot",
    "second_half",
    "late_game",
    "added_time_flag",
]
CATEGORICAL = ["body_part", "situation", "score_state"]

#: Metric-corrected geometry, used only in the "alternative distance and angle
#: definitions" sensitivity analysis (proposal 5.10).
NUMERIC_ALT = [
    "distance_m",
    "angle_m",
    "visible_goal_angle",
    "player_x",
    "player_y",
    "lateral_offset",
    "centrality",
    "minute",
    "score_diff_before",
]

TARGET = "goal"


def feature_frame(df: pd.DataFrame, numeric=None, binary=None, categorical=None):
    numeric = NUMERIC if numeric is None else numeric
    binary = BINARY if binary is None else binary
    categorical = CATEGORICAL if categorical is None else categorical
    cols = list(numeric) + list(binary) + list(categorical)
    return df[cols].copy(), numeric, binary, categorical


def make_preprocessor(numeric, binary, categorical, scale=True):
    num_tf = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("num", num_tf, list(numeric)),
            ("bin", "passthrough", list(binary)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False),
                list(categorical),
            ),
        ],
        remainder="drop",
    )


def logistic_pipeline(numeric, binary, categorical, C=1.0):
    from sklearn.linear_model import LogisticRegression

    return Pipeline(
        [
            ("pre", make_preprocessor(numeric, binary, categorical, scale=True)),
            ("clf", LogisticRegression(C=C, max_iter=5000, solver="lbfgs")),
        ]
    )


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Equal-frequency ECE: mean |observed - predicted| weighted by bin size."""
    order = np.argsort(p)
    y_s, p_s = y[order], p[order]
    bins = np.array_split(np.arange(len(p)), n_bins)
    total = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        total += len(b) * abs(y_s[b].mean() - p_s[b].mean())
    return total / len(p)


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Cox calibration: slope from y ~ logit(p); intercept from y ~ offset(logit(p)).

    A perfectly calibrated model has slope 1 and intercept 0. Slope < 1 signals
    over-extreme predictions; a negative intercept signals systematic
    over-prediction of the event.
    """
    lp = _logit(p)
    slope_fit = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    slope = float(slope_fit.params[1])
    int_fit = sm.GLM(
        y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp
    ).fit()
    intercept = float(int_fit.params[0])
    return slope, intercept


def evaluate(y, p, base_rate: float, label: str = "") -> dict:
    """Discrimination, overall accuracy and calibration of a probability model."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    brier = brier_score_loss(y, p)
    brier_ref = brier_score_loss(y, np.full_like(p, base_rate))
    # A constant-probability model is undefined for AUC; guard for Model 0.
    if len(np.unique(p)) > 1:
        auc = roc_auc_score(y, p)
        pr_auc = average_precision_score(y, p)
        slope, intercept = calibration_slope_intercept(y, p)
        ece = expected_calibration_error(y, p)
    else:
        auc, pr_auc, slope, intercept = np.nan, np.nan, np.nan, np.nan
        ece = abs(y.mean() - p.mean())
    return {
        "Model": label,
        "N shots": len(y),
        "AUC": auc,
        "PR-AUC": pr_auc,
        "Log loss": log_loss(y, p),
        "Brier score": brier,
        "Brier skill score": 1 - brier / brier_ref,
        "Calibration slope": slope,
        "Calibration intercept": intercept,
        "ECE": ece,
    }


def bootstrap_ci(y, p, metric_fn, n_boot: int = 500, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI for a metric of (y, p)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            stats[i] = np.nan
            continue
        stats[i] = metric_fn(y[idx], p[idx])
    stats = stats[~np.isnan(stats)]
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))
