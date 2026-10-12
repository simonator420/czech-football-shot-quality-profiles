"""Step 16 - Subgroup calibration of the holdout (2024/25) shot-quality model.

Reviewer request for Supplementary Table S3: check whether the 2024/25
holdout over-prediction (909.2 predicted xG vs. 718 observed goals;
calibration intercept -0.320) is uniform across home/away shots and across
teams that were strong vs. weak in the preceding (2023/24) season.

Uses the original, unrecalibrated CatBoost holdout predictions
(`shot_quality_holdout` in data/processed_release/shots_scored.parquet) - never the
recalibrated or out-of-fold values. Strong/weak grouping is fixed from the
2023/24 final league table (points-based ranking, same method as
`team_results` in s12_core_validation.py) before any 2024/25 results are
used, so the grouping does not leak holdout information:
    strong = final position 1-8 in 2023/24
    weak   = final position 9-15 in 2023/24, plus the club newly promoted
             into 2024/25 (absent from the 2023/24 table)

Calibration intercept uses the same Cox-style method as
`modelling.calibration_slope_intercept`: a logistic regression of the
observed outcome on an offset of logit(predicted probability) with the
slope fixed at 1 (intercept-only GLM with offset).

Produces
    tables/tableS3_holdout_calibration_subgroups.csv
    tables/tableS3b_holdout_calibration_contrasts.csv
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import RANDOM_STATE, header, load_frame, save_table
from s12_core_validation import team_results

warnings.filterwarnings("ignore")

N_BOOT = 2000


def _logit(p: np.ndarray) -> np.ndarray:
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fixed_slope_intercept(y: np.ndarray, p: np.ndarray) -> float:
    lp = _logit(p)
    fit = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
    return float(fit.params[0])


def intercept_arr(y: np.ndarray, p: np.ndarray) -> float:
    try:
        return fixed_slope_intercept(y, p)
    except Exception:
        return np.nan


def group_metrics(df: pd.DataFrame, label: str) -> dict:
    y = df["goal"].to_numpy(dtype=float)
    p = df["shot_quality_holdout"].to_numpy(dtype=float)
    observed = float(y.sum())
    predicted = float(p.sum())
    return {
        "Group": label,
        "N shots": int(len(df)),
        "Observed goals": observed,
        "Predicted xG": round(predicted, 1),
        "O/E ratio": round(observed / predicted, 3) if predicted else np.nan,
        "Calibration intercept": round(fixed_slope_intercept(y, p), 3),
    }


def oe_ratio_arr(y: np.ndarray, p: np.ndarray) -> float:
    total = p.sum()
    return float(y.sum() / total) if total else np.nan


def by_match_arrays(df: pd.DataFrame) -> dict:
    out = {}
    for m, g in df.groupby("match_id"):
        out[m] = (
            g["goal"].to_numpy(dtype=float),
            g["shot_quality_holdout"].to_numpy(dtype=float),
        )
    return out


def bootstrap_match_contrast(
    a: pd.DataFrame, b: pd.DataFrame, metric_fn, n_boot: int = N_BOOT
) -> tuple[float, str]:
    """Cluster bootstrap at match level for a metric difference a - b."""
    rng = np.random.default_rng(RANDOM_STATE)
    a_by_match = by_match_arrays(a)
    b_by_match = by_match_arrays(b)
    a_ids = np.array(list(a_by_match.keys()))
    b_ids = np.array(list(b_by_match.keys()))

    point = metric_fn(
        a["goal"].to_numpy(dtype=float), a["shot_quality_holdout"].to_numpy(dtype=float)
    ) - metric_fn(
        b["goal"].to_numpy(dtype=float), b["shot_quality_holdout"].to_numpy(dtype=float)
    )

    diffs = []
    for _ in range(n_boot):
        a_draw = rng.choice(a_ids, size=len(a_ids), replace=True)
        b_draw = rng.choice(b_ids, size=len(b_ids), replace=True)
        ay = np.concatenate([a_by_match[m][0] for m in a_draw])
        ap = np.concatenate([a_by_match[m][1] for m in a_draw])
        by = np.concatenate([b_by_match[m][0] for m in b_draw])
        bp = np.concatenate([b_by_match[m][1] for m in b_draw])
        da = metric_fn(ay, ap)
        db = metric_fn(by, bp)
        if np.isfinite(da) and np.isfinite(db):
            diffs.append(da - db)
    diffs = np.asarray(diffs, dtype=float)
    ci = f"[{np.quantile(diffs, .025):.3f}, {np.quantile(diffs, .975):.3f}]" if len(diffs) else ""
    return float(point), ci


def main() -> None:
    header("STEP 16  Holdout (2024/25) calibration by subgroup")

    shots = load_frame("shots_scored")
    matches = load_frame("matches")

    holdout = shots[shots["season_name"] == "2024/25"].copy()
    print(f"  holdout shots: {len(holdout)}; observed goals {int(holdout['goal'].sum())}; "
          f"predicted xG {holdout['shot_quality_holdout'].sum():.1f}")

    # ---- strong/weak grouping fixed from the 2023/24 final table -----------
    standings = team_results(
        matches[matches["season_name"] == "2023/24"],
        shots[shots["season_name"] == "2023/24"],
    )
    strong_ids = set(standings.loc[standings["final_position"] <= 8, "team_id"])
    weak_ids_2324 = set(standings.loc[standings["final_position"] > 8, "team_id"])
    holdout_teams = set(holdout["team_id"].unique())
    promoted_ids = holdout_teams - strong_ids - weak_ids_2324
    weak_ids = weak_ids_2324 | promoted_ids

    print(f"  2023/24 standings: {len(standings)} teams; strong n={len(strong_ids)}, "
          f"weak (incl. promoted) n={len(weak_ids & holdout_teams)} "
          f"(promoted/new club(s): {promoted_ids})")

    home = holdout[holdout["is_home_shot"] == 1]
    away = holdout[holdout["is_home_shot"] == 0]
    strong = holdout[holdout["team_id"].isin(strong_ids)]
    weak = holdout[holdout["team_id"].isin(weak_ids)]

    rows = [
        group_metrics(holdout, "All shots"),
        group_metrics(home, "Home"),
        group_metrics(away, "Away"),
        group_metrics(strong, "Strong teams (2023/24 rank 1-8)"),
        group_metrics(weak, "Weak teams (2023/24 rank 9-15 + promoted)"),
    ]
    table = pd.DataFrame(rows)
    print("\n  subgroup calibration:")
    print(table.to_string(index=False))
    save_table(table, "tableS3_holdout_calibration_subgroups")

    # ---- bootstrap contrasts (match-level clusters, 2000 resamples) -------
    contrasts = []
    for label, a, b in [
        ("Home - Away", home, away),
        ("Strong - Weak", strong, weak),
    ]:
        oe_point, oe_ci = bootstrap_match_contrast(a, b, oe_ratio_arr)
        inter_point, inter_ci = bootstrap_match_contrast(a, b, intercept_arr)
        contrasts.append(
            {
                "Comparison": label,
                "O/E ratio difference": round(oe_point, 3),
                "O/E ratio 95% bootstrap CI": oe_ci,
                "Calibration intercept difference": round(inter_point, 3),
                "Calibration intercept 95% bootstrap CI": inter_ci,
            }
        )
    contrast_table = pd.DataFrame(contrasts)
    print("\n  bootstrap contrasts (2,000 match-cluster resamples):")
    print(contrast_table.to_string(index=False))
    save_table(contrast_table, "tableS3b_holdout_calibration_contrasts")


if __name__ == "__main__":
    main()
