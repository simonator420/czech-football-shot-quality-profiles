"""Reviewer sensitivity analysis: time-varying opponent defensive strength.

The primary opponent adjustment uses a single opponent effect.  This script
checks whether team-season attacking ratings change when opponent strength is
allowed to evolve within a season, using (i) opponent-by-season-half fixed
effects and (ii) strictly lagged 5- and 10-match rolling defensive records.

Produces
    tables/table_dynamic_opponent_model_fit.csv
    tables/table_dynamic_opponent_rank_robustness.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

from common import header, load_frame, save_table


OUTCOMES = {
    "xg_per_match": "total_xg",
    "shots_per_match": "shots",
    "mean_shot_quality": "mean_shot_quality",
}


def add_dynamic_defence(tm: pd.DataFrame) -> pd.DataFrame:
    """Attach only information available before the current match."""
    d = tm.copy().sort_values(["kickoff", "match_id", "team_id"]).reset_index(drop=True)
    d["team_season"] = d["team_id"].astype(str) + "_" + d["season_name"]
    d["season_half"] = np.where(d["match_number"] <= 15, "H1", "H2")
    d["opponent_period"] = (
        d["opponent_team_id"].astype(str)
        + "_"
        + d["season_name"].astype(str)
        + "_"
        + d["season_half"]
    )

    # Each attacking row is also one observation of what its opponent conceded.
    # Grouping by opponent_team_id therefore reconstructs the opponent's defensive
    # history. shift(1) prevents the current match entering its own adjustment.
    for label, col in OUTCOMES.items():
        for window in (5, 10):
            d[f"opp_prev_{window}_{label}"] = d.groupby("opponent_team_id", sort=False)[
                col
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
    return d


def fit_and_ratings(d: pd.DataFrame, outcome: str, specification: str, rolling_window=None):
    ycol = OUTCOMES[outcome]
    if specification == "Static opponent FE":
        formula = f"{ycol} ~ C(team_season) + C(opponent_team_id) + is_home + C(season_name)"
        used = d.copy()
    elif specification == "Opponent-by-half FE":
        formula = f"{ycol} ~ C(team_season) + C(opponent_period) + is_home + C(season_name)"
        used = d.copy()
    else:
        covar = f"opp_prev_{rolling_window}_{outcome}"
        used = d.dropna(subset=[covar]).copy()
        formula = f"{ycol} ~ C(team_season) + {covar} + is_home + C(season_name)"

    fit = smf.ols(formula, used).fit()
    base = float(fit.params["Intercept"])
    ratings = []
    for key in sorted(used["team_season"].unique()):
        term = f"C(team_season)[T.{key}]"
        ratings.append(
            {
                "team_season_key": key,
                "Specification": specification,
                "Outcome": outcome,
                "Adjusted rating": base + float(fit.params.get(term, 0.0)),
            }
        )
    ratings = pd.DataFrame(ratings)
    ratings[["team_id", "season_name"]] = ratings["team_season_key"].str.split(
        "_", n=1, expand=True
    )
    ratings["team_id"] = ratings["team_id"].astype(int)
    ratings["Adjusted rank"] = ratings.groupby("season_name")["Adjusted rating"].rank(
        ascending=False
    )

    rolling_beta = np.nan
    rolling_p = np.nan
    if rolling_window is not None:
        covar = f"opp_prev_{rolling_window}_{outcome}"
        rolling_beta = float(fit.params[covar])
        rolling_p = float(fit.pvalues[covar])
    fit_row = {
        "Outcome": outcome,
        "Specification": specification,
        "N team-matches": int(fit.nobs),
        "Adjusted R2": float(fit.rsquared_adj),
        "RMSE": float(np.sqrt(np.mean(fit.resid**2))),
        "AIC": float(fit.aic),
        "Rolling opponent coefficient": rolling_beta,
        "Rolling opponent p-value": rolling_p,
    }
    return fit_row, ratings


def main() -> None:
    header("STEP 17  Dynamic opponent-strength sensitivity")
    d = add_dynamic_defence(load_frame("team_match_profiles"))

    fit_rows = []
    rating_frames = []
    specifications = [
        ("Static opponent FE", None),
        ("Opponent-by-half FE", None),
        ("Lagged 5-match opponent defence", 5),
        ("Lagged 10-match opponent defence", 10),
    ]
    for outcome in OUTCOMES:
        for spec, window in specifications:
            row, ratings = fit_and_ratings(d, outcome, spec, window)
            fit_rows.append(row)
            rating_frames.append(ratings)

    fit_table = pd.DataFrame(fit_rows)
    save_table(fit_table.round(5), "table_dynamic_opponent_model_fit")

    all_ratings = pd.concat(rating_frames, ignore_index=True)
    comparison_rows = []
    for outcome in OUTCOMES:
        wide = all_ratings[all_ratings["Outcome"] == outcome].pivot(
            index=["team_id", "season_name"],
            columns="Specification",
            values="Adjusted rank",
        )
        ref = wide["Static opponent FE"]
        for spec, _ in specifications[1:]:
            valid = wide[["Static opponent FE", spec]].dropna()
            delta = valid[spec] - valid["Static opponent FE"]
            comparison_rows.append(
                {
                    "Outcome": outcome,
                    "Comparison with static opponent FE": spec,
                    "N team-seasons": len(valid),
                    "Spearman rank correlation": float(
                        spearmanr(valid["Static opponent FE"], valid[spec]).correlation
                    ),
                    "Mean absolute rank change": float(delta.abs().mean()),
                    "Maximum absolute rank change": float(delta.abs().max()),
                }
            )
    rank_table = pd.DataFrame(comparison_rows)
    save_table(rank_table.round(4), "table_dynamic_opponent_rank_robustness")

    print("\n  model fit and rolling-strength coefficients:")
    print(fit_table.round(4).to_string(index=False))
    print("\n  robustness of adjusted team-season ranks:")
    print(rank_table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
