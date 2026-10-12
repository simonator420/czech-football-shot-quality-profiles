"""Goal-independent shot-process axes requested in peer review.

No feature in either axis uses goals, xG, model probabilities, shots on target,
blocked-shot outcomes, or conversion.  The primary version uses volume, raw
shot geometry, body part and chance-source labels.  A taxonomy-robust version
also removes assisted, set-piece and fast-break provider labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

from common import RANDOM_STATE, SEASONS, header, load_frame, save_table
from s04_profiles import _core_aggregates
from s06_stability import icc1
from s12_core_validation import cluster_bootstrap_ci, format_p, team_results


FEATURE_SETS = {
    "Goal-independent full": [
        "shots_per_match",
        "mean_distance_m",
        "mean_angle_proxy",
        "mean_centrality",
        "header_share",
        "assisted_share",
        "set_piece_share",
        "fast_break_share",
        "close_range_share",
        "inside_box_share",
        "wide_angle_share",
    ],
    "Goal-independent taxonomy-robust": [
        "shots_per_match",
        "mean_distance_m",
        "mean_angle_proxy",
        "mean_centrality",
        "header_share",
        "close_range_share",
        "inside_box_share",
        "wide_angle_share",
    ],
}

N_BOOT = 2000


def fit_reference(ts: pd.DataFrame, features: list[str]) -> dict:
    by_season = {}
    z = np.empty((len(ts), len(features)), dtype=float)
    for season, g in ts.groupby("season_name"):
        idx = ts.index.get_indexer(g.index)
        block = g[features].to_numpy(float)
        mu, sd = block.mean(0), block.std(0, ddof=0)
        sd[sd == 0] = 1.0
        by_season[season] = (mu, sd)
        z[idx] = (block - mu) / sd
    pca = PCA().fit(z)

    # Orient without referring to goals or xG: higher means more shots and a
    # larger share of close/central/inside-box attempts at shorter distance.
    threat = (
        z[:, features.index("shots_per_match")]
        - z[:, features.index("mean_distance_m")]
        + z[:, features.index("mean_centrality")]
        + z[:, features.index("close_range_share")]
        + z[:, features.index("inside_box_share")]
        - z[:, features.index("wide_angle_share")]
    )
    raw = pca.transform(z)[:, 0]
    sign = 1.0 if np.corrcoef(raw, threat)[0, 1] >= 0 else -1.0
    return {"features": features, "pca": pca, "by_season": by_season, "sign": sign}


def apply_axis(df: pd.DataFrame, ref: dict, name: str) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    scores = np.full(len(out), np.nan)
    for season, idx in out.groupby("season_name").groups.items():
        mu, sd = ref["by_season"][season]
        z = (out.loc[idx, ref["features"]].to_numpy(float) - mu) / sd
        scores[np.asarray(list(idx), dtype=int)] = (
            ref["pca"].transform(z)[:, 0] * ref["sign"]
        )
    out[name] = scores
    return out


def aggregate_half(shots, tm, first: bool, ref: dict, name: str) -> pd.DataFrame:
    rows = []
    for (team_id, season), g in tm.groupby(["team_id", "season_name"]):
        midpoint = int(np.floor(g["match_number"].max() / 2))
        selected = g[g["match_number"] <= midpoint] if first else g[g["match_number"] > midpoint]
        sub = shots[
            shots["match_id"].isin(selected["match_id"])
            & (shots["team_id"] == team_id)
        ]
        if len(sub) < 20:
            continue
        rec = {"team_id": team_id, "season_name": season}
        rec.update(_core_aggregates(sub))
        rec["shots_per_match"] = len(sub) / len(selected)
        rows.append(rec)
    return apply_axis(pd.DataFrame(rows), ref, name)


def reliability(ts_scored, shots, tm, ref, axis_name):
    x, y = [], []
    for a, b in zip(SEASONS[:-1], SEASONS[1:]):
        aa = ts_scored[ts_scored.season_name == a].set_index("team_id")[axis_name]
        bb = ts_scored[ts_scored.season_name == b].set_index("team_id")[axis_name]
        common = aa.index.intersection(bb.index)
        x.extend(aa.loc[common]); y.extend(bb.loc[common])
    x, y = np.asarray(x), np.asarray(y)
    r, p = pearsonr(x, y)
    z, se = np.arctanh(r), 1 / np.sqrt(len(x) - 3)
    between_ci = np.tanh([z - 1.96 * se, z + 1.96 * se])

    first = aggregate_half(shots, tm, True, ref, axis_name)
    second = aggregate_half(shots, tm, False, ref, axis_name)
    split = first.merge(second, on=["team_id", "season_name"], suffixes=("_first", "_second"))
    a, b = split[f"{axis_name}_first"].to_numpy(), split[f"{axis_name}_second"].to_numpy()
    within_r = pearsonr(a, b).statistic
    within_rho = spearmanr(a, b).correlation
    rng = np.random.default_rng(RANDOM_STATE)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(a), len(a))
        if np.std(a[idx]) and np.std(b[idx]):
            boots.append(pearsonr(a[idx], b[idx]).statistic)
    lo, hi = np.quantile(boots, [.025, .975])
    return {
        "Axis": axis_name,
        "Between-season Pearson r": r,
        "Between-season 95% CI": f"[{between_ci[0]:.3f}, {between_ci[1]:.3f}]",
        "Between-season p": format_p(p),
        "ICC(1,1)": icc1(ts_scored[["team_id", axis_name]], "team_id", axis_name),
        "Between-season n pairs": len(x),
        "Within-season Pearson r": within_r,
        "Within-season 95% bootstrap CI": f"[{lo:.3f}, {hi:.3f}]",
        "Within-season Spearman rho": within_rho,
        "Within-season n": len(a),
    }


def validity(scored, matches, shots, axis_name):
    d = scored.merge(
        team_results(matches, shots),
        on=["team_id", "season_name", "team_name"],
        how="inner",
    )
    rows = []
    for outcome in ("points_per_match", "goal_difference_per_match", "nonpen_goal_difference_per_match"):
        formula = f"{outcome} ~ {axis_name} + finishing_axis + C(season_name)"
        fit = smf.ols(formula, d).fit(cov_type="cluster", cov_kwds={"groups": d["team_id"]})
        boot_ci, boot_p = cluster_bootstrap_ci(d, formula, axis_name, n_boot=N_BOOT)
        rows.append({
            "Axis": axis_name,
            "Outcome": outcome,
            "Estimate adjusted for finishing axis and season": fit.params[axis_name],
            "Cluster-robust SE": fit.bse[axis_name],
            "Cluster-robust p": format_p(fit.pvalues[axis_name]),
            "Team-cluster bootstrap 95% CI": boot_ci,
            "Team-cluster bootstrap p": format_p(boot_p),
            "Spearman rho (unadjusted)": spearmanr(d[axis_name], d[outcome]).correlation,
            "Model R2": fit.rsquared,
        })
    return rows


def main():
    header("STEP 18  Completely goal-independent process indicators")
    ts = load_frame("team_season_clustered").reset_index(drop=True)
    shots = load_frame("shots_scored")
    tm = load_frame("team_match_profiles")
    matches = load_frame("matches")

    reliability_rows, validity_rows, loading_rows = [], [], []
    for label, features in FEATURE_SETS.items():
        axis_name = "goal_independent_axis" if label.endswith("full") else "goal_independent_robust_axis"
        ref = fit_reference(ts, features)
        scored = apply_axis(ts, ref, axis_name)
        reliability_rows.append(reliability(scored, shots, tm, ref, axis_name))
        validity_rows.extend(validity(scored, matches, shots, axis_name))
        for feature, loading in zip(features, ref["pca"].components_[0] * ref["sign"]):
            loading_rows.append({
                "Axis": axis_name,
                "Feature": feature,
                "PC1 loading": loading,
                "PC1 variance explained": ref["pca"].explained_variance_ratio_[0],
            })

    rel = pd.DataFrame(reliability_rows)
    val = pd.DataFrame(validity_rows)
    loads = pd.DataFrame(loading_rows)
    save_table(rel.round(4), "tableS5_goal_independent_axis_reliability")
    save_table(val.round(4), "tableS5b_goal_independent_axis_validity")
    save_table(loads.round(4), "tableS5c_goal_independent_axis_loadings")
    print("\nReliability:\n", rel.round(4).to_string(index=False))
    print("\nValidity:\n", val.round(4).to_string(index=False))
    print("\nLoadings:\n", loads.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
