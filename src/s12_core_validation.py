"""Step 12 - Core validation analyses for the main claims.

This step adds four checks that are easier to read as direct tests of the
paper's headline claims than the broad exploratory tables:

* split-half reliability and regression to the mean;
* practical validity of a process-only chance-creation axis;
* opponent-adjusted attacking ratings;
* temporal development/final holdout shot-quality validation.

Produces
    tables/table8_split_half_reliability.csv
    tables/table8b_split_half_differences.csv
    tables/table9_practical_validity.csv
    tables/table9b_opponent_adjusted_profiles.csv
    tables/table3c_temporal_validation.csv
    figures/figure10_split_half_reliability.pdf
    figures/figure11_practical_validity.pdf
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

import plotstyle
from common import MODELS, RANDOM_STATE, SEASONS, TEST_SEASONS, TRAIN_SEASONS, header, load_frame, save_table
from modelling import BINARY, CATEGORICAL, NUMERIC, TARGET, evaluate
from s03_shot_quality import build_candidates, refit_full
from s04_profiles import PROFILE_FEATURES, _core_aggregates
from s05_clustering import standardise_within_season

warnings.filterwarnings("ignore")

SPLIT_FEATURES = [
    "shots_per_match",
    "xg_per_match",
    "mean_shot_quality",
    "high_quality_share",
    "chance_creation_axis",
    "goals_per_match",
    "goals_minus_xg_per_match",
    "on_target_rate",
    "finishing_axis",
]

PROCESS_FEATURES = [
    "shots_per_match",
    "xg_per_match",
    "mean_shot_quality",
    "median_shot_quality",
    "high_quality_share",
    "low_quality_share",
    "mean_distance_m",
    "mean_centrality",
    "set_piece_share",
    "header_share",
    "close_range_share",
]


def fit_axis_reference(ts: pd.DataFrame) -> dict:
    Z = standardise_within_season(ts, PROFILE_FEATURES)
    pca = PCA().fit(Z)
    signs = np.array([
        np.sign(pd.Series(pca.components_[0], index=PROFILE_FEATURES)["mean_shot_quality"]) or 1.0,
        np.sign(pd.Series(pca.components_[1], index=PROFILE_FEATURES)["goals_minus_xg_per_match"]) or 1.0,
    ])
    by_season = {}
    for season, g in ts.groupby("season_name"):
        block = g[PROFILE_FEATURES].to_numpy(dtype=float)
        mu = block.mean(axis=0)
        sd = block.std(axis=0, ddof=0)
        sd[sd == 0] = 1.0
        by_season[season] = (mu, sd)
    return {"pca": pca, "signs": signs, "by_season": by_season}


def add_axes(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    out = df.copy()
    scores = np.full((len(out), 2), np.nan)
    for season, idx in out.groupby("season_name").groups.items():
        mu, sd = ref["by_season"][season]
        z = (out.loc[idx, PROFILE_FEATURES].to_numpy(dtype=float) - mu) / sd
        scores[out.index.get_indexer(idx)] = ref["pca"].transform(z)[:, :2] * ref["signs"]
    out["chance_creation_axis"] = scores[:, 0]
    out["finishing_axis"] = scores[:, 1]
    return out


def aggregate_team_window(shots: pd.DataFrame, tm: pd.DataFrame, selector, axis_ref: dict) -> pd.DataFrame:
    rows = []
    for (team_id, season), g in tm.groupby(["team_id", "season_name"]):
        sel = selector(g)
        if sel.empty:
            continue
        sub = shots[shots["match_id"].isin(sel["match_id"]) & (shots["team_id"] == team_id)]
        if len(sub) < 20:
            continue
        rec = {
            "team_id": team_id,
            "season_name": season,
            "team_name": g["team_name"].iloc[0],
            "matches": len(sel),
        }
        agg = _core_aggregates(sub)
        rec.update(agg)
        rec["shots_per_match"] = agg["shots"] / len(sel)
        rec["xg_per_match"] = agg["total_xg"] / len(sel)
        rec["goals_per_match"] = agg["goals"] / len(sel)
        rec["goals_minus_xg_per_match"] = (agg["goals"] - agg["total_xg"]) / len(sel)
        rows.append(rec)
    return add_axes(pd.DataFrame(rows), axis_ref)


def ci(vals: np.ndarray) -> str:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return f"[{np.quantile(vals, .025):.3f}, {np.quantile(vals, .975):.3f}]" if len(vals) else ""


def corr_pair(data: pd.DataFrame, feature: str) -> float:
    a, b = f"{feature}_first", f"{feature}_second"
    ok = data[[a, b]].dropna()
    if len(ok) < 3 or ok[a].std() == 0 or ok[b].std() == 0:
        return np.nan
    return float(pearsonr(ok[a], ok[b]).statistic)


def paired_cluster_corr_contrasts(
    data: pd.DataFrame,
    pearson_estimates: dict[str, float],
    contrast_specs: list[tuple[str, str, str]],
    n_boot: int = 2000,
) -> pd.DataFrame:
    """Bootstrap correlation differences using the same resampled teams."""
    rng = np.random.default_rng(RANDOM_STATE)
    teams = data["team_id"].drop_duplicates().to_numpy()
    boot = {label: [] for label, _, _ in contrast_specs}
    for _ in range(n_boot):
        sampled = rng.choice(teams, size=len(teams), replace=True)
        parts = []
        for i, team in enumerate(sampled):
            block = data[data["team_id"] == team].copy()
            block["boot_team_id"] = f"{team}_{i}"
            parts.append(block)
        draw = pd.concat(parts, ignore_index=True)
        for label, left, right in contrast_specs:
            left_r = corr_pair(draw, left)
            right_r = corr_pair(draw, right)
            if np.isfinite(left_r) and np.isfinite(right_r):
                boot[label].append(left_r - right_r)

    rows = []
    for label, left, right in contrast_specs:
        diff = np.asarray(boot[label], dtype=float)
        p_two = np.nan
        if len(diff):
            p_two = min(1.0, float(2 * min((diff <= 0).mean(), (diff >= 0).mean())))
        rows.append(
            {
                "Comparison": label,
                "Pearson r difference": round(
                    round(float(pearson_estimates[left]), 3)
                    - round(float(pearson_estimates[right]), 3),
                    3,
                ),
                "95% paired cluster bootstrap CI": ci(diff),
                "Two-sided paired bootstrap p": format_p(p_two),
            }
        )
    return pd.DataFrame(rows)


def split_half_reliability(shots, tm, axis_ref):
    first = aggregate_team_window(
        shots,
        tm,
        lambda g: g[g["match_number"] <= int(np.floor(g["match_number"].max() / 2))],
        axis_ref,
    )
    second = aggregate_team_window(
        shots,
        tm,
        lambda g: g[g["match_number"] > int(np.floor(g["match_number"].max() / 2))],
        axis_ref,
    )
    d = first.merge(second, on=["team_id", "season_name"], suffixes=("_first", "_second"))
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    pearson_estimates = {}
    for f in SPLIT_FEATURES:
        a, b = f"{f}_first", f"{f}_second"
        ok = d[[a, b]].dropna()
        pear = pearsonr(ok[a], ok[b]).statistic
        pearson_estimates[f] = float(pear)
        spear = spearmanr(ok[a], ok[b]).correlation
        mae = mean_abs = float(np.mean(np.abs(ok[a] - ok[b])))
        pear_boot, spear_boot = [], []
        for _ in range(2000):
            idx = rng.integers(0, len(ok), len(ok))
            aa, bb = ok[a].to_numpy()[idx], ok[b].to_numpy()[idx]
            if np.std(aa) == 0 or np.std(bb) == 0:
                continue
            pear_boot.append(pearsonr(aa, bb).statistic)
            spear_boot.append(spearmanr(aa, bb).correlation)
        rows.append(
            {
                "Feature": f,
                "Pearson r": round(pear, 3),
                "Pearson 95% bootstrap CI": ci(pear_boot),
                "Spearman rho": round(spear, 3),
                "Spearman 95% bootstrap CI": ci(spear_boot),
                "MAE first vs second half": round(mae, 3),
                "n team-seasons": len(ok),
            }
        )

    contrast_specs = [
        ("chance_creation_axis minus finishing_axis", "chance_creation_axis", "finishing_axis"),
        ("xg_per_match minus goals_minus_xg_per_match", "xg_per_match", "goals_minus_xg_per_match"),
    ]
    diff_rows = paired_cluster_corr_contrasts(d, pearson_estimates, contrast_specs)
    return pd.DataFrame(rows), diff_rows, d


def team_results(matches: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, m in matches.iterrows():
        hs, aas = int(m["home_score_current"]), int(m["away_score_current"])
        for side in ("home", "away"):
            is_home = side == "home"
            gf, ga = (hs, aas) if is_home else (aas, hs)
            rows.append(
                {
                    "team_id": int(m[f"{side}_team_id"]),
                    "season_name": m["season_name"],
                    "team_name": m[f"{side}_team_name"],
                    "match_id": m["match_id"],
                    "points": 3 if gf > ga else (1 if gf == ga else 0),
                    "goals_for": gf,
                    "goals_against": ga,
                }
            )
    res = pd.DataFrame(rows)
    out = res.groupby(["team_id", "season_name", "team_name"], as_index=False).agg(
        matches=("match_id", "nunique"),
        points=("points", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
    )
    out["points_per_match"] = out["points"] / out["matches"]
    out["goal_difference_per_match"] = (out["goals_for"] - out["goals_against"]) / out["matches"]

    nonpen = shots[shots["is_penalty"] == 0].groupby(
        ["team_id", "season_name"], as_index=False
    )["goal"].sum().rename(columns={"goal": "nonpen_goals_for"})
    against = shots[shots["is_penalty"] == 0].groupby(
        ["opponent_team_id", "season_name"], as_index=False
    )["goal"].sum().rename(columns={"opponent_team_id": "team_id", "goal": "nonpen_goals_against"})
    out = out.merge(nonpen, on=["team_id", "season_name"], how="left")
    out = out.merge(against, on=["team_id", "season_name"], how="left")
    out[["nonpen_goals_for", "nonpen_goals_against"]] = out[
        ["nonpen_goals_for", "nonpen_goals_against"]
    ].fillna(0)
    out["nonpen_goal_difference_per_match"] = (
        out["nonpen_goals_for"] - out["nonpen_goals_against"]
    ) / out["matches"]
    out["final_position"] = out.groupby("season_name")["points"].rank(
        method="first", ascending=False
    ).astype(int)
    return out


def add_process_axis(ts: pd.DataFrame) -> pd.DataFrame:
    out = ts.copy()
    feats = [f for f in PROCESS_FEATURES if f in out.columns]
    Z = standardise_within_season(out, feats)
    pca = PCA().fit(Z)
    scores = pca.transform(Z)[:, 0]
    sign = np.sign(pd.Series(pca.components_[0], index=feats)["xg_per_match"]) or 1.0
    out["process_chance_creation_axis"] = scores * sign
    return out


def format_p(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "< .001"
    return f"{p:.3f}"


def cluster_bootstrap_ci(data: pd.DataFrame, formula: str, term: str, n_boot: int = 2000):
    rng = np.random.default_rng(RANDOM_STATE)
    teams = data["team_id"].drop_duplicates().to_numpy()
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(teams, size=len(teams), replace=True)
        parts = []
        for i, team in enumerate(sampled):
            block = data[data["team_id"] == team].copy()
            block["boot_team_id"] = f"{team}_{i}"
            parts.append(block)
        boot = pd.concat(parts, ignore_index=True)
        try:
            vals.append(float(smf.ols(formula, boot).fit().params[term]))
        except Exception:
            continue
    vals = np.asarray(vals, dtype=float)
    est = float(smf.ols(formula, data).fit().params[term])
    p_two = float(2 * min((vals <= 0).mean(), (vals >= 0).mean())) if len(vals) else np.nan
    return ci(vals), p_two


def practical_validity(ts, matches, shots):
    ts2 = add_process_axis(ts)
    res = team_results(matches, shots)
    d = ts2.merge(res, on=["team_id", "season_name", "team_name"], how="inner")
    formulas = {
        "points_per_match": "points_per_match ~ process_chance_creation_axis + finishing_axis + C(season_name)",
        "goal_difference_per_match": "goal_difference_per_match ~ process_chance_creation_axis + finishing_axis + C(season_name)",
        "nonpen_goal_difference_per_match": "nonpen_goal_difference_per_match ~ process_chance_creation_axis + finishing_axis + C(season_name)",
    }
    rows = []
    for outcome, formula in formulas.items():
        fit = smf.ols(formula, d).fit(cov_type="cluster", cov_kwds={"groups": d["team_id"]})
        for term in ("process_chance_creation_axis", "finishing_axis"):
            boot_ci, boot_p = cluster_bootstrap_ci(d, formula, term)
            rows.append(
                {
                    "Outcome": outcome,
                    "Term": term,
                    "Estimate": round(float(fit.params[term]), 4),
                    "SE cluster(team)": round(float(fit.bse[term]), 4),
                    "p cluster(team)": format_p(float(fit.pvalues[term])),
                    "Team-cluster bootstrap 95% CI": boot_ci,
                    "Team-cluster bootstrap p": format_p(boot_p),
                    "Model R2": round(float(fit.rsquared), 3),
                    "Inference note": "Exploratory concurrent association; team-cluster bootstrap checks small-cluster sensitivity",
                }
            )
    for x in ("process_chance_creation_axis", "finishing_axis"):
        for y in ("points_per_match", "goal_difference_per_match"):
            rows.append(
                {
                    "Outcome": y,
                    "Term": f"Spearman({x})",
                    "Estimate": round(float(spearmanr(d[x], d[y]).correlation), 4),
                    "SE cluster(team)": np.nan,
                    "p cluster(team)": "",
                    "Team-cluster bootstrap 95% CI": "",
                    "Team-cluster bootstrap p": "",
                    "Model R2": np.nan,
                    "Inference note": "Descriptive rank association",
                }
            )
    return pd.DataFrame(rows), d


def opponent_adjusted(tm: pd.DataFrame, ts: pd.DataFrame) -> pd.DataFrame:
    d = tm.copy()
    d["team_season"] = d["team_id"].astype(str) + "_" + d["season_name"]
    d["xg_per_match"] = d["total_xg"]
    d["shots_per_match"] = d["shots"]
    rows = []
    for outcome in ("xg_per_match", "shots_per_match", "mean_shot_quality"):
        fit = smf.ols(
            f"{outcome} ~ C(team_season) + C(opponent_team_id) + is_home + C(season_name)",
            d,
        ).fit()
        base = float(fit.params["Intercept"])
        ratings = []
        for team_season in sorted(d["team_season"].unique()):
            term = f"C(team_season)[T.{team_season}]"
            ratings.append({"team_season_key": team_season, f"adjusted_{outcome}": base + float(fit.params.get(term, 0.0))})
        adj = pd.DataFrame(ratings)
        rows.append(adj)
    out = rows[0]
    for r in rows[1:]:
        out = out.merge(r, on="team_season_key")
    out[["team_id", "season_name"]] = out["team_season_key"].str.split("_", n=1, expand=True)
    out["team_id"] = out["team_id"].astype(int)
    out = out.merge(
        ts[["team_id", "season_name", "team_name", "chance_creation_axis", "xg_per_match"]],
        on=["team_id", "season_name"],
    )
    out["raw_xg_rank"] = out.groupby("season_name")["xg_per_match"].rank(ascending=False)
    out["adjusted_xg_rank"] = out.groupby("season_name")["adjusted_xg_per_match"].rank(ascending=False)
    out["rank_change_adjusted_minus_raw"] = out["adjusted_xg_rank"] - out["raw_xg_rank"]
    rho = spearmanr(out["adjusted_xg_per_match"], out["chance_creation_axis"]).correlation
    out["Spearman adjusted xG vs chance axis"] = round(float(rho), 3)
    return out.sort_values(["season_name", "adjusted_xg_rank"])


def best_config():
    log = json.loads((MODELS / "tuning_log.json").read_text())
    best = min(log, key=lambda r: r["inner_logloss"])
    return best["family"], best["config"], best.get("best_iter")


def rolling_origin_validation(shots: pd.DataFrame) -> pd.DataFrame:
    family, config, best_iter = best_config()
    factory = dict(build_candidates(NUMERIC, BINARY, CATEGORICAL)[family])[config]
    cols = NUMERIC + BINARY + CATEGORICAL
    rows = []
    for train_seasons, test_season in (
        (["2022/23"], "2023/24"),
        (TRAIN_SEASONS, TEST_SEASONS[0]),
    ):
        tr = shots["season_name"].isin(train_seasons).to_numpy()
        te = (shots["season_name"] == test_season).to_numpy()
        pipe = refit_full(factory, shots.loc[tr, cols], shots.loc[tr, TARGET], best_iter)
        pred = pipe.predict_proba(shots.loc[te, cols])[:, 1]
        base = float(shots.loc[tr, TARGET].mean())
        label = (
            "Temporal development validation: 2022/23 -> 2023/24"
            if train_seasons == ["2022/23"]
            else "Final temporal holdout: 2022/23 + 2023/24 -> 2024/25"
        )
        row = evaluate(shots.loc[te, TARGET].to_numpy(), pred, base, label)
        rows.append(row)
    out = pd.DataFrame(rows)
    for c in ("AUC", "PR-AUC", "Log loss", "Brier score", "Brier skill score", "Calibration slope", "Calibration intercept", "ECE"):
        out[c] = out[c].astype(float).round(4)
    return out


def main() -> None:
    header("STEP 12  Core validation analyses")
    plotstyle.apply()

    shots = load_frame("shots_scored")
    tm = load_frame("team_match_profiles")
    ts = load_frame("team_season_clustered")
    matches = load_frame("matches")
    axis_ref = fit_axis_reference(ts)

    split, diffs, split_data = split_half_reliability(shots, tm, axis_ref)
    save_table(split, "table8_split_half_reliability")
    save_table(diffs, "table8b_split_half_differences")
    print("\n  split-half reliability:")
    print(split.to_string(index=False))
    print(diffs.to_string(index=False))

    practical, practical_data = practical_validity(ts, matches, shots)
    save_table(practical, "table9_practical_validity")
    print("\n  practical validity models:")
    print(practical.to_string(index=False))

    adjusted = opponent_adjusted(tm, ts)
    save_table(adjusted.round(4), "table9b_opponent_adjusted_profiles")
    print("\n  opponent-adjusted xG rating vs chance-creation axis:")
    print(adjusted[["Spearman adjusted xG vs chance axis"]].head(1).to_string(index=False))

    rolling = rolling_origin_validation(shots)
    save_table(rolling, "table3c_temporal_validation")
    print("\n  temporal shot-quality validation:")
    print(rolling.to_string(index=False))

    stab = pd.read_csv("outputs/tables/tableS2_stabilisation_curves.csv")
    make_split_half_figure(split, split_data, stab)
    make_practical_figure(practical_data)


def make_split_half_figure(split: pd.DataFrame, d: pd.DataFrame, stab: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(plotstyle.W_DOUBLE, 5.3))
    axes = axes.ravel()
    panels = [
        ("xg_per_match", "xG per match"),
        ("goals_minus_xg_per_match", "Goals minus xG per match"),
    ]
    for ax, (feat, title) in zip(axes[:2], panels):
        x, y = d[f"{feat}_first"], d[f"{feat}_second"]
        ax.scatter(x, y, s=28, color=plotstyle.CATEGORICAL[0], edgecolor="white", linewidth=0.5)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color=plotstyle.INK_MUTED, lw=0.8, ls=(0, (3, 3)))
        ax.set_xlabel("First half")
        ax.set_ylabel("Second half")
        ax.set_title(title, loc="left")
    ax = axes[2]
    rr = split.sort_values("Pearson r")
    ax.barh(range(len(rr)), rr["Pearson r"], color=plotstyle.CATEGORICAL[1], height=0.7)
    ax.set_yticks(range(len(rr)))
    ax.set_yticklabels([f.replace("_", " ") for f in rr["Feature"]], fontsize=6)
    ax.set_xlabel("First-half vs second-half Pearson r")
    ax.set_title("Reliability by indicator", loc="left")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    ax = axes[3]
    piv = stab.pivot(index="Matches", columns="Feature", values="Pearson r with remainder")
    for i, feat in enumerate(["xg_per_match", "chance_creation_axis", "goals_minus_xg_per_match", "finishing_axis"]):
        if feat not in piv:
            continue
        ax.plot(
            piv.index,
            piv[feat],
            marker=plotstyle.MARKERS[i % len(plotstyle.MARKERS)],
            color=plotstyle.CATEGORICAL[i % len(plotstyle.CATEGORICAL)],
            label=feat.replace("_", " "),
            markersize=3.2,
            markeredgecolor="white",
            markeredgewidth=0.4,
        )
    ax.axhline(0.70, color=plotstyle.INK_MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.set_xlabel("Matches elapsed")
    ax.set_ylabel("Pearson r with remainder")
    ax.set_title("Stabilisation across the season", loc="left")
    ax.legend(frameon=False, fontsize=6.2)
    ax.grid(axis="y")
    fig.tight_layout()
    plotstyle.save(fig, "figure10_split_half_reliability")


def make_practical_figure(d: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(plotstyle.W_DOUBLE, 3.1))
    for ax, y, title in (
        (axes[0], "points_per_match", "Points per match"),
        (axes[1], "goal_difference_per_match", "Goal difference per match"),
    ):
        for i, season in enumerate(SEASONS):
            sub = d[d["season_name"] == season]
            ax.scatter(
                sub["process_chance_creation_axis"],
                sub[y],
                s=30,
                color=plotstyle.CATEGORICAL[i],
                label=season,
                edgecolor="white",
                linewidth=0.5,
            )
        fit = smf.ols(f"{y} ~ process_chance_creation_axis", d).fit()
        xs = np.linspace(d["process_chance_creation_axis"].min(), d["process_chance_creation_axis"].max(), 100)
        pred = fit.get_prediction(pd.DataFrame({"process_chance_creation_axis": xs})).summary_frame()
        ax.plot(xs, pred["mean"], color=plotstyle.INK, lw=1.1)
        ax.fill_between(xs, pred["mean_ci_lower"], pred["mean_ci_upper"], color=plotstyle.INK_MUTED, alpha=0.18)
        ax.set_xlabel("Process-only chance-creation axis")
        ax.set_ylabel(title)
        ax.set_title(title, loc="left")
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    plotstyle.save(fig, "figure11_practical_validity")


if __name__ == "__main__":
    main()
