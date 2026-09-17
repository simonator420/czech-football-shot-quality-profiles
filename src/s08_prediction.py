"""Step 8 - Early-season prediction without part-whole overlap.

Research question 3 asks whether early-season attacking behaviour has practical
predictive value. The target must therefore exclude the matches used to build
the predictors: first-5, first-10 and first-30%-of-season windows are used to
predict only the remainder of the same season.

Produces
    tables/tableS2_stabilisation_curves.csv
    tables/tableS3_early_season_regression.csv
    figures/figureS1_stabilisation.pdf
    figures/figureS2_early_season_shap.pdf
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import plotstyle
from common import RANDOM_STATE, TEST_SEASONS, TRAIN_SEASONS, header, load_frame, save_table
from s04_profiles import PROFILE_FEATURES, _core_aggregates
from s05_clustering import standardise_within_season

warnings.filterwarnings("ignore")

WINDOWS = {"First 5 matches": 5, "First 10 matches": 10, "First 30% of season": None}
STABILISATION_WINDOWS = [3, 5, 7, 10, 12, 15, 18, 20]
AXES = ["chance_creation_axis", "finishing_axis"]
TARGETS = AXES + ["xg_per_match", "goals_minus_xg_per_match"]

EARLY_FEATURES = [
    "chance_creation_axis",
    "finishing_axis",
    "mean_shot_quality",
    "median_shot_quality",
    "xg_per_match",
    "shots_per_match",
    "goals_minus_xg_per_match",
    "high_quality_share",
    "low_quality_share",
    "mean_distance_m",
    "mean_angle_proxy",
    "close_range_share",
    "header_share",
    "set_piece_share",
    "assisted_share",
    "fast_break_share",
    "on_target_rate",
    "blocked_rate",
    "goal_rate",
]

STABILISATION_FEATURES = [
    "chance_creation_axis",
    "finishing_axis",
    "xg_per_match",
    "shots_per_match",
    "mean_shot_quality",
    "goals_minus_xg_per_match",
    "on_target_rate",
]


def fit_axis_reference(ts: pd.DataFrame) -> dict:
    """Fit profile axes from training seasons only.

    Training seasons keep their within-season scaling. Unseen seasons use the
    pooled training-season mean and SD so the transformation does not depend on
    future full-season information from the holdout year.
    """
    train = ts[ts["season_name"].isin(TRAIN_SEASONS)].copy()
    Z = standardise_within_season(train, PROFILE_FEATURES)
    pca = PCA().fit(Z)
    load1 = pd.Series(pca.components_[0], index=PROFILE_FEATURES)
    load2 = pd.Series(pca.components_[1], index=PROFILE_FEATURES)
    sign1 = np.sign(load1["mean_shot_quality"]) or 1.0
    sign2 = np.sign(load2["goals_minus_xg_per_match"]) or 1.0
    by_season = {}
    for season, g in train.groupby("season_name"):
        block = g[PROFILE_FEATURES].to_numpy(dtype=float)
        mu = block.mean(axis=0)
        sd = block.std(axis=0, ddof=0)
        sd[sd == 0] = 1.0
        by_season[season] = (mu, sd)
    block = train[PROFILE_FEATURES].to_numpy(dtype=float)
    pooled_mu = block.mean(axis=0)
    pooled_sd = block.std(axis=0, ddof=0)
    pooled_sd[pooled_sd == 0] = 1.0
    return {
        "pca": pca,
        "signs": np.array([sign1, sign2]),
        "by_season": by_season,
        "fallback": (pooled_mu, pooled_sd),
    }


def add_profile_axes(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    out = df.copy()
    scores = np.full((len(out), 2), np.nan)
    for season, idx in out.groupby("season_name").groups.items():
        mu, sd = ref["by_season"].get(season, ref["fallback"])
        block = out.loc[idx, PROFILE_FEATURES].to_numpy(dtype=float)
        z = (block - mu) / sd
        scores[out.index.get_indexer(idx)] = ref["pca"].transform(z)[:, :2] * ref["signs"]
    out["chance_creation_axis"] = scores[:, 0]
    out["finishing_axis"] = scores[:, 1]
    return out


def use_temporal_holdout_shot_quality(shots: pd.DataFrame) -> pd.DataFrame:
    """Use train-only shot-quality predictions for the held-out test season."""
    out = shots.copy()
    test = out["season_name"].isin(TEST_SEASONS) & out["shot_quality_holdout"].notna()
    out.loc[test, "shot_quality"] = out.loc[test, "shot_quality_holdout"]
    return out


def early_window_features(
    shots: pd.DataFrame,
    tm: pd.DataFrame,
    n_matches,
    axis_ref: dict,
    complement: bool = False,
) -> pd.DataFrame:
    rows = []
    for (team_id, season), g in tm.groupby(["team_id", "season_name"]):
        total = int(g["match_number"].max())
        cutoff = int(np.ceil(0.30 * total)) if n_matches is None else int(n_matches)
        sel = g[g["match_number"] > cutoff] if complement else g[g["match_number"] <= cutoff]
        if sel.empty:
            continue
        sub = shots[shots["match_id"].isin(sel["match_id"]) & (shots["team_id"] == team_id)]
        if len(sub) < 20:
            continue
        rec = {
            "team_id": team_id,
            "season_name": season,
            "team_name": g["team_name"].iloc[0],
            "window_matches": cutoff,
            "matches_in_window": len(sel),
        }
        agg = _core_aggregates(sub)
        rec.update(agg)
        rec["shots_per_match"] = agg["shots"] / len(sel)
        rec["xg_per_match"] = agg["total_xg"] / len(sel)
        rec["goals_minus_xg_per_match"] = (agg["goals"] - agg["total_xg"]) / len(sel)
        rows.append(rec)
    out = pd.DataFrame(rows)
    return add_profile_axes(out, axis_ref) if not out.empty else out


def window_exclusions(shots: pd.DataFrame, tm: pd.DataFrame, n_matches) -> tuple[int, str]:
    """Explain team-seasons dropped by the >=20 shot rule for a window."""
    notes = []
    for (team_id, season), g in tm.groupby(["team_id", "season_name"]):
        total = int(g["match_number"].max())
        cutoff = int(np.ceil(0.30 * total)) if n_matches is None else int(n_matches)
        early = g[g["match_number"] <= cutoff]
        rest = g[g["match_number"] > cutoff]
        early_shots = len(
            shots[shots["match_id"].isin(early["match_id"]) & (shots["team_id"] == team_id)]
        )
        rest_shots = len(
            shots[shots["match_id"].isin(rest["match_id"]) & (shots["team_id"] == team_id)]
        )
        reasons = []
        if early_shots < 20:
            reasons.append(f"early window {early_shots} shots")
        if rest_shots < 20:
            reasons.append(f"remainder window {rest_shots} shots")
        if reasons:
            notes.append(f"{g['team_name'].iloc[0]} {season} ({'; '.join(reasons)})")
    return len(notes), "; ".join(notes)


def metric_ci(y, pred, metric, n_boot: int = 2000) -> str:
    rng = np.random.default_rng(RANDOM_STATE)
    vals = []
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], pred[idx]
        if metric == "r2":
            vals.append(r2_score(yy, pp))
        elif metric == "mae":
            vals.append(mean_absolute_error(yy, pp))
        elif metric == "spearman":
            vals.append(spearmanr(yy, pp).correlation if np.std(pp) > 0 else np.nan)
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return f"[{np.quantile(vals, .025):.3f}, {np.quantile(vals, .975):.3f}]" if len(vals) else ""


def stabilisation_curves(shots: pd.DataFrame, tm: pd.DataFrame, axis_ref: dict) -> pd.DataFrame:
    rows = []
    for m in STABILISATION_WINDOWS:
        early = early_window_features(shots, tm, m, axis_ref)
        rest = early_window_features(shots, tm, m, axis_ref, complement=True)
        n_excluded, exclusion_note = window_exclusions(shots, tm, m)
        if early.empty or rest.empty:
            continue
        merged = early.merge(rest, on=["team_id", "season_name"], suffixes=("_early", "_rest"))
        for f in STABILISATION_FEATURES:
            a, b = f"{f}_early", f"{f}_rest"
            ok = merged[[a, b]].dropna()
            if len(ok) < 10 or ok[a].std() == 0 or ok[b].std() == 0:
                continue
            rows.append(
                {
                    "Matches": m,
                    "Feature": f,
                    "Pearson r with remainder": round(ok[a].corr(ok[b]), 3),
                    "Spearman rho with remainder": round(spearmanr(ok[a], ok[b]).correlation, 3),
                    "n": len(ok),
                    "Excluded team-seasons": n_excluded,
                    "Exclusion note": exclusion_note,
                }
            )
    return pd.DataFrame(rows)


def fit_predict_model(model, X_tr, y_tr, X_te):
    fitted = copy.deepcopy(model)
    fitted.fit(X_tr, y_tr)
    return fitted.predict(X_te), fitted


def prediction_rows(early: pd.DataFrame, rest: pd.DataFrame, wname: str) -> tuple[list[dict], tuple]:
    data = early.merge(rest, on=["team_id", "season_name"], suffixes=("_early", "_rest"))
    feats = [f"{f}_early" for f in EARLY_FEATURES if f"{f}_early" in data.columns]
    X = data[feats].to_numpy(dtype=float)
    tr = data["season_name"].isin(TRAIN_SEASONS).to_numpy()
    te = data["season_name"].isin(TEST_SEASONS).to_numpy()

    models = {
        "Ridge regression": Pipeline(
            [("scale", StandardScaler()), ("model", RidgeCV(alphas=np.logspace(-2, 3, 40)))]
        ),
        "Random forest sensitivity": RandomForestRegressor(
            n_estimators=600,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    rows = []
    shap_payload = None
    for target in TARGETS:
        y = data[f"{target}_rest"].to_numpy(dtype=float)
        naive = data[f"{target}_early"].to_numpy(dtype=float)
        candidates = [("Naive early-value baseline", naive[te], None)]
        for name, model in models.items():
            pred, fitted = fit_predict_model(model, X[tr], y[tr], X[te])
            candidates.append((name, pred, fitted))
            if wname == "First 10 matches" and target == "chance_creation_axis" and name.startswith("Random"):
                shap_payload = (X, y, feats)

        for name, pred, _ in candidates:
            yy = y[te]
            rho = spearmanr(yy, pred).correlation if np.std(pred) > 0 else np.nan
            rows.append(
                {
                    "Window": wname,
                    "Target": target.replace("_", " "),
                    "Model": name,
                    "Holdout R2 (2024/25 remainder)": round(r2_score(yy, pred), 3),
                    "R2 95% bootstrap CI": metric_ci(yy, pred, "r2"),
                    "Holdout MAE": round(mean_absolute_error(yy, pred), 3),
                    "MAE 95% bootstrap CI": metric_ci(yy, pred, "mae"),
                    "Holdout Spearman rho": round(rho, 3) if np.isfinite(rho) else np.nan,
                    "Spearman 95% bootstrap CI": metric_ci(yy, pred, "spearman"),
                    "N train": int(tr.sum()),
                    "N test": int(te.sum()),
                }
            )
    return rows, shap_payload


def main() -> None:
    header("STEP 8  Early-season prediction")
    plotstyle.apply()

    shots = use_temporal_holdout_shot_quality(load_frame("shots_scored"))
    tm = load_frame("team_match_profiles")
    ts = load_frame("team_season_clustered")
    axis_ref = fit_axis_reference(ts)

    stab = stabilisation_curves(shots, tm, axis_ref)
    save_table(stab, "tableS2_stabilisation_curves")
    piv = stab.pivot(index="Matches", columns="Feature", values="Pearson r with remainder")
    earliest = piv.apply(
        lambda col: col.index[col.ge(0.70).argmax()] if col.ge(0.70).any() else np.nan
    ).sort_values()
    print("  matches needed to reach Pearson r >= 0.70 with remainder-season performance:")
    print(earliest.dropna().to_string() if earliest.notna().any() else "  none")

    reg_rows = []
    shap_payload = None
    for wname, wsize in WINDOWS.items():
        early = early_window_features(shots, tm, wsize, axis_ref)
        rest = early_window_features(shots, tm, wsize, axis_ref, complement=True)
        rows, payload = prediction_rows(early, rest, wname)
        reg_rows.extend(rows)
        if payload is not None:
            shap_payload = payload
        print(f"\n  {wname}: {len(early)} early profiles, {len(rest)} remainder profiles")

    out = pd.DataFrame(reg_rows)
    save_table(out, "tableS3_early_season_regression")
    print("\n  early-to-remainder prediction:")
    print(
        out[out["Model"] != "Random forest sensitivity"][
            ["Window", "Target", "Model", "Holdout R2 (2024/25 remainder)",
             "Holdout MAE", "Holdout Spearman rho"]
        ].to_string(index=False)
    )

    make_stabilisation_figure(stab)
    if shap_payload:
        make_shap_figure(*shap_payload)


def make_stabilisation_figure(stab: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    piv = stab.pivot(index="Matches", columns="Feature", values="Pearson r with remainder")
    show = [
        "chance_creation_axis",
        "finishing_axis",
        "xg_per_match",
        "goals_minus_xg_per_match",
        "shots_per_match",
        "on_target_rate",
    ]
    fig, ax = plt.subplots(figsize=(plotstyle.W_DOUBLE * 0.70, 3.35))
    for i, f in enumerate([f for f in show if f in piv.columns]):
        ax.plot(
            piv.index,
            piv[f],
            marker=plotstyle.MARKERS[i % len(plotstyle.MARKERS)],
            color=plotstyle.CATEGORICAL[i % len(plotstyle.CATEGORICAL)],
            label=f.replace("_", " "),
            markersize=3.5,
            markeredgecolor="white",
            markeredgewidth=0.4,
        )
    ax.axhline(0.70, color=plotstyle.INK_MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.text(piv.index.max(), 0.71, "r = 0.70", ha="right", fontsize=6.3,
            color=plotstyle.INK_MUTED)
    ax.set_xlabel("Matches elapsed")
    ax.set_ylabel("Pearson r with remainder of season")
    ax.set_title("Stabilisation against non-overlapping remainder-season performance", loc="left")
    ax.set_ylim(-0.15, 1.02)
    ax.legend(loc="lower right", fontsize=6.3)
    plotstyle.save(fig, "figureS1_stabilisation")


def make_shap_figure(X, y, feats) -> None:
    import matplotlib.pyplot as plt
    import shap

    model = RandomForestRegressor(
        n_estimators=600,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(X, y)
    sv = shap.TreeExplainer(model).shap_values(X)
    imp = np.abs(np.asarray(sv)).mean(axis=0)
    order = np.argsort(imp)[::-1][:12][::-1]

    fig, ax = plt.subplots(figsize=(plotstyle.W_SINGLE * 1.35, 3.0))
    ax.barh(
        range(len(order)),
        imp[order],
        color=plotstyle.CATEGORICAL[0],
        height=0.68,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feats[i].replace("_early", "").replace("_", " ") for i in order])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(
        "Early predictors of remainder-season chance creation\n(first 10 matches)",
        loc="left",
        fontsize=8,
    )
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    plotstyle.save(fig, "figureS2_early_season_shap")


if __name__ == "__main__":
    main()
