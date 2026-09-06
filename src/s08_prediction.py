"""Step 8 - Early-season profile prediction (proposal 5.9).

Research question 3 asks whether early-season attacking behaviour predicts the
full-season profile, and which event-level features stabilise earliest.

Three analyses answer it, in increasing order of statistical power:

  1. *Stabilisation curves* - for each profile feature, the split-half
     correlation between matches 1..m and matches m+1..end, as a function of m.
     The two windows are disjoint, so the curve is not inflated by part-whole
     overlap. Computed on all 48 team-seasons and needs no model.
  2. *Continuous profile prediction* - regression of the full-season
     chance-creation and finishing axes on early-season features. With 48
     team-seasons this is far better powered than classification and is treated
     as the primary predictive result.
  3. *Cluster-membership classification* - the analysis as literally specified
     in the proposal. With 32 training and 16 test team-seasons it is
     underpowered by construction, so it is reported against a majority-class
     baseline with bootstrap intervals and interpreted with corresponding
     caution.

Produces
    tables/tableS2_stabilisation_curves.csv
    tables/tableS3_early_season_regression.csv
    tables/tableS4_early_season_classification.csv
    figures/figureS1_stabilisation.pdf
    figures/figureS2_early_season_shap.pdf
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import plotstyle
from common import RANDOM_STATE, TEST_SEASONS, TRAIN_SEASONS, header, load_frame, save_table
from s04_profiles import PROFILE_FEATURES, _core_aggregates

warnings.filterwarnings("ignore")

WINDOWS = {"First 5 matches": 5, "First 10 matches": 10, "First 30% of season": None}
AXES = ["chance_creation_axis", "finishing_axis"]

#: Early-season descriptors offered to the models.
EARLY_FEATURES = [
    "mean_shot_quality",
    "shots_per_match",
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


def early_window_features(
    shots: pd.DataFrame, tm: pd.DataFrame, n_matches, complement: bool = False
) -> pd.DataFrame:
    """Aggregate a team-season's first `n_matches` matches into a profile.

    With ``complement=True`` the *remaining* matches of the season are
    aggregated instead. That is what the stabilisation curves need: correlating
    the first m matches against the full season would share those same matches
    on both sides and inflate the correlation by part-whole overlap.
    """
    rows = []
    for (team_id, season), g in tm.groupby(["team_id", "season_name"]):
        total = g["match_number"].max()
        cutoff = int(np.ceil(0.30 * total)) if n_matches is None else n_matches
        sel = g[g["match_number"] > cutoff] if complement else g[g["match_number"] <= cutoff]
        if sel.empty:
            continue
        sub = shots[
            shots["match_id"].isin(sel["match_id"]) & (shots["team_id"] == team_id)
        ]
        if len(sub) < 20:
            continue
        rec = {"team_id": team_id, "season_name": season, "window_matches": cutoff}
        agg = _core_aggregates(sub)
        rec.update(agg)
        rec["shots_per_match"] = agg["shots"] / len(sel)
        rec["xg_per_match"] = agg["total_xg"] / len(sel)
        rec["goals_minus_xg_per_match"] = (agg["goals"] - agg["total_xg"]) / len(sel)
        rows.append(rec)
    return pd.DataFrame(rows)


def stabilisation_curves(shots: pd.DataFrame, tm: pd.DataFrame, ts: pd.DataFrame) -> pd.DataFrame:
    """How early each feature settles, measured on disjoint halves of a season.

    For every cut-point m the feature is computed twice: once from matches
    1..m and once from matches m+1..end. The two windows share no matches, so
    the correlation between them is a clean split-half estimate of how quickly
    a team's attacking profile becomes recognisable, uncontaminated by the
    part-whole overlap that comparing against the full season would introduce.
    """
    # A season runs 30-35 matches, so beyond about m = 20 the complement window
    # is too short to estimate a stable profile and the correlation falls for
    # want of data rather than because the profile has stopped settling. The
    # curve is therefore cut where the complement drops below ten matches.
    min_complement = 10
    max_m = int(tm.groupby(["team_id", "season_name"])["match_number"].max().min()) - min_complement

    rows = []
    for m in range(3, max(max_m + 1, 4), 2):
        early = early_window_features(shots, tm, m)
        rest = early_window_features(shots, tm, m, complement=True)
        if early.empty or rest.empty:
            continue
        merged = early.merge(
            rest, on=["team_id", "season_name"], suffixes=("_early", "_rest")
        )
        if len(merged) < 10:
            continue
        for f in EARLY_FEATURES:
            a, b = f"{f}_early", f"{f}_rest"
            if a not in merged or b not in merged:
                continue
            ok = merged[[a, b]].dropna()
            if len(ok) < 10 or ok[a].std() == 0 or ok[b].std() == 0:
                continue
            rows.append({"Matches": m, "Feature": f,
                         "r with rest of season": round(ok[a].corr(ok[b]), 3),
                         "n": len(ok)})
    return pd.DataFrame(rows)


def main() -> None:
    header("STEP 8  Early-season profile prediction")
    plotstyle.apply()

    shots = load_frame("shots_scored")
    tm = load_frame("team_match_profiles")
    ts = load_frame("team_season_clustered")

    # ---- 1. stabilisation curves -----------------------------------------
    stab = stabilisation_curves(shots, tm, ts)
    save_table(stab, "tableS2_stabilisation_curves")
    piv = stab.pivot(index="Matches", columns="Feature", values="r with rest of season")
    earliest = (
        piv.apply(lambda col: col.index[col.ge(0.70).argmax()] if col.ge(0.70).any() else np.nan)
        .sort_values()
    )
    print("  matches needed for a feature to reach r >= 0.70 with the rest of the season:")
    print(earliest.dropna().head(12).to_string())
    never = earliest[earliest.isna()].index.tolist()
    if never:
        print(f"  never reaching r >= 0.70 within {int(piv.index.max())} matches "
              f"({len(never)} of {piv.shape[1]}): {', '.join(never)}")

    # ---- 2 & 3. prediction from each early window ------------------------
    reg_rows, clf_rows = [], []
    shap_payload = None

    for wname, wsize in WINDOWS.items():
        early = early_window_features(shots, tm, wsize)
        data = early.merge(
            ts[["team_id", "season_name", "cluster"] + AXES],
            on=["team_id", "season_name"],
        )
        feats = [f for f in EARLY_FEATURES if f in data.columns]
        X = data[feats].to_numpy(dtype=float)
        tr = data["season_name"].isin(TRAIN_SEASONS).to_numpy()
        te = data["season_name"].isin(TEST_SEASONS).to_numpy()
        print(f"\n  {wname}: {len(data)} team-seasons "
              f"({tr.sum()} train / {te.sum()} test), {len(feats)} features")

        # --- continuous ---
        for axis in AXES:
            y = data[axis].to_numpy(dtype=float)
            models = {
                "Ridge regression": Pipeline([("s", StandardScaler()),
                                              ("m", RidgeCV(alphas=np.logspace(-2, 3, 30)))]),
                "Random forest": RandomForestRegressor(
                    n_estimators=500, min_samples_leaf=3,
                    random_state=RANDOM_STATE, n_jobs=-1),
                "Mean baseline": DummyRegressor(strategy="mean"),
            }
            for mname, model in models.items():
                model.fit(X[tr], y[tr])
                pred = model.predict(X[te])
                # Repeated CV over all team-seasons as a supplementary estimate.
                cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
                cv_scores = []
                for i_tr, i_te in cv.split(X):
                    import copy
                    m2 = copy.deepcopy(model)
                    m2.fit(X[i_tr], y[i_tr])
                    cv_scores.append(r2_score(y[i_te], m2.predict(X[i_te])))
                reg_rows.append({
                    "Window": wname, "Target": axis.replace("_", " "),
                    "Model": mname,
                    "Holdout R2 (2024/25)": round(r2_score(y[te], pred), 3),
                    "Holdout MAE": round(mean_absolute_error(y[te], pred), 3),
                    "Holdout r": round(np.corrcoef(y[te], pred)[0, 1], 3)
                    if np.std(pred) > 0 else np.nan,
                    "Repeated-CV R2 (mean)": round(float(np.mean(cv_scores)), 3),
                    "Repeated-CV R2 (SD)": round(float(np.std(cv_scores)), 3),
                    "N train": int(tr.sum()), "N test": int(te.sum()),
                })

        # --- classification ---
        yc = data["cluster"].to_numpy()
        clf_models = {
            "Multinomial logistic": Pipeline([
                ("s", StandardScaler()),
                ("m", LogisticRegression(max_iter=5000, C=0.5)),
            ]),
            "Random forest": RandomForestClassifier(
                n_estimators=500, min_samples_leaf=2,
                random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
            "Majority-class baseline": DummyClassifier(strategy="most_frequent"),
        }
        for mname, model in clf_models.items():
            model.fit(X[tr], yc[tr])
            pred = model.predict(X[te])
            acc = (pred == yc[te]).mean()
            # Bootstrap interval on the 16-team test set.
            rng = np.random.default_rng(RANDOM_STATE)
            boots = [
                (pred[i] == yc[te][i]).mean()
                for i in (rng.integers(0, te.sum(), te.sum()) for _ in range(2000))
            ]
            skf = RepeatedStratifiedKFold(n_splits=4, n_repeats=10, random_state=RANDOM_STATE)
            cv_acc = []
            for i_tr, i_te in skf.split(X, yc):
                import copy
                m2 = copy.deepcopy(model)
                m2.fit(X[i_tr], yc[i_tr])
                cv_acc.append((m2.predict(X[i_te]) == yc[i_te]).mean())
            clf_rows.append({
                "Window": wname, "Model": mname,
                "Holdout accuracy": round(acc, 3),
                "95% bootstrap CI": f"[{np.quantile(boots, .025):.2f}, {np.quantile(boots, .975):.2f}]",
                "Macro F1": round(f1_score(yc[te], pred, average="macro", zero_division=0), 3),
                "Balanced accuracy": round(balanced_accuracy_score(yc[te], pred), 3),
                "Repeated-CV accuracy": round(float(np.mean(cv_acc)), 3),
                "N train": int(tr.sum()), "N test": int(te.sum()),
            })

        if wname == "First 10 matches":
            shap_payload = (X, data[AXES[0]].to_numpy(dtype=float), feats)

    t7b = pd.DataFrame(reg_rows)
    t7c = pd.DataFrame(clf_rows)
    print("\n  continuous prediction of the full-season profile axes:")
    print(t7b[t7b["Model"] != "Mean baseline"][
        ["Window", "Target", "Model", "Holdout R2 (2024/25)", "Holdout r",
         "Repeated-CV R2 (mean)"]].to_string(index=False))
    print("\n  cluster-membership classification:")
    print(t7c[["Window", "Model", "Holdout accuracy", "95% bootstrap CI",
               "Macro F1", "Repeated-CV accuracy"]].to_string(index=False))
    save_table(t7b, "tableS3_early_season_regression")
    save_table(t7c, "tableS4_early_season_classification")

    make_stabilisation_figure(stab)
    if shap_payload:
        make_shap_figure(*shap_payload)


def make_stabilisation_figure(stab: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    piv = stab.pivot(index="Matches", columns="Feature", values="r with rest of season")
    final = piv.iloc[-1].sort_values(ascending=False)
    top = list(final.head(6).index)

    fig, ax = plt.subplots(figsize=(plotstyle.W_DOUBLE * 0.62, 3.2))
    for i, f in enumerate(top):
        ax.plot(piv.index, piv[f], marker=plotstyle.MARKERS[i % len(plotstyle.MARKERS)],
                color=plotstyle.CATEGORICAL[i % len(plotstyle.CATEGORICAL)],
                label=f.replace("_", " "), markersize=3.5,
                markeredgecolor="white", markeredgewidth=0.4)
    others = [c for c in piv.columns if c not in top]
    if others:
        ax.plot(piv.index, piv[others].mean(axis=1), color=plotstyle.INK_MUTED,
                lw=1.1, ls=(0, (4, 3)), label="Other features (mean)")
    ax.axhline(0.70, color=plotstyle.INK_MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.text(piv.index.max(), 0.71, "r = 0.70", ha="right", fontsize=6.3,
            color=plotstyle.INK_MUTED)
    ax.set_xlabel("Matches elapsed in the season")
    ax.set_ylabel("Split-half correlation with the rest of the season")
    ax.set_title("Stabilisation of attacking-profile features\n(disjoint split-half within season)", loc="left", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=6.3, ncol=1)
    plotstyle.save(fig, "figureS1_stabilisation")


def make_shap_figure(X, y, feats) -> None:
    import matplotlib.pyplot as plt
    import shap
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                                  random_state=RANDOM_STATE, n_jobs=-1).fit(X, y)
    sv = shap.TreeExplainer(model).shap_values(X)
    imp = np.abs(np.asarray(sv)).mean(axis=0)
    order = np.argsort(imp)[::-1][:12][::-1]

    fig, ax = plt.subplots(figsize=(plotstyle.W_SINGLE * 1.35, 3.0))
    ax.barh(range(len(order)), imp[order], color=plotstyle.CATEGORICAL[0],
            height=0.68, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feats[i].replace("_", " ") for i in order])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Early-season predictors of the full-season chance-creation axis\n"
                 "(first 10 matches)", loc="left", fontsize=8)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    plotstyle.save(fig, "figureS2_early_season_shap")


if __name__ == "__main__":
    main()
