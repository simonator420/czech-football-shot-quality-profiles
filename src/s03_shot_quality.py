"""Step 3 - Shot-quality (expected-goals) model development (proposal 5.3).

Validation design
-----------------
The proposal specifies a chronological train/validation/test split. With the
three-season sample the split is:

    inner tuning   train 2022/23        -> validate 2023/24
    final fit      train 2022/23 + 2023/24
    evaluation     test  2024/25        (touched exactly once)

Because downstream profile, clustering and load analyses need a shot-quality
value for *every* shot, a second set of predictions is generated out-of-fold
using 5-fold cross-validation grouped by match, so no shot is ever scored by a
model that saw the same match. Headline performance in Table 3 always comes
from the strict chronological hold-out, never from the out-of-fold values.

Produces
    tables/table3_shot_quality_models.csv
    tables/table3b_shot_quality_by_season.csv
    figures/figure2_calibration.pdf
    figures/figure3_shap_importance.pdf
    data/shots_scored.parquet
    models/shot_quality_model.joblib
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import plotstyle
from common import (
    MODELS,
    RANDOM_STATE,
    TEST_SEASONS,
    TRAIN_SEASONS,
    header,
    load_frame,
    save_frame,
    save_table,
)
from modelling import (
    BINARY,
    CATEGORICAL,
    NUMERIC,
    TARGET,
    bootstrap_ci,
    calibration_slope_intercept,
    evaluate,
    logistic_pipeline,
    make_preprocessor,
)


# --------------------------------------------------------------------------
# Candidate models
# --------------------------------------------------------------------------


def build_candidates(numeric, binary, categorical):
    """Candidate model zoo and tuning grids.

    Each entry is a ``(tag, factory)`` pair rather than a fitted estimator:
    CatBoost refuses parameter changes once fitted, and every refit must start
    from a clean estimator anyway so that no state leaks between the tuning,
    final-fit and out-of-fold stages.
    """
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from sklearn.pipeline import Pipeline

    def rf(**kw):
        return Pipeline(
            [
                ("pre", make_preprocessor(numeric, binary, categorical, scale=False)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=600,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                        **kw,
                    ),
                ),
            ]
        )

    def xgbm(**kw):
        params = dict(
            n_estimators=2000,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            early_stopping_rounds=100,
        )
        params.update(kw)
        return Pipeline(
            [
                ("pre", make_preprocessor(numeric, binary, categorical, scale=False)),
                ("clf", xgb.XGBClassifier(**params)),
            ]
        )

    def cat(**kw):
        params = dict(
            iterations=3000,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=RANDOM_STATE,
            verbose=0,
            early_stopping_rounds=100,
        )
        params.update(kw)
        return Pipeline(
            [
                ("pre", make_preprocessor(numeric, binary, categorical, scale=False)),
                ("clf", CatBoostClassifier(**params)),
            ]
        )

    return {
        "Regularised logistic regression": [
            (f"C={c}", (lambda c=c: logistic_pipeline(numeric, binary, categorical, C=c)))
            for c in (0.1, 1.0, 10.0)
        ],
        "Random forest": [
            (f"depth={d},leaf={l}", (lambda d=d, l=l: rf(max_depth=d, min_samples_leaf=l)))
            for d in (8, 12, None)
            for l in (20, 50)
        ],
        "XGBoost": [
            (f"depth={d},mcw={m}", (lambda d=d, m=m: xgbm(max_depth=d, min_child_weight=m)))
            for d in (3, 4, 5)
            for m in (5, 20)
        ],
        "CatBoost": [
            (f"depth={d},l2={l}", (lambda d=d, l=l: cat(depth=d, l2_leaf_reg=l)))
            for d in (4, 6)
            for l in (3, 10)
        ],
    }


def fit_with_validation(pipe, X_tr, y_tr, X_va, y_va):
    """Fit a pipeline, wiring early stopping through to the boosting stage."""
    name = type(pipe.named_steps["clf"]).__name__
    if name in ("XGBClassifier", "CatBoostClassifier"):
        pre = pipe.named_steps["pre"]
        Xt = pre.fit_transform(X_tr)
        Xv = pre.transform(X_va)
        clf = pipe.named_steps["clf"]
        if name == "XGBClassifier":
            clf.fit(Xt, y_tr, eval_set=[(Xv, y_va)], verbose=False)
        else:
            clf.fit(Xt, y_tr, eval_set=(Xv, y_va))
        return pipe
    pipe.fit(X_tr, y_tr)
    return pipe


def best_iteration(pipe) -> int | None:
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "best_iteration"):
        return clf.best_iteration
    if hasattr(clf, "get_best_iteration"):
        return clf.get_best_iteration()
    return None


def refit_full(factory, X, y, n_iter):
    """Refit a tuned configuration on the full training window.

    Early stopping needs a validation set that does not exist once training
    spans the whole window, so the boosting budget is frozen at the iteration
    count chosen during inner validation.
    """
    pipe = factory()
    clf = pipe.named_steps["clf"]
    name = type(clf).__name__
    if name == "XGBClassifier" and n_iter:
        clf.set_params(n_estimators=max(int(n_iter) + 1, 50), early_stopping_rounds=None)
    elif name == "CatBoostClassifier" and n_iter:
        clf.set_params(iterations=max(int(n_iter) + 1, 50), early_stopping_rounds=None)
    pipe.fit(X, y)
    return pipe


# --------------------------------------------------------------------------


def main() -> None:
    header("STEP 3  Shot-quality model development and validation")
    plotstyle.apply()

    df = load_frame("shots_features")
    df = df.sort_values("start_datetime_utc").reset_index(drop=True)

    feat_cols = NUMERIC + BINARY + CATEGORICAL
    df[NUMERIC] = df[NUMERIC].astype(float)
    y_all = df[TARGET].to_numpy()

    is_train = df["season_name"].isin(TRAIN_SEASONS).to_numpy()
    is_test = df["season_name"].isin(TEST_SEASONS).to_numpy()
    inner_tr = (df["season_name"] == TRAIN_SEASONS[0]).to_numpy()
    inner_va = (df["season_name"] == TRAIN_SEASONS[1]).to_numpy()

    X = df[feat_cols]
    base_rate = float(y_all[is_train].mean())
    print(
        f"  train {is_train.sum():,} shots ({TRAIN_SEASONS}), "
        f"test {is_test.sum():,} shots ({TEST_SEASONS}); "
        f"training base rate {base_rate:.4f}"
    )

    results = []

    # ---- Model 0: overall goal-rate baseline -----------------------------
    p0 = np.full(is_test.sum(), base_rate)
    results.append(evaluate(y_all[is_test], p0, base_rate, "Model 0: overall goal-rate baseline"))

    # ---- Model 1: spatial logistic regression ----------------------------
    m1 = logistic_pipeline(["distance_proxy", "angle_proxy"], [], [])
    m1.fit(X[is_train], y_all[is_train])
    results.append(
        evaluate(
            y_all[is_test],
            m1.predict_proba(X[is_test])[:, 1],
            base_rate,
            "Model 1: spatial logistic regression",
        )
    )

    # ---- Model 2: contextual logistic regression -------------------------
    m2 = logistic_pipeline(
        ["distance_proxy", "angle_proxy", "minute"], ["is_home_shot"], ["body_part", "situation"]
    )
    m2.fit(X[is_train], y_all[is_train])
    results.append(
        evaluate(
            y_all[is_test],
            m2.predict_proba(X[is_test])[:, 1],
            base_rate,
            "Model 2: contextual logistic regression",
        )
    )

    # ---- Machine-learning candidates -------------------------------------
    candidates = build_candidates(NUMERIC, BINARY, CATEGORICAL)
    tuning_log = []
    family_best = {}

    for family, specs in candidates.items():
        print(f"\n  tuning {family} ({len(specs)} configurations)")
        rows = []
        for tag, factory in specs:
            pipe = fit_with_validation(
                factory(), X[inner_tr], y_all[inner_tr], X[inner_va], y_all[inner_va]
            )
            p_va = pipe.predict_proba(X[inner_va])[:, 1]
            from sklearn.metrics import log_loss

            ll = log_loss(y_all[inner_va], p_va)
            auc = roc_auc_score(y_all[inner_va], p_va)
            rows.append(
                {
                    "family": family,
                    "config": tag,
                    "inner_logloss": ll,
                    "inner_auc": auc,
                    "best_iter": best_iteration(pipe),
                }
            )
            print(f"    {tag:<20} inner log loss {ll:.5f}   AUC {auc:.4f}")
        rows.sort(key=lambda r: r["inner_logloss"])
        tuning_log.extend(rows)
        family_best[family] = rows[0]

        # Refit the winning configuration on the full training window.
        winner_tag = rows[0]["config"]
        winner_pipe = refit_full(
            dict(specs)[winner_tag], X[is_train], y_all[is_train], rows[0]["best_iter"]
        )
        p_test = winner_pipe.predict_proba(X[is_test])[:, 1]
        results.append(
            evaluate(y_all[is_test], p_test, base_rate, f"{family} ({winner_tag})")
        )
        family_best[family]["fitted"] = winner_pipe
        family_best[family]["p_test"] = p_test

    # ---- Select the main model on inner-validation log loss --------------
    ml_families = [f for f in family_best if f != "Model 0"]
    main_family = min(ml_families, key=lambda f: family_best[f]["inner_logloss"])
    main = family_best[main_family]
    main_pipe = main["fitted"]
    p_test = main["p_test"]
    print(
        f"\n  selected main model: {main_family} ({main['config']}) "
        f"on inner-validation log loss {main['inner_logloss']:.5f}"
    )

    table3 = pd.DataFrame(results)
    for c in ("AUC", "PR-AUC", "Log loss", "Brier score", "Brier skill score",
              "Calibration slope", "Calibration intercept", "ECE"):
        table3[c] = table3[c].astype(float).round(4)
    table3["Main model"] = [
        "yes" if m.startswith(main_family) else "" for m in table3["Model"]
    ]

    # Bootstrap CIs for the main model's headline metrics.
    y_te = y_all[is_test]
    auc_lo, auc_hi = bootstrap_ci(y_te, p_test, roc_auc_score, seed=RANDOM_STATE)
    from sklearn.metrics import brier_score_loss

    br_lo, br_hi = bootstrap_ci(y_te, p_test, brier_score_loss, seed=RANDOM_STATE)
    print(f"  main model AUC {roc_auc_score(y_te, p_test):.4f} [{auc_lo:.4f}, {auc_hi:.4f}]")
    print(f"  main model Brier {brier_score_loss(y_te, p_test):.5f} [{br_lo:.5f}, {br_hi:.5f}]")

    save_table(table3, "table3_shot_quality_models")

    # ---- Performance excluding penalties and by season -------------------
    extra = []
    no_pen = (df.loc[is_test, "is_penalty"] == 0).to_numpy()
    extra.append(
        evaluate(y_te[no_pen], p_test[no_pen], base_rate, "Test season, penalties excluded")
    )
    extra.append(evaluate(y_te, p_test, base_rate, "Test season, all shots"))
    open_play = (df.loc[is_test, "is_open_play"] == 1).to_numpy()
    extra.append(
        evaluate(y_te[open_play], p_test[open_play], base_rate, "Test season, open play only")
    )
    save_table(pd.DataFrame(extra).round(4), "table3b_shot_quality_by_subset")

    # ---- Out-of-fold shot quality for every shot -------------------------
    print("\n  generating out-of-fold shot quality (5-fold, grouped by match)")
    oof = np.full(len(df), np.nan)
    gkf = GroupKFold(n_splits=5)
    for k, (tr, va) in enumerate(gkf.split(X, y_all, groups=df["match_id"])):
        # An inner chronological slice of the fold's training data drives early
        # stopping, so no fold ever tunes on its own validation shots.
        n_inner = int(0.85 * len(tr))
        pipe = fit_with_validation(
            dict(candidates[main_family])[main["config"]](),
            X.iloc[tr[:n_inner]],
            y_all[tr[:n_inner]],
            X.iloc[tr[n_inner:]],
            y_all[tr[n_inner:]],
        )
        oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
        print(f"    fold {k + 1}/5 done")

    df["shot_quality"] = oof
    df["shot_quality_holdout"] = np.nan
    df.loc[is_test, "shot_quality_holdout"] = p_test

    oof_slope, oof_int = calibration_slope_intercept(y_all, oof)
    print(
        f"  out-of-fold: AUC {roc_auc_score(y_all, oof):.4f}, "
        f"calibration slope {oof_slope:.3f}, intercept {oof_int:.3f}, "
        f"mean SQ {oof.mean():.4f} vs observed goal rate {y_all.mean():.4f}"
    )

    save_frame(df, "shots_scored")

    import joblib

    joblib.dump(main_pipe, MODELS / "shot_quality_model.joblib")
    (MODELS / "tuning_log.json").write_text(
        json.dumps(
            [{k: v for k, v in r.items() if k not in ("fitted", "p_test")} for r in tuning_log],
            indent=2,
            default=str,
        )
    )

    # ---- Figure 2: calibration -------------------------------------------
    make_calibration_figure(y_te, p_test, results, df, is_test)

    # ---- Figure 3: SHAP ---------------------------------------------------
    make_shap_figure(main_pipe, X[is_test], main_family)


def make_calibration_figure(y_te, p_test, results, df, is_test) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(plotstyle.W_DOUBLE, 3.0))

    ax = axes[0]
    n_bins = 10
    order = np.argsort(p_test)
    ys, ps = y_te[order], p_test[order]
    bins = np.array_split(np.arange(len(ps)), n_bins)
    obs = np.array([ys[b].mean() for b in bins])
    pred = np.array([ps[b].mean() for b in bins])
    # Wilson interval for the observed proportion in each decile.
    ns = np.array([len(b) for b in bins])
    z = 1.96
    denom = 1 + z**2 / ns
    centre = (obs + z**2 / (2 * ns)) / denom
    half = z * np.sqrt(obs * (1 - obs) / ns + z**2 / (4 * ns**2)) / denom
    lo, hi = centre - half, centre + half

    lim = max(pred.max(), obs.max()) * 1.08
    ax.plot([0, lim], [0, lim], color=plotstyle.INK_MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.errorbar(
        pred, obs, yerr=[obs - lo, hi - obs],
        fmt="o", color=plotstyle.CATEGORICAL[0], markersize=4.5,
        ecolor=plotstyle.CATEGORICAL[0], elinewidth=1.0, capsize=2, zorder=3,
        markeredgecolor="white", markeredgewidth=0.6,
    )
    ax.set_xlabel("Predicted shot quality (decile mean)")
    ax.set_ylabel("Observed goal proportion")
    ax.set_title("a  Calibration, held-out season 2024/25", loc="left")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.text(
        0.97, 0.06, "dashed line = perfect calibration",
        transform=ax.transAxes, ha="right", color=plotstyle.INK_MUTED, fontsize=7,
    )

    ax = axes[1]
    ax.hist(
        p_test, bins=40, color=plotstyle.CATEGORICAL[0], alpha=0.85,
        edgecolor="white", linewidth=0.4,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Predicted shot quality")
    ax.set_ylabel("Shots (log scale)")
    ax.set_title("b  Distribution of predicted shot quality", loc="left")
    ax.grid(axis="y")

    plotstyle.save(fig, "figure2_calibration")


def make_shap_figure(pipe, X_test, family: str) -> None:
    import matplotlib.pyplot as plt
    import shap

    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    Xt = pre.transform(X_test)
    names = list(pre.get_feature_names_out())
    names = [n.split("__", 1)[-1] for n in names]

    try:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt)
    except Exception as exc:  # linear main model
        print(f"    TreeExplainer unavailable ({exc}); falling back to LinearExplainer")
        explainer = shap.LinearExplainer(clf, Xt)
        sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, -1]

    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(imp)[::-1][:15][::-1]

    fig, ax = plt.subplots(figsize=(plotstyle.W_SINGLE * 1.35, 3.4))
    ax.barh(
        range(len(order)), imp[order],
        color=plotstyle.CATEGORICAL[0], height=0.68,
        edgecolor="white", linewidth=0.5,
    )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order])
    ax.set_xlabel("Mean |SHAP value| (log-odds of a goal)")
    ax.set_title(f"Feature contributions, {family}", loc="left")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(imp[order]):
        ax.text(v, i, f"  {v:.3f}", va="center", fontsize=6.8, color=plotstyle.INK_MUTED)
    ax.set_xlim(0, imp[order].max() * 1.22)
    plotstyle.save(fig, "figure3_shap_importance")

    np.save(MODELS / "shap_values_test.npy", sv)
    pd.DataFrame({"feature": names, "mean_abs_shap": imp}).sort_values(
        "mean_abs_shap", ascending=False
    ).to_csv(MODELS / "shap_importance.csv", index=False)


if __name__ == "__main__":
    main()
