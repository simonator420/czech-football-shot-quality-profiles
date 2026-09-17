"""Step 9 - Sensitivity and robustness analyses (proposal 5.10).

Every check listed in the proposal is run, plus two added after the data audit:
a shot-quality/profile-construction workflow re-run without the taxonomy-
affected `situation` variable, and a comparison of pooled against within-
season standardisation for the profile vectors.

Produces
    tables/table7_robustness.csv
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, roc_auc_score, silhouette_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from common import (
    MODELS,
    PLAYER_SHOT_THRESHOLD_SENSITIVITY,
    RANDOM_STATE,
    SEASONS,
    TEST_SEASONS,
    TRAIN_SEASONS,
    header,
    load_frame,
    save_table,
)
from modelling import (
    BINARY,
    CATEGORICAL,
    NUMERIC,
    NUMERIC_ALT,
    TARGET,
    calibration_slope_intercept,
    evaluate,
)
from s03_shot_quality import build_candidates, fit_with_validation, refit_full
from s04_profiles import (
    PROFILE_FEATURES,
    SHOT_QUALITY_FEATURES,
    SHOT_SELECTION_FEATURES,
    SPATIAL_ONLY_FEATURES,
    LOAD_RESPONSE_FEATURES,
    build_team_match,
    build_team_season,
    build_player_season,
)
from s05_clustering import bootstrap_stability, standardise_within_season

warnings.filterwarnings("ignore")

ROWS: list[dict] = []
PRIMARY_CLUSTER_K = 2
ROBUSTNESS_CLUSTER_BOOT = 400

SITUATION_DERIVED_PROFILE_FEATURES = {
    "assisted_share",
    "set_piece_share",
    "fast_break_share",
}
NON_SITUATION_PROFILE_FEATURES = [
    f for f in PROFILE_FEATURES if f not in SITUATION_DERIVED_PROFILE_FEATURES
]


def record(analysis, variation, metric, value, reference=None, note=""):
    ROWS.append(
        {
            "Analysis": analysis,
            "Variation": variation,
            "Metric": metric,
            "Value": round(value, 4) if isinstance(value, (int, float, np.floating)) else value,
            "Primary-analysis value": (
                round(reference, 4)
                if isinstance(reference, (int, float, np.floating))
                else reference
            ),
            "Note": note,
        }
    )


def best_config() -> tuple[str, str, int | None]:
    log = json.loads((MODELS / "tuning_log.json").read_text())
    best = min(log, key=lambda r: r["inner_logloss"])
    return best["family"], best["config"], best.get("best_iter")


# --------------------------------------------------------------------------
# A. Shot-quality model
# --------------------------------------------------------------------------


def fit_shot_quality(df, numeric, binary, categorical, train_mask, test_mask):
    family, config, _ = best_config()
    cands = build_candidates(numeric, binary, categorical)
    factory = dict(cands[family])[config]
    cols = list(numeric) + list(binary) + list(categorical)
    X, y = df[cols], df[TARGET].to_numpy()

    inner = df["season_name"] == TRAIN_SEASONS[0]
    inner_va = df["season_name"] == TRAIN_SEASONS[1]
    inner = (inner & train_mask).to_numpy()
    inner_va = (inner_va & train_mask).to_numpy()
    if inner.sum() < 200 or inner_va.sum() < 200:
        pipe = factory()
        pipe.fit(X[train_mask], y[train_mask])
    else:
        tuned = fit_with_validation(factory(), X[inner], y[inner], X[inner_va], y[inner_va])
        n_iter = getattr(tuned.named_steps["clf"], "best_iteration", None)
        if n_iter is None and hasattr(tuned.named_steps["clf"], "get_best_iteration"):
            n_iter = tuned.named_steps["clf"].get_best_iteration()
        pipe = refit_full(factory, X[train_mask], y[train_mask], n_iter)
    return pipe.predict_proba(X[test_mask])[:, 1], y[test_mask]


def retained_profile_scores(data: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Match step 5 by retaining enough PCs to explain at least 80% variance."""
    Z = standardise_within_season(data, features)
    pca_full = PCA().fit(Z)
    n_comp = int(np.searchsorted(np.cumsum(pca_full.explained_variance_ratio_), 0.80) + 1)
    n_comp = min(n_comp, Z.shape[1], Z.shape[0])
    return PCA(n_components=n_comp, random_state=RANDOM_STATE).fit_transform(Z)


def oof_shot_quality(df, numeric, binary, categorical) -> np.ndarray:
    """Generate grouped out-of-fold predictions for a sensitivity workflow."""
    family, config, _ = best_config()
    factory = dict(build_candidates(numeric, binary, categorical)[family])[config]
    cols = list(numeric) + list(binary) + list(categorical)
    X, y = df[cols], df[TARGET].to_numpy()

    out = np.full(len(df), np.nan)
    gkf = GroupKFold(n_splits=5)
    for k, (tr, va) in enumerate(gkf.split(X, y, groups=df["match_id"])):
        n_inner = int(0.85 * len(tr))
        pipe = fit_with_validation(
            factory(),
            X.iloc[tr[:n_inner]],
            y[tr[:n_inner]],
            X.iloc[tr[n_inner:]],
            y[tr[n_inner:]],
        )
        out[va] = pipe.predict_proba(X.iloc[va])[:, 1]
        print(f"    no-situation profile workflow fold {k + 1}/5 done")
    if np.isnan(out).any():
        raise RuntimeError("out-of-fold prediction failed for at least one shot")
    return out


def shot_quality_checks(df):
    header("A  Shot-quality model sensitivity")
    train = df["season_name"].isin(TRAIN_SEASONS)
    test = df["season_name"].isin(TEST_SEASONS)
    base_rate = float(df.loc[train, TARGET].mean())

    p_ref, y_ref = fit_shot_quality(df, NUMERIC, BINARY, CATEGORICAL, train, test)
    ref = evaluate(y_ref, p_ref, base_rate, "reference")
    record("Shot-quality model", "Primary specification", "AUC (test season)",
           ref["AUC"], ref["AUC"])
    record("Shot-quality model", "Primary specification", "Brier skill score",
           ref["Brier skill score"], ref["Brier skill score"])
    print(f"  reference AUC {ref['AUC']:.4f}, BSS {ref['Brier skill score']:.4f}")

    restrictions = {
        "Excluding penalties": df["is_penalty"] == 0,
        "Excluding added-time shots": df["added_time_flag"] == 0,
        "Excluding set-piece shots": df["is_set_piece"] == 0,
        "Open play only": df["is_open_play"] == 1,
    }
    for label, mask in restrictions.items():
        tr, te = train & mask, test & mask
        p, y = fit_shot_quality(df, NUMERIC, BINARY, CATEGORICAL, tr, te)
        r = evaluate(y, p, float(df.loc[tr, TARGET].mean()), label)
        record("Shot-quality model", label, "AUC (test season)", r["AUC"], ref["AUC"],
               f"n = {int(te.sum()):,}")
        record("Shot-quality model", label, "Brier skill score",
               r["Brier skill score"], ref["Brier skill score"])
        print(f"  {label:<32} AUC {r['AUC']:.4f}  BSS {r['Brier skill score']:.4f}")

    # Alternative geometry.
    p, y = fit_shot_quality(df, NUMERIC_ALT, BINARY, CATEGORICAL, train, test)
    r = evaluate(y, p, base_rate, "alt geometry")
    record("Shot-quality model", "Metric-corrected distance/angle + visible goal angle",
           "AUC (test season)", r["AUC"], ref["AUC"],
           "Replaces the normalised proxies pre-specified in 5.2")
    print(f"  {'Metric-corrected geometry':<32} AUC {r['AUC']:.4f}  BSS {r['Brier skill score']:.4f}")

    # Without the taxonomy-affected predictor.
    cat_no_sit = [c for c in CATEGORICAL if c != "situation"]
    p, y = fit_shot_quality(df, NUMERIC, BINARY, cat_no_sit, train, test)
    r = evaluate(y, p, base_rate, "no situation")
    record("Shot-quality model", "Excluding the `situation` predictor",
           "AUC (test season)", r["AUC"], ref["AUC"],
           "`situation` taxonomy changes at the 2024/25 boundary (step 1b)")
    record("Shot-quality model", "Excluding the `situation` predictor",
           "Calibration intercept", r["Calibration intercept"], ref["Calibration intercept"])
    print(f"  {'Excluding situation predictor':<32} AUC {r['AUC']:.4f}  "
          f"calib. intercept {r['Calibration intercept']:.3f} (ref {ref['Calibration intercept']:.3f})")

    # Calibration separately by season, using out-of-fold predictions.
    for s in SEASONS:
        m = (df["season_name"] == s).to_numpy()
        slope, inter = calibration_slope_intercept(
            df.loc[m, TARGET].to_numpy(), df.loc[m, "shot_quality"].to_numpy()
        )
        obs = df.loc[m, TARGET].mean()
        pred = df.loc[m, "shot_quality"].mean()
        record("Shot-quality model", f"Calibration in {s}", "Calibration slope", slope, 1.0)
        record("Shot-quality model", f"Calibration in {s}", "Calibration intercept", inter, 0.0)
        record("Shot-quality model", f"Calibration in {s}",
               "Observed / predicted goals", obs / pred, 1.0)
        print(f"  calibration {s}: slope {slope:.3f}, intercept {inter:+.3f}, "
              f"obs/pred {obs / pred:.3f}")

    # Season-by-season refit: train on one season, test on the next.
    for a, b in zip(SEASONS[:-1], SEASONS[1:]):
        tr = (df["season_name"] == a)
        te = (df["season_name"] == b)
        cols = NUMERIC + BINARY + CATEGORICAL
        family, config, _ = best_config()
        pipe = dict(build_candidates(NUMERIC, BINARY, CATEGORICAL)[family])[config]()
        pipe.named_steps["clf"].set_params(iterations=400, early_stopping_rounds=None)
        pipe.fit(df.loc[tr, cols], df.loc[tr, TARGET])
        p = pipe.predict_proba(df.loc[te, cols])[:, 1]
        auc = roc_auc_score(df.loc[te, TARGET], p)
        record("Shot-quality model", f"Train {a} -> test {b}", "AUC", auc, ref["AUC"])
        print(f"  train {a} -> test {b}: AUC {auc:.4f}")


# --------------------------------------------------------------------------
# B. Clustering
# --------------------------------------------------------------------------


def clustering_checks(ts):
    header("B  Clustering sensitivity")
    variants = {
        "All profile features (primary)": PROFILE_FEATURES,
        "Spatial features only": [f for f in SPATIAL_ONLY_FEATURES if f in ts.columns],
        "Shot-quality features only": SHOT_QUALITY_FEATURES,
        "Shot-selection features only": SHOT_SELECTION_FEATURES,
        "Profile + competitive-load response features": (
            PROFILE_FEATURES + [f for f in LOAD_RESPONSE_FEATURES if ts[f].notna().all()]
        ),
    }
    ref_sil = None
    ref_jaccard = None
    for label, feats in variants.items():
        feats = [f for f in feats if f in ts.columns and ts[f].notna().all()]
        if len(feats) < 3:
            continue
        S = retained_profile_scores(ts, feats)
        lab = KMeans(
            n_clusters=PRIMARY_CLUSTER_K,
            n_init=50,
            random_state=RANDOM_STATE,
        ).fit_predict(S)
        sil = silhouette_score(S, lab)
        stab = bootstrap_stability(
            "k-means",
            S,
            PRIMARY_CLUSTER_K,
            n_boot=ROBUSTNESS_CLUSTER_BOOT,
        )
        if ref_sil is None:
            ref_sil = sil
            ref_jaccard = stab.min()
        record("Clustering", label, f"Silhouette (k = {PRIMARY_CLUSTER_K})", sil, ref_sil,
               f"{len(feats)} features")
        record(
            "Clustering",
            label,
            f"Minimum bootstrap Jaccard (k = {PRIMARY_CLUSTER_K})",
            stab.min(),
            ref_jaccard,
            "0.60 is the floor for a reportable cluster",
        )
        print(f"  {label:<46} silhouette {sil:.3f}  min Jaccard {stab.min():.3f}")

    # Pooled versus within-season standardisation.
    for label, Z in (
        ("Within-season standardisation (primary)",
         standardise_within_season(ts, PROFILE_FEATURES)),
        ("Pooled standardisation",
         StandardScaler().fit_transform(ts[PROFILE_FEATURES].to_numpy(dtype=float))),
    ):
        pca_full = PCA().fit(Z)
        n_comp = int(np.searchsorted(np.cumsum(pca_full.explained_variance_ratio_), 0.80) + 1)
        S = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit_transform(Z)
        lab = KMeans(
            n_clusters=PRIMARY_CLUSTER_K,
            n_init=50,
            random_state=RANDOM_STATE,
        ).fit_predict(S)
        stab = bootstrap_stability(
            "k-means",
            S,
            PRIMARY_CLUSTER_K,
            n_boot=ROBUSTNESS_CLUSTER_BOOT,
        )
        if label.startswith("Within-season"):
            ref_jaccard = stab.min()
        record(
            "Clustering",
            label,
            f"Minimum bootstrap Jaccard (k = {PRIMARY_CLUSTER_K})",
            stab.min(),
            ref_jaccard,
        )
        print(f"  {label:<46} min Jaccard {stab.min():.3f}")


def no_situation_profile_workflow(shots, primary_ts):
    header("B2  No-situation profile-construction workflow")
    load = load_frame("team_match_load")
    cat_no_sit = [c for c in CATEGORICAL if c != "situation"]

    rebuilt = shots.copy()
    rebuilt["shot_quality"] = oof_shot_quality(rebuilt, NUMERIC, BINARY, cat_no_sit)
    tm_no_sit = build_team_match(rebuilt, load)
    ts_no_sit = build_team_season(rebuilt, tm_no_sit)

    keys = ["team_id", "season_name"]
    merged = primary_ts[keys + PROFILE_FEATURES].merge(
        ts_no_sit[keys + PROFILE_FEATURES],
        on=keys,
        suffixes=("_primary", "_no_situation"),
        validate="one_to_one",
    )
    record(
        "Profile construction",
        "No-situation workflow",
        "Team-seasons retained",
        len(ts_no_sit),
        len(primary_ts),
        "OOF shot quality refit without `situation`; "
        "situation-derived profile features excluded below",
    )

    comparison_rows = []
    for f in NON_SITUATION_PROFILE_FEATURES:
        a = merged[f"{f}_primary"]
        b = merged[f"{f}_no_situation"]
        corr = a.corr(b)
        mad = (a - b).abs().mean()
        comparison_rows.append(
            {
                "Feature": f,
                "Primary mean": a.mean(),
                "No-situation mean": b.mean(),
                "Pearson r": corr,
                "Mean absolute difference": mad,
                "Maximum absolute difference": (a - b).abs().max(),
            }
        )
        if f in {"mean_shot_quality", "xg_per_match", "high_quality_share"}:
            record(
                "Profile construction",
                "No-situation workflow",
                f"{f} correlation",
                corr,
                1.0,
                "Compared with the primary team-season profile values",
            )
            record(
                "Profile construction",
                "No-situation workflow",
                f"{f} mean absolute difference",
                mad,
                0.0,
            )
    save_table(
        pd.DataFrame(comparison_rows).round(4),
        "table7b_no_situation_profile_comparison",
    )

    primary_ns = primary_ts[keys + NON_SITUATION_PROFILE_FEATURES].copy()
    rebuilt_ns = ts_no_sit[keys + NON_SITUATION_PROFILE_FEATURES].copy()
    primary_ns = primary_ns.merge(rebuilt_ns[keys], on=keys, validate="one_to_one")

    def profile_space(data):
        S = retained_profile_scores(data, NON_SITUATION_PROFILE_FEATURES)
        labels = KMeans(
            n_clusters=PRIMARY_CLUSTER_K,
            n_init=50,
            random_state=RANDOM_STATE,
        ).fit_predict(S)
        return S, labels

    S_primary, labels_primary = profile_space(primary_ns)
    S_no_sit, labels_no_sit = profile_space(rebuilt_ns)
    ari = adjusted_rand_score(labels_primary, labels_no_sit)
    sil = silhouette_score(S_no_sit, labels_no_sit)
    stab = bootstrap_stability(
        "k-means",
        S_no_sit,
        PRIMARY_CLUSTER_K,
        n_boot=ROBUSTNESS_CLUSTER_BOOT,
    )

    record(
        "Profile construction",
        "No-situation workflow",
        "Cluster adjusted Rand index",
        ari,
        1.0,
        "Primary and no-situation workflows both exclude situation-derived profile features",
    )
    record(
        "Profile construction",
        "No-situation workflow",
        f"Silhouette (k = {PRIMARY_CLUSTER_K})",
        sil,
        silhouette_score(S_primary, labels_primary),
    )
    record(
        "Profile construction",
        "No-situation workflow",
        f"Minimum bootstrap Jaccard (k = {PRIMARY_CLUSTER_K})",
        stab.min(),
        0.60,
        "0.60 is the floor for a reportable cluster",
    )
    print(
        f"  no-situation workflow: {len(ts_no_sit)} team-seasons, "
        f"ARI {ari:.3f}, silhouette {sil:.3f}, min Jaccard {stab.min():.3f}"
    )


# --------------------------------------------------------------------------
# C. Competitive-load models
# --------------------------------------------------------------------------


def load_checks(shots):
    header("C  Competitive-load model sensitivity")
    from s07_load_analysis import prepare

    d = prepare(shots)
    base = "minute_c + C(score_state) + is_home_shot + C(season_name) + post_break"

    def controls_for(data) -> str:
        """Drop controls that lost their variation under the current subset.

        Restricting the sample (for example to non-break fixtures) can make a
        control constant, which leaves the design matrix singular. Such terms
        carry no information in that subset and are removed rather than fitted.
        """
        terms = []
        for term, col in (
            ("minute_c", "minute_c"),
            ("C(score_state)", "score_state"),
            ("is_home_shot", "is_home_shot"),
            ("C(season_name)", "season_name"),
            ("post_break", "post_break"),
        ):
            if data[col].nunique() > 1:
                terms.append(term)
        return " + ".join(terms)

    def effect(data, formula, term, groups="team_id"):
        try:
            m = smf.mixedlm(formula, data, groups=data[groups]).fit(reml=True, method="lbfgs")
        except Exception as exc:
            print(f"      [skipped: {type(exc).__name__}]")
            return (np.nan,) * 3
        key = next((p for p in m.params.index if p.startswith(term)), None)
        return (float(m.params[key]), float(m.bse[key]), float(m.pvalues[key])) if key else (np.nan,) * 3

    b_ref, se_ref, p_ref = effect(d, f"logit_shot_quality ~ short_rest + {base}", "short_rest")
    record("Competitive load", "Primary specification",
           "Short-rest effect on logit shot quality", b_ref, b_ref, f"p = {p_ref:.3f}")
    print(f"  reference short-rest effect {b_ref:+.4f} (SE {se_ref:.4f}, p = {p_ref:.3f})")

    variations = {
        "Alternative rest cut-points (<=4 / 5-7 / >=8 days)":
            (d.assign(sr=(d["rest_category_alt"] == "short").astype(int)), "sr", d),
        "League-only calendar":
            (d.assign(sr=(d["rest_category_league_only"] == "short").astype(int)), "sr", d),
        "Excluding post-break fixtures":
            (d[d["post_break"] == 0].assign(sr=lambda x: x["short_rest"]), "sr", d),
        "Excluding penalties":
            (d[d["is_penalty"] == 0].assign(sr=lambda x: x["short_rest"]), "sr", d),
        "Open play only":
            (d[d["is_open_play"] == 1].assign(sr=lambda x: x["short_rest"]), "sr", d),
        "European participants only":
            (d[d["euro_participant"] == 1].assign(sr=lambda x: x["short_rest"]), "sr", d),
        "Non-European participants only":
            (d[d["euro_participant"] == 0].assign(sr=lambda x: x["short_rest"]), "sr", d),
    }
    for label, (data, term, _) in variations.items():
        if len(data) < 500 or data[term].nunique() < 2:
            continue
        b, se, p = effect(data, f"logit_shot_quality ~ {term} + {controls_for(data)}", term)
        if not np.isfinite(b):
            continue
        record("Competitive load", label, "Short-rest effect on logit shot quality",
               b, b_ref, f"n = {len(data):,}, p = {p:.3f}")
        print(f"  {label:<50} {b:+.4f} (p = {p:.3f}, n = {len(data):,})")

    # Adding cup and European exposure as covariates.
    b, se, p = effect(
        d, f"logit_shot_quality ~ short_rest + cup_last_7 + euro_last_7 + {base}", "short_rest"
    )
    record("Competitive load", "Adjusting for cup and European exposure",
           "Short-rest effect on logit shot quality", b, b_ref, f"p = {p:.3f}")
    print(f"  {'Adjusting for cup + European exposure':<50} {b:+.4f} (p = {p:.3f})")

    # Team and opponent fixed effects instead of a team random intercept.
    ols = smf.ols(
        f"logit_shot_quality ~ short_rest + C(team_id) + C(opponent_team_id) + {base}", d
    ).fit(cov_type="cluster", cov_kwds={"groups": d["match_id"]})
    record("Competitive load", "Team and opponent fixed effects (cluster-robust SE)",
           "Short-rest effect on logit shot quality",
           float(ols.params["short_rest"]), b_ref,
           f"p = {float(ols.pvalues['short_rest']):.3f}")
    print(f"  {'Team + opponent fixed effects':<50} "
          f"{float(ols.params['short_rest']):+.4f} (p = {float(ols.pvalues['short_rest']):.3f})")

    # No team term at all.
    ols2 = smf.ols(f"logit_shot_quality ~ short_rest + {base}", d).fit(
        cov_type="cluster", cov_kwds={"groups": d["match_id"]}
    )
    record("Competitive load", "No team identifiers", "Short-rest effect on logit shot quality",
           float(ols2.params["short_rest"]), b_ref,
           f"p = {float(ols2.pvalues['short_rest']):.3f}")
    print(f"  {'No team identifiers':<50} {float(ols2.params['short_rest']):+.4f} "
          f"(p = {float(ols2.pvalues['short_rest']):.3f})")

    for s in SEASONS:
        sub = d[d["season_name"] == s]
        if sub["short_rest"].nunique() < 2:
            continue
        b, se, p = effect(
            sub, f"logit_shot_quality ~ short_rest + {controls_for(sub)}", "short_rest"
        )
        if not np.isfinite(b):
            continue
        record("Competitive load", f"Season {s} only",
               "Short-rest effect on logit shot quality", b, b_ref, f"p = {p:.3f}")
        print(f"  season {s:<43} {b:+.4f} (p = {p:.3f})")


# --------------------------------------------------------------------------
# D. Player profiles
# --------------------------------------------------------------------------


def player_checks(shots):
    header("D  Player-profile threshold sensitivity")
    ref_n = None
    for thr in PLAYER_SHOT_THRESHOLD_SENSITIVITY:
        ps = build_player_season(shots, thr)
        if ps.empty:
            continue
        corr = ps["mean_shot_quality"].corr(ps["goal_rate"])
        foe_sd = ps["finishing_over_expectation"].std()
        if ref_n is None:
            ref_n = len(ps)
        record("Player profiles", f"Minimum {thr} shots per player-season",
               "Player-seasons retained", len(ps), ref_n)
        record("Player profiles", f"Minimum {thr} shots per player-season",
               "SD of finishing over expectation", foe_sd, None,
               "Reported as threshold sensitivity; retained-player composition changes with the cutoff")
        print(f"  >= {thr:>2} shots: {len(ps):>3} player-seasons, "
              f"SD(finishing over expectation) {foe_sd:.3f}, "
              f"r(shot quality, goal rate) {corr:.3f}")


def main() -> None:
    shots = load_frame("shots_scored")
    ts = load_frame("team_season_clustered")

    shot_quality_checks(shots)
    clustering_checks(ts)
    no_situation_profile_workflow(shots, ts)
    load_checks(shots)
    player_checks(shots)

    out = pd.DataFrame(ROWS)
    save_table(out, "table7_robustness")
    print(f"\n  {len(out)} robustness rows written")


if __name__ == "__main__":
    main()
