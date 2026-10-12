"""Step 15 - Reliability of the separate 11-indicator process-based axis.

Reviewer request: report between-season and within-season reliability for the
process-based chance-creation axis used in Sections 2.6/3.5 (11 indicators,
realised-conversion outcomes excluded), using the same methodology as
Table 6 (Panels A and B), which was built for the original 17-feature PC1.

Methodological constraint carried over from Table 6: the PCA that defines the
axis is fitted once on the full team-season data (`fit_process_axis_reference`)
and then applied with the *same* loadings, sign convention and per-season
standardisation statistics to (a) every team-season (between-season analysis)
and (b) every first-half / second-half split (within-season analysis). No PCA
is refitted on a half-season or on a single season in isolation; this is the
direct analogue of `fit_axis_reference` / `add_axes` in s12_core_validation.py,
specialised to the 11-feature `PROCESS_FEATURES` set.

Produces
    tables/table_process_axis_reliability_between.csv
    tables/table_process_axis_reliability_within.csv
    tables/table_process_axis_reliability_summary.csv
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

from common import RANDOM_STATE, SEASONS, header, load_frame, save_table
from s04_profiles import _core_aggregates
from s06_stability import icc1
from s12_core_validation import PROCESS_FEATURES, ci, format_p

warnings.filterwarnings("ignore")

N_BOOT = 2000


# --------------------------------------------------------------------------
# Single, globally fixed PCA transformation for the 11-indicator axis
# --------------------------------------------------------------------------


def fit_process_axis_reference(ts: pd.DataFrame) -> dict:
    """Fit the process-axis PCA once on the full team-season table.

    Mirrors `fit_axis_reference` in s12_core_validation.py: standardise each
    feature within its own season, fit a single PCA on the pooled z-scores,
    fix the sign so higher values mean greater xG, and store the per-season
    mean/SD used for standardisation so the *same* transformation can be
    re-applied later to any subset of team-seasons (e.g. half-seasons)
    without ever refitting the PCA.
    """
    feats = [f for f in PROCESS_FEATURES if f in ts.columns]
    by_season = {}
    Z = np.empty((len(ts), len(feats)), dtype=float)
    for season, g in ts.groupby("season_name"):
        idx = ts.index.get_indexer(g.index)
        block = g[feats].to_numpy(dtype=float)
        mu = block.mean(axis=0)
        sd = block.std(axis=0, ddof=0)
        sd[sd == 0] = 1.0
        by_season[season] = (mu, sd)
        Z[idx] = (block - mu) / sd

    pca = PCA().fit(Z)
    sign = np.sign(pd.Series(pca.components_[0], index=feats)["xg_per_match"]) or 1.0
    return {"pca": pca, "sign": sign, "by_season": by_season, "features": feats}


def apply_process_axis(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Score any team-season table on the fixed reference PCA.

    Uses the per-season mean/SD captured in `ref` (estimated on the *full*
    season), never statistics recomputed from `df` itself, so a half-season
    frame is projected into exactly the same coordinate system as the
    full-season frame it was split from.
    """
    out = df.copy()
    feats = ref["features"]
    scores = np.full(len(out), np.nan)
    for season, idx in out.groupby("season_name").groups.items():
        mu, sd = ref["by_season"][season]
        z = (out.loc[idx, feats].to_numpy(dtype=float) - mu) / sd
        scores[out.index.get_indexer(idx)] = ref["pca"].transform(z)[:, 0] * ref["sign"]
    out["process_chance_creation_axis"] = scores
    return out


# --------------------------------------------------------------------------
# Between-season reliability (Table 6, Panel A analogue)
# --------------------------------------------------------------------------


def between_season_reliability(ts_scored: pd.DataFrame, feature: str) -> dict:
    xs, ys = [], []
    for a, b in zip(SEASONS[:-1], SEASONS[1:]):
        left = ts_scored[ts_scored["season_name"] == a].set_index("team_id")[feature]
        right = ts_scored[ts_scored["season_name"] == b].set_index("team_id")[feature]
        common = left.index.intersection(right.index)
        xs.extend(left.loc[common].to_numpy())
        ys.extend(right.loc[common].to_numpy())
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]

    r, p = stats.pearsonr(xs, ys)
    z = np.arctanh(r)
    se = 1 / np.sqrt(len(xs) - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)

    icc = icc1(ts_scored[["team_id", feature]], "team_id", feature)

    return {
        "Feature": feature,
        "Pairs": int(len(xs)),
        "Year-to-year r": round(float(r), 3),
        "95% CI": f"[{lo:.2f}, {hi:.2f}]",
        "p": format_p(float(p)),
        "ICC(1,1)": round(float(icc), 3) if np.isfinite(icc) else np.nan,
    }


# --------------------------------------------------------------------------
# Within-season reliability (Table 6, Panel B analogue)
# --------------------------------------------------------------------------


def aggregate_half(shots: pd.DataFrame, tm: pd.DataFrame, selector, ref: dict) -> pd.DataFrame:
    rows = []
    for (team_id, season), g in tm.groupby(["team_id", "season_name"]):
        sel = selector(g)
        if sel.empty:
            continue
        sub = shots[shots["match_id"].isin(sel["match_id"]) & (shots["team_id"] == team_id)]
        if len(sub) < 20:
            continue
        rec = {"team_id": team_id, "season_name": season, "matches": len(sel)}
        agg = _core_aggregates(sub)
        rec.update(agg)
        rec["shots_per_match"] = agg["shots"] / len(sel)
        rec["xg_per_match"] = agg["total_xg"] / len(sel)
        rows.append(rec)
    return apply_process_axis(pd.DataFrame(rows), ref)


def within_season_reliability(shots: pd.DataFrame, tm: pd.DataFrame, ref: dict, feature: str) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    first = aggregate_half(
        shots, tm,
        lambda g: g[g["match_number"] <= int(np.floor(g["match_number"].max() / 2))],
        ref,
    )
    second = aggregate_half(
        shots, tm,
        lambda g: g[g["match_number"] > int(np.floor(g["match_number"].max() / 2))],
        ref,
    )
    d = first.merge(second, on=["team_id", "season_name"], suffixes=("_first", "_second"))
    a, b = f"{feature}_first", f"{feature}_second"
    ok = d[[a, b]].dropna()

    pear = pearsonr(ok[a], ok[b]).statistic
    spear = spearmanr(ok[a], ok[b]).correlation
    mae = float(np.mean(np.abs(ok[a] - ok[b])))

    pear_boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(ok), len(ok))
        aa, bb = ok[a].to_numpy()[idx], ok[b].to_numpy()[idx]
        if np.std(aa) == 0 or np.std(bb) == 0:
            continue
        pear_boot.append(pearsonr(aa, bb).statistic)

    return {
        "Indicator": feature,
        "Pearson r": round(float(pear), 3),
        "95% CI": ci(pear_boot),
        "Spearman rho": round(float(spear), 3),
        "MAE": round(mae, 3),
        "n": int(len(ok)),
    }, d


# --------------------------------------------------------------------------


def main() -> None:
    header("STEP 15  Reliability of the 11-indicator process-based axis")

    shots = load_frame("shots_scored")
    tm = load_frame("team_match_profiles")
    ts = load_frame("team_season_clustered")

    ref = fit_process_axis_reference(ts)
    print(f"  process-axis PCA fitted once on {len(ts)} team-seasons; "
          f"features = {ref['features']}")

    ts_scored = apply_process_axis(ts, ref)

    between = between_season_reliability(ts_scored, "process_chance_creation_axis")
    print("\n  between-season reliability (Panel A analogue):")
    print(pd.DataFrame([between]).to_string(index=False))

    within, split_data = within_season_reliability(
        shots, tm, ref, "process_chance_creation_axis"
    )
    print("\n  within-season reliability (Panel B analogue):")
    print(pd.DataFrame([within]).to_string(index=False))

    save_table(pd.DataFrame([between]), "table_process_axis_reliability_between")
    save_table(pd.DataFrame([within]), "table_process_axis_reliability_within")

    summary = pd.DataFrame(
        [
            {
                "Analysis": "Between-season",
                "Pearson r": between["Year-to-year r"],
                "95% CI": between["95% CI"],
                "p": between["p"],
                "ICC(1,1)": between["ICC(1,1)"],
                "Spearman rho": "—",
                "MAE": "—",
                "n": between["Pairs"],
            },
            {
                "Analysis": "Within-season",
                "Pearson r": within["Pearson r"],
                "95% CI": within["95% CI"],
                "p": "—",
                "ICC(1,1)": "—",
                "Spearman rho": within["Spearman rho"],
                "MAE": within["MAE"],
                "n": within["n"],
            },
        ]
    )
    save_table(summary, "table_process_axis_reliability_summary")
    print("\n  summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
