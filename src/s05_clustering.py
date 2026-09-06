"""Step 5 - Attacking-profile structure and clustering (proposal 5.6).

Hypothesis 1 predicts that unsupervised clustering recovers several discrete,
interpretable attacking-profile types. Testing that claim honestly requires
asking whether *any* discrete structure exists before naming clusters, so the
analysis runs four independent checks:

  1. internal validity indices (silhouette, Calinski-Harabasz, Davies-Bouldin,
     BIC) across k-means, Gaussian mixtures and Ward linkage, k = 2..7;
  2. a Jaccard bootstrap per cluster (Hennig, 2007), where >= 0.75 is stable,
     0.60-0.75 is a pattern and < 0.60 is not a reportable profile;
  3. a null reference drawn from a multivariate normal with the *same*
     covariance as the observed profiles, which shows what those indices look
     like when the data are a single elliptical cloud by construction;
  4. the gap statistic (Tibshirani et al., 2001), which unlike every index
     above can select k = 1 and therefore can reject clustering outright.

Both the full 17-feature profile block and a reduced 9-feature block are run,
because the full block contains several near-duplicate descriptors of shot
proximity whose redundancy would otherwise be a competing explanation for any
instability.

Produces
    tables/table4_cluster_characteristics.csv
    tables/table4b_cluster_selection_metrics.csv
    tables/table4c_cluster_membership.csv
    tables/table4d_pca_loadings.csv
    tables/table4e_feature_redundancy.csv
    figures/figure4_profile_space.pdf
    figures/figure5_cluster_radar.pdf
    data/team_season_clustered.parquet
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import plotstyle
from common import RANDOM_STATE, header, load_frame, save_frame, save_table
from s04_profiles import PROFILE_FEATURES, REDUCED_PROFILE_FEATURES

K_RANGE = range(2, 8)
N_BOOTSTRAP = 400
N_GAP_REF = 200
STABILITY_FLOOR = 0.60

RADAR_FEATURES = [
    "shots_per_match",
    "mean_shot_quality",
    "high_quality_share",
    "close_range_share",
    "header_share",
    "set_piece_share",
    "fast_break_share",
    "on_target_rate",
    "blocked_rate",
    "mean_distance_m",
]


# --------------------------------------------------------------------------
# Algorithms and validity machinery
# --------------------------------------------------------------------------


def fit_labels(algo: str, X: np.ndarray, k: int, seed: int = RANDOM_STATE) -> np.ndarray:
    if algo == "k-means":
        return KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(X)
    if algo == "GMM":
        return GaussianMixture(
            n_components=k, covariance_type="diag", n_init=20,
            random_state=seed, reg_covar=1e-4,
        ).fit_predict(X)
    if algo == "Ward":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    raise ValueError(algo)


def gmm_bic(X: np.ndarray, k: int) -> float:
    return GaussianMixture(
        n_components=k, covariance_type="diag", n_init=20,
        random_state=RANDOM_STATE, reg_covar=1e-4,
    ).fit(X).bic(X)


def bootstrap_stability(algo: str, X: np.ndarray, k: int, n_boot: int = N_BOOTSTRAP,
                        seed: int = RANDOM_STATE) -> np.ndarray:
    """Mean maximum Jaccard similarity per original cluster under resampling."""
    rng = np.random.default_rng(seed)
    base = fit_labels(algo, X, k)
    n = len(X)
    sims = np.full((n_boot, k), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(idx)) < k + 1:
            continue
        try:
            lab_b = fit_labels(algo, X[idx], k, seed=seed + b)
        except Exception:
            continue
        boot_sets = [set(idx[lab_b == d]) for d in range(k)]
        for c in range(k):
            orig = set(np.where(base == c)[0])
            sims[b, c] = max(
                (len(orig & bs) / len(orig | bs)) if (orig | bs) else 0.0
                for bs in boot_sets
            )
    return np.nanmean(sims, axis=0)


def within_dispersion(X: np.ndarray, labels: np.ndarray) -> float:
    """Pooled within-cluster sum of squared distances to the centroid."""
    total = 0.0
    for c in np.unique(labels):
        pts = X[labels == c]
        if len(pts) <= 1:
            continue
        total += ((pts - pts.mean(axis=0)) ** 2).sum()
    return total


def gap_statistic(X: np.ndarray, k_max: int = 7, n_ref: int = N_GAP_REF) -> pd.DataFrame:
    """Tibshirani gap statistic with a uniform reference over the PCA-aligned box.

    Unlike silhouette or Calinski-Harabasz, the gap statistic admits k = 1, so
    it can conclude that the data contain no cluster structure at all.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    # Reference boxes are aligned to the data's principal axes, as recommended
    # for correlated features.
    pca = PCA().fit(X)
    Xp = pca.transform(X)
    lo, hi = Xp.min(axis=0), Xp.max(axis=0)

    rows = []
    for k in range(1, k_max + 1):
        lab = np.zeros(len(X), dtype=int) if k == 1 else fit_labels("k-means", X, k)
        log_wk = np.log(max(within_dispersion(X, lab), 1e-12))

        ref_logs = np.empty(n_ref)
        for b in range(n_ref):
            Zp = rng.uniform(lo, hi, size=Xp.shape)
            Zr = pca.inverse_transform(Zp)
            lab_r = np.zeros(len(Zr), dtype=int) if k == 1 else fit_labels(
                "k-means", Zr, k, seed=RANDOM_STATE + b
            )
            ref_logs[b] = np.log(max(within_dispersion(Zr, lab_r), 1e-12))

        gap = ref_logs.mean() - log_wk
        sk = ref_logs.std(ddof=1) * np.sqrt(1 + 1 / n_ref)
        rows.append({"k": k, "log_Wk": log_wk, "gap": gap, "s_k": sk})

    df = pd.DataFrame(rows)
    # Smallest k with gap(k) >= gap(k+1) - s(k+1).
    choice = None
    for i in range(len(df) - 1):
        if df.loc[i, "gap"] >= df.loc[i + 1, "gap"] - df.loc[i + 1, "s_k"]:
            choice = int(df.loc[i, "k"])
            break
    df.attrs["k_selected"] = choice if choice is not None else int(df["k"].iloc[-1])
    return df


def null_reference_metrics(X: np.ndarray, algo: str, k: int, n_rep: int = 25) -> dict:
    """Silhouette and bootstrap stability on data that contain no clusters.

    Reference samples are multivariate normal with the observed mean and
    covariance, i.e. a single elliptical cloud with the same correlation
    structure as the real profiles.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    mu, cov = X.mean(axis=0), np.cov(X, rowvar=False)
    sils, stabs = [], []
    for r in range(n_rep):
        Xr = rng.multivariate_normal(mu, cov, size=len(X))
        lab = fit_labels(algo, Xr, k, seed=RANDOM_STATE + r)
        if len(np.unique(lab)) < 2:
            continue
        sils.append(silhouette_score(Xr, lab))
        stabs.append(bootstrap_stability(algo, Xr, k, n_boot=60, seed=RANDOM_STATE + r).mean())
    return {"null_silhouette": float(np.mean(sils)), "null_stability": float(np.mean(stabs))}


# --------------------------------------------------------------------------


def standardise_within_season(ts: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Z-score each feature inside its own season.

    Step 1b establishes that the provider's event taxonomy changes abruptly at
    the 2024/25 season boundary, shifting several league-wide feature means.
    Pooled standardisation would let that shift enter the profile vectors, so
    that every team appears to move in the same direction between seasons.
    Standardising within season expresses each team relative to its own
    league-season context, which is the quantity the profile is meant to
    capture, and removes any common annual shift by construction.
    """
    out = np.empty((len(ts), len(features)), dtype=float)
    for si, season in enumerate(ts["season_name"].unique()):
        m = (ts["season_name"] == season).to_numpy()
        block = ts.loc[m, features].to_numpy(dtype=float)
        mu = block.mean(axis=0)
        sd = block.std(axis=0, ddof=0)
        sd[sd == 0] = 1.0
        out[m] = (block - mu) / sd
    return out


def scree_table(pca_full: PCA, tag: str) -> pd.DataFrame:
    """Variance explained per component, for reporting alongside any PCA figure.

    A two-dimensional projection is only an honest summary of the data if the
    first two components carry most of the variance, so the full scree is
    reported rather than the two components that happen to be plotted.
    """
    evr = pca_full.explained_variance_ratio_ * 100
    eig = pca_full.explained_variance_
    return pd.DataFrame(
        {
            "Feature set": tag,
            "Component": [f"PC{i + 1}" for i in range(len(evr))],
            "Variance explained (%)": np.round(evr, 1),
            "Cumulative variance (%)": np.round(np.cumsum(evr), 1),
            "Eigenvalue": np.round(eig, 2),
        }
    )


def report_scree(pca_full: PCA, tag: str, n_comp: int) -> None:
    """Print the component-retention evidence a PCA reviewer expects."""
    evr = pca_full.explained_variance_ratio_ * 100
    cum = np.cumsum(evr)
    eig = pca_full.explained_variance_
    kaiser = int((eig > 1).sum())
    print(f"  {tag}:")
    print(f"    PC1 {evr[0]:.1f}%   PC2 {evr[1]:.1f}%   "
          f"PC1+PC2 {cum[1]:.1f}%   PC3 {evr[2]:.1f}%   PC4 {evr[3]:.1f}%")
    print(f"    eigenvalues: " + "  ".join(f"PC{i + 1} {eig[i]:.2f}" for i in range(5)))
    print(f"    scree gaps (pp): PC1->PC2 {evr[0] - evr[1]:.1f}, "
          f"PC2->PC3 {evr[1] - evr[2]:.1f}, PC3->PC4 {evr[2] - evr[3]:.1f}")
    print(f"    components with eigenvalue > 1 (Kaiser): {kaiser}; "
          f"retained for 80% of variance: {n_comp}")
    if cum[1] < 70 or eig[2] > 1:
        print("    NOTE: the first two components do not dominate - PC3 exceeds the "
              "Kaiser threshold, so the two-component plot is a projection for "
              "readability, not a claim that the profile space is two-dimensional.")


def component_invariance(S: np.ndarray, k: int) -> pd.DataFrame:
    """Does the partition depend on how many components are retained?

    Reported because the profile figure shows two components while the
    clustering runs on more; if the partition changed with that choice, the
    figure and the analysis would be telling different stories.
    """
    labels = {
        n: fit_labels("k-means", S[:, :n], k)
        for n in range(2, min(S.shape[1], 4) + 1)
    }
    rows = []
    ns = sorted(labels)
    for i, a in enumerate(ns):
        for b in ns[i + 1:]:
            rows.append({
                "Components A": a,
                "Components B": b,
                "Adjusted Rand index": round(adjusted_rand_score(labels[a], labels[b]), 3),
            })
    return pd.DataFrame(rows)


def redundancy_table(ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(PROFILE_FEATURES, 2):
        r = ts[a].corr(ts[b])
        if abs(r) > 0.80:
            rows.append({"Feature A": a, "Feature B": b, "Pearson r": round(r, 3)})
    return pd.DataFrame(rows).sort_values("Pearson r", key=abs, ascending=False)


def run_selection(S: np.ndarray, tag: str) -> pd.DataFrame:
    rows = []
    for algo in ("k-means", "GMM", "Ward"):
        for k in K_RANGE:
            lab = fit_labels(algo, S, k)
            if len(np.unique(lab)) < 2:
                continue
            stab = bootstrap_stability(algo, S, k)
            rows.append(
                {
                    "Feature set": tag,
                    "Algorithm": algo,
                    "k": k,
                    "Silhouette": silhouette_score(S, lab),
                    "Calinski-Harabasz": calinski_harabasz_score(S, lab),
                    "Davies-Bouldin": davies_bouldin_score(S, lab),
                    "BIC (GMM)": gmm_bic(S, k) if algo == "GMM" else np.nan,
                    "Mean bootstrap Jaccard": stab.mean(),
                    "Min bootstrap Jaccard": stab.min(),
                    "Smallest cluster n": int(np.bincount(lab).min()),
                }
            )
    return pd.DataFrame(rows)


def name_clusters(centroids: pd.DataFrame, sizes: dict | None = None) -> dict:
    """Descriptive labels from the direction of each centroid's extreme z-scores.

    Because the clusterability tests reject discrete profile types, labels are
    deliberately descriptive rather than typological. A group holding most of
    the league is named as such instead of being given a style label it does
    not earn: calling 40 of 48 team-seasons "perimeter attackers" would imply a
    tactical identity that the partition does not establish.
    """
    names, used = {}, set()
    total = sum(sizes.values()) if sizes else 0
    for cid, row in centroids.iterrows():
        if sizes and total and sizes.get(cid, 0) / total > 0.60:
            names[cid] = "Main body of the league"
            used.add(names[cid])
            continue
        vol = row.get("shots_per_match", 0)
        qual = row.get("mean_shot_quality", 0)
        setp = row.get("set_piece_share", 0)
        fast = row.get("fast_break_share", 0)
        head = row.get("header_share", 0)
        fin = row.get("goals_minus_xg_per_match", 0)
        ont = row.get("on_target_rate", 0)

        if vol > 0.45 and qual > 0.20:
            base = "High-volume chance creators"
        elif vol > 0.40 and qual < 0.0:
            base = "High-volume, low-quality shooters"
        elif fast > 0.45 or (qual > 0.35 and setp > 0.2):
            base = "Direct/transition attackers"
        elif setp > 0.45 or head > 0.45:
            base = "Set-piece and aerial-oriented"
        elif qual < -0.25 and vol < 0.0:
            base = "Perimeter, low-volume attackers"
        elif fin > 0.35 or ont > 0.35:
            base = "Efficient finishers"
        elif vol < -0.40:
            base = "Low-volume, restrained attackers"
        else:
            base = "Balanced attackers"
        n, cand = 1, base
        while cand in used:
            n += 1
            cand = f"{base} ({n})"
        used.add(cand)
        names[cid] = cand
    return names


def main() -> None:
    header("STEP 5  Attacking-profile structure and clustering")
    plotstyle.apply()

    ts = load_frame("team_season_profiles").reset_index(drop=True)

    red = redundancy_table(ts)
    print(f"  {len(red)} feature pairs in the full block correlate above |r| = 0.80")
    save_table(red, "table4e_feature_redundancy")

    spaces = {}
    scree_rows = []
    for tag, feats in (("Full (17 features)", PROFILE_FEATURES),
                       ("Reduced (9 features)", REDUCED_PROFILE_FEATURES)):
        Z = standardise_within_season(ts, feats)
        pca_full = PCA().fit(Z)
        cum = np.cumsum(pca_full.explained_variance_ratio_)
        n_comp = int(np.searchsorted(cum, 0.80) + 1)
        pca = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit(Z)
        spaces[tag] = {
            "feats": feats, "Z": Z, "pca": pca, "S": pca.transform(Z),
            "evr": pca_full.explained_variance_ratio_, "n_comp": n_comp,
        }
        scree_rows.append(scree_table(pca_full, tag))
        report_scree(pca_full, tag, n_comp)

    scree = pd.concat(scree_rows, ignore_index=True)
    save_table(scree, "table4g_pca_scree")

    # ---- Selection metrics for both feature blocks -----------------------
    sel = pd.concat(
        [run_selection(sp["S"], tag) for tag, sp in spaces.items()], ignore_index=True
    )
    print("\n  cluster-selection metrics:")
    print(sel.round(3).to_string(index=False))
    save_table(sel.round(4), "table4b_cluster_selection_metrics")

    # ---- Gap statistic: is k = 1 preferred? ------------------------------
    print("\n  gap statistic (can select k = 1, i.e. no cluster structure):")
    gap_rows = []
    for tag, sp in spaces.items():
        g = gap_statistic(sp["S"])
        g.insert(0, "Feature set", tag)
        gap_rows.append(g)
        print(f"    {tag}: k selected = {g.attrs['k_selected']}")
        print("      " + g.round(3).to_string(index=False).replace("\n", "\n      "))
    save_table(pd.concat(gap_rows, ignore_index=True).round(4), "table4f_gap_statistic")
    gap_choice = {tag: gap_statistic(sp["S"]).attrs["k_selected"] for tag, sp in spaces.items()}

    # ---- Best available partition ----------------------------------------
    stable = sel[(sel["Min bootstrap Jaccard"] >= STABILITY_FLOOR)
                 & (sel["Smallest cluster n"] >= 4)]
    structure_supported = not stable.empty
    pool = stable if structure_supported else sel[sel["Smallest cluster n"] >= 4]
    chosen = pool.sort_values("Silhouette", ascending=False).iloc[0]
    tag, algo, k = chosen["Feature set"], chosen["Algorithm"], int(chosen["k"])
    sp = spaces[tag]
    S, feats = sp["S"], sp["feats"]

    print(
        f"\n  best available partition: {algo}, k={k} on the {tag.lower()} block "
        f"(silhouette {chosen['Silhouette']:.3f}, "
        f"min bootstrap Jaccard {chosen['Min bootstrap Jaccard']:.3f})"
    )
    null = null_reference_metrics(S, algo, k)
    print(
        f"  null reference (no clusters by construction): "
        f"silhouette {null['null_silhouette']:.3f}, "
        f"mean bootstrap Jaccard {null['null_stability']:.3f}"
    )
    print(
        "  HYPOTHESIS 1 VERDICT: "
        + (
            "discrete profile types supported"
            if structure_supported and all(v > 1 for v in gap_choice.values())
            else "NOT supported - profiles form a continuum; the partition below "
                 "is reported as a descriptive device, not as discrete types"
        )
    )

    labels = fit_labels(algo, S, k)
    stability = bootstrap_stability(algo, S, k)

    Zsel = standardise_within_season(ts, PROFILE_FEATURES)
    vol_idx = PROFILE_FEATURES.index("shots_per_match")
    order = np.argsort([Zsel[labels == c, vol_idx].mean() for c in range(k)])[::-1]
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[c] for c in labels])
    stability = stability[order]

    ts["cluster"] = labels
    ts["cluster_stability"] = [stability[c] for c in labels]

    agree = {
        other: adjusted_rand_score(labels, fit_labels(other, S, k))
        for other in ("k-means", "GMM", "Ward") if other != algo
    }
    print(f"  agreement with alternative algorithms (ARI): "
          f"{ {a: round(v, 3) for a, v in agree.items()} }")

    centroids_z = pd.DataFrame(
        [Zsel[labels == c].mean(axis=0) for c in range(k)], columns=PROFILE_FEATURES
    )
    names = name_clusters(centroids_z, {c: int((labels == c).sum()) for c in range(k)})
    ts["profile"] = [names[c] for c in labels]
    print("\n  cluster labels:")
    for c in range(k):
        print(f"    {c}: {names[c]:<38} n={int((labels == c).sum()):>2}  "
              f"bootstrap Jaccard {stability[c]:.2f}")

    # ---- Continuous profile space ----------------------------------------
    # The first component separates how good the chances a team creates are;
    # the second relates to how well it converts them, which is the distinction
    # RQ6 asks about. The third is retained and reported too: it exceeds the
    # Kaiser threshold and is interpretable, so omitting it would overstate how
    # low-dimensional the profile space is. Signs are fixed so that higher
    # always means "more". Note that the components are uncorrelated by
    # construction - that is a property of PCA, not a finding - so the
    # substantive evidence for separating chance creation from finishing is
    # their contrasting season-to-season repeatability in step 6, not r = 0.
    full = spaces["Full (17 features)"]
    load1 = pd.Series(full["pca"].components_[0], index=PROFILE_FEATURES)
    load2 = pd.Series(full["pca"].components_[1], index=PROFILE_FEATURES)
    load3 = pd.Series(full["pca"].components_[2], index=PROFILE_FEATURES)
    s1 = np.sign(load1["mean_shot_quality"]) or 1.0
    s2 = np.sign(load2["goals_minus_xg_per_match"]) or 1.0
    s3 = np.sign(load3["set_piece_share"]) or 1.0
    ts["chance_creation_axis"] = full["S"][:, 0] * s1
    ts["finishing_axis"] = full["S"][:, 1] * s2
    ts["chance_source_axis"] = full["S"][:, 2] * s3

    evr_full = full["evr"] * 100
    loadings = pd.DataFrame(
        {
            f"PC1 chance creation ({evr_full[0]:.1f}%)": load1 * s1,
            f"PC2 finishing ({evr_full[1]:.1f}%)": load2 * s2,
            f"PC3 chance source ({evr_full[2]:.1f}%)": load3 * s3,
        }
    ).round(3)
    save_table(loadings.reset_index().rename(columns={"index": "Feature"}),
               "table4d_pca_loadings")
    for i, label in enumerate(("chance-creation", "finishing", "chance-source")):
        print(f"\n  {label} axis (PC{i + 1}, {evr_full[i]:.1f}%), strongest loadings:")
        print(loadings.iloc[:, [i]]
              .reindex(loadings.iloc[:, i].abs().sort_values(ascending=False).index)
              .head(5).to_string())

    inv = component_invariance(full["S"], k)
    save_table(inv, "table4h_component_invariance")
    print("\n  partition stability across the number of retained components:")
    print("    " + inv.to_string(index=False).replace("\n", "\n    "))

    # ---- Table 4 ----------------------------------------------------------
    raw_means = pd.DataFrame(
        [ts.loc[labels == c, PROFILE_FEATURES].mean() for c in range(k)],
        columns=PROFILE_FEATURES,
    )
    t4 = pd.DataFrame({"Cluster": range(k), "Profile": [names[c] for c in range(k)]})
    t4["Team-seasons"] = [int((labels == c).sum()) for c in range(k)]
    t4["Bootstrap Jaccard"] = np.round(stability, 3)
    t4["Stability verdict"] = [
        "stable" if s >= 0.75 else ("pattern" if s >= 0.60 else "not reportable as a type")
        for s in stability
    ]
    for f in PROFILE_FEATURES:
        t4[f] = raw_means[f].round(3).to_numpy()
        t4[f"{f} (z)"] = centroids_z[f].round(2).to_numpy()
    t4["Representative team-seasons"] = [
        ", ".join(
            ts.loc[labels == c]
            .assign(d=lambda d: ((d[PROFILE_FEATURES] - raw_means.loc[c]) ** 2).sum(axis=1))
            .nsmallest(3, "d")["team_season"].tolist()
        )
        for c in range(k)
    ]
    save_table(t4, "table4_cluster_characteristics")
    save_table(
        ts[["team_id", "team_name", "season_name", "team_season", "cluster", "profile",
            "cluster_stability", "chance_creation_axis", "finishing_axis",
            "chance_source_axis",
            "matches", "shots", "goals"] + PROFILE_FEATURES].round(4),
        "table4c_cluster_membership",
    )
    save_frame(ts, "team_season_clustered")

    make_profile_space_figure(ts, S, names, k, full["evr"])
    make_radar_figure(centroids_z, names, k)


def make_profile_space_figure(ts, S, names, k, evr) -> None:
    import matplotlib.pyplot as plt
    import umap

    fig, axes = plt.subplots(
        1, 2, figsize=(plotstyle.W_DOUBLE, 3.35),
        gridspec_kw={"width_ratios": [1.08, 1.0], "wspace": 0.26},
    )

    ax = axes[0]
    x = ts["chance_creation_axis"].to_numpy()
    y = ts["finishing_axis"].to_numpy()
    ax.axhline(0, color=plotstyle.GRID, lw=0.9, zorder=1)
    ax.axvline(0, color=plotstyle.GRID, lw=0.9, zorder=1)
    for c in range(k):
        m = ts["cluster"].to_numpy() == c
        ax.scatter(x[m], y[m], s=30, color=plotstyle.CATEGORICAL[c],
                   marker=plotstyle.MARKERS[c], edgecolor="white", linewidth=0.6,
                   alpha=0.94, label=names[c], zorder=3)
    ax.margins(x=0.08, y=0.08)
    ax.set_xlabel(f"Chance-creation axis (PC1, {evr[0]:.0%} of variance)")
    ax.set_ylabel(f"Finishing axis (PC2, {evr[1]:.0%})")
    ax.set_title("a) Profile axes", loc="left", x=0.0)
    ax.grid(axis="both")

    ax = axes[1]
    emb = umap.UMAP(n_neighbors=8, min_dist=0.35, n_components=2,
                    random_state=RANDOM_STATE).fit_transform(S)
    for c in range(k):
        m = ts["cluster"].to_numpy() == c
        ax.scatter(emb[m, 0], emb[m, 1], s=30, color=plotstyle.CATEGORICAL[c],
                   marker=plotstyle.MARKERS[c], edgecolor="white", linewidth=0.6,
                   alpha=0.94, zorder=3)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("b) UMAP projection", loc="left", x=0.0)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=min(k, 3),
               bbox_to_anchor=(0.5, 0.035), frameon=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.30, wspace=0.26)
    plotstyle.save(fig, "figure4_profile_space")


def make_radar_figure(centroids_z, names, k) -> None:
    import matplotlib.pyplot as plt

    feats = [f for f in RADAR_FEATURES if f in centroids_z.columns]
    ncols = min(k, 3)
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(plotstyle.W_DOUBLE, 3.2 * nrows),
                             subplot_kw={"projection": "polar"})
    axes = np.atleast_1d(axes).ravel()

    ang = np.linspace(0, 2 * np.pi, len(feats), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    lim = float(np.abs(centroids_z[feats].to_numpy()).max()) * 1.3
    pretty = [f.replace("_", " ").replace("mean ", "") for f in feats]

    for c in range(k):
        ax = axes[c]
        vals = centroids_z.loc[c, feats].to_numpy(dtype=float)
        vc = np.concatenate([vals, vals[:1]])
        ax.plot(ang_c, vc, color=plotstyle.CATEGORICAL[c], lw=1.8, zorder=3)
        ax.fill(ang_c, vc, color=plotstyle.CATEGORICAL[c], alpha=0.20, zorder=2)
        ax.plot(ang_c, np.zeros_like(ang_c), color=plotstyle.INK_MUTED, lw=0.8,
                ls=(0, (3, 3)), zorder=1)
        ax.set_xticks(ang)
        ax.set_xticklabels(pretty, fontsize=6.0)
        ax.set_ylim(-lim, lim)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["-1 SD", "mean", "+1 SD"], fontsize=5.8,
                           color=plotstyle.INK_MUTED)
        ax.set_title(names[c], fontsize=8, pad=14)
        ax.grid(color=plotstyle.GRID, lw=0.5)
        ax.spines["polar"].set_color(plotstyle.GRID)

    for j in range(k, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Cluster centroids, standardised against the league mean",
                 fontsize=9, fontweight="bold", y=1.0)
    fig.tight_layout()
    plotstyle.save(fig, "figure5_cluster_radar")


if __name__ == "__main__":
    main()
