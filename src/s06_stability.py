"""Step 6 - Temporal stability and transition analysis (proposal 5.7).

Hypothesis 2 predicts moderate year-to-year stability of team attacking
profiles and greater variability at player level. Stability is quantified on
two tracks, because step 5 showed the profile space to be continuous rather
than made of discrete types:

  * categorical - same-cluster retention, transition probability matrices,
    Cramer's V and the adjusted Rand index between consecutive seasons;
  * continuous - season-to-season repeatability of each profile feature and of
    the two profile axes, an intraclass correlation with team as a random
    effect, and the Euclidean distance between a team's own standardised
    profile vectors compared against a permutation null.

The continuous track is the better powered of the two with 15 teams observed
across all three seasons, and is treated as primary.

Produces
    tables/table5_transition_matrix.csv
    tables/table5b_profile_repeatability.csv
    tables/table5c_trajectory_typology.csv
    figures/figure6_profile_transitions.pdf
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score

import plotstyle
from common import RANDOM_STATE, SEASONS, header, load_frame, save_table
from s04_profiles import PROFILE_FEATURES
from s05_clustering import standardise_within_season

warnings.filterwarnings("ignore", category=RuntimeWarning)

AXES = ["chance_creation_axis", "finishing_axis", "chance_source_axis"]
REPEATABILITY_FEATURES = [
    "chance_creation_axis",
    "finishing_axis",
    "chance_source_axis",
    "mean_shot_quality",
    "median_shot_quality",
    "xg_per_match",
    "goals_minus_xg_per_match",
    "mean_distance_m",
    "mean_angle_proxy",
    "high_quality_share",
    "low_quality_share",
    "shots_per_match",
    "on_target_rate",
    "blocked_rate",
    "goal_rate",
    "header_share",
    "assisted_share",
    "set_piece_share",
    "close_range_share",
    "fast_break_share",
]


# --------------------------------------------------------------------------
# Categorical stability
# --------------------------------------------------------------------------


def cramers_v(table: np.ndarray) -> float:
    """Bias-corrected Cramer's V (Bergsma, 2013)."""
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    if n == 0:
        return np.nan
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else np.nan


def transition_analysis(ts: pd.DataFrame, k: int):
    pairs = []
    for a, b in zip(SEASONS[:-1], SEASONS[1:]):
        left = ts[ts["season_name"] == a].set_index("team_id")
        right = ts[ts["season_name"] == b].set_index("team_id")
        common = left.index.intersection(right.index)
        for t in common:
            pairs.append(
                {
                    "team_id": t,
                    "team_name": left.loc[t, "team_name"],
                    "from_season": a,
                    "to_season": b,
                    "from_cluster": int(left.loc[t, "cluster"]),
                    "to_cluster": int(right.loc[t, "cluster"]),
                }
            )
    tr = pd.DataFrame(pairs)

    counts = np.zeros((k, k), dtype=int)
    for _, r in tr.iterrows():
        counts[r["from_cluster"], r["to_cluster"]] += 1

    retention = float(np.trace(counts) / counts.sum()) if counts.sum() else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        probs = counts / counts.sum(axis=1, keepdims=True)

    aris = []
    for a, b in zip(SEASONS[:-1], SEASONS[1:]):
        left = ts[ts["season_name"] == a].set_index("team_id")
        right = ts[ts["season_name"] == b].set_index("team_id")
        common = left.index.intersection(right.index)
        if len(common) > 2:
            aris.append(
                adjusted_rand_score(
                    left.loc[common, "cluster"], right.loc[common, "cluster"]
                )
            )
    return tr, counts, probs, retention, float(np.mean(aris)) if aris else np.nan


# --------------------------------------------------------------------------
# Continuous stability
# --------------------------------------------------------------------------


def icc1(values: pd.DataFrame, group: str, value: str) -> float:
    """ICC(1) from one-way ANOVA variance components.

    Computed directly rather than through a mixed model: with 14-18 teams
    observed over three seasons the restricted-likelihood fit frequently lands
    on the variance boundary and returns an uninterpretable estimate, whereas
    the ANOVA decomposition is exact for this design.
    """
    d = values[[group, value]].dropna()
    groups = [g[value].to_numpy(dtype=float) for _, g in d.groupby(group) if len(g) > 1]
    if len(groups) < 3:
        return np.nan
    n_i = np.array([len(g) for g in groups])
    k = n_i.mean()
    grand = np.concatenate(groups).mean()
    ms_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (len(groups) - 1)
    df_w = sum(len(g) - 1 for g in groups)
    if df_w == 0:
        return np.nan
    ms_within = sum(((g - g.mean()) ** 2).sum() for g in groups) / df_w
    denom = ms_between + (k - 1) * ms_within
    return float((ms_between - ms_within) / denom) if denom != 0 else np.nan


def repeatability(ts: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Consecutive-season correlation and ICC for each profile feature."""
    rows = []
    for f in features:
        xs, ys = [], []
        for a, b in zip(SEASONS[:-1], SEASONS[1:]):
            left = ts[ts["season_name"] == a].set_index("team_id")[f]
            right = ts[ts["season_name"] == b].set_index("team_id")[f]
            common = left.index.intersection(right.index)
            xs.extend(left.loc[common].to_numpy())
            ys.extend(right.loc[common].to_numpy())
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.sum() < 4:
            continue
        r, p = stats.pearsonr(xs[ok], ys[ok])
        # Fisher z confidence interval.
        z = np.arctanh(r)
        se = 1 / np.sqrt(ok.sum() - 3)
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)

        icc = icc1(ts[["team_id", f]], "team_id", f)
        rows.append(
            {
                "Feature": f,
                "n team-season pairs": int(ok.sum()),
                "Year-to-year r": round(r, 3),
                "95% CI": f"[{lo:.2f}, {hi:.2f}]",
                "p": round(p, 4),
                "ICC (team)": round(icc, 3) if np.isfinite(icc) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def vector_distance_vs_null(ts: pd.DataFrame, n_perm: int = 10000):
    """Distance between a team's own consecutive profile vectors vs a null.

    The null pairs each team's season-t vector with a *different* team's
    season-t+1 vector. Significance is assessed by permuting the team labels of
    the later season and recomputing the mean distance, so the reference
    distribution is over mean distances rather than over individual pairs.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    Zmat = standardise_within_season(ts, PROFILE_FEATURES)
    Z = pd.DataFrame(Zmat, columns=PROFILE_FEATURES, index=ts.index)
    Z["team_id"] = ts["team_id"].to_numpy()
    Z["season_name"] = ts["season_name"].to_numpy()

    pairs = []
    for a, b in zip(SEASONS[:-1], SEASONS[1:]):
        left = Z[Z["season_name"] == a].set_index("team_id")[PROFILE_FEATURES]
        right = Z[Z["season_name"] == b].set_index("team_id")[PROFILE_FEATURES]
        common = list(left.index.intersection(right.index))
        if len(common) < 3:
            continue
        pairs.append((left.loc[common].to_numpy(), right.loc[common].to_numpy()))

    own = np.concatenate([
        np.linalg.norm(L - R, axis=1) for L, R in pairs
    ])
    null_pairwise = np.concatenate([
        np.linalg.norm(L[i] - R[j], axis=0)[None]
        for L, R in pairs
        for i in range(len(L)) for j in range(len(R)) if i != j
    ])

    observed = own.mean()
    perm_means = np.empty(n_perm)
    for p_i in range(n_perm):
        vals = []
        for L, R in pairs:
            idx = rng.permutation(len(R))
            vals.append(np.linalg.norm(L - R[idx], axis=1))
        perm_means[p_i] = np.concatenate(vals).mean()
    p_value = float(((perm_means <= observed).sum() + 1) / (n_perm + 1))
    return own, null_pairwise, p_value


def classify_trajectories(ts: pd.DataFrame) -> pd.DataFrame:
    """Multi-season trajectory typology (proposal 5.7)."""
    rows = []
    for team_id, g in ts.sort_values("season_name").groupby("team_id"):
        if len(g) < 3:
            continue
        cc = g["chance_creation_axis"].to_numpy()
        clusters = g["cluster"].to_numpy()
        diffs = np.diff(cc)
        span = cc.max() - cc.min()

        if np.all(diffs > 0.5):
            traj = "Progressive improvement"
        elif np.all(diffs < -0.5):
            traj = "Progressive decline"
        elif len(set(clusters)) == 1 and span < 1.5:
            traj = "Stable profile"
        elif np.sign(diffs[0]) != np.sign(diffs[-1]) and span >= 1.5:
            traj = "Oscillating profile"
        else:
            traj = "Stable profile" if span < 1.5 else "Oscillating profile"

        # A load-sensitive team drops its shot quality most under short rest.
        delta = g["short_rest_sq_delta"].mean()
        rows.append(
            {
                "Team": g["team_name"].iloc[0],
                "Seasons": len(g),
                "Chance-creation axis by season": ", ".join(f"{v:+.2f}" for v in cc),
                "Clusters": "->".join(str(c) for c in clusters),
                "Range on chance-creation axis": round(span, 2),
                "Mean short-rest shot-quality delta": round(delta, 4),
                "Trajectory": traj,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        thresh = out["Mean short-rest shot-quality delta"].quantile(0.25)
        out.loc[
            out["Mean short-rest shot-quality delta"] <= thresh, "Trajectory"
        ] += " (load-sensitive)"
    return out.sort_values("Team")


# --------------------------------------------------------------------------


def main() -> None:
    header("STEP 6  Temporal stability and transition analysis")
    plotstyle.apply()

    ts = load_frame("team_season_clustered")
    k = int(ts["cluster"].max()) + 1

    n_all3 = (ts.groupby("team_id").size() == 3).sum()
    print(f"  {ts['team_id'].nunique()} teams, {n_all3} present in all three seasons")

    # ---- categorical ------------------------------------------------------
    tr, counts, probs, retention, ari = transition_analysis(ts, k)
    v = cramers_v(counts)
    print(f"\n  consecutive-season transitions: n = {len(tr)}")
    print(f"  same-cluster retention rate: {retention:.3f}")
    print(f"  Cramer's V (from-cluster x to-cluster): {v:.3f}")
    print(f"  mean adjusted Rand index between consecutive seasons: {ari:.3f}")
    # Chance retention if transitions were independent of the origin cluster.
    marg = counts.sum(axis=0) / counts.sum()
    chance = float((counts.sum(axis=1) / counts.sum() * marg).sum())
    print(f"  retention expected by chance: {chance:.3f}")

    t5 = pd.DataFrame(counts, columns=[f"To cluster {c}" for c in range(k)])
    t5.insert(0, "From cluster", [f"Cluster {c}" for c in range(k)])
    for c in range(k):
        t5[f"P(to {c})"] = np.round(probs[:, c], 3)
    t5.loc[len(t5)] = (
        ["Summary"] + [""] * k
        + [f"retention {retention:.3f}", f"Cramer's V {v:.3f}", f"ARI {ari:.3f}"][:k]
    )
    save_table(t5, "table5_transition_matrix")

    # ---- continuous -------------------------------------------------------
    rep = repeatability(ts, REPEATABILITY_FEATURES)
    print("\n  season-to-season repeatability (profile axes first):")
    print(rep.head(8).to_string(index=False))
    save_table(rep, "table5b_profile_repeatability")

    own, null, p_null = vector_distance_vs_null(ts)
    print(
        f"\n  own-team profile distance {own.mean():.2f} (SD {own.std():.2f}) vs "
        f"different-team null {null.mean():.2f} (SD {null.std():.2f}); "
        f"permutation p = {p_null:.4f}"
    )

    traj = classify_trajectories(ts)
    print(f"\n  trajectory typology for {len(traj)} teams:")
    print(traj["Trajectory"].value_counts().to_string())
    save_table(traj, "table5c_trajectory_typology")

    make_transition_figure(ts, tr, counts, probs, k, own, null, rep)


def make_transition_figure(ts, tr, counts, probs, k, own, null, rep) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    import matplotlib.patches as patches

    fig = plt.figure(figsize=(plotstyle.W_DOUBLE, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.5, wspace=0.30)

    # --- (a) Sankey-style transition ribbons ------------------------------
    ax = fig.add_subplot(gs[0, 0])
    seasons = list(dict.fromkeys(list(tr["from_season"]) + list(tr["to_season"])))
    x_pos = np.linspace(0, 1, len(seasons))
    bar_w = 0.035
    node_h = {}

    for si, season in enumerate(seasons):
        sizes = ts[ts["season_name"] == season]["cluster"].value_counts().reindex(
            range(k), fill_value=0
        )
        total = max(sizes.sum(), 1)
        y = 0.0
        for c in range(k):
            h = sizes[c] / total
            node_h[(si, c)] = (y, h)
            ax.add_patch(
                patches.Rectangle(
                    (x_pos[si] - bar_w / 2, y), bar_w, h,
                    facecolor=plotstyle.CATEGORICAL[c], edgecolor="white", linewidth=0.8,
                    zorder=4,
                )
            )
            if h > 0.04:
                ax.text(
                    x_pos[si] - bar_w / 2 - 0.012, y + h / 2, f"C{c}",
                    ha="right", va="center", fontsize=6.5, color=plotstyle.INK_MUTED,
                )
            y += h + 0.012

    for si, (a, b) in enumerate(zip(seasons[:-1], seasons[1:])):
        sub = tr[(tr["from_season"] == a) & (tr["to_season"] == b)]
        cm = np.zeros((k, k), dtype=int)
        for _, r in sub.iterrows():
            cm[r["from_cluster"], r["to_cluster"]] += 1
        total = max(cm.sum(), 1)
        out_off = {c: node_h[(si, c)][0] for c in range(k)}
        in_off = {c: node_h[(si + 1, c)][0] for c in range(k)}
        for src in range(k):
            for dst in range(k):
                n = cm[src, dst]
                if n == 0:
                    continue
                h = n / total
                y0, y1 = out_off[src], in_off[dst]
                out_off[src] += h
                in_off[dst] += h
                x0 = x_pos[si] + bar_w / 2
                x1 = x_pos[si + 1] - bar_w / 2
                xm = (x0 + x1) / 2
                verts = [
                    (x0, y0), (xm, y0), (xm, y1), (x1, y1),
                    (x1, y1 + h), (xm, y1 + h), (xm, y0 + h), (x0, y0 + h), (x0, y0),
                ]
                codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                         Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
                ax.add_patch(
                    patches.PathPatch(
                        Path(verts, codes),
                        facecolor=plotstyle.CATEGORICAL[src],
                        alpha=0.42 if src != dst else 0.68,
                        edgecolor="none", zorder=2,
                    )
                )

    ax.set_xlim(-0.09, 1.05)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(seasons)
    ax.set_yticks([])
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title("a  Movement between profile clusters", loc="left")
    ax.text(
        0.02, 1.02,
        f"same-cluster retention {np.trace(counts) / counts.sum():.0%}",
        transform=ax.transAxes, fontsize=6.5, color=plotstyle.INK_MUTED, va="bottom",
    )

    # --- (b) continuous trajectories on the chance-creation axis ----------
    # With an 8-versus-40 partition, categorical retention is high almost by
    # construction, so the continuous axis carries the trajectory information.
    ax = fig.add_subplot(gs[0, 1])
    xs = np.arange(len(seasons))
    for team_id, g in ts.sort_values("season_name").groupby("team_id"):
        if len(g) < 2:
            continue
        idx = [seasons.index(s) for s in g["season_name"] if s in seasons]
        vals = g.loc[g["season_name"].isin(seasons), "chance_creation_axis"].to_numpy()
        if len(idx) != len(vals) or len(idx) < 2:
            continue
        col = plotstyle.CATEGORICAL[int(g["cluster"].iloc[-1])]
        ax.plot(idx, vals, color=col, lw=1.0, alpha=0.75, marker="o", markersize=3,
                markeredgecolor="white", markeredgewidth=0.4, zorder=3)
        ax.annotate(g["team_name"].iloc[-1], (idx[-1], vals[-1]),
                    textcoords="offset points", xytext=(4, 0), va="center",
                    fontsize=5.2, color=plotstyle.INK_MUTED)
    ax.axhline(0, color=plotstyle.INK_MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.set_xticks(xs)
    ax.set_xticklabels(seasons)
    ax.set_xlim(-0.15, len(seasons) - 1 + 1.35)
    ax.set_ylabel("Chance-creation axis")
    ax.set_title("b  Team trajectories on the chance-creation axis", loc="left")
    ax.grid(axis="y")

    # --- (c) own vs null profile distance ---------------------------------
    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, max(null.max(), own.max()), 34)
    ax.hist(null, bins=bins, density=True, color=plotstyle.INK_MUTED, alpha=0.35,
            label="Different teams (null)", edgecolor="white", linewidth=0.3)
    ax.hist(own, bins=bins, density=True, color=plotstyle.CATEGORICAL[0], alpha=0.85,
            label="Same team, consecutive seasons", edgecolor="white", linewidth=0.3)
    ax.axvline(own.mean(), color=plotstyle.CATEGORICAL[0], lw=1.4)
    ax.axvline(null.mean(), color=plotstyle.INK_MUTED, lw=1.4, ls=(0, (4, 3)))
    ax.set_xlabel("Euclidean distance between standardised profile vectors")
    ax.set_ylabel("Density")
    ax.set_title("c  Profile persistence", loc="left")
    ax.legend(loc="upper right", fontsize=6.5)

    # --- (d) per-feature repeatability ------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    rr = rep[rep["Feature"].isin(AXES + PROFILE_FEATURES)].copy()
    rr = rr.sort_values("Year-to-year r").tail(12)
    ypos = np.arange(len(rr))
    colors = [
        plotstyle.CATEGORICAL[1] if f in AXES else plotstyle.CATEGORICAL[0]
        for f in rr["Feature"]
    ]
    ax.barh(ypos, rr["Year-to-year r"], color=colors, height=0.7,
            edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=plotstyle.INK_MUTED, lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f.replace("_", " ") for f in rr["Feature"]], fontsize=6.3)
    ax.set_xlabel("Year-to-year correlation")
    ax.set_title("d  Repeatability by feature", loc="left")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    plotstyle.save(fig, "figure6_profile_transitions")


if __name__ == "__main__":
    main()
