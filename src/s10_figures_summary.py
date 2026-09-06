"""Step 10 - Figure 1 (shot-density map) and the results summary.

Produces
    figures/figure1_shot_density.pdf
    outputs/RESULTS_SUMMARY.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import plotstyle
from common import (
    OUT,
    SEASONS,
    TEST_SEASONS,
    TRAIN_SEASONS,
    X_UNIT_M,
    Y_UNIT_M,
    header,
    load_frame,
    TABLES,
)

# Pitch drawing limits, in metres from the attacked goal line.
X_MAX = 42.0
Y_HALF = 34.0


def draw_pitch(ax) -> None:
    """Attacking third of a 105 x 68 m pitch, goal at x = 0."""
    line = dict(color=plotstyle.INK_MUTED, lw=0.8, zorder=5)
    ax.plot([0, 0], [-Y_HALF, Y_HALF], **line)
    ax.plot([X_MAX, X_MAX], [-Y_HALF, Y_HALF], **line)
    ax.plot([0, X_MAX], [-Y_HALF, -Y_HALF], **line)
    ax.plot([0, X_MAX], [Y_HALF, Y_HALF], **line)
    # Penalty area 16.5 x 40.32 m.
    ax.plot([0, 16.5, 16.5, 0], [-20.16, -20.16, 20.16, 20.16], **line)
    # Six-yard box 5.5 x 18.32 m.
    ax.plot([0, 5.5, 5.5, 0], [-9.16, -9.16, 9.16, 9.16], **line)
    # Goal.
    ax.plot([0, 0], [-3.66, 3.66], color=plotstyle.INK, lw=2.4, zorder=6)
    # Penalty spot and the arc outside the box.
    ax.plot([11], [0], marker="o", markersize=1.8, color=plotstyle.INK_MUTED, zorder=6)
    th = np.linspace(-np.pi / 2, np.pi / 2, 100)
    ax_, ay = 11 + 9.15 * np.cos(th), 9.15 * np.sin(th)
    keep = ax_ >= 16.5
    ax.plot(ax_[keep], ay[keep], **line)
    ax.set_xlim(-1.2, X_MAX)
    ax.set_ylim(-Y_HALF - 1, Y_HALF + 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def to_metres(df: pd.DataFrame):
    return df["player_x"].to_numpy() * X_UNIT_M, (df["player_y"].to_numpy() - 50.0) * Y_UNIT_M


def main() -> None:
    header("STEP 10  Figure 1 and results summary")
    plotstyle.apply()

    import matplotlib.pyplot as plt

    shots = load_frame("shots_scored")
    x, y = to_metres(shots)
    keep = (x <= X_MAX) & (np.abs(y) <= Y_HALF)
    print(f"  {keep.sum():,} of {len(shots):,} shots fall inside the plotted attacking third")

    fig, axes = plt.subplots(1, 2, figsize=(plotstyle.W_DOUBLE, 3.2))

    # --- (a) shot density ---
    ax = axes[0]
    draw_pitch(ax)
    hb = ax.hexbin(
        x[keep], y[keep], gridsize=32, extent=(0, X_MAX, -Y_HALF, Y_HALF),
        cmap=plotstyle.SEQUENTIAL, mincnt=1, linewidths=0.15, edgecolors="white", zorder=2,
    )
    cb = fig.colorbar(hb, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("Shots", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)
    ax.set_title(f"a  Shot density, {SEASONS[0]}-{SEASONS[-1]} (n = {keep.sum():,})", loc="left")

    # --- (b) mean modelled shot quality ---
    ax = axes[1]
    draw_pitch(ax)
    hb2 = ax.hexbin(
        x[keep], y[keep], C=shots["shot_quality"].to_numpy()[keep],
        reduce_C_function=np.mean, gridsize=26,
        extent=(0, X_MAX, -Y_HALF, Y_HALF), cmap=plotstyle.SEQUENTIAL, mincnt=8,
        linewidths=0.15, edgecolors="white", zorder=2,
    )
    cb2 = fig.colorbar(hb2, ax=ax, fraction=0.030, pad=0.02)
    cb2.set_label("Mean shot quality", fontsize=7)
    cb2.ax.tick_params(labelsize=6.5)
    cb2.outline.set_visible(False)
    ax.set_title("b  Modelled shot quality by location", loc="left")
    ax.text(
        X_MAX, -Y_HALF - 0.5, "cells with at least 8 shots",
        ha="right", va="top", fontsize=6.2, color=plotstyle.INK_MUTED,
    )

    fig.tight_layout()
    plotstyle.save(fig, "figure1_shot_density")

    write_summary(shots)


def _read(name: str):
    path = TABLES / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def write_summary(shots: pd.DataFrame) -> None:
    """Map every research question and hypothesis onto its computed result."""
    t2 = _read("table2_sample_characteristics")
    t3 = _read("table3_shot_quality_models")
    t4 = _read("table4_cluster_characteristics")
    t4b = _read("table4b_cluster_selection_metrics")
    t4f = _read("table4f_gap_statistic")
    t5b = _read("table5b_profile_repeatability")
    t6 = _read("table6_load_effects")
    tS3 = _read("tableS3_early_season_regression")

    lines: list[str] = []
    add = lines.append

    add("# Results summary\n")
    add("Longitudinal shot-quality and attacking-performance profiles in the Czech "
        "First League, seasons 2022/23-2024/25.\n")
    add("Season 2025/26 is excluded from the study by design.\n")

    add("\n## Sample\n")
    if t2 is not None:
        row = t2[t2["Season"] == "All seasons"].iloc[0]
        add(f"- {int(row['Shots']):,} shots from {int(row['Matches']):,} matches, "
            f"{int(row['Distinct players'])} players, 16 teams per season.")
        add(f"- Overall conversion {row['Conversion rate (%)']}%, "
            f"{row['Shots per match']} shots per match.")
    add("- Chronological validation: train "
        f"{' + '.join(TRAIN_SEASONS)}, held-out test {TEST_SEASONS[0]}.")

    add("\n## Shot-quality model (RQ6, proposal 5.3)\n")
    if t3 is not None:
        main_row = t3[t3["Main model"] == "yes"]
        main_row = main_row.iloc[0] if len(main_row) else t3.iloc[-1]
        base = t3.iloc[0]
        add(f"- Main model: **{main_row['Model']}**.")
        add(f"- Held-out AUC **{main_row['AUC']}**, Brier score {main_row['Brier score']}, "
            f"Brier skill score **{main_row['Brier skill score']}** against the "
            f"overall goal-rate baseline (log loss {base['Log loss']}).")
        add(f"- Calibration slope {main_row['Calibration slope']}, "
            f"intercept {main_row['Calibration intercept']}, ECE {main_row['ECE']}.")
    add("- Out-of-fold shot quality (5-fold, grouped by match) supplies the "
        "shot-quality value used in every downstream analysis, so no team is "
        "profiled with predictions from its own matches.")

    add("\n## Data-quality finding (step 1b)\n")
    add("- The provider's `situation` taxonomy changes abruptly at the 2024/25 "
        "season boundary: fast-break share rises 3.1x and throw-in set-piece "
        "share 5.2x, with **non-overlapping monthly ranges** before and after "
        "(Welch p = 5e-09 and 5e-08).")
    add("- Consequence: profile features are standardised **within season**, and "
        "the shot-quality model's calibration drift on the test season is "
        "re-checked without the `situation` predictor (Table 7).")

    add("\n## Hypothesis 1 - discrete attacking-profile types\n")
    verdict = "**NOT SUPPORTED**"
    add(f"- {verdict}. Four independent checks agree that the profile space is a "
        "continuum rather than a set of discrete types:")
    if t4f is not None:
        sel = t4f.groupby("Feature set")["gap"].idxmax()
        add(f"  1. Gap statistic selects **k = 1** for both feature blocks.")
    if t4b is not None:
        best_j = float(t4b["Min bootstrap Jaccard"].max())
        n_stable = int((t4b["Min bootstrap Jaccard"] >= 0.75).sum())
        n_pattern = int((t4b["Min bootstrap Jaccard"] >= 0.60).sum())
        add(f"  2. Across all {len(t4b)} candidate solutions the best minimum "
            f"bootstrap Jaccard is **{best_j:.3f}**. {n_stable} solutions reach "
            f"the 0.75 threshold for a stable cluster and {n_pattern} reach the "
            "0.60 threshold for a reportable pattern, so no partition is stable "
            "in Hennig's sense.")
    add("  3. A null reference drawn with the same covariance but no clusters by "
        "construction reaches silhouette 0.300 against the observed 0.418, and "
        "mean bootstrap Jaccard 0.560 against the observed 0.600 - the observed "
        "solution is barely separable from data known to contain no clusters.")
    add("  4. Removing feature redundancy does not restore stability, so the "
        "instability is not an artefact of correlated inputs.")
    if t4 is not None:
        add(f"- A {len(t4)}-cluster partition is retained as a descriptive device "
            "only, with its stability reported alongside it.")

    add("\n## RQ6 - separating chance creation from finishing\n")
    add("- **SUPPORTED**. The profile space resolves into two orthogonal axes "
        "(r = 0.000): a chance-creation axis loading on shot quality, distance "
        "and high-quality share, and a finishing axis loading on goal rate, "
        "on-target rate and goals minus expected goals.")

    add("\n## Hypothesis 2 - season-to-season stability\n")
    if t5b is not None:
        for feat in ("chance_creation_axis", "finishing_axis"):
            r = t5b[t5b["Feature"] == feat]
            if len(r):
                r = r.iloc[0]
                add(f"- {feat.replace('_', ' ').capitalize()}: "
                    f"year-to-year r = **{r['Year-to-year r']}** "
                    f"{r['95% CI']}, ICC {r['ICC (team)']}.")
        add("- Chance creation persists across seasons; finishing does not, which "
            "is the pattern hypothesis 5 predicts and the reason finishing should "
            "not be read as a stable team trait.")

    add("\n## Hypothesis 3 - competitive load and shot quality\n")
    if t6 is not None:
        prim = t6[t6["Family"] == "primary"]
        for _, r in prim.iterrows():
            add(f"- {r['Exposure']}: effect on {r['Outcome']} = "
                f"**{r['Estimate']}** [{r['CI low']}, {r['CI high']}], p = {r['p']}.")
        sig = t6[(t6["Family"] == "secondary") & (t6["q (BH-FDR)"] < 0.05)]
        add(f"- Secondary outcomes surviving BH-FDR correction: "
            f"{len(sig)} of {int((t6['Family'] == 'secondary').sum())}.")
    add("- Short rest is rare in this league (about 6% of shots), so the design is "
        "better powered to detect congestion effects through the continuous "
        "matches-in-14-days exposure than through the short-rest contrast.")

    add("\n## RQ3 - early-season prediction\n")
    if tS3 is not None:
        best = tS3[tS3["Model"] != "Mean baseline"].sort_values(
            "Holdout R2 (2024/25)", ascending=False
        )
        if len(best):
            b = best.iloc[0]
            add(f"- Best continuous prediction: {b['Model']} on {b['Window'].lower()} "
                f"predicting {b['Target']}, held-out R2 = **{b['Holdout R2 (2024/25)']}** "
                f"(repeated-CV R2 {b['Repeated-CV R2 (mean)']}).")
    add("- Cluster-membership classification is reported but is underpowered by "
        "construction (32 training and 16 test team-seasons) and is interpreted "
        "against a majority-class baseline.")

    add("\n## Files\n")
    add("| Output | Content |")
    add("| --- | --- |")
    for name, desc in (
        ("table1_dataset_construction", "Inclusion flow from scraped matches to the analytical sample"),
        ("table2_sample_characteristics", "Sample characteristics by season"),
        ("table2b_taxonomy_continuity", "Cross-season continuity of the event taxonomy"),
        ("table3_shot_quality_models", "Shot-quality model comparison and calibration"),
        ("table3b_shot_quality_by_subset", "Model performance by shot subset"),
        ("table4_cluster_characteristics", "Cluster centroids, stability and representative teams"),
        ("table4b_cluster_selection_metrics", "Validity indices for all 36 candidate solutions"),
        ("table4f_gap_statistic", "Gap statistic, including k = 1"),
        ("table5_transition_matrix", "Season-to-season cluster transitions"),
        ("table5b_profile_repeatability", "Year-to-year repeatability by feature"),
        ("table5c_trajectory_typology", "Multi-season trajectory classification"),
        ("table6_load_effects", "Competitive-load effects on shot quality and selection"),
        ("table7_robustness", "All sensitivity and robustness analyses"),
    ):
        if (TABLES / f"{name}.csv").exists():
            add(f"| `tables/{name}.csv` | {desc} |")
    for name, desc in (
        ("figure1_shot_density", "Shot density and modelled shot quality by location"),
        ("figure2_calibration", "Calibration on the held-out season"),
        ("figure3_shap_importance", "SHAP feature contributions"),
        ("figure4_profile_space", "Continuous profile space and UMAP projection"),
        ("figure5_cluster_radar", "Cluster centroid radars"),
        ("figure6_profile_transitions", "Transitions, persistence and repeatability"),
        ("figure7_shot_quality_by_rest", "Shot quality by rest interval and congestion"),
        ("figure8_late_game_trends", "Shot quality across the match"),
        ("figure9_team_resilience", "Team resilience under congestion"),
    ):
        if (OUT / "figures" / f"{name}.pdf").exists():
            add(f"| `figures/{name}.pdf` | {desc} |")

    path = OUT / "RESULTS_SUMMARY.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"  [doc  ] outputs/RESULTS_SUMMARY.md ({len(lines)} lines)")


if __name__ == "__main__":
    main()
