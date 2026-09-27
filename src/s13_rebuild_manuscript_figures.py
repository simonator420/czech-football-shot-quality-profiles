"""Rebuild the five manuscript figures with readable submission spacing.

This presentation-only step uses saved analytical data and tables. It writes
matching PNG, PDF and SVG files under ``outputs/figures``; generated figures
are intentionally excluded from version control.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from statsmodels.formula.api import ols

import plotstyle
from common import SEASONS, X_UNIT_M, Y_UNIT_M, load_frame
from s10_figures_summary import (
    HEX_GRID_DENSITY,
    HEX_GRID_QUALITY,
    QUALITY_MINCNT,
    X_MAX,
    Y_HALF,
    draw_pitch,
)
from s12_core_validation import add_process_axis, team_results


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"

SEQ_READABLE = LinearSegmentedColormap.from_list(
    "shotq_readable", ["#edf4f8", "#c6dbef", "#6baed6", "#2878a7", "#17365d"]
)


def save(fig, stem: str, *, tight: bool = True) -> None:
    # ``None`` would fall back to the global savefig.bbox="tight" setting.
    # Use the explicit full-canvas box when exact visual centring matters.
    bbox = "tight" if tight else fig.bbox_inches
    for ext in ("svg", "pdf", "png"):
        fig.savefig(
            FIGURES / f"{stem}.{ext}",
            dpi=400 if ext == "png" else None,
            bbox_inches=bbox,
            pad_inches=0.16,
            facecolor="white",
        )


def figure1(shots: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    x = shots["player_x"].to_numpy() * X_UNIT_M
    y = (shots["player_y"].to_numpy() - 50.0) * Y_UNIT_M
    q = shots["shot_quality"].to_numpy()
    keep = (x <= X_MAX) & (np.abs(y) <= Y_HALF)
    fig, axes = plt.subplots(1, 2, figsize=(8.15, 4.55))

    draw_pitch(axes[0])
    hb = axes[0].hexbin(
        x[keep], y[keep], gridsize=HEX_GRID_DENSITY,
        extent=(0, X_MAX, -Y_HALF, Y_HALF), cmap=SEQ_READABLE,
        norm=PowerNorm(gamma=0.40, vmin=1), mincnt=1,
        linewidths=0.12, edgecolors="white", zorder=2,
    )
    cb = fig.colorbar(hb, ax=axes[0], fraction=0.034, pad=0.035)
    cb.set_label("Shots", labelpad=10)
    cb.outline.set_visible(False)
    axes[0].set_title(f"Shot density, {SEASONS[0]}-{SEASONS[-1]}", pad=10)

    draw_pitch(axes[1])
    hb = axes[1].hexbin(
        x[keep], y[keep], C=q[keep], reduce_C_function=np.mean,
        gridsize=HEX_GRID_QUALITY, extent=(0, X_MAX, -Y_HALF, Y_HALF),
        cmap=SEQ_READABLE, norm=PowerNorm(gamma=0.48, vmin=0.02, vmax=0.62),
        mincnt=QUALITY_MINCNT, linewidths=0.12, edgecolors="white", zorder=2,
    )
    cb = fig.colorbar(hb, ax=axes[1], fraction=0.034, pad=0.035)
    cb.set_label("Mean modelled scoring probability", labelpad=10)
    cb.outline.set_visible(False)
    axes[1].set_title("Modelled shot quality by location", pad=10)

    fig.suptitle("Spatial distribution of the analytical shot sample", x=0.5, y=0.965,
                 ha="center", fontsize=10.5, fontweight="bold", color=plotstyle.INK)
    # Equal-aspect pitch axes occupy only the centre of their subplot slots.
    # Shift the two-panel block left to balance the outer whitespace created by
    # the two colour bars and their right-side labels.
    fig.subplots_adjust(left=-0.063, right=0.867, bottom=0.065, top=0.81, wspace=0.25)
    save(fig, "manuscript_figure1_spatial_distribution", tight=False)
    plt.close(fig)


def calibration_arrays(shots: pd.DataFrame):
    test = shots[(shots["season_name"] == "2024/25") & shots["shot_quality_holdout"].notna()]
    y = test["goal"].to_numpy(dtype=float)
    p = test["shot_quality_holdout"].to_numpy(dtype=float)
    order = np.argsort(p)
    y, p = y[order], p[order]
    bins = np.array_split(np.arange(len(p)), 10)
    obs = np.array([y[b].mean() for b in bins])
    pred = np.array([p[b].mean() for b in bins])
    ns = np.array([len(b) for b in bins])
    z = 1.96
    den = 1 + z**2 / ns
    centre = (obs + z**2 / (2 * ns)) / den
    half = z * np.sqrt(obs * (1 - obs) / ns + z**2 / (4 * ns**2)) / den
    return p, pred, obs, centre - half, centre + half


def figure2(shots: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    _, pred, obs, lo, hi = calibration_arrays(shots)
    fig, ax = plt.subplots(figsize=(6.6, 4.35))
    lim = max(pred.max(), obs.max()) * 1.10
    ax.plot([0, lim], [0, lim], color=plotstyle.INK_MUTED, lw=0.9,
            ls=(0, (4, 3)), label="Perfect calibration")
    ax.errorbar(
        pred, obs, yerr=[obs - lo, hi - obs], fmt="o",
        color=plotstyle.BLU, ecolor=plotstyle.BLU, markersize=4.8,
        elinewidth=1, capsize=2.5, markeredgecolor="white", markeredgewidth=0.6,
        label="Observed rate (95% Wilson CI)",
    )
    ax.set(xlabel="Predicted shot quality (equal-frequency bin mean)",
           ylabel="Observed goal proportion", xlim=(0, lim), ylim=(0, lim))
    ax.set_title("Calibration in held-out 2024/25", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              fontsize=7.3, columnspacing=1.7, handlelength=2.5)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.25, top=0.88)
    save(fig, "manuscript_figure2_calibration")
    plt.close(fig)


def figure3(ts: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.75))
    ax.axhline(0, color=plotstyle.GRID, lw=0.9)
    ax.axvline(0, color=plotstyle.GRID, lw=0.9)
    for i, season in enumerate(SEASONS):
        s = ts[ts["season_name"] == season]
        ax.scatter(s["chance_creation_axis"], s["finishing_axis"], s=38,
                   color=plotstyle.CATEGORICAL[i], edgecolor="white", linewidth=0.6,
                   label=season, zorder=3)
    ax.set_xlabel("PC1 - Chance creation (42.9% variance explained)")
    ax.set_ylabel("PC2 - Finishing and shot outcome (16.5% variance explained)")
    ax.set_title("Continuous team-season attacking-profile space", pad=11)
    ax.grid(axis="both")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3,
              columnspacing=1.8, handletextpad=0.6)
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.25, top=0.85)
    save(fig, "manuscript_figure3_profile_space")
    plt.close(fig)


def pretty_feature(value: str) -> str:
    return {
        "xg_per_match": "xG per match",
        "shots_per_match": "Shots per match",
        "goals_per_match": "Goals per match",
        "chance_creation_axis": "Chance-creation axis",
        "mean_shot_quality": "Mean shot quality",
        "high_quality_share": "High-quality share",
        "finishing_axis": "Finishing axis",
        "goals_minus_xg_per_match": "Goals minus xG per match",
        "on_target_rate": "On-target rate",
    }.get(value, value.replace("_", " ").title())


def figure4() -> None:
    import matplotlib.pyplot as plt

    stab = pd.read_csv(TABLES / "tableS2_stabilisation_curves.csv")
    fig, ax = plt.subplots(figsize=(6.7, 4.45))

    order = ["xg_per_match", "chance_creation_axis", "goals_minus_xg_per_match", "finishing_axis"]
    for i, feat in enumerate(order):
        s = stab[stab["Feature"] == feat].sort_values("Matches")
        ax.plot(s["Matches"], s["Pearson r with remainder"],
                marker=plotstyle.MARKERS[i], color=plotstyle.CATEGORICAL[i],
                markeredgecolor="white", markeredgewidth=0.5,
                label=pretty_feature(feat))
        if feat in ("xg_per_match", "chance_creation_axis") and \
                "Pearson 95% team-cluster bootstrap CI lower" in s:
            ax.fill_between(
                s["Matches"].to_numpy(),
                s["Pearson 95% team-cluster bootstrap CI lower"].to_numpy(),
                s["Pearson 95% team-cluster bootstrap CI upper"].to_numpy(),
                color=plotstyle.CATEGORICAL[i], alpha=0.12, linewidth=0,
            )
    ticks = sorted(stab["Matches"].unique().astype(int))
    ax.set_xticks(ticks)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Matches elapsed")
    ax.set_ylabel("Pearson r with non-overlapping remainder")
    ax.set_title("Stabilisation across the season", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=2,
              columnspacing=1.4, handlelength=2.0)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.25, top=0.88)
    save(fig, "manuscript_figure4_stabilisation")
    plt.close(fig)


def process_data(ts: pd.DataFrame, matches: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    # Use the same scoring function as the inferential analysis so the plotted
    # coordinates and reported estimates cannot silently diverge.
    return add_process_axis(ts).merge(
        team_results(matches, shots),
        on=["team_id", "season_name", "team_name"],
    )


def figure5(d: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.15, 3.85))
    outcomes = [("points_per_match", "Points per match"),
                ("goal_difference_per_match", "Goal difference per match")]
    for ax, (y, ylabel) in zip(axes, outcomes):
        for i, season in enumerate(SEASONS):
            s = d[d["season_name"] == season]
            ax.scatter(s["process_chance_creation_axis"], s[y], s=34,
                       color=plotstyle.CATEGORICAL[i], edgecolor="white", linewidth=0.55,
                       label=season, zorder=3)
        fit = ols(f"{y} ~ process_chance_creation_axis", d).fit()
        xs = np.linspace(d["process_chance_creation_axis"].min(),
                         d["process_chance_creation_axis"].max(), 150)
        pred = fit.get_prediction(pd.DataFrame({"process_chance_creation_axis": xs})).summary_frame()
        ax.plot(xs, pred["mean"], color=plotstyle.INK, lw=1.2)
        ax.fill_between(xs, pred["mean_ci_lower"], pred["mean_ci_upper"],
                        color=plotstyle.INK_MUTED, alpha=0.17, linewidth=0)
        ax.set_xlabel("Process-based chance-creation axis (PCA)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, pad=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.035),
               ncol=3, columnspacing=1.8, handletextpad=0.6)
    fig.suptitle("Concurrent practical validity of the process-based axis", y=0.985,
                 fontsize=10.5, fontweight="bold", color=plotstyle.INK)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.26, top=0.77, wspace=0.28)
    save(fig, "manuscript_figure5_practical_validity")
    plt.close(fig)


def main() -> None:
    plotstyle.apply()
    shots = load_frame("shots_scored")
    ts = load_frame("team_season_clustered")
    matches = load_frame("matches")
    figure1(shots)
    figure2(shots)
    figure3(ts)
    figure4()
    figure5(process_data(ts, matches, shots))


if __name__ == "__main__":
    main()
