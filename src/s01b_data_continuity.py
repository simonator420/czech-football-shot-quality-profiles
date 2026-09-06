"""Step 1b - Cross-season continuity of the event-data taxonomy.

Multi-season event-data studies assume the provider annotates events the same
way in every season. That assumption is testable, and here it fails: the
distribution of `situation` labels changes abruptly at the 2024/25 season
boundary rather than drifting, which is the signature of a provider taxonomy
revision rather than of tactical evolution in the league.

Quantifying the break matters for three downstream decisions:

  * profile features are standardised *within season*, so a team's profile is
    expressed relative to its own league-season context and a league-wide
    annotation shift cannot masquerade as team development;
  * the shot-quality model's chronological split crosses the break, so its
    calibration drift on the test season is interpreted in that light and
    re-checked without the affected predictor;
  * the break is reported as a limitation and as a warning for other users of
    the same public data source.

Produces
    tables/table2b_taxonomy_continuity.csv
    figures/figureS3_taxonomy_break.pdf
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import plotstyle
from common import SEASONS, header, load_frame, save_table

#: Variables whose season-to-season stability is checked.
SITUATION_LABELS = [
    "assisted", "corner", "open_play", "set_piece", "throw_in",
    "free_kick", "fast_break", "penalty",
]
SPATIAL_CHECKS = {
    "mean_distance_m": ("distance_m", "mean"),
    "close_range_share": ("distance_m", lambda s: (s < 11).mean()),
    "inside_box_share": ("inside_box", "mean"),
    "header_share": ("is_header", "mean"),
}


def main() -> None:
    header("STEP 1b  Cross-season continuity of the event-data taxonomy")
    plotstyle.apply()

    d = load_frame("shots_scored")
    d["kickoff"] = pd.to_datetime(d["start_datetime_utc"])

    rows = []
    for lab in SITUATION_LABELS:
        share = d.groupby("season_name")["situation"].apply(lambda s: (s == lab).mean())
        rec = {"Variable": f"situation = {lab}", "Type": "Technical annotation"}
        for s in SEASONS:
            rec[s] = round(float(share.get(s, np.nan)), 4)
        base = share.reindex(SEASONS[:2]).mean()
        last = share.get(SEASONS[-1], np.nan)
        rec["Ratio 2024/25 : earlier seasons"] = (
            round(float(last / base), 2) if base and base > 0 else np.nan
        )
        rows.append(rec)

    for name, (col, fn) in SPATIAL_CHECKS.items():
        vals = d.groupby("season_name")[col].agg(fn)
        rec = {"Variable": name, "Type": "Spatial / derived"}
        for s in SEASONS:
            rec[s] = round(float(vals.get(s, np.nan)), 4)
        base = vals.reindex(SEASONS[:2]).mean()
        rec["Ratio 2024/25 : earlier seasons"] = round(float(vals.get(SEASONS[-1]) / base), 2)
        rows.append(rec)

    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))
    save_table(tbl, "table2b_taxonomy_continuity")

    flagged = tbl[
        (tbl["Ratio 2024/25 : earlier seasons"] > 1.5)
        | (tbl["Ratio 2024/25 : earlier seasons"] < 0.67)
    ]
    print(f"\n  {len(flagged)} variables shift by more than 50% at the season boundary:")
    print(flagged[["Variable", "Ratio 2024/25 : earlier seasons"]].to_string(index=False))

    # --- is the change a step or a trend? ---
    monthly = d.set_index("kickoff").sort_index()
    m = monthly.resample("ME").agg(
        n=("goal", "size"),
        fast_break=("is_fast_break", "mean"),
        assisted=("is_assisted", "mean"),
        distance=("distance_m", "mean"),
    )
    m["throw_in"] = monthly.resample("ME")["situation"].apply(lambda s: (s == "throw_in").mean())
    m = m[m["n"] >= 100]

    boundary = pd.Timestamp("2024-07-01")
    print("\n  step test (mean before vs after the 2024/25 season boundary):")
    for c in ("fast_break", "throw_in", "assisted", "distance"):
        before = m.loc[m.index < boundary, c]
        after = m.loc[m.index >= boundary, c]
        from scipy import stats

        t, p = stats.ttest_ind(before, after, equal_var=False)
        print(
            f"    {c:<12} before {before.mean():.4f} (SD {before.std():.4f}), "
            f"after {after.mean():.4f} (SD {after.std():.4f}), "
            f"Welch p = {p:.2e}, non-overlapping ranges: "
            f"{'yes' if before.max() < after.min() or after.max() < before.min() else 'no'}"
        )

    make_figure(m, boundary)


def make_figure(m: pd.DataFrame, boundary) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    series = [
        ("fast_break", "Fast-break share"),
        ("throw_in", "Throw-in set-piece share"),
        ("assisted", "Assisted share"),
        ("distance", "Mean shot distance (m)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(plotstyle.W_DOUBLE, 4.2), sharex=True)
    axes = axes.ravel()
    for i, (col, label) in enumerate(series):
        ax = axes[i]
        ax.plot(m.index, m[col], color=plotstyle.CATEGORICAL[i],
                marker=plotstyle.MARKERS[i], markersize=3,
                markeredgecolor="white", markeredgewidth=0.4)
        ax.axvline(boundary, color=plotstyle.INK_MUTED, lw=1.0, ls=(0, (4, 3)))
        ax.set_ylabel("")
        ax.set_title(label, loc="left", x=0.0, fontsize=8.5, pad=6)
        if col != "distance":
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(axis="y")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", labelbottom=True)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(0)
            lbl.set_fontsize(6.5)
    boundary_handle = Line2D(
        [0], [0], color=plotstyle.INK_MUTED, lw=1.0, ls=(0, (4, 3)),
        label="2024/25 season boundary",
    )
    fig.legend(handles=[boundary_handle], loc="lower center",
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    plotstyle.save(fig, "figureS3_taxonomy_break")


if __name__ == "__main__":
    main()
