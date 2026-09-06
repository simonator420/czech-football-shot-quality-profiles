"""Step 7 - Competitive-load analysis (proposal 5.8).

Hypothesis 3 predicts that short rest and fixture congestion depress shot
quality, push shots further out and to wider angles, and raise the share of
blocked or low-quality attempts. Hypothesis 5 predicts that between-team
differences under load show up in shot selection rather than in finishing.

Design decisions that matter for interpretation:

  * The Czech First League runs a ten-week winter break, so raw rest intervals
    reach 78 days. Break fixtures are flagged and the continuous rest variable
    is winsorised at 14 days, otherwise the "long rest" contrast would be
    dominated by teams returning from a competition break rather than by
    recovery inside a congested schedule.
  * Shots are nested in team-matches and team-matches in teams. Continuous
    outcomes use mixed models with a team random intercept; binary outcomes use
    GEE with an exchangeable working correlation clustered on the match, which
    gives cluster-robust standard errors without assuming a variance structure
    the data cannot support.
  * Effects are reported as associations. Load is measured through calendar
    proxies, not physiological monitoring, and the design is observational.
  * The secondary-outcome family is corrected for multiple testing with the
    Benjamini-Hochberg false discovery rate.

Produces
    tables/table6_load_effects.csv
    tables/table6b_calendar_definitions.csv
    tables/table6c_european_load.csv
    figures/figure7_shot_quality_by_rest.pdf
    figures/figure8_late_game_trends.pdf
    figures/figure9_team_resilience.pdf
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

import plotstyle
from common import header, load_frame, save_table

warnings.filterwarnings("ignore")

EPS = 1e-6

#: Primary and secondary outcomes (proposal 5.8).
CONTINUOUS_OUTCOMES = {
    "logit_shot_quality": "Shot quality (logit)",
    "distance_m": "Shot distance (m)",
    "angle_proxy": "Shot angle proxy (rad)",
}
BINARY_OUTCOMES = {
    "on_target": "Shot on target",
    "blocked": "Shot blocked",
    "goal": "Goal",
    "high_quality": "High-quality shot (SQ >= 0.15)",
    "low_quality": "Low-quality shot (SQ < 0.05)",
}

#: Load exposures of interest, each tested in its own model.
EXPOSURES = {
    "short_rest": "Short rest (<= 3 days vs >= 4)",
    "matches_last_14_c": "Matches in previous 14 days (per match)",
    "days_since_any_capped_c": "Days since previous fixture (per day, capped at 14)",
}

BASE_CONTROLS = "minute_c + C(score_state) + is_home_shot + C(season_name) + post_break"


def prepare(shots: pd.DataFrame) -> pd.DataFrame:
    d = shots.copy()
    d = d[d["rest_category"] != "unknown"].copy()
    d["logit_shot_quality"] = np.log(
        np.clip(d["shot_quality"], EPS, 1 - EPS) / (1 - np.clip(d["shot_quality"], EPS, 1 - EPS))
    )
    d["high_quality"] = (d["shot_quality"] >= 0.15).astype(int)
    d["low_quality"] = (d["shot_quality"] < 0.05).astype(int)
    d["minute_c"] = (d["minute"] - d["minute"].mean()) / 10.0
    d["matches_last_14_c"] = d["matches_last_14"] - d["matches_last_14"].mean()
    d["days_since_any_capped_c"] = (
        d["days_since_any_capped"] - d["days_since_any_capped"].mean()
    )
    d["team_id"] = d["team_id"].astype(int)
    d["opponent_team_id"] = d["opponent_team_id"].astype(int)
    return d.dropna(subset=["days_since_any_capped_c", "matches_last_14_c"])


# --------------------------------------------------------------------------
# Model fitting
# --------------------------------------------------------------------------


def fit_continuous(d: pd.DataFrame, outcome: str, exposure: str, controls: str = BASE_CONTROLS):
    """Linear mixed model with a team random intercept."""
    f = f"{outcome} ~ {exposure} + {controls}"
    m = smf.mixedlm(f, d, groups=d["team_id"]).fit(reml=True, method="lbfgs")
    return _extract(m, exposure, n=len(d), model="LMM (team random intercept)")


def fit_binary(d: pd.DataFrame, outcome: str, exposure: str, controls: str = BASE_CONTROLS):
    """GEE logistic model with exchangeable correlation clustered on the match."""
    f = f"{outcome} ~ {exposure} + {controls}"
    m = smf.gee(
        f, groups="match_id", data=d,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    return _extract(m, exposure, n=len(d), model="GEE logistic (clustered on match)", exp=True)


def _extract(res, term: str, n: int, model: str, exp: bool = False) -> dict:
    key = next((p for p in res.params.index if p.startswith(term)), None)
    if key is None:
        return {}
    beta = float(res.params[key])
    se = float(res.bse[key])
    lo, hi = beta - 1.96 * se, beta + 1.96 * se
    p = float(res.pvalues[key])
    out = {
        "Estimate": np.exp(beta) if exp else beta,
        "CI low": np.exp(lo) if exp else lo,
        "CI high": np.exp(hi) if exp else hi,
        "p": p,
        "N": n,
        "Model": model,
        "Scale": "odds ratio" if exp else "units of outcome",
    }
    return out


# --------------------------------------------------------------------------


def main() -> None:
    header("STEP 7  Competitive-load analysis")
    plotstyle.apply()

    shots = load_frame("shots_scored")
    tm = load_frame("team_match_profiles")
    d = prepare(shots)
    print(f"  {len(d):,} shots with complete calendar-load information")
    print(
        "  exposure distribution: "
        f"short rest {d['short_rest'].mean():.1%} of shots, "
        f"mean matches in previous 14 days {d['matches_last_14'].mean():.2f}"
    )

    # ---- Table 6: primary and secondary outcomes -------------------------
    rows = []
    for exp_var, exp_label in EXPOSURES.items():
        for out_var, out_label in CONTINUOUS_OUTCOMES.items():
            r = fit_continuous(d, out_var, exp_var)
            if r:
                rows.append({"Exposure": exp_label, "Outcome": out_label,
                             "Family": "primary" if out_var == "logit_shot_quality"
                             else "secondary", **r})
        for out_var, out_label in BINARY_OUTCOMES.items():
            r = fit_binary(d, out_var, exp_var)
            if r:
                rows.append({"Exposure": exp_label, "Outcome": out_label,
                             "Family": "secondary", **r})
        print(f"    fitted all outcomes for exposure: {exp_label}")

    t6 = pd.DataFrame(rows)
    sec = t6["Family"] == "secondary"
    t6["q (BH-FDR)"] = np.nan
    t6.loc[sec, "q (BH-FDR)"] = multipletests(t6.loc[sec, "p"], method="fdr_bh")[1]
    for c in ("Estimate", "CI low", "CI high"):
        t6[c] = t6[c].round(4)
    t6["p"] = t6["p"].round(4)
    t6["q (BH-FDR)"] = t6["q (BH-FDR)"].round(4)

    print("\n  primary outcome (shot quality):")
    print(
        t6[t6["Family"] == "primary"][
            ["Exposure", "Outcome", "Estimate", "CI low", "CI high", "p"]
        ].to_string(index=False)
    )
    print("\n  secondary outcomes with q < 0.05:")
    signif = t6[(sec) & (t6["q (BH-FDR)"] < 0.05)]
    print(
        signif[["Exposure", "Outcome", "Estimate", "CI low", "CI high", "q (BH-FDR)"]]
        .to_string(index=False)
        if len(signif)
        else "    none"
    )
    save_table(t6, "table6_load_effects")

    # ---- Table 6b: alternative calendar definitions ----------------------
    cal_rows = []
    for label, col in (
        ("League-only calendar", "rest_category_league_only"),
        ("All-competition calendar", "rest_category"),
        ("All-competition, alternative cut-points", "rest_category_alt"),
    ):
        dd = d[d[col] != "unknown"].copy()
        dd["short_rest_def"] = (dd[col] == "short").astype(int)
        r = fit_continuous(dd, "logit_shot_quality", "short_rest_def")
        if r:
            cal_rows.append({"Calendar definition": label,
                             "Short-rest shots": int(dd["short_rest_def"].sum()), **r})
    t6b = pd.DataFrame(cal_rows).round(4)
    print("\n  short-rest effect on shot quality under three calendar definitions:")
    print(t6b[["Calendar definition", "Short-rest shots", "Estimate", "CI low", "CI high", "p"]]
          .to_string(index=False))
    save_table(t6b, "table6b_calendar_definitions")

    # ---- Table 6c: European and cup exposure -----------------------------
    eu_rows = []
    d_eu = d[d["euro_participant"] == 1].copy()
    for label, var, data in (
        ("European match in previous 7 days", "euro_last_7", d_eu),
        ("Cup match in previous 7 days", "cup_last_7", d),
        ("Upcoming cup/European fixture within 3 days", "upcoming_cup_or_euro_3d", d),
        ("European participant (team-season)", "euro_participant", d),
    ):
        if data[var].nunique() < 2:
            continue
        r = fit_continuous(data, "logit_shot_quality", var)
        if r:
            eu_rows.append({"Exposure": label, "Exposed shots": int((data[var] > 0).sum()), **r})
    t6c = pd.DataFrame(eu_rows).round(4)
    print("\n  broader competitive-load exposures (shot quality):")
    print(t6c[["Exposure", "Exposed shots", "Estimate", "CI low", "CI high", "p"]]
          .to_string(index=False))
    save_table(t6c, "table6c_european_load")

    make_rest_figure(d)
    make_late_game_figure(d)
    make_resilience_figure(tm, d)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _mean_ci(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return m, m - 1.96 * se, m + 1.96 * se


def make_rest_figure(d: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(plotstyle.W_DOUBLE, 3.0))

    ax = axes[0]
    cats = ["short", "normal", "weekly", "extended", "post_break"]
    labels = ["<=3 d", "4-6 d", "7-8 d", "9-14 d", ">14 d\n(break)"]
    present = [(c, l) for c, l in zip(cats, labels) if (d["rest_category_detailed"] == c).any()]
    xs, ms, los, his, ns = [], [], [], [], []
    for i, (c, _) in enumerate(present):
        sub = d[d["rest_category_detailed"] == c]["shot_quality"]
        m, lo, hi = _mean_ci(sub)
        xs.append(i); ms.append(m); los.append(lo); his.append(hi); ns.append(len(sub))
    ax.errorbar(xs, ms, yerr=[np.array(ms) - los, np.array(his) - ms],
                fmt="o", color=plotstyle.CATEGORICAL[0], markersize=5,
                capsize=3, elinewidth=1.1, markeredgecolor="white", markeredgewidth=0.6)
    ax.axhline(d["shot_quality"].mean(), color=plotstyle.INK_MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.set_xticks(xs)
    ax.set_xticklabels([l for _, l in present])
    ax.set_xlabel("Days since previous fixture (all competitions)")
    ax.set_ylabel("Mean shot quality")
    ax.set_title("a  Shot quality by rest interval", loc="left")
    # Sample sizes sit on a reserved strip below the data so they never collide
    # with the intervals.
    lo_lim = min(los) - (max(his) - min(los)) * 0.28
    ax.set_ylim(lo_lim, max(his) + (max(his) - min(los)) * 0.10)
    for x, n in zip(xs, ns):
        ax.annotate(f"n={n:,}", (x, lo_lim), textcoords="offset points", xytext=(0, 4),
                    ha="center", va="bottom", fontsize=6, color=plotstyle.INK_MUTED)
    ax.text(0.98, 0.955, "dashed line = league mean", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.5, color=plotstyle.INK_MUTED)

    ax = axes[1]
    grp = d.groupby("matches_last_14")["shot_quality"]
    xs = sorted(g for g in grp.groups if grp.get_group(g).size >= 40)
    ms, los, his = zip(*[_mean_ci(grp.get_group(g)) for g in xs])
    ax.errorbar(xs, ms, yerr=[np.array(ms) - np.array(los), np.array(his) - np.array(ms)],
                fmt="s", color=plotstyle.CATEGORICAL[1], markersize=5,
                capsize=3, elinewidth=1.1, markeredgecolor="white", markeredgewidth=0.6)
    ax.axhline(d["shot_quality"].mean(), color=plotstyle.INK_MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.set_xlabel("Matches played in the previous 14 days")
    ax.set_ylabel("Mean shot quality")
    ax.set_title("b  Shot quality by fixture congestion", loc="left")
    ax.set_xticks(xs)

    plotstyle.save(fig, "figure7_shot_quality_by_rest")


def make_late_game_figure(d: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(plotstyle.W_DOUBLE, 3.0))

    bins = [0, 15, 30, 45, 60, 75, 90, 200]
    labels = ["0-15", "16-30", "31-45", "46-60", "61-75", "76-90", "90+"]
    d = d.copy()
    d["minute_bin"] = pd.cut(d["minute"], bins=bins, labels=labels, right=True)

    ax = axes[0]
    xs, ms, los, his = [], [], [], []
    for i, lab in enumerate(labels):
        sub = d[d["minute_bin"] == lab]["shot_quality"]
        if len(sub) < 30:
            continue
        m, lo, hi = _mean_ci(sub)
        xs.append(i); ms.append(m); los.append(lo); his.append(hi)
    ax.errorbar(xs, ms, yerr=[np.array(ms) - los, np.array(his) - ms],
                fmt="o-", color=plotstyle.CATEGORICAL[0], markersize=4.5,
                capsize=2.5, elinewidth=1.0, markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([labels[i] for i in xs])
    ax.set_xlabel("Match minute")
    ax.set_ylabel("Mean shot quality")
    ax.set_title("a  Shot quality across the match", loc="left")

    ax = axes[1]
    for j, (cat, lab) in enumerate((("short", "Short rest (<=3 d)"),
                                    ("normal", "Normal rest (4-6 d)"),
                                    ("long", "Long rest (>=7 d)"))):
        sub_all = d[d["rest_category"] == cat]
        xs, ms = [], []
        for i, lab_b in enumerate(labels):
            sub = sub_all[sub_all["minute_bin"] == lab_b]["shot_quality"]
            if len(sub) < 25:
                continue
            xs.append(i); ms.append(sub.mean())
        ax.plot(xs, ms, marker=plotstyle.MARKERS[j], color=plotstyle.CATEGORICAL[j],
                label=lab, markersize=4, markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Match minute")
    ax.set_ylabel("Mean shot quality")
    ax.set_title("b  Late-game trend by rest category", loc="left")
    ax.legend(loc="upper left", fontsize=6.5)

    plotstyle.save(fig, "figure8_late_game_trends")


def make_resilience_figure(tm: pd.DataFrame, d: pd.DataFrame) -> None:
    """Per-team change in shot quality under short rest, with bootstrap CIs."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    rows = []
    for team, g in d.groupby("team_name"):
        short = g[g["rest_category"] == "short"]["shot_quality"].to_numpy()
        rest = g[g["rest_category"] != "short"]["shot_quality"].to_numpy()
        if len(short) < 40 or len(rest) < 100:
            continue
        delta = short.mean() - rest.mean()
        boots = np.array([
            rng.choice(short, len(short), replace=True).mean()
            - rng.choice(rest, len(rest), replace=True).mean()
            for _ in range(2000)
        ])
        rows.append({
            "Team": team, "delta": delta,
            "lo": np.quantile(boots, 0.025), "hi": np.quantile(boots, 0.975),
            "n_short": len(short),
        })
    res = pd.DataFrame(rows).sort_values("delta")
    if res.empty:
        print("  [fig  ] figure9 skipped - no team met the short-rest sample threshold")
        return

    fig, ax = plt.subplots(figsize=(plotstyle.W_SINGLE * 1.5, 0.22 * len(res) + 1.5))
    ypos = np.arange(len(res))
    colors = [
        plotstyle.CATEGORICAL[1] if hi < 0 else
        (plotstyle.CATEGORICAL[2] if lo > 0 else plotstyle.INK_MUTED)
        for lo, hi in zip(res["lo"], res["hi"])
    ]
    # matplotlib's `ecolor` takes a single colour, so each interval is drawn
    # separately to colour it by whether it excludes zero.
    for yi, (_, r), col in zip(ypos, res.iterrows(), colors):
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=col, lw=1.1, solid_capstyle="butt",
                zorder=2)
        for xb in (r["lo"], r["hi"]):
            ax.plot([xb, xb], [yi - 0.18, yi + 0.18], color=col, lw=1.1, zorder=2)
    ax.scatter(res["delta"], ypos, s=26, color=colors, zorder=3,
               edgecolor="white", linewidth=0.6)
    ax.axvline(0, color=plotstyle.INK_MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{t}  (n={n})" for t, n in zip(res["Team"], res["n_short"])],
                       fontsize=6.5)
    ax.set_xlabel("Change in mean shot quality under short rest (<=3 days)")
    ax.set_title("Team resilience of shot quality under fixture congestion", loc="left")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    n_sig = int(sum(1 for lo, hi in zip(res["lo"], res["hi"]) if lo > 0 or hi < 0))
    ax.text(
        0.99, 0.02,
        "coloured where the 95% bootstrap interval excludes zero\n"
        f"({n_sig} of {len(res)} teams; about {0.05 * len(res):.1f} expected by chance, "
        "intervals are unadjusted)",
        transform=ax.transAxes, ha="right", fontsize=6.2, color=plotstyle.INK_MUTED,
    )
    plotstyle.save(fig, "figure9_team_resilience")
    save_table(res.round(4), "table6d_team_resilience")


if __name__ == "__main__":
    main()
