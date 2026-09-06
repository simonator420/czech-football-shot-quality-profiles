"""Step 4 - Team and player attacking-profile construction (proposal 5.4, 5.5).

Aggregation units
    team-season    primary unit for profile clustering and stability analysis
    team-match     unit for the competitive-load models and early-season windows
    player-season  secondary, exploratory scouting output (>= 25 shots)

Shot quality throughout is the out-of-fold model prediction from step 3, so a
team's profile is never built from predictions that saw that team's own match.

Produces
    data/team_season_profiles.parquet
    data/team_match_profiles.parquet
    data/player_season_profiles.parquet
    tables/tableS1_player_profiles_top.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    MIN_PLAYER_SEASON_SHOTS,
    MIN_TEAM_SEASON_MATCHES,
    header,
    load_frame,
    save_frame,
    save_table,
)

#: A shot the model rates at or above this probability is treated as a clear
#: scoring opportunity; below the lower bound it is a speculative attempt.
HIGH_QUALITY_THRESHOLD = 0.15
LOW_QUALITY_THRESHOLD = 0.05
CLOSE_RANGE_M = 11.0

#: Features that define the attacking profile itself. Competitive-load response
#: features are built alongside these but deliberately kept out of the primary
#: clustering vector: they are undefined for team-seasons with few short-rest
#: fixtures, and with 48 team-seasons an imputed block of columns would drive
#: the solution. They enter the load-sensitivity typology (5.7) and the
#: all-feature clustering variant in the sensitivity analysis (5.10).
SHOT_QUALITY_FEATURES = [
    "mean_shot_quality",
    "median_shot_quality",
    "xg_per_match",
    "goals_minus_xg_per_match",
    "mean_distance_m",
    "mean_angle_proxy",
    "high_quality_share",
    "low_quality_share",
]
SHOT_SELECTION_FEATURES = [
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
PROFILE_FEATURES = SHOT_QUALITY_FEATURES + SHOT_SELECTION_FEATURES

#: The full profile block contains several near-duplicate descriptors of the
#: same construct (mean and median shot quality, high/low-quality shares, mean
#: distance and close-range share all correlate above |r| = 0.80). Feeding all
#: of them to a distance-based clustering algorithm silently triples the weight
#: of shot proximity. This reduced block keeps one representative per construct
#: and is used to test whether any instability in the cluster solution is an
#: artefact of that redundancy.
REDUCED_PROFILE_FEATURES = [
    "mean_shot_quality",
    "shots_per_match",
    "goals_minus_xg_per_match",
    "on_target_rate",
    "blocked_rate",
    "header_share",
    "set_piece_share",
    "fast_break_share",
    "mean_angle_proxy",
]

LOAD_RESPONSE_FEATURES = [
    "mean_sq_short_rest",
    "mean_sq_normal_rest",
    "mean_sq_long_rest",
    "short_rest_sq_delta",
    "shots_per_match_short_rest",
    "blocked_rate_short_rest",
]

SPATIAL_ONLY_FEATURES = [
    "mean_distance_m",
    "mean_angle_proxy",
    "close_range_share",
    "header_share",
    "inside_box_share",
    "mean_centrality",
]


def _core_aggregates(g: pd.DataFrame) -> dict:
    """Shot-quality and shot-selection descriptors for one group of shots."""
    n = len(g)
    return {
        "shots": n,
        "goals": int(g["goal"].sum()),
        "mean_shot_quality": g["shot_quality"].mean(),
        "median_shot_quality": g["shot_quality"].median(),
        "total_xg": g["shot_quality"].sum(),
        "mean_distance_m": g["distance_m"].mean(),
        "mean_distance_proxy": g["distance_proxy"].mean(),
        "mean_angle_proxy": g["angle_proxy"].mean(),
        "mean_centrality": g["centrality"].mean(),
        "high_quality_share": (g["shot_quality"] >= HIGH_QUALITY_THRESHOLD).mean(),
        "low_quality_share": (g["shot_quality"] < LOW_QUALITY_THRESHOLD).mean(),
        "on_target_rate": g["on_target"].mean(),
        "blocked_rate": g["blocked"].mean(),
        "goal_rate": g["goal"].mean(),
        "header_share": g["is_header"].mean(),
        "assisted_share": g["is_assisted"].mean(),
        "set_piece_share": g["is_set_piece"].mean(),
        "fast_break_share": g["is_fast_break"].mean(),
        "close_range_share": (g["distance_m"] < CLOSE_RANGE_M).mean(),
        "inside_box_share": g["inside_box"].mean(),
        "wide_angle_share": g["wide_angle"].mean(),
    }


def build_team_match(shots: pd.DataFrame, load: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (match_id, team_id), g in shots.groupby(["match_id", "team_id"]):
        rec = {"match_id": match_id, "team_id": team_id}
        rec.update(_core_aggregates(g))
        rec["season_name"] = g["season_name"].iloc[0]
        rec["team_name"] = g["team_name"].iloc[0]
        rec["opponent_team_id"] = g["opponent_team_id"].iloc[0]
        rec["is_home"] = g["is_home_shot"].iloc[0]
        rec["kickoff"] = g["start_datetime_utc"].iloc[0]
        rows.append(rec)
    tm = pd.DataFrame(rows)
    tm["goals_minus_xg"] = tm["goals"] - tm["total_xg"]

    load_cols = [c for c in load.columns if c not in ("season_name", "is_home", "kickoff")]
    tm = tm.merge(load[load_cols], on=["match_id", "team_id"], how="left")

    # Match order within a season drives the early-season prediction windows.
    tm = tm.sort_values(["team_id", "season_name", "kickoff"])
    tm["match_number"] = tm.groupby(["team_id", "season_name"]).cumcount() + 1
    return tm.reset_index(drop=True)


def build_team_season(shots: pd.DataFrame, team_match: pd.DataFrame) -> pd.DataFrame:
    n_matches = (
        team_match.groupby(["team_id", "season_name"])["match_id"].nunique().rename("matches")
    )

    rows = []
    for (team_id, season), g in shots.groupby(["team_id", "season_name"]):
        rec = {"team_id": team_id, "season_name": season, "team_name": g["team_name"].iloc[0]}
        rec.update(_core_aggregates(g))
        m = int(n_matches.get((team_id, season), 0))
        rec["matches"] = m
        rec["shots_per_match"] = rec["shots"] / m if m else np.nan
        rec["xg_per_match"] = rec["total_xg"] / m if m else np.nan
        rec["goals_per_match"] = rec["goals"] / m if m else np.nan
        rec["goals_minus_xg"] = rec["goals"] - rec["total_xg"]
        rec["goals_minus_xg_per_match"] = rec["goals_minus_xg"] / m if m else np.nan

        # --- competitive-load response ---
        for cat, key in (("short", "short"), ("normal", "normal"), ("long", "long")):
            sub = g[g["rest_category"] == cat]
            rec[f"mean_sq_{key}_rest"] = sub["shot_quality"].mean() if len(sub) else np.nan
            rec[f"shots_{key}_rest"] = len(sub)
        sub_short = g[g["rest_category"] == "short"]
        rec["blocked_rate_short_rest"] = (
            sub_short["blocked"].mean() if len(sub_short) else np.nan
        )
        tm_g = team_match[
            (team_match["team_id"] == team_id) & (team_match["season_name"] == season)
        ]
        short_matches = tm_g[tm_g["rest_category"] == "short"]
        rec["short_rest_matches"] = len(short_matches)
        rec["shots_per_match_short_rest"] = (
            short_matches["shots"].mean() if len(short_matches) else np.nan
        )
        rec["short_rest_sq_delta"] = rec["mean_sq_short_rest"] - rec["mean_shot_quality"]
        rec["euro_participant"] = int(tm_g["euro_participant"].max() or 0) if len(tm_g) else 0
        rows.append(rec)

    ts = pd.DataFrame(rows)
    ts = ts[ts["matches"] >= MIN_TEAM_SEASON_MATCHES].reset_index(drop=True)
    ts["team_season"] = ts["team_name"] + " " + ts["season_name"]
    return ts


def build_player_season(shots: pd.DataFrame, threshold: int) -> pd.DataFrame:
    rows = []
    for (player_id, season), g in shots.groupby(["player_id", "season_name"]):
        if len(g) < threshold:
            continue
        rec = {
            "player_id": player_id,
            "season_name": season,
            "player_name": g["player_name"].iloc[0],
            "position": g["player_position"].iloc[0],
            "team_name": g["team_name"].mode().iloc[0],
        }
        rec.update(_core_aggregates(g))
        rec["finishing_over_expectation"] = rec["goals"] - rec["total_xg"]
        rec["finishing_over_expectation_per_shot"] = (
            rec["finishing_over_expectation"] / rec["shots"]
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    header("STEP 4  Team and player attacking-profile construction")

    shots = load_frame("shots_scored")
    load = load_frame("team_match_load")

    team_match = build_team_match(shots, load)
    print(f"  team-match profiles: {len(team_match):,}")

    team_season = build_team_season(shots, team_match)
    print(
        f"  team-season profiles: {len(team_season)} "
        f"(>= {MIN_TEAM_SEASON_MATCHES} matches; "
        f"{team_season['season_name'].value_counts().sort_index().to_dict()})"
    )

    player_season = build_player_season(shots, MIN_PLAYER_SEASON_SHOTS)
    print(f"  player-season profiles: {len(player_season)} (>= {MIN_PLAYER_SEASON_SHOTS} shots)")

    missing = team_season[PROFILE_FEATURES].isna().sum()
    if missing.any():
        print("  WARNING - missing values in profile features:")
        print(missing[missing > 0].to_string())
    else:
        print("  no missing values in the primary profile feature block")

    print(
        "\n  short-rest coverage: "
        f"{(team_season['short_rest_matches'] == 0).sum()} of {len(team_season)} "
        "team-seasons had no short-rest match "
        f"(median {team_season['short_rest_matches'].median():.0f} such matches)"
    )

    save_frame(team_match, "team_match_profiles")
    save_frame(team_season, "team_season_profiles")
    save_frame(player_season, "player_season_profiles")

    top = (
        player_season.sort_values("finishing_over_expectation", ascending=False)
        .head(20)[
            [
                "player_name",
                "team_name",
                "season_name",
                "position",
                "shots",
                "goals",
                "mean_shot_quality",
                "total_xg",
                "finishing_over_expectation",
                "close_range_share",
                "header_share",
                "on_target_rate",
            ]
        ]
        .round(3)
    )
    save_table(top, "tableS1_player_profiles_top")

    print("\n  team-season profile summary (primary clustering features):")
    print(team_season[PROFILE_FEATURES].describe().T[["mean", "std", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
