"""Step 2 - Shot-level feature engineering (proposal 5.2).

Builds spatial, technical, match-context and competitive-load features.

Two deliberate deviations from the feature list in proposal 5.2, both required
to keep the shot-quality model free of outcome leakage:

  * `shot_type` in the source database is the shot *outcome*
    (goal/save/miss/block/post), not a pre-shot descriptor. It is used to
    derive the modelling target and the on-target/blocked outcomes, and is
    never offered to the model as a predictor.
  * `goal_mouth_location` records where the ball crossed the goal frame and is
    only defined once a shot has been struck and kept on target. It is
    therefore post-outcome information and is excluded from the predictor set,
    while remaining available for descriptive reporting.

Produces
    data/shots_features.parquet
    data/team_match_load.parquet
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    BOX_DEPTH_X,
    BOX_HALF_WIDTH_Y,
    GOAL_HALF_WIDTH_Y,
    GOAL_WIDTH_M,
    REST_CATEGORY_BINS,
    REST_CATEGORY_BINS_ALT,
    REST_CATEGORY_LABELS,
    SET_PIECE_SITUATIONS,
    SIX_YARD_DEPTH_X,
    SIX_YARD_HALF_WIDTH_Y,
    X_UNIT_M,
    Y_UNIT_M,
    header,
    load_frame,
    save_frame,
)

# --------------------------------------------------------------------------
# Spatial features
# --------------------------------------------------------------------------


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    x = df["player_x"].to_numpy(dtype=float)
    y = df["player_y"].to_numpy(dtype=float)
    dy = y - 50.0

    # --- primary proxies, exactly as pre-specified in proposal 5.2 ---
    df["distance_proxy"] = np.sqrt(x**2 + dy**2)
    df["angle_proxy"] = np.arctan(np.abs(dy) / np.maximum(x, 1e-6))

    # --- derived normalised-space descriptors ---
    df["lateral_offset"] = np.abs(dy)
    df["centrality"] = 1.0 - np.abs(dy) / 50.0
    df["inside_box"] = ((x <= BOX_DEPTH_X) & (np.abs(dy) <= BOX_HALF_WIDTH_Y)).astype(int)
    df["inside_six_yard"] = (
        (x <= SIX_YARD_DEPTH_X) & (np.abs(dy) <= SIX_YARD_HALF_WIDTH_Y)
    ).astype(int)
    df["central_corridor"] = (np.abs(dy) <= GOAL_HALF_WIDTH_Y).astype(int)

    # --- metric-corrected alternatives (sensitivity: "alternative distance and
    #     angle definitions", proposal 5.10). The two axes are normalised
    #     independently, so raw normalised units mix 1.05 m steps in x with
    #     0.68 m steps in y. ---
    dx_m = x * X_UNIT_M
    dy_m = dy * Y_UNIT_M
    df["distance_m"] = np.sqrt(dx_m**2 + dy_m**2)
    df["angle_m"] = np.arctan(np.abs(dy_m) / np.maximum(dx_m, 1e-6))

    # Angle subtended by the goal posts at the shot location - the standard
    # geometric shot-quality descriptor.
    half = GOAL_WIDTH_M / 2.0
    num = GOAL_WIDTH_M * dx_m
    den = dx_m**2 + dy_m**2 - half**2
    visible = np.arctan2(num, den)
    visible = np.where(visible < 0, visible + np.pi, visible)
    df["visible_goal_angle"] = visible

    # Quantile bands used for descriptive reporting only.
    df["distance_band"] = pd.cut(
        df["distance_m"],
        bins=[0, 6, 11, 16.5, 25, np.inf],
        labels=["<6m", "6-11m", "11-16.5m", "16.5-25m", ">25m"],
    ).astype(str)
    df["wide_angle"] = (df["angle_proxy"] > df["angle_proxy"].median()).astype(int)
    return df


# --------------------------------------------------------------------------
# Technical and match-context features
# --------------------------------------------------------------------------


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_home_shot"] = df["is_home"].astype(float).fillna(0).astype(int)
    df["minute"] = df["minute"].fillna(0).astype(float)
    df["added_time_flag"] = (df["added_time"].fillna(0) > 0).astype(int)
    df["second_half"] = (df["minute"] > 45).astype(int)
    df["late_game"] = (df["minute"] >= 75).astype(int)
    df["match_phase"] = pd.cut(
        df["minute"],
        bins=[-np.inf, 15, 30, 45, 60, 75, np.inf],
        labels=["0-15", "16-30", "31-45", "46-60", "61-75", "76+"],
    ).astype(str)

    df["is_set_piece"] = df["situation"].isin(SET_PIECE_SITUATIONS).astype(int)
    df["is_open_play"] = (~df["situation"].isin(SET_PIECE_SITUATIONS)).astype(int)
    df["is_assisted"] = (df["situation"] == "assisted").astype(int)
    df["is_corner"] = (df["situation"] == "corner").astype(int)
    df["is_fast_break"] = (df["situation"] == "fast_break").astype(int)
    df["is_free_kick"] = (df["situation"] == "free_kick").astype(int)
    df["preferred_foot_unknown"] = (df["body_part"] == "other").astype(int)
    return df


# --------------------------------------------------------------------------
# Competitive-load features
# --------------------------------------------------------------------------


def build_team_match_load(matches: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Calendar-load indicators for every team-match in the shot sample.

    Rest intervals and rolling fixture counts are computed within season, so a
    summer break is never counted as "long rest". Season-opening matches carry
    no rest information and are flagged rather than imputed.
    """
    matches = matches.copy()
    matches["start_datetime_utc"] = pd.to_datetime(matches["start_datetime_utc"])
    long = pd.concat(
        [
            matches[["match_id", "season_name", "start_datetime_utc", "home_team_id"]]
            .rename(columns={"home_team_id": "team_id"})
            .assign(is_home=1),
            matches[["match_id", "season_name", "start_datetime_utc", "away_team_id"]]
            .rename(columns={"away_team_id": "team_id"})
            .assign(is_home=0),
        ],
        ignore_index=True,
    ).rename(columns={"start_datetime_utc": "kickoff"})

    fixtures = fixtures.copy()
    fixtures["kickoff"] = pd.to_datetime(fixtures["kickoff"])

    records = []
    for (team_id, season), grp in long.groupby(["team_id", "season_name"]):
        cal = fixtures[
            (fixtures["team_id"] == team_id) & (fixtures["season_name"] == season)
        ].sort_values("kickoff")
        cal_times = cal["kickoff"].to_numpy()
        cal_comp = cal["competition"].to_numpy()

        league_times = cal_times[cal_comp == "league"]
        cup_times = cal_times[cal_comp == "cup"]
        euro_times = cal_times[cal_comp == "european"]
        has_euro = len(euro_times) > 0

        for _, row in grp.sort_values("kickoff").iterrows():
            t = np.datetime64(row["kickoff"])

            def days_since(times):
                prev = times[times < t]
                if len(prev) == 0:
                    return np.nan
                return (t - prev[-1]) / np.timedelta64(1, "D")

            def days_until(times):
                nxt = times[times > t]
                if len(nxt) == 0:
                    return np.nan
                return (nxt[0] - t) / np.timedelta64(1, "D")

            def count_prev(times, days):
                lo = t - np.timedelta64(int(days * 24), "h")
                return int(((times >= lo) & (times < t)).sum())

            rec = {
                "match_id": row["match_id"],
                "team_id": team_id,
                "season_name": season,
                "kickoff": row["kickoff"],
                "is_home": row["is_home"],
                "days_since_league": days_since(league_times),
                "days_since_any": days_since(cal_times),
                "matches_last_7": count_prev(cal_times, 7),
                "matches_last_14": count_prev(cal_times, 14),
                "matches_last_21": count_prev(cal_times, 21),
                "league_matches_last_14": count_prev(league_times, 14),
                "cup_last_3": count_prev(cup_times, 3),
                "cup_last_7": count_prev(cup_times, 7),
                "cup_last_14": count_prev(cup_times, 14),
                "euro_last_3": count_prev(euro_times, 3),
                "euro_last_7": count_prev(euro_times, 7),
                "euro_last_14": count_prev(euro_times, 14),
                "days_since_euro": days_since(euro_times),
                "days_until_next_any": days_until(cal_times),
                "euro_participant": int(has_euro),
            }
            rec["upcoming_cup_or_euro_3d"] = int(
                (days_until(np.concatenate([cup_times, euro_times])) or np.inf) <= 3
                if len(cup_times) + len(euro_times) > 0
                else 0
            )
            records.append(rec)

    load = pd.DataFrame(records)
    load["is_season_opener"] = load["days_since_any"].isna().astype(int)

    # Rest categories: <=3 days short, 4-6 normal, >=7 long (all-competition
    # calendar). Two further calendar definitions are retained for the
    # sensitivity analysis in proposal 5.10.
    load["rest_category"] = pd.cut(
        load["days_since_any"], bins=REST_CATEGORY_BINS, labels=REST_CATEGORY_LABELS
    ).astype(object)
    load["rest_category_league_only"] = pd.cut(
        load["days_since_league"], bins=REST_CATEGORY_BINS, labels=REST_CATEGORY_LABELS
    ).astype(object)
    load["rest_category_alt"] = pd.cut(
        load["days_since_any"], bins=REST_CATEGORY_BINS_ALT, labels=REST_CATEGORY_LABELS
    ).astype(object)
    for col in ("rest_category", "rest_category_league_only", "rest_category_alt"):
        load[col] = load[col].where(load[col].notna(), "unknown")

    load["short_rest"] = (load["rest_category"] == "short").astype(int)
    load["congested"] = (load["matches_last_14"] >= 4).astype(int)

    # The Czech First League runs a winter break of roughly ten weeks, so the
    # raw rest interval reaches 78 days. Such a gap is a competition break, not
    # recovery within a congested schedule, and pooling it with ordinary long
    # rest would bias the load contrast. Break matches are flagged and the
    # continuous rest variable is winsorised at 14 days for the regression
    # models; a finer category set separates the weekly cycle from genuinely
    # extended gaps.
    load["post_break"] = (load["days_since_any"] > 14).astype(int)
    load["days_since_any_capped"] = load["days_since_any"].clip(upper=14)
    load["rest_category_detailed"] = pd.cut(
        load["days_since_any"],
        bins=[-np.inf, 3, 6, 8, 14, np.inf],
        labels=["short", "normal", "weekly", "extended", "post_break"],
    ).astype(object)
    load["rest_category_detailed"] = load["rest_category_detailed"].where(
        load["rest_category_detailed"].notna(), "unknown"
    )
    return load


# --------------------------------------------------------------------------


def main() -> None:
    header("STEP 2  Shot-level feature engineering")

    shots = load_frame("shots_clean")
    matches = load_frame("matches")
    fixtures = load_frame("fixtures")

    shots = add_spatial_features(shots)
    shots = add_context_features(shots)
    print(f"  spatial + context features built for {len(shots):,} shots")

    load = build_team_match_load(matches, fixtures)
    print(f"  calendar-load features built for {len(load):,} team-matches")

    shots = shots.merge(
        load.drop(columns=["kickoff", "is_home", "season_name"]),
        on=["match_id", "team_id"],
        how="left",
        validate="many_to_one",
    )

    # Opponent-side load, used to control for the fact that both teams'
    # schedules shape the match context in which a shot is taken.
    opp = load[["match_id", "team_id", "days_since_any", "matches_last_14"]].rename(
        columns={
            "team_id": "opponent_team_id",
            "days_since_any": "opp_days_since_any",
            "matches_last_14": "opp_matches_last_14",
        }
    )
    shots = shots.merge(opp, on=["match_id", "opponent_team_id"], how="left")

    print("\n  rest-category distribution of shots:")
    print(shots["rest_category"].value_counts(normalize=True).round(3).to_string())
    print("\n  goal rate by rest category:")
    print(shots.groupby("rest_category")["goal"].agg(["mean", "size"]).round(4).to_string())
    print("\n  missingness in key load features:")
    for c in ("days_since_any", "days_since_league", "matches_last_14"):
        print(f"    {c:<22} {shots[c].isna().mean():.3%}")

    save_frame(shots, "shots_features")
    save_frame(load, "team_match_load")


if __name__ == "__main__":
    main()
