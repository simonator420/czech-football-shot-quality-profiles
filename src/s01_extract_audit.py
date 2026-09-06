"""Step 1 - Data extraction, cleaning and dataset audit (proposal 4.3 / 5.1).

Produces
    data/matches.parquet          league matches entering the study
    data/shots_clean.parquet      analytical shot sample (pre-feature-engineering)
    data/fixtures.parquet         all competitive fixtures per team (calendar load)
    tables/table1_dataset_construction.csv
    tables/table2_sample_characteristics.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    LEAGUE_TOURNAMENTS,
    PLAYOFF_TOURNAMENTS,
    SEASONS,
    BODY_PART_MAP,
    SHOT_TYPE_MAP,
    SITUATION_MAP,
    header,
    read_sql,
    save_frame,
    save_table,
)

SEASON_SQL = "', '".join(SEASONS)


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def fetch_matches() -> pd.DataFrame:
    return read_sql(
        f"""
        SELECT m.id AS match_id,
               m.sofascore_event_id,
               m.season_name,
               m.tournament_name,
               m.round_number,
               m.status_type,
               m.shotmap_status,
               m.start_datetime_utc,
               m.home_team_id,
               m.away_team_id,
               m.home_score_current,
               m.away_score_current,
               ht.name AS home_team_name,
               at.name AS away_team_name
        FROM FootballMatches m
        LEFT JOIN FootballTeams ht ON ht.id = m.home_team_id
        LEFT JOIN FootballTeams at ON at.id = m.away_team_id
        WHERE m.season_name IN ('{SEASON_SQL}')
        """
    )


def fetch_shots() -> pd.DataFrame:
    return read_sql(
        f"""
        SELECT s.id AS shot_row_id,
               s.match_id,
               s.sofascore_shot_id,
               s.shot_index,
               s.team_id,
               s.player_id,
               s.is_home,
               s.minute,
               s.added_time,
               s.time_seconds,
               s.shot_type,
               s.situation,
               s.body_part,
               s.goal_mouth_location,
               s.goal_type,
               s.player_x,
               s.player_y,
               s.goal_x,
               s.goal_y,
               s.goal_z,
               s.block_x,
               s.block_y,
               p.name AS player_name,
               p.position AS player_position,
               t.name AS team_name
        FROM FootballShots s
        JOIN FootballMatches m ON m.id = s.match_id
        LEFT JOIN FootballPlayers p ON p.id = s.player_id
        LEFT JOIN FootballTeams t ON t.id = s.team_id
        WHERE m.season_name IN ('{SEASON_SQL}')
        """
    )


def fetch_goal_incidents() -> pd.DataFrame:
    return read_sql(
        f"""
        SELECT i.match_id, i.time, i.added_time, i.incident_class,
               i.is_home, i.home_score, i.away_score
        FROM FootballIncidents i
        JOIN FootballMatches m ON m.id = i.match_id
        WHERE m.season_name IN ('{SEASON_SQL}')
          AND i.incident_type = 'goal'
          AND i.home_score IS NOT NULL
        """
    )


def fetch_fixtures(matches: pd.DataFrame) -> pd.DataFrame:
    """All competitive fixtures per team, used for calendar-load features.

    Combines domestic league matches (including the relegation/promotion
    play-off, which counts as calendar load even though it contributes no
    shots), domestic cup fixtures and European fixtures.
    """
    teams = read_sql("SELECT id AS team_id, sofascore_team_id, name FROM FootballTeams")
    ss_to_id = dict(zip(teams["sofascore_team_id"], teams["team_id"]))
    name_to_id = dict(zip(teams["name"], teams["team_id"]))

    dom = matches.loc[matches["status_type"] == "finished"].copy()
    home = dom[["home_team_id", "start_datetime_utc", "season_name", "match_id"]].rename(
        columns={"home_team_id": "team_id", "start_datetime_utc": "kickoff"}
    )
    away = dom[["away_team_id", "start_datetime_utc", "season_name", "match_id"]].rename(
        columns={"away_team_id": "team_id", "start_datetime_utc": "kickoff"}
    )
    dom_long = pd.concat([home, away], ignore_index=True)
    dom_long["competition"] = "league"

    frames = [dom_long]

    for table, comp in (("FootballCupFixtures", "cup"), ("FootballEuropeanFixtures", "european")):
        raw = read_sql(
            f"""
            SELECT season_name, match_datetime AS kickoff,
                   home_team_name, away_team_name,
                   home_team_id_ss, away_team_id_ss
            FROM {table}
            WHERE season_name IN ('{SEASON_SQL}')
            """
        )
        for side in ("home", "away"):
            part = raw[["season_name", "kickoff", f"{side}_team_id_ss", f"{side}_team_name"]].copy()
            part.columns = ["season_name", "kickoff", "ss_id", "team_name_src"]
            part["team_id"] = part["ss_id"].map(ss_to_id)
            unmapped = part["team_id"].isna()
            part.loc[unmapped, "team_id"] = part.loc[unmapped, "team_name_src"].map(name_to_id)
            part = part.dropna(subset=["team_id", "kickoff"])
            part["team_id"] = part["team_id"].astype(int)
            part["competition"] = comp
            part["match_id"] = np.nan
            frames.append(part[["team_id", "kickoff", "season_name", "match_id", "competition"]])

    fixtures = pd.concat(frames, ignore_index=True)
    fixtures["kickoff"] = pd.to_datetime(fixtures["kickoff"])
    fixtures = fixtures.dropna(subset=["team_id", "kickoff"])
    fixtures["team_id"] = fixtures["team_id"].astype(int)
    fixtures = fixtures.drop_duplicates(subset=["team_id", "kickoff", "competition"])
    return fixtures.sort_values(["team_id", "kickoff"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Score-state reconstruction
# --------------------------------------------------------------------------


def _time_key(minute: pd.Series, added: pd.Series) -> pd.Series:
    """Ordering key combining minute and added time (added time < 100 min)."""
    return minute.fillna(0).astype(float) + added.fillna(0).astype(float) / 100.0


def attach_score_state(shots: pd.DataFrame, goals: pd.DataFrame) -> pd.DataFrame:
    """Score line immediately *before* each shot, rebuilt from goal incidents.

    `FootballShots.home_score` / `away_score` are NULL throughout this database,
    so the pre-shot score state is reconstructed from the goal incident
    timeline. For each shot the score is taken from the last goal that occurred
    strictly earlier in the match; shots sharing a time key with a goal (the
    scoring shot itself, or a rebound in the same minute) therefore receive the
    score state that preceded that goal.
    """
    goals = goals.copy()
    goals["t_key"] = _time_key(goals["time"], goals["added_time"])
    goals = goals.sort_values(["match_id", "t_key"])

    shots = shots.copy()
    shots["t_key"] = _time_key(shots["minute"], shots["added_time"])

    home_before = np.zeros(len(shots), dtype=float)
    away_before = np.zeros(len(shots), dtype=float)

    goal_groups = {mid: g for mid, g in goals.groupby("match_id")}
    positions = np.arange(len(shots))
    for mid, idx in shots.groupby("match_id").groups.items():
        g = goal_groups.get(mid)
        if g is None or g.empty:
            continue
        rows = shots.loc[idx]
        pos = positions[shots.index.get_indexer(idx)]
        k = np.searchsorted(g["t_key"].to_numpy(), rows["t_key"].to_numpy(), side="left")
        hs = np.concatenate([[0.0], g["home_score"].to_numpy(dtype=float)])
        aws = np.concatenate([[0.0], g["away_score"].to_numpy(dtype=float)])
        home_before[pos] = hs[k]
        away_before[pos] = aws[k]

    shots["home_score_before"] = home_before
    shots["away_score_before"] = away_before
    # Score difference from the perspective of the shooting team.
    is_home = shots["is_home"].astype(float)
    shots["score_diff_before"] = np.where(
        is_home == 1,
        shots["home_score_before"] - shots["away_score_before"],
        shots["away_score_before"] - shots["home_score_before"],
    )
    shots["score_state"] = pd.cut(
        shots["score_diff_before"],
        bins=[-np.inf, -0.5, 0.5, np.inf],
        labels=["trailing", "level", "leading"],
    ).astype(str)
    return shots


# --------------------------------------------------------------------------
# Cleaning pipeline with audit trail
# --------------------------------------------------------------------------


def build_sample(matches: pd.DataFrame, shots: pd.DataFrame):
    """Apply inclusion criteria, recording the count at every step."""
    audit: list[dict] = []

    def step(label, n_matches, n_shots, note=""):
        audit.append(
            {
                "Step": label,
                "Matches": n_matches,
                "Shots": n_shots,
                "Note": note,
            }
        )

    step(
        "Matches scraped, seasons 2022/23-2024/25",
        len(matches),
        len(shots),
        "Season 2025/26 excluded from the study by design",
    )

    fin = matches[matches["status_type"] == "finished"]
    sh = shots[shots["match_id"].isin(fin["match_id"])]
    step("Finished matches", len(fin), len(sh), "Postponed/cancelled fixtures removed")

    withmap = fin[fin["shotmap_status"] == "processed"]
    sh = sh[sh["match_id"].isin(withmap["match_id"])]
    step(
        "Matches with shot-map data",
        len(withmap),
        len(sh),
        f"{len(fin) - len(withmap)} finished matches had no shot map available",
    )

    league = withmap[withmap["tournament_name"].isin(LEAGUE_TOURNAMENTS)]
    n_playoff = int(withmap["tournament_name"].isin(PLAYOFF_TOURNAMENTS).sum())
    sh = sh[sh["match_id"].isin(league["match_id"])]
    step(
        "Czech First League matches (regular season + championship/relegation group)",
        len(league),
        len(sh),
        f"{n_playoff} relegation/promotion play-off matches vs second-tier "
        "opposition excluded from the shot sample (retained as calendar load)",
    )

    before = len(sh)
    sh = sh.drop_duplicates(subset=["match_id", "sofascore_shot_id"])
    step("Duplicate shot records removed", len(league), len(sh), f"{before - len(sh)} duplicates")

    before = len(sh)
    sh = sh[sh["situation"] != "shootout"]
    step(
        "Penalty shoot-out attempts removed",
        len(league),
        len(sh),
        f"{before - len(sh)} shoot-out attempts",
    )

    before = len(sh)
    sh = sh[sh["goal_type"] != "own"]
    step(
        "Own goals removed",
        len(league),
        len(sh),
        f"{before - len(sh)} own goals (not attempts on the opponent's goal)",
    )

    before = len(sh)
    sh = sh[sh["player_x"].notna() & sh["player_y"].notna()]
    step(
        "Shots with valid spatial coordinates",
        len(league),
        len(sh),
        f"{before - len(sh)} shots without coordinates",
    )

    before = len(sh)
    sh = sh[
        sh["player_x"].between(0, 100)
        & sh["player_y"].between(0, 100)
        & (sh["player_x"] > 0)
    ]
    step(
        "Coordinates within the normalised pitch grid",
        len(league),
        len(sh),
        f"{before - len(sh)} shots with out-of-range coordinates",
    )

    before = len(sh)
    sh = sh[sh["team_id"].notna() & sh["player_id"].notna()]
    step(
        "Shots with valid team and player identifiers",
        len(league),
        len(sh),
        f"{before - len(sh)} shots without team/player link",
    )

    before = len(sh)
    sh = sh[sh["shot_type"].notna() & sh["body_part"].notna() & sh["situation"].notna()]
    step(
        "Final analytical shot sample",
        len(league),
        len(sh),
        f"{before - len(sh)} shots with missing shot type/body part/situation",
    )

    return league.reset_index(drop=True), sh.reset_index(drop=True), pd.DataFrame(audit)


def standardise_categories(shots: pd.DataFrame) -> pd.DataFrame:
    shots = shots.copy()
    shots["body_part"] = shots["body_part"].map(BODY_PART_MAP).fillna("other")
    shots["situation"] = shots["situation"].map(SITUATION_MAP).fillna("other")
    shots["outcome"] = shots["shot_type"].map(SHOT_TYPE_MAP).fillna("other")
    shots["goal"] = (shots["outcome"] == "goal").astype(int)
    shots["on_target"] = shots["outcome"].isin(["goal", "saved"]).astype(int)
    shots["blocked"] = (shots["outcome"] == "blocked").astype(int)
    shots["is_penalty"] = (shots["situation"] == "penalty").astype(int)
    shots["is_header"] = (shots["body_part"] == "head").astype(int)
    return shots


def build_table2(shots: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Table 2 - sample characteristics by season."""
    rows = []
    for season in SEASONS + ["All seasons"]:
        if season == "All seasons":
            s, m = shots, matches
        else:
            s = shots[shots["season_name"] == season]
            m = matches[matches["season_name"] == season]
        n_matches = m["match_id"].nunique()
        goals = int(s["goal"].sum())
        rows.append(
            {
                "Season": season,
                "Matches": n_matches,
                "Teams": len(
                    set(m["home_team_id"]).union(set(m["away_team_id"]))
                ),
                "Shots": len(s),
                "Shots per match": round(len(s) / n_matches, 2),
                "Goals": goals,
                "Goals per match": round(goals / n_matches, 2),
                "Conversion rate (%)": round(100 * goals / len(s), 2),
                "On-target (%)": round(100 * s["on_target"].mean(), 1),
                "Blocked (%)": round(100 * s["blocked"].mean(), 1),
                "Headers (%)": round(100 * s["is_header"].mean(), 1),
                "Penalties": int(s["is_penalty"].sum()),
                "Set-piece origin (%)": round(
                    100 * s["situation"].isin(["corner", "set_piece", "throw_in", "free_kick", "penalty"]).mean(),
                    1,
                ),
                "Distinct players": s["player_id"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    header("STEP 1  Data extraction, cleaning and dataset audit")

    matches = fetch_matches()
    shots = fetch_shots()
    print(f"  fetched {len(matches):,} matches and {len(shots):,} shot records")

    league_matches, sample, audit = build_sample(matches, shots)
    print("\n  Inclusion flow:")
    for _, r in audit.iterrows():
        print(f"    {r['Step']:<70} matches={r['Matches']:>5}  shots={r['Shots']:>7,}")

    goals = fetch_goal_incidents()
    sample = attach_score_state(sample, goals)
    sample = standardise_categories(sample)

    meta_cols = [
        "match_id",
        "season_name",
        "tournament_name",
        "round_number",
        "start_datetime_utc",
        "home_team_id",
        "away_team_id",
        "home_team_name",
        "away_team_name",
    ]
    sample = sample.merge(league_matches[meta_cols], on="match_id", how="left")
    sample["opponent_team_id"] = np.where(
        sample["is_home"] == 1, sample["away_team_id"], sample["home_team_id"]
    )
    sample["start_datetime_utc"] = pd.to_datetime(sample["start_datetime_utc"])

    fixtures = fetch_fixtures(matches)
    print(
        f"\n  calendar fixtures: {len(fixtures):,} team-fixtures "
        f"({fixtures['competition'].value_counts().to_dict()})"
    )

    save_frame(league_matches, "matches")
    save_frame(sample, "shots_clean")
    save_frame(fixtures, "fixtures")

    save_table(audit, "table1_dataset_construction")
    save_table(build_table2(sample, league_matches), "table2_sample_characteristics")

    print("\n  score-state reconstruction check:")
    print(sample["score_state"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
