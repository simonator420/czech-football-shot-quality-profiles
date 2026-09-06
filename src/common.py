"""Shared configuration, IO helpers and constants for the shot-quality study.

Study: Longitudinal Shot-Quality and Attacking Performance Profiles in Czech
Professional Football (Czech First League, seasons 2022/23-2024/25).

Season 2025/26 is deliberately excluded from the analytical sample; see
`SEASONS` below and the note in `table1_dataset_construction`.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore", message="pandas only supports SQLAlchemy connectable", category=UserWarning
)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
DATA = Path(os.environ.get("FOOTBALL_PROCESSED_DIR", ROOT / "data" / "processed_release"))
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
MODELS = OUT / "models"
LOGS = OUT / "logs"

for _d in (DATA, TABLES, FIGURES, MODELS, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", 8889)),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", "root"),
    "database": os.environ.get("DB_NAME", "czech_soccer"),
}


def connect():
    import pymysql

    return pymysql.connect(charset="utf8mb4", **DB_CONFIG)


def read_sql(query: str, params=None) -> pd.DataFrame:
    conn = connect()
    try:
        return pd.read_sql(query, conn, params=params)
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Study design constants
# --------------------------------------------------------------------------

SEASONS = ["2022/23", "2023/24", "2024/25"]

#: Chronological validation split (proposal 5.3.3, adapted to the three-season
#: sample): the two earliest seasons train the model, the most recent season is
#: a fully held-out test set. Hyper-parameters are tuned inside TRAIN_SEASONS
#: using a time-ordered inner split, so the test season is never touched.
TRAIN_SEASONS = ["2022/23", "2023/24"]
TEST_SEASONS = ["2024/25"]

#: Tournament labels that constitute the Czech First League proper. The
#: "Relegation/Promotion" play-off is contested against second-tier opposition
#: and is therefore excluded from the shot sample, while still counting towards
#: each First League team's calendar load.
LEAGUE_TOURNAMENTS = (
    "Czech First League",
    "Czech First League, Championship",
    "Czech First League, Relegation",
)
PLAYOFF_TOURNAMENTS = (
    "Czech First League, Relegation/Promotion",
    "1. Liga, Qualification Playoffs",
)

#: Minimum league matches for a team-season to enter profile clustering.
MIN_TEAM_SEASON_MATCHES = 20

#: Minimum shots for a player-season to enter player profiling (proposal 4.3).
MIN_PLAYER_SEASON_SHOTS = 25
PLAYER_SHOT_THRESHOLD_SENSITIVITY = [25, 30, 40, 50]

RANDOM_STATE = 20260728

# --------------------------------------------------------------------------
# Pitch geometry
# --------------------------------------------------------------------------
# SofaScore shot-map coordinates are normalised to a 0-100 grid on each axis,
# with the attacked goal at x = 0 and the pitch mid-line at y = 50. Because the
# two axes are normalised independently, one x-unit and one y-unit do not span
# the same physical distance. The proposal's primary distance/angle proxies are
# computed in raw normalised units (as pre-specified); metric-corrected
# equivalents are computed alongside them and used in the sensitivity analysis
# "alternative distance and angle definitions" (proposal 5.10).

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
X_UNIT_M = PITCH_LENGTH_M / 100.0  # 1.05 m per normalised x unit
Y_UNIT_M = PITCH_WIDTH_M / 100.0  # 0.68 m per normalised y unit
GOAL_WIDTH_M = 7.32
#: Half goal width expressed in normalised y units (7.32 / 2 / 0.68).
GOAL_HALF_WIDTH_Y = (GOAL_WIDTH_M / 2.0) / Y_UNIT_M
#: Penalty box: 16.5 m deep, 40.32 m wide.
BOX_DEPTH_X = 16.5 / X_UNIT_M
BOX_HALF_WIDTH_Y = (40.32 / 2.0) / Y_UNIT_M
#: Six-yard box: 5.5 m deep, 18.32 m wide.
SIX_YARD_DEPTH_X = 5.5 / X_UNIT_M
SIX_YARD_HALF_WIDTH_Y = (18.32 / 2.0) / Y_UNIT_M

# --------------------------------------------------------------------------
# Category standardisation (proposal 5.1)
# --------------------------------------------------------------------------

BODY_PART_MAP = {
    "right-foot": "right_foot",
    "left-foot": "left_foot",
    "head": "head",
    "other": "other",
}

SITUATION_MAP = {
    "regular": "open_play",
    "assisted": "assisted",
    "fast-break": "fast_break",
    "corner": "corner",
    "set-piece": "set_piece",
    "throw-in-set-piece": "throw_in",
    "free-kick": "free_kick",
    "penalty": "penalty",
}

SHOT_TYPE_MAP = {
    "goal": "goal",
    "save": "saved",
    "miss": "off_target",
    "block": "blocked",
    "post": "woodwork",
}

#: Situations that originate from a dead ball.
SET_PIECE_SITUATIONS = {"corner", "set_piece", "throw_in", "free_kick", "penalty"}

REST_CATEGORY_BINS = [-np.inf, 3, 6, np.inf]
REST_CATEGORY_LABELS = ["short", "normal", "long"]
#: Alternative cut-points used in the "alternative rest categories" sensitivity.
REST_CATEGORY_BINS_ALT = [-np.inf, 4, 7, np.inf]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    """Write a results table as CSV and return the path."""
    path = TABLES / f"{name}.csv"
    df.to_csv(path, index=index)
    print(f"  [table] {path.relative_to(ROOT)}  ({len(df)} rows)")
    return path


def save_frame(df: pd.DataFrame, name: str) -> Path:
    """Persist an intermediate analytical frame as parquet."""
    path = DATA / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"  [data ] {path.relative_to(ROOT)}  ({len(df):,} rows x {df.shape[1]} cols)")
    return path


def load_frame(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / f"{name}.parquet")


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
