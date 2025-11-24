import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np


# per game object to store game data
@dataclass
class GameRecord:
    season_start_year: int
    team: str
    date: datetime
    state: int  # 0 = loss, 1 = win
    factors: np.ndarray  # vector of format [e_fg, tov, orb, ft_fga]


def infer_season(game_date):
    """
    infers nba season from a game date
    example: oct 2000 - jun 2001 -> 2000
    """
    month = game_date.month
    year = game_date.year
    if month >= 10:
        return year
    return year - 1


def read_games_from_csv(csv_path):
    """
    reads regular season games from a csv file and groups them by season
    """
    season_to_games: Dict[int, List[GameRecord]] = defaultdict(list)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # only looking at regular season games
            if row.get("IsRegular", "0") != "1":
                continue

            # parse date and season
            game_date = datetime.strptime(row["date"], "%m/%d/%Y")
            season = infer_season(game_date)

            team = row["team"]
            state = int(row["W/L"])

            # extract four factors
            e_fg = float(row["eFG%"])
            tov = float(row["TOV%"])
            orb = float(row["ORB%"])
            ft_fga = float(row["FT/FGA"])

            factors = np.array([e_fg, tov, orb, ft_fga], dtype=float)

            record = GameRecord(
                season=season,
                team=team,
                date=game_date,
                state=state,
                factors=factors,
            )
            season_to_games[season].append(record)

    return season_to_games


def compute_season_factor_stats(season_to_games):
    """
    computes per-season mean and std for the four factors
    returns mapping: season -> (mean_vec, std_vec)
    """
    stats = {}
    for season, games in season_to_games.items():
        all_factors = np.stack([g.factors for g in games], axis=0)
        mean_vec = all_factors.mean(axis=0)
        std_vec = all_factors.std(axis=0, ddof=0)

        # avoid division by zero when standardizing
        std_vec = np.where(std_vec > 0.0, std_vec, 1.0)
        stats[season] = (mean_vec, std_vec)
    return stats


def standardize_and_bin_games(season_to_games, season_stats, bin_edges):
    """
    computes per-game z-scores per season and bins into discrete observation symbols
    """
    season_sequences: Dict[int, Dict[str, List[Tuple[np.ndarray, np.ndarray]]]] = {}

    for season, games in season_to_games.items():
        mean_vec, std_vec = season_stats[season]

        # group games by team for this season
        team_to_games: Dict[str, List[GameRecord]] = defaultdict(list)
        for g in games:
            team_to_games[g.team].append(g)

        team_sequences: List[Tuple[np.ndarray, np.ndarray]] = []

        for _, team_games in team_to_games.items():
            # sort by date to form the game sequence for this team-season
            team_games_sorted = sorted(team_games, key=lambda r: r.date)

            states = np.fromiter((g.state for g in team_games_sorted), dtype=int)
            raw_factors = np.stack([g.factors for g in team_games_sorted], axis=0)

            # per-season z-scores for each factor
            z_factors = (raw_factors - mean_vec[None, :]) / std_vec[None, :]

            # flip sign of turnovers so that higher is better for the weighted score
            z_e_fg = z_factors[:, 0]
            z_tov = -z_factors[:, 1]
            z_orb = z_factors[:, 2]
            z_ft_fga = z_factors[:, 3]

            # dean oliver weights: shooting 40, turnovers 25, rebounding 20, ft 15
            weighted_z = 0.4 * z_e_fg + 0.25 * z_tov + 0.2 * z_orb + 0.15 * z_ft_fga

            # bin the weighted z-score into discrete symbols
            # np.digitize returns indices in 1..len(bin_edges); we shift to 0-based
            obs_symbols = np.digitize(weighted_z, bin_edges, right=False) - 1

            team_sequences.append((states, obs_symbols))

        season_sequences[season] = {
            "sequences": team_sequences,
        }

    return season_sequences


def build_sequences_from_csv(csv_path):
    """
    builds train and test sequences from a csv file of games
    """
    holdout_seasons = (2018, 2024)
    bin_edges = np.array(
        [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf],
        dtype=float,
    )

    season_to_games = read_games_from_csv(csv_path)
    season_stats = compute_season_factor_stats(season_to_games)
    season_sequences = standardize_and_bin_games(
        season_to_games=season_to_games,
        season_stats=season_stats,
        bin_edges=bin_edges,
    )

    train_states: List[np.ndarray] = []
    train_obs: List[np.ndarray] = []
    test_states: List[np.ndarray] = []
    test_obs: List[np.ndarray] = []

    for season, payload in season_sequences.items():
        sequences = payload["sequences"]
        for states, obs in sequences:
            # ignore empty sequences if they appear
            if states.size == 0:
                continue

            if season in holdout_seasons:
                test_states.append(states)
                test_obs.append(obs)
            else:
                train_states.append(states)
                train_obs.append(obs)

    return train_states, train_obs, test_states, test_obs


if __name__ == "__main__":
    # simple sanity check on temp.csv-shaped data
    train_y, train_x, test_y, test_x = build_sequences_from_csv("temp.csv")
    print(f"num train sequences: {len(train_y)}")
    print(f"num test sequences: {len(test_y)}")
