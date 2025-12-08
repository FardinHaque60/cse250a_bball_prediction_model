import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

# Add project root to path
PROJECT_ROOT = Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hmm.preprocess import read_games_from_csv, compute_season_factor_stats

DATA_PATH = str(PROJECT_ROOT / "data" / "allseasons.csv")


def optimize_weights():
    season_to_games = read_games_from_csv(DATA_PATH)
    season_stats = compute_season_factor_stats(season_to_games)

    X_list = []
    y_list = []

    for season, games in season_to_games.items():
        if season == 2024 or season == 2018:
            continue  # skip test seasons

        mean_vec, std_vec = season_stats[season]

        for g in games:
            # Z-score
            z_factors = (g.factors - mean_vec) / std_vec
            # Flip TOV
            z_factors[1] = -z_factors[1]

            X_list.append(z_factors)
            y_list.append(g.state)

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"Training Logistic Regression on {len(X)} games...")
    clf = LogisticRegression(
        fit_intercept=False
    )  # We want just the weights for the factors
    clf.fit(X, y)

    print("Learned Coefficients:")
    factors = ["eFG%", "TOV%", "ORB%", "FT/FGA"]
    weights = clf.coef_[0]

    # Normalize weights so they sum to 1 (like Dean Oliver's roughly sum to 1: 0.4+0.25+0.2+0.15 = 1.0)
    norm_weights = weights / np.sum(np.abs(weights))

    print(f"Optimized Weights: {norm_weights}")

    print(f"Dean Oliver: 0.40, 0.25, 0.20, 0.15")

    return norm_weights


if __name__ == "__main__":
    optimize_weights()
