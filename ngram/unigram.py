import random
from tqdm import tqdm
import csv

# will include "initial" and "transition" CPTs
MODEL_CPTS = {}
SEASON_LENGTH = 82
DATA_PATH = "data/temp.csv"

def sequence_accuracy(true_sequence, pred_sequence):
    """
    computes simple per-game accuracy across a single sequence
    """
    correct = 0
    total = 0

    T = min(len(true_sequence), len(pred_sequence))
    for t in range(T):
        if true_sequence[t] == pred_sequence[t]:
            correct += 1
        total += 1

    if total == 0:
        return 0.0
    return correct / total

def read_data(path):
    team_results = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['IsRegular'] == '1':
                team = row['team']
                result = int(row['W/L'])
                if team not in team_results:
                    team_results[team] = []
                team_results[team].append(result)

    team_results = list(team_results.values())
    return team_results

# generates one bball season of random data for a single team
def generate_season_data():
    '''
        generates random sequence of 82 1's or 0's to represent outcomes for a bball season
    '''
    return [random.choice([1, 0]) for _ in range(82)]

# trains MLE model on all data
def train_unigram(all_data):
    cpt = {"W": 0, "L": 0}
    all_games = 0
    
    for season in tqdm(all_data, desc="Processing seasons"):
        for i in range(len(season)):
            game_outcome = season[i]
            cpt[game_outcome] += 1
            all_games += 1
    
    cpt["W"] /= all_games
    cpt["L"] /= all_games
    
    MODEL_CPTS["initial"] = cpt
    print("completed training MLE model")

def infer_season():
    season_predictions = [random.choices(["W", "L"], weights=[MODEL_CPTS["initial"]["W"], MODEL_CPTS["initial"]["L"]])[0] for _ in range(SEASON_LENGTH)]
    
    return season_predictions

if __name__ == "__main__":
    data = read_data(DATA_PATH)

    model = train_unigram(data) # TODO add param for degree for general handling of n previous games
    print("MODEL CPTs:", MODEL_CPTS)

    season_pred = infer_season()
    mock_actual_season = generate_season_data()
    print("season_pred", season_pred)
    print("mock_actual_season", mock_actual_season)

    print("accuracy:", sequence_accuracy(mock_actual_season, season_pred))