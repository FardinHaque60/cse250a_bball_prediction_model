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
def train_trigram(all_data):
    initial_cpt = {
        "<start>": {"W": 0, "L": 0},
        "W": {"W": 0, "L": 0},
        "L": {"W": 0, "L": 0}
    }
    # First key is 2 games ago
    # Second key is 1 game ago (previous)
    transition_cpt = {
        "W": {
            "W": {"W": 0, "L": 0},
            "L": {"W": 0, "L": 0}
        },
        "L": {
            "W": {"W": 0, "L": 0},
            "L": {"W": 0, "L": 0}
        }
    }
    prev_bigram_counts = {
        "W": {"W": 0, "L": 0},
        "L": {"W": 0, "L": 0},
    } # total counts of previous bigrams for normalization

    for season in tqdm(all_data, desc="Processing seasons"):
        # count initial bigrams
        initial_cpt["<start>"][season[0]] += 1
        initial_cpt[season[0]][season[1]] += 1
        # count transitions
        for i in range(2, len(season)):
            # 2 games ago
            prev2_game = season[i - 2]
            # 1 game ago (prev game)
            prev1_game = season[i - 1]
            curr_game = season[i]
            transition_cpt[prev2_game][prev1_game][curr_game] += 1
            prev_bigram_counts[prev2_game][prev1_game] += 1

    # convert counts to probabilities
    for prev_game in initial_cpt:
        for curr_game in initial_cpt[prev_game]:
            initial_cpt[prev_game][curr_game] /= len(all_data)
            
    for prev2_game in transition_cpt:
        for prev1_game in transition_cpt[prev2_game]:
            for curr_game in transition_cpt[prev2_game][prev1_game]:
                transition_cpt[prev2_game][prev1_game][curr_game] /= prev_bigram_counts[prev2_game][prev1_game]
                
    MODEL_CPTS["initial"] = initial_cpt
    MODEL_CPTS["transition"] = transition_cpt
    print("completed training MLE model")

def infer_season():
    season_predictions = []

    first_game_pred = random.choices(["W", "L"], weights=[MODEL_CPTS["initial"]["<start>"]["W"], MODEL_CPTS["initial"]["<start>"]["L"]])[0]
    season_predictions.append(first_game_pred)
    second_game_pred = random.choices(["W", "L"], weights=[MODEL_CPTS["initial"][first_game_pred]["W"], MODEL_CPTS["initial"][first_game_pred]["L"]])[0]
    season_predictions.append(second_game_pred)
    
    for game_num in range(2, SEASON_LENGTH):
        prev2_outcome = season_predictions[game_num - 2]
        prev1_outcome = season_predictions[game_num - 1]
        win_prob = MODEL_CPTS["transition"][prev2_outcome][prev1_outcome]["W"]
        loss_prob = MODEL_CPTS["transition"][prev2_outcome][prev1_outcome]["L"]
        outcome = random.choices(["W", "L"], weights=[win_prob, loss_prob])[0]
        season_predictions.append(outcome)

    return season_predictions

if __name__ == "__main__":
    data = read_data(DATA_PATH)

    model = train_trigram(data) # TODO add param for degree for general handling of n previous games
    print("MODEL CPTs:", MODEL_CPTS)

    season_pred = infer_season()
    mock_actual_season = generate_season_data()
    print("season_pred", season_pred)
    print("mock_actual_season", mock_actual_season)

    print("accuracy:", sequence_accuracy(mock_actual_season, season_pred))