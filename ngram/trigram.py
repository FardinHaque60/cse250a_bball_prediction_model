import random
from tqdm import tqdm

# will include "initial" and "transition" CPTs
SEASON_LENGTH = 82
DATA_PATH = "data/temp.csv"

# trains MLE model on all data
def train_trigram(all_data):
    initial_cpt = {
        "<start>": {1: 0, 0: 0},
        1: {1: 0, 0: 0},
        0: {1: 0, 0: 0}
    }
    prev_first_game_counts = { # used to divide initial CPTs for normalization
        1: 0,
        0: 0,
    }
    # First key is 2 games ago
    # Second key is 1 game ago (previous)
    transition_cpt = {
        1: {
            1: {1: 0, 0: 0},
            0: {1: 0, 0: 0}
        },
        0: {
            1: {1: 0, 0: 0},
            0: {1: 0, 0: 0}
        }
    }
    prev_bigram_counts = {
        1: {1: 0, 0: 0},
        0: {1: 0, 0: 0},
    } # total counts of previous bigrams for normalization

    for season in tqdm(all_data, desc="Processing team records"):
        # count initial bigrams
        initial_cpt["<start>"][season[0]] += 1
        initial_cpt[season[0]][season[1]] += 1
        prev_first_game_counts[season[0]] += 1
        # count transitions
        for i in range(2, len(season)):
            # 2 games ago
            prev2_game = season[i - 2]
            # 1 game ago (prev game)
            prev1_game = season[i - 1]
            curr_game = season[i]
            transition_cpt[prev2_game][prev1_game][curr_game] += 1
            prev_bigram_counts[prev2_game][prev1_game] += 1

    # convert initial counts to probabilities
    initial_cpt["<start>"][1] /= len(all_data) # divide by number of seasons
    initial_cpt["<start>"][0] /= len(all_data)

    initial_cpt[1][1] /= prev_first_game_counts[1]
    initial_cpt[1][0] /= prev_first_game_counts[1]

    initial_cpt[0][1] /= prev_first_game_counts[0]
    initial_cpt[0][0] /= prev_first_game_counts[0]
            
    for prev2_game in transition_cpt:
        for prev1_game in transition_cpt[prev2_game]:
            for curr_game in transition_cpt[prev2_game][prev1_game]:
                transition_cpt[prev2_game][prev1_game][curr_game] /= prev_bigram_counts[prev2_game][prev1_game]
                
    model = {}
    model["initial"] = initial_cpt
    model["transition"] = transition_cpt
    print("completed training trigram model")

    return model

def infer_trigram_season(model):
    season_predictions = []

    first_game_pred = random.choices([1, 0], weights=[model["initial"]["<start>"][1], model["initial"]["<start>"][0]])[0]
    season_predictions.append(first_game_pred)
    second_game_pred = random.choices([1, 0], weights=[model["initial"][first_game_pred][1], model["initial"][first_game_pred][0]])[0]
    season_predictions.append(second_game_pred)
    
    for game_num in range(2, SEASON_LENGTH):
        prev2_outcome = season_predictions[game_num - 2]
        prev1_outcome = season_predictions[game_num - 1]
        win_prob = model["transition"][prev2_outcome][prev1_outcome][1]
        loss_prob = model["transition"][prev2_outcome][prev1_outcome][0]
        outcome = random.choices([1, 0], weights=[win_prob, loss_prob])[0]
        season_predictions.append(outcome)

    return season_predictions