import random
from tqdm import tqdm

SEASON_LENGTH = 82
DATA_PATH = "data/temp.csv"

# trains MLE model on all data
def train_bigram(all_data):
    '''
        expects a 2d list of data
        ex: [
                [1, 0, 1, ...], # team record
                [0, 0, 1, ...], 
                ...
            ]
    '''
    initial_cpt = {1: 0, 0: 0} # increment counts for each first game
    transition_cpt = { # key represents previous game outcome
        1: {1: 0, 0: 0}, # if previous game was a win, then count wins
        0: {1: 0, 0: 0} # if previous game was a loss, then count wins
    }
    prev_game_counts = {1: 0, 0: 0} # total counts of previous games for normalization

    for season in tqdm(all_data, desc="Processing team records"):
        # count initial game
        first_game = season[0]
        initial_cpt[first_game] += 1
        # count transitions
        for i in range(1, len(season)):
            prev_game = season[i - 1]
            curr_game = season[i]
            transition_cpt[prev_game][curr_game] += 1
            prev_game_counts[prev_game] += 1

    #print("SANITY CHECKS:")
    #print("Initial CPT counts:", initial_cpt)
    #print("Transition CPT counts:", transition_cpt)
    #print("Previous game counts:", prev_game_counts)

    # convert counts to probabilities
    initial_cpt[1] /= len(all_data) # divide by number of seasons
    initial_cpt[0] /= len(all_data)

    transition_cpt[1][1] /= prev_game_counts[1]
    transition_cpt[1][0] /= prev_game_counts[1]

    transition_cpt[0][1] /= prev_game_counts[0]
    transition_cpt[0][0] /= prev_game_counts[0]

    model = {}
    model["initial"] = initial_cpt
    model["transition"] = transition_cpt
    print("completed training bigram model")

    return model

def infer_bigram_season(model):
    season_predictions = []

    first_game_pred = random.choices([1, 0], weights=[model["initial"][1], model["initial"][0]])[0]
    season_predictions.append(first_game_pred)

    for game_num in range(1, SEASON_LENGTH): # calc for game index 1 to 81
        prev_outcome = season_predictions[game_num - 1]
        win_prob = model["transition"][prev_outcome][1]
        loss_prob = model["transition"][prev_outcome][0]
        outcome = random.choices([1, 0], weights=[win_prob, loss_prob])[0]
        season_predictions.append(outcome)
    return season_predictions