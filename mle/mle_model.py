import random
from tqdm import tqdm

# will include "initial" and "transition" CPTs
MODEL_CPTS = {}
SEASON_LENGTH = 82

# generates one bball season of random data
def generate_season_data():
    '''
        generates random sequence of 82 "W" or "L"'s to represent outcomes for a bball season
    '''
    return [random.choice(["W", "L"]) for _ in range(82)]

# trains MLE model on all data
def train_mle_model(all_data):
    '''
        expects a 2d list of data
        ex: [
                ["W", "L", "W", ...], # team record
                ["L", "L", "W", ...], 
                ...
            ]
    '''
    initial_cpt = {"W": 0, "L": 0} # increment counts for each first game
    transition_cpt = { # key represents previous game outcome
        "W": {"W": 0, "L": 0}, # if previous game was a win, then count wins
        "L": {"W": 0, "L": 0} # if previous game was a loss, then count wins
    }
    prev_game_counts = {"W": 0, "L": 0} # total counts of previous games for normalization

    for season in tqdm(all_data, desc="Processing seasons"):
        # count initial game
        first_game = season[0]
        initial_cpt[first_game] += 1
        # count transitions
        for i in range(1, len(season)):
            prev_game = season[i - 1]
            curr_game = season[i]
            transition_cpt[prev_game][curr_game] += 1
            prev_game_counts[prev_game] += 1

    # print("SANITY CHECKS:")
    #print("Initial CPT counts:", initial_cpt)
    #print("Transition CPT counts:", transition_cpt)
    #print("Previous game counts:", prev_game_counts)

    # convert counts to probabilities
    initial_cpt["W"] /= len(all_data) # divide by number of seasons
    initial_cpt["L"] /= len(all_data)

    transition_cpt["W"]["W"] /= prev_game_counts["W"]
    transition_cpt["W"]["L"] /= prev_game_counts["W"]

    transition_cpt["L"]["W"] /= prev_game_counts["L"]
    transition_cpt["L"]["L"] /= prev_game_counts["L"]

    MODEL_CPTS["initial"] = initial_cpt
    MODEL_CPTS["transition"] = transition_cpt
    print("completed training MLE model")

def infer_season():
    season_predictions = []

    first_game_pred = random.choices(["W", "L"], weights=[MODEL_CPTS["initial"]["W"], MODEL_CPTS["initial"]["L"]])[0]
    season_predictions.append(first_game_pred)

    for game_num in range(1, SEASON_LENGTH): # calc for game index 1 to 81
        prev_outcome = season_predictions[game_num - 1]
        win_prob = MODEL_CPTS["transition"][prev_outcome]["W"]
        loss_prob = MODEL_CPTS["transition"][prev_outcome]["L"]
        outcome = random.choices(["W", "L"], weights=[win_prob, loss_prob])[0]
        season_predictions.append(outcome)
    return season_predictions

if __name__ == "__main__":
    mock_data = [generate_season_data() for _ in range(50)] # generate 1000 seasons of random data

    model = train_mle_model(mock_data) # TODO add param for degree for general handling of n previous games
    print("MODEL CPTs:", MODEL_CPTS)

    season_pred = infer_season()
    print(season_pred)