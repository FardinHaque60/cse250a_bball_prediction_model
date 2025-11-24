import random
from tqdm import tqdm

# will include "initial" and "transition" CPTs
SEASON_LENGTH = 82
DATA_PATH = "data/temp.csv"

# trains MLE model on all data
def train_unigram(all_data):
    cpt = {1: 0, 0: 0}
    all_games = 0
    
    for season in tqdm(all_data, desc="Processing team records"):
        for i in range(len(season)):
            game_outcome = season[i]
            cpt[game_outcome] += 1
            all_games += 1
    
    cpt[1] /= all_games
    cpt[0] /= all_games
    
    model = {}
    model["initial"] = cpt
    print("completed training unigram model")

    return model

def infer_unigram_season(model):
    season_predictions = [random.choices([1, 0], weights=[model["initial"][1], model["initial"][0]])[0] for _ in range(SEASON_LENGTH)]
    
    return season_predictions