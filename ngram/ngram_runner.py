# use this file to run {uni, bi, tri}gram models for testing
from bigram import *
from trigram import *
from unigram import *
from lib import read_data, sequence_accuracy, generate_season_data

# constants
DATA_PATH = "data/scrape_results_2019_2022.csv" # path where data is located

if __name__ == "__main__":
    # TODO change what data is used, mock or real data
    raw_data = read_data(DATA_PATH) # real data
    data = list(raw_data.values())
    # data = [generate_season_data() for _ in range(50)]  # mock data

    # TODO change what model is used, can use train_unigram, train_bigram, train_trigram
    model = train_trigram(data)
    print("MODEL CPTs:", model)
    print()

    # TODO change inference function based on model used
    season_pred = infer_trigram_season(model)

    mock_actual_season = generate_season_data()
    print("season_pred", season_pred)
    print("mock_actual_season", mock_actual_season)

    print("accuracy:", sequence_accuracy(mock_actual_season, season_pred))