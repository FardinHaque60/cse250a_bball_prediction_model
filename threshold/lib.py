import csv

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

def get_ff_scores(path):
    """
    computes and returns the four factors score for each game in the data path
    """
    ff_scores = {}
    with open(path, "r") as f:
        csv_reader = csv.DictReader(f)
        for i,row in enumerate(csv_reader):
            if i == 0:
                continue
            team_season = row['team'] + row['Season']
            ff = 0.4*float(row['eFG%']) + 0.25*float(row['TOV%']) + 0.2*float(row['ORB%']) + 0.15*float(row['FT/FGA'])
            result = int(row['W/L'])
            if team_season not in ff_scores:
                ff_scores[team_season] = []
            ff_scores[team_season].append([ff, result])
            
    return ff_scores