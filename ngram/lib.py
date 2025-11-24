# contains common functions for ngram models
import random
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