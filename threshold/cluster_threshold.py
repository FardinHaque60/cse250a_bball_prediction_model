import numpy as np

def optimal_threshold(cluster_data):
    """
    find optimal threshold given training data for a specific tier cluster. takes all unique four factors score across cluster data, sorts the scores, and uses the midpoints between adjacent sorted four factor scores as potential thresholds. the threshold that results in the best accuracy (correct game outcome predicitons) is deemed the optimal threshold.
    """
    cluster = sorted(cluster_data)
    cluster_ff_scores = np.array([c[0] for c in cluster])
    cluster_outcomes = np.array([int(c[1]) for c in cluster])
    unique_ff_scores = np.unique(cluster_ff_scores)
    
    thresholds = []
    for a,b in zip(unique_ff_scores[:-1], unique_ff_scores[1:]):
        thresholds.append(0.5 * (a+b))
    
    best_threshold = None
    best_acc = -1
    for t in thresholds:
        preds = (cluster_ff_scores >= t).astype(int)
        acc = (preds == cluster_outcomes).mean()
        if acc > best_acc:
            best_acc = acc
            best_threshold = t
    return best_threshold

def infer_threshold_season(wins, clusters, tiered_teams, season_ff_scores):
    """
    Infer the outcomes for a team's season based on the four factors score for each game and the true total amount of wins the team amassed that season.
    """
    # Determine team's tier based on their number of wins and how many clusters
    # 2 clusters
    if clusters == 2:
        if wins > 37.5:
            tier = "tier_1"
        else:
            tier = "tier_0"
    # 3 clusters
    elif clusters == 3:
        if wins > 45:
            tier = "tier_2"
        elif wins > 30.5:
            tier = "tier_1"
        else:
            tier = "tier_0"
    # 4 clusters
    elif clusters == 4:
        if wins > 52.5:
            tier = "tier_3"
        elif wins > 42:
            tier = "tier_2"
        elif wins > 30:
            tier = "tier_1"
        else:
            tier = "tier_0"
    # 5 clusters
    elif clusters == 5:
        if wins > 59:
            tier = "tier_4"
        elif wins > 51:
            tier = "tier_3"
        elif wins > 42:
            tier = "tier_2"
        elif wins > 30:
            tier = "tier_1"
        else:
            tier = "tier_0"
    threshold = optimal_threshold(tiered_teams[tier])
    
    predicted_wins = []
    for ff in season_ff_scores:
        if ff >= threshold:
            outcome = 1
        else:
            outcome = 0
        predicted_wins.append(outcome)
        
    return predicted_wins, tier