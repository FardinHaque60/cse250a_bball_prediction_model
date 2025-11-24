import numpy as np
from numpy.typing import NDArray
from typing import List


def viterbi(emissions, initials, observations, transitions):
    """
    implements the viterbi algorithm for a discrete-emission hmm in log space
    emissions: shape (s, m) where s = num states, m = num observation symbols
    initials: shape (s,)
    observations: shape (t,) of integers in [0, m)
    transitions: shape (s, s)
    """
    S = emissions.shape[0]
    T = int(observations.shape[0])

    log_transitions = np.log(transitions)
    log_emissions = np.log(emissions)
    log_initials = np.log(initials)

    viterbi_matrix = np.zeros((S, T), dtype=float)
    backpointer = np.zeros((S, T), dtype=int)

    # initialization
    first_obs = observations[0]
    viterbi_matrix[:, 0] = log_initials + log_emissions[:, first_obs]

    # dynamic programming forward pass
    for t in range(1, T):
        obs_t = observations[t]

        # shape (s, s): previous score for each prev state + transition log probs
        transition_scores = viterbi_matrix[:, t - 1][:, np.newaxis] + log_transitions

        backpointer[:, t] = np.argmax(transition_scores, axis=0)
        viterbi_matrix[:, t] = (
            transition_scores[backpointer[:, t], np.arange(S)] + log_emissions[:, obs_t]
        )

    # backtracking
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(viterbi_matrix[:, -1])

    for t in range(T - 2, -1, -1):
        path[t] = backpointer[path[t + 1], t + 1]

    return path


def viterbi_on_sequences(emissions, initials, transitions, obs_sequences):
    """
    runs viterbi on a list of observation sequences
    obs_sequences: list of 1d integer arrays
    """
    paths: List[NDArray] = []
    for obs in obs_sequences:
        if obs.size == 0:
            paths.append(np.array([], dtype=int))
            continue
        path = viterbi(
            emissions=emissions,
            initials=initials,
            observations=obs,
            transitions=transitions,
        )
        paths.append(path)
    return paths
