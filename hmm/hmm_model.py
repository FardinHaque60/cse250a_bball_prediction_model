import numpy as np


def estimate_initial_and_transition(state_sequences, num_states=2, smoothing=1.0):
    """
    estimates initial state distribution and transition matrix via supervised mle
    state_sequences: list of 1d arrays of ints in [0, num_states)
    """
    initial_counts = np.zeros(num_states, dtype=float)
    transition_counts = np.zeros((num_states, num_states), dtype=float)

    for seq in state_sequences:
        if seq.size == 0:
            continue

        # initial state count
        initial_counts[seq[0]] += 1.0

        if seq.size > 1:
            prev_states = seq[:-1]
            next_states = seq[1:]

            # accumulate transitions for this sequence using bincount
            idx = prev_states * num_states + next_states
            seq_counts = np.bincount(
                idx,
                minlength=num_states * num_states,
            ).reshape(num_states, num_states)
            transition_counts += seq_counts

    # apply laplace smoothing to avoid zeros
    initial_counts += smoothing
    initial_probs = initial_counts / initial_counts.sum()

    transition_counts += smoothing
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_probs = transition_counts / row_sums

    return initial_probs, transition_probs


def estimate_emissions(
    state_sequences, obs_sequences, num_states, num_symbols, smoothing=1.0
):
    """
    estimates discrete emission probabilities via supervised mle
    state_sequences and obs_sequences must be aligned lists of 1d arrays
    """
    emission_counts = np.zeros((num_states, num_symbols), dtype=float)

    for states, obs in zip(state_sequences, obs_sequences):
        if states.size == 0:
            continue

        # guard against mismatched lengths
        T = min(states.size, obs.size)
        s = states[:T]
        o = obs[:T]

        idx = s * num_symbols + o
        seq_counts = np.bincount(
            idx,
            minlength=num_states * num_symbols,
        ).reshape(num_states, num_symbols)
        emission_counts += seq_counts

    emission_counts += smoothing
    row_sums = emission_counts.sum(axis=1, keepdims=True)
    emission_probs = emission_counts / row_sums

    return emission_probs


def train_supervised_hmm(
    state_sequences,
    obs_sequences,
    num_states=2,
    num_symbols=None,
    smoothing=1.0,
):
    """
    trains a supervised discrete-emission hmm using known win/loss states
    returns:
        (initial_probs, transition_probs, emission_probs)
    """
    if num_symbols is None:
        max_symbol = -1
        for obs in obs_sequences:
            if obs.size == 0:
                continue
            max_symbol = max(max_symbol, int(obs.max()))
        if max_symbol < 0:
            raise ValueError("no observations provided for emission estimation")
        num_symbols = max_symbol + 1

    initial_probs, transition_probs = estimate_initial_and_transition(
        state_sequences=state_sequences,
        num_states=num_states,
        smoothing=smoothing,
    )

    emission_probs = estimate_emissions(
        state_sequences=state_sequences,
        obs_sequences=obs_sequences,
        num_states=num_states,
        num_symbols=num_symbols,
        smoothing=smoothing,
    )

    return initial_probs, transition_probs, emission_probs


def sequence_accuracy(true_sequences, pred_sequences):
    """
    computes simple per-game accuracy across all provided sequences
    """
    correct = 0
    total = 0

    for y_true, y_pred in zip(true_sequences, pred_sequences):
        if y_true.size == 0 or y_pred.size == 0:
            continue
        T = min(y_true.size, y_pred.size)
        correct += np.sum(y_true[:T] == y_pred[:T])
        total += T

    if total == 0:
        return 0.0

    return correct / total
