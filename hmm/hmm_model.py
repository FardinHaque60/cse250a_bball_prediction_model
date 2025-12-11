from typing import List, Sequence

import numpy as np


def infer_num_symbols(obs_sequences: Sequence[np.ndarray]) -> int:
    """
    determines the number of discrete observation symbols present in the data
    """
    max_symbol = -1
    for obs in obs_sequences:
        if obs.size == 0:
            continue
        max_symbol = max(max_symbol, int(obs.max()))
    if max_symbol < 0:
        raise ValueError("no observations provided for emission estimation")
    return max_symbol + 1


def estimate_initial_and_transition(state_sequences, num_states, smoothing=1.0):
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
    num_states,
    num_symbols=None,
    smoothing=1.0,
):
    """
    trains a supervised discrete-emission hmm using known win/loss states
    returns:
        (initial_probs, transition_probs, emission_probs)
    """
    if num_symbols is None:
        num_symbols = infer_num_symbols(obs_sequences)

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


def random_stochastic_matrix(num_rows, num_cols, rng):
    samples = rng.random((num_rows, num_cols)) + 1e-6
    samples /= samples.sum(axis=1, keepdims=True)
    return samples


def forward_backward(obs, emissions, initials, transitions):
    """
    scaled forward-backward pass for a single observation sequence
    returns log likelihood, gammas, and xi accumulators
    """
    S = emissions.shape[0]
    T = obs.size

    alpha = np.zeros((S, T), dtype=float)
    beta = np.zeros((S, T), dtype=float)
    scales = np.zeros(T, dtype=float)

    # initial step
    alpha[:, 0] = initials * emissions[:, obs[0]]
    scales[0] = alpha[:, 0].sum()
    if scales[0] == 0.0:
        scales[0] = 1e-12
    alpha[:, 0] /= scales[0]

    # forward pass
    for t in range(1, T):
        alpha[:, t] = (alpha[:, t - 1] @ transitions) * emissions[:, obs[t]]
        scales[t] = alpha[:, t].sum()
        if scales[t] == 0.0:
            scales[t] = 1e-12
        alpha[:, t] /= scales[t]

    # backward pass
    beta[:, -1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[:, t] = transitions @ (emissions[:, obs[t + 1]] * beta[:, t + 1])
        beta[:, t] /= scales[t + 1]

    log_likelihood = -np.sum(np.log(scales))
    gamma = alpha * beta
    gamma /= np.maximum(gamma.sum(axis=0, keepdims=True), 1e-12)

    xi_accum = np.zeros((S, S), dtype=float)
    for t in range(T - 1):
        xi = (
            alpha[:, t][:, None]
            * transitions
            * emissions[:, obs[t + 1]][None, :]
            * beta[:, t + 1][None, :]
        )
        xi_sum = xi.sum()
        if xi_sum == 0.0:
            xi_sum = 1e-12
        xi /= xi_sum
        xi_accum += xi

    return log_likelihood, gamma, xi_accum


def train_unsupervised_hmm(
    obs_sequences,
    num_states,
    num_symbols=None,
    smoothing=1e-3,
    max_iters=50,
    tol=1e-3,
    random_state=None,
):
    """
    trains a discrete hmm with baum-welch (unsupervised em)
    returns (initial_probs, transition_probs, emission_probs, log_likelihood_trace)
    """
    if num_symbols is None:
        num_symbols = infer_num_symbols(obs_sequences)

    rng = np.random.default_rng(random_state)
    initials = rng.dirichlet(np.ones(num_states))
    transitions = random_stochastic_matrix(num_states, num_states, rng)
    emissions = random_stochastic_matrix(num_states, num_symbols, rng)

    log_likelihoods = []
    prev_ll = -np.inf

    for _ in range(max_iters):
        init_accum = np.zeros(num_states, dtype=float)
        trans_accum = np.zeros((num_states, num_states), dtype=float)
        emit_accum = np.zeros((num_states, num_symbols), dtype=float)
        total_ll = 0.0

        for obs in obs_sequences:
            if obs.size == 0:
                continue
            ll, gamma, xi = forward_backward(
                obs=obs,
                emissions=emissions,
                initials=initials,
                transitions=transitions,
            )
            total_ll += ll
            init_accum += gamma[:, 0]
            trans_accum += xi
            for t, sym in enumerate(obs):
                emit_accum[:, sym] += gamma[:, t]

        initials = init_accum + smoothing
        initials /= initials.sum()

        transitions = trans_accum + smoothing
        transitions /= transitions.sum(axis=1, keepdims=True)

        emissions = emit_accum + smoothing
        emissions /= emissions.sum(axis=1, keepdims=True)

        log_likelihoods.append(total_ll)
        if abs(total_ll - prev_ll) < tol:
            break
        prev_ll = total_ll

    return initials, transitions, emissions, log_likelihoods


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


def map_hidden_states_to_outcomes(
    hidden_sequences, true_sequences, num_hidden_states, fallback_label=None
):
    """
    derives a mapping from hidden states to observed win/loss labels
    """
    counts = np.zeros((num_hidden_states, 2), dtype=float)
    global_counts = np.zeros(2, dtype=float)

    for hidden, true in zip(hidden_sequences, true_sequences):
        if hidden.size == 0 or true.size == 0:
            continue
        T = min(hidden.size, true.size)
        h = hidden[:T]
        t = true[:T]
        counts[h, t] += 1.0
        global_counts += np.bincount(t, minlength=2)

    if fallback_label is None:
        fallback_label = 0 if global_counts[0] >= global_counts[1] else 1

    mapping = np.full(num_hidden_states, fallback_label, dtype=int)
    for state in range(num_hidden_states):
        if counts[state].sum() == 0.0:
            continue
        mapping[state] = int(np.argmax(counts[state]))
    return mapping


def convert_hidden_to_observed(hidden_sequences, mapping):
    """
    applies a hidden->observed mapping to a list of sequences
    """
    mapped: List[np.ndarray] = []
    for seq in hidden_sequences:
        if seq.size == 0:
            mapped.append(seq.copy())
            continue
        mapped.append(mapping[seq])
    return mapped
