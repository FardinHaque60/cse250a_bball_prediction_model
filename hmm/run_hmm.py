from hmm.preprocess import build_sequences_from_csv
from hmm.hmm_model import train_supervised_hmm, sequence_accuracy
from hmm.hmm_viterbi import viterbi_on_sequences

data_path = None  # TODO: add data path

# build sequences using 2018-19 and 2024-25 as test set
train_states, train_obs, test_states, test_obs = build_sequences_from_csv(
    data_path, holdout_seasons=(2018, 2024)
)

# train supervised hmm (mle using known win/loss states)
pi, A, B = train_supervised_hmm(train_states, train_obs, num_states=2)

# run viterbi on test observation sequences
pred_paths = viterbi_on_sequences(B, pi, A, test_obs)

# evaluate prediction accuracy vs actual w/l
acc = sequence_accuracy(test_states, pred_paths)
print("test accuracy:", acc)
